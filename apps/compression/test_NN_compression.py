#!/usr/bin/env python3

import argparse
import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from MLP import init_model, model_label, predict, train

jax.config.update("jax_enable_x64", True)

RUN_DIR = "compression-test-POD"
FILE_INDEX = 0
Y_INDEX = None
Nopt = {"adam": 500, "BGFS": 1000}
N_SAMPLES_PER_DIR = 64
LR = 5e-2
ARCHITECTURE = "mlp"  # "siren" or "mlp"
MLP_ACTIVATION = "sin"  # "sin" or "tanh" (only used when ARCHITECTURE == "mlp")
OMEGA_0 = 2.0  # only used when ARCHITECTURE == "siren"
LAYER_SIZES = [4, 16, 16, 16, 1]


def load_f_4d(run_dir, file_index=0):
    baseline = os.path.join(run_dir, "branch_baseline")
    snapshot = os.path.join(baseline, f"GYSELALIBXX_{file_index:05d}.h5")
    initstate = os.path.join(baseline, "GYSELALIBXX_initstate.h5")

    with h5py.File(initstate, "r") as h5:
        x = np.asarray(h5["MeshX"][:], dtype=np.float64)
        y = np.asarray(h5["MeshY"][:], dtype=np.float64)
        vx = np.asarray(h5["MeshVx"][:], dtype=np.float64)
        vy = np.asarray(h5["MeshVy"][:], dtype=np.float64)

    with h5py.File(snapshot, "r") as h5:
        f = np.asarray(h5["fdistribu"][0], dtype=np.float64)

    x_period = float(x[-1] - x[0])
    y_period = float(y[-1] - y[0])

    return x, y, vx, vy, f, x_period, y_period, snapshot


def make_training_set(x, y, vx, vy, f, n_per_dir, seed=0):
    """Random subgrid: n_per_dir indices along each of x, y, vx, vy."""
    rng = np.random.default_rng(seed)
    nx, ny, nvx, nvy = f.shape

    ix = rng.choice(nx, min(n_per_dir, nx), replace=False)
    iy = rng.choice(ny, min(n_per_dir, ny), replace=False)
    ivx = rng.choice(nvx, min(n_per_dir, nvx), replace=False)
    ivy = rng.choice(nvy, min(n_per_dir, nvy), replace=False)

    f_sub = f[np.ix_(ix, iy, ivx, ivy)]
    x_sub, y_sub, vx_sub, vy_sub = x[ix], y[iy], vx[ivx], vy[ivy]

    X, Y, VX, VY = np.meshgrid(x_sub, y_sub, vx_sub, vy_sub, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), VX.ravel(), VY.ravel()], axis=1)
    targets = f_sub.ravel()

    return jnp.asarray(coords), jnp.asarray(targets)


def x_vx_slice_at_mid_y_vy(f, x, vx, y, vy, y_index=None):
    y_idx = f.shape[1] // 2 if y_index is None else y_index
    vy_idx = vy.size // 2
    slice_x_vx = f[:, y_idx, :, vy_idx]
    y_val = y[y_idx]
    vy_val = vy[vy_idx]

    xx, vvx = np.meshgrid(x, vx, indexing="ij")
    coords = np.stack(
        [
            xx.ravel(),
            np.full(xx.size, y_val),
            vvx.ravel(),
            np.full(xx.size, vy_val),
        ],
        axis=1,
    )
    return slice_x_vx, coords, y_val, vy_val


def predict_slice(model, coords, nx, nvx):
    pred = np.asarray(predict(model, coords))
    return pred.reshape(nx, nvx)


def plot_slice(ax, x, vx, data, title):
    ax.pcolormesh(x, vx, data.T, shading="auto")
    ax.set_xlabel("x")
    ax.set_ylabel("vx")
    ax.set_title(title)


def plot_loss(losses):
    plt.figure(figsize=(7, 3))
    plt.semilogy(losses)
    plt.xlabel("step")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIREN or MLP on fdistribu.")
    parser.add_argument("run_dir", nargs="?", default=RUN_DIR)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--arch",
        choices=("siren", "mlp"),
        default=ARCHITECTURE,
        help="Network architecture (default: %(default)s)",
    )
    parser.add_argument(
        "--activation",
        choices=("sin", "tanh"),
        default=MLP_ACTIVATION,
        help="Hidden activation for mlp only: sin or tanh (default: %(default)s)",
    )
    parser.add_argument(
        "--omega-0",
        type=float,
        default=OMEGA_0,
        help="SIREN omega_0 for the first layer (default: %(default)g)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    lr = args.lr

    x, y, vx, vy, f, x_period, y_period, snapshot = load_f_4d(run_dir, FILE_INDEX)
    coords_train, targets_train = make_training_set(
        x, y, vx, vy, f, N_SAMPLES_PER_DIR
    )

    slice_true, coords_slice, _, _ = x_vx_slice_at_mid_y_vy(f, x, vx, y, vy, Y_INDEX)

    key = jax.random.PRNGKey(0)
    model = init_model(
        key,
        LAYER_SIZES,
        lx=x_period,
        ly=y_period,
        arch=args.arch,
        activation=args.activation,
        omega_0=args.omega_0,
    )

    print(f"Training on {coords_train.shape[0]} points, f shape {f.shape}")
    print(f"Lx={float(model['lx']):.6g}, Ly={float(model['ly']):.6g} (fixed on model)")
    print(f"Architecture: {model_label(model)}")
    print(f"Then BFGS: {Nopt['BGFS']} max iterations ...")

    model, losses = train(
        model,
        coords_train,
        targets_train,
        Nopt["adam"],
        Nopt["BGFS"],
        lr=lr,
    )
    print(f"  final loss = {losses[-1]:.6e}")

    slice_pred = predict_slice(model, coords_slice, x.size, vx.size)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plot_slice(axes[0], x, vx, slice_true, "original")
    plot_slice(axes[1], x, vx, slice_pred, model_label(model))
    fig.savefig("slice_x_vx_nn.png", dpi=150)
    plt.close(fig)

    plot_loss(losses)
    print("saved slice_x_vx_nn.png and training_loss.png")
