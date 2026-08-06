"""Standalone experiment: run the user's Gauss-Newton (Levenberg-Marquardt-style)
training loop under the SAME conditions as a real compress_array call --
same grid resolution/bounds (read live from apps/compression/params_two_stream.yaml's
SplineMesh section) and same arch/batch_size (from its compression.NN section) --
so the error message (if any) matches what plugging this into
NeuralNetworkCompressor._fit_on_species (in place of the L-BFGS phase) would
actually produce.

This is a throwaway diagnostic script, NOT part of the NeuralNetworkCompressor
class or the pytest suite (run it directly, not via pytest, so a crash here
can't take down a real compression job).

Caveat: the *field values* used as regression targets are a synthetic
Maxwellian-plus-perturbation (same shape as test_inr_compressor.py's
generate_synthetic_fdistribu), not an actual two_stream simulation snapshot --
we don't have a real fdistribu.h5 to load here. That's irrelevant for this
experiment: the memory blow-up is entirely structural (driven by grid
resolution and n_params), independent of what values the field holds.

Usage:
    python src/tests/gauss_newton_experiment.py # use default apps/compression/params_two_stream.yaml
    python src/tests/gauss_newton_experiment.py --n-map 2000 --n-iterations 2
    python src/tests/gauss_newton_experiment.py --params-yaml apps/compression/params_two_stream.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax.flatten_util import ravel_pytree

from compression_methods.neural_network import NeuralNetworkCompressor, _losses_function
from scimba_jax.nonlinear_approximation.optimizers.optimizers import ScimbaAdam

jax.config.update("jax_enable_x64", True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PARAMS_YAML = os.path.join(REPO_ROOT, "apps", "compression", "params_two_stream.yaml")


def load_production_config(params_yaml_path):
    """Read the exact grid bounds/resolution and NN hyperparameters
    launch_benchmark.py would use for a real run of this yaml.
    """
    with open(params_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    mesh = config["SplineMesh"]
    nn_cfg = config["compression"]["NN"]

    return {
        "x_min": float(mesh["x_min"]), "x_max": float(mesh["x_max"]), "nx": int(mesh["x_ncells"]),
        "y_min": float(mesh["y_min"]), "y_max": float(mesh["y_max"]), "ny": int(mesh["y_ncells"]),
        "vx_min": float(mesh["vx_min"]), "vx_max": float(mesh["vx_max"]), "nvx": int(mesh["vx_ncells"]),
        "vy_min": float(mesh["vy_min"]), "vy_max": float(mesh["vy_max"]), "nvy": int(mesh["vy_ncells"]),
        "arch": nn_cfg["arch"],
        "batch_size": int(nn_cfg["batch_size"]),
        "lbfgs_iters": int(nn_cfg["lbfgs_iters"]),
    }


def make_synthetic_fdistribu(cfg):
    """Same Maxwellian-plus-perturbation shape as test_inr_compressor.py's
    generate_synthetic_fdistribu, evaluated on the real production grid.
    """
    x = np.linspace(cfg["x_min"], cfg["x_max"], cfg["nx"], endpoint=False)
    y = np.linspace(cfg["y_min"], cfg["y_max"], cfg["ny"], endpoint=False)
    vx = np.linspace(cfg["vx_min"], cfg["vx_max"], cfg["nvx"], endpoint=True)
    vy = np.linspace(cfg["vy_min"], cfg["vy_max"], cfg["nvy"], endpoint=True)
    Xg, Yg, VXg, VYg = np.meshgrid(x, y, vx, vy, indexing="ij")
    f = np.exp(-0.5 * (VXg ** 2 + VYg ** 2)) * (1.0 + 0.1 * np.cos(Xg) * np.sin(Yg))
    return f.reshape(-1, 1)


def predict_map(model, coords, add_identity=False):
    """Glue for train_map_gn's `predict_map` call: our INR models take a single
    (4,) input and return a (1,) scalar, vmapped over a batch of coords.
    `add_identity` is a no-op here (that flag belongs to the coordinate-map
    networks it was written for, not our scalar-field INRs).
    """
    del add_identity
    return jax.vmap(model)(coords)


def _run_scan_with_progress(step_fn, initial_state, n_iterations, desc):
    print(f"[{desc}] tracing + running jax.lax.scan over {n_iterations} iterations ...")
    xs = jnp.arange(n_iterations)
    final_state, loss_hist = jax.lax.scan(step_fn, initial_state, xs)
    return final_state, loss_hist


# --- verbatim training loop, as pasted by the user (only predict_map/make_data/
# _run_scan_with_progress are supplied as glue) ---

def train_map_gn(n_map, make_data, params, n_iterations=100, init_damping=1e-2):
    key = jax.random.PRNGKey(42)
    flat_params, unflatten = ravel_pytree(params)

    def residual_fn(p, coords, t):
        pred = predict_map(unflatten(p), coords, add_identity=False)
        return (pred - t).reshape(-1)

    initial_state = (flat_params, init_damping, key)

    def step_fn(state, i):
        curr_p, damping, key = state
        coords_train, target, key = make_data(n_map, key)
        r = residual_fn(curr_p, coords_train, target)
        J = jax.jacfwd(residual_fn)(curr_p, coords_train, target)
        current_loss = jnp.mean(r ** 2)
        grad = J.T @ r
        JTJ = J.T @ J
        num_params = JTJ.shape[0]
        step = jnp.linalg.solve(JTJ + damping * jnp.eye(num_params), -grad)
        direction_derivative = jnp.dot(grad, step)

        def ls_cond(ls_state):
            alpha, count, _, loss_cand = ls_state
            return (loss_cand > current_loss + 1e-4 * alpha * direction_derivative) & (
                count < 8
            )

        def ls_body(ls_state):
            alpha, count, _, _ = ls_state
            new_alpha = alpha * 0.5
            new_p = curr_p + new_alpha * step
            new_loss = jnp.mean(residual_fn(new_p, coords_train, target) ** 2)
            return new_alpha, count + 1, new_p, new_loss

        init_p_cand = curr_p + 1.0 * step
        init_l_cand = jnp.mean(residual_fn(init_p_cand, coords_train, target) ** 2)
        final_alpha, ls_count, final_p, final_loss = jax.lax.while_loop(
            ls_cond, ls_body, (1.0, 0, init_p_cand, init_l_cand)
        )
        new_damping = jnp.where(
            ls_count >= 8,
            damping * 10.0,
            jnp.where(ls_count == 0, damping / 2.0, damping),
        )
        new_damping = jnp.clip(new_damping, 1e-5, 1e2)
        success = final_loss < current_loss
        actual_p = jnp.where(success, final_p, curr_p)
        actual_loss = jnp.where(success, final_loss, current_loss)
        return (actual_p, new_damping, key), actual_loss

    final_state, loss_hist = _run_scan_with_progress(step_fn, initial_state, n_iterations, "GN")
    p_final, _, _ = final_state
    return unflatten(p_final), loss_hist


# --- driver ---

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params-yaml", default=DEFAULT_PARAMS_YAML,
                         help="GYSELA yaml to read grid bounds/resolution + NN config from (default: apps/compression/params_two_stream.yaml)")
    parser.add_argument("--arch", default=None,
                         help="Override the yaml's compression.NN.arch (e.g. periodic_siren_small_32), "
                              "to test GN on a smaller network without editing the yaml.")
    parser.add_argument("--n-map", type=int, default=None,
                         help="Gauss-Newton mini-batch size per step. Default: the FULL production grid "
                              "(nx*ny*nvx*nvy), matching what NeuralNetworkCompressor's L-BFGS phase "
                              "currently trains on -- i.e. exactly what GN would see if plugged in as its replacement.")
    parser.add_argument("--n-iterations", type=int, default=2)
    parser.add_argument("--init-damping", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-batch", action="store_true",
                         help="Re-fit the SAME mini-batch at every GN iteration instead of "
                              "resampling a fresh one each step, to isolate whether resampling "
                              "noise (vs. the GN direction itself) explains poor convergence.")
    parser.add_argument("--adam-warmup-iters", type=int, default=0,
                         help="Run ScimbaAdam for this many resampled-mini-batch iterations "
                              "BEFORE handing the model to train_map_gn, as Philipp suggested "
                              "(precondition with Adam instead of starting GN from random init).")
    parser.add_argument("--adam-warmup-batch-size", type=int, default=8000)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("JAX devices:", jax.devices())

    cfg = load_production_config(args.params_yaml)
    if args.arch is not None:
        cfg["arch"] = args.arch
    print(f"\nLoaded production config from {args.params_yaml}:")
    print(f"  grid (nx,ny,nvx,nvy) = ({cfg['nx']}, {cfg['ny']}, {cfg['nvx']}, {cfg['nvy']})"
          f"  -> {cfg['nx']*cfg['ny']*cfg['nvx']*cfg['nvy']:,} points")
    print(f"  bounds: x[{cfg['x_min']},{cfg['x_max']}] y[{cfg['y_min']},{cfg['y_max']}] "
          f"vx[{cfg['vx_min']},{cfg['vx_max']}] vy[{cfg['vy_min']},{cfg['vy_max']}]")
    print(f"  arch = {cfg['arch']}, batch_size (ADAM/L-BFGS full-batch reference) = {cfg['batch_size']}, "
          f"lbfgs_iters = {cfg['lbfgs_iters']}")

    # Reuse the real compressor's own grid-building/normalization logic so the
    # coordinates fed to the model are byte-for-byte what compress_array would build.
    compressor = NeuralNetworkCompressor(
        x_min=cfg["x_min"], x_max=cfg["x_max"],
        y_min=cfg["y_min"], y_max=cfg["y_max"],
        vx_min=cfg["vx_min"], vx_max=cfg["vx_max"],
        vy_min=cfg["vy_min"], vy_max=cfg["vy_max"],
        arch=cfg["arch"], verbose=False,
    )
    inputs = compressor._build_inputs(cfg["nx"], cfg["ny"], cfg["nvx"], cfg["nvy"])
    targets = jnp.asarray(make_synthetic_fdistribu(cfg))
    n_points = inputs.shape[0]
    print(f"\nBuilt real production grid: inputs.shape={inputs.shape}, targets.shape={targets.shape}")

    n_map = args.n_map if args.n_map is not None else n_points
    print(f"n_map (GN mini-batch size) = {n_map:,}"
          + (" (= full grid, matching L-BFGS's current full-batch phase)" if n_map == n_points else ""))

    if args.fixed_batch:
        fixed_key = jax.random.PRNGKey(args.seed + 1)
        fixed_idx = jax.random.choice(fixed_key, n_points, shape=(n_map,), replace=(n_map > n_points))
        fixed_inputs, fixed_targets = inputs[fixed_idx], targets[fixed_idx]
        print("--fixed-batch: GN will re-fit the SAME batch at every iteration (no resampling).")

        def make_data(n_map, key):
            del n_map
            return fixed_inputs, fixed_targets, key
    else:
        def make_data(n_map, key):
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, n_points, shape=(n_map,), replace=(n_map > n_points))
            return inputs[idx], targets[idx], key

    key = jax.random.PRNGKey(args.seed)
    model = get_inr_model_from_compressor(compressor, key)
    flat_params, _ = ravel_pytree(model)
    n_params = flat_params.shape[0]
    print(f"\nn_params = {n_params}")
    print(f"expected JTJ size (float64) = {n_params * n_params * 8 / 1e9:.2f} GB")
    print(f"n_iterations = {args.n_iterations}\n")

    if args.adam_warmup_iters > 0:
        print(f"=== Adam warmup: {args.adam_warmup_iters} iters, "
              f"batch_size={args.adam_warmup_batch_size}, lr={args.adam_lr} ===")
        adam_opt = ScimbaAdam(model, _losses_function, learning_rate=args.adam_lr)
        t_warm0 = time.perf_counter()
        for i in range(args.adam_warmup_iters):
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, n_points, shape=(args.adam_warmup_batch_size,), replace=False)
            batch = (inputs[idx], targets[idx])
            loss_dict, model, adam_opt = adam_opt.update(model, batch)
            if i % 200 == 0 or i == args.adam_warmup_iters - 1:
                print(f"  [Adam warmup] iter {i:5d}  loss={float(loss_dict['total']):.6e}  "
                      f"t={time.perf_counter() - t_warm0:.2f}s")
        print(f"Adam warmup done in {time.perf_counter() - t_warm0:.2f}s\n")

    t0 = time.perf_counter()
    try:
        final_model, loss_hist = train_map_gn(
            n_map, make_data, model,
            n_iterations=args.n_iterations, init_damping=args.init_damping,
        )
        jax.block_until_ready(loss_hist)
        t1 = time.perf_counter()
        print(f"\nOK: completed {args.n_iterations} GN iterations in {t1 - t0:.2f}s")
        print(f"loss history: {loss_hist}")

        # Held-out check: the printed loss_hist (fixed-batch mode) is measured on the
        # SAME points the model was fit on, so a low value could just mean memorizing
        # those 8000 points rather than generalizing. Evaluate on a fresh random sample
        # (never used for training) to get a number comparable to Adam's resampled-batch loss.
        eval_key = jax.random.PRNGKey(args.seed + 999)
        eval_idx = jax.random.choice(eval_key, n_points, shape=(min(50_000, n_points),), replace=False)
        eval_pred = jax.vmap(final_model)(inputs[eval_idx])
        eval_loss = float(jnp.mean((eval_pred - targets[eval_idx]) ** 2))
        print(f"held-out loss (fresh 50,000-point sample, never used for training): {eval_loss:.6e}")
    except Exception as e:
        t1 = time.perf_counter()
        print(f"\nFAILED after {t1 - t0:.2f}s: {type(e).__name__}")
        print(str(e)[:1500])
        sys.exit(1)


def get_inr_model_from_compressor(compressor, key):
    from compression_methods.neural_network import get_inr_model
    return get_inr_model(compressor.arch, key)


if __name__ == "__main__":
    main()
