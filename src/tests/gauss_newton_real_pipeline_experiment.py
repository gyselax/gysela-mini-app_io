"""Run Philipp's Gauss-Newton (train_map_gn), Adam-preconditioned, INSIDE the real
NeuralNetworkCompressor.compress_array pipeline -- same grid-building, same
warm-start plumbing, same species loop -- WITHOUT touching neural_network.py's
own L-BFGS phase at all.

How: subclass NeuralNetworkCompressor and override only _fit_on_species. Phase 1
(ADAM) is copied verbatim from the real method (same code, same behavior). Phase 2
substitutes train_map_gn (imported from gauss_newton_experiment.py, so it reuses the
exact same glue/predict_map already validated there) in place of ScimbaLBfgs. Nothing
in src/python/compression_methods/neural_network.py is modified.

Caveat: synthetic Maxwellian-plus-perturbation target (no real fdistribu.h5), same as
gauss_newton_experiment.py -- irrelevant here since we're comparing optimizer
behavior/generalization, not physical accuracy.

Usage:
    python src/tests/gauss_newton_real_pipeline_experiment.py --arch periodic_siren_small_32 \
        --adam-iters 2000 --adam-batch-size 8000 --gn-iterations 50 --gn-n-map 8000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
sys.path.insert(0, os.path.dirname(__file__))  # so `from gauss_newton_experiment import ...` resolves

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from compression_methods.neural_network import (
    NeuralNetworkCompressor, get_inr_model, _losses_function, _training_progress,
)
from scimba_jax.nonlinear_approximation.optimizers.optimizers import ScimbaAdam

from gauss_newton_experiment import train_map_gn  # reuses its own module-level predict_map

jax.config.update("jax_enable_x64", True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PARAMS_YAML = os.path.join(REPO_ROOT, "apps", "compression", "params_two_stream.yaml")


def load_production_config(params_yaml_path):
    with open(params_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    mesh = config["SplineMesh"]
    return {
        "x_min": float(mesh["x_min"]), "x_max": float(mesh["x_max"]), "nx": int(mesh["x_ncells"]),
        "y_min": float(mesh["y_min"]), "y_max": float(mesh["y_max"]), "ny": int(mesh["y_ncells"]),
        "vx_min": float(mesh["vx_min"]), "vx_max": float(mesh["vx_max"]), "nvx": int(mesh["vx_ncells"]),
        "vy_min": float(mesh["vy_min"]), "vy_max": float(mesh["vy_max"]), "nvy": int(mesh["vy_ncells"]),
    }


def make_synthetic_fdistribu(cfg):
    x = np.linspace(cfg["x_min"], cfg["x_max"], cfg["nx"], endpoint=False)
    y = np.linspace(cfg["y_min"], cfg["y_max"], cfg["ny"], endpoint=False)
    vx = np.linspace(cfg["vx_min"], cfg["vx_max"], cfg["nvx"], endpoint=True)
    vy = np.linspace(cfg["vy_min"], cfg["vy_max"], cfg["nvy"], endpoint=True)
    Xg, Yg, VXg, VYg = np.meshgrid(x, y, vx, vy, indexing="ij")
    f = np.exp(-0.5 * (VXg ** 2 + VYg ** 2)) * (1.0 + 0.1 * np.cos(Xg) * np.sin(Yg))
    return f[None, ...]  # (1 species, nx, ny, nvx, nvy)


def load_real_fdistribu(h5_path):
    """Load a real GYSELALIBXX fdistribu snapshot (already-run two_stream simulation
    output, no need to re-run anything), shape (n_species, nx, ny, nvx, nvy).
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        fdistribu = np.asarray(f["fdistribu"])
    return fdistribu


class GaussNewtonNeuralNetworkCompressor(NeuralNetworkCompressor):
    """NeuralNetworkCompressor with the L-BFGS polish phase swapped for train_map_gn.

    Phase 1 (ADAM) below is a verbatim copy of the real _fit_on_species's Phase 1 --
    same optax schedule logic, same batch sampling -- so this class's behavior only
    differs from production in Phase 2. Everything else (compress_array, warm-start
    payload handling, species loop, grid building) is inherited unchanged.
    """

    def __init__(self, *args, gn_iterations=50, gn_n_map=8000, gn_init_damping=1e-2, **kwargs):
        super().__init__(*args, **kwargs)
        self.gn_iterations = gn_iterations
        self.gn_n_map = gn_n_map
        self.gn_init_damping = gn_init_damping
        self.gn_loss_hist = None  # exposed for the driver to print/inspect after the call

    def _fit_on_species(self, inputs, targets, key, warm_model):
        import optax  # local import, same as neural_network.py's module-level one

        is_warm = warm_model is not None
        if is_warm:
            model = warm_model
        else:
            key, subkey = jax.random.split(key)
            model = get_inr_model(self.arch, subkey)

        total_points = inputs.shape[0]
        loss_history = []
        tag = "warm" if is_warm else "cold"
        best_model, best_loss = model, float("inf")

        # --- Phase 1: ADAM (verbatim copy of NeuralNetworkCompressor._fit_on_species) ---
        if is_warm:
            learning_rate = optax.cosine_decay_schedule(
                init_value=self.lr, decay_steps=self.max_iters, alpha=self.lr_decay_alpha
            )
        else:
            learning_rate = self.lr
        adam_opt = ScimbaAdam(model, _losses_function, learning_rate=learning_rate)
        pbar = _training_progress(self.max_iters, f"[INR/{self.arch}][ADAM]", self.verbose)
        for i in pbar:
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.choice(subkey, total_points, shape=(self.batch_size,), replace=False)
            batch = (inputs[batch_idx], targets[batch_idx])
            loss_dict, model, adam_opt = adam_opt.update(model, batch)
            loss_val = float(loss_dict["total"])
            loss_history.append(loss_val)
            if loss_val < best_loss:
                best_model, best_loss = model, loss_val
            if self.verbose and i % 100 == 0:
                print(f"  [INR/{self.arch}][ADAM/{tag}] iter {i:4d} - loss: {loss_val:.2e}")
            if loss_val < self.threshold:
                pbar.write(f"  [INR/{self.arch}][ADAM] early convergence at iter {i}")
                break

        # --- Phase 2: Gauss-Newton (train_map_gn) INSTEAD OF ScimbaLBfgs ---
        model = best_model

        def make_data(n_map, k):
            k, sk = jax.random.split(k)
            idx = jax.random.choice(sk, total_points, shape=(n_map,), replace=(n_map > total_points))
            return inputs[idx], targets[idx], k

        t_gn0 = time.perf_counter()
        gn_model, gn_loss_hist = train_map_gn(
            self.gn_n_map, make_data, model,
            n_iterations=self.gn_iterations, init_damping=self.gn_init_damping,
        )
        jax.block_until_ready(gn_loss_hist)
        gn_loss_hist_np = np.asarray(gn_loss_hist)
        self.gn_loss_hist = gn_loss_hist_np
        if self.verbose:
            print(f"  [INR/{self.arch}][GN/{tag}] {self.gn_iterations} iters "
                  f"in {time.perf_counter() - t_gn0:.2f}s, final loss {gn_loss_hist_np[-1]:.6e}")
        loss_history.extend(gn_loss_hist_np.tolist())

        final_gn_loss = float(gn_loss_hist_np[-1])
        if final_gn_loss < best_loss:
            best_model, best_loss = gn_model, final_gn_loss

        return best_model, jnp.array(loss_history)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params-yaml", default=DEFAULT_PARAMS_YAML)
    parser.add_argument("--arch", default="periodic_siren_small_32")
    parser.add_argument("--adam-iters", type=int, default=400,
                         help="max_iters for the real ADAM phase (mirrors compression_config.py's max_iters).")
    parser.add_argument("--adam-batch-size", type=int, default=8000,
                         help="batch_size for the real ADAM phase.")
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--gn-iterations", type=int, default=50)
    parser.add_argument("--gn-n-map", type=int, default=8000)
    parser.add_argument("--gn-init-damping", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-lbfgs", action="store_true",
                         help="Use the REAL, unmodified NeuralNetworkCompressor (Adam + ScimbaLBfgs) "
                             "instead of the Gauss-Newton subclass, with lbfgs_iters=--gn-iterations "
                             "for an iteration-count-matched comparison.")
    parser.add_argument("--real-fdistribu", default=None,
                         help="Path to a real GYSELALIBXX_*.h5 snapshot (e.g. "
                              "results_two_stream/branch_baseline/GYSELALIBXX_00050.h5) to fit "
                              "instead of the synthetic Maxwellian-plus-perturbation target.")
    args = parser.parse_args()

    print("JAX devices:", jax.devices())

    cfg = load_production_config(args.params_yaml)
    print(f"\ngrid (nx,ny,nvx,nvy) = ({cfg['nx']}, {cfg['ny']}, {cfg['nvx']}, {cfg['nvy']})"
          f" -> {cfg['nx']*cfg['ny']*cfg['nvx']*cfg['nvy']:,} points")
    print(f"arch = {args.arch}, adam_iters={args.adam_iters}, adam_batch_size={args.adam_batch_size}, "
          f"gn_iterations={args.gn_iterations}, gn_n_map={args.gn_n_map}")

    if args.real_fdistribu is not None:
        fdistribu = load_real_fdistribu(args.real_fdistribu)
        print(f"Loaded REAL fdistribu from {args.real_fdistribu}: shape={fdistribu.shape}")
    else:
        fdistribu = make_synthetic_fdistribu(cfg)  # (1, nx, ny, nvx, nvy)

    if args.baseline_lbfgs:
        print(f"[baseline] using the REAL, unmodified NeuralNetworkCompressor "
              f"(Adam + ScimbaLBfgs), lbfgs_iters={args.gn_iterations}")
        compressor = NeuralNetworkCompressor(
            x_min=cfg["x_min"], x_max=cfg["x_max"],
            y_min=cfg["y_min"], y_max=cfg["y_max"],
            vx_min=cfg["vx_min"], vx_max=cfg["vx_max"],
            vy_min=cfg["vy_min"], vy_max=cfg["vy_max"],
            arch=args.arch,
            lr=args.adam_lr,
            max_iters=args.adam_iters,
            batch_size=args.adam_batch_size,
            lbfgs_iters=args.gn_iterations,
            lbfgs_chunk_size=200_000,
            seed=args.seed,
            verbose=True,
        )
        label = "ADAM + L-BFGS (real, unmodified NeuralNetworkCompressor)"
    else:
        compressor = GaussNewtonNeuralNetworkCompressor(
            x_min=cfg["x_min"], x_max=cfg["x_max"],
            y_min=cfg["y_min"], y_max=cfg["y_max"],
            vx_min=cfg["vx_min"], vx_max=cfg["vx_max"],
            vy_min=cfg["vy_min"], vy_max=cfg["vy_max"],
            arch=args.arch,
            lr=args.adam_lr,
            max_iters=args.adam_iters,
            batch_size=args.adam_batch_size,
            seed=args.seed,
            verbose=True,
            gn_iterations=args.gn_iterations,
            gn_n_map=args.gn_n_map,
            gn_init_damping=args.gn_init_damping,
        )
        label = "ADAM + Gauss-Newton (train_map_gn)"

    t0 = time.perf_counter()
    try:
        compressed = compressor.compress_array(fdistribu)
    except Exception as e:
        t1 = time.perf_counter()
        print(f"\nFAILED after {t1 - t0:.2f}s: {type(e).__name__}")
        print(str(e)[:1500])
        sys.exit(1)
    t1 = time.perf_counter()

    print(f"\nOK: real compress_array ({label}) completed in {t1 - t0:.2f}s")
    print(f"final best loss: {float(jnp.min(compressor.loss_histories[0])):.6e}")

    # Held-out check, exactly as in gauss_newton_experiment.py: evaluate on a fresh
    # random sample never used for training, to catch overfitting to the GN batch.
    # Shape read from the actual fdistribu array (not cfg), since a real snapshot's
    # (nx,ny,nvx,nvy) can differ from the yaml's *_ncells (e.g. nvx=66 vs vx_ncells=65,
    # off by one because velocity grids include both endpoints).
    _, real_nx, real_ny, real_nvx, real_nvy = fdistribu.shape
    inputs = compressor._build_inputs(real_nx, real_ny, real_nvx, real_nvy)
    targets = jnp.asarray(fdistribu[0].reshape(-1, 1))
    eval_key = jax.random.PRNGKey(args.seed + 999)
    eval_idx = jax.random.choice(eval_key, inputs.shape[0], shape=(50_000,), replace=False)
    final_model = compressor.models[0]
    eval_pred = jax.vmap(final_model)(inputs[eval_idx])
    eval_loss = float(jnp.mean((eval_pred - targets[eval_idx]) ** 2))
    print(f"held-out loss (fresh 50,000-point sample, never used for training): {eval_loss:.6e}")


if __name__ == "__main__":
    main()
