"""Standalone experiment: chunked + gradient-checkpointed loss/grad for the
L-BFGS full-batch phase, tested under the SAME conditions as a real
compress_array call (grid resolution/bounds read live from
apps/compression/params_two_stream.yaml, real arch, real ScimbaLBfgs).

Goal: replace `_losses_function`'s single `jax.vmap(model)(inputs)` over the
ENTIRE grid (the actual OOM cause -- peak memory scales O(N) with grid size,
already ~17.3M points at the current 64,64,65,65 resolution, heading to
~277M at 128,128,129,129) with a version that processes the batch in fixed
`chunk_size` chunks via `jax.lax.scan`, wrapped in `jax.checkpoint` so the
backward pass recomputes each chunk's forward activations instead of storing
all chunks' activations at once. Peak memory becomes O(chunk_size),
independent of total grid resolution; loss and gradient stay mathematically
IDENTICAL to the unchunked full-batch computation (this is exact gradient
accumulation, not a stochastic approximation) -- important since L-BFGS's
curvature estimate needs a deterministic, consistent objective.

This is a throwaway diagnostic script, NOT part of NeuralNetworkCompressor or
the pytest suite (run it directly). If it checks out, the same
make_chunked_losses_function is meant to be ported into neural_network.py's
_losses_function.

Usage:
    # 1) correctness only (tiny grid, chunked vs unchunked must match exactly)
    python src/tests/chunked_lbfgs_experiment.py --skip-scale-test

    # 2) full run: correctness check + real production grid/arch L-BFGS iterations
    python src/tests/chunked_lbfgs_experiment.py --chunk-size 200000 --n-iterations 5

    # 3) tune chunk-size for your own GPU's free memory
    python src/tests/chunked_lbfgs_experiment.py --chunk-size 50000 --n-iterations 3
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

from compression_methods.neural_network import (
    NeuralNetworkCompressor, get_inr_model, _losses_function, make_chunked_losses_function,
)
from scimba_jax.nonlinear_approximation.optimizers.optimizers import ScimbaLBfgs

jax.config.update("jax_enable_x64", True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PARAMS_YAML = os.path.join(REPO_ROOT, "apps", "compression", "params_two_stream.yaml")

# make_chunked_losses_function is imported directly from neural_network.py (not
# duplicated here) so this script exercises the exact function NeuralNetworkCompressor
# ships with, not a copy that could silently drift from it.


# --- glue: read the real production grid/config, same as gauss_newton_experiment.py ---

def load_production_config(params_yaml_path):
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
        "lbfgs_iters": int(nn_cfg["lbfgs_iters"]),
    }


def make_synthetic_fdistribu(cfg):
    x = np.linspace(cfg["x_min"], cfg["x_max"], cfg["nx"], endpoint=False)
    y = np.linspace(cfg["y_min"], cfg["y_max"], cfg["ny"], endpoint=False)
    vx = np.linspace(cfg["vx_min"], cfg["vx_max"], cfg["nvx"], endpoint=True)
    vy = np.linspace(cfg["vy_min"], cfg["vy_max"], cfg["nvy"], endpoint=True)
    Xg, Yg, VXg, VYg = np.meshgrid(x, y, vx, vy, indexing="ij")
    f = np.exp(-0.5 * (VXg ** 2 + VYg ** 2)) * (1.0 + 0.1 * np.cos(Xg) * np.sin(Yg))
    return f.reshape(-1, 1)


# --- step 1: correctness check on a tiny grid where the unchunked path still fits ---

def check_correctness(arch, chunk_size, seed):
    print("=== Correctness check (chunked vs unchunked, must match to fp64 precision) ===")
    cfg = {
        "x_min": 0.0, "x_max": 2 * np.pi, "nx": 8,
        "y_min": 0.0, "y_max": 2 * np.pi, "ny": 8,
        "vx_min": -6.0, "vx_max": 6.0, "nvx": 8,
        "vy_min": -6.0, "vy_max": 6.0, "nvy": 8,
    }
    compressor = NeuralNetworkCompressor(
        x_min=cfg["x_min"], x_max=cfg["x_max"], y_min=cfg["y_min"], y_max=cfg["y_max"],
        vx_min=cfg["vx_min"], vx_max=cfg["vx_max"], vy_min=cfg["vy_min"], vy_max=cfg["vy_max"],
        arch=arch, verbose=False,
    )
    inputs = compressor._build_inputs(cfg["nx"], cfg["ny"], cfg["nvx"], cfg["nvy"])
    targets = jnp.asarray(make_synthetic_fdistribu(cfg))
    batch = (inputs, targets)
    n = inputs.shape[0]
    small_chunk_size = min(chunk_size, max(1, n // 3))  # force >1 chunk even on this tiny grid
    print(f"tiny grid n_points={n}, using chunk_size={small_chunk_size} ({-(-n // small_chunk_size)} chunks)")

    key = jax.random.PRNGKey(seed)
    model = get_inr_model(arch, key)

    loss_ref = _losses_function(model, batch)["total"]
    grad_ref = jax.grad(lambda m: _losses_function(m, batch)["total"])(model)

    chunked_fn = make_chunked_losses_function(small_chunk_size)
    loss_chunked = chunked_fn(model, batch)["total"]
    grad_chunked = jax.grad(lambda m: chunked_fn(m, batch)["total"])(model)

    grad_flat_ref, _ = ravel_pytree(grad_ref)
    grad_flat_chunked, _ = ravel_pytree(grad_chunked)

    loss_diff = abs(float(loss_ref) - float(loss_chunked))
    grad_diff = float(jnp.max(jnp.abs(grad_flat_ref - grad_flat_chunked)))

    print(f"loss (unchunked) = {float(loss_ref):.15e}")
    print(f"loss (chunked)   = {float(loss_chunked):.15e}")
    print(f"|loss diff|      = {loss_diff:.3e}")
    print(f"max |grad diff|  = {grad_diff:.3e}")

    ok = loss_diff < 1e-10 and grad_diff < 1e-8
    print("=> " + ("PASS: chunked loss/grad match the unchunked reference." if ok
                    else "FAIL: chunked loss/grad DO NOT match -- do not trust the scale test below."))
    print()
    return ok


# --- step 2: real production grid + real arch + real ScimbaLBfgs, chunked loss swapped in ---

def run_scale_test(cfg, chunk_size, n_iterations, seed):
    print("=== Scale test: real production grid/arch, chunked loss, real ScimbaLBfgs ===")
    print(f"grid (nx,ny,nvx,nvy) = ({cfg['nx']}, {cfg['ny']}, {cfg['nvx']}, {cfg['nvy']})"
          f" -> {cfg['nx']*cfg['ny']*cfg['nvx']*cfg['nvy']:,} points")
    print(f"arch = {cfg['arch']}")

    compressor = NeuralNetworkCompressor(
        x_min=cfg["x_min"], x_max=cfg["x_max"], y_min=cfg["y_min"], y_max=cfg["y_max"],
        vx_min=cfg["vx_min"], vx_max=cfg["vx_max"], vy_min=cfg["vy_min"], vy_max=cfg["vy_max"],
        arch=cfg["arch"], verbose=False,
    )
    inputs = compressor._build_inputs(cfg["nx"], cfg["ny"], cfg["nvx"], cfg["nvy"])
    targets = jnp.asarray(make_synthetic_fdistribu(cfg))
    n = inputs.shape[0]
    n_chunks = -(-n // chunk_size)
    print(f"chunk_size = {chunk_size:,}  ->  n_chunks = {n_chunks}")

    key = jax.random.PRNGKey(seed)
    model = get_inr_model(cfg["arch"], key)
    flat_params, _ = ravel_pytree(model)
    print(f"n_params = {flat_params.shape[0]}\n")

    chunked_fn = make_chunked_losses_function(chunk_size)
    lbfgs_opt = ScimbaLBfgs(model, chunked_fn)
    full_batch = (inputs, targets)

    for i in range(n_iterations):
        t0 = time.perf_counter()
        loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
        loss_val = float(loss_dict["total"])
        t1 = time.perf_counter()
        print(f"  [L-BFGS/chunked] iter {i}: loss={loss_val:.6e}  ({t1 - t0:.2f}s)")

    print("\n=> Completed without OOM.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params-yaml", default=DEFAULT_PARAMS_YAML)
    parser.add_argument("--chunk-size", type=int, default=200_000,
                         help="Points processed per scan step. Tune down if you still OOM, "
                              "up if you have memory to spare and want fewer (faster) chunks.")
    parser.add_argument("--n-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-correctness-check", action="store_true")
    parser.add_argument("--skip-scale-test", action="store_true",
                         help="Only run the tiny-grid correctness check, skip the real production-grid run.")
    args = parser.parse_args()

    print("JAX devices:", jax.devices(), "\n")

    cfg = load_production_config(args.params_yaml)

    if not args.skip_correctness_check:
        ok = check_correctness(cfg["arch"], args.chunk_size, args.seed)
        if not ok and not args.skip_scale_test:
            print("Aborting scale test since correctness already failed.")
            sys.exit(1)

    if not args.skip_scale_test:
        t0 = time.perf_counter()
        try:
            run_scale_test(cfg, args.chunk_size, args.n_iterations, args.seed)
        except Exception as e:
            t1 = time.perf_counter()
            print(f"\nFAILED after {t1 - t0:.2f}s: {type(e).__name__}")
            print(str(e)[:1500])
            sys.exit(1)


if __name__ == "__main__":
    main()
