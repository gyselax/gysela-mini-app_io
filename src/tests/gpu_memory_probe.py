"""Standalone, MPI-free reproduction of point 4's GPU memory measurement: N
concurrent OS processes -- one per --nprocs, each with its own JAX runtime,
exactly like a real MPI rank's embedded Python/JAX runtime under PDI's pycall
plugin -- all calling the REAL OnlineNeuralNetworkCompressor.compress_decompress_array
on a synthetic local chunk, to measure per-rank GPU memory availability under
N-way contention on one GPU. No mpirun / gys_compress / PDI needed, so it's
faster to (re)run and immune to the C++/Kokkos abort-hangs-the-orchestrator
failure mode the real pipeline has.

This is how gn_chunk_size=300 was found for online (see apps/compression/
params_two_stream.yaml's online_NN section and
[[project-online-distributed-pipeline-port]]): gn_chunk_size=2000 (offline-tuned)
and 1000 both OOM'd whichever rank reached Gauss-Newton last, under real 4-rank
MPI contention; 300 survived cleanly across 5 compression events. Same pattern
as gauss_newton_real_pipeline_experiment.py: reuse the REAL production class via
a thin subclass instead of the full launch_benchmark.py CLI / MPI machinery.
`_fit_one_species` below is a verbatim copy of OnlineNeuralNetworkCompressor's
own method (src/python/compression_methods/neural_network.py) with two probe()
calls added around the Gauss-Newton branch -- if that method changes, re-sync
this copy by hand (same tradeoff the GN experiment script already accepts for
its own copied Phase 1).

Caveat: no physics simulation runs concurrently here (gys_compress/Kokkos,
which in the real online pipeline shares the same GPU too), so this
UNDER-estimates true contention somewhat -- treat results as an upper bound on
available memory, not an exact match to a real run. Re-validate any new
gn_chunk_size/gn_n_map with the full pipeline (launch_benchmark.py --online)
before trusting it in production.

Usage:
    python src/tests/gpu_memory_probe.py --nprocs 4 --gn-chunk-size 300 \
        --arch periodic_siren_small_32_l5 --polish-optimizer gauss_newton

    # push it further to find a new breaking point after an arch/nprocs change:
    python src/tests/gpu_memory_probe.py --nprocs 4 --gn-chunk-size 1000
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PARAMS_YAML = os.path.join(REPO_ROOT, "apps", "compression", "params_two_stream.yaml")


def _nvidia_smi_memory_mib(gpu_index: int = 0):
    """Device-wide used/free/total memory (MiB), as seen by nvidia-smi -- reflects
    every other rank's JAX runtime on the same GPU, unlike jax.devices()[0].memory_stats()
    which only sees this process's own XLA arena."""
    out = subprocess.run(
        [
            "nvidia-smi", "-i", str(gpu_index),
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=5, check=True,
    ).stdout.strip()
    used, free, total = (int(x) for x in out.split(","))
    return used, free, total


def probe(tag: str, rank, call_idx: int):
    """Record one GPU memory snapshot to gpu_memory_probe_rank{rank:03d}.csv in the cwd."""
    import jax  # local import: workers set XLA_PYTHON_CLIENT_PREALLOCATE before importing jax

    try:
        used_mib, free_mib, total_mib = _nvidia_smi_memory_mib()
    except (subprocess.SubprocessError, OSError, ValueError):
        used_mib = free_mib = total_mib = None

    stats = jax.devices()[0].memory_stats() or {}

    rank_tag = f"{rank:03d}" if rank is not None else "na"
    record = {
        "call_idx": call_idx,
        "rank": rank,
        "tag": tag,
        "nvidia_smi_used_mib": used_mib,
        "nvidia_smi_free_mib": free_mib,
        "nvidia_smi_total_mib": total_mib,
        "jax_bytes_in_use_mib": stats.get("bytes_in_use", 0) // (1024 * 1024),
        "jax_peak_bytes_in_use_mib": stats.get("peak_bytes_in_use", 0) // (1024 * 1024),
    }

    path = Path(f"gpu_memory_probe_rank{rank_tag}.csv")
    write_header = not path.is_file()
    with open(path, mode="a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def load_production_config(params_yaml_path):
    with open(params_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    mesh = config["SplineMesh"]
    return {
        "nx": int(mesh["x_ncells"]),
        "ny": int(mesh["y_ncells"]),
        "nvx": int(mesh["vx_ncells"]) + 1,  # velocity grids include both endpoints
        "nvy": int(mesh["vy_ncells"]) + 1,
    }


def factorize_grid(nprocs):
    """Split nprocs into (px, py) as close to a square as possible, e.g. 4 -> (2, 2)."""
    px = int(np.sqrt(nprocs))
    while px > 1 and nprocs % px != 0:
        px -= 1
    return px, nprocs // px


def local_chunk_shape(cfg, rank, nprocs):
    """This rank's local (nx, ny, nvx, nvy): only vx, vy are split across ranks
    (x, y stay whole), mirroring the real online path's domain decomposition."""
    px, py = factorize_grid(nprocs)
    ix, iy = divmod(rank, py)
    nvx_local = cfg["nvx"] // px + (1 if ix < cfg["nvx"] % px else 0)
    nvy_local = cfg["nvy"] // py + (1 if iy < cfg["nvy"] % py else 0)
    return cfg["nx"], cfg["ny"], max(nvx_local, 1), max(nvy_local, 1)


def make_synthetic_local_chunk(n_species, nx, ny, nvx, nvy):
    """Shape is what drives GPU memory, not physical content -- unlike
    gauss_newton_real_pipeline_experiment.py this isn't testing accuracy."""
    x = np.linspace(0.0, 2 * np.pi, nx, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, ny, endpoint=False)
    vx = np.linspace(-1.0, 1.0, nvx, endpoint=True)
    vy = np.linspace(-1.0, 1.0, nvy, endpoint=True)
    Xg, Yg, VXg, VYg = np.meshgrid(x, y, vx, vy, indexing="ij")
    f = np.exp(-0.5 * (VXg ** 2 + VYg ** 2) * 9) * (1.0 + 0.1 * np.cos(Xg) * np.sin(Yg))
    return np.tile(f[None, ...], (n_species, 1, 1, 1, 1))


def build_probed_compressor_class():
    """Deferred import: workers must set XLA_PYTHON_CLIENT_PREALLOCATE before
    `import jax` (transitively pulled in by compression_methods.neural_network)
    -- see launch_benchmark.py's own top-of-file fix for why."""
    from compression_methods.neural_network import (
        OnlineNeuralNetworkCompressor, ScimbaLBfgs, _losses_function, _training_progress, train_map_gn,
    )
    import jax

    class ProbedOnlineNeuralNetworkCompressor(OnlineNeuralNetworkCompressor):
        """OnlineNeuralNetworkCompressor with probe() calls bracketing Gauss-Newton."""

        def compress_decompress_array(self, array, rank=None, local_bounds=None):
            self._probe_rank = rank
            probe("call_entry", rank, self.n_calls)
            return super().compress_decompress_array(array, rank=rank, local_bounds=local_bounds)

        def _fit_one_species(
            self, model, opt, inputs, targets, n_iters_adam, n_iters_lbfgs, n_iters_gn, desc: str = "",
        ):
            rank = getattr(self, "_probe_rank", None)
            total_points = inputs.shape[0]
            bs = min(self.batch_size, total_points) if self.batch_size is not None else total_points
            loss_val = None
            best_model, best_loss = model, float("inf")

            pbar = _training_progress(n_iters_adam, f"{desc}[ADAM]", self.verbose)
            for _ in pbar:
                if bs < total_points:
                    self._key, subkey = jax.random.split(self._key)
                    idx = jax.random.choice(subkey, total_points, shape=(bs,), replace=False)
                    batch = (inputs[idx], targets[idx])
                else:
                    batch = (inputs, targets)
                loss_dict, model, opt = opt.update(model, batch)
                loss_val = float(loss_dict["total"])
                pbar.set_postfix(loss=f"{loss_val:.2e}")
                if loss_val < best_loss:
                    best_model, best_loss = model, loss_val
                if loss_val < self.threshold:
                    break

            n_iters_polish = n_iters_lbfgs if self.polish_optimizer == "lbfgs" else n_iters_gn
            if n_iters_polish > 0 and (loss_val is None or loss_val >= self.threshold):
                if self.polish_optimizer == "lbfgs":
                    full_batch = (inputs, targets)
                    lbfgs_opt = ScimbaLBfgs(model, _losses_function)
                    pbar = _training_progress(n_iters_lbfgs, f"{desc}[L-BFGS]", self.verbose)
                    for _ in pbar:
                        loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
                        loss_val = float(loss_dict["total"])
                        pbar.set_postfix(loss=f"{loss_val:.2e}")
                        if loss_val < best_loss:
                            best_model, best_loss = model, loss_val
                        if loss_val < self.threshold:
                            break
                else:  # gauss_newton
                    def make_data(n_map, k):
                        k, subkey = jax.random.split(k)
                        idx = jax.random.choice(subkey, total_points, shape=(n_map,), replace=(n_map > total_points))
                        return inputs[idx], targets[idx], k

                    probe("pre_gn", rank, self.n_calls)
                    model, gn_loss_hist = train_map_gn(
                        self.gn_n_map, make_data, model,
                        n_iterations=n_iters_gn, init_damping=self.gn_init_damping,
                        chunk_size=self.gn_chunk_size,
                    )
                    probe("post_gn", rank, self.n_calls)
                    gn_loss_hist = np.asarray(gn_loss_hist)
                    if gn_loss_hist.size:
                        best_gn_loss = float(gn_loss_hist.min())
                        loss_val = float(gn_loss_hist[-1])
                        if best_gn_loss < best_loss:
                            best_model, best_loss = model, best_gn_loss

            return best_model, opt, best_loss

    return ProbedOnlineNeuralNetworkCompressor


def worker_main(args):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.makedirs(args.out_dir, exist_ok=True)
    os.chdir(args.out_dir)

    ProbedOnlineNeuralNetworkCompressor = build_probed_compressor_class()

    cfg = load_production_config(args.params_yaml)
    nx, ny, nvx, nvy = local_chunk_shape(cfg, args.rank, args.nprocs)
    print(f"[rank {args.rank}] local chunk (nx,ny,nvx,nvy)=({nx},{ny},{nvx},{nvy}) "
          f"-> {nx*ny*nvx*nvy:,} points", flush=True)

    local_fdistribu = make_synthetic_local_chunk(args.n_species, nx, ny, nvx, nvy)

    compressor = ProbedOnlineNeuralNetworkCompressor(
        arch=args.arch,
        lr=args.lr,
        batch_size=args.batch_size,
        warm_iters_adam=args.warm_iters_adam,
        warm_iters_lbfgs=args.warm_iters_lbfgs,
        refine_iters_adam=args.refine_iters_adam,
        refine_iters_lbfgs=args.refine_iters_lbfgs,
        polish_optimizer=args.polish_optimizer,
        warm_iters_gn=args.warm_iters_gn,
        refine_iters_gn=args.refine_iters_gn,
        gn_n_map=args.gn_n_map,
        gn_init_damping=args.gn_init_damping,
        gn_chunk_size=args.gn_chunk_size,
        seed=42 + args.rank,  # mirrors compression_config.py's build_online_compressor rank-seed convention
        verbose=args.verbose,
    )

    for call_idx in range(args.n_calls):
        t0 = time.perf_counter()
        try:
            compressor.compress_decompress_array(local_fdistribu, rank=args.rank)
        except Exception as e:
            print(f"[rank {args.rank}] call {call_idx} FAILED after {time.perf_counter() - t0:.2f}s: "
                  f"{type(e).__name__}: {str(e)[:300]}", flush=True)
            sys.exit(1)
        print(f"[rank {args.rank}] call {call_idx} OK in {time.perf_counter() - t0:.2f}s", flush=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nprocs", type=int, default=4, help="Concurrent rank processes sharing the GPU.")
    parser.add_argument("--n-calls", type=int, default=2,
                         help="compress_decompress_array calls per rank (call 0 = cold start, rest = warm refine).")
    parser.add_argument("--n-species", type=int, default=1)
    parser.add_argument("--arch", default="periodic_siren_small_32_l5")
    parser.add_argument("--polish-optimizer", default="gauss_newton", choices=["lbfgs", "gauss_newton"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--warm-iters-adam", type=int, default=5000)
    parser.add_argument("--warm-iters-lbfgs", type=int, default=100)
    parser.add_argument("--refine-iters-adam", type=int, default=500)
    parser.add_argument("--refine-iters-lbfgs", type=int, default=10)
    parser.add_argument("--warm-iters-gn", type=int, default=150)
    parser.add_argument("--refine-iters-gn", type=int, default=30)
    parser.add_argument("--gn-n-map", type=int, default=16000)
    parser.add_argument("--gn-init-damping", type=float, default=1e-2)
    parser.add_argument("--gn-chunk-size", type=int, default=300)
    parser.add_argument("--params-yaml", default=DEFAULT_PARAMS_YAML)
    parser.add_argument("--out-dir", default="gpu_memory_probe_out")
    parser.add_argument("--verbose", action="store_true")
    # Internal, set only on the workers this script spawns itself:
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.worker:
        worker_main(args)
        return

    print(f"Spawning {args.nprocs} concurrent rank processes (no MPI -- real N-way GPU "
          f"contention, same failure mode as the real online pipeline) ...")
    shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    base_argv = sys.argv[1:] + ["--worker"]
    procs = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__)] + base_argv + ["--rank", str(r)])
        for r in range(args.nprocs)
    ]
    failed = [r for r, p in enumerate(procs) if p.wait() != 0]

    print("\n=== Summary ===")
    print(f"Ranks that FAILED (likely OOM): {failed}" if failed else "All ranks completed all calls without error.")

    worst = {}
    for r in range(args.nprocs):
        csv_path = os.path.join(args.out_dir, f"gpu_memory_probe_rank{r:03d}.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not row["nvidia_smi_free_mib"]:
                    continue
                free = int(row["nvidia_smi_free_mib"])
                if row["tag"] not in worst or free < worst[row["tag"]][0]:
                    worst[row["tag"]] = (free, r, row["call_idx"])
    for tag, (free, r, call_idx) in sorted(worst.items()):
        print(f"  worst-case free at '{tag}': {free} MiB (rank {r}, call {call_idx})")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
