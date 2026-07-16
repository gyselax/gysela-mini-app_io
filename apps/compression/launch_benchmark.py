#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import yaml
import csv

# ------------------------------------------------------------------
# Compression params / names
# ------------------------------------------------------------------
from evaluate_compression import plot_diags
from compression_config import build_offline_compressor


EXEC_CMD = ["mpirun", "-n", "4", "./build/apps/compression/gys_compress"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SOURCE_GYSELA_YAML = os.path.join(SCRIPT_DIR, "params_landau_damping.yaml")
SOURCE_PDI_YAML = os.path.join(SCRIPT_DIR, "pdi_out_diags.yaml")
ANALYTICS_SCRIPT = os.path.join(BASE_DIR, "src", "python", "diagnostics.py")
COMPRESSION_DIAGNOSTICS_SCRIPT = os.path.join(os.path.dirname(ANALYTICS_SCRIPT), "compression_diagnostics.py")
ACTIVATE_SCRIPT = os.path.join(BASE_DIR, "apps", "io", "activate_deisa_spack_env.sh")
SCHEFILE = os.path.join(BASE_DIR, "scheduler.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the compression benchmark pipeline.")

    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help=(
            "Output compression_run directory. "
            "If omitted, a timestamped compression_run_YYYYMMDD_HHMMSS "
            "directory is created in the project root."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow using an existing non-empty run directory. "
            "Without this option, the script refuses to write into "
            "a non-empty directory."
        ),
    )

    parser.add_argument(
        "--keep-payloads",
        action="store_true",
        help=(
            "Keep restart_iter_XXXXX_compressed.npz payload files. "
            "By default, they are deleted after their size and metrics are "
            "stored in compression_events.yaml."
        ),
    )

    parser.add_argument(
        "--keep-restart-approximations",
        action="store_true",
        help=(
            "Keep restart_iter_XXXXX_approx.h5 files after they have been "
            "used for restart. By default, each approximation is deleted after "
            "the segment that consumes it has finished."
        ),
    )

    parser.add_argument(
        "--keep-segment-configs",
        action="store_true",
        help=(
            "Keep config_compressed_segment_XXX.yaml files. "
            "By default, segment configs are removed after each segment run."
        ),
    )

    parser.add_argument(
        "--keep-pdi-copy",
        action="store_true",
        help=(
            "Keep the copied pdi_out.yaml inside the run directory. "
            "By default, it is removed after the workflow finishes."
        ),
    )

    parser.add_argument(
        "--dask-workers",
        type=int,
        default=1,
        help="Number of Dask workers to launch per simulation segment (default: 1).",
    )

    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Use online in-situ compression instead of the offline."
        ),
    )

    return parser.parse_args()


def remove_file_if_exists(path, description):
    if path is None or path == "none":
        return

    if os.path.exists(path):
        print(f"  [Cleanup] Removing {description}: {os.path.basename(path)}")
        os.remove(path)


def create_or_select_run_dir(requested_run_dir=None, overwrite=False):
    if requested_run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(BASE_DIR, f"compression_run_{timestamp}")
    else:
        run_dir = os.path.abspath(requested_run_dir)

    if os.path.exists(run_dir):
        if not os.path.isdir(run_dir):
            raise RuntimeError(f"Requested run path exists but is not a directory: {run_dir}")

        if os.listdir(run_dir) and not overwrite:
            raise RuntimeError(
                f"Run directory already exists and is not empty: {run_dir}\n"
                "Use --overwrite if you intentionally want to write into it."
            )
    else:
        os.makedirs(run_dir, exist_ok=True)

    return run_dir


def assert_file_exists(path, description):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing {description}: {path}")

def read_mesh_config(config):
    """Extracts the grid boundaries from the GYSELA YAML file"""
    try:
        mesh = config["SplineMesh"]
        return {
            "x_min": float(mesh["x_min"]),
            "x_max": float(mesh["x_max"]),
            "y_min": float(mesh["y_min"]),
            "y_max": float(mesh["y_max"]),
            "vx_min": float(mesh["vx_min"]),
            "vx_max": float(mesh["vx_max"]),
            "vy_min": float(mesh["vy_min"]),
            "vy_max": float(mesh["vy_max"]),
        }
    except KeyError as exc:
        raise RuntimeError("Missing SplineMesh parameters in the GYSELA YAML template.") from exc

def read_benchmark_config(config):
    try:
        iter_total = int(config["Algorithm"]["nbiter"])
        compression_period = int(config["CompressionBenchmark"]["compression_period"])
    except KeyError as exc:
        raise RuntimeError(
            "Missing required benchmark parameter in the GYSELA input template "
            f"({os.path.basename(SOURCE_GYSELA_YAML)}). "
            "Expected Algorithm.nbiter and CompressionBenchmark.compression_period."
        ) from exc

    if iter_total <= 0:
        raise RuntimeError(f"Algorithm.nbiter must be positive. Got {iter_total}.")

    if compression_period <= 0:
        raise RuntimeError(f"CompressionBenchmark.compression_period must be positive. " f"Got {compression_period}.")

    if compression_period >= iter_total:
        raise RuntimeError(
            f"Compression period must be smaller than total iterations. "
            f"Got compression_period={compression_period}, nbiter={iter_total}."
        )

    return iter_total, compression_period


def compute_diagnostic_step(config):
    nbstep_diag = int(config["Output"]["nbiter_diag"])

    if nbstep_diag <= 0:
        raise RuntimeError(f"Invalid diagnostic step: Output.nbiter_diag={nbstep_diag} must be positive.")

    return nbstep_diag


def assert_iterations_are_diagnostic_outputs(iter_total, compression_period, nbstep_diag):
    if compression_period % nbstep_diag != 0:
        raise RuntimeError(
            f"Compression period must be a multiple of nbstep_diag={nbstep_diag}. "
            f"Got compression_period={compression_period}."
        )

    if iter_total % nbstep_diag != 0:
        raise RuntimeError(
            f"Algorithm.nbiter must be a multiple of nbstep_diag={nbstep_diag}. " f"Got nbiter={iter_total}."
        )


def format_param_summary(metrics):
    params = metrics.get("params") or {}

    if not params:
        return "no parameters"

    return ", ".join(f"{key}={value}" for key, value in params.items())


def compress_decompress(input_h5, output_h5, compressed_path, compressor_kwargs):
    compressor = build_offline_compressor(**compressor_kwargs)

    print(
        f"  [{compressor.method_name} Compression] "
        f"{os.path.basename(input_h5)} -> {os.path.basename(output_h5)}"
    )
    print(f"  [Parameters] {compressor.printable_name()}")
    print(f"  [Compressed Payload] {os.path.basename(compressed_path)}")

    metrics = compressor.compress_decompress_h5(
        input_h5=input_h5,
        output_h5=output_h5,
        compressed_path=compressed_path,
    )

    print(f"  Method = {metrics['method_name']} ({format_param_summary(metrics)})")

    explained = metrics.get("explained_variance_ratio_sum")
    if explained is not None:
        print(f"  Explained variance ratio sum = {explained:.12e}")

    print("  Relative L2 reconstruction error = " f"{metrics['relative_l2_error']:.12e}")
    print("  Max abs reconstruction error = " f"{metrics['max_abs_error']:.12e}")

    if metrics.get("mean_abs_error") is not None:
        print("  Mean abs reconstruction error = " f"{metrics['mean_abs_error']:.12e}")

    if metrics.get("rmse") is not None:
        print("  RMSE reconstruction error = " f"{metrics['rmse']:.12e}")

    if metrics["compression_ratio"] is not None:
        print(f"  Compression ratio = {metrics['compression_ratio']:.6f}x")

    return metrics


def create_yaml_override(
    base_yaml_path,
    output_yaml_path,
    nb_restart,
    fdist_file,
    nbiter,
    iter_offset,
    compression_period_online=0,
):
    with open(base_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("Input", {})
    config["Input"]["nb_restart"] = nb_restart
    config["Input"]["fdistribu_filename"] = fdist_file
    config["Input"]["iter_offset"] = iter_offset

    config.setdefault("Algorithm", {})
    config["Algorithm"]["nbiter"] = nbiter

    config.setdefault("CompressionBenchmark", {})
    config["CompressionBenchmark"]["compression_period"] = compression_period_online

    with open(output_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ------------------------------------------------------------------
# Dask infrastructure
# ------------------------------------------------------------------

def load_deisa_env():
    """Source the spack+venv activation script and capture the resulting environment."""
    if not os.path.exists(ACTIVATE_SCRIPT):
        raise RuntimeError(
            f"Deisa activation script not found: {ACTIVATE_SCRIPT}\n"
            "Ensure the spack environment and venv are set up as described "
            "in apps/io/README.md."
        )
    result = subprocess.run(
        ["bash", "-c",
         f'. "{ACTIVATE_SCRIPT}" && python3 -c '
         '"import os,json,sys; sys.stdout.write(json.dumps(dict(os.environ)))"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def start_dask(deisa_env, n_workers=1):
    """Start Dask scheduler and workers. Returns (sch_proc, worker_proc, updated_env)."""
    if os.path.exists(SCHEFILE):
        os.remove(SCHEFILE)

    sch_proc = subprocess.Popen(
        ["dask-scheduler", f"--scheduler-file={SCHEFILE}"],
        env=deisa_env,
    )

    print("  Waiting for Dask scheduler", end="", flush=True)
    deadline = time.time() + 60
    while not os.path.exists(SCHEFILE):
        if time.time() > deadline:
            sch_proc.kill()
            raise RuntimeError("Dask scheduler did not start within 60 seconds.")
        time.sleep(1)
        print(".", end="", flush=True)
    print(" ready")

    with open(SCHEFILE) as f:
        scheduler_address = json.load(f)["address"]

    deisa_env = dict(deisa_env)
    deisa_env["DEISA_DASK_SCHEDULER_ADDRESS"] = scheduler_address

    worker_proc = subprocess.Popen(
        [
            "dask-worker",
            f"--nworkers={n_workers}",
            "--local-directory=/tmp",
            f"--scheduler-file={SCHEFILE}",
        ],
        env=deisa_env,
    )

    print(f"  Waiting 10 s for {n_workers} worker(s) to connect...")
    time.sleep(10)

    return sch_proc, worker_proc, deisa_env


def stop_dask(sch_proc, worker_proc):
    """Kill Dask scheduler and workers."""
    for proc in (worker_proc, sch_proc):
        if proc is not None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass


def run_sim_with_diagnostics(branch_name, gysela_yaml, pdi_yaml, work_dir, n_workers=1):
    """Start Dask, run simulation and diagnostics in parallel, stop Dask.

    Both the simulation and diagnostics.py run with work_dir as CWD so that
    HDF5 snapshots and diagnostics.csv are written there.
    """
    print(f"\n--- Running: {branch_name} ---")
    os.makedirs(work_dir, exist_ok=True)

    deisa_env = load_deisa_env()
    sch_proc, worker_proc, deisa_env = start_dask(deisa_env, n_workers)

    try:
        exec_path = os.path.abspath(EXEC_CMD[-1])
        abs_gysela = os.path.abspath(gysela_yaml)
        abs_pdi = os.path.abspath(pdi_yaml)
        sim_cmd = EXEC_CMD[:-1] + [exec_path, abs_gysela, abs_pdi]

        analytics_proc = subprocess.Popen(
            ["python3", ANALYTICS_SCRIPT],
            cwd=work_dir,
            env=deisa_env,
        )

        sim_proc = subprocess.Popen(
            sim_cmd,
            cwd=work_dir,
            env=deisa_env,
        )

        analytics_rc = analytics_proc.wait()
        sim_rc = sim_proc.wait()

        if analytics_rc != 0:
            raise RuntimeError(
                f"Analytics process for '{branch_name}' exited with return code {analytics_rc}."
            )
        if sim_rc != 0:
            raise RuntimeError(
                f"Simulation '{branch_name}' exited with return code {sim_rc}."
            )
    finally:
        stop_dask(sch_proc, worker_proc)


def write_compression_manifest(run_dir, compression_events):
    manifest_path = os.path.join(run_dir, "compression_events.yaml")

    with open(manifest_path, "w") as f:
        yaml.dump(
            {"compression_events": compression_events},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"\nCompression event manifest written to: {manifest_path}")


def run_baseline(run_dir, run_pdi_yaml, iter_total, n_workers=1):
    dir_baseline = os.path.join(run_dir, "branch_baseline")
    yaml_baseline = os.path.join(run_dir, "config_baseline.yaml")

    create_yaml_override(
        SOURCE_GYSELA_YAML,
        yaml_baseline,
        nb_restart=0,
        fdist_file="none",
        nbiter=iter_total,
        iter_offset=0,
    )

    run_sim_with_diagnostics(
        branch_name="Baseline",
        gysela_yaml=yaml_baseline,
        pdi_yaml=run_pdi_yaml,
        work_dir=dir_baseline,
        n_workers=n_workers,
    )

    return dir_baseline


def run_periodic_compressed_branch(
    run_dir,
    run_pdi_yaml,
    iter_total,
    compression_period,
    mesh_kwargs,
    n_workers=1,
    keep_payloads=False,
    keep_restart_approximations=False,
    keep_segment_configs=False,
):
    dir_compressed = os.path.join(run_dir, "branch_compressed")
    restart_dir = os.path.join(run_dir, "periodic_restarts")

    os.makedirs(dir_compressed, exist_ok=True)
    os.makedirs(restart_dir, exist_ok=True)

    current_iter = 0
    segment_id = 0
    restart_file = "none"
    compression_events = []
    
    #tracking the previous payload for warm-starting
    previous_compressed_payload = None

    while current_iter < iter_total:
        remaining_iter = iter_total - current_iter
        segment_nbiter = min(compression_period, remaining_iter)
        next_iter = current_iter + segment_nbiter

        yaml_segment = os.path.join(
            run_dir,
            f"config_compressed_segment_{segment_id:03d}.yaml",
        )

        nb_restart = 0 if segment_id == 0 else segment_id
        restart_file_used_by_segment = restart_file

        create_yaml_override(
            SOURCE_GYSELA_YAML,
            yaml_segment,
            nb_restart=nb_restart,
            fdist_file=restart_file,
            nbiter=segment_nbiter,
            iter_offset=current_iter,
        )

        run_sim_with_diagnostics(
            branch_name=(f"Compressed segment {segment_id} ({current_iter} -> {next_iter})"),
            gysela_yaml=yaml_segment,
            pdi_yaml=run_pdi_yaml,
            work_dir=dir_compressed,
            n_workers=n_workers,
        )

        if nb_restart > 0 and not keep_restart_approximations:
            remove_file_if_exists(
                restart_file_used_by_segment,
                "consumed restart approximation",
            )

        if not keep_segment_configs:
            remove_file_if_exists(
                yaml_segment,
                "temporary compressed-segment config",
            )

        current_iter = next_iter

        if current_iter >= iter_total:
            break

        if current_iter % compression_period != 0:
            raise RuntimeError(
                f"Cannot compress at iteration {current_iter}: "
                f"not a multiple of compression_period={compression_period}."
            )

        raw_restart = os.path.join(
            dir_compressed,
            f"GYSELALIBXX_{current_iter:05d}.h5",
        )

        approx_restart = os.path.join(
            restart_dir,
            f"restart_iter_{current_iter:05d}_approx.h5",
        )

        compressed_payload = os.path.join(
            restart_dir,
            f"restart_iter_{current_iter:05d}_compressed.npz",
        )

        assert_file_exists(
            raw_restart,
            f"restart source file at iteration {current_iter}",
        )

        raw_restart_size = os.path.getsize(raw_restart)
        
        # preparing dynamic arguments for the compressor
        compressor_kwargs = {**mesh_kwargs}
        if previous_compressed_payload is not None:
            compressor_kwargs["warm_start_payload"] = previous_compressed_payload

        metrics = compress_decompress(
            input_h5=raw_restart,
            output_h5=approx_restart,
            compressed_path=compressed_payload,
            compressor_kwargs=compressor_kwargs,
        )
        
        # Compression is complete: we can now safely delete the old payload
        if previous_compressed_payload is not None and not keep_payloads:
            remove_file_if_exists(
                previous_compressed_payload,
                "consumed compressed INR payload",
            )

        approx_restart_size = os.path.getsize(approx_restart) if os.path.exists(approx_restart) else None

        compressed_payload_size = os.path.getsize(compressed_payload) if os.path.exists(compressed_payload) else None

        compression_events.append(
            {
                "segment_id": segment_id,
                "iteration": current_iter,
                "file_index": current_iter,
                "branch_restart": os.path.relpath(raw_restart, run_dir),
                "approx_restart": os.path.relpath(approx_restart, run_dir),
                "compressed_payload": (os.path.relpath(compressed_payload, run_dir) if keep_payloads else None),
                "method_name": metrics.get("method_name"),
                "param_names": metrics.get("param_names", []),
                "params": metrics.get("params", {}),
                "n_components": metrics.get("param_n_components"),
                "raw_restart_size": raw_restart_size,
                "approx_restart_size": approx_restart_size,
                "compressed_payload_size": compressed_payload_size,
                "explained_variance_ratio_sum": metrics.get("explained_variance_ratio_sum"),
                "relative_l2_error": float(metrics["relative_l2_error"]),
                "max_abs_error": float(metrics["max_abs_error"]),
                "mean_abs_error": float(metrics["mean_abs_error"]),
                "rmse": float(metrics["rmse"]),
                "compression_seconds": metrics.get("compression_seconds"),
                "decompression_seconds": metrics.get("decompression_seconds"),
                "compression_ratio": (
                    None if metrics["compression_ratio"] is None else float(metrics["compression_ratio"])
                ),
                "approx_restart_kept": keep_restart_approximations,
                "compressed_payload_kept": keep_payloads,
            }
        )
        """ 
        if not keep_payloads:
            remove_file_if_exists(
                compressed_payload,
                "compressed PCA payload",
            )
        """
        remove_file_if_exists(raw_restart, "consumed raw restart")

        restart_file = os.path.abspath(approx_restart)
        #save the path of the new payload for the next segment
        previous_compressed_payload = os.path.abspath(compressed_payload)
        
        segment_id += 1

    if previous_compressed_payload is not None and not keep_payloads:
        remove_file_if_exists(
            previous_compressed_payload,
            "final consumed compressed INR payload",
        )
    
    write_compression_manifest(run_dir, compression_events)

    return dir_compressed

def run_online_compressed_branch(run_dir, run_pdi_yaml, iter_total, compression_period, n_workers=1):
    dir_online = os.path.join(run_dir, "branch_online_compressed")
    yaml_online = os.path.join(run_dir, "config_online_compressed.yaml")

    create_yaml_override(
        SOURCE_GYSELA_YAML,
        yaml_online,
        nb_restart=0,
        fdist_file="none",
        nbiter=iter_total,
        iter_offset=0,
        compression_period_online=compression_period,
    )

    run_sim_with_diagnostics(
        branch_name="Online compressed",
        gysela_yaml=yaml_online,
        pdi_yaml=run_pdi_yaml,
        work_dir=dir_online,
        n_workers=n_workers,
    )

    events = _collect_online_compression_events(dir_online)
    write_compression_manifest(run_dir, events)

    return dir_online


def _collect_online_compression_events(work_dir):
    events = []
    for entry in sorted(os.listdir(work_dir)):
        if entry.startswith("compression_events_rank") and entry.endswith(".csv"):
            with open(os.path.join(work_dir, entry), newline="") as fh:
                events.extend(dict(row) for row in csv.DictReader(fh))
    return events


def compare_results(run_dir):
    diag_files = []
    for entry in sorted(os.listdir(run_dir)):
        csv_path = os.path.join(run_dir, entry, "diagnostics.csv")
        if os.path.exists(csv_path):
            diag_files.append(csv_path)

    if not diag_files:
        print("\nNo diagnostics.csv files found — skipping comparison plot.")
        return

    output = os.path.join(run_dir, "diags_comparison.png")
    plot_diags(diag_files, output=output)


def main():
    args = parse_args()

    assert_file_exists(SOURCE_GYSELA_YAML, "base GYSELA input template")
    assert_file_exists(SOURCE_PDI_YAML, "base PDI input template")
    assert_file_exists(ANALYTICS_SCRIPT, "in-situ diagnostics script")
    if args.online:
        assert_file_exists(COMPRESSION_DIAGNOSTICS_SCRIPT, "online in-situ compression script")

    run_dir = create_or_select_run_dir(
        requested_run_dir=args.run_dir,
        overwrite=args.overwrite,
    )

    print(f"Master directory initialised: {run_dir}")

    run_pdi_yaml = os.path.join(run_dir, "pdi_out_diags.yaml")
    shutil.copy2(SOURCE_PDI_YAML, run_pdi_yaml)

    with open(SOURCE_GYSELA_YAML, "r") as f:
        base_cfg = yaml.safe_load(f)

    iter_total, compression_period = read_benchmark_config(base_cfg)
    nbstep_diag = compute_diagnostic_step(base_cfg)
    mesh_kwargs = read_mesh_config(base_cfg)

    assert_iterations_are_diagnostic_outputs(
        iter_total=iter_total,
        compression_period=compression_period,
        nbstep_diag=nbstep_diag,
    )

    print(f"Total iterations       : {iter_total}")
    print(f"Compression period K   : {compression_period}")
    print(f"Diagnostic step        : {nbstep_diag}")

    if args.overwrite:
        print("\n--- Skipping reference simulation (--overwrite): reusing existing baseline ---")
    else:
        run_baseline(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            n_workers=args.dask_workers,
        )

    if args.online:
        run_online_compressed_branch(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            compression_period=compression_period,
            n_workers=args.dask_workers,
        )
    else:
        run_periodic_compressed_branch(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            compression_period=compression_period,
            mesh_kwargs=mesh_kwargs,
            n_workers=args.dask_workers,
            keep_payloads=args.keep_payloads,
            keep_restart_approximations=args.keep_restart_approximations,
            keep_segment_configs=args.keep_segment_configs,
        )

    if not args.keep_pdi_copy:
        remove_file_if_exists(run_pdi_yaml, "copied PDI config")

    

    print(f"\nWorkflow complete. All files are in: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_dir = main()
    compare_results(run_dir)
