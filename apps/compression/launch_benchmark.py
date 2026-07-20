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


def create_yaml_override(
    base_yaml_path,
    output_yaml_path,
    nb_restart,
    fdist_file,
    nbiter,
    iter_offset,
    compression_period=0,
    compression_mode=0,
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
    config["CompressionBenchmark"]["compression_period"] = compression_period
    config["CompressionBenchmark"]["compression_mode"] = compression_mode

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

def print_branch_banner(branch_name):
    label = f" {branch_name} "
    width = max(70, len(label) + 4)
    bar = "=" * (width - 2)
    cyan = "\033[96m"
    bold = "\033[1m"
    reset = "\033[0m"
    print()
    print(f"{cyan}{bold}+{bar}+{reset}")
    print(f"{cyan}{bold}|{label.center(width - 2)}|{reset}")
    print(f"{cyan}{bold}+{bar}+{reset}")
    print()


def run_sim_with_diagnostics(branch_name, gysela_yaml, pdi_yaml, work_dir, n_workers=1, extra_env=None):
    """Start Dask, run simulation and diagnostics in parallel, stop Dask.

    Both the simulation and diagnostics.py run with work_dir as CWD so that
    HDF5 snapshots and diagnostics.csv are written there.
    """
    print_branch_banner(branch_name)
    os.makedirs(work_dir, exist_ok=True)

    deisa_env = load_deisa_env()
    sch_proc, worker_proc, deisa_env = start_dask(deisa_env, n_workers)

    if extra_env:
        deisa_env = dict(deisa_env)
        deisa_env.update({k: str(v) for k, v in extra_env.items()})

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


def run_offline_compressed_branch(run_dir, run_pdi_yaml, iter_total, compression_period, mesh_kwargs=None, n_workers=1):
    dir_offline = os.path.join(run_dir, "branch_offline_compressed")
    yaml_offline = os.path.join(run_dir, "config_offline_compressed.yaml")

    create_yaml_override(
        SOURCE_GYSELA_YAML,
        yaml_offline,
        nb_restart=0,
        fdist_file="none",
        nbiter=iter_total,
        iter_offset=0,
        compression_period=compression_period,
        compression_mode=2,
    )

    extra_env = {"COMPRESSION_MESH_KWARGS": json.dumps(mesh_kwargs)} if mesh_kwargs else None

    run_sim_with_diagnostics(
        branch_name="Offline compressed",
        gysela_yaml=yaml_offline,
        pdi_yaml=run_pdi_yaml,
        work_dir=dir_offline,
        n_workers=n_workers,
        extra_env=extra_env,
    )

    events = _collect_offline_compression_events(dir_offline)
    write_compression_manifest(run_dir, events)

    return dir_offline


def _collect_offline_compression_events(work_dir):
    csv_path = os.path.join(work_dir, "compression_events_offline.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


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
        compression_period=compression_period,
        compression_mode=1,
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
        run_offline_compressed_branch(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            compression_period=compression_period,
            mesh_kwargs=mesh_kwargs,
            n_workers=args.dask_workers,
        )

    if not args.keep_pdi_copy:
        remove_file_if_exists(run_pdi_yaml, "copied PDI config")

    print(f"\nWorkflow complete. All files are in: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run_dir = main()
    compare_results(run_dir)
