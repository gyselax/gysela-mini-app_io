#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime

import yaml
import csv

# ------------------------------------------------------------------
# Compression params / names
# ------------------------------------------------------------------
from evaluate_compression import plot_diags, plot_final_snapshot_comparison


GYS_COMPRESS_BIN = "./build/apps/compression/gys_compress"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SOURCE_GYSELA_YAML = os.path.join(SCRIPT_DIR, "params_two_stream.yaml")
SOURCE_PDI_YAML = os.path.join(SCRIPT_DIR, "pdi_out_diags.yaml")
ANALYTICS_SCRIPT = os.path.join(BASE_DIR, "src", "python", "diagnostics.py")
COMPRESSION_DIAGNOSTICS_SCRIPT = os.path.join(os.path.dirname(ANALYTICS_SCRIPT), "compression_diagnostics.py")

# ------------------------------------------------------------------
# Toolchain resolution: a single --arch flag picks
# toolchains/<site>/<arch>/environment.sh, where <site> is chosen using
# resolve_site.
# On Adastra, --arch also becomes 'srun/#SBATCH --constraint='.
# ------------------------------------------------------------------
ENV_SCRIPT = None  # resolved from --arch
VENV_ACTIVATE = None  # resolved from --venv

# 3 distinct nodes on Adastra: one for the Dask scheduler, the Dask
# worker(s), and the simulation.
ADASTRA_NODES = 3
DEFAULT_N_PROCS = 4
ADASTRA_N_PROCS = None  # simulation MPI ranks -- resolved from --nprocs, see configure_toolchain()

GENOA_LOGICAL_THREADS_PER_NODE = 384
DEFAULT_SIM_OMP_NUM_THREADS = 1

SIM_OMP_THREADS = None  # resolved from --arch, see configure_toolchain()

# Dask worker processes on its dedicated node: 1 per CPU socket
WORKER_RANKS_PER_NODE_BY_ARCH = {"genoa": 2, "mi250": 1}
DEFAULT_WORKER_RANKS_PER_NODE = 2


def resolve_site():
    """Which toolchains/<site>/ subtree we're on.
    Confirm via Slurm's ClusterName (`scontrol show config`).
    """
    scontrol = shutil.which("scontrol")
    if scontrol:
        try:
            result = subprocess.run(
                [scontrol, "show", "config"], capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and "adastra" in result.stdout.lower():
                return "adastra"
        except (OSError, subprocess.TimeoutExpired):
            pass

    if "adastra" in socket.gethostname().lower():
        return "adastra"

    return "persee"


def configure_toolchain(args):
    """Resolve the toolchains/<site>/<arch>/environment.sh, venv activate path,
    the simulation's MPI rank count, and its OMP_NUM_THREADS, from --arch/--venv/--nprocs."""
    global ENV_SCRIPT, VENV_ACTIVATE, SIM_OMP_THREADS, ADASTRA_N_PROCS
    ENV_SCRIPT = os.path.join(BASE_DIR, "toolchains", resolve_site(), args.arch.lower(), "environment.sh")
    venv_dir = os.path.abspath(args.venv) if args.venv else os.path.join(BASE_DIR, ".gys_env")
    VENV_ACTIVATE = os.path.join(venv_dir, "bin", "activate")

    ADASTRA_N_PROCS = args.nprocs

    arch_lower = args.arch.lower()
    if arch_lower == "genoa":
        SIM_OMP_THREADS = max(1, GENOA_LOGICAL_THREADS_PER_NODE // ADASTRA_N_PROCS)
    elif arch_lower == "mi250":
        SIM_OMP_THREADS = 1
    else:
        SIM_OMP_THREADS = DEFAULT_SIM_OMP_NUM_THREADS


def on_adastra():
    """sbatch on PATH and not already inside a job => submit ourselves as a batch job."""
    return resolve_site() == "adastra" and "SLURM_JOB_ID" not in os.environ


def effective_n_workers(args):
    """Dask worker process count: explicit --dask-workers, else 1 rank/socket by --arch on Adastra, else 1."""
    if args.dask_workers is not None:
        return args.dask_workers
    if resolve_site() == "adastra":
        return WORKER_RANKS_PER_NODE_BY_ARCH.get(args.arch.lower(), DEFAULT_WORKER_RANKS_PER_NODE)
    return 1


def resolve_role_nodes():
    """(scheduler_node, worker_node, sim_node) from the current Slurm allocation, or all-None outside one."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return None, None, None

    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True
    )
    nodes = [h for h in result.stdout.split() if h]
    if len(nodes) < 3:
        raise RuntimeError(
            f"Need 3 distinct nodes (scheduler, worker, simulation); "
            f"allocation only has {len(nodes)}. Check ADASTRA_NODES / #SBATCH --nodes."
        )
    return nodes[0], nodes[1], nodes[2]


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
        default=None,
        help=(
            "Number of Dask worker processes to launch on the dedicated "
            "Dask worker node. If not given: on Adastra, defaults to 1 "
            "rank per CPU socket for --arch "
            f"({WORKER_RANKS_PER_NODE_BY_ARCH}, or {DEFAULT_WORKER_RANKS_PER_NODE} "
            "for an unlisted --arch); on persee, defaults to 1."
        ),
    )

    parser.add_argument(
        "--nprocs",
        type=int,
        default=DEFAULT_N_PROCS,
        help=(
            "Number of MPI ranks to launch the simulation with "
            f"(default: {DEFAULT_N_PROCS}). On Adastra (GENOA), also sets "
            "OMP_NUM_THREADS per rank to threads_per_node/nprocs."
        ),
    )

    parser.add_argument(
        "--online",
        action="store_true",
        help=(
            "Use online in-situ compression instead of the offline."
        ),
    )
    
    parser.add_argument(
        "--compression", 
        type=str, 
        choices=["POD", "NN"], 
        default=None,
        help="Override for the offline compressor defined in compression_config.py."
    )
    
    parser.add_argument(
        "--rank", 
        type=int, 
        default=32, 
        help="n_components for POD."
    )
    
    parser.add_argument(
        "--arch-nn", 
        type=str, 
        default="periodic_siren_deep_128",
        dest="arch_nn",
        choices=["periodic_siren_deep_128", "periodic_fourier_mlp_deep_128"],
        help="INR architecture (if --compression NN)"
    )

    parser.add_argument(
        "--arch",
        default="GENOA" if resolve_site() == "adastra" else "xeon",
        help=(
            "Node arch/toolchain to activate: toolchains/<site>/<arch>/"
            "environment.sh, where <site> is 'adastra' or 'persee' depending on "
            "the current machine. On Adastra, also used for 'srun/#SBATCH "
            "--constraint=' and to pick the Dask worker's rank count. "
            "Default: GENOA on Adastra, xeon on persee."
        ),
    )

    parser.add_argument(
        "--time",
        default="01:00:00",
        help="Time limit for the Adastra batch job (HH:MM:SS). Default: 01:00:00.",
    )

    parser.add_argument(
        "--venv",
        default=None,
        help=(
            "Path (relative or absolute) to the personal venv directory "
            "to activate alongside the toolchain -- its bin/activate is "
            "sourced automatically, just point this at the venv directory "
            "itself. Default: '.gys_env' at the repo root."
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

        contents = [e for e in os.listdir(run_dir) if e not in ("batch.log", "submit.sbatch")]
        if contents and not overwrite:
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
        compression_period = int(config["compression"]["compression_period"])
    except KeyError as exc:
        raise RuntimeError(
            "Missing required benchmark parameter in the GYSELA input template "
            f"({os.path.basename(SOURCE_GYSELA_YAML)}). "
            "Expected Algorithm.nbiter and compression.compression_period."
        ) from exc

    if iter_total <= 0:
        raise RuntimeError(f"Algorithm.nbiter must be positive. Got {iter_total}.")

    if compression_period <= 0:
        raise RuntimeError(f"compression.compression_period must be positive. " f"Got {compression_period}.")

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
    """Source the toolchain environment.sh + personal venv and capture the resulting environment."""
    if not os.path.exists(ENV_SCRIPT):
        raise RuntimeError(
            f"Toolchain environment script not found: {ENV_SCRIPT}\n"
            "Check --arch matches an existing toolchains/<site>/<arch>/ directory."
        )

    source_cmds = f'. "{ENV_SCRIPT}" 1>&2'
    if os.path.isfile(VENV_ACTIVATE):
        source_cmds += f' && . "{VENV_ACTIVATE}" 1>&2'
    else:
        print(f"[Warning] Personal venv not found at {VENV_ACTIVATE}; skipping.")

    result = subprocess.run(
        ["bash", "-c",
         f'{source_cmds} && python3 -c '
         '"import os,json,sys; sys.stdout.write(json.dumps(dict(os.environ)))"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def start_dask(deisa_env, work_dir, n_workers=1):
    """Start Dask scheduler and workers. Returns (sch_proc, worker_proc, updated_env).

    The scheduler file lives inside work_dir (rather than a fixed repo-wide
    path) so that concurrent launch_benchmark.py runs against different
    run_dirs don't race on the same file -- each branch's own work_dir is
    unique per invocation.
    """
    schefile = os.path.join(work_dir, "scheduler.json")
    if os.path.exists(schefile):
        os.remove(schefile)

    sch_proc = subprocess.Popen(
        [
            "dask-scheduler",
            f"--scheduler-file={schefile}",
            "--port", "0",
            "--dashboard-address", ":0",
        ],
        env=deisa_env,
    )

    print("  Waiting for Dask scheduler", end="", flush=True)
    deadline = time.time() + 60
    while not os.path.exists(schefile):
        if time.time() > deadline:
            sch_proc.kill()
            raise RuntimeError("Dask scheduler did not start within 60 seconds.")
        time.sleep(1)
        print(".", end="", flush=True)
    print(" ready")

    with open(schefile) as f:
        scheduler_address = json.load(f)["address"]

    deisa_env = dict(deisa_env)
    deisa_env["DEISA_DASK_SCHEDULER_ADDRESS"] = scheduler_address

    worker_proc = subprocess.Popen(
        [
            "dask-worker",
            f"--nworkers={n_workers}",
            "--local-directory=/tmp",
            f"--scheduler-file={schefile}",
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


def sim_launch_prefix():
    """argv prefix to launch gys_compress: srun on the dedicated sim node inside our Adastra job, else mpirun."""
    if "SLURM_JOB_ID" in os.environ:
        _, _, sim_node = resolve_role_nodes()
        prefix = ["srun"]
        if sim_node:
            prefix += ["-w", sim_node]
        prefix += [
            "-N", "1", "--ntasks-per-node", str(ADASTRA_N_PROCS),
            "--cpus-per-task", str(SIM_OMP_THREADS),
        ]
        if SIM_OMP_THREADS > 1:
            prefix += ["--threads-per-core", "2"]
        prefix += ["--overlap"]
        return prefix
    return ["mpirun", "-n", str(ADASTRA_N_PROCS)]


def run_sim_with_diagnostics(branch_name, gysela_yaml, pdi_yaml, work_dir, n_workers=1, extra_env=None):
    """Start Dask, run simulation and diagnostics in parallel, stop Dask.

    Both the simulation and diagnostics.py run with work_dir as CWD so that
    HDF5 snapshots and diagnostics.csv are written there.
    """
    print_branch_banner(branch_name)
    os.makedirs(work_dir, exist_ok=True)

    deisa_env = load_deisa_env()
    sch_proc, worker_proc, deisa_env = start_dask(deisa_env, work_dir, n_workers)

    if extra_env:
        deisa_env = dict(deisa_env)
        deisa_env.update({k: str(v) for k, v in extra_env.items()})

    try:
        exec_path = os.path.abspath(GYS_COMPRESS_BIN)
        abs_gysela = os.path.abspath(gysela_yaml)
        abs_pdi = os.path.abspath(pdi_yaml)
        sim_cmd = sim_launch_prefix() + [exec_path, abs_gysela, abs_pdi]

        sim_env = deisa_env
        if "SLURM_JOB_ID" in os.environ:
            sim_env = dict(deisa_env)
            sim_env["OMP_NUM_THREADS"] = str(SIM_OMP_THREADS)
            sim_env["OMP_PROC_BIND"] = "CLOSE"
            sim_env["OMP_PLACES"] = "THREADS"

        analytics_proc = subprocess.Popen(
            ["python3", ANALYTICS_SCRIPT],
            cwd=work_dir,
            env=deisa_env,
        )

        sim_proc = subprocess.Popen(
            sim_cmd,
            cwd=work_dir,
            env=sim_env,
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


def run_offline_compressed_branch(run_dir, run_pdi_yaml, iter_total, compression_period,
                                   mesh_kwargs=None, method_override=None, n_workers=1):
    branch_name = _offline_branch_name(method_override)
    dir_offline = os.path.join(run_dir, branch_name)
    yaml_offline = os.path.join(run_dir, f"config_{branch_name}.yaml")

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

    extra_env = {}
    if mesh_kwargs:
        extra_env["COMPRESSION_MESH_KWARGS"] = json.dumps(mesh_kwargs)
    if method_override:
        extra_env["COMPRESSION_METHOD_OVERRIDE"] = json.dumps(method_override)

    run_sim_with_diagnostics(
        branch_name=branch_name.replace("branch_", ""),
        gysela_yaml=yaml_offline, 
        pdi_yaml=run_pdi_yaml, 
        work_dir=dir_offline,
        n_workers=n_workers, 
        extra_env=extra_env or None,
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

def _offline_branch_name(method_override):
    if not method_override:
        return "branch_offline_compressed"
    if method_override["class"] == "PCA":
        return f"branch_offline_compressed_POD_r{method_override['params']['n_components']}"
    if method_override["class"] == "NeuralNetwork":
        return f"branch_offline_compressed_NN_{method_override['params']['arch']}"
    return "branch_offline_compressed"


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

    snapshot_output = os.path.join(run_dir, "final_snapshot_comparison.png")
    plot_final_snapshot_comparison(run_dir, output=snapshot_output)


# ------------------------------------------------------------------
# Adastra batch submission
# ------------------------------------------------------------------

def _job_is_active(job_id):
    result = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
    return bool(result.stdout.strip())


def _get_job_state(job_id):
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "-X"],
        capture_output=True, text=True,
    )
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    return lines[0] if lines else None


def _stream_until_job_done(job_id, log_path):
    """Print new lines from log_path as they're written, until the job leaves the queue."""
    pos = 0
    while True:
        if os.path.exists(log_path):
            with open(log_path) as f:
                f.seek(pos)
                chunk = f.read()
                if chunk:
                    print(chunk, end="", flush=True)
                    pos = f.tell()
        if not _job_is_active(job_id):
            break
        time.sleep(2)

    if os.path.exists(log_path):
        with open(log_path) as f:
            f.seek(pos)
            print(f.read(), end="", flush=True)


def submit_batch_and_wait(args):
    """Submit ourselves as a single sbatch job on Adastra, then stream its output until it finishes."""
    account = os.environ.get("ACTIVE_PROJECT")
    if account:
        print(f"[Account] Using Slurm account: {account}")

    run_dir = create_or_select_run_dir(requested_run_dir=args.run_dir, overwrite=args.overwrite)
    print(f"Master directory initialised: {run_dir}")

    log_path = os.path.join(run_dir, "batch.log")
    script_path = os.path.join(run_dir, "submit.sbatch")

    forwarded = [shutil.which("python3") or "python3", os.path.abspath(__file__), run_dir]
    if args.overwrite:
        forwarded.append("--overwrite")
    if args.keep_pdi_copy:
        forwarded.append("--keep-pdi-copy")
    if args.online:
        forwarded.append("--online")
    if args.dask_workers is not None:
        forwarded += ["--dask-workers", str(args.dask_workers)]
    forwarded += ["--nprocs", str(args.nprocs)]
    forwarded += ["--arch", args.arch]
    if args.venv:
        forwarded += ["--venv", args.venv]

    n_workers = effective_n_workers(args)
    max_ranks_per_node = max(ADASTRA_N_PROCS, n_workers, 1)

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gys-compression-benchmark",
        f"#SBATCH --nodes={ADASTRA_NODES}",
        f"#SBATCH --ntasks-per-node={max_ranks_per_node}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --constraint={args.arch}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
    ]
    if account:
        sbatch_lines.append(f"#SBATCH --account={account}")
    sbatch_lines += ["", " ".join(f'"{a}"' for a in forwarded)]

    with open(script_path, "w") as f:
        f.write("\n".join(sbatch_lines) + "\n")

    print(f"[Batch] Submitting {script_path}")
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {result.returncode}):\n{result.stderr.strip() or result.stdout.strip()}"
        )
    print(f"  {result.stdout.strip()}")
    job_id = result.stdout.strip().split()[-1]

    print(f"[Batch] Streaming {log_path} until job {job_id} finishes...\n")
    _stream_until_job_done(job_id, log_path)

    state = _get_job_state(job_id)
    if state and not state.startswith("COMPLETED"):
        raise RuntimeError(f"Batch job {job_id} finished with state {state} (see {log_path}).")


def run_pipeline(args):
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
    
    comp_cfg = base_cfg.get("compression", {})
    selected_method = args.compression if args.compression else comp_cfg.get("method", "PCA")
    method_override = None
    
    if selected_method in ["POD", "PCA"]:
        pod_cfg = comp_cfg.get("POD", {})
        method_override = {
            "class": "PCA",
            "params": {
                "n_components": args.rank if args.rank else pod_cfg.get("n_components", 32),
                "normalisation": pod_cfg.get("normalisation", "none"),
                "clip_nonnegative": pod_cfg.get("clip_nonnegative", False),
            }
        }
    elif selected_method == "NN":
        nn_cfg = comp_cfg.get("NN", {})
        method_override = {
            "class": "NeuralNetwork",
            "params": {
                "arch": args.arch_nn if getattr(args, 'arch_nn', None) else nn_cfg.get("arch", "periodic_siren_deep_128"),
                "lr": float(nn_cfg.get("lr", 1e-3)),
                "max_iters": int(nn_cfg.get("max_iters", 2000)),
                "batch_size": int(nn_cfg.get("batch_size", 2000)),
                "lbfgs_iters": int(nn_cfg.get("lbfgs_iters", 50)),
            }
        }

    n_workers = effective_n_workers(args)

    if args.overwrite:
        print("\n--- Skipping reference simulation (--overwrite): reusing existing baseline ---")
    else:
        run_baseline(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            n_workers=n_workers,
        )

    if args.online:
        run_online_compressed_branch(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            compression_period=compression_period,
            n_workers=n_workers,
        )
    else:
        run_offline_compressed_branch(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
            compression_period=compression_period,
            mesh_kwargs=mesh_kwargs,
            method_override=method_override,
            n_workers=args.dask_workers,
        )
    if not args.keep_pdi_copy:
        remove_file_if_exists(run_pdi_yaml, "copied PDI config")

    print(f"\nWorkflow complete. All files are in: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    configure_toolchain(args)

    if on_adastra():
        submit_batch_and_wait(args)
        return

    run_dir = run_pipeline(args)
    compare_results(run_dir)


if __name__ == "__main__":
    main()
