#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
from datetime import datetime

import yaml

# ------------------------------------------------------------------
# Compression params / names
# ------------------------------------------------------------------
from compression_methods.PCA import PCACompressor

COMPRESSOR_CLASS = PCACompressor
COMPRESSOR_PARAMS = {
    "n_components": 8,
    "normalisation": "none",
    "clip_nonnegative": False,
}

EXEC_CMD = ["mpirun", "-n", "4", "./build/apps/compression/compression_app"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SOURCE_GYSELA_YAML = os.path.join(SCRIPT_DIR, "params.yaml")
SOURCE_PDI_YAML = os.path.join(SCRIPT_DIR, "pdi_out.yaml")


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


def read_benchmark_config(config):
    try:
        iter_total = int(config["Algorithm"]["nbiter"])
        compression_period = int(config["CompressionBenchmark"]["compression_period"])
    except KeyError as exc:
        raise RuntimeError(
            "Missing required benchmark parameter in params.yaml. "
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
    dt = float(config["Algorithm"]["deltat"])
    time_diag = float(config["Output"]["time_diag"])
    nbstep_diag = int(time_diag / dt)

    if nbstep_diag <= 0:
        raise RuntimeError(f"Invalid diagnostic step: time_diag={time_diag}, deltat={dt}.")

    if abs(nbstep_diag * dt - time_diag) > 1e-12:
        raise RuntimeError(
            f"Output.time_diag must be an integer multiple of Algorithm.deltat. "
            f"Got time_diag={time_diag}, deltat={dt}, time_diag/deltat={time_diag / dt}."
        )

    return nbstep_diag


def assert_iterations_are_diagnostic_outputs(
    iter_total,
    compression_period,
    nbstep_diag,
):
    if compression_period % nbstep_diag != 0:
        raise RuntimeError(
            f"Compression period must be a multiple of nbstep_diag={nbstep_diag}. "
            f"Got compression_period={compression_period}."
        )

    if iter_total % nbstep_diag != 0:
        raise RuntimeError(
            f"Algorithm.nbiter must be a multiple of nbstep_diag={nbstep_diag}. " f"Got nbiter={iter_total}."
        )


def assert_complete_branch(branch_dir, file_index_total):
    missing = []

    for idx in range(file_index_total + 1):
        filepath = os.path.join(branch_dir, f"GYSELALIBXX_{idx:05d}.h5")

        if not os.path.exists(filepath):
            missing.append(idx)

    if missing:
        preview = ", ".join(f"{idx:05d}" for idx in missing[:10])
        suffix = "..." if len(missing) > 10 else ""

        raise RuntimeError(
            f"Branch {os.path.basename(branch_dir)} is incomplete. "
            f"Missing {len(missing)} diagnostic file(s): {preview}{suffix}"
        )


def build_compressor():
    return COMPRESSOR_CLASS(**COMPRESSOR_PARAMS)


def format_param_summary(metrics):
    params = metrics.get("params") or {}

    if not params:
        return "no parameters"

    return ", ".join(f"{key}={value}" for key, value in params.items())


def compress_decompress(input_h5, output_h5, compressed_path):
    compressor = build_compressor()

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
):
    with open(base_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    config.setdefault("Input", {})
    config["Input"]["nb_restart"] = nb_restart
    config["Input"]["fdistribu_filename"] = fdist_file
    config["Input"]["iter_offset"] = iter_offset

    config.setdefault("Algorithm", {})
    config["Algorithm"]["nbiter"] = nbiter

    with open(output_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def remove_restart_output_before_rewrite(
    branch_dir,
    current_iter,
    nbstep_diag,
):
    if current_iter % nbstep_diag != 0:
        raise RuntimeError(f"Restart iteration {current_iter} is not aligned with " f"nbstep_diag={nbstep_diag}.")

    file_index = current_iter // nbstep_diag

    filepath = os.path.join(
        branch_dir,
        f"GYSELALIBXX_{file_index:05d}.h5",
    )

    if os.path.exists(filepath):
        print(f"  [Restart Output Refresh] Removing existing " f"{os.path.basename(filepath)} before restart rewrite")
        os.remove(filepath)


def run_sim(branch_name, gysela_yaml, pdi_yaml, work_dir):
    print(f"\n--- Running: {branch_name} ---")

    os.makedirs(work_dir, exist_ok=True)

    exec_path = os.path.abspath(EXEC_CMD[-1])
    abs_gysela = os.path.abspath(gysela_yaml)
    abs_pdi = os.path.abspath(pdi_yaml)

    cmd = EXEC_CMD[:-1] + [exec_path, abs_gysela, abs_pdi]

    subprocess.run(
        cmd,
        cwd=work_dir,
        env=os.environ.copy(),
        check=True,
    )


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


def run_baseline(run_dir, run_pdi_yaml, iter_total):
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

    run_sim(
        branch_name="Baseline",
        gysela_yaml=yaml_baseline,
        pdi_yaml=run_pdi_yaml,
        work_dir=dir_baseline,
    )

    return dir_baseline


def run_periodic_compressed_branch(
    run_dir,
    run_pdi_yaml,
    iter_total,
    compression_period,
    nbstep_diag,
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

        if nb_restart > 0:
            remove_restart_output_before_rewrite(
                branch_dir=dir_compressed,
                current_iter=current_iter,
                nbstep_diag=nbstep_diag,
            )

        run_sim(
            branch_name=(f"Compressed segment {segment_id} " f"({current_iter} -> {next_iter})"),
            gysela_yaml=yaml_segment,
            pdi_yaml=run_pdi_yaml,
            work_dir=dir_compressed,
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

        if current_iter % nbstep_diag != 0:
            raise RuntimeError(
                f"Cannot compress at iteration {current_iter}: " f"not a multiple of nbstep_diag={nbstep_diag}."
            )

        file_index = current_iter // nbstep_diag

        raw_restart = os.path.join(
            dir_compressed,
            f"GYSELALIBXX_{file_index:05d}.h5",
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

        metrics = compress_decompress(
            input_h5=raw_restart,
            output_h5=approx_restart,
            compressed_path=compressed_payload,
        )

        approx_restart_size = os.path.getsize(approx_restart) if os.path.exists(approx_restart) else None

        compressed_payload_size = os.path.getsize(compressed_payload) if os.path.exists(compressed_payload) else None

        compression_events.append(
            {
                "segment_id": segment_id,
                "iteration": current_iter,
                "file_index": file_index,
                "branch_restart": os.path.relpath(raw_restart, run_dir),
                "approx_restart": os.path.relpath(approx_restart, run_dir),
                "compressed_payload": (os.path.relpath(compressed_payload, run_dir) if keep_payloads else None),
                "method_name": metrics.get("method_name"),
                "param_names": metrics.get("param_names", []),
                "params": metrics.get("params", {}),
                # Backward-compatible convenience key for PCA runs.
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

        if not keep_payloads:
            remove_file_if_exists(
                compressed_payload,
                "compressed PCA payload",
            )

        restart_file = os.path.abspath(approx_restart)
        segment_id += 1

    write_compression_manifest(run_dir, compression_events)

    return dir_compressed


def main():
    args = parse_args()

    assert_file_exists(SOURCE_GYSELA_YAML, "base GYSELA input template")
    assert_file_exists(SOURCE_PDI_YAML, "base PDI input template")

    run_dir = create_or_select_run_dir(
        requested_run_dir=args.run_dir,
        overwrite=args.overwrite,
    )

    print(f"Master directory initialised: {run_dir}")

    run_pdi_yaml = os.path.join(run_dir, "pdi_out.yaml")
    shutil.copy2(SOURCE_PDI_YAML, run_pdi_yaml)

    with open(SOURCE_GYSELA_YAML, "r") as f:
        base_cfg = yaml.safe_load(f)

    iter_total, compression_period = read_benchmark_config(base_cfg)
    nbstep_diag = compute_diagnostic_step(base_cfg)

    assert_iterations_are_diagnostic_outputs(
        iter_total=iter_total,
        compression_period=compression_period,
        nbstep_diag=nbstep_diag,
    )

    file_index_total = iter_total // nbstep_diag

    print(f"Total iterations       : {iter_total}")
    print(f"Compression period K   : {compression_period}")
    print(f"Diagnostic step        : {nbstep_diag}")
    print(f"Final diagnostic index : {file_index_total}")

    if args.overwrite:
        dir_baseline = os.path.join(run_dir, "branch_baseline")
        print("\n--- Skipping reference simulation (--overwrite): reusing existing reference data without compression")
    else:
        dir_baseline = run_baseline(
            run_dir=run_dir,
            run_pdi_yaml=run_pdi_yaml,
            iter_total=iter_total,
        )

    assert_complete_branch(dir_baseline, file_index_total)

    dir_compressed = run_periodic_compressed_branch(
        run_dir=run_dir,
        run_pdi_yaml=run_pdi_yaml,
        iter_total=iter_total,
        compression_period=compression_period,
        nbstep_diag=nbstep_diag,
        keep_payloads=args.keep_payloads,
        keep_restart_approximations=args.keep_restart_approximations,
        keep_segment_configs=args.keep_segment_configs,
    )

    assert_complete_branch(dir_compressed, file_index_total)

    if not args.keep_pdi_copy:
        remove_file_if_exists(run_pdi_yaml, "copied PDI config")

    print(f"\nWorkflow complete. All files are populated inside: {run_dir}")


if __name__ == "__main__":
    main()
