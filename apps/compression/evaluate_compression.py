import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate compression benchmark results.")

    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help=(
            "Path to the compression_run_* directory to analyse. "
            "If omitted, the latest compression_run_* directory is used."
        ),
    )

    return parser.parse_args()


def find_run_dir(requested_run_dir=None):
    if requested_run_dir is not None:
        run_dir = os.path.abspath(requested_run_dir)

        if not os.path.isdir(run_dir):
            raise RuntimeError(f"Run directory does not exist: {run_dir}")

        return run_dir

    run_dirs = sorted(glob.glob("compression_run_*"))

    if not run_dirs:
        raise RuntimeError("No compression_run_* folders found.")

    return os.path.abspath(run_dirs[-1])


def load_branch_diagnostics(branch_dir):
    """Read diagnostics.csv produced by diagnostics.py and return arrays.

    The returned dict has keys matching the plot_analysis interface:
    idx, time, M, Px, Py, P, E_kin, E_pot, E_tot.

    Duplicate rows at restart boundaries are removed (keeping first occurrence).
    """
    csv_path = os.path.join(branch_dir, "diagnostics.csv")

    if not os.path.exists(csv_path):
        raise RuntimeError(f"Diagnostics CSV not found: {csv_path}")

    rows = {
        "idx": [], "time": [],
        "M": [], "Px": [], "Py": [], "P": [],
        "E_kin": [], "E_pot": [], "E_tot": [],
    }
    seen_iters = set()

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            it = int(row["iter"])
            if it in seen_iters:
                continue
            seen_iters.add(it)
            rows["idx"].append(it)
            rows["time"].append(float(row["time"]))
            rows["M"].append(float(row["mass"]))
            rows["Px"].append(float(row["momentum_x"]))
            rows["Py"].append(float(row["momentum_y"]))
            rows["P"].append(float(row["momentum"]))
            rows["E_kin"].append(float(row["ekin"]))
            rows["E_pot"].append(float(row["epot"]))
            rows["E_tot"].append(float(row["etot"]))

    order = np.argsort(rows["idx"])
    return {k: np.array(v)[order] for k, v in rows.items()}


def load_benchmark_config(run_dir):
    config_file = os.path.join(run_dir, "config_baseline.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    try:
        iter_total = int(config["Algorithm"]["nbiter"])
        compression_period = int(config["CompressionBenchmark"]["compression_period"])
    except KeyError as exc:
        raise RuntimeError(
            "Missing required benchmark parameter in config_baseline.yaml. "
            "Expected Algorithm.nbiter and CompressionBenchmark.compression_period."
        ) from exc

    if iter_total <= 0:
        raise RuntimeError(f"Invalid nbiter={iter_total}.")

    if compression_period <= 0:
        raise RuntimeError(f"Invalid compression_period={compression_period}.")

    return iter_total, compression_period


def load_compression_events(run_dir):
    manifest = os.path.join(run_dir, "compression_events.yaml")

    if not os.path.exists(manifest):
        return []

    with open(manifest, "r") as f:
        data = yaml.safe_load(f) or {}

    return data.get("compression_events") or []


def assert_same_timeline(branches):
    reference_indices = branches["Baseline"]["idx"]
    reference_time = branches["Baseline"]["time"]

    for name, data in branches.items():
        if len(data["idx"]) != len(reference_indices):
            raise RuntimeError(f"{name} has {len(data['idx'])} outputs, " f"but baseline has {len(reference_indices)}.")

        if not np.array_equal(data["idx"], reference_indices):
            raise RuntimeError(f"{name} does not share the same diagnostic indices as baseline.")

        if not np.allclose(data["time"], reference_time):
            raise RuntimeError(f"{name} does not share the same time grid as baseline.")


def bytes_to_human(size_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]

    size = float(size_bytes)
    sign = "-" if size < 0 else ""
    size = abs(size)

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{sign}{size:.2f} {unit}"

        size /= 1024.0

    raise RuntimeError("Unreachable size conversion state.")


def event_value(event, key):
    value = event.get(key)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_event_with_max_metric(compression_events, metric_name):
    valid_events = [event for event in compression_events if event_value(event, metric_name) is not None]

    if not valid_events:
        return None

    return max(valid_events, key=lambda event: event_value(event, metric_name))


def compute_compression_stats(run_dir, compression_events):
    stats = []

    if not compression_events:
        return stats

    last_event = compression_events[-1]

    worst_l2_event = find_event_with_max_metric(compression_events, "relative_l2_error")
    worst_max_abs_event = find_event_with_max_metric(compression_events, "max_abs_error")
    worst_mean_abs_event = find_event_with_max_metric(compression_events, "mean_abs_error")
    worst_rmse_event = find_event_with_max_metric(compression_events, "rmse")
    worst_ratio_event = find_event_with_max_metric(compression_events, "compression_ratio")

    raw_restart_size = last_event.get("raw_restart_size")
    approx_restart_size = last_event.get("approx_restart_size")
    compressed_payload_size = last_event.get("compressed_payload_size")

    compression_ratio = last_event.get("compression_ratio")

    if (
        compression_ratio is None
        and raw_restart_size is not None
        and compressed_payload_size is not None
        and compressed_payload_size > 0
    ):
        compression_ratio = raw_restart_size / compressed_payload_size

    compressed_change = None
    compressed_change_percent = None

    if raw_restart_size is not None and compressed_payload_size is not None:
        compressed_change = compressed_payload_size - raw_restart_size
        compressed_change_percent = 100.0 * compressed_change / raw_restart_size

    approx_change = None
    approx_change_percent = None

    if raw_restart_size is not None and approx_restart_size is not None:
        approx_change = approx_restart_size - raw_restart_size
        approx_change_percent = 100.0 * approx_change / raw_restart_size

    result = {
        "label": (f"{len(compression_events)} compression events, " f"last iter {int(last_event['iteration'])}"),
        "method_name": last_event.get("method_name", "unknown"),
        "param_names": last_event.get("param_names", []),
        "params": last_event.get("params", {}),
        "raw_restart_size": raw_restart_size,
        "approx_restart_size": approx_restart_size,
        "compressed_payload_size": compressed_payload_size,
        "compression_ratio": compression_ratio,
        "compressed_change": compressed_change,
        "compressed_change_percent": compressed_change_percent,
        "approx_change": approx_change,
        "approx_change_percent": approx_change_percent,
        "relative_l2_error": None if worst_l2_event is None else event_value(worst_l2_event, "relative_l2_error"),
        "max_abs_error": None if worst_max_abs_event is None else event_value(worst_max_abs_event, "max_abs_error"),
        "mean_abs_error": None if worst_mean_abs_event is None else event_value(worst_mean_abs_event, "mean_abs_error"),
        "rmse": None if worst_rmse_event is None else event_value(worst_rmse_event, "rmse"),
        "worst_relative_l2_iteration": (None if worst_l2_event is None else int(worst_l2_event["iteration"])),
        "worst_relative_l2_file_index": (None if worst_l2_event is None else int(worst_l2_event["file_index"])),
        "worst_max_abs_iteration": (None if worst_max_abs_event is None else int(worst_max_abs_event["iteration"])),
        "worst_max_abs_file_index": (None if worst_max_abs_event is None else int(worst_max_abs_event["file_index"])),
        "worst_mean_abs_iteration": (None if worst_mean_abs_event is None else int(worst_mean_abs_event["iteration"])),
        "worst_mean_abs_file_index": (None if worst_mean_abs_event is None else int(worst_mean_abs_event["file_index"])),
        "worst_rmse_iteration": (None if worst_rmse_event is None else int(worst_rmse_event["iteration"])),
        "worst_rmse_file_index": (None if worst_rmse_event is None else int(worst_rmse_event["file_index"])),
        "worst_compression_ratio": (
            None if worst_ratio_event is None else event_value(worst_ratio_event, "compression_ratio")
        ),
        "worst_compression_ratio_iteration": (
            None if worst_ratio_event is None else int(worst_ratio_event["iteration"])
        ),
        "explained_variance_ratio_sum": last_event.get("explained_variance_ratio_sum"),
        "n_components": last_event.get("n_components"),
        "metrics_source": "compression_events.yaml",
        "approx_restart_kept": last_event.get("approx_restart_kept"),
        "compressed_payload_kept": last_event.get("compressed_payload_kept"),
    }

    stats.append(result)

    return stats


def format_compression_stats(stats):
    lines = [
        "Compression statistics",
        "Worst reconstruction errors over all events",
    ]

    if not stats:
        lines.append("")
        lines.append("No compression event found.")
        return "\n".join(lines)

    for item in stats:
        lines.append("")

        method_name = item.get("method_name", "unknown")
        params = item.get("params") or {}
        param_text = ", ".join(f"{key}={value}" for key, value in params.items())
        method_text = method_name if not param_text else f"{method_name}({param_text})"

        lines.append(f"{item['label']}\n{method_text}")

        raw_size = item.get("raw_restart_size")
        compressed_size = item.get("compressed_payload_size")
        approx_size = item.get("approx_restart_size")

        if raw_size is not None and compressed_size is not None:
            lines.append(
                f"  Raw restart:      {bytes_to_human(raw_size):>10}    "
                f"Compressed payload: {bytes_to_human(compressed_size):>10}"
            )
        elif compressed_size is not None:
            lines.append(
                f"  Raw restart:      unavailable    "
                f"Compressed payload: {bytes_to_human(compressed_size):>10}"
            )
        else:
            lines.append("  File sizes: unavailable")

        if approx_size is not None:
            lines.append(f"  Approx restart:   {bytes_to_human(approx_size):>10}")

        compression_ratio = item.get("compression_ratio")
        compressed_change = item.get("compressed_change")
        compressed_change_percent = item.get("compressed_change_percent")

        if compression_ratio is not None and compressed_change is not None and compressed_change_percent is not None:
            lines.append(
                f"Compression ratio: {compression_ratio:.3f}x "
                f"({bytes_to_human(compressed_change)}, "
                f"{compressed_change_percent:+.2f}%)"
            )
        elif compression_ratio is not None:
            lines.append(f"Compression ratio: {compression_ratio:.3f}x")
        else:
            lines.append("Compression ratio: unavailable")

        explained_variance_ratio_sum = item.get("explained_variance_ratio_sum")

        if explained_variance_ratio_sum is not None:
            lines.append("  Explained variance ratio sum: " f"{explained_variance_ratio_sum:.6e}")

        relative_l2_error = item.get("relative_l2_error")
        max_abs_error = item.get("max_abs_error")

        if relative_l2_error is not None:
            worst_iter = item.get("worst_relative_l2_iteration")
            worst_idx = item.get("worst_relative_l2_file_index")

            location = (
                f" at iter {worst_iter}, file {worst_idx:05d}"
                if worst_iter is not None and worst_idx is not None
                else ""
            )

            lines.append(f"  Worst relative L2 error: {relative_l2_error:.3e}{location}")
        else:
            lines.append("  Worst relative L2 error: unavailable")

        if max_abs_error is not None:
            worst_iter = item.get("worst_max_abs_iteration")
            worst_idx = item.get("worst_max_abs_file_index")

            location = (
                f" at iter {worst_iter}, file {worst_idx:05d}"
                if worst_iter is not None and worst_idx is not None
                else ""
            )

            lines.append(f"  Worst max abs error:    {max_abs_error:.3e}{location}")
        else:
            lines.append("  Worst max abs error: unavailable")

        mean_abs_error = item.get("mean_abs_error")
        if mean_abs_error is not None:
            worst_iter = item.get("worst_mean_abs_iteration")
            worst_idx = item.get("worst_mean_abs_file_index")
            location = (
                f" at iter {worst_iter}, file {worst_idx:05d}"
                if worst_iter is not None and worst_idx is not None
                else ""
            )
            lines.append(f"  Worst mean abs error:   {mean_abs_error:.3e}{location}")

        rmse = item.get("rmse")
        if rmse is not None:
            worst_iter = item.get("worst_rmse_iteration")
            worst_idx = item.get("worst_rmse_file_index")
            location = (
                f" at iter {worst_iter}, file {worst_idx:05d}"
                if worst_iter is not None and worst_idx is not None
                else ""
            )
            lines.append(f"  Worst RMSE:             {rmse:.3e}{location}")

    return "\n".join(lines)


def add_compression_markers(axs, compression_times):
    if not compression_times:
        return

    for ax in axs:
        ax.scatter(
            compression_times,
            np.zeros(len(compression_times)),
            transform=ax.get_xaxis_transform(),
            marker="v",
            s=42,
            color="orange",
            edgecolor="gold",
            linewidth=0.8,
            clip_on=False,
            zorder=10,
            label="_nolegend_",
        )


def compute_relative_errors_vs_baseline(branch, baseline):
    eps = 1e-30

    rel_total_energy = np.abs(branch["E_tot"] - baseline["E_tot"]) / np.maximum(np.abs(baseline["E_tot"]), eps)

    rel_potential_energy = np.abs(branch["E_pot"] - baseline["E_pot"]) / np.maximum(np.abs(baseline["E_pot"]), eps)

    rel_mass = np.abs(branch["M"] - baseline["M"]) / np.maximum(np.abs(baseline["M"]), eps)

    delta_px = branch["Px"] - baseline["Px"]
    delta_py = branch["Py"] - baseline["Py"]

    rel_momentum = np.sqrt(delta_px**2 + delta_py**2) / np.maximum(np.abs(baseline["M"]), eps)

    return {
        "time": baseline["time"],
        "E_tot": rel_total_energy,
        "E_pot": rel_potential_energy,
        "M": rel_mass,
        "P": rel_momentum,
    }


def plot_combined_errors_vs_baseline(ax, branches):
    baseline = branches["Baseline"]

    eps_plot = 1e-16

    quantity_styles = {
        "E_tot": {"label": r"$\mathcal{E}_{tot}$", "marker": None},
        "E_pot": {"label": r"$E_{pot}$", "marker": None},
        "M": {"label": r"$M$", "marker": None},
        "P": {"label": r"$P$", "marker": None},
    }

    for branch_name, branch in branches.items():
        if branch_name == "Baseline":
            continue

        errors = compute_relative_errors_vs_baseline(branch=branch, baseline=baseline)

        t = errors["time"]

        for quantity, style in quantity_styles.items():
            ax.semilogy(
                t[1:],
                np.clip(errors[quantity][1:], eps_plot, None),
                marker=style["marker"],
                markevery=max(1, len(t) // 25),
                linewidth=1.3,
                label=f"{branch_name} {style['label']}",
            )

    ax.set_title("Relative Errors vs Baseline")
    ax.set_ylabel("relative error")
    ax.set_xlabel("Time (t)")
    ax.grid(True, which="both")
    ax.legend(loc="upper right", bbox_to_anchor=(1.23, 1.0), fontsize=8)


def plot_analysis(branches, latest_run, compression_stats=None, compression_times=None):
    assert_same_timeline(branches)

    mass_0 = branches["Baseline"]["M"][0]
    energy_0 = branches["Baseline"]["E_tot"][0]
    px_0 = branches["Baseline"]["Px"][0]
    py_0 = branches["Baseline"]["Py"][0]

    colors = {
        "Baseline": "black",
        "Compressed": "tab:blue",
    }

    linestyles = {
        "Baseline": "-",
        "Compressed": "--",
    }

    eps_conservation = 1e-16
    eps_energy = 1e-30

    plt.style.use("seaborn-v0_8-whitegrid")

    if compression_stats:
        fig = plt.figure(figsize=(15, 29))
        gs = fig.add_gridspec(
            7,
            1,
            height_ratios=[1.25, 2.2, 2.2, 2.2, 2.2, 2.2, 2.6],
            hspace=0.58,
        )

        ax_stats = fig.add_subplot(gs[0, 0])
        axs = [fig.add_subplot(gs[i, 0]) for i in range(1, 7)]

        ax_stats.axis("off")

        ax_stats.text(
            0.5,
            0.5,
            format_compression_stats(compression_stats),
            transform=ax_stats.transAxes,
            fontsize=10,
            va="center",
            ha="center",
            family="monospace",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": "white",
                "alpha": 0.95,
                "edgecolor": "gray",
            },
        )
    else:
        fig, axs = plt.subplots(
            6,
            1,
            figsize=(15, 26),
            sharex=False,
        )

    main_axes = axs[:5]
    error_axis = axs[5]

    for name, data in branches.items():
        if len(data["time"]) == 0:
            continue

        t = data["time"]
        color = colors[name]
        linestyle = linestyles[name]

        total_energy_variation = np.abs(data["E_tot"] - energy_0) / abs(energy_0)

        delta_px = data["Px"] - px_0
        delta_py = data["Py"] - py_0

        momentum_variation = np.sqrt(delta_px**2 + delta_py**2) / abs(mass_0)

        mass_variation = np.abs(data["M"] - mass_0) / abs(mass_0)

        main_axes[0].plot(t[1:], data["E_tot"][1:], color=color, linestyle=linestyle, label=name)

        main_axes[1].semilogy(
            t[1:],
            np.clip(data["E_pot"][1:], eps_energy, None),
            color=color,
            linestyle=linestyle,
            label=name,
        )

        main_axes[2].semilogy(
            t[1:],
            np.clip(total_energy_variation[1:], eps_conservation, None),
            color=color,
            linestyle=linestyle,
            label=name,
        )

        main_axes[3].semilogy(
            t[1:],
            np.clip(momentum_variation[1:], eps_conservation, None),
            color=color,
            linestyle=linestyle,
            label=name,
        )

        main_axes[4].semilogy(
            t[1:],
            np.clip(mass_variation[1:], eps_conservation, None),
            color=color,
            linestyle=linestyle,
            label=name,
        )

    plot_combined_errors_vs_baseline(error_axis, branches)

    all_plot_axes = list(main_axes) + [error_axis]
    add_compression_markers(all_plot_axes, compression_times)

    main_axes[0].set_ylabel(r"$\mathcal{E}_{tot}$")
    main_axes[0].set_title(r"Total Energy: $\mathcal{E}_{tot} = \mathcal{E}_{kin} + \mathcal{E}_{pot}$")

    main_axes[1].set_ylabel(r"$\mathcal{E}_{pot}$")
    main_axes[1].set_title(r"Potential Energy: $\mathcal{E}_{pot} = \frac{1}{2}\int |E|^2\,dx\,dy$")

    main_axes[2].set_ylabel(r"$|\Delta \mathcal{E}| / \mathcal{E}_0$")
    main_axes[2].set_title("Total Energy Variation")

    main_axes[3].set_ylabel(r"$|\Delta \mathcal{P}| / \mathcal{M}_0$")
    main_axes[3].set_title("Normalised Momentum Variation")

    main_axes[4].set_ylabel(r"$|\Delta \mathcal{M}| / \mathcal{M}_0$")
    main_axes[4].set_title("Relative Mass Variation")

    for ax in main_axes:
        ax.set_xlabel("Time (t)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.23, 1.0))

    output_pdf = os.path.join(latest_run, "compression_analysis.pdf")

    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Analysis plot written to: {output_pdf}")


def main():
    args = parse_args()

    try:
        run_dir = find_run_dir(args.run_dir)
    except RuntimeError as exc:
        print(exc)
        return

    print(f"Analysing data from: {run_dir}")

    iter_total, compression_period = load_benchmark_config(run_dir)
    compression_events = load_compression_events(run_dir)
    compression_stats = compute_compression_stats(run_dir, compression_events)

    print(f"Compression events found  = {len(compression_events)}")

    baseline_data = load_branch_diagnostics(os.path.join(run_dir, "branch_baseline"))
    compressed_data = load_branch_diagnostics(os.path.join(run_dir, "branch_compressed"))

    branches = {
        "Baseline": baseline_data,
        "Compressed": compressed_data,
    }

    # Derive dt from the baseline CSV time column for compression event markers.
    baseline_time = baseline_data["time"]
    dt = float(baseline_time[1] - baseline_time[0]) if len(baseline_time) >= 2 else 0.0
    compression_times = [int(event["iteration"]) * dt for event in compression_events]

    plot_analysis(
        branches=branches,
        latest_run=run_dir,
        compression_stats=compression_stats,
        compression_times=compression_times,
    )

    print("Analysis complete.")


if __name__ == "__main__":
    main()
