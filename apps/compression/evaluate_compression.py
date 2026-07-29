import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_QUANTITIES = [
    ("ekin", r"$\mathcal{E}_{kin}$"),
    ("epot", r"$\mathcal{E}_{pot}$"),
]

CONSERVED_QUANTITIES = [
    ("etot",     r"$|\Delta\mathcal{E}_{tot}|/\mathcal{E}_{tot,0}$"),
    ("l2norm",   r"$|\Delta L_2|/L_{2,0}$"),
    ("mass",     r"$|\Delta M|/M_0$"),
    ("momentum", r"$|\Delta P|/P_0$"),
]

PHYSICAL_QUANTITIES = ["epot","ekin","etot","mass","momentum","l2norm"]

def load_diags(filename):
    """Read a diagnostics CSV and return a dict of numpy arrays keyed by column name."""
    rows = {}
    seen = set()

    with open(filename, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            it = int(row["iter"])
            if it in seen:
                continue
            seen.add(it)
            for key, val in row.items():
                rows.setdefault(key, []).append(float(val))

    order = np.argsort(rows["iter"])
    return {k: np.array(v)[order] for k, v in rows.items()}

def load_compression_events(data_dir):
    path = Path(data_dir) / "compression_events_offline.csv"
    if not path.exists():
        return None
    iters, data = [], {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            for k, v in row.items():
                if k == "iter":
                    continue
                try:
                    data.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass
    return iters, data 

def case_label(csv_path):
    p = Path(csv_path).resolve()
    return p.parent.name.replace("branch_","")

def save_fig(fig, out_path):
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.png')}")

def plot_diags(diags_filenames, output=None):
    """Plot diagnostic quantities vs time.

    diags_filenames: a single CSV path or a list of CSV paths.
    If multiple files are given they are overlaid on each other.
    Saves to output path if given, otherwise shows interactively.
    """
    if isinstance(diags_filenames, (str, os.PathLike)):
        diags_filenames = [diags_filenames]

    datasets = [(case_label(p), load_diags(p)) for p in diags_filenames]

    n = len(RAW_QUANTITIES) + len(CONSERVED_QUANTITIES)
    fig, axs = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)

    for ax, (col, ylabel) in zip(axs, RAW_QUANTITIES):
        for label, data in datasets:
            if col not in data:
                continue
            ax.semilogy(data["time"], np.abs(data[col]), label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        if len(datasets) > 1:
            ax.legend(fontsize=8)

    for ax, (col, ylabel) in zip(axs[len(RAW_QUANTITIES):], CONSERVED_QUANTITIES):
        for label, data in datasets:
            if col not in data:
                continue
            q0 = data[col][0]
            rel = np.abs(data[col] - q0) / np.abs(q0)
            ax.semilogy(data["time"], rel, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        if len(datasets) > 1:
            ax.legend(fontsize=8)

    axs[-1].set_xlabel("Time")
    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"Plot written to: {output}")
    else:
        plt.show()

    plt.close(fig)
    
#specific figures for compression

def plot_physical(cases, quantities, out_dir):
    for q in quantities:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for label, times, data in cases:
            if q not in data:
                continue
            style = dict(linewidth=2.5)
            if "baseline" in label.lower():
                style.update(color="black", linestyle="--", linewidth=3.0, zorder=5)
            ax.plot(times, data[q], label=label, **style)
        ax.set_xlabel("Time")
        ax.set_ylabel(q)
        ax.set_title(f"{q} over time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        save_fig(fig, out_dir / f"{q}_comparison")
        plt.close(fig)
        
def plot_frobenius(compression_cases, out_dir, filt=None, name="frob_error_comparison"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for label, iters, data in compression_cases:
        if filt and filt.lower() not in label.lower():
            continue
        if not data.get("relative_l2_error"):
            continue
        ax.semilogy(iters, data["relative_l2_error"], marker="o", markersize=4, label=label)
        plotted = True
    if not plotted:
        print(f"Nothing to plot for {name}.")
        plt.close(fig)
        return
    ax.set_xlabel("iter")
    ax.set_ylabel("Relative L2 (Frobenius) error")
    ax.set_title("Compression error vs iteration")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_dir / name)
    plt.close(fig)
    
def plot_cpu_time(compression_cases, out_dir):
    labels, comp_t, decomp_t, sim_t = [], [], [], []
    for label, iters, data in compression_cases:
        labels.append(label)
        comp_t.append(float(np.sum(data.get("compression_seconds", [0.0]))))
        decomp_t.append(float(np.sum(data.get("decompression_seconds", [0.0]))))
        sim_vals = [v for v in data.get("sim_time_approx", []) if v]
        sim_t.append(float(np.sum(sim_vals)) if sim_vals else 0.0)
        
    if not labels:
        print("No compression_events_offline.csv found for --cpu-time.")
        return 

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, sim_t, label="Simulation (approx.)", color="#1f77b4")
    ax.bar(x, comp_t, bottom=sim_t, label="Compression", color="#ff7f0e")
    ax.bar(x, decomp_t, bottom=np.array(sim_t) + np.array(comp_t), label="Decompression", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative CPU time (s)")
    ax.set_title("Computational cost")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_dir / "cpu_time_comparison")
    plt.close(fig)
    
def plot_svd_spectrum(data_dirs, compression_cases, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    found = False
    rank_by_dir = {
        Path(dd): int(data["param_n_components"][0])
        for (label, _, data), dd in zip(compression_cases, data_dirs)
        if "param_n_components" in data and data["param_n_components"]
    }
    
    color_cycle = plt.cm.tab10.colors
    
    for i, data_dir in enumerate(data_dirs):
        spectrum_dir = Path(data_dir) / "svd_spectrums"
        if not spectrum_dir.exists():
            continue
        files = sorted(spectrum_dir.glob("spectrum_iter*.csv"))
        if not files:
            continue
        found = True
        # last checkpoint spectrum
        arr = np.genfromtxt(files[-1], delimiter=",", skip_header=1)
        label = case_label(str(data_dir / "diagnostics.csv"))
        color = color_cycle[i % len(color_cycle)]
        ax.semilogy(arr[:, 0], arr[:, 1], color=color, marker="o", markersize=3, label=label)
        if data_dir in rank_by_dir:
            ax.axvline(rank_by_dir[data_dir], color=color, linestyle="--", alpha=0.7)
                
    if not found:
        print("No svd_spectrum/ found for --svd-spectrum.")
        plt.close(fig)
        return
    
    ax.set_xlabel("Singular value index i")
    ax.set_ylabel(r"$\sigma_i/\sigma_1$")
    ax.set_title("SVD spectrum decay (last checkpoint)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_dir / "svd_spectrum")
    plt.close(fig)
    
def plot_inr_loss(data_dirs, out_dir):
    found = False
    for data_dir in data_dirs:
        loss_dir = Path(data_dir) / "loss_histories"
        if not loss_dir.exists():
            continue
        for fpath in sorted(loss_dir.glob("loss_iter*.npy")):
            found = True
            hist = np.load(fpath)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.semilogy(hist, color="tab:blue", linewidth=2)
            ax.set_xlabel("Optimization step (ADAM then L-BFGS)")
            ax.set_ylabel("MSE loss")
            ax.set_title(fpath.stem)
            ax.grid(True, which="both", alpha=0.4)
            fig.tight_layout()
            save_fig(fig, out_dir / fpath.stem)
            plt.close(fig)
    if not found:
        print("No loss_histories/ found for --inr-loss.")
    
def resolve_out_dir(args):
    if args.output_dir:
        return Path(args.outpu_dir)
    
    base = Path(args.params[0]).resolve().parent.parent / "comparisons"
    if len(args.params) <= 1:
        return base
    
    all_paths_str = " ".join(args.params)
    has_pod = "POD" in all_paths_str
    has_nn = "NN" in all_paths_str
    if has_pod and not has_nn:
        sub = "baseline_vs_POD"
    elif has_nn and not has_pod:
        sub = "baseline_vs_INR"
    else:
        sub = "mixed_comparisons"
    
    return base / sub

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process gysela-mini-app_io compression benchmarks.")
    parser.add_argument("--params", type=str, nargs="+", required=True, help="Path(s) to diagnostics.csv (one per branch/run to compare).")
    parser.add_argument("--summary", action="store_true", help="Multi-panel summary figure (raw + relative retained), auto-generated style.")
    for q in PHYSICAL_QUANTITIES:
        parser.add_argument(f"--{q}", action="store_true")
    parser.add_argument("--frob", action="store_true")
    parser.add_argument("--frob-pod", action="store_true")
    parser.add_argument("--frob-inr", action="store_true")
    parser.add_argument("--svd-spectrum", action="store_true")
    parser.add_argument("--inr-loss", action="store_true")
    parser.add_argument("--cpu-time", action="store_true")
    parser.add_argument("--compression-ratio", action="store_true")
    parser.add_argument("-o","--output-dir", type=str, default=None, help="Output directory for figures (also used by --summary)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    quantities = [q for q in PHYSICAL_QUANTITIES if getattr(args, q.replace("-","_"))]
    
    physical_cases, compression_cases, data_dirs = [], [], []
    for p in args.params:
        p = Path(p)
        if not p.exists():
            print(f"Warning: {p} not found, skipping.")
            continue
        
        data = load_diags(p)
        times = data["time"]
        label = case_label(p)
        physical_cases.append((label, times, data))
        data_dir = p.parent 
        data_dirs.append(data_dir)
        ce = load_compression_events(data_dir)
        if ce is not None:
            compression_cases.append((label, ce[0], ce[1]))
            
    out_dir = resolve_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.summary:
        plot_diags(args.params, output=out_dir / "diags_comparison.png")
    if quantities:
        plot_physical(physical_cases, quantities, out_dir)
    if args.frob:
        plot_frobenius(compression_cases, out_dir)
    if args.frob_pod:
        plot_frobenius(compression_cases, out_dir, filt="pod", name="frob_error_pod")
    if args.frob_inr:
        plot_frobenius(compression_cases, out_dir, filt="inr", name="frob_error_nn")
    if args.cpu_time:
        plot_cpu_time(compression_cases, out_dir)
    if args.svd_spectrum:
        plot_svd_spectrum(data_dirs, compression_cases, out_dir)
    if args.inr_loss:
        plot_inr_loss(data_dirs, out_dir)
        
    print(f"\nDone. Figures in {out_dir}")

if __name__ == "__main__":
    main()
