#!/usr/bin/env python3
"""CLI post-processing for offline compression benchmarks gysela-mini-app_io.
Plasmax-diags mirror. Entry point: gysela-diags."""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PHYSICAL_QUANTITIES = ["epot", "ekin", "etot", "mass", "momentum", "l2norm"]

def load_diagnostics_csv(csv_path):
    times, data = [], {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            times.append(float(row["time"]))
            for k, v in row.items():
                if k not in ("iter", "time"):
                    try:
                        data.setdefault(k, []).append(float(v))
                    except ValueError:
                        pass
                    
    return times, data 

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
    return p.parent.name.replace("branch_", "")

def save_fig(fig, out_path):
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path.with_suffix('.png')}")
    
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
        ax.set_xlabel("Time"); ax.set_ylabel(q); ax.set_title(f"{q} over time")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
        fig.tight_layout(); save_fig(fig, out_dir / f"{q}_comparison"); plt.close(fig)


def plot_frobenius(compression_cases, out_dir, filt=None, name="frob_error_comparison"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for label, iters, data in compression_cases:
        if filt and filt.lower() not in label.lower():
            continue
        if "relative_l2_error" not in data:
            continue
        ax.semilogy(iters, data["relative_l2_error"], marker="o", markersize=4, label=label)
        plotted = True
    if not plotted:
        print(f"Nothing to plot for {name}."); plt.close(fig); return
    ax.set_xlabel("iter"); ax.set_ylabel("Relative L2 (Frobenius) error")
    ax.set_title("Compression error vs iteration")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout(); save_fig(fig, out_dir / name); plt.close(fig)

def plot_cpu_time(compression_cases, out_dir):
    labels, comp_t, decomp_t, sim_t = [], [], [], []
    for label, iters, data in compression_cases:
        labels.append(label)
        comp_t.append(float(np.sum(data.get("compression_seconds", [0.0]))))
        decomp_t.append(float(np.sum(data.get("decompression_seconds", [0.0]))))
        sim_vals = [v for v in data.get("sim_time_approx", []) if v]
        sim_t.append(float(np.sum(sim_vals)) if sim_vals else 0.0)
    if not labels:
        print("No compression_events_offline.csv found for --cpu-time."); return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, sim_t, label="Simulation (approx.)", color="#1f77b4")
    ax.bar(x, comp_t, bottom=sim_t, label="Compression", color="#ff7f0e")
    ax.bar(x, decomp_t, bottom=np.array(sim_t) + np.array(comp_t), label="Decompression", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative CPU time (s)"); ax.set_title("Computational cost")
    ax.legend(); ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout(); save_fig(fig, out_dir / "cpu_time_comparison"); plt.close(fig)

def plot_svd_spectrum(data_dirs, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    found = False
    for data_dir in data_dirs:
        spectrum_dir = Path(data_dir) / "svd_spectrums"
        if not spectrum_dir.exists():
            continue
        files = sorted(spectrum_dir.glob("spectrum_iter*.csv"))
        cmap = plt.cm.viridis
        for idx, fpath in enumerate(files):
            found = True
            arr = np.genfromtxt(fpath, delimiter=",", skip_header=1)
            it = int(fpath.stem.replace("spectrum_iter", ""))
            ax.semilogy(arr[:, 0], arr[:, 1], color=cmap(idx / max(len(files) - 1, 1)),
                        marker="o", markersize=3, label=f"iter {it}")
    if not found:
        print("No svd_spectrums/ found for --svd-spectrum."); plt.close(fig); return
    ax.set_xlabel("Singular value index i"); ax.set_ylabel(r"$\sigma_i/\sigma_1$")
    ax.set_title("SVD spectrum decay"); ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout(); save_fig(fig, out_dir / "svd_spectrum"); plt.close(fig)
    
def plot_inr_loss(data_dirs, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    found = False
    for data_dir in data_dirs:
        loss_dir = Path(data_dir) / "loss_histories"
        if not loss_dir.exists():
            continue
        for fpath in sorted(loss_dir.glob("loss_iter*.npy")):
            found = True
            ax.semilogy(np.load(fpath), label=fpath.stem)
    if not found:
        print("No loss_histories/ found for --inr-loss."); plt.close(fig); return
    ax.set_xlabel("Optimization step (ADAM then L-BFGS)"); ax.set_ylabel("MSE loss")
    ax.set_title("INR training curve"); ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout(); save_fig(fig, out_dir / "inr_loss_convergence"); plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Post-process gysela-mini-app_io compression benchmarks.")
    parser.add_argument("--params", type=str, nargs="+", required=True,
                         help="Path(s) to diagnostics.csv (one per branch/run to compare).")
    for q in PHYSICAL_QUANTITIES:
        parser.add_argument(f"--{q}", action="store_true")
    parser.add_argument("--frob", action="store_true")
    parser.add_argument("--frob-pod", action="store_true")
    parser.add_argument("--frob-inr", action="store_true")
    parser.add_argument("--svd-spectrum", action="store_true")
    parser.add_argument("--inr-loss", action="store_true")
    parser.add_argument("--cpu-time", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    
    quantities = [q for q in PHYSICAL_QUANTITIES if getattr(args, q.replace("-", "_"))]
    
    physical_cases, compression_cases, data_dirs = [], [], []
    for p in args.params:
        p = Path(p)
        if not p.exists():
            print(f"Warning: {p} not found, skipping."); continue
        times, data = load_diagnostics_csv(p)
        label = case_label(p)
        physical_cases.append((label, times, data))
        data_dir = p.parent
        data_dirs.append(data_dir)
        ce = load_compression_events(data_dir)
        if ce is not None:
            compression_cases.append((label, ce[0], ce[1]))

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params[0]).resolve().parent.parent / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if quantities:
        plot_physical(physical_cases, quantities, out_dir)
    if args.frob:
        plot_frobenius(compression_cases, out_dir)
    if args.frob_pod:
        plot_frobenius(compression_cases, out_dir, filt="POD", name="frob_error_pod")
    if args.frob_inr:
        plot_frobenius(compression_cases, out_dir, filt="NN", name="frob_error_nn")
    if args.cpu_time:
        plot_cpu_time(compression_cases, out_dir)
    if args.svd_spectrum:
        plot_svd_spectrum(data_dirs, out_dir)
    if args.inr_loss:
        plot_inr_loss(data_dirs, out_dir)

    print(f"\nDone. Figures in {out_dir}")
    
if __name__ == "__main__":
    main()