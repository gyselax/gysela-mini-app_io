import argparse
import csv
import os

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


def plot_diags(diags_filenames, output=None):
    """Plot diagnostic quantities vs time.

    diags_filenames: a single CSV path or a list of CSV paths.
    If multiple files are given they are overlaid on each other.
    Saves to output path if given, otherwise shows interactively.
    """
    if isinstance(diags_filenames, (str, os.PathLike)):
        diags_filenames = [diags_filenames]

    datasets = []
    for path in diags_filenames:
        data = load_diags(path)
        label = os.path.basename(os.path.dirname(os.path.abspath(path))) or os.path.basename(path)
        datasets.append((label, data))

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


def parse_args():
    parser = argparse.ArgumentParser(description="Plot diagnostics CSV files.")
    parser.add_argument("csv_files", nargs="+", help="One or more diagnostics.csv files to plot.")
    parser.add_argument("-o", "--output", default=None, help="Output file (PDF/PNG). Omit to show interactively.")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_diags(args.csv_files, output=args.output)


if __name__ == "__main__":
    main()
