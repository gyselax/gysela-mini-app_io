# SPDX-License-Identifier: MIT

"""
Plots of the fdistribu initialisation benchmark, from the IOPS results CSV
(results_init_bench.csv):

    bandwidth_vs_nodes.png   bandwidth vs node count, best configuration of each family
    time_split.png           compute / write / fsync time at the largest node count
    configurations_table.png every configuration: time and bandwidth per node count, speed-up,
                             peak memory per node
    configurations_table.csv the same table as a csv

Times are time_total_synced (compute + write + fsync); repetitions are averaged.

Usage:
    python processing/initialisation/plot_summary.py results_init_bench.csv -o figures/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Families, in display order, and associated colors
FAMILIES = {
    "fpp": {"name": "One file per rank", "color": "#2a78d6", "ls": "-", "lw": 2.5},
    "collective": {"name": "Shared file, collective", "color": "#0b0b0b", "ls": "-", "lw": 2.0},
    "independent": {"name": "Shared file, independent", "color": "#52514e", "ls": (0, (6, 4)), "lw": 2.0},
    "posix": {"name": "Shared file, POSIX pwrite", "color": "#9a9993", "ls": (0, (2, 4)), "lw": 2.5},
}
LIBRARIES = {"raw": "raw", "netcdf": "netCDF-4", "hdf5": "HDF5"}
MODES = {"collective": "collective", "independent": "independent", "posix": "POSIX pwrite"}
SPLIT = [("compute", "Compute (Maxwellian)", "#9a9993"), ("write", "Write (open, write, close)", "#2a78d6"), ("sync", "fsync", "#0b0b0b")]
GRID_COLOR, TEXT2 = "#e3e2de", "#52514e"


def load(csv_path, grid=None):
    """Read the IOPS CSV and return one row per test run with short column names."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda c: c.split(".", 1)[1] if c.startswith(("vars.", "metrics.")) else c)
    df = df.dropna(subset=["time_total_synced"])
    grids = sorted(df["grid"].unique())
    if grid is None:
        if len(grids) > 1:
            raise SystemExit(f"several grids in the CSV ({', '.join(grids)}): choose one with --grid")
        grid = grids[0]
    df = df[df["grid"] == grid].copy()
    if df.empty:
        raise SystemExit(f"no successful run for grid '{grid}'")

    df["family"] = df.apply(lambda r: "fpp" if r["layout"] == "fpp" else r["io_mode"], axis=1)
    df["stripe"] = df["stripe_count"].map(lambda s: "all" if int(s) == -1 else str(int(s)))
    df["config"] = df.apply(
        lambda r: LIBRARIES[r["backend"]]
        if r["family"] == "fpp"
        else f"{LIBRARIES[r['backend']]}, {MODES[r['io_mode']]}, stripe {r['stripe']}",
        axis=1,
    )
    return df, grid


def summarise(df):
    """Mean (and min/max) per configuration and node count; memory is the peak over repetitions."""
    columns = {
        "time": ("time_total_synced", "mean"),
        "time_min": ("time_total_synced", "min"),
        "time_max": ("time_total_synced", "max"),
        "runs": ("time_total_synced", "size"),
        "compute": ("time_compute", "mean"),
        "write": ("time_write", "mean"),
        "sync": ("time_sync", "mean"),
        "data_GB": ("data_GB", "first"),
    }
    if "mem_peak_node_GB" in df:
        columns["mem_node_GB"] = ("mem_peak_node_GB", "max")
        columns["mem_rank_MB"] = ("mem_peak_rank_MB", "max")
    agg = df.groupby(["family", "config", "backend", "stripe", "nodes"]).agg(**columns).reset_index()
    agg["bw"] = agg["data_GB"] / agg["time"]
    order = {f: i for i, f in enumerate(FAMILIES)}
    lib_order = {b: i for i, b in enumerate(LIBRARIES)}
    agg["_o"] = agg["family"].map(order) * 100 + agg["backend"].map(lib_order) * 10 + (agg["stripe"] == "all")
    return agg.sort_values(["_o", "nodes"]).drop(columns="_o")


def best_per_family(agg, nodes):
    sub = agg[agg["nodes"] == nodes]
    return sub.loc[sub.groupby("family")["time"].idxmin()].set_index("family")


def style_axes(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT2, length=0)


def plot_bandwidth(agg, nodes, title, path):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ends = []
    for fam, st in FAMILIES.items():
        rows = [best_per_family(agg, n).loc[fam] for n in nodes if fam in best_per_family(agg, n).index]
        if not rows:
            continue
        x = [r["nodes"] for r in rows]
        y = [r["bw"] for r in rows]
        # error bar = bandwidth range over the repetitions of the best configuration
        yerr = [[r["bw"] - r["data_GB"] / r["time_max"] for r in rows], [r["data_GB"] / r["time_min"] - r["bw"] for r in rows]]
        ax.errorbar(x, y, yerr=yerr if max(r["runs"] for r in rows) > 1 else None, color=st["color"], ls=st["ls"],
                    lw=st["lw"], marker="o", ms=6, capsize=3, label=st["name"])
        ends.append([y[-1], f"{st['name']} {y[-1]:.1f}", st["color"] if fam == "fpp" else "#0b0b0b", fam == "fpp"])
    # end labels, pushed apart so they never overlap
    ymax = max(e[0] for e in ends) * 1.15
    gap = ymax * 0.055
    ends.sort(key=lambda e: e[0], reverse=True)
    for k in range(1, len(ends)):
        ends[k][0] = min(ends[k][0], ends[k - 1][0] - gap)
    for yv, text, color, bold in ends:
        ax.annotate(text, (nodes[-1], yv), xytext=(10, 0), textcoords="offset points", va="center",
                    color=color, fontweight="bold" if bold else "normal", fontsize=9.5)
    ax.set_xticks(nodes, [f"{n} node{'s' if n > 1 else ''}" for n in nodes])
    ax.set_xlim(nodes[0] - 0.3 * (nodes[-1] - nodes[0] or 1), nodes[-1] + 0.25 * (nodes[-1] - nodes[0] or 1))
    ax.set_ylim(0, ymax)
    ax.set_ylabel("GB/s", color=TEXT2)
    ax.grid(axis="y", color=GRID_COLOR)
    style_axes(ax)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=2)
    fig.subplots_adjust(right=0.72)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_split(agg, nodes, title, path):
    best = best_per_family(agg, nodes)
    fams = [f for f in FAMILIES if f in best.index]
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(fams) + 1.2))
    xmax = (best["compute"] + best["write"] + best["sync"]).max()
    for i, fam in enumerate(fams):
        r, left = best.loc[fam], 0.0
        for key, label, color in SPLIT:
            ax.barh(i, r[key], left=left, color=color, height=0.55, label=label if i == 0 else None)
            left += r[key]
        ax.text(left + xmax * 0.01, i, f"{r['time']:.1f} s · compute {100 * r['compute'] / r['time']:.0f}%",
                va="center", fontsize=9, color=TEXT2)
    ax.set_yticks(range(len(fams)), [FAMILIES[f]["name"] for f in fams])
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.25)
    ax.set_xlabel("seconds", color=TEXT2)
    ax.grid(axis="x", color=GRID_COLOR)
    ax.set_axisbelow(True)
    style_axes(ax)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, ncols=3, loc="upper left", bbox_to_anchor=(0, -0.25 if len(fams) > 2 else -0.4))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_table(agg, nodes):
    """One row per configuration, grouped by family: time + bandwidth per node count, speed-up."""
    best_last = best_per_family(agg, nodes[-1])
    rows = []
    for fam, st in FAMILIES.items():
        sub = agg[agg["family"] == fam]
        if sub.empty:
            continue
        for config, g in sub.groupby("config", sort=False):
            g = g.set_index("nodes")
            row = {"family": st["name"], "configuration": config}
            for n in nodes:
                row[f"time_{n}n_s"] = g["time"].get(n)
                row[f"bw_{n}n_GBps"] = g["bw"].get(n)
                row[f"runs_{n}n"] = g["runs"].get(n)
                if "mem_node_GB" in g:
                    row[f"mem_node_{n}n_GB"] = g["mem_node_GB"].get(n)
                    row[f"mem_rank_{n}n_MB"] = g["mem_rank_MB"].get(n)
            t0, t1 = g["time"].get(nodes[0]), g["time"].get(nodes[-1])
            row["speedup"] = t0 / t1 if t0 and t1 and len(nodes) > 1 else None
            row["best"] = fam in best_last.index and best_last.loc[fam, "config"] == config
            rows.append(row)
    return pd.DataFrame(rows)


def plot_table(table, nodes, title, path):
    label = lambda n: f"{n} node{'s' if n > 1 else ''}"  # noqa: E731
    has_mem = f"mem_node_{nodes[0]}n_GB" in table
    header = ["Configuration"] + [label(n) for n in nodes]
    widths = [0.40] + [0.22] * len(nodes)
    if len(nodes) > 1:
        header.append(f"Speed-up\n{nodes[0]} to {nodes[-1]} nodes")
        widths.append(0.13)
    if has_mem:
        header += [f"Peak mem./node\n{label(n)}" for n in nodes]
        widths += [0.19] * len(nodes)
    cells, kinds = [], []  # kinds: "group" header line, or row best flag
    for fam, g in table.groupby("family", sort=False):
        cells.append([fam.upper()] + [""] * (len(header) - 1))
        kinds.append("group")
        for _, r in g.iterrows():
            line = [r["configuration"]]
            for n in nodes:
                t = r[f"time_{n}n_s"]
                line.append("–" if pd.isna(t) else f"{t:.1f} s  ({r[f'bw_{n}n_GBps']:.1f} GB/s)")
            if len(nodes) > 1:
                line.append("–" if pd.isna(r["speedup"]) else f"{r['speedup']:.2f}×")
            if has_mem:
                for n in nodes:
                    m = r[f"mem_node_{n}n_GB"]
                    line.append("–" if pd.isna(m) else f"{m:.1f} GB")
            cells.append(line)
            kinds.append(bool(r["best"]))
    total = sum(widths)
    fig, ax = plt.subplots(figsize=(8.6 * total, 0.27 * (len(cells) + 1) + 0.6))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=header, loc="upper left", cellLoc="right", colLoc="right",
                   colWidths=[w / total for w in widths])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.3)
    for (i, j), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.visible_edges = "B"
        if j == 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if i == 0:
            cell.set_text_props(color=TEXT2, va="bottom")
            cell.set_height(cell.get_height() * 1.9)  # two-line headers
        elif kinds[i - 1] == "group":
            cell.set_text_props(color=TEXT2, fontsize=7.5)
            cell.visible_edges = ""
        elif kinds[i - 1]:
            cell.set_text_props(fontweight="bold")
    ax.set_title(title, loc="left", fontsize=11)
    if has_mem:
        # Footnote
        fig.canvas.draw()
        box = tbl.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(box.x0 + 0.005, box.y0 - 0.012, "Peak mem./node: sum of the per-rank peak resident memory on the "
                 "most loaded node (peak over repetitions).", fontsize=8, color=TEXT2, ha="left", va="top")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def setup_parser(parser):
    parser.add_argument("csv", type=Path, help="IOPS results CSV (results_init_bench.csv)")
    parser.add_argument("-o", "--output", type=Path, default=Path("."), help="output folder (default: .)")
    parser.add_argument("--grid", help="grid to plot when the CSV holds several (small, medium, large)")
    parser.add_argument("--format", default="png", help="figure format: png, pdf (default: png)")

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        setup_parser(parser)
        args = parser.parse_args()

    df, grid = load(args.csv, args.grid)
    agg = summarise(df)
    nodes = sorted(int(n) for n in agg["nodes"].unique())
    data_gb = agg["data_GB"].iloc[0]
    reps = int(agg["runs"].max())
    tag = f"{grid} grid, {data_gb:.1f} GB, {reps} run{'s' if reps > 1 else ''} per point"
    args.output.mkdir(parents=True, exist_ok=True)

    plot_bandwidth(agg, nodes, f"Bandwidth incl. compute and fsync, best configuration per family ({tag})",
                   args.output / f"bandwidth_vs_nodes.{args.format}")
    plot_split(agg, nodes[-1], f"Where the time goes on {nodes[-1]} node{'s' if nodes[-1] > 1 else ''} ({tag})",
               args.output / f"time_split.{args.format}")
    table = build_table(agg, nodes)
    table.to_csv(args.output / "configurations_table.csv", index=False, float_format="%.4g")
    plot_table(table, nodes, f"All configurations, time to compute and write ({tag})",
               args.output / f"configurations_table.{args.format}")

    best = best_per_family(agg, nodes[-1])
    print(f"{tag}; best per family on {nodes[-1]} node(s):")
    for fam in FAMILIES:
        if fam in best.index:
            r = best.loc[fam]
            print(f"  {FAMILIES[fam]['name']:<28} {r['config']:<32} {r['time']:7.1f} s {r['bw']:6.2f} GB/s")
    print(f"figures written to {args.output}/")


if __name__ == "__main__":
    main()

