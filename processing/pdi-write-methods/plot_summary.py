"""Draw the summary plots of an IOPS run (gys_write_methods / gys_scaling) from its results CSV.

Call it once a run is finished, with the CSV written by IOPS:

    python plot_summary.py output/results_write_methods.csv
    python plot_summary.py output/results_scaling_256GiB.csv
    python plot_summary.py output/results_write_methods_16GiB.csv output/results_write_methods_128GiB.csv \\
                           output/results_scaling_256GiB.csv -o figures --format png pdf

Output (in -o, default: next to the first CSV):
  --format png       one file per figure (default: png)
  --format pdf       one multi-page file, plot_summary.pdf
  --theme dark       dark background
A summary table (mean, min and max time, bandwidth, number of runs per configuration) is printed, with
the configurations that have no valid run (timeouts, failed checks).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

THEMES = {
    "light": dict(bg="#fcfcfb", tx="#0b0b0b", tx2="#52514e", grid="#e3e2de", acc="#2a78d6", acc_l="#a9c8f0",
                  dk="#0b0b0b", md="#52514e", lt="#9a9993"),
    "dark": dict(bg="#1a1a19", tx="#f0efec", tx2="#c3c2b7", grid="#34342f", acc="#3987e5", acc_l="#27507f",
                 dk="#f0efec", md="#c3c2b7", lt="#8d8c85"),
}
C: dict = {}

METHODS = [  # order used in the charts
    ("per_rank_netcdf", "One file per rank, netCDF4"),
    ("per_rank_hdf5", "One file per rank, HDF5"),
    ("shared_hdf5", "Shared file, HDF5"),
    ("subfiling_hdf5", "HDF5 subfiling"),
    ("shared_netcdf", "Shared file, netCDF4"),
]
NAME = dict(METHODS)
STRIPE_LABEL = {1: "1", 8: "8", -1: "all OSTs"}


def method_style(m: str) -> dict:
    """Line style per method (stripe-count panels)."""
    return {
        "per_rank_netcdf": dict(c=C["acc"], ls="-", hollow=False),
        "per_rank_hdf5": dict(c=C["acc"], ls="-", hollow=True),
        "shared_hdf5": dict(c=C["dk"], ls="-", hollow=False),
        "subfiling_hdf5": dict(c=C["md"], ls=(0, (6, 4)), hollow=False),
        "shared_netcdf": dict(c=C["lt"], ls=(0, (2, 4)), hollow=False),
    }.get(m, dict(c=C["md"], ls="-", hollow=False))


def pick_font() -> str:
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    return next((f for f in ("Inter", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans") if f in have),
                "sans-serif")


def apply_theme(name: str) -> None:
    C.clear()
    C.update(THEMES[name])
    plt.rcParams.update({
        "figure.facecolor": C["bg"], "axes.facecolor": C["bg"], "savefig.facecolor": C["bg"],
        "text.color": C["tx"], "axes.labelcolor": C["tx2"], "xtick.color": C["tx2"], "ytick.color": C["tx2"],
        "axes.edgecolor": C["grid"], "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.9,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "font.family": pick_font(), "font.size": 10,
        "pdf.fonttype": 42,
    })


class Run:
    """One results CSV: valid runs, payload size and the configurations that produced nothing."""

    def __init__(self, path: Path):
        raw = pd.read_csv(path)
        for c in ("vars.method", "metrics.time_write"):
            if c not in raw.columns:
                raise SystemExit(f"{path}: column {c} missing - is this an IOPS results CSV of these benchmarks?")

        def num(c):
            return pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else pd.Series(np.nan, index=raw.index)

        df = pd.DataFrame({"method": raw["vars.method"].astype(str), "stripe": num("vars.stripe_count"),
                           "nodes": num("vars.nodes"), "tpn": num("vars.tasks_per_nodes"),
                           "ioc": num("vars.ioc_per_node"), "time": num("metrics.time_write"),
                           "bytes": num("metrics.bytes_written")})
        bad = df[~(df["time"] > 0)]
        self.df = df[df["time"] > 0].copy()          # timeouts / failed checks have no measurement
        if self.df.empty:
            raise SystemExit(f"{path}: no run with a valid metrics.time_write")
        self.path, self.total = path, len(df)
        self.dropped = len(bad)
        valid_cfgs = set(map(tuple, self.df[self.key_cols(self.df)].itertuples(index=False)))
        self.missing = sorted({t for t in map(tuple, bad[self.key_cols(bad)].itertuples(index=False))
                               if t not in valid_cfgs}, key=str)
        gib = self.df["bytes"].median() / 2**30       # drop the few-MB metadata overhead
        self.bytes = round(gib) * 2**30 if np.isfinite(gib) else np.nan
        self.is_scaling = self.df["nodes"].nunique() > 1

    @staticmethod
    def key_cols(df):
        return ["method", "stripe", "nodes", "ioc"]

    @property
    def size(self) -> str:
        return f"{self.bytes / 2**30:.0f} GiB" if np.isfinite(self.bytes) else self.path.stem

    def bw(self, t):
        return self.bytes / 1e9 / t


def group_name(method: str, ioc, tpn) -> str:
    if method.startswith("per_rank"):
        return "One file per rank"
    if method == "subfiling_hdf5":
        if pd.notna(ioc) and pd.notna(tpn) and ioc == tpn:
            return "Subfiling, 1 concentrator per rank"
        return f"Subfiling, {int(ioc)} per node" if pd.notna(ioc) else "HDF5 subfiling"
    return NAME.get(method, method)


def place_end_labels(ax, items, dx=12, gap=15.0) -> None:
    """Direct labels at the end of the lines, pushed apart so they never overlap.
    items: (x, y, text, style) in data coordinates."""
    ax.figure.canvas.draw()
    ppp = 72.0 / ax.figure.dpi
    pts = sorted((ax.transData.transform((x, y))[1] * ppp, x, y, t, st) for x, y, t, st in items)
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (np.mean([p[0] for p in pts]) - np.mean(ys)) if pts else 0
    for (y0, x, y, t, st), y1 in zip(pts, ys):
        bold = st.get("bold", False)
        ax.annotate(t, (x, y), xytext=(dx, y1 + shift - y0), textcoords="offset points", va="center",
                    fontsize=10.5, fontweight="bold" if bold else "normal",
                    color=C["acc"] if bold else st.get("tc", C["tx"]))


def label_points(ax, pts, clear=13.0) -> None:
    """Value labels above (else below) each point; a label that would touch a neighbour is skipped."""
    ax.figure.canvas.draw()
    ppp = 72.0 / ax.figure.dpi
    cols = {}
    for x, t in pts:
        cols.setdefault(x, []).append((ax.transData.transform((x, t))[1] * ppp, t))
    for x, items in cols.items():
        ys = sorted(i[0] for i in items)
        for y, t in items:
            above = all(not (0 < o - y < clear * 1.6) for o in ys)
            below = all(not (0 < y - o < clear * 1.6) for o in ys)
            if above:
                ax.annotate(f"{t:.1f}", (x, t), xytext=(0, 8), textcoords="offset points", ha="center",
                            fontsize=8.5, color=C["tx"])
            elif below:
                ax.annotate(f"{t:.1f}", (x, t), xytext=(0, -15), textcoords="offset points", ha="center",
                            fontsize=8.5, color=C["tx"])


def plain_log_y(ax, lo, hi) -> None:
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(mt.FixedLocator([v for v in (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000)
                                                if lo <= v <= hi][::1]))
    ax.yaxis.set_major_formatter(mt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mt.NullFormatter())


def title(ax, text) -> None:
    ax.set_title(text, loc="left", fontsize=13, fontweight="bold", color=C["tx"])


def ds_labels(runs: list[Run]) -> list[str]:
    """'128 GiB', plus the node count when the CSVs do not all use the same one."""
    nodes = [int(r.df["nodes"].max()) if r.df["nodes"].notna().any() else None for r in runs]
    show = len({n for n in nodes}) > 1
    return [f"{r.size}, {n} nodes" if show and n else r.size for r, n in zip(runs, nodes)]


def chart_bandwidth(runs: list[Run]) -> plt.Figure:
    methods = [(m, n) for m, n in METHODS if any(m in r.df["method"].values for r in runs)]
    k = len(runs)
    ramp = ([C["acc_l"], C["acc"]] if k == 2 else [C["acc"]] if k == 1 else
            [matplotlib.colors.to_hex(c) for c in
             np.linspace(matplotlib.colors.to_rgb(C["acc_l"]), matplotlib.colors.to_rgb(C["acc"]), k)])
    vals = {}
    for m, _ in methods:
        for j, r in enumerate(runs):
            s = r.df[r.df["method"] == m]
            if not s.empty:
                vals[(m, j)] = r.bw(s.groupby("stripe", dropna=False)["time"].mean().min())  # best stripe count

    fig, ax = plt.subplots(figsize=(9.2, 1.0 + len(methods) * (0.38 + 0.22 * k)))
    bh = 0.8 / k
    mx = max(vals.values()) * 1.12
    for i, (m, n) in enumerate(methods):
        for j in range(k):
            if (m, j) not in vals:
                continue
            y = i + (j - (k - 1) / 2) * bh
            ax.barh(y, vals[(m, j)], height=bh * 0.82, color=ramp[j], zorder=3)
            ax.text(vals[(m, j)] + mx * 0.012, y, f"{vals[(m, j)]:.1f}", va="center", fontsize=9, color=C["tx"])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([n for _, n in methods], fontsize=10.5, color=C["tx"])
    ax.invert_yaxis()
    ax.set_xlim(0, mx)
    ax.xaxis.set_major_locator(mt.MaxNLocator(5))
    ax.grid(axis="y", visible=False)
    ax.tick_params(length=0)
    ax.set_xlabel("write bandwidth at the best stripe count (GB/s), higher is better")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ramp[j]) for j in range(k)], ds_labels(runs),
              loc="lower right", frameon=False, ncol=k, labelcolor=C["tx2"])
    fig.tight_layout()
    return fig


def chart_stripes(runs: list[Run]) -> plt.Figure | None:
    if all(r.df["stripe"].nunique() < 2 for r in runs):
        return None
    order = [1, 8, -1]
    allt = pd.concat([r.df.groupby(["method", "stripe"])["time"].mean() for r in runs])
    lo, hi = allt.min() * 0.7, allt.max() * 1.8
    k = len(runs)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k + 0.6, 5.2), sharey=True, squeeze=False)
    for ax, r, lab in zip(axes[0], runs, ds_labels(runs)):
        present = set(r.df["stripe"].dropna().astype(int))
        stripes = [s for s in order if s in present] + sorted(s for s in present if s not in order)
        pos = {s: i for i, s in enumerate(stripes)}
        a = r.df.groupby(["method", "stripe"], dropna=False)["time"].mean().reset_index()
        pts = []
        for m, _ in METHODS:
            s = a[a["method"] == m].copy()
            if s.empty:
                continue
            s["x"] = s["stripe"].map(lambda v: pos.get(int(v)))
            s = s.sort_values("x")
            st = method_style(m)
            ax.plot(s["x"], s["time"], color=st["c"], lw=2, ls=st["ls"], zorder=3)
            ax.scatter(s["x"], s["time"], s=46, zorder=4, edgecolor=st["c"], linewidth=2,
                       facecolor=C["bg"] if st["hollow"] else st["c"])
            if m != "per_rank_hdf5":      # the hollow duplicate is not labelled
                pts += list(zip(s["x"], s["time"]))
        plain_log_y(ax, lo, hi)
        ax.set_xlim(-0.35, len(stripes) - 0.65)
        label_points(ax, pts)
        ax.set_xticks(range(len(stripes)))
        ax.set_xticklabels([STRIPE_LABEL.get(s, str(s)) for s in stripes])
        ax.grid(axis="x", visible=False)
        ax.tick_params(length=0, which="both")
        title(ax, lab)
        ax.set_xlabel("stripe count")
    axes[0][0].set_ylabel("seconds (log scale), lower is better")
    used = [(m, n) for m, n in METHODS if any(m in r.df["method"].values for r in runs)]
    handles = [plt.Line2D([], [], color=method_style(m)["c"], lw=2, ls=method_style(m)["ls"], marker="o",
                          markerfacecolor=C["bg"] if method_style(m)["hollow"] else method_style(m)["c"],
                          markeredgewidth=2, ms=6) for m, _ in used]
    fig.legend(handles, [n for _, n in used], loc="upper center", ncol=len(used), frameon=False,
               labelcolor=C["tx2"], fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


SCALING_STYLES = lambda: [  # noqa: E731  fastest group in the accent colour, the others in greys
    dict(c=C["acc"], ls="-", w=2.6, bold=True), dict(c=C["dk"], ls="-", w=2.0),
    dict(c=C["md"], ls=(0, (6, 4)), w=2.0), dict(c=C["lt"], ls=(0, (2, 4)), w=2.6),
    dict(c=C["lt"], ls=(0, (6, 2, 1, 2)), w=2.0)]


def best_by_group(r: Run) -> tuple[pd.DataFrame, list[str], list[float]]:
    """Best configuration (stripe count) of every method, at every node count. Groups sorted fastest first."""
    df = r.df.copy()
    df["group"] = [group_name(m, i, t) for m, i, t in zip(df["method"], df["ioc"], df["tpn"])]
    df["cfg"] = df["group"] + "|" + df["stripe"].astype(str)
    nodes = sorted(df["nodes"].dropna().unique())
    mean = df.groupby(["group", "cfg", "nodes"])["time"].mean().reset_index()
    best = mean.loc[mean.groupby(["group", "nodes"])["time"].idxmin()]
    last = best[best["nodes"] == nodes[-1]].sort_values("time")
    return best, list(last["group"]), nodes


def _scaling(r: Run, what: str) -> plt.Figure:
    best, groups, nodes = best_by_group(r)
    tpn = r.df["tpn"].dropna().iloc[0] if r.df["tpn"].notna().any() else None
    xp = {n: np.log2(n) for n in nodes}
    styles = SCALING_STYLES()
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    labels, allv = [], []
    for i, g in enumerate(groups):
        st = styles[i % len(styles)]
        s = best[best["group"] == g].sort_values("nodes")
        x = [xp[n] for n in s["nodes"]]
        y = [r.bw(t) for t in s["time"]] if what == "bw" else list(s["time"])
        ax.plot(x, y, color=st["c"], lw=st["w"], ls=st["ls"], zorder=3)
        ax.scatter(x, y, s=44, color=st["c"], zorder=4)
        labels.append((x[-1], y[-1], f"{g} {y[-1]:.1f}", st))
        allv += y
    if what == "time":
        fast = best[best["group"] == groups[0]].sort_values("nodes")
        n0, t0 = fast["nodes"].iloc[0], fast["time"].iloc[0]
        xs = np.array([xp[n] for n in nodes])
        ax.plot(xs, [t0 * n0 / n for n in nodes], ls=(0, (1, 3)), color=C["tx2"], lw=1.2, zorder=2)
        labels.append((xs[-1], t0 * n0 / nodes[-1], "ideal scaling", dict(tc=C["tx2"])))
        allv += [t0 * n0 / nodes[-1]]
        plain_log_y(ax, min(allv) * 0.75, max(allv) * 1.5)
    else:
        ax.set_ylim(0, max(allv) * 1.08)
    place_end_labels(ax, labels)
    ax.set_xticks([xp[n] for n in nodes])
    ax.set_xticklabels([f"{int(n)} nodes" + (f" ({int(n * tpn)} ranks)" if tpn else "") for n in nodes])
    ax.set_xlim(xp[nodes[0]] - 0.3, xp[nodes[-1]] + 1.9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0, which="both")
    if what == "bw":
        ax.set_ylabel("GB/s, higher is better")
        title(ax, f"Write bandwidth at each method's best stripe count · {r.size}")
    else:
        ax.set_ylabel("seconds (log scale), lower is better")
        title(ax, f"Write time at each method's best stripe count · {r.size}")
    fig.tight_layout()
    return fig


def summary_table(r: Run) -> str:
    df = r.df.copy()
    df["method"] = [group_name(m, i, t) if m == "subfiling_hdf5" else m for m, i, t in
                    zip(df["method"], df["ioc"], df["tpn"])]
    df["stripe"] = df["stripe"].map(lambda s: "" if pd.isna(s) else STRIPE_LABEL.get(int(s), str(int(s))))
    by = ["method", "stripe"] + (["nodes"] if r.is_scaling else [])
    a = df.groupby(by, dropna=False).agg(time_s=("time", "mean"), min_s=("time", "min"), max_s=("time", "max"),
                                         runs=("time", "size")).reset_index()
    a["GB/s"] = r.bw(a["time_s"])
    a = a.sort_values((["nodes"] if r.is_scaling else []) + ["time_s"])
    return a.to_string(index=False, float_format=lambda v: f"{v:.2f}")


def report_run(r: Run) -> None:
    kind = "scaling" if r.is_scaling else "comparison"
    print(f"\n== {r.path.name}: {r.size}, {kind}, {len(r.df)} valid runs of {r.total}")
    print(summary_table(r))
    if r.dropped:
        print(f"-- {r.dropped} run(s) without a valid time_write (timeout or failed check), not plotted:")
        for m, s, n, i in r.missing:
            bits = [str(m), "stripe " + ("?" if pd.isna(s) else STRIPE_LABEL.get(int(s), str(int(s))))]
            if pd.notna(n):
                bits.append(f"{int(n)} nodes")
            if pd.notna(i) and m == "subfiling_hdf5":
                bits.append(f"ioc_per_node={int(i)}")
            print("   no valid run at all for:", ", ".join(bits))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+", type=Path, help="IOPS results CSV(s), the only input")
    parser.add_argument("-o", "--outdir", type=Path, default=None, help="output folder (default: next to the first CSV)")
    parser.add_argument("--format", nargs="+", choices=["png", "pdf"], default=["png"])
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--no-table", action="store_true", help="do not print the summary tables")
    parsed = parser.parse_args()

    apply_theme(parsed.theme)
    outdir = parsed.outdir or parsed.csv[0].resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [Run(p) for p in parsed.csv]
    if not parsed.no_table:
        for r in runs:
            report_run(r)

    figs: list[tuple[str, plt.Figure]] = []
    comp = sorted((r for r in runs if not r.is_scaling), key=lambda r: (r.bytes, r.path.name))
    if comp:
        figs.append(("bandwidth_per_method", chart_bandwidth(comp)))
        f = chart_stripes(comp)
        if f is not None:
            figs.append(("time_vs_stripe_count", f))
    scal = [r for r in runs if r.is_scaling]
    for r in scal:
        suffix = f"_{r.size.replace(' ', '')}" if len(scal) > 1 else ""
        figs.append((f"scaling_bandwidth{suffix}", _scaling(r, "bw")))
        figs.append((f"scaling_time{suffix}", _scaling(r, "time")))

    print()
    for ext in parsed.format:
        if ext == "pdf":
            p = outdir / "plot_summary.pdf"
            with PdfPages(p) as pdf:
                for _, f in figs:
                    pdf.savefig(f, bbox_inches="tight")
            print("wrote", p)
        else:
            for name, f in figs:
                p = outdir / f"{name}.{ext}"
                f.savefig(p, dpi=200, bbox_inches="tight")
                print("wrote", p)


if __name__ == "__main__":
    main()
