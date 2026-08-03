import argparse
import csv
import glob
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from compression_methods.neural_network import (
    AVAILABLE_INR_ARCHS,
    OnlineNeuralNetworkCompressor,
    _DEFAULT_RECON_CHUNK_SIZE,
    _vmap_in_chunks,
    assemble_global_field,
    continue_training_offline,
    load_online_params,
)


# ---------------------------------------------------------------------------
# Plot style -- tweak here to resize every figure at once. Current values are
# sized for paper figures, not screen viewing.
# ---------------------------------------------------------------------------
TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 22
TICK_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16
RANK_LABEL_FONTSIZE = 25
BOX_LINEWIDTH = 6           # global-row rank boxes; local-row frames match this explicitly (not via rcParams --
                            # axes.linewidth would also thicken the global row's own (uncolored) axes border)
DIAG_FIG_WIDTH = 12         # inches, plot_diags figure width
DIAG_ROW_HEIGHT = 3.5       # inches per row, plot_diags figure
COMBINED_COL_WIDTH = 5    # inches per column, plot_combined figure
COMBINED_ROW_HEIGHT = 4   # inches per row, plot_combined figure

plt.rcParams.update({
    "font.size": TICK_LABEL_FONTSIZE,
    "axes.titlesize": TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "patch.linewidth": BOX_LINEWIDTH,
})


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

    has_cpu_time = any("cpu_time" in data for _, data in datasets)
    n = len(RAW_QUANTITIES) + len(CONSERVED_QUANTITIES) + (1 if has_cpu_time else 0)
    fig, axs = plt.subplots(n, 1, figsize=(DIAG_FIG_WIDTH, DIAG_ROW_HEIGHT * n), sharex=True)

    for ax, (col, ylabel) in zip(axs, RAW_QUANTITIES):
        for label, data in datasets:
            if col not in data:
                continue
            ax.semilogy(data["time"], np.abs(data[col]), label=label)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        if len(datasets) > 1:
            ax.legend()

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
            ax.legend()

    if has_cpu_time:
        ax = axs[-1]
        for label, data in datasets:
            if "cpu_time" not in data:
                continue
            ax.plot(data["time"], data["cpu_time"], label=label)
        ax.set_ylabel("CPU time [s]")
        ax.grid(True, which="both")
        if len(datasets) > 1:
            ax.legend()

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
        return Path(args.output_dir)
    
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

def _load_final_snapshot(branch_dir, species=0):
    """Load the final fdistribu snapshot (one species, global domain) and mesh
    coordinates written by the "last_iteration" PDI event for one branch's
    work directory (GYSELALIBXX_<iter>.h5 + GYSELALIBXX_initstate.h5).
    """
    import h5py

    snapshot_paths = sorted(glob.glob(os.path.join(branch_dir, "GYSELALIBXX_[0-9]*.h5")))
    if not snapshot_paths:
        raise FileNotFoundError(f"No final snapshot GYSELALIBXX_<iter>.h5 found in {branch_dir!r}")

    with h5py.File(snapshot_paths[-1], "r") as f:
        fdistribu = np.asarray(f["fdistribu"][species])
        time_saved = float(np.asarray(f["time_saved"]))

    initstate_path = os.path.join(branch_dir, "GYSELALIBXX_initstate.h5")
    with h5py.File(initstate_path, "r") as f:
        bounds = (
            float(np.asarray(f["MeshX"]).min()), float(np.asarray(f["MeshX"]).max()),
            float(np.asarray(f["MeshY"]).min()), float(np.asarray(f["MeshY"]).max()),
            float(np.asarray(f["MeshVx"]).min()), float(np.asarray(f["MeshVx"]).max()),
            float(np.asarray(f["MeshVy"]).min()), float(np.asarray(f["MeshVy"]).max()),
        )

    return fdistribu, bounds, time_saved


def plot_final_snapshot_comparison(run_dir, plane="xvx", species=0, index=None, output=None):
    """Compare the final fdistribu snapshot of the baseline branch against the
    first compressed branch found, side by side, plus their difference.

    Looks for a "branch_baseline" directory and the first other "branch_*"
    directory under run_dir that has a final GYSELALIBXX_<iter>.h5 snapshot.
    Saves to output path if given, otherwise shows interactively.
    """
    branch_dirs = sorted(
        d for d in glob.glob(os.path.join(run_dir, "branch_*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "GYSELALIBXX_[0-9]*.h5"))
    )
    baseline_dir = next((d for d in branch_dirs if os.path.basename(d) == "branch_baseline"), None)
    other_dirs = [d for d in branch_dirs if d != baseline_dir]

    if baseline_dir is None or not other_dirs:
        print("\nMissing baseline or compressed final snapshot -- skipping snapshot comparison plot.")
        return
    compressed_dir = other_dirs[0]

    baseline_f, bounds, _ = _load_final_snapshot(baseline_dir, species=species)
    compressed_f, _, _ = _load_final_snapshot(compressed_dir, species=species)

    extent = _plane_extent(bounds, plane)
    baseline_2d, axes_labels = _slice_2d(baseline_f, plane=plane, index=index)
    compressed_2d, _ = _slice_2d(compressed_f, plane=plane, index=index)
    diff_2d = compressed_2d - baseline_2d

    fig, axs = plt.subplots(1, 3, figsize=(3 * COMBINED_COL_WIDTH, COMBINED_ROW_HEIGHT))
    for ax, data, title in (
        (axs[0], baseline_2d, os.path.basename(baseline_dir)),
        (axs[1], compressed_2d, os.path.basename(compressed_dir)),
        (axs[2], diff_2d, "difference"),
    ):
        im = ax.imshow(np.asarray(data).T, origin="lower", aspect="auto", extent=extent)
        ax.set_title(title)
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"Plot written to: {output}")
    else:
        plt.show()

    plt.close(fig)


# Online (in-situ) neural-network evaluation


def _find_rank_files(data_dir, it):
    pattern = os.path.join(data_dir, f"params_iter{it:05d}_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No saved params found for iter={it} in {data_dir!r} (pattern: {pattern})")
    return paths


def _slice_2d(array_4d, plane="xvx", index=None):
    """Reduce a (nx, ny, nvx, nvy) array to a 2D slice for plotting.

    plane="xvx": fix (y, vy) at `index` (default: central indices), return (nx, nvx).
    plane="xy": fix (vx, vy) at `index`, return (nx, ny).
    plane="yvy": fix (x, vx) at `index`, return (ny, nvy).
    plane="vxvy": fix (x, y) at `index`, return (nvx, nvy).
    """
    nx, ny, nvx, nvy = array_4d.shape
    if plane == "xvx":
        iy, ivy = index if index is not None else (ny // 2, nvy // 2)
        return array_4d[:, iy, :, ivy], (r"$x$", r"$v_x$")
    elif plane == "xy":
        ivx, ivy = index if index is not None else (nvx // 2, nvy // 2)
        return array_4d[:, :, ivx, ivy], (r"$x$", r"$y$")
    elif plane == "yvy":
        ix, ivx = index if index is not None else (nx // 2, nvx // 2)
        return array_4d[ix, :, ivx, :], (r"$y$", r"$v_y$")
    elif plane == "vxvy":
        ix, iy = index if index is not None else (nx // 2, ny // 2)
        return array_4d[ix, iy, :, :], (r"$v_x$", r"$v_y$")
    else:
        raise ValueError(f"Unknown plane {plane!r}, expected 'xvx', 'xy', 'yvy', or 'vxvy'")


def evaluate_rank(data_dir, it, rank, species=0):
    """Load one rank's saved network + the local data it was fit on, and
    evaluate that same network on its own local grid (local-only reconstruction).
    """
    path = os.path.join(data_dir, f"params_iter{it:05d}_rank{rank:03d}.npz")
    payload = load_online_params(path)

    model = payload["models"][species]
    nx, ny, nvx, nvy = payload["local_shape"]
    inputs = OnlineNeuralNetworkCompressor._build_local_inputs(nx, ny, nvx, nvy)
    recon = np.asarray(jax.vmap(model)(inputs)).reshape(nx, ny, nvx, nvy)
    target = payload["target"][species]
    plt.figure()
    plt.pcolormesh(target[:,16,:, 16])
    plt.savefig("target.png")
    plt.show()

    return payload, target, recon


_AXIS_ENDPOINT = (False, False, True, True)  # x, y periodic (half-open); vx, vy inclusive


def _axis_coords(shape, bounds):
    """Physical coordinate arrays (x, y, vx, vy) for one rank's own local grid,
    using the same half-open/inclusive convention as the assembled global grid.
    """
    return tuple(
        np.linspace(bounds[2 * axis], bounds[2 * axis + 1], shape[axis], endpoint=_AXIS_ENDPOINT[axis])
        for axis in range(4)
    )


def _assemble_global_grid(local_shapes, local_bounds_list):
    """Infer the full global grid coordinates and each rank's integer offset into
    it along each of the 4 physical axes (x, y, vx, vy), from a set of per-rank
    (local_shape, local_bounds) tiles that partition the domain edge-to-edge with
    no overlap (as recorded by OnlineNeuralNetworkCompressor / apply_online_compression).

    Returns ((x_coords, y_coords, vx_coords, vy_coords), offsets), where offsets[i]
    is a 4-tuple of starting indices for rank i's chunk in the assembled global array.
    """
    coords_per_axis = []
    starts_per_axis = []  # one {interval_key: start_index} dict per axis

    for axis in range(4):
        intervals = {}
        for shape, bounds in zip(local_shapes, local_bounds_list):
            lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
            intervals[(round(lo, 9), round(hi, 9))] = shape[axis]

        offset = 0
        starts = {}
        chunks = []
        for key in sorted(intervals):
            lo, hi = key
            n = intervals[key]
            chunks.append(np.linspace(lo, hi, n, endpoint=_AXIS_ENDPOINT[axis]))
            starts[key] = offset
            offset += n

        coords_per_axis.append(np.concatenate(chunks))
        starts_per_axis.append(starts)

    offsets = [
        tuple(
            starts_per_axis[axis][(round(bounds[2 * axis], 9), round(bounds[2 * axis + 1], 9))]
            for axis in range(4)
        )
        for bounds in local_bounds_list
    ]

    return tuple(coords_per_axis), offsets


def evaluate_global_domain(data_dir, it, species=0):
    """Assemble the FULL global domain from every rank's saved network at this
    iteration. Returns (global_coords, target, recon, rank_regions): global_coords
    is (x_c, y_c, vx_c, vy_c); rank_regions maps rank -> (local_shape, local_bounds).
    """
    paths = _find_rank_files(data_dir, it)
    payloads = [load_online_params(p) for p in paths]

    missing = [p["rank"] for p in payloads if p["local_bounds"] is None]
    if missing:
        raise ValueError(
            f"Cannot assemble a global reconstruction: ranks {missing} have no "
            "local_bounds recorded (pass local_bounds= to apply_online_compression)."
        )

    local_models = [p["models"] for p in payloads]
    local_bounds = [p["local_bounds"] for p in payloads]
    local_shapes = [p["local_shape"] for p in payloads]

    (x_c, y_c, vx_c, vy_c), offsets = _assemble_global_grid(local_shapes, local_bounds)
    nx, ny, nvx, nvy = len(x_c), len(y_c), len(vx_c), len(vy_c)
    n_species = payloads[0]["target"].shape[0]

    global_target = np.zeros((n_species, nx, ny, nvx, nvy))
    for payload, (ox, oy, ovx, ovy) in zip(payloads, offsets):
        nxi, nyi, nvxi, nvyi = payload["local_shape"]
        global_target[:, ox:ox + nxi, oy:oy + nyi, ovx:ovx + nvxi, ovy:ovy + nvyi] = payload["target"]

    Xg, Yg, VXg, VYg = np.meshgrid(x_c, y_c, vx_c, vy_c, indexing="ij")
    query_points = np.stack([Xg.ravel(), Yg.ravel(), VXg.ravel(), VYg.ravel()], axis=-1)

    global_out = assemble_global_field(local_models, local_bounds, jnp.asarray(query_points))
    recon = np.asarray(global_out[species]).reshape(nx, ny, nvx, nvy)

    rank_regions = {
        payload["rank"]: (payload["local_shape"], payload["local_bounds"]) for payload in payloads
    }

    return (x_c, y_c, vx_c, vy_c), global_target[species], recon, rank_regions


_PLANE_AXES = {"xvx": (0, 2), "xy": (0, 1), "yvy": (1, 3), "vxvy": (2, 3)}  # (x, y, vx, vy) -> plane axis indices
_PLANE_FIXED_AXES = {"xvx": (1, 3), "xy": (2, 3), "yvy": (0, 2), "vxvy": (0, 1)}  # axes held fixed, _slice_2d's index order
_RANK_COLORS = [f"C{i}" for i in range(10)]  # matplotlib's default color cycle
_RANK_LINESTYLES = ["-", "--", ":", "-."]


def _color_for_index(i):
    return _RANK_COLORS[i % len(_RANK_COLORS)]


def _linestyle_for_index(i):
    return _RANK_LINESTYLES[i % len(_RANK_LINESTYLES)]


def _local_slice_index(local_shape, local_bounds, plane, global_coords):
    """Index into this rank's own local grid, along the plane's fixed axes,
    nearest the physical point the global slice uses for those axes -- so a
    rank's local row shows the same physical cut as the global row instead of
    an index centered on its own (possibly disjoint) subdomain.
    """
    fixed_axes = _PLANE_FIXED_AXES[plane]
    local_axis_coords = _axis_coords(local_shape, local_bounds)
    return tuple(
        int(np.argmin(np.abs(local_axis_coords[axis] - global_coords[axis][len(global_coords[axis]) // 2])))
        for axis in fixed_axes
    )


def _plane_extent(bounds, plane):
    """Physical (xmin, xmax, ymin, ymax) for the plane's shown axes, from an
    8-tuple of (x,y,vx,vy) bounds -- used both as imshow's extent and, for the
    global row, as the box marking where a rank's subdomain sits.
    """
    a0, a1 = _PLANE_AXES[plane]
    return bounds[2 * a0], bounds[2 * a0 + 1], bounds[2 * a1], bounds[2 * a1 + 1]


def _global_bounds_from_coords(global_coords):
    """(x,y,vx,vy) bounds 8-tuple spanning the full assembled grid, in the
    same layout as a rank's own local_bounds, for reuse with `_plane_extent`.
    """
    return tuple(v for c in global_coords for v in (float(c.min()), float(c.max())))


def _rank_box_in_plane(shape, bounds, plane):
    """Physical (x0, y0, width, height) box for one rank's region within a
    2D slice plane of the full global domain, for drawing e.g. a Rectangle patch.

    Periodic axes (x, y) are sampled half-open (`endpoint=False`): the last
    grid point -- and thus the global plot's extent, built from actual grid
    values -- falls one cell short of the nominal upper bound. Pull the box's
    upper edge in by one cell on those axes so it lines up with the plot
    instead of poking out past it.
    """
    a0, a1 = _PLANE_AXES[plane]
    x0, x1, y0, y1 = _plane_extent(bounds, plane)
    if not _AXIS_ENDPOINT[a0]:
        x1 -= (x1 - x0) / shape[a0]
    if not _AXIS_ENDPOINT[a1]:
        y1 -= (y1 - y0) / shape[a1]
    return x0, y0, x1 - x0, y1 - y0


def _plot_row(
    fig, axs_row, target_2d, recon_2d, axes_labels, title_prefix, column_ranges,
    extent=None, boxes=None, add_rank_labels=False, frame_color=None, frame_linestyle="-",
):
    diff_2d = recon_2d - target_2d
    for ax, data, label in zip(axs_row, [target_2d, recon_2d, diff_2d], ["target", "reconstruction", "error"]):
        vmin, vmax = column_ranges[label]
        im = ax.imshow(np.asarray(data).T, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix}: {label}")
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        fig.colorbar(im, ax=ax, fraction=0.046)
        for rank, box, color, linestyle in boxes or []:
            x0, y0, w, h = box
            ax.add_patch(Rectangle(
                (x0, y0), w, h, edgecolor=color, facecolor="none", linestyle=linestyle, clip_on=False,
            ))
            if add_rank_labels:
                ax.text(
                    x0 + w / 2, y0 + h / 2, f"rank {rank}", fontsize=RANK_LABEL_FONTSIZE,
                    color=color, ha="center", va="center", fontweight="bold", clip_on=False,
                )
        if frame_color is not None:
            for spine in ax.spines.values():
                spine.set_edgecolor(frame_color)
                spine.set_linestyle(frame_linestyle)
                spine.set_linewidth(BOX_LINEWIDTH)


def plot_combined(
    local_entries, global_target, global_recon, it, plane="xvx", output=None, global_extent=None,
    add_rank_labels=False,
):
    """Top row: full-domain reconstruction vs. data, with one colored/dashed
    box per rank. Each following row: one rank's own network vs. its local
    data, sliced at the same physical point as the global row, framed to
    match its box. Axes show physical coordinates, not grid indices.

    local_entries: list of (rank, target, recon, box, color, linestyle,
    index, extent) tuples, one per requested rank.

    Each of the 3 columns (target, reconstruction, error) gets its own shared
    colorbar range across rows.
    """
    global_t2d, axes_labels = _slice_2d(global_target, plane=plane)
    global_r2d, _ = _slice_2d(global_recon, plane=plane)

    local_slices = [
        (
            rank,
            _slice_2d(target, plane=plane, index=index)[0],
            _slice_2d(recon, plane=plane, index=index)[0],
            box, color, linestyle, extent,
        )
        for rank, target, recon, box, color, linestyle, index, extent in local_entries
    ]

    def _range(arrays):
        return (min(float(np.min(a)) for a in arrays), max(float(np.max(a)) for a in arrays))

    all_targets = [global_t2d] + [t2d for _, t2d, _, _, _, _, _ in local_slices]
    all_recons = [global_r2d] + [r2d for _, _, r2d, _, _, _, _ in local_slices]
    all_diffs = [global_r2d - global_t2d] + [r2d - t2d for _, t2d, r2d, _, _, _, _ in local_slices]
    column_ranges = {
        "target": _range(all_targets),
        "reconstruction": _range(all_recons),
        "error": _range(all_diffs),
    }

    n_rows = 1 + len(local_slices)
    fig, axs = plt.subplots(n_rows, 3, figsize=(3 * COMBINED_COL_WIDTH, COMBINED_ROW_HEIGHT * n_rows))
    axs = np.atleast_2d(axs)

    boxes = [
        (rank, box, color, linestyle) for rank, _, _, box, color, linestyle, _, _ in local_entries if box is not None
    ]
    _plot_row(
        fig, axs[0], global_t2d, global_r2d, axes_labels, f"global (iter={it})",
        column_ranges, extent=global_extent, boxes=boxes, add_rank_labels=add_rank_labels,
    )

    for row, (rank, t2d, r2d, _, color, linestyle, extent) in enumerate(local_slices, start=1):
        _plot_row(
            fig, axs[row], t2d, r2d, axes_labels, f"local (rank={rank})",
            column_ranges, extent=extent, frame_color=color, frame_linestyle=linestyle,
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight")
        print(f"Plot written to: {output}")
    else:
        plt.show()
    plt.close(fig)


def _online_networks_output_path(data_dir, it, ranks, species, plane):
    rank_label = "-".join(f"{r:03d}" for r in ranks)
    return os.path.join(data_dir, f"eval_iter{it:05d}_rank{rank_label}_species{species}_{plane}.png")


def run_online_networks(data_dir, it, ranks=None, species=0, plane="xvx", add_rank_labels=False):
    """Evaluate/plot saved online INR networks (params_iterXXXXX_rankXXX.npz):
    top row is the full-domain reconstruction assembled from every rank's network
    at that iteration vs. the full-domain data; each following row is one
    requested rank's own network vs. the local data it was trained on,
    color-matched to its box in the top row. `ranks=None` uses every rank found.
    """
    global_coords, global_target, global_recon, rank_regions = evaluate_global_domain(data_dir, it, species=species)
    global_extent = _plane_extent(_global_bounds_from_coords(global_coords), plane)

    ranks = ranks if ranks is not None else sorted(rank_regions)

    local_entries = []
    for i, rank in enumerate(ranks):
        _, target, recon = evaluate_rank(data_dir, it, rank, species=species)

        color = _color_for_index(i)
        linestyle = _linestyle_for_index(i)
        if rank in rank_regions:
            shape, bounds = rank_regions[rank]
            box = _rank_box_in_plane(shape, bounds, plane)
            index = _local_slice_index(shape, bounds, plane, global_coords)
            extent = _plane_extent(bounds, plane)
        else:
            box, index, extent = None, None, global_extent
        local_entries.append((rank, target, recon, box, color, linestyle, index, extent))

    out = _online_networks_output_path(data_dir, it, ranks, species, plane)
    plot_combined(
        local_entries, global_target, global_recon, it=it, plane=plane, output=out,
        global_extent=global_extent, add_rank_labels=add_rank_labels,
    )


# Offline fine-tuning (continue saved online networks without rerunning the simulation)


def finetune_rank_offline(data_dir, it, rank, out_dir=None, out_iter=None, **kwargs):
    """Continue one rank's saved online INR(s) offline against its saved
    target data (see `continue_training_offline`), then optionally save the
    result as a new params_iterXXXXX_rankXXX.npz -- same file layout as the
    online pipeline, so `evaluate_rank` / `run_online_networks` can load and
    plot the fine-tuned checkpoint unchanged.
    """
    path = os.path.join(data_dir, f"params_iter{it:05d}_rank{rank:03d}.npz")
    payload = load_online_params(path)
    result = continue_training_offline(payload, **kwargs)

    if out_dir is not None:
        save_it = out_iter if out_iter is not None else it
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"params_iter{save_it:05d}_rank{rank:03d}.npz")
        result["compressor"].save_params(out_path, rank=rank, timestep=save_it)
        print(f"Fine-tuned params written to: {out_path}")

    return result


def finetune_offline(data_dir, it, ranks=None, out_dir=None, out_iter=None, **kwargs):
    """Continue training every (or selected) rank's saved online INR(s)
    offline against its own already-captured local target data -- the full
    distributed pipeline's per-rank fits, run standalone without the
    simulation. `ranks=None` uses every rank found for this iteration.
    """
    if ranks is None:
        paths = _find_rank_files(data_dir, it)
        ranks = sorted(int(os.path.basename(p).rsplit("_rank", 1)[1].split(".")[0]) for p in paths)

    results = {}
    for rank in ranks:
        print(f"=== rank {rank} ===", flush=True)
        results[rank] = finetune_rank_offline(data_dir, it, rank, out_dir=out_dir, out_iter=out_iter, **kwargs)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate compression benchmark runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diag_parser = subparsers.add_parser(
        "diagnostics", help="Plot conserved-quantity diagnostics CSV files (offline/baseline benchmark)."
    )
    diag_parser.add_argument("csv_files", nargs="+", help="One or more diagnostics.csv files to plot.")
    diag_parser.add_argument("-o", "--output", default=None, help="Output file (PDF/PNG). Omit to show interactively.")

    online_parser = subparsers.add_parser(
        "online-networks", help="Evaluate/plot saved online INR networks (params_iterXXXXX_rankXXX.npz)."
    )
    online_parser.add_argument("data_dir", help="Directory containing params_iterXXXXX_rankXXX.npz files.")
    online_parser.add_argument("--iter", type=int, required=True, help="Iteration/timestep to evaluate.")
    online_parser.add_argument(
        "--rank", type=int, nargs="+", default=None,
        help="Rank(s) whose local network/data to show, one row each (each gets its own color). "
        "Omit to use every rank found for this iteration.",
    )
    online_parser.add_argument("--species", type=int, default=0, help="Species index to plot (default: 0).")
    online_parser.add_argument(
        "--plane", choices=["xvx", "xy", "vxvy"], default="xvx", help="2D slice plane to plot (default: xvx).",
    )
    online_parser.add_argument(
        "--add-rank-labels", action="store_true", help="Label each rank's box with \"rank X\" at its center.",
    )

    finetune_parser = subparsers.add_parser(
        "finetune-offline",
        help="Continue training saved online INR networks against their saved target data, "
        "without rerunning the simulation.",
    )
    finetune_parser.add_argument("data_dir", help="Directory containing params_iterXXXXX_rankXXX.npz files.")
    finetune_parser.add_argument("--iter", type=int, required=True, help="Saved iteration/timestep to load.")
    finetune_parser.add_argument(
        "--rank", type=int, nargs="+", default=None,
        help="Rank(s) to fine-tune. Omit to use every rank found for this iteration.",
    )
    finetune_parser.add_argument(
        "--iters-adam", type=int, default=2000, help="Total ADAM steps to run (default: 2000).",
    )
    finetune_parser.add_argument(
        "--iters-lbfgs", type=int, default=10, help="Total L-BFGS steps to run (default: 10).",
    )
    finetune_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    finetune_parser.add_argument(
        "--arch", default=None, choices=AVAILABLE_INR_ARCHS,
        help="Architecture to train. Default: reuse the saved arch, warm-started from the saved "
        "weights. A different arch trains fresh models of that arch from scratch on the same "
        "saved target data (architecture search).",
    )
    finetune_parser.add_argument(
        "--no-warm-start", action="store_true",
        help="Ignore saved weights and start from a fresh random init (still trains on the saved target data).",
    )
    finetune_parser.add_argument(
        "--species", type=int, nargs="+", default=None,
        help="Species indices to fine-tune. Omit to fine-tune every species.",
    )
    finetune_parser.add_argument(
        "--out-dir", default=None, help="Directory to write fine-tuned params_iterXXXXX_rankXXX.npz files to. "
        "Omit (without --overwrite) to skip saving.",
    )
    finetune_parser.add_argument(
        "--out-iter", type=int, default=None,
        help="Iteration tag for saved output files (default: same as --iter -- pass a different "
        "value to avoid overwriting the original checkpoint).",
    )
    finetune_parser.add_argument(
        "--overwrite", action="store_true",
        help="Save fine-tuned params back into data_dir at the same --iter, in place of the "
        "original checkpoint. Shorthand for --out-dir <data_dir> --out-iter <iter>; "
        "mutually exclusive with --out-dir/--out-iter.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Post-process gysela-mini-app_io compression benchmarks (offline multi-branch comparison)."
    )
    compare_parser.add_argument(
        "--params", type=str, nargs="+", required=True,
        help="Path(s) to diagnostics.csv (one per branch/run to compare).",
    )
    compare_parser.add_argument(
        "--summary", action="store_true",
        help="Multi-panel summary figure (raw + relative retained), auto-generated style.",
    )
    for q in PHYSICAL_QUANTITIES:
        compare_parser.add_argument(f"--{q}", action="store_true")
    compare_parser.add_argument("--frob", action="store_true")
    compare_parser.add_argument("--frob-pod", action="store_true")
    compare_parser.add_argument("--frob-inr", action="store_true")
    compare_parser.add_argument("--svd-spectrum", action="store_true")
    compare_parser.add_argument("--inr-loss", action="store_true")
    compare_parser.add_argument("--cpu-time", action="store_true")
    compare_parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory for figures (also used by --summary).",
    )

    args = parser.parse_args()
    if args.command == "finetune-offline" and args.overwrite and (args.out_dir is not None or args.out_iter is not None):
        parser.error("--overwrite is mutually exclusive with --out-dir/--out-iter")
    return args

def main():
    args = parse_args()

    if args.command == "diagnostics":
        plot_diags(args.csv_files, output=args.output)
    elif args.command == "online-networks":
        run_online_networks(
            args.data_dir, args.iter, ranks=args.rank, species=args.species, plane=args.plane,
            add_rank_labels=args.add_rank_labels,
        )
    elif args.command == "finetune-offline":
        out_dir = args.data_dir if args.overwrite else args.out_dir
        out_iter = args.iter if args.overwrite else args.out_iter
        finetune_offline(
            args.data_dir, args.iter,
            ranks=args.rank, out_dir=out_dir, out_iter=out_iter,
            arch=args.arch, species=args.species, lr=args.lr,
            iters_adam=args.iters_adam, iters_lbfgs=args.iters_lbfgs,
            warm_start=not args.no_warm_start,
        )
    elif args.command == "compare":
        quantities = [q for q in PHYSICAL_QUANTITIES if getattr(args, q.replace("-", "_"))]

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
            plot_frobenius(compression_cases, out_dir, filt="nn", name="frob_error_nn")
        if args.cpu_time:
            plot_cpu_time(compression_cases, out_dir)
        if args.svd_spectrum:
            plot_svd_spectrum(data_dirs, compression_cases, out_dir)
        if args.inr_loss:
            plot_inr_loss(data_dirs, out_dir)

        print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
