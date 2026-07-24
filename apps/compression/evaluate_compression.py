import argparse
import csv
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from compression_methods.neural_network import (
    AVAILABLE_INR_ARCHS,
    OnlineNeuralNetworkCompressor,
    assemble_global_field,
    continue_training_offline,
    load_online_params,
)


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
    plane="vxvy": fix (x, y) at `index`, return (nvx, nvy).
    """
    nx, ny, nvx, nvy = array_4d.shape
    if plane == "xvx":
        iy, ivy = index if index is not None else (ny // 2, nvy // 2)
        return array_4d[:, iy, :, ivy], ("x", "vx")
    elif plane == "xy":
        ivx, ivy = index if index is not None else (nvx // 2, nvy // 2)
        return array_4d[:, :, ivx, ivy], ("x", "y")
    elif plane == "vxvy":
        ix, iy = index if index is not None else (nx // 2, ny // 2)
        return array_4d[ix, iy, :, :], ("vx", "vy")
    else:
        raise ValueError(f"Unknown plane {plane!r}, expected 'xvx', 'xy', or 'vxvy'")


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


def _assemble_global_grid(local_shapes, local_bounds_list):
    """Infer the full global grid coordinates and each rank's integer offset into
    it along each of the 4 physical axes (x, y, vx, vy), from a set of per-rank
    (local_shape, local_bounds) tiles that partition the domain edge-to-edge with
    no overlap (as recorded by OnlineNeuralNetworkCompressor / apply_online_compression).

    Returns ((x_coords, y_coords, vx_coords, vy_coords), offsets), where offsets[i]
    is a 4-tuple of starting indices for rank i's chunk in the assembled global array.
    """
    endpoint_per_axis = (False, False, True, True)  # x, y periodic; vx, vy inclusive
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
            chunks.append(np.linspace(lo, hi, n, endpoint=endpoint_per_axis[axis]))
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
    iteration: the exact per-rank ground-truth chunks tiled together, and the
    partition-of-unity reconstruction (`assemble_global_field`) evaluated on
    that same full grid.
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
        payload["rank"]: (offset, payload["local_shape"]) for payload, offset in zip(payloads, offsets)
    }

    return global_target[species], recon, rank_regions


_PLANE_AXES = {"xvx": (0, 2), "xy": (0, 1), "vxvy": (2, 3)}  # (x, y, vx, vy) -> plane axis indices
_RANK_COLORS = [f"C{i}" for i in range(10)]  # matplotlib's default color cycle


def _color_for_index(i):
    return _RANK_COLORS[i % len(_RANK_COLORS)]


def _rank_box_in_plane(offset, shape, plane):
    """Pixel-index (x0, y0, width, height) box for one rank's region within a
    2D slice plane of the full global array, for drawing e.g. a Rectangle patch.
    """
    a0, a1 = _PLANE_AXES[plane]
    return offset[a0] - 0.5, offset[a1] - 0.5, shape[a0], shape[a1]


def _plot_row(fig, axs_row, target_2d, recon_2d, axes_labels, title_prefix, column_ranges, boxes=None, frame_color=None):
    diff_2d = recon_2d - target_2d
    for ax, data, label in zip(axs_row, [target_2d, recon_2d, diff_2d], ["target", "reconstruction", "error"]):
        vmin, vmax = column_ranges[label]
        im = ax.imshow(np.asarray(data).T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix}: {label}")
        ax.set_xlabel(axes_labels[0])
        ax.set_ylabel(axes_labels[1])
        fig.colorbar(im, ax=ax, fraction=0.046)
        for box, color in boxes or []:
            x0, y0, w, h = box
            ax.add_patch(Rectangle((x0, y0), w, h, edgecolor=color, facecolor="none", linewidth=2))
        if frame_color is not None:
            for spine in ax.spines.values():
                spine.set_edgecolor(frame_color)
                spine.set_linewidth(2)


def plot_combined(local_entries, global_target, global_recon, it, plane="xvx", output=None):
    """One figure: top row is the full-domain reconstruction assembled from every
    rank's network vs. the full-domain data, with one colored frame per requested
    rank marking where its subdomain sits within the full domain; each following
    row is one rank's own network vs. its local training data, framed in the same
    color as its box in the top row.

    local_entries: list of (rank, target, recon, box, color) tuples, one per
    requested rank, in the order their rows should appear.

    Each of the 3 columns (target, reconstruction, error) gets its own colorbar
    range, shared across every row so the global assembly and each rank's local
    network are directly comparable within that column -- but target,
    reconstruction, and error are not forced onto the same range as each other.
    """
    global_t2d, axes_labels = _slice_2d(global_target, plane=plane)
    global_r2d, _ = _slice_2d(global_recon, plane=plane)

    local_slices = [
        (rank, *_slice_2d(target, plane=plane)[:1], _slice_2d(recon, plane=plane)[0], box, color)
        for rank, target, recon, box, color in local_entries
    ]

    def _range(arrays):
        return (min(float(np.min(a)) for a in arrays), max(float(np.max(a)) for a in arrays))

    all_targets = [global_t2d] + [t2d for _, t2d, _, _, _ in local_slices]
    all_recons = [global_r2d] + [r2d for _, _, r2d, _, _ in local_slices]
    all_diffs = [global_r2d - global_t2d] + [r2d - t2d for _, t2d, r2d, _, _ in local_slices]
    column_ranges = {
        "target": _range(all_targets),
        "reconstruction": _range(all_recons),
        "error": _range(all_diffs),
    }

    n_rows = 1 + len(local_slices)
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axs = np.atleast_2d(axs)

    boxes = [(box, color) for _, _, _, box, color in local_entries if box is not None]
    _plot_row(
        fig, axs[0], global_t2d, global_r2d, axes_labels, f"global (iter={it}, all ranks)",
        column_ranges, boxes=boxes,
    )

    for row, (rank, t2d, r2d, _, color) in enumerate(local_slices, start=1):
        _plot_row(
            fig, axs[row], t2d, r2d, axes_labels, f"local (iter={it}, rank={rank})",
            column_ranges, frame_color=color,
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


def run_online_networks(data_dir, it, ranks=None, species=0, plane="xvx"):
    """Evaluate/plot saved online INR networks (params_iterXXXXX_rankXXX.npz):
    top row is the full-domain reconstruction assembled from every rank's network
    at that iteration vs. the full-domain data; each following row is one
    requested rank's own network vs. the local data it was trained on,
    color-matched to its box in the top row. `ranks=None` uses every rank found.
    """
    global_target, global_recon, rank_regions = evaluate_global_domain(data_dir, it, species=species)

    ranks = ranks if ranks is not None else sorted(rank_regions)

    local_entries = []
    for i, rank in enumerate(ranks):
        _, target, recon = evaluate_rank(data_dir, it, rank, species=species)

        color = _color_for_index(i)
        box = _rank_box_in_plane(*rank_regions[rank], plane) if rank in rank_regions else None
        local_entries.append((rank, target, recon, box, color))

    out = _online_networks_output_path(data_dir, it, ranks, species, plane)
    plot_combined(local_entries, global_target, global_recon, it=it, plane=plane, output=out)


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


# CLI


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


if __name__ == "__main__":
    main()
