"""Reduced diagnostics: rank-local partial sums finished either by a deisa callback or post hoc."""

from pathlib import Path
import csv

import numpy as np

PARTIAL_FIELDS = ["mass", "ekin", "l2norm_sq", "momentum_x", "momentum_y"]
N_PARTIALS = len(PARTIAL_FIELDS)


def local_reduce(f_local, starts, mesh_vx, mesh_vy, nsp_global):
    """Reduce the local f chunk over (x, y, vx, vy) -> partial sums of shape (Nsp, 5)."""
    f = np.asarray(f_local)
    s0, svx, svy = int(starts[0]), int(starts[3]), int(starts[4])
    vx = np.asarray(mesh_vx)[svx: svx + f.shape[3]][None, None, None, :, None]
    vy = np.asarray(mesh_vy)[svy: svy + f.shape[4]][None, None, None, None, :]
    axes = (1, 2, 3, 4)

    partial = np.zeros((int(nsp_global), N_PARTIALS))
    sp = slice(s0, s0 + f.shape[0])
    partial[sp, 0] = f.sum(axis=axes)
    partial[sp, 1] = 0.5 * (f * (vx ** 2 + vy ** 2)).sum(axis=axes)
    partial[sp, 2] = (f ** 2).sum(axis=axes)
    partial[sp, 3] = (f * vx).sum(axis=axes)
    partial[sp, 4] = (f * vy).sum(axis=axes)
    return partial


def append_partial_csv(partial, it, t_actual, rank, data_dir="."):
    """Append the rank-local partial sums to partial_diagnostics_rank<rank>.csv.

    Called synchronously and in iteration order from within a single MPI rank's
    pycall event, so (unlike write_diagnostics_rows below) there's no risk of an
    out-of-order write here -- a plain it==0 reset is safe.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"partial_diagnostics_rank{rank:03d}.csv"

    reset = it == 0
    write_header = reset or not path.is_file()
    with open(path, mode="w" if reset else "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["iter", "time", "species"] + PARTIAL_FIELDS)
        for isp in range(partial.shape[0]):
            writer.writerow([it, float(t_actual), isp] + [float(v) for v in partial[isp]])


def finalize_row(partial_sp, dV_4D, it, t_actual):
    """Scale summed partials by dV_4D into a diagnostics.csv row.

    epot/etot need the field solve, which isn't possible on scattered data -> nan.
    """
    mass, ekin, l2norm_sq, momentum_x, momentum_y = (float(v) * dV_4D for v in partial_sp)
    epot = float("nan")
    return {
        "iter":       it,
        "time":       float(t_actual),
        "ekin":       ekin,
        "epot":       epot,
        "etot":       ekin + epot,
        "l2norm":     float(np.sqrt(l2norm_sq)),
        "mass":       mass,
        "momentum":   float(np.hypot(momentum_x, momentum_y)),
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }


def write_diagnostics_rows(partials, dV_4D, it, t_actual, initialized_paths, data_dir="."):
    """Append one diagnostics.csv row per species, laid out as in diagnostics.measure.

    initialized_paths is a set the caller owns and passes on every call (e.g.
    diagnostics.py's _INITIALIZED_DIAG_FILES, or a fresh set kept alive across
    a post-hoc loop) - a path resets (mode 'w') only the first time it's seen.
    """
    nsp = partials.shape[0]
    for isp in range(nsp):
        sp_dir = Path(data_dir) if nsp == 1 else Path(data_dir) / f"species_{isp}"
        sp_dir.mkdir(parents=True, exist_ok=True)
        path = sp_dir / "diagnostics.csv"

        data = finalize_row(partials[isp], dV_4D, it, t_actual)
        reset = path not in initialized_paths
        initialized_paths.add(path)
        write_header = reset or not path.is_file()
        with open(path, mode="w" if reset else "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(data)
