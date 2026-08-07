#!/usr/bin/env python3
"""Post hoc finalisation of the per-rank partial diagnostics written with GYS_DIAG_MODE=2."""

import time
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

import reduced_diagnostics


def read_dV_4D(work_dir):
    """Read the mesh from GYSELALIBXX_initstate.h5 and return dx * dy * dvx * dvy."""
    with h5py.File(Path(work_dir) / "GYSELALIBXX_initstate.h5", "r") as fh5:
        return float(np.prod([fh5[m][1] - fh5[m][0] for m in ("MeshX", "MeshY", "MeshVx", "MeshVy")]))


def main():
    parser = argparse.ArgumentParser(description="Finish the reduced diagnostics from per-rank partial csv files.")
    parser.add_argument("work_dir", help="Branch directory containing partial_diagnostics_rank*.csv")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    partial_files = sorted(work_dir.glob("partial_diagnostics_rank*.csv"))
    if not partial_files:
        raise RuntimeError(f"No partial_diagnostics_rank*.csv found in {work_dir}")

    # summed[(iter, time)][species] -> partial sums accumulated over ranks
    summed = defaultdict(lambda: defaultdict(lambda: np.zeros(reduced_diagnostics.N_PARTIALS)))
    for path in partial_files:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                key = (int(row["iter"]), float(row["time"]))
                summed[key][int(row["species"])] += np.array(
                    [float(row[name]) for name in reduced_diagnostics.PARTIAL_FIELDS]
                )

    dV_4D = read_dV_4D(work_dir)
    initialized_paths = set()
    for it, t_actual in sorted(summed):
        by_species = summed[(it, t_actual)]
        partials = np.stack([by_species[isp] for isp in sorted(by_species)])
        reduced_diagnostics.write_diagnostics_rows(partials, dV_4D, it, t_actual, initialized_paths, work_dir)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Time post-hoc diagnostics:", time.time() - t0, flush=True)

