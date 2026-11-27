# SPDX-License-Identifier: MIT

"""
Benchmark of the different ways to write the initial distribution function of
Gysela-X (see gyselax/initialisation/init_distribution_function.py) to disk.

fdistribu(species, tor1, tor2[, tor3], vpar, mu) is a Maxwellian

    n / (2 pi T)^1.5 * exp(-(0.5 (vpar - Upar)^2 + mu B) / T)

Each MPI rank owns a block of the leading (species, tor1, tor2, ...) axes and
streams it to disk in pieces of at most --piece-mb, so the full array is never
held in memory.
"""

import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np
from mpi4py import MPI

try:
    from .writers import WRITERS
except ImportError:
    from writers import WRITERS

try:
    import numexpr as ne
except ImportError:  # numpy fallback, slower and with temporaries
    ne = None

DIM_NAMES_5D = ["species", "tor1", "tor2", "vpar", "mu"]
DIM_NAMES_6D = ["species", "tor1", "tor2", "tor3", "vpar", "mu"]


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def make_coords(shape):
    names = DIM_NAMES_6D if len(shape) == 6 else DIM_NAMES_5D
    coords = {}
    for name, n in zip(names, shape):
        if name == "species":
            values = np.arange(n, dtype=np.float64)
        elif name == "tor1":
            values = np.linspace(0.0, 1.0, n)
        elif name in ("tor2", "tor3"):
            values = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        elif name == "vpar":
            values = np.linspace(-7.0, 7.0, n)
        else:  # mu
            values = np.linspace(0.0, 12.0, n)
        coords[name] = values
    return coords


def maxwellian_inputs(coords, start, count):
    """Broadcastable inputs of the Maxwellian on the sub-block [start, start+count)."""
    names = list(coords)
    ndim = len(names)
    sl = {name: coords[name][s : s + c] for name, s, c in zip(names, start, count)}

    def along(name):
        shape = [1] * ndim
        shape[names.index(name)] = -1
        return sl[name].reshape(shape)

    species = along("species").astype(np.int64)
    r = along("tor1")
    theta = along("tor2")
    phi = along("tor3") if "tor3" in names else 0.0

    upar0 = np.where(species == 0, 0.0, 3.0)
    temp0 = np.where(species == 0, 1.0, 1.2)
    n_equil = 1.0 - 0.5 * r**2
    n_fluct = 1.0e-3 * np.exp(-(((r - 0.5) / 0.1) ** 2)) * np.cos(5 * theta - 3 * phi)

    return {
        "n_np": n_equil * (1.0 + n_fluct) + 0.0 * species,
        "t_np": temp0 * np.exp(-0.3 * np.tanh((r - 0.5) / 0.2)),
        "upar_np": upar0 * (1.0 - r**2),
        "b_norm_np": 1.0 / (1.0 + 0.3 * r * np.cos(theta)),
        "vpar_np": along("vpar"),
        "mu_np": along("mu"),
    }


EXPR = "n_np / (2.0 * pi * t_np)**1.5 * exp(-(0.5 * (vpar_np - upar_np)**2 + mu_np * b_norm_np) / t_np)"


def compute_maxwellian(coords, start, count, out):
    """Fill out fdistribu in place."""
    args = maxwellian_inputs(coords, start, count)
    if ne is not None:
        ne.evaluate(EXPR, local_dict={**args, "pi": np.pi}, out=out, casting="unsafe")
    else:
        energy = 0.5 * (args["vpar_np"] - args["upar_np"]) ** 2 + args["mu_np"] * args["b_norm_np"]
        out[...] = args["n_np"] / (2.0 * np.pi * args["t_np"]) ** 1.5 * np.exp(-energy / args["t_np"])
    return out


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------
def split(n, p, i):
    """
    Start and count of block i when n points are split in p near-equal blocks.
    Decides which point one rank gets along an axis.

    For example:
        tor1 = 126 + 32 parts + block 0 -> split(126,32,0) = (0,4)
        tor1 = 126 + 32 parts + block 31 -> split(126,32,31) = (123,3)
    """
    base, extra = divmod(n, p)
    return i * base + min(i, extra), base + (1 if i < extra else 0)


def decompose(nprocs, shape):
    """
    Process grid over the leading axes (never vpar/mu), filled outermost first.
    Decides how many ranks go along each axis.
    
    For example:
        medium grid (2,126,128,64,64,64) + 64 ranks -> [2,32,1,1,1,1] 
    """
    grid = [1] * len(shape)
    rest = nprocs
    for d in range(len(shape) - 2):
        divisors = [p for p in range(1, min(rest, shape[d]) + 1) if rest % p == 0]
        grid[d] = max(divisors)
        rest //= grid[d]
    if rest != 1:
        raise ValueError(f"cannot decompose {nprocs} ranks over the leading axes of {shape}")
    return grid


def local_block(rank, grid, shape):
    """
    Returns which part of fdistribu the current rank owns.

    For example on the medium grid (2, 126, 128, 64, 64, 64) using the grid from
    the decomposition above [2, 32, 1, 1, 1, 1]:
        - unravel_index(37, [2, 32, 1, 1, 1, 1]) -> (1, 5, 0, 0, 0, 0)
        - call split once per axis:
            species : split(2, 2, 1) -> start = 1, count = 1
            tor1 : split(126, 32, 5) -> start = 20, count = 4
            tor2, tor3, vpar, mu -> start = 0, count = n (whole axis)
        - meaning rank 37 would own :
            starts = [1, 20, 0, 0, 0, 0]
            counts = [1, 4, 128, 64, 64, 64]
    """
    coords = np.unravel_index(rank, grid)
    starts, counts = zip(*(split(n, p, c) for n, p, c in zip(shape, grid, coords)))
    return list(starts), list(counts)


def plan_pieces(comm, local_count, itemsize, piece_bytes):
    """
    Number of pieces per axis, identical on every rank.
    Cuts the block into pieces of size --pieces-mb
    """
    max_bytes = comm.allreduce(int(np.prod(local_count, dtype=np.int64)) * itemsize, op=MPI.MAX)
    min_count = np.empty(len(local_count), dtype=np.int64)
    comm.Allreduce(np.asarray(local_count, dtype=np.int64), min_count, op=MPI.MIN)
    wanted = max(1, math.ceil(max_bytes / piece_bytes))
    npieces = [1] * len(local_count)
    for d in range(len(local_count) - 1):  # never split mu
        if wanted <= 1:
            break
        npieces[d] = int(min(wanted, min_count[d]))
        wanted = math.ceil(wanted / npieces[d])
    return npieces


def iter_pieces(local_start, local_count, npieces):
    per_axis = [[split(c, p, i) for i in range(p)] for c, p in zip(local_count, npieces)]
    for combo in np.ndindex(*npieces):
        start = [ls + per_axis[d][i][0] for d, (ls, i) in enumerate(zip(local_start, combo))]
        count = [per_axis[d][i][1] for d, i in enumerate(combo)]
        yield start, count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_hints(text):
    hints = {}
    for item in filter(None, (t.strip() for t in text.split(","))):
        key, _, value = item.partition("=")
        hints[key.strip()] = value.strip()
    return hints


def peak_rss_bytes():
    """Peak resident memory of this process so far."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # Linux reports KiB


def memory_report(comm, baseline):
    """
    Peak memory of the largest rank, of the most loaded node and of the whole
    job (bytes in, MiB/GiB out). Node and job values sum the per-rank peaks.
    """
    peak = peak_rss_bytes()
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_sum = node.allreduce(peak, op=MPI.SUM)
    node_rank = node.Get_rank()
    node.Free()
    return {
        "mem_baseline_rank_MB": comm.allreduce(baseline, op=MPI.MAX) / 2**20,
        "mem_peak_rank_MB": comm.allreduce(peak, op=MPI.MAX) / 2**20,
        "mem_peak_node_GB": comm.allreduce(node_sum if node_rank == 0 else 0, op=MPI.MAX) / 2**30,
        "mem_peak_total_GB": comm.allreduce(peak, op=MPI.SUM) / 2**30,
    }


def fsync_files(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def check_sizes(comm, writer, local_bytes, total_bytes):
    """
    Every fdistribu file must be at least as large as the payload it holds: the
    local block for a per-rank file, all of fdistribu for the shared file (HDF5
    Returns (size, allocated, bad files), summed over the job.
    `allocated` (st_blocks) also exposes holes, i.e. byte
    ranges that were never written, which st_size alone does not.
    """
    if writer.layout == "fpp":
        mine = [(path, local_bytes) for path in writer.files()]
    else:
        mine = [(writer.files()[0], total_bytes)] if comm.Get_rank() == 0 else []
    size = alloc = bad = 0
    for path, expected in mine:
        st = os.stat(path)
        size += st.st_size
        alloc += st.st_blocks * 512
        if st.st_size < expected:
            bad += 1
            print(f"### ERROR: {path} holds {st.st_size} bytes, expected at least {expected}", flush=True)
    return (comm.allreduce(size, op=MPI.SUM), comm.allreduce(alloc, op=MPI.SUM), comm.allreduce(bad, op=MPI.SUM))


def verify_points(pieces, local_start, local_count, n_random, max_pieces=64, seed=0):
    """
    Points to read back: the first and last element of every piece and `n_random` random points.
    """
    step = max(1, math.ceil(len(pieces) / max_pieces))
    points = []
    for start, count in pieces[::step]:
        points.append(list(start))
        points.append([s + c - 1 for s, c in zip(start, count)])
    rng = np.random.default_rng(seed)
    points += [[int(s + rng.integers(c)) for s, c in zip(local_start, local_count)] for _ in range(n_random)]
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=sorted(WRITERS), required=True)
    parser.add_argument("--layout", choices=["shared", "fpp"], required=True, help="one shared file, or one file per MPI rank")
    parser.add_argument(
        "--io-mode",
        choices=["collective", "independent", "posix", "serial"],
        default="collective",
        help="shared file only: MPI-IO transfer mode (posix: raw pwrite, raw backend only)",
    )
    parser.add_argument("--shape", required=True, help="species,tor1,tor2[,tor3],vpar,mu (number of points)")
    parser.add_argument("--out", required=True, help="output directory (stripe it beforehand on Lustre)")
    parser.add_argument("--piece-mb", type=float, default=512, help="max size of a piece computed+written at once, per rank")
    parser.add_argument("--alignment", type=int, default=0, help="HDF5 object alignment in bytes (use the Lustre stripe size)")
    parser.add_argument("--nc-format", default="NETCDF4", help="NETCDF4 (parallel HDF5) or NETCDF3_64BIT_DATA (PnetCDF)")
    parser.add_argument("--mpi-hints", default="", help="MPI-IO hints, e.g. romio_cb_write=enable,cb_nodes=8")
    parser.add_argument("--fsync", action=argparse.BooleanOptionalAction, default=True, help="time an fsync of the written files")
    parser.add_argument(
        "--verify",
        type=int,
        default=8,
        help="read back both ends of every piece plus this many random points per rank (0: no read-back; "
        "the on-disk size check always runs)",
    )
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank, nprocs = comm.Get_rank(), comm.Get_size()

    shape = [int(n) for n in args.shape.split(",")]
    if len(shape) not in (5, 6):
        parser.error("--shape needs 5 (2D toroidal) or 6 (3D toroidal) extents")
    if args.layout == "fpp":
        args.io_mode = "serial"
    elif args.io_mode == "serial":
        parser.error("--io-mode serial only makes sense with --layout fpp")
    if args.io_mode == "posix" and args.backend != "raw":
        parser.error("--io-mode posix is only available for the raw backend")
    if ne is not None:
        ne.set_num_threads(int(os.environ.get("NUMEXPR_NUM_THREADS", "1")))

    coords = make_coords(shape)
    grid = decompose(nprocs, shape)
    local_start, local_count = local_block(rank, grid, shape)
    itemsize = np.dtype(np.float64).itemsize
    npieces = plan_pieces(comm, local_count, itemsize, int(args.piece_mb * 2**20))
    pieces = list(iter_pieces(local_start, local_count, npieces))
    max_piece = max(int(np.prod(c)) for _, c in pieces)
    mem_baseline = peak_rss_bytes()  # interpreter + libraries, before the piece buffer
    buf = np.empty(max_piece, dtype=np.float64)  # reused for every piece

    total_bytes = int(np.prod(shape, dtype=np.int64)) * itemsize
    if rank == 0:
        print(f"backend={args.backend} layout={args.layout} io_mode={args.io_mode} nprocs={nprocs}", flush=True)
        print(f"shape={shape} ({total_bytes / 1e9:.3f} GB) process grid={grid}", flush=True)
        print(f"pieces per rank={len(pieces)} grid={npieces} max piece={max_piece * itemsize / 2**20:.1f} MiB", flush=True)
        print(f"numexpr={'yes' if ne is not None else 'no (numpy fallback)'}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    opts = {
        "layout": args.layout,
        "io_mode": args.io_mode,
        "alignment": args.alignment,
        "nc_format": args.nc_format,
        "mpi_hints": parse_hints(args.mpi_hints),
    }
    writer = WRITERS[args.backend](comm, args.out, shape, local_start, local_count, coords, opts)

    t = {"open": 0.0, "compute": 0.0, "write": 0.0, "close": 0.0, "sync": 0.0}
    comm.Barrier()
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    writer.open()
    t["open"] = time.perf_counter() - t0

    for start, count in pieces:
        block = buf[: int(np.prod(count))].reshape(count)
        t0 = time.perf_counter()
        compute_maxwellian(coords, start, count, block)
        t1 = time.perf_counter()
        writer.write(start, block)
        t2 = time.perf_counter()
        t["compute"] += t1 - t0
        t["write"] += t2 - t1

    t0 = time.perf_counter()
    writer.close()
    t["close"] = time.perf_counter() - t0
    comm.Barrier()
    t_written = time.perf_counter()

    if args.fsync:
        t0 = time.perf_counter()
        fsync_files(writer.files())
        t["sync"] = time.perf_counter() - t0
        comm.Barrier()
    t_synced = time.perf_counter()
    mem = memory_report(comm, mem_baseline)

    # checks
    # 1. size on disk, always: a run that wrote less than the payload fails.
    local_bytes = int(np.prod(local_count, dtype=np.int64)) * itemsize
    disk_bytes, disk_alloc, size_errors = check_sizes(comm, writer, local_bytes, total_bytes)
    if rank == 0 and disk_alloc < 0.99 * total_bytes:
        print(f"### WARNING: {disk_alloc} bytes allocated on disk for a {total_bytes}-byte payload (holes?)", flush=True)

    # 2. read-back with the same library: both ends of every piece + random points.
    errors = n_checked = 0
    if args.verify > 0:
        path = writer.files()[0]
        for idx in verify_points(pieces, local_start, local_count, args.verify, seed=1234 + rank):
            expected = compute_maxwellian(coords, idx, [1] * len(idx), np.empty([1] * len(idx)))[(0,) * len(idx)]
            n_checked += 1
            try:
                got = writer.read_point(path, writer.file_start(idx))
            except Exception as exc:  # pylint: disable=broad-except
                # A truncated or corrupt file: count it, keep every rank in step.
                errors += 1
                print(f"[rank {rank}] verify read failed at {idx}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if not np.isclose(got, expected, rtol=1e-12, atol=0.0):
                errors += 1
                print(f"[rank {rank}] verify mismatch at {idx}: {got} != {expected}", flush=True)
        errors = comm.allreduce(errors, op=MPI.SUM)
        n_checked = comm.allreduce(n_checked, op=MPI.SUM)

    # results
    tmax = {k: comm.allreduce(v, op=MPI.MAX) for k, v in t.items()}
    tmin_write = comm.allreduce(t["write"], op=MPI.MIN)
    time_total = t_written - t_start
    time_total_synced = t_synced - t_start
    time_write = comm.allreduce(t["open"] + t["write"] + t["close"], op=MPI.MAX)
    piece_mb = comm.allreduce(max_piece * itemsize, op=MPI.MAX) / 2**20
    if rank == 0:
        gb = total_bytes / 1e9
        result = {
            "data_GB": gb,
            "nprocs": nprocs,
            "time_total": time_total,
            "time_total_synced": time_total_synced,
            "time_write": time_write,
            "time_open": tmax["open"],
            "time_compute": tmax["compute"],
            "time_write_calls": tmax["write"],
            "time_write_calls_min": tmin_write,
            "time_close": tmax["close"],
            "time_sync": tmax["sync"],
            "bw_write_GBps": gb / time_write,
            "bw_total_GBps": gb / time_total,
            "bw_synced_GBps": gb / time_total_synced,
            "piece_MB": piece_mb,
            **mem,
            "disk_GB": disk_bytes / 1e9,
            "disk_alloc_GB": disk_alloc / 1e9,
            "size_errors": size_errors,
            "verify_points": n_checked,
            "verify_errors": errors,
        }
        for key, value in result.items():
            print(f"{key:>22}: {value:.4f}" if isinstance(value, float) else f"{key:>22}: {value}", flush=True)
        print("INIT_BENCH_RESULT " + json.dumps(result), flush=True)
        if size_errors or errors:
            print(f"### FAILED: {size_errors} file(s) smaller than their payload, {errors} wrong value(s) read back", flush=True)
    # Non-zero exit makes srun, and so the IOPS test, fail.
    return 1 if size_errors or errors else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # pylint: disable=broad-except
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
