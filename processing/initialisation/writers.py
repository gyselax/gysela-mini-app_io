# SPDX-License-Identifier: MIT

"""
Writers used by the fdistribu initialisation benchmark.

Every writer follows the same life cycle, called by *all* MPI ranks:

    w = Writer(...)
    w.open()                       # create the file(s), define the variable
    for start, block in pieces:    # same number of pieces on every rank
        w.write(start, block)      # one (possibly collective) write per piece
    w.close()
    w.files()

`start` is the global index of `block[0, 0, ...]` in fdistribu.  Writers for the
file-per-process layout convert it to an index inside the local block.

Backends x layouts:

    hdf5    shared : parallel HDF5 (mpio driver), collective or independent
            fpp    : one serial HDF5 file per rank (sec2 driver)
    netcdf  shared : parallel netCDF-4 (HDF5 based) or CDF5 through PnetCDF
            fpp    : one serial netCDF-4 file per rank
    raw     shared : MPI-IO (collective / independent) or plain pwrite
            fpp    : one file per rank, plain pwrite
"""

import json
import os
from pathlib import Path

import numpy as np
from mpi4py import MPI

FDISTRIBU = "fdistribu"


def mpi_info(hints):
    """Build an MPI.Info from a {key: value} dict (MPI.INFO_NULL when empty)."""
    if not hints:
        return MPI.INFO_NULL
    info = MPI.Info.Create()
    for key, value in hints.items():
        info.Set(key, str(value))
    return info


def contiguous_runs(shape, start, count):
    """
    Yield (element_offset, outer_index, run_length) for every contiguous run
    that a C-ordered sub-block `count` at `start` occupies in an array `shape`.

    All axes after the last partially-selected axis are full, so each run spans
    count[k] * prod(shape[k+1:]) elements.

    For example in the medium shared file (2, 126, 128, 64, 64, 64), one tor1
    plane = 128 * 64**3 = 33,554,432 elements:
        piece start = [1, 20, 0, 0, 0, 0], count = [1, 2, 128, 64, 64, 64]:
            last partial axis k = 1 (tor1), run = 2 planes = 67,108,864 elements
            -> (4,898,947,072, (0,), 67,108,864) : one 512 MiB pwrite
               (146 planes in: 126 of species 0 + 20)
        piece start = [0, 0, 64, 0, 0, 0], count = [1, 2, 64, 64, 64, 64]:
            last partial axis k = 2 (tor2), run = 64 * 64**3 = 16,777,216 elements
            -> (16,777,216, (0, 0), 16,777,216)
            -> (50,331,648, (0, 1), 16,777,216) : two 128 MiB pwrites, one per tor1 plane
    """
    ndim = len(shape)
    k = 0
    for d in range(ndim):
        if count[d] != shape[d]:
            k = d
    inner = int(np.prod(shape[k + 1 :], dtype=np.int64))
    run = count[k] * inner
    strides = [int(np.prod(shape[d + 1 :], dtype=np.int64)) for d in range(ndim)]
    for outer in np.ndindex(*count[:k]):
        offset = sum((start[d] + outer[d]) * strides[d] for d in range(k)) + start[k] * strides[k]
        yield offset, outer, run


def pwrite_all(fd, buf, offset):
    """os.pwrite until the whole buffer is written."""
    view = memoryview(buf).cast("B")
    done = 0
    while done < len(view):
        done += os.pwrite(fd, view[done:], offset + done)


class Writer:
    """Common state shared by all writers."""

    suffix = ""

    def __init__(self, comm, out_dir, global_shape, local_start, local_count, coords, opts):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.out_dir = Path(out_dir)
        self.global_shape = tuple(int(n) for n in global_shape)
        self.local_start = tuple(int(s) for s in local_start)
        self.local_count = tuple(int(c) for c in local_count)
        self.coords = coords  # {dim_name: 1D np.ndarray}, ordered like the shape
        self.opts = opts
        self.layout = opts["layout"]
        self.io_mode = opts["io_mode"]

    # -- helpers ------------------------------------------------------------
    @property
    def path(self):
        if self.layout == "shared":
            return self.out_dir / f"{FDISTRIBU}{self.suffix}"
        return self.out_dir / f"{FDISTRIBU}_{self.rank:06d}{self.suffix}"

    @property
    def file_shape(self):
        return self.global_shape if self.layout == "shared" else self.local_count

    def file_start(self, start):
        if self.layout == "shared":
            return tuple(start)
        return tuple(s - l for s, l in zip(start, self.local_start))

    def files(self):
        return [self.path]

    # -- interface ----------------------------------------------------------
    def open(self):
        raise NotImplementedError

    def write(self, start, block):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def read_point(self, path, index):
        """Read one value at `index` (file coordinates) for verification."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HDF5
# --------------------------------------------------------------------------- #
class HDF5Writer(Writer):
    suffix = ".h5"

    def open(self):
        import h5py
        from h5py import h5d, h5f, h5fd, h5p, h5s, h5t

        self._h5py = h5py
        fapl = h5p.create(h5p.FILE_ACCESS)
        if self.layout == "shared":
            if not h5py.get_config().mpi:
                raise RuntimeError("hdf5/shared needs h5py built against parallel HDF5 (h5py.get_config().mpi is False)")
            fapl.set_fapl_mpio(self.comm, mpi_info(self.opts["mpi_hints"]))
            # Collective metadata
            for name in ("set_all_coll_metadata_ops", "set_coll_metadata_write"):
                if hasattr(fapl, name):
                    getattr(fapl, name)(True)
        if self.opts["alignment"] > 0:
            # Align large objects (the fdistribu dataset) on the stripe boundaries.
            fapl.set_alignment(1024 * 1024, self.opts["alignment"])
        self.fid = h5f.create(str(self.path).encode(), h5f.ACC_TRUNC, fapl=fapl)

        dcpl = h5p.create(h5p.DATASET_CREATE)
        dcpl.set_alloc_time(h5d.ALLOC_TIME_EARLY)
        dcpl.set_fill_time(h5d.FILL_TIME_NEVER)  # otherwise the whole dataset is written twice
        self.dset = h5d.create(
            self.fid, FDISTRIBU.encode(), h5t.NATIVE_DOUBLE, h5s.create_simple(self.file_shape), dcpl=dcpl
        )

        self.dxpl = h5p.create(h5p.DATASET_XFER)
        indep_dxpl = h5p.create(h5p.DATASET_XFER)
        if self.layout == "shared":
            mode = h5fd.MPIO_COLLECTIVE if self.io_mode == "collective" else h5fd.MPIO_INDEPENDENT
            self.dxpl.set_dxpl_mpio(mode)
            indep_dxpl.set_dxpl_mpio(h5fd.MPIO_INDEPENDENT)

        # Small coordinate datasets: created by all ranks (collective metadata),
        # written by a single rank.
        coord_writer = self.layout == "fpp" or self.rank == 0
        for i, (name, values) in enumerate(self.coords.items()):
            if self.layout == "fpp":
                values = values[self.local_start[i] : self.local_start[i] + self.local_count[i]]
            values = np.ascontiguousarray(values, dtype=np.float64)
            cid = h5d.create(self.fid, name.encode(), h5t.NATIVE_DOUBLE, h5s.create_simple(values.shape))
            if coord_writer:
                cid.write(h5s.create_simple(values.shape), cid.get_space(), values, dxpl=indep_dxpl)
            cid.close()

    def write(self, start, block):
        from h5py import h5s

        fspace = self.dset.get_space()
        fspace.select_hyperslab(self.file_start(start), block.shape)
        self.dset.write(h5s.create_simple(block.shape), fspace, block, dxpl=self.dxpl)

    def close(self):
        f = self._h5py.File(self.fid)
        f.attrs["global_shape"] = np.asarray(self.global_shape, dtype=np.int64)
        f.attrs["local_start"] = np.asarray(self.local_start if self.layout == "fpp" else [0] * len(self.global_shape))
        f.attrs["dims"] = list(self.coords)
        self.dset.close()
        f.close()

    def read_point(self, path, index):
        import h5py

        with h5py.File(path, "r") as f:
            return float(f[FDISTRIBU][index])


# --------------------------------------------------------------------------- #
# netCDF (netCDF4-python)
# --------------------------------------------------------------------------- #
class NetCDFWriter(Writer):
    suffix = ".nc"

    def open(self):
        import netCDF4

        fmt = self.opts["nc_format"]
        if self.layout == "shared":
            if fmt.startswith("NETCDF4") and not netCDF4.__has_parallel4_support__:
                raise RuntimeError("netcdf/shared with NETCDF4 needs netcdf-c built with parallel HDF5 (+mpi)")
            if fmt.startswith("NETCDF3") and not netCDF4.__has_pnetcdf_support__:
                raise RuntimeError(f"netcdf/shared with {fmt} needs netcdf-c built with PnetCDF (+parallel-netcdf)")
            self.ds = netCDF4.Dataset(
                str(self.path),
                "w",
                format=fmt,
                parallel=True,
                comm=self.comm,
                info=mpi_info(self.opts["mpi_hints"]),
            )
        else:
            self.ds = netCDF4.Dataset(str(self.path), "w", format=fmt)
        self.ds.set_fill_off()

        dims = list(self.coords)
        for name, n in zip(dims, self.file_shape):
            self.ds.createDimension(name, n)
        extra = {"contiguous": True} if fmt.startswith("NETCDF4") else {}
        coord_vars = {name: self.ds.createVariable(name, "f8", (name,), **extra) for name in dims}
        self.var = self.ds.createVariable(FDISTRIBU, "f8", tuple(dims), fill_value=False, **extra)
        # Plain ndarray in, no masked-array / scale handling on the write path.
        self.var.set_auto_maskandscale(False)
        self.var.global_shape = np.asarray(self.global_shape, dtype=np.int64)
        self.var.local_start = np.asarray(
            self.local_start if self.layout == "fpp" else [0] * len(self.global_shape), dtype=np.int64
        )

        # Variables default to independent access: rank 0 writes the coordinates.
        for i, name in enumerate(dims):
            values = self.coords[name]
            if self.layout == "fpp":
                values = values[self.local_start[i] : self.local_start[i] + self.local_count[i]]
            if self.layout == "fpp" or self.rank == 0:
                coord_vars[name][:] = values
        if self.layout == "shared" and self.io_mode == "collective":
            self.var.set_collective(True)

    def write(self, start, block):
        fs = self.file_start(start)
        self.var[tuple(slice(s, s + n) for s, n in zip(fs, block.shape))] = block

    def close(self):
        self.ds.close()

    def read_point(self, path, index):
        import netCDF4

        with netCDF4.Dataset(str(path), "r") as ds:
            var = ds[FDISTRIBU]
            var.set_auto_maskandscale(False)
            return float(var[index])


# --------------------------------------------------------------------------- #
# Custom: C-ordered float64, plus a JSON file describing the layout
# --------------------------------------------------------------------------- #
class RawWriter(Writer):
    suffix = ".bin"

    def open(self):
        if self.layout == "shared" and self.io_mode in ("collective", "independent"):
            info = mpi_info(self.opts["mpi_hints"])
            if self.rank == 0 and self.path.exists():
                MPI.File.Delete(str(self.path), info)
            self.comm.Barrier()
            self.fh = MPI.File.Open(self.comm, str(self.path), MPI.MODE_CREATE | MPI.MODE_WRONLY, info)
            self.fd = None
        else:
            # CUSTOM : shared file truncated once by rank 0, or one file per rank.
            if self.layout == "shared":
                if self.rank == 0:
                    os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644))
                self.comm.Barrier()
                self.fd = os.open(self.path, os.O_WRONLY)
            else:
                self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            self.fh = None

    def write(self, start, block):
        fs = self.file_start(start)
        if self.fh is not None:
            # Subarray file view on the piece: one MPI-IO call per piece
            ftype = MPI.DOUBLE.Create_subarray(self.file_shape, block.shape, fs, order=MPI.ORDER_C).Commit()
            self.fh.Set_view(0, MPI.DOUBLE, ftype, "native")
            if self.io_mode == "collective":
                self.fh.Write_all(block)
            else:
                self.fh.Write(block)
            ftype.Free()
        else:
            flat = block.reshape(-1)
            done = 0
            for offset, _, run in contiguous_runs(self.file_shape, fs, block.shape):
                pwrite_all(self.fd, flat[done : done + run], offset * block.itemsize)
                done += run

    def close(self):
        if self.fh is not None:
            self.fh.Close()
        else:
            os.close(self.fd)
        starts = self.comm.gather(self.local_start, root=0)
        counts = self.comm.gather(self.local_count, root=0)
        if self.rank == 0:
            meta = {
                "dtype": np.dtype(np.float64).str,
                "order": "C",
                "dims": list(self.coords),
                "global_shape": list(self.global_shape),
                "layout": self.layout,
                "files": [str(self.path.name)]
                if self.layout == "shared"
                else [f"{FDISTRIBU}_{r:06d}{self.suffix}" for r in range(self.comm.Get_size())],
                "local_starts": [list(s) for s in starts],
                "local_counts": [list(c) for c in counts],
                "coords": {k: v.tolist() for k, v in self.coords.items()},
            }
            with open(self.out_dir / f"{FDISTRIBU}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)

    def read_point(self, path, index):
        mm = np.memmap(path, dtype=np.float64, mode="r", shape=self.file_shape)
        return float(mm[index])


WRITERS = {"hdf5": HDF5Writer, "netcdf": NetCDFWriter, "raw": RawWriter}
