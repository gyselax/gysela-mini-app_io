"""Compression logic shared by the rank-local online path and the bridge-based offline path:
builds compressors, runs compress/decompress round trips, and logs results to CSV."""

import os
import csv
import json
import dask.array as da
import h5py
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from compression_config import build_online_compressor, build_offline_compressor


@dataclass
class CompressionConfig:
    data_dir: Path
    compressor: object


_COMPRESSION_CFG = None
_OFFLINE_COMPRESSION_CFG = None


def get_compression_config(data_dir=".", rank=None):
    global _COMPRESSION_CFG
    if _COMPRESSION_CFG is None:
        _COMPRESSION_CFG = CompressionConfig(Path(data_dir), build_online_compressor(rank=rank))
    return _COMPRESSION_CFG


def get_offline_compression_config(data_dir="."):
    global _OFFLINE_COMPRESSION_CFG
    if _OFFLINE_COMPRESSION_CFG is None:
        mesh_kwargs = json.loads(os.environ.get("COMPRESSION_MESH_KWARGS", "{}"))
        _OFFLINE_COMPRESSION_CFG = CompressionConfig(Path(data_dir), build_offline_compressor(**mesh_kwargs))
    return _OFFLINE_COMPRESSION_CFG


def _write_compression_event_csv(event_path, record):
    write_header = not event_path.is_file()
    with open(event_path, mode="a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def apply_online_compression(fdistribu, timestep, rank, local_bounds=None):
    """Rank-local compress/decompress round trip. Mutates fdistribu in place.

    local_bounds, if given, is the local chunk's physical bounding box
    (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max) within the
    global mesh; compressors that fit a local model (e.g.
    OnlineNeuralNetworkCompressor) record it so a downstream tool can later
    reassemble a global field from the per-rank local models.
    """
    cfg = get_compression_config(rank=rank)
    approx, metrics = cfg.compressor.compress_decompress_array(fdistribu, rank=rank, local_bounds=local_bounds)

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(cfg.compressor, "save_params"):
        params_path = cfg.data_dir / f"params_iter{timestep:05d}_rank{rank:03d}.npz"
        cfg.compressor.save_params(params_path, rank=rank, timestep=timestep)

    event_path = cfg.data_dir / f"compression_events_rank{rank:03d}.csv"

    record = {"iter": timestep, "rank": rank}
    record.update({k: v for k, v in metrics.items() if k != "params"})
    for key, value in (metrics.get("params") or {}).items():
        record[f"param_{key}"] = value

    _write_compression_event_csv(event_path, record)
    
    fdistribu[...] = approx
    return fdistribu


def _offline_compressed_restart_path(data_dir, timestep):
    return Path(data_dir) / f"GYSELALIBXX_compressed_{timestep:05d}.h5"


def _write_reconstruction_h5(out_path, reconstructed, dataset_name="fdistribu"):
    """A dask reconstruction is fetched and written one block at a time, so the
    global field is never resident in this process.
    """
    if not isinstance(reconstructed, da.Array):
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset(dataset_name, data=reconstructed)
        return out_path

    offsets = [np.cumsum((0,) + chunks) for chunks in reconstructed.chunks]

    with h5py.File(out_path, "w") as h5:
        dataset = h5.create_dataset(dataset_name, shape=reconstructed.shape, dtype=reconstructed.dtype)
        for index in np.ndindex(*reconstructed.numblocks):
            selection = tuple(slice(offset[i], offset[i + 1]) for offset, i in zip(offsets, index))
            dataset[selection] = reconstructed.blocks[index].compute()

    return out_path


def run_offline_compression_on_global_array(fdistribu_global, timestep, data_dir="."):
    """Compress the globally assembled array and write the
    reconstruction to HDF5.
    """
    cfg = get_offline_compression_config(data_dir)

    if not cfg.compressor.accepts_dask:
        fdistribu_global = np.asarray(fdistribu_global)

    cfg.compressor.compress_decompress_array(fdistribu_global)
