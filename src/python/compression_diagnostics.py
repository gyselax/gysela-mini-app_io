"""Compression logic shared by the rank-local online path and the bridge-based offline path:
builds compressors, runs compress/decompress round trips, and logs results to CSV."""

import os
import csv
import json
import h5py
from dataclasses import dataclass
from pathlib import Path

from compression_config import build_online_compressor, build_offline_compressor


@dataclass
class CompressionConfig:
    data_dir: Path
    compressor: object


_COMPRESSION_CFG = None
_OFFLINE_COMPRESSION_CFG = None


def get_compression_config(data_dir="."):
    global _COMPRESSION_CFG
    if _COMPRESSION_CFG is None:
        _COMPRESSION_CFG = CompressionConfig(Path(data_dir), build_online_compressor())
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
    cfg = get_compression_config()
    approx, metrics = cfg.compressor.compress_decompress_array(fdistribu, rank=rank, local_bounds=local_bounds)
    fdistribu[...] = approx

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    event_path = cfg.data_dir / f"compression_events_rank{rank:03d}.csv"

    record = {"iter": timestep, "rank": rank}
    record.update({k: v for k, v in metrics.items() if k != "params"})
    for key, value in (metrics.get("params") or {}).items():
        record[f"param_{key}"] = value

    _write_compression_event_csv(event_path, record)
    return fdistribu


def _offline_compressed_restart_path(data_dir, timestep):
    return Path(data_dir) / f"GYSELALIBXX_compressed_{timestep:05d}.h5"


def run_offline_compression_on_global_array(fdistribu_global, timestep, data_dir="."):
    """Compress the globally assembled array and write the
    reconstruction to HDF5.
    """
    cfg = get_offline_compression_config(data_dir)

    _coefficients, reconstructed = cfg.compressor.compress_decompress_array(fdistribu_global)
    metrics = cfg.compressor.compute_metrics(
        f_original=fdistribu_global,
        f_reconstructed=reconstructed,
    )

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = _offline_compressed_restart_path(cfg.data_dir, timestep)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("fdistribu", data=reconstructed)

    event_path = cfg.data_dir / "compression_events_offline.csv"
    record = {"iter": timestep}
    record.update({k: v for k, v in metrics.items() if k != "params"})
    for key, value in (metrics.get("params") or {}).items():
        record[f"param_{key}"] = value
    _write_compression_event_csv(event_path, record)

    return out_path
