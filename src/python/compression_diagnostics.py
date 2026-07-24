"""Compression logic shared by the rank-local online path and the bridge-based offline path:
builds compressors, runs compress/decompress round trips, and logs results to CSV."""

import os
import csv
import json
import h5py
import time
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


def apply_online_compression(fdistribu, timestep, rank):
    """Rank-local compress/decompress round trip. Mutates fdistribu in place.
    """
    cfg = get_compression_config()
    approx, metrics = cfg.compressor.compress_decompress_array(fdistribu, rank=rank)
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

_LAST_EVENT_WALLTIME = {}

def run_offline_compression_on_global_array(fdistribu_global, timestep, data_dir="."):
    """Compress the globally assembled array and write the
    reconstruction to HDF5.
    """
    cfg = get_offline_compression_config(data_dir)

    t_now = time.perf_counter()
    prev_t = _LAST_EVENT_WALLTIME.get(str(cfg.data_dir))
    _LAST_EVENT_WALLTIME[str(cfg.data_dir)] = t_now

    _coefficients, reconstructed = cfg.compressor.compress_decompress_array(fdistribu_global)
    
    metrics = cfg.compressor.compute_metrics(
        f_original=fdistribu_global,
        f_reconstructed=reconstructed,
    )
    
    # Wall-clock approximation of pure simulation time (no exact PDI measurement at this stage)
    if prev_t is not None:
        delta_wall = t_now - prev_t
        comp_t = metrics.get("compression_seconds") or 0.0 
        decomp_t = metrics.get("decompression_seconds") or 0.0
        metrics["sim_time_approx"] = max(0.0, delta_wall - comp_t - decomp_t)
    else:
        metrics["sim_time_approx"] = None

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    out_path = _offline_compressed_restart_path(cfg.data_dir, timestep)
    #save the compressed payload for warm-starting
    payload_path = cfg.data_dir / f"payload_iter{timestep:05d}.npz"
    cfg.compressor.save_compressed_payload(str(payload_path), _coefficients)
    
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("fdistribu", data=reconstructed)

    if hasattr(cfg.compressor, "save_svd_spectrum"):
        cfg.compressor.save_svd_spectrum(cfg.data_dir / "svd_spectrums", timestep)
    if hasattr(cfg.compressor, "save_loss_histories"):
        cfg.compressor.save_loss_histories(cfg.data_dir / "loss_histories", timestep)
    
    event_path = cfg.data_dir / "compression_events_offline.csv"
    record = {"iter": timestep}
    record.update({k: v for k, v in metrics.items() if k != "params"})
    for key, value in (metrics.get("params") or {}).items():
        record[f"param_{key}"] = value
    _write_compression_event_csv(event_path, record)

    return out_path
