"""In-situ online compression: Mutates fdistribu in place.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from compression_config import build_online_compressor


@dataclass
class CompressionConfig:
    data_dir: Path
    compressor: object


_COMPRESSION_CFG = None


def get_compression_config(data_dir="."):
    global _COMPRESSION_CFG
    if _COMPRESSION_CFG is None:
        _COMPRESSION_CFG = CompressionConfig(Path(data_dir), build_online_compressor())
    return _COMPRESSION_CFG


def apply_online_compression(fdistribu, timestep, rank):
    """Rank-local compress/decompress round trip. Mutates fdistribu in place.
    """
    cfg = get_compression_config()
    approx, metrics = cfg.compressor.compress_decompress_array(fdistribu)
    fdistribu[...] = approx

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    event_path = cfg.data_dir / f"compression_events_rank{rank:03d}.csv"

    record = {"iter": timestep, "rank": rank}
    record.update({k: v for k, v in metrics.items() if k != "params"})
    for key, value in (metrics.get("params") or {}).items():
        record[f"param_{key}"] = value

    write_header = not event_path.is_file()
    with open(event_path, mode="a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return fdistribu
