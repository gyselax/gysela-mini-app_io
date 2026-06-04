"""
Generic compressor interface for compression benchmarks.

New compression methods should subclass ``Compressor`` and implement only the
method-specific parts:

    - compress_array
    - decompress_array
    - optionally save_compressed_payload / load payload helpers
    - optionally get_extra_metrics
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional

import h5py
import numpy as np


class Compressor:
    """
    Base placeholder/interface for benchmark compressors.

    Subclasses are expected to implement ``compress_array`` and
    ``decompress_array``.  The base class handles:

    - method naming;
    - parameter naming/value serialisation;
    - HDF5 copy/replace workflow;
    - wall-clock timings;
    - common reconstruction and size metrics;
    """

    method_name = "placeholder"
    payload_extension = ".bin"

    def __init__(self, method_name: Optional[str] = None, **params: Any) -> None:
        self.method_name = method_name or self.method_name or self.__class__.__name__
        self.params: "OrderedDict[str, Any]" = OrderedDict(params)
        self._last_compression_seconds: Optional[float] = None
        self._last_decompression_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def param_names(self):
        return list(self.params.keys())

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [Compressor._json_safe(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): Compressor._json_safe(val) for key, val in value.items()}
        return value

    def describe(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "param_names": self.param_names,
            "params": self._json_safe(self.params),
        }

    def printable_name(self) -> str:
        if not self.params:
            return self.method_name
        params = ", ".join(f"{key}={value}" for key, value in self.params.items())
        return f"{self.method_name}({params})"

    def get_extra_metrics(self) -> Dict[str, Any]:
        """Subclasses can override this to expose method-specific metrics."""
        return {}

    # ------------------------------------------------------------------
    # Method-specific API expected from subclasses
    # ------------------------------------------------------------------

    def compress_array(self, array: np.ndarray) -> Any:
        raise NotImplementedError("Subclasses must implement compress_array.")

    def decompress_array(self, compressed: Any) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement decompress_array.")

    def save_compressed_payload(self, compressed_path: str, compressed: Any) -> None:
        """
        Optional payload writer.

        Subclasses should override this when the compressed representation is
        not directly serialisable by ``np.savez_compressed``.
        """
        os.makedirs(os.path.dirname(os.path.abspath(compressed_path)), exist_ok=True)
        np.savez_compressed(
            compressed_path,
            compressed=compressed,
            metadata_json=np.array(json.dumps(self.describe())),
        )

    # ------------------------------------------------------------------
    # Generic benchmark workflow
    # ------------------------------------------------------------------

    def compress_decompress_array(self, array: np.ndarray) -> tuple[Any, np.ndarray]:
        t0 = time.perf_counter()
        compressed = self.compress_array(array)
        self._last_compression_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        reconstructed = self.decompress_array(compressed)
        self._last_decompression_seconds = time.perf_counter() - t0

        return compressed, reconstructed

    def compress_decompress_h5(
        self,
        input_h5: str,
        output_h5: str,
        compressed_path: Optional[str] = None,
        dataset_name: str = "fdistribu",
    ) -> Dict[str, Any]:
        """
        Compress and immediately reconstruct one HDF5 dataset.

        The whole HDF5 file is copied first, then only ``dataset_name`` is
        replaced by the reconstruction. This keeps the restart container and
        metadata usable by the solver.
        """
        if not os.path.exists(input_h5):
            raise FileNotFoundError(input_h5)

        os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)
        shutil.copy2(input_h5, output_h5)

        with h5py.File(output_h5, "r+") as h5:
            if dataset_name not in h5:
                raise KeyError(f"Dataset '{dataset_name}' not found in {output_h5}.")

            f_original = h5[dataset_name][:]
            compressed, f_reconstructed = self.compress_decompress_array(f_original)

            if h5[dataset_name].shape != f_reconstructed.shape:
                raise RuntimeError(
                    "Reconstructed shape mismatch: "
                    f"expected {h5[dataset_name].shape}, got {f_reconstructed.shape}."
                )

            h5[dataset_name][...] = f_reconstructed

        if compressed_path is not None:
            self.save_compressed_payload(compressed_path, compressed)

        return self.compute_metrics(
            f_original=f_original,
            f_reconstructed=f_reconstructed,
            input_h5=input_h5,
            output_h5=output_h5,
            compressed_path=compressed_path,
        )

    def compress_decompress(self, *args, **kwargs) -> Dict[str, Any]:
        return self.compress_decompress_h5(*args, **kwargs)

    # ------------------------------------------------------------------
    # Generic metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_file_size(path: Optional[str]) -> Optional[int]:
        if path is None or path == "none" or not os.path.exists(path):
            return None
        return os.path.getsize(path)

    def compute_metrics(
        self,
        f_original: np.ndarray,
        f_reconstructed: np.ndarray,
        input_h5: Optional[str] = None,
        output_h5: Optional[str] = None,
        compressed_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        f_original = np.asarray(f_original)
        f_reconstructed = np.asarray(f_reconstructed)

        diff = f_original - f_reconstructed
        original_norm = np.linalg.norm(f_original.ravel())
        diff_norm = np.linalg.norm(diff.ravel())

        relative_l2_error = diff_norm / original_norm if original_norm > 0.0 else np.nan
        max_abs_error = np.max(np.abs(diff)) if diff.size else np.nan
        mean_abs_error = np.mean(np.abs(diff)) if diff.size else np.nan
        rmse = np.sqrt(np.mean(diff * diff)) if diff.size else np.nan

        original_size = self._safe_file_size(input_h5)
        reconstructed_size = self._safe_file_size(output_h5)
        compressed_size = self._safe_file_size(compressed_path)

        compression_ratio = (
            original_size / compressed_size
            if original_size is not None and compressed_size is not None and compressed_size > 0
            else None
        )

        metrics: Dict[str, Any] = {
            "input_h5": input_h5,
            "output_h5": output_h5,
            "compressed_path": compressed_path,
            "method_name": self.method_name,
            "param_names": self.param_names,
            "params": self._json_safe(self.params),
            **{f"param_{key}": self._json_safe(value) for key, value in self.params.items()},
            "relative_l2_error": float(relative_l2_error),
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(mean_abs_error),
            "rmse": float(rmse),
            "original_size": original_size,
            "reconstructed_size": reconstructed_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_seconds": self._last_compression_seconds,
            "decompression_seconds": self._last_decompression_seconds,
        }

        metrics.update(self._json_safe(self.get_extra_metrics()))
        return metrics
