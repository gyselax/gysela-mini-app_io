import time
import numpy as np


class RandomNoiseCompressor:
    """Injects noise scaled to the local chunk's amplitude. Exposes
    the array-level interface compression methods are expected to implement."""

    method_name = "RandomNoise"

    def __init__(self, relative_noise_level=0.01, seed=None):
        self.relative_noise_level = relative_noise_level
        self._rng = np.random.default_rng(seed)

    def printable_name(self):
        return f"{self.method_name}(relative_noise_level={self.relative_noise_level})"

    def compress_decompress_array(self, array, rank=None, local_bounds=None):
        t0 = time.perf_counter()
        scale = self.relative_noise_level * np.abs(array).mean()
        noise = self._rng.normal(loc=0.0, scale=scale, size=array.shape)
        t1 = time.perf_counter()
        approx = array + noise
        t2 = time.perf_counter()

        diff = approx - array
        l2_ref = float(np.linalg.norm(array))
        metrics = {
            "method_name": self.method_name,
            "params": {"relative_noise_level": self.relative_noise_level},
            "relative_l2_error": float(np.linalg.norm(diff) / l2_ref) if l2_ref > 0 else 0.0,
            "max_abs_error": float(np.max(np.abs(diff))),
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "compression_seconds": t1 - t0,
            "decompression_seconds": t2 - t1,
            "compression_ratio": None,
        }
        return approx, metrics
