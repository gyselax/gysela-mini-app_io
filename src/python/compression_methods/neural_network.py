import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

class NeuralNetworkCompressor:
    """Placeholder neural-network compressor. Training is not implemented yet;
    instead the training data is plotted and saved to a png. Exposes the
    array-level interface compression methods are expected to implement."""

    method_name = "NeuralNetwork"

    def __init__(self, latent_dim=16, plot_path="nn_training_data.png"):
        self.latent_dim = latent_dim
        self.plot_path = plot_path

    def printable_name(self):
        return f"{self.method_name}(latent_dim={self.latent_dim})"

    def _plot_path_for_rank(self, rank):
        if rank is None:
            return self.plot_path
        path = Path(self.plot_path)
        return str(path.with_name(f"{path.stem}_rank{rank:03d}{path.suffix}"))

    def _train(self, array, rank=None):
        print(f"Plotting training data for rank {rank}")
        print(f"data shape: {array.shape}")
        # array is fdistribu[sp, x, y, vx, vy]; slice to f[x, vx] (species 0, summed over y, vy)
        f_xvx =array[0,:,8,:,8]
        ariel = jnp.zeros(4)
        print(f"ariel: {jnp.sum(ariel)}", flush=True)
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(f_xvx, cmap="viridis", shading="auto")
        fig.colorbar(mesh, ax=ax, label="f(x, vx)")
        ax.set_xlabel("vx index")
        ax.set_ylabel("x index")
        ax.set_title("Training data slice (x, vx)")
        fig.savefig(self._plot_path_for_rank(rank))
        plt.close(fig)

    def compress_decompress_array(self, array, rank=None):
        t0 = time.perf_counter()
        self._train(array, rank=rank)
        t1 = time.perf_counter()
        approx = array.copy()
        t2 = time.perf_counter()

        diff = approx - array
        l2_ref = float(np.linalg.norm(array))
        metrics = {
            "method_name": self.method_name,
            "params": {"latent_dim": self.latent_dim},
            "relative_l2_error": float(np.linalg.norm(diff) / l2_ref) if l2_ref > 0 else 0.0,
            "max_abs_error": float(np.max(np.abs(diff))),
            "mean_abs_error": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "compression_seconds": t1 - t0,
            "decompression_seconds": t2 - t1,
            "compression_ratio": None,
        }
        return approx, metrics
