"""Compressor selection for the offline and online compression pipelines."""

import inspect
from compression_methods.PCA import PCACompressor
from compression_methods.random_noise import RandomNoiseCompressor
from compression_methods.neural_network import NeuralNetworkCompressor, OnlineNeuralNetworkCompressor


"""
OFFLINE_COMPRESSOR_CLASS = PCACompressor
OFFLINE_COMPRESSOR_PARAMS = {
    "n_components": 8,
    "normalisation": "none",
    "clip_nonnegative": False,
}
"""

OFFLINE_COMPRESSOR_CLASS = NeuralNetworkCompressor
OFFLINE_COMPRESSOR_PARAMS = {
    "arch": "periodic_siren_deep_128",
    "lr": 1e-3,
    "max_iters": 500,
    "batch_size": 2000,
    "lbfgs_iters": 50,
}

"""
ONLINE_COMPRESSOR_CLASS = RandomNoiseCompressor
ONLINE_COMPRESSOR_PARAMS = {
   "relative_noise_level": 0.01,
}
"""

ONLINE_COMPRESSOR_CLASS = OnlineNeuralNetworkCompressor
ONLINE_COMPRESSOR_PARAMS = {
    "arch": "periodic_siren_small_32",
    "lr": 1e-3,
    "warm_iters_adam": 3,
    "warm_iters_lbfgs": 3,
    "refine_iters_adam": 2,
    "refine_iters_lbfgs": 2,
    "verbose": True,
    "debug_plot": True,
}

def build_offline_compressor(**overrides):
    accepted = set(inspect.signature(OFFLINE_COMPRESSOR_CLASS.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in overrides.items() if k in accepted}
    params = {**OFFLINE_COMPRESSOR_PARAMS, **filtered}
    return OFFLINE_COMPRESSOR_CLASS(**params)

def build_online_compressor(rank=None):
    params = dict(ONLINE_COMPRESSOR_PARAMS)
    if rank is not None:
        # Every MPI rank otherwise builds its compressor from the same default
        # seed, so all ranks' networks start from identical initial weights
        # and stay highly correlated (looking like duplicated output once
        # assembled) unless training runs long enough to fully diverge.
        params["seed"] = params.get("seed", 42) + rank
    return ONLINE_COMPRESSOR_CLASS(**params)
