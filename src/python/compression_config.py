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
    "warm_iters": 200,
    "refine_iters": 20,
}

def build_offline_compressor(**overrides):
    accepted = set(inspect.signature(OFFLINE_COMPRESSOR_CLASS.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in overrides.items() if k in accepted}
    params = {**OFFLINE_COMPRESSOR_PARAMS, **filtered}
    return OFFLINE_COMPRESSOR_CLASS(**params)

def build_online_compressor():
    return ONLINE_COMPRESSOR_CLASS(**ONLINE_COMPRESSOR_PARAMS)
