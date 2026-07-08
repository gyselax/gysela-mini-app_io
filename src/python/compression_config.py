"""Compressor selection for the offline and online compression pipelines."""

from compression_methods.PCA import PCACompressor
from compression_methods.random_noise import RandomNoiseCompressor
from compression_methods.neural_network import NeuralNetworkCompressor


OFFLINE_COMPRESSOR_CLASS = PCACompressor
OFFLINE_COMPRESSOR_PARAMS = {
    "n_components": 8,
    "normalisation": "none",
    "clip_nonnegative": False,
}

# ONLINE_COMPRESSOR_CLASS = RandomNoiseCompressor
# ONLINE_COMPRESSOR_PARAMS = {
#     "relative_noise_level": 0.01,
# }

ONLINE_COMPRESSOR_CLASS = NeuralNetworkCompressor
ONLINE_COMPRESSOR_PARAMS = {
    "latent_dim": 16,
}


def build_offline_compressor():
    return OFFLINE_COMPRESSOR_CLASS(**OFFLINE_COMPRESSOR_PARAMS)


def build_online_compressor():
    return ONLINE_COMPRESSOR_CLASS(**ONLINE_COMPRESSOR_PARAMS)
