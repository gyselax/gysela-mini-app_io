"""Compressor selection for the offline and online compression pipelines."""

import inspect
import os, json
from compression_methods.PCA import PCACompressor
from compression_methods.random_noise import RandomNoiseCompressor
from compression_methods.neural_network import NeuralNetworkCompressor

OFFLINE_COMPRESSOR_CLASS = PCACompressor
OFFLINE_COMPRESSOR_PARAMS = {
    "n_components": 8,
    "normalisation": "none",
    "clip_nonnegative": False,
}

OFFLINE_COMPRESSOR_CLASS = NeuralNetworkCompressor
OFFLINE_COMPRESSOR_PARAMS = {
    "arch": "periodic_siren_deep_128",
    "lr": 1e-3,
    "max_iters": 2000,
    "batch_size": 2000,
    "lbfgs_iters": 50,
}

ONLINE_COMPRESSOR_CLASS = RandomNoiseCompressor
ONLINE_COMPRESSOR_PARAMS = {
   "relative_noise_level": 0.01,
}

"""
def build_offline_compressor(**overrides):
    accepted = set(inspect.signature(OFFLINE_COMPRESSOR_CLASS.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in overrides.items() if k in accepted}
    params = {**OFFLINE_COMPRESSOR_PARAMS, **filtered}
    return OFFLINE_COMPRESSOR_CLASS(**params)
"""

def build_offline_compressor(**overrides):
    method_override = os.environ.get("COMPRESSION_METHOD_OVERRIDE")
    compressor_class = OFFLINE_COMPRESSOR_CLASS
    base_params = dict(OFFLINE_COMPRESSOR_PARAMS)

    if method_override:
        choice = json.loads(method_override)
        class_name = choice.get("class")
        if class_name == "PCA":
            compressor_class = PCACompressor
        elif class_name == "NeuralNetwork":
            compressor_class = NeuralNetworkCompressor
        else:
            raise ValueError(f"Unknown COMPRESSION_METHOD_OVERRIDE class: {class_name!r}")
        base_params = choice.get("params", {})

    accepted = set(inspect.signature(compressor_class.__init__).parameters) - {"self"}
    filtered = {k: v for k, v in overrides.items() if k in accepted}
    params = {**base_params, **filtered}
    return compressor_class(**params)

def build_online_compressor():
    return ONLINE_COMPRESSOR_CLASS(**ONLINE_COMPRESSOR_PARAMS)
