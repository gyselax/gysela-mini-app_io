"""Compressor selection for the offline and online compression pipelines."""

import inspect
import os, json
from compression_methods.PCA import PCACompressor
from compression_methods.random_noise import RandomNoiseCompressor
from compression_methods.neural_network import NeuralNetworkCompressor, OnlineNeuralNetworkCompressor



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
    "warm_max_iters": 2000,
    "batch_size": 2000,
    "polish_optimizer": "lbfgs",
    "lbfgs_iters": 50,
    "lbfgs_chunk_size": 200_000,
    "gn_iters": 50,
    "gn_n_map": 8000,
    "gn_init_damping": 1e-2,
    "gn_chunk_size": 2000,
}

ONLINE_COMPRESSOR_CLASS = RandomNoiseCompressor
ONLINE_COMPRESSOR_PARAMS = {
   "relative_noise_level": 0.01,
}

ONLINE_COMPRESSOR_CLASS = OnlineNeuralNetworkCompressor
ONLINE_COMPRESSOR_PARAMS = {
    "arch": "periodic_siren_deep_128",
    "lr": 1e-4,
    "warm_iters_adam": 5000,
    "warm_iters_lbfgs": 100,
    "refine_iters_adam": 500,
    "refine_iters_lbfgs": 10,
    #"batch_size": 2000,
    "verbose": True,
    "debug_plot": True,
}

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

def build_online_compressor(rank=None):
    params = dict(ONLINE_COMPRESSOR_PARAMS)
    if rank is not None:
        # Every MPI rank otherwise builds its compressor from the same default
        # seed, so all ranks' networks start from identical initial weights
        # and stay highly correlated (looking like duplicated output once
        # assembled) unless training runs long enough to fully diverge.
        params["seed"] = params.get("seed", 42) + rank
    return ONLINE_COMPRESSOR_CLASS(**params)
