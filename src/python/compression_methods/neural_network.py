import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.optimizers.optimizers import (ScimbaAdam, ScimbaLBfgs)

from Compressor import Compressor

jax.config.update("jax_enable_x64", True)


def _training_progress(n_iters: int, desc: str, verbose: bool):
    """tqdm progress bar (iteration count, elapsed/ETA, rate) over a training
    loop; live loss is attached per-step via `pbar.set_postfix(...)`. A no-op
    passthrough when `verbose` is False so silent runs pay no overhead.
    """
    return tqdm(range(n_iters), desc=desc, disable=not verbose, leave=True, dynamic_ncols=True)

def periodic_embedding(x_input: jnp.ndarray) -> jnp.ndarray:
    """Embed raw (x, y, vx, vy) into periodic features for x and y only.
    Returns shape (6,). vx, vy pass through unchanged (non-periodic dimensions).
    Every periodic architecture below feeds its raw input through this
    single function
    """
    x_coord = x_input[0:1]
    y_coord = x_input[1:2]
    v_coords = x_input[2:4]
 
    kx = 2 * jnp.pi
    ky = 2 * jnp.pi
    trig = jnp.concatenate(
        [
            jnp.cos(kx * x_coord),
            jnp.sin(kx * x_coord),
            jnp.cos(ky * y_coord),
            jnp.sin(ky * y_coord),
        ],
        axis=-1,
    )
    return jnp.concatenate([trig, v_coords], axis=-1)

# Network architectures
def _siren_init(layer: eqx.nn.Linear, in_size: int, omega_0: float, is_first: bool, key: jax.Array) -> eqx.nn.Linear:
    """Re-initialize a SIREN linear layer's weight per Sitzmann et al. 2020.

    The first layer samples U(-1/fan_in, 1/fan_in); every later layer
    (including the final linear readout) samples U(-sqrt(6/fan_in)/omega_0,
    sqrt(6/fan_in)/omega_0), which compensates for the omega_0 factor applied
    inside each sine activation so pre-activations keep unit-ish variance
    regardless of omega_0. Equinox's default Linear init ignores omega_0
    entirely, which leaves SIREN badly conditioned at init.
    """
    lim = 1.0 / in_size if is_first else jnp.sqrt(6.0 / in_size) / omega_0
    new_weight = jax.random.uniform(key, layer.weight.shape, minval=-lim, maxval=lim, dtype=layer.weight.dtype)
    return eqx.tree_at(lambda l: l.weight, layer, new_weight)


class SIRENScimbaINR(eqx.Module):
    """SIREN network, non-periodic. Raw input (x, y, vx, vy) -> in_size=4"""

    layers: tuple
    omega_0: float = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, hidden_sizes: list[int], omega_0: float, key: jax.Array):
        self.omega_0 = omega_0
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size] + hidden_sizes + [out_size]

        layers = []
        for i in range(len(sizes) - 1):
            layer = eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            layer = _siren_init(layer, sizes[i], omega_0, is_first=(i == 0), key=keys[i])
            layers.append(layer)
        self.layers = tuple(layers)

    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        x = jnp.sin(self.omega_0 * self.layers[0](x_input))
        for layer in self.layers[1:-1]:
            x = jnp.sin(self.omega_0 * layer(x))
        return self.layers[-1](x)
    
    def ndof(self) -> int:
        flat_params, _ = jax.tree_util.tree_flatten(self)
        return sum(p.size for p in flat_params if isinstance(p, jnp.ndarray))
    
class PeriodicSIRENScimbaINR(eqx.Module):
    """SIREN wrapped with periodic embedding on (x,y). in_size=6 internally"""
    network: SIRENScimbaINR
    
    def __init__(self, hidden_sizes: list[int], omega_0: float, key: jax.Array):
        self.network = SIRENScimbaINR(
            in_size=6, 
            out_size=1, 
            hidden_sizes=hidden_sizes, 
            omega_0=omega_0, 
            key=key
        )
        
    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        return self.network(periodic_embedding(x_input))
    
class FourierScimbaINR(eqx.Module):
    """Random Fourier Features, non-periodic. Raw input (x, y, vx, vy) -> in_features=4"""
    network: MLP
    B: jnp.ndarray
    
    def __init__(self, in_features: int, n_freqs: int, hidden_sizes: list[int], sigma: float, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.B = jax.random.normal(k1, (in_features, n_freqs)) * sigma
        self.network = MLP(
            in_size=2 * n_freqs, # cos and sin features
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation="tanh",
            key=k2  
        )
    
    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        proj = x_input @ self.B
        h = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        
        return self.network(h)
    
class PeriodicFourierScimbaINR(eqx.Module):
    """Random Fourier Features + periodic embedding on (x, y). B has shape (6, n_freqs)"""
    network: MLP
    B: jnp.ndarray
    
    def __init__(self, n_freqs: int, hidden_sizes: list[int], sigma: float, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.B = jax.random.normal(k1, (6, n_freqs)) * sigma
        self.network = MLP(
            in_size=2 * n_freqs, # cos and sin features
            out_size=1,
            hidden_sizes=hidden_sizes,
            activation="tanh",
            key=k2  
        )
        
    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        h = periodic_embedding(x_input)
        proj = h @ jax.lax.stop_gradient(self.B)
        h_fourier = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        
        return self.network(h_fourier)
    
AVAILABLE_INR_ARCHS = [
    "periodic_siren_deep_128",
    "periodic_fourier_mlp_deep_128",
    "periodic_siren_small_32",
    "periodic_fourier_mlp_small_32",
]

def get_inr_model(arch: str, key: jax.Array) -> eqx.Module:
    """Instantiate an INR architecture"""
    if arch == "periodic_siren_deep_128": return PeriodicSIRENScimbaINR([128]*5, 30.0, key)
    elif arch == "periodic_fourier_mlp_deep_128": return PeriodicFourierScimbaINR(16, [128]*5, 10.0, key)
    elif arch == "periodic_siren_small_32": return PeriodicSIRENScimbaINR([32]*3, 30.0, key)
    elif arch == "periodic_fourier_mlp_small_32": return PeriodicFourierScimbaINR(8, [32]*3, 10.0, key)
    else:
        raise ValueError(f"Unknown INR architecture: {arch}. Available: {AVAILABLE_INR_ARCHS}")
    
# Training losses
@jax.jit
def _losses_function(model: eqx.Module, batch: tuple) -> dict:
    inputs, targets = batch
    predictions = jax.vmap(model)(inputs)
    mse = jnp.mean((predictions - targets) ** 2)
    
    return {"total":mse}
    
# Compressor (offline)

class NeuralNetworkCompressor(Compressor):
    """INR compressor for fdistribu[species, x, y, vx, vy] restart fields.
 
    Fits one INR per species over the physical grid inferred from the array
    shape, using an ADAM (exploration) + L-BFGS (exploitation) two-phase
    routine ported from PlasmaX's compress_inr.
 
    NOTE: this class conforms to the Compressor blueprint (offline workflow,
    called via compress_decompress_h5 / launch_benchmark.py-style pipelines).
    It does NOT implement the array-in/array-out online interface used by
    compression_diagnostics.py's apply_online_compression.
    """
    
    method_name = "NeuralNetwork"
    
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        vx_min: float,
        vx_max: float,
        vy_min: float,
        vy_max: float,
        arch: str = "periodic_siren_deep_128",
        lr: float = 1e-3,
        max_iters: int = 2000,
        batch_size: int = 2000,
        lbfgs_iters: int = 50,
        threshold: float = 1e-8,
        seed: int = 42,
        warm_start_payload: Optional[str] = None,
        verbose: bool = True,
    ):
        if arch not in AVAILABLE_INR_ARCHS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {AVAILABLE_INR_ARCHS}")
            
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.vx_min, self.vx_max = float(vx_min), float(vx_max)
        self.vy_min, self.vy_max = float(vy_min), float(vy_max)
 
        self.arch = arch
        self.lr = float(lr)
        self.max_iters = int(max_iters)
        self.batch_size = int(batch_size)
        self.lbfgs_iters = int(lbfgs_iters)
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.warm_start_payload = warm_start_payload
        self.verbose = bool(verbose)
        
        super().__init__(
            method_name=self.method_name,
            arch=self.arch,
            lr=self.lr,
            max_iters=self.max_iters,
            batch_size=self.batch_size,
            lbfgs_iters=self.lbfgs_iters,
        )
        
        self.original_shape = None
        self.models: list[eqx.Module] = []
        self.loss_histories: list[jnp.ndarray] = []
        
    # Grid reconstruction
    
    def _build_grid(self, nx: int, ny: int, nvx: int, nvy: int):
        x = jnp.linspace(self.x_min, self.x_max, nx, endpoint=False)
        y = jnp.linspace(self.y_min, self.y_max, ny, endpoint=False)
        vx = jnp.linspace(self.vx_min, self.vx_max, nvx, endpoint=True)
        vy = jnp.linspace(self.vy_min, self.vy_max, nvy, endpoint=True)
        
        return x, y, vx, vy
    
    def _build_inputs(self, nx:int, ny: int, nvx: int, nvy: int) -> jnp.ndarray:
        x, y, vx, vy = self._build_grid(nx, ny, nvx, nvy)
        Xg, Yg, VXg, VYg = jnp.meshgrid(x, y, vx, vy, indexing="ij")
        #normalization
        x_norm = (Xg.ravel() - self.x_min) / (self.x_max - self.x_min)
        y_norm = (Yg.ravel() - self.y_min) / (self.y_max - self.y_min)
        vx_norm = 2.0 * (VXg.ravel() - self.vx_min) / (self.vx_max - self.vx_min) - 1.0
        vy_norm = 2.0 * (VYg.ravel() - self.vy_min) / (self.vy_max - self.vy_min) - 1.0 
        
        inputs = jnp.stack([x_norm, y_norm, vx_norm, vy_norm], axis=-1)
        
        return inputs
    
    def _load_warm_start_models(self, n_species: int, key: jax.Array) -> list[Optional[eqx.Module]]:
        if self.warm_start_payload is None or not os.path.exists(self.warm_start_payload):
            return [None] * n_species
        
        with np.load(self.warm_start_payload, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata.get("arch") != self.arch:
                raise ValueError(
                    f"warm_start_payload was trained with arch={metadata.get('arch')!r}"
                    f"but this compressor uses arch={self.arch!r}"
                )
                
            n_saved = int(metadata["n_species"])
            if n_saved != n_species:
                raise ValueError(
                    f"warm_start_payload has {n_saved} species, array has {n_species}"
                )
            
            warm_models = []
            for isp in range(n_species):
                template = get_inr_model(self.arch, key)
                flat_template, unravel_fn = ravel_pytree(template)
                loaded_flat = jnp.asarray(payload[f"weights_{isp}"])
                if loaded_flat.shape != flat_template.shape:
                    raise ValueError(
                        f"Shape mismatch loading warm start for species {isp}: "
                        f"expected {flat_template.shape}, got {loaded_flat.shape}"
                    )
                warm_models.append(unravel_fn(loaded_flat))
            return warm_models
        
    # Training (one species at a time)
    
    def _fit_on_species(self, inputs: jnp.ndarray, targets: jnp.ndarray, key: jax.Array, warm_model):
        if warm_model is not None:
            model = warm_model
        else:
            key, subkey = jax.random.split(key)
            model = get_inr_model(self.arch, subkey)
        
        total_points = inputs.shape[0]
        loss_history = []
        best_model, best_loss = model, float("inf")

        #Phase 1: ADAM, mini-batches
        adam_opt = ScimbaAdam(model, _losses_function, learning_rate=self.lr)
        pbar = _training_progress(self.max_iters, f"[INR/{self.arch}][ADAM]", self.verbose)
        for i in pbar:
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.choice(subkey, total_points, shape=(self.batch_size,), replace=False)
            batch = (inputs[batch_idx], targets[batch_idx])

            loss_dict, model, adam_opt = adam_opt.update(model, batch)
            loss_val = float(loss_dict["total"])
            loss_history.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.2e}")

            if loss_val < best_loss:
                best_model, best_loss = model, loss_val

            if loss_val < self.threshold:
                pbar.write(f"  [INR/{self.arch}][ADAM] early convergence at iter {i}")
                break

        #Phase 2: L-BFGS, full-batch
        full_batch = (inputs, targets)
        lbfgs_opt = ScimbaLBfgs(model, _losses_function)
        pbar = _training_progress(self.lbfgs_iters, f"[INR/{self.arch}][L-BFGS]", self.verbose)
        for i in pbar:
            loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
            loss_val = float(loss_dict["total"])
            loss_history.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.2e}")

            if loss_val < best_loss:
                best_model, best_loss = model, loss_val

            if loss_val < self.threshold:
                pbar.write(f"  [INR/{self.arch}][L-BFGS] convergence at iter {i}")
                break

        return best_model, jnp.array(loss_history)
    
    # Compressor interface
    
    def compress_array(self, f:np.ndarray) -> dict:
        f = jnp.asarray(f, dtype=jnp.float64)
        if f.ndim != 5:
            raise ValueError(
                "Expected fdistribu with rank 5 (Nspecies, Nx, Ny, Nvx, Nvy)"
                f"got shape {f.shape}"
            )
            
        n_species, nx, ny, nvx, nvy = f.shape
        self.original_shape = f.shape
        
        inputs = self._build_inputs(nx, ny, nvx, nvy)
        
        key = jax.random.PRNGKey(self.seed)
        warm_models = self._load_warm_start_models(n_species, key)
        
        self.models = []
        self.loss_histories = []

        if self.verbose:
            print(f"[INR/{self.arch}] device: {jax.devices()}", flush=True)

        for isp in range(n_species):
            targets = f[isp].reshape(-1, 1)
            key, subkey = jax.random.split(key)
            t0 = time.perf_counter()
            model, loss_hist = self._fit_on_species(inputs, targets, subkey, warm_models[isp])
            t1 = time.perf_counter()
            
            if self.verbose:
                print(
                    f"[INR/{self.arch}] species {isp}: best loss "
                    f"{float(jnp.min(loss_hist)):.2e} ({t1 - t0:.2f}s)",
                    flush=True,
                )
            
            self.models.append(model)
            self.loss_histories.append(loss_hist)
        
        return {"models": self.models, "grid_shape": (nx, ny, nvx, nvy)}
    
    def decompress_array(self, compressed: dict) -> jnp.ndarray:
        models = compressed["models"]
        nx, ny, nvx, nvy = compressed["grid_shape"]
        inputs = self._build_inputs(nx, ny, nvx, nvy)
        
        species_out = []
        for model in models:
            pred = jax.vmap(model)(inputs).reshape(nx, ny, nvx, nvy)
            species_out.append(pred)
            
        return jnp.stack(species_out)
    
    def save_compressed_payload(self, compressed_path: str, compressed: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(compressed_path)), exist_ok=True)
        
        metadata = {
            "arch": self.arch,
            "n_species": len(compressed["models"]),
            "grid_shape": compressed["grid_shape"],
            "bounds":{
                "x_min": self.x_min, "x_max": self.x_max,
                "y_min": self.y_min, "y_max": self.y_max,
                "vx_min": self.vx_min, "vx_max": self.vx_max,
                "vy_min": self.vy_min, "vy_max": self.vy_max,
            },
        }
        
        payload: dict[str, Any] = {"metadata_json": np.array(json.dumps(metadata))}
        for isp, model in enumerate(compressed["models"]):
            flat, _ = ravel_pytree(model)
            payload[f"weights_{isp}"] = np.asarray(flat)
        
        np.savez_compressed(compressed_path, **payload)
        
    def get_extra_metrics(self) -> dict:
        return {
            "final_loss_per_species": [float(jnp.min(h)) for h in self.loss_histories] if self.loss_histories else None,
            "n_train_steps_per_species": [int(h.shape[0]) for h in self.loss_histories] if self.loss_histories else None,
        }

# Compressor (online / in-situ)

class OnlineNeuralNetworkCompressor:
    """In-situ INR compressor for one rank's local fdistribu[species, x, y, vx, vy] chunk.

    Fits one small NN f_theta^(i)(z) per species directly on rank i's own
    subdomain, with no cross-rank communication. Coordinates are normalized
    to the local chunk's own unit cell ([0,1) for x,y, [-1,1] for vx,vy);
    `local_bounds`, if passed, is stored so `assemble_global_field` can later
    map each rank's unit cell back to physical space.

    Runs inside the live timestepping loop, so models and ADAM state persist
    across calls (warm start): only `refine_iters_adam`/`refine_iters_lbfgs`
    steps run after the first call for a given chunk shape (`warm_iters_adam`
    /`warm_iters_lbfgs` on cold start). Same two-phase ADAM + L-BFGS routine
    as `NeuralNetworkCompressor` above; the L-BFGS optimizer is re-instantiated
    fresh each call since its history is only meaningful right after ADAM.
    """

    method_name = "OnlineNeuralNetwork"

    def __init__(
        self,
        arch: str = "periodic_siren_small_32",
        lr: float = 1e-3,
        warm_iters_adam: int = 200,
        warm_iters_lbfgs: int = 20,
        refine_iters_adam: int = 20,
        refine_iters_lbfgs: int = 5,
        batch_size: Optional[int] = None,
        threshold: float = 1e-8,
        seed: int = 42,
        verbose: bool = False,
        debug_plot: bool = False,
        save_target: bool = False,
    ):
        if arch not in AVAILABLE_INR_ARCHS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {AVAILABLE_INR_ARCHS}")

        self.arch = arch
        self.lr = float(lr)
        self.warm_iters_adam = int(warm_iters_adam)
        self.warm_iters_lbfgs = int(warm_iters_lbfgs)
        self.refine_iters_adam = int(refine_iters_adam)
        self.refine_iters_lbfgs = int(refine_iters_lbfgs)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.debug_plot = bool(debug_plot)
        self.save_target = bool(save_target)

        self._key = jax.random.PRNGKey(self.seed)
        self.models: Optional[list] = None        # one eqx.Module per species, warm-started
        self._opts: Optional[list] = None          # one ScimbaAdam per species, warm-started
        self.local_shape: Optional[tuple] = None   # (nx, ny, nvx, nvy) of the local chunk
        self.local_bounds: Optional[tuple] = None  # physical bbox of the local chunk, if known
        self.n_calls = 0
        self._last_local_array: Optional[np.ndarray] = None  # (n_species, nx, ny, nvx, nvy), last fit target

    def printable_name(self) -> str:
        return (
            f"{self.method_name}(arch={self.arch}, lr={self.lr}, "
            f"warm_iters_adam={self.warm_iters_adam}, warm_iters_lbfgs={self.warm_iters_lbfgs}, "
            f"refine_iters_adam={self.refine_iters_adam}, refine_iters_lbfgs={self.refine_iters_lbfgs})"
        )

    @staticmethod
    def _build_local_inputs(nx: int, ny: int, nvx: int, nvy: int) -> jnp.ndarray:
        x = jnp.linspace(0.0, 1.0, nx, endpoint=False)
        y = jnp.linspace(0.0, 1.0, ny, endpoint=False)
        vx = jnp.linspace(-1.0, 1.0, nvx, endpoint=True)
        vy = jnp.linspace(-1.0, 1.0, nvy, endpoint=True)
        Xg, Yg, VXg, VYg = jnp.meshgrid(x, y, vx, vy, indexing="ij")
        return jnp.stack([Xg.ravel(), Yg.ravel(), VXg.ravel(), VYg.ravel()], axis=-1)

    def _fit_one_species(
        self, model, opt, inputs: jnp.ndarray, targets: jnp.ndarray, n_iters_adam: int, n_iters_lbfgs: int,
        desc: str = "",
    ):
        total_points = inputs.shape[0]
        bs = min(self.batch_size, total_points) if self.batch_size is not None else total_points
        loss_val = None
        best_model, best_loss = model, float("inf")

        # Phase 1: ADAM, warm-started across calls, mini-batches
        pbar = _training_progress(n_iters_adam, f"{desc}[ADAM]", self.verbose)
        for _ in pbar:
            if bs < total_points:
                self._key, subkey = jax.random.split(self._key)
                idx = jax.random.choice(subkey, total_points, shape=(bs,), replace=False)
                batch = (inputs[idx], targets[idx])
            else:
                batch = (inputs, targets)

            loss_dict, model, opt = opt.update(model, batch)
            loss_val = float(loss_dict["total"])
            pbar.set_postfix(loss=f"{loss_val:.2e}")
            if loss_val < best_loss:
                best_model, best_loss = model, loss_val
            if loss_val < self.threshold:
                break

        # Phase 2: L-BFGS, full-batch, freshly instantiated each call to polish
        # the ADAM warm-started fit (same two-phase routine as the offline compressor)
        if n_iters_lbfgs > 0 and (loss_val is None or loss_val >= self.threshold):
            full_batch = (inputs, targets)
            lbfgs_opt = ScimbaLBfgs(model, _losses_function)
            pbar = _training_progress(n_iters_lbfgs, f"{desc}[L-BFGS]", self.verbose)
            for _ in pbar:
                loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
                loss_val = float(loss_dict["total"])
                pbar.set_postfix(loss=f"{loss_val:.2e}")
                if loss_val < best_loss:
                    best_model, best_loss = model, loss_val
                if loss_val < self.threshold:
                    break

        return best_model, opt, best_loss

    def _plot_local_xvx_debug(self, f_orig, f_approx, rank: Optional[int] = None):
        """Save a quick x–vx plot of local original vs reconstruction (species 0)."""
        import matplotlib.pyplot as plt

        f0 = np.asarray(f_orig[0])
        a0 = np.asarray(f_approx[0])
        # Marginalize over y and vy -> (x, vx)
        f_xvx = np.sum(f0, axis=(1, 3))
        a_xvx = np.sum(a0, axis=(1, 3))

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, data, title in (
            (axs[0], f_xvx, "local original"),
            (axs[1], a_xvx, "local reconstruction"),
            (axs[2], np.abs(f_xvx - a_xvx), "|error|"),
        ):
            im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("vx index")
            ax.set_ylabel("x index")
            fig.colorbar(im, ax=ax, fraction=0.046)

        rank_tag = f"{rank:03d}" if rank is not None else "na"
        out = f"nn_online_local_rank{rank_tag}_call{self.n_calls:04d}_xvx.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        if self.verbose:
            print(f"[OnlineINR] debug plot written: {out}", flush=True)

    def compress_decompress_array(self, array: np.ndarray, rank: Optional[int] = None, local_bounds: Optional[tuple] = None):
        f = jnp.asarray(array, dtype=jnp.float64)
        if f.ndim != 5:
            raise ValueError(
                "Expected local fdistribu with rank 5 (Nspecies, Nx, Ny, Nvx, Nvy), "
                f"got shape {f.shape}"
            )

        n_species, nx, ny, nvx, nvy = f.shape
        local_shape = (nx, ny, nvx, nvy)

        if local_bounds is not None:
            self.local_bounds = tuple(float(b) for b in local_bounds)

        cold_start = self.models is None or self.local_shape != local_shape
        if cold_start:
            self.local_shape = local_shape
            keys = jax.random.split(self._key, n_species + 1)
            self._key = keys[0]
            self.models = [get_inr_model(self.arch, keys[i + 1]) for i in range(n_species)]
            self._opts = [ScimbaAdam(self.models[i], _losses_function, learning_rate=self.lr) for i in range(n_species)]

        inputs = self._build_local_inputs(nx, ny, nvx, nvy)
        n_iters_adam = self.warm_iters_adam if cold_start else self.refine_iters_adam
        n_iters_lbfgs = self.warm_iters_lbfgs if cold_start else self.refine_iters_lbfgs

        t0 = time.perf_counter()
        final_losses = []
        for isp in range(n_species):
            targets = f[isp].reshape(-1, 1)
            model, opt, loss_val = self._fit_one_species(
                self.models[isp], self._opts[isp], inputs, targets, n_iters_adam, n_iters_lbfgs,
                desc=f"[OnlineINR/{self.arch}][rank={rank}][sp={isp}]",
            )
            self.models[isp] = model
            self._opts[isp] = opt
            final_losses.append(loss_val)
        t1 = time.perf_counter()

        recon_species = [jax.vmap(self.models[isp])(inputs).reshape(nx, ny, nvx, nvy) for isp in range(n_species)]
        approx = jnp.stack(recon_species)
        t2 = time.perf_counter()

        if self.verbose:
            tag = "cold" if cold_start else "warm"
            print(f"[OnlineINR/{self.arch}] rank {rank}: device {jax.default_backend()}", flush=True)
            print(
                f"[OnlineINR/{self.arch}] rank {rank}: {tag} fit "
                f"(ADAM {n_iters_adam} + L-BFGS {n_iters_lbfgs} iters) "
                f"final losses {['%.2e' % l for l in final_losses]}",
                flush=True,
            )

        diff = approx - f
        l2_ref = float(jnp.linalg.norm(f))

        metrics = {
            "method_name": self.method_name,
            "params": {
                "arch": self.arch,
                "lr": self.lr,
                "warm_iters_adam": self.warm_iters_adam,
                "warm_iters_lbfgs": self.warm_iters_lbfgs,
                "refine_iters_adam": self.refine_iters_adam,
                "refine_iters_lbfgs": self.refine_iters_lbfgs,
                "cold_start": cold_start,
            },
            "relative_l2_error": float(jnp.linalg.norm(diff) / l2_ref) if l2_ref > 0 else 0.0,
            "max_abs_error": float(jnp.max(jnp.abs(diff))),
            "mean_abs_error": float(jnp.mean(jnp.abs(diff))),
            "rmse": float(jnp.sqrt(jnp.mean(diff ** 2))),
            "final_loss_per_species": final_losses,
            "compression_seconds": t1 - t0,
            "decompression_seconds": t2 - t1,
            "compression_ratio": None,
            "local_bounds": self.local_bounds,
        }

        self.n_calls += 1
        self._last_local_array = np.asarray(f)

        if self.debug_plot:
            self._plot_local_xvx_debug(f, approx, rank=rank)

        return np.asarray(approx), metrics

    def save_params(self, path: str, rank: Optional[int] = None, timestep: Optional[int] = None) -> None:
        """Serialize the current per-species models plus the local data chunk
        they were just fit on, so `load_online_params` can later reload and
        evaluate/plot them without rerunning the simulation.
        """
        if self.models is None or self._last_local_array is None:
            raise RuntimeError("save_params called before any compress_decompress_array call")

        path = str(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

        metadata = {
            "arch": self.arch,
            "n_species": len(self.models),
            "local_shape": self.local_shape,
            "local_bounds": self.local_bounds,
            "rank": rank,
            "iter": timestep,
        }

        payload: dict[str, Any] = {"metadata_json": np.array(json.dumps(metadata))}
        for isp, model in enumerate(self.models):
            flat, _ = ravel_pytree(model)
            payload[f"weights_{isp}"] = np.asarray(flat)
            if self.save_target:
                payload[f"target_{isp}"] = self._last_local_array[isp]

        np.savez_compressed(path, **payload)


def load_online_params(path: str) -> dict:
    """Reload a payload written by `OnlineNeuralNetworkCompressor.save_params`.

    Returns a dict with keys: arch, rank, iter, local_shape, local_bounds,
    models (list of eqx.Module, one per species, ready to evaluate on the
    unit-cell grid built by `OnlineNeuralNetworkCompressor._build_local_inputs`),
    and target (np.ndarray, shape (n_species, nx, ny, nvx, nvy) -- the local
    data the models were fit on -- or None if the payload was saved with
    `save_target=False`).
    """
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        n_species = int(metadata["n_species"])
        arch = metadata["arch"]
        has_target = f"target_0" in payload.files

        key = jax.random.PRNGKey(0)
        models = []
        targets = []
        for isp in range(n_species):
            template = get_inr_model(arch, key)
            _, unravel_fn = ravel_pytree(template)
            flat = jnp.asarray(payload[f"weights_{isp}"])
            models.append(unravel_fn(flat))
            if has_target:
                targets.append(np.asarray(payload[f"target_{isp}"]))

    local_bounds = metadata.get("local_bounds")
    return {
        "arch": arch,
        "rank": metadata.get("rank"),
        "iter": metadata.get("iter"),
        "local_shape": tuple(metadata["local_shape"]),
        "local_bounds": tuple(local_bounds) if local_bounds is not None else None,
        "models": models,
        "target": np.stack(targets) if has_target else None,
    }

# Offline continuation (fine-tune a saved online network without rerunning the simulation)


def continue_training_offline(
    payload: dict,
    arch: Optional[str] = None,
    species: Optional[list] = None,
    lr: float = 1e-3,
    iters_adam: int = 1000,
    iters_lbfgs: int = 50,
    batch_size: Optional[int] = None,
    threshold: float = 1e-8,
    warm_start: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Continue training one rank's saved online INR(s) offline.

    Drives a single `OnlineNeuralNetworkCompressor.compress_decompress_array`
    call -- the exact fitting routine used in-situ (ADAM warm start + L-BFGS
    polish), for `iters_adam` + `iters_lbfgs` steps -- directly from a
    payload already loaded by `load_online_params`, instead of from a live
    simulation call. This lets you iterate on architecture and training
    hyperparameters against already-captured local target data without
    rerunning the distributed simulation: `payload["target"]` is the fixed
    local ground truth the rank was fit on, and `payload["models"]` /
    `payload["arch"]` are its saved weights. When `verbose`, a live tqdm
    progress bar (iteration count, current loss, ETA) tracks each phase --
    the same progress reporting `_fit_one_species` gives the regular in-situ
    online path.

    arch: architecture to train. Defaults to the payload's saved arch (warm
        start from the saved weights). Passing a *different* arch instead
        trains fresh models of that arch from scratch on the same saved
        target data -- for architecture search without touching the
        simulation.
    species: species indices to fine-tune (default: every species in the
        payload). Species not selected keep their saved weights untouched.
    warm_start: if True and `arch` matches the payload's saved arch, continue
        from the saved weights; if False (or `arch` differs), start the
        selected species from a fresh random init instead.

    Returns {"models", "final_losses" (species -> final loss), "metrics"
    (the compress_decompress_array metrics dict), "local_shape",
    "local_bounds", "arch", "target", "compressor"}. Pass `compressor` to
    `.save_params(...)` to write a params_iterXXXXX_rankXXX.npz payload the
    existing evaluate_compression.py tooling (`evaluate_rank`,
    `run_online_networks`, ...) can load unchanged.
    """
    target = payload["target"]
    if target is None:
        raise ValueError(
            "continue_training_offline requires target data, but this payload "
            "was saved with save_target=False."
        )
    n_species = target.shape[0]
    species_idx = list(range(n_species)) if species is None else list(species)

    use_arch = arch or payload["arch"]
    reuse_weights = warm_start and use_arch == payload["arch"]

    compressor = OnlineNeuralNetworkCompressor(
        arch=use_arch,
        lr=lr,
        refine_iters_adam=iters_adam,
        refine_iters_lbfgs=iters_lbfgs,
        batch_size=batch_size,
        threshold=threshold,
        seed=seed,
        verbose=verbose,
    )
    # Pre-populate local_shape/models so compress_decompress_array's
    # cold_start branch never fires: our own choice of starting weights below
    # (warm-started or freshly initialized) stays authoritative, and this
    # call runs exactly `iters_adam` + `iters_lbfgs` steps.
    compressor.local_shape = payload["local_shape"]
    compressor.local_bounds = payload["local_bounds"]

    if reuse_weights:
        compressor.models = list(payload["models"])
    else:
        keys = jax.random.split(compressor._key, n_species + 1)
        compressor._key = keys[0]
        compressor.models = [
            get_inr_model(use_arch, keys[isp + 1]) if isp in species_idx else payload["models"][isp]
            for isp in range(n_species)
        ]
    compressor._opts = [
        ScimbaAdam(compressor.models[isp], _losses_function, learning_rate=lr) for isp in range(n_species)
    ]

    _, metrics = compressor.compress_decompress_array(
        target, rank=payload.get("rank"), local_bounds=payload["local_bounds"]
    )
    compressor._last_local_array = np.asarray(target)

    final_losses = {isp: metrics["final_loss_per_species"][isp] for isp in species_idx}

    return {
        "models": compressor.models,
        "final_losses": final_losses,
        "metrics": metrics,
        "local_shape": compressor.local_shape,
        "local_bounds": compressor.local_bounds,
        "arch": use_arch,
        "target": target,
        "compressor": compressor,
    }

# Global assembly (offline reconstruction / visualization utility)

def assemble_global_field(
    local_models: list,
    local_bounds: list,
    query_points: jnp.ndarray,
) -> jnp.ndarray:
    """Reassemble a global field from per-rank local INRs by exact-domain dispatch.

    Each query point is evaluated by exactly one rank's network -- whichever
    one's `local_bounds` contains it -- with no cross-rank blending. This
    matches the actual decomposition: every rank's network already spans the
    full x,y domain (only vx,vy are split across ranks), and that vx,vy split
    is a hard partition of the discrete grid with no shared points, so each
    physical grid point belongs to exactly one rank.

    This is a post-hoc reconstruction/visualization utility, not part of the
    online round trip -- during the simulation each rank only ever writes
    its own reconstructed chunk back in place.

    Args:
        local_models: local_models[i] is the list of per-species eqx.Module
            models trained by rank i's OnlineNeuralNetworkCompressor.
        local_bounds: local_bounds[i] is rank i's physical bounding box
            (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max).
        query_points: physical (x, y, vx, vy) points, shape (N, 4).

    Returns:
        Array of shape (n_species, N).
    """
    n_ranks = len(local_models)
    if len(local_bounds) != n_ranks:
        raise ValueError("local_models and local_bounds must have the same length")
    n_species = len(local_models[0])

    def _in_bounds(z, lo, hi):
        # `_assemble_global_grid` (evaluate_compression.py) rounds bounds to 9
        # decimals before rebuilding grid coordinates from them, so a query
        # point at the true edge can land a hair outside [lo, hi]; tolerate
        # that without opening a real overlap band.
        eps = 1e-9 * jnp.maximum(hi - lo, 1.0)
        return (z >= lo - eps) & (z <= hi + eps)

    x, y, vx, vy = query_points[:, 0], query_points[:, 1], query_points[:, 2], query_points[:, 3]

    out = jnp.zeros((n_species, query_points.shape[0]))
    for irank, (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max) in enumerate(local_bounds):
        mask = (
            _in_bounds(x, x_min, x_max)
            & _in_bounds(y, y_min, y_max)
            & _in_bounds(vx, vx_min, vx_max)
            & _in_bounds(vy, vy_min, vy_max)
        )

        x_n = (x - x_min) / jnp.maximum(x_max - x_min, 1e-12)
        y_n = (y - y_min) / jnp.maximum(y_max - y_min, 1e-12)
        vx_n = 2.0 * (vx - vx_min) / jnp.maximum(vx_max - vx_min, 1e-12) - 1.0
        vy_n = 2.0 * (vy - vy_min) / jnp.maximum(vy_max - vy_min, 1e-12) - 1.0
        coords = jnp.stack([x_n, y_n, vx_n, vy_n], axis=-1)

        for isp in range(n_species):
            pred = jax.vmap(local_models[irank][isp])(coords).squeeze(-1)
            out = out.at[isp].add(jnp.where(mask, pred, 0.0))

    return out