
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

from scimba_jax.nonlinear_approximation.networks.mlp import MLP
from scimba_jax.nonlinear_approximation.optimizers.optimizers import (ScimbaAdam, ScimbaLBfgs)

from Compressor import Compressor 

jax.config.update("jax_enable_x64", True)

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
            layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))
        self.layers = tuple(layers)

    def __call__(self, x_input: jnp.ndarray) -> jnp.ndarray:
        x = jnp.sin(self.omega_0 * self.layers[0](x_input))
        for layer in self.layers[1:-1]:
            x = jnp.sin(layer(x))
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
        
        #Phase 1: ADAM, mini-batches
        adam_opt = ScimbaAdam(model, _losses_function, learning_rate=self.lr)
        for i in range(self.max_iters):
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.choice(subkey, total_points, shape=(self.batch_size,), replace=False)
            batch = (inputs[batch_idx], targets[batch_idx])
            
            loss_dict, model, adam_opt = adam_opt.update(model, batch)
            loss_val = float(loss_dict["total"])
            loss_history.append(loss_val)
            
            if self.verbose and i % 100 == 0:
                 print(f"  [INR/{self.arch}][ADAM] iter {i:4d} - loss: {loss_val:.2e}")
            if loss_val < self.threshold:
                if self.verbose:
                    print(f"  [INR/{self.arch}][ADAM] early convergence at iter {i}")
                break
        
        #Phase 2: L-BFGS, full-batch
        full_batch = (inputs, targets)
        lbfgs_opt = ScimbaLBfgs(model, _losses_function)
        for i in range(self.lbfgs_iters):
            loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
            loss_val = float(loss_dict["total"])
            loss_history.append(loss_val)
            
            if self.verbose and (i % 10 == 0 or i == self.lbfgs_iters - 1):
                print(f"  [INR/{self.arch}][L-BFGS] iter {i:3d} - loss: {loss_val:.2e}")
            if loss_val < self.threshold:
                if self.verbose:
                    print(f"  [INR/{self.arch}][L-BFGS] convergence at iter {i}")
                break
            
        return model, jnp.array(loss_history)
    
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
        
        for isp in range(n_species):
            targets = f[isp].reshape(-1, 1)
            key, subkey = jax.random.split(key)
            t0 = time.perf_counter()
            model, loss_hist = self._fit_on_species(inputs, targets, subkey, warm_models[isp])
            t1 = time.perf_counter()
            
            if self.verbose:
                print(
                    f"[INR/{self.arch}] species {isp}: final loss "
                    f"{float(loss_hist[-1]):.2e} ({t1 - t0:.2f}s)"
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
            "final_loss_per_species": [float(h[-1]) for h in self.loss_histories] if self.loss_histories else None,
            "n_train_steps_per_species": [int(h.shape[0]) for h in self.loss_histories] if self.loss_histories else None,
        }

# Compressor (online / in-situ)

class OnlineNeuralNetworkCompressor:
    """In-situ INR compressor for one rank's local fdistribu[species, x, y, vx, vy] chunk.

    Implements the array-in/array-out online interface used by
    compression_diagnostics.py's apply_online_compression (same contract as
    RandomNoiseCompressor.compress_decompress_array). pycal already
    decomposes the domain across MPI ranks, so each rank calling this
    compressor on its own local chunk *is* the "parallel local solves"
    picture: one small NN f_theta^(i)(z) fit directly on subdomain Omega_i,
    with no cross-rank communication needed for the round trip itself.

    Coordinates are normalized to the LOCAL chunk's own unit cell
    ([0,1) for x,y, [-1,1] for vx,vy) since apply_online_compression only
    receives the local array -- not the chunk's placement in the global
    mesh. If the caller also passes `local_bounds` (the chunk's physical
    bounding box within the global domain, e.g. derived from
    local_fdistribu_starts + the global mesh arrays), it is stored and
    reported in the metrics so a downstream tool can map each rank's unit
    cell back to physical space and reassemble a smooth global field --
    see `assemble_global_field` below, which implements
    f_theta(z, t) = sum_i omega_i(z) f_theta^(i)(z, t).

    Since this runs inside the live timestepping loop, training must be
    cheap: the per-species models and ADAM states persist across calls
    (warm start), so only `refine_iters` steps are needed after the first
    call for a given local chunk shape (`warm_iters` steps, cold start).
    """

    method_name = "OnlineNeuralNetwork"

    def __init__(
        self,
        arch: str = "periodic_siren_small_32",
        lr: float = 1e-3,
        warm_iters: int = 200,
        refine_iters: int = 20,
        batch_size: Optional[int] = None,
        threshold: float = 1e-8,
        seed: int = 42,
        verbose: bool = False,
    ):
        if arch not in AVAILABLE_INR_ARCHS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {AVAILABLE_INR_ARCHS}")

        self.arch = arch
        self.lr = float(lr)
        self.warm_iters = int(warm_iters)
        self.refine_iters = int(refine_iters)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.verbose = bool(verbose)

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
            f"warm_iters={self.warm_iters}, refine_iters={self.refine_iters})"
        )

    @staticmethod
    def _build_local_inputs(nx: int, ny: int, nvx: int, nvy: int) -> jnp.ndarray:
        x = jnp.linspace(0.0, 1.0, nx, endpoint=False)
        y = jnp.linspace(0.0, 1.0, ny, endpoint=False)
        vx = jnp.linspace(-1.0, 1.0, nvx, endpoint=True)
        vy = jnp.linspace(-1.0, 1.0, nvy, endpoint=True)
        Xg, Yg, VXg, VYg = jnp.meshgrid(x, y, vx, vy, indexing="ij")
        return jnp.stack([Xg.ravel(), Yg.ravel(), VXg.ravel(), VYg.ravel()], axis=-1)

    def _fit_one_species(self, model, opt, inputs: jnp.ndarray, targets: jnp.ndarray, n_iters: int):
        total_points = inputs.shape[0]
        bs = min(self.batch_size, total_points) if self.batch_size is not None else total_points
        loss_val = None
        for _ in range(n_iters):
            if bs < total_points:
                self._key, subkey = jax.random.split(self._key)
                idx = jax.random.choice(subkey, total_points, shape=(bs,), replace=False)
                batch = (inputs[idx], targets[idx])
            else:
                batch = (inputs, targets)

            loss_dict, model, opt = opt.update(model, batch)
            loss_val = float(loss_dict["total"])
            if loss_val < self.threshold:
                break
        return model, opt, loss_val

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
        n_iters = self.warm_iters if cold_start else self.refine_iters

        t0 = time.perf_counter()
        final_losses = []
        for isp in range(n_species):
            targets = f[isp].reshape(-1, 1)
            model, opt, loss_val = self._fit_one_species(
                self.models[isp], self._opts[isp], inputs, targets, n_iters
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
            print(
                f"[OnlineINR/{self.arch}] rank {rank}: {tag} fit ({n_iters} iters) "
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
                "warm_iters": self.warm_iters,
                "refine_iters": self.refine_iters,
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
            payload[f"target_{isp}"] = self._last_local_array[isp]

        np.savez_compressed(path, **payload)


def load_online_params(path: str) -> dict:
    """Reload a payload written by `OnlineNeuralNetworkCompressor.save_params`.

    Returns a dict with keys: arch, rank, iter, local_shape, local_bounds,
    models (list of eqx.Module, one per species, ready to evaluate on the
    unit-cell grid built by `OnlineNeuralNetworkCompressor._build_local_inputs`),
    and target (np.ndarray, shape (n_species, nx, ny, nvx, nvy) -- the local
    data the models were fit on).
    """
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        n_species = int(metadata["n_species"])
        arch = metadata["arch"]

        key = jax.random.PRNGKey(0)
        models = []
        targets = []
        for isp in range(n_species):
            template = get_inr_model(arch, key)
            _, unravel_fn = ravel_pytree(template)
            flat = jnp.asarray(payload[f"weights_{isp}"])
            models.append(unravel_fn(flat))
            targets.append(np.asarray(payload[f"target_{isp}"]))

    local_bounds = metadata.get("local_bounds")
    return {
        "arch": arch,
        "rank": metadata.get("rank"),
        "iter": metadata.get("iter"),
        "local_shape": tuple(metadata["local_shape"]),
        "local_bounds": tuple(local_bounds) if local_bounds is not None else None,
        "models": models,
        "target": np.stack(targets),
    }

# Global assembly (offline reconstruction / visualization utility)

def assemble_global_field(
    local_models: list,
    local_bounds: list,
    query_points: jnp.ndarray,
    overlap_frac: float = 0.1,
) -> jnp.ndarray:
    """Reassemble a global field from per-rank local INRs via a partition of unity.

    Implements f_theta(z, t) = sum_i omega_i(z) f_theta^(i)(z, t): each rank i
    contributes the local model(s) it trained online on its own subdomain
    Omega_i (see OnlineNeuralNetworkCompressor), and omega_i is a smooth
    partition-of-unity weight built from `local_bounds[i]`, cosine-feathered
    over `overlap_frac` of each subdomain's width near its edges and zero
    outside it.

    This is a post-hoc reconstruction/visualization utility, not part of the
    online round trip -- during the simulation each rank only ever writes
    its own reconstructed chunk back in place. It does not handle periodic
    wrap-around at the global x/y domain edges; ranks are assumed to tile
    the interior of the domain.

    Args:
        local_models: local_models[i] is the list of per-species eqx.Module
            models trained by rank i's OnlineNeuralNetworkCompressor.
        local_bounds: local_bounds[i] is rank i's physical bounding box
            (x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max).
        query_points: physical (x, y, vx, vy) points, shape (N, 4).
        overlap_frac: fraction of each subdomain's width used for feathering
            near its edges (matches the "Overlap" bands in the decomposition
            picture).

    Returns:
        Array of shape (n_species, N).
    """
    n_ranks = len(local_models)
    if len(local_bounds) != n_ranks:
        raise ValueError("local_models and local_bounds must have the same length")
    n_species = len(local_models[0])

    def _ramp(z, lo, hi):
        # local_bounds tile the domain edge-to-edge with no ghost overlap (each
        # rank's chunk is exclusive), so the weight must extend *past* its own
        # [lo, hi] by a margin `o` into the neighbour's territory -- otherwise
        # adjacent ramps both hit zero exactly at the shared edge and the raw
        # weight sum collapses to 0 there. Weight is 1 throughout [lo, hi],
        # cosine-decays to 0 over a margin `o` on each side beyond that.
        width = jnp.maximum(hi - lo, 1e-12)
        o = jnp.maximum(overlap_frac * width, 1e-12)
        left = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip((z - (lo - o)) / o, 0.0, 1.0)))
        right = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(((hi + o) - z) / o, 0.0, 1.0)))
        return jnp.minimum(left, right)

    x, y, vx, vy = query_points[:, 0], query_points[:, 1], query_points[:, 2], query_points[:, 3]

    weights = []
    local_coords_per_rank = []
    for x_min, x_max, y_min, y_max, vx_min, vx_max, vy_min, vy_max in local_bounds:
        w = _ramp(x, x_min, x_max) * _ramp(y, y_min, y_max) * _ramp(vx, vx_min, vx_max) * _ramp(vy, vy_min, vy_max)
        weights.append(w)

        x_n = (x - x_min) / jnp.maximum(x_max - x_min, 1e-12)
        y_n = (y - y_min) / jnp.maximum(y_max - y_min, 1e-12)
        vx_n = 2.0 * (vx - vx_min) / jnp.maximum(vx_max - vx_min, 1e-12) - 1.0
        vy_n = 2.0 * (vy - vy_min) / jnp.maximum(vy_max - vy_min, 1e-12) - 1.0
        local_coords_per_rank.append(jnp.stack([x_n, y_n, vx_n, vy_n], axis=-1))

    weights = jnp.stack(weights, axis=0)  # (n_ranks, N)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=0, keepdims=True), 1e-12)

    out = jnp.zeros((n_species, query_points.shape[0]))
    for irank in range(n_ranks):
        coords = local_coords_per_rank[irank]
        for isp in range(n_species):
            pred = jax.vmap(local_models[irank][isp])(coords).squeeze(-1)
            out = out.at[isp].add(weights[irank] * pred)

    return out