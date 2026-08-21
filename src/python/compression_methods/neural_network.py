import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
    "periodic_siren_small_32_l5",
    "periodic_siren_small_32_l8",
    "periodic_siren_small_16_l3",
    "periodic_siren_small_16_l5",
    "periodic_siren_small_16_l8",
]

# Choices for the polish-phase optimizer
AVAILABLE_POLISH_OPTIMIZERS = ["lbfgs", "gauss_newton"]

def get_inr_model(arch: str, key: jax.Array) -> eqx.Module:
    """Instantiate an INR architecture"""
    if arch == "periodic_siren_deep_128": return PeriodicSIRENScimbaINR([128]*5, 30.0, key)
    elif arch == "periodic_fourier_mlp_deep_128": return PeriodicFourierScimbaINR(16, [128]*5, 10.0, key)
    elif arch == "periodic_siren_small_32": return PeriodicSIRENScimbaINR([32]*3, 30.0, key)
    elif arch == "periodic_fourier_mlp_small_32": return PeriodicFourierScimbaINR(8, [32]*3, 10.0, key)
    elif arch == "periodic_siren_small_32_l5": return PeriodicSIRENScimbaINR([32]*5, 30.0, key)
    elif arch == "periodic_siren_small_32_l8": return PeriodicSIRENScimbaINR([32]*8, 30.0, key)
    elif arch == "periodic_siren_small_16_l3": return PeriodicSIRENScimbaINR([16]*3, 30.0, key)
    elif arch == "periodic_siren_small_16_l5": return PeriodicSIRENScimbaINR([16]*5, 30.0, key)
    elif arch == "periodic_siren_small_16_l8": return PeriodicSIRENScimbaINR([16]*8, 30.0, key)
    else:
        raise ValueError(f"Unknown INR architecture: {arch}. Available: {AVAILABLE_INR_ARCHS}")
    
# Training losses
@jax.jit
def _losses_function(model: eqx.Module, batch: tuple) -> dict:
    inputs, targets = batch
    predictions = jax.vmap(model)(inputs)
    mse = jnp.mean((predictions - targets) ** 2)

    return {"total":mse}

# Chunked losses for L-BFGS
def make_chunked_losses_function(chunk_size: int):
    """
    L-BFGS trains full-batch so that its curvature estimate sees a consistent, deterministic objective across iterations. 
    At high grid resolutions a single jax.vmap(model) over the
    entire batch needs more activation memory than fits on the GPU. This builds
    a drop-in replacement for `_losses_function` (same (model, batch) -> {"total"}
    contract, so it plugs directly into ScimbaLBfgs/ScimbaAdam) that processes the
    batch in fixed-size chunks via jax.lax.scan, each chunk wrapped in
    jax.checkpoint so the backward pass recomputes its activations instead of
    keeping every chunk's activations resident at once. Peak memory becomes
    O(chunk_size) instead of O(N); the loss and gradient stay mathematically
    identical to the unchunked full-batch computation (MSE is a plain average of
    independent per-point terms, so summing it in chunks is exact, not an
    approximation) -- see src/tests/chunked_lbfgs_experiment.py for the
    correctness check (chunked vs unchunked gradient differs by ~1e-17, float64
    epsilon) and the real-grid OOM validation this was ported from.
    """
    
    @jax.jit
    def _chunked_losses_function(model: eqx.Module, batch: tuple) -> dict:
        inputs, targets = batch
        n = inputs.shape[0]
        n_chunks = -(-n // chunk_size)  # ceil division, static python int
        n_pad = n_chunks * chunk_size - n

        if n_pad > 0:
            pad_in = jnp.zeros((n_pad,) + inputs.shape[1:], dtype=inputs.dtype)
            pad_t = jnp.zeros((n_pad,) + targets.shape[1:], dtype=targets.dtype)
            inputs_p = jnp.concatenate([inputs, pad_in], axis=0)
            targets_p = jnp.concatenate([targets, pad_t], axis=0)
            mask = jnp.concatenate([
                jnp.ones((n,), dtype=inputs.dtype),
                jnp.zeros((n_pad,), dtype=inputs.dtype),
            ])
        else:
            inputs_p, targets_p, mask = inputs, targets, jnp.ones((n,), dtype=inputs.dtype)

        inputs_c = inputs_p.reshape((n_chunks, chunk_size) + inputs.shape[1:])
        targets_c = targets_p.reshape((n_chunks, chunk_size) + targets.shape[1:])
        mask_c = mask.reshape((n_chunks, chunk_size))

        @jax.checkpoint
        def chunk_step(carry, xs):
            x_chunk, t_chunk, m_chunk = xs
            pred = jax.vmap(model)(x_chunk)
            sq_err = jnp.sum(((pred - t_chunk) ** 2).squeeze(-1) * m_chunk)
            return carry + sq_err, None

        total_sse, _ = jax.lax.scan(chunk_step, jnp.zeros((), dtype=targets.dtype),
                                     (inputs_c, targets_c, mask_c))
        mse = total_sse / n
        return {"total": mse}

    return _chunked_losses_function

# Chunked Gauss-Newton normal equations: J^T J, J^T r and sum(r^2) are all
# plain sums of independent per-point contributions, so accumulating them
# chunk by chunk (jax.lax.scan, mirroring make_chunked_losses_function above)
# is mathematically exact -- not an approximation of the unchunked result.
def _chunked_gn_normal_equations(residual_fn, p, coords, target, chunk_size):
    """Peak-memory-bounded (J^T J, J^T r, mean(r^2)) for one Gauss-Newton batch.

    jax.jacfwd(residual_fn) materializes O(chunk_size x n_params x hidden_width)
    intermediate activations for whatever batch it's handed -- by far the
    dominant memory cost of Gauss-Newton, much bigger than the O(n_params^2)
    J^T J matrix itself. Computing it chunk_size points at a time instead of
    n_map points at once bounds peak memory by chunk_size independent of
    n_map, letting n_map be set purely for a well-posed batch (n_map >>
    n_params) rather than for whatever fits in GPU memory that run -- the
    same role lbfgs_chunk_size already plays for L-BFGS's full-grid loss.
    """
    n = coords.shape[0]
    n_params = p.shape[0]
    n_chunks = -(-n // chunk_size)  # ceil division, static python int
    n_pad = n_chunks * chunk_size - n

    if n_pad > 0:
        pad_c = jnp.zeros((n_pad,) + coords.shape[1:], dtype=coords.dtype)
        pad_t = jnp.zeros((n_pad,) + target.shape[1:], dtype=target.dtype)
        coords_p = jnp.concatenate([coords, pad_c], axis=0)
        target_p = jnp.concatenate([target, pad_t], axis=0)
        mask = jnp.concatenate([
            jnp.ones((n,), dtype=coords.dtype),
            jnp.zeros((n_pad,), dtype=coords.dtype),
        ])
    else:
        coords_p, target_p, mask = coords, target, jnp.ones((n,), dtype=coords.dtype)

    coords_c = coords_p.reshape((n_chunks, chunk_size) + coords.shape[1:])
    target_c = target_p.reshape((n_chunks, chunk_size) + target.shape[1:])
    mask_c = mask.reshape((n_chunks, chunk_size))

    def masked_residual(p_, c_chunk, t_chunk, m_chunk):
        # Zeroing padded rows here (rather than masking coords/target directly)
        # also zeroes their row of J = d(residual)/dp under jacfwd, via the
        # chain rule d(m_i * r_i)/dp = m_i * dr_i/dp -- so padded points
        # contribute exactly zero to JTJ/grad/sse below, not an approximation.
        return residual_fn(p_, c_chunk, t_chunk) * m_chunk

    def chunk_step(carry, xs):
        JTJ_acc, grad_acc, sse_acc = carry
        c_chunk, t_chunk, m_chunk = xs
        r_chunk = masked_residual(p, c_chunk, t_chunk, m_chunk)
        J_chunk = jax.jacfwd(masked_residual)(p, c_chunk, t_chunk, m_chunk)
        return (
            JTJ_acc + J_chunk.T @ J_chunk,
            grad_acc + J_chunk.T @ r_chunk,
            sse_acc + jnp.sum(r_chunk ** 2),
        ), None

    init = (
        jnp.zeros((n_params, n_params), dtype=p.dtype),
        jnp.zeros((n_params,), dtype=p.dtype),
        jnp.zeros((), dtype=p.dtype),
    )
    (JTJ, grad, sse), _ = jax.lax.scan(chunk_step, init, (coords_c, target_c, mask_c))
    return JTJ, grad, sse / n


# Gauss-Newton polish optimizer (alternative to L-BFGS, used after ADAM warmup)
def train_map_gn(
    n_map: int,
    make_data,
    params: eqx.Module,
    n_iterations: int = 50,
    init_damping: float = 1e-2,
    chunk_size: int = 2000,
) -> tuple[eqx.Module, jnp.ndarray]:
    """Fit `params` (an INR model) to data produced by `make_data(n_map, key) ->
    (coords, targets, new_key)`, via damped Gauss-Newton.

    chunk_size bounds the peak memory of the J^T J / J^T r computation (see
    _chunked_gn_normal_equations) independent of n_map -- so n_map can be set
    purely for a well-posed batch (n_map >> n_params, needed for Gauss-Newton
    to converge reliably instead of overfitting a too-small random batch)
    without needing to fit the whole batch's Jacobian in GPU memory at once.
    chunk_size itself still needs to leave room for one chunk's Jacobian
    alongside whatever else shares the GPU (e.g. the concurrently-running
    physics simulation) -- 2000 is the value validated to fit under that
    contention for periodic_siren_small_32 (~2369 params).

    IMPORTANT -- only tractable on small architectures regardless of
    chunk_size: every iteration forms the dense Gauss-Newton matrix J^T J
    (size n_params x n_params), independent of batch size or chunking. For
    periodic_siren_deep_128 (~67k params) this alone needs ~36GB and reliably
    OOMs, confirmed even with a batch as small as n_map=100 -- do not use
    polish_optimizer="gauss_newton" with the "_deep_128" architectures.

    Returns (best_model, loss_history), loss_history of shape (n_iterations,).
    best_model corresponds to the LOWEST loss encountered across all
    n_iterations
    """
    key = jax.random.PRNGKey(42)
    flat_params, unflatten = ravel_pytree(params)

    def residual_fn(p, coords, t):
        pred = jax.vmap(unflatten(p))(coords)
        return (pred - t).reshape(-1)

    # carry: (curr_p, damping, key, best_p, best_loss)
    initial_state = (flat_params, init_damping, key, flat_params, jnp.inf)

    def step_fn(state, _):
        curr_p, damping, step_key, best_p, best_loss = state
        coords_train, target, step_key = make_data(n_map, step_key)
        JTJ, grad, current_loss = _chunked_gn_normal_equations(
            residual_fn, curr_p, coords_train, target, chunk_size
        )
        num_params = JTJ.shape[0]
        step = jnp.linalg.solve(JTJ + damping * jnp.eye(num_params), -grad)
        direction_derivative = jnp.dot(grad, step)

        def ls_cond(ls_state):
            alpha, count, _, loss_cand = ls_state
            return (loss_cand > current_loss + 1e-4 * alpha * direction_derivative) & (
                count < 8
            )

        def ls_body(ls_state):
            alpha, count, _, _ = ls_state
            new_alpha = alpha * 0.5
            new_p = curr_p + new_alpha * step
            new_loss = jnp.mean(residual_fn(new_p, coords_train, target) ** 2)
            return new_alpha, count + 1, new_p, new_loss

        init_p_cand = curr_p + 1.0 * step
        init_l_cand = jnp.mean(residual_fn(init_p_cand, coords_train, target) ** 2)
        _, ls_count, final_p, final_loss = jax.lax.while_loop(
            ls_cond, ls_body, (1.0, 0, init_p_cand, init_l_cand)
        )
        new_damping = jnp.where(
            ls_count >= 8,
            damping * 10.0,
            jnp.where(ls_count == 0, damping / 2.0, damping),
        )
        new_damping = jnp.clip(new_damping, 1e-5, 1e2)
        success = final_loss < current_loss
        actual_p = jnp.where(success, final_p, curr_p)
        actual_loss = jnp.where(success, final_loss, current_loss)

        is_best = actual_loss < best_loss
        new_best_p = jnp.where(is_best, actual_p, best_p)
        new_best_loss = jnp.where(is_best, actual_loss, best_loss)

        return (actual_p, new_damping, step_key, new_best_p, new_best_loss), actual_loss

    xs = jnp.arange(n_iterations)
    (_, _, _, best_p, _), loss_hist = jax.lax.scan(step_fn, initial_state, xs)
    return unflatten(best_p), loss_hist


# Reconstruction: unlike training, this always evaluates the *entire* grid, so
# (unlike batch_size, which only bounds training mini-batches) it needs its own
# chunking to keep peak memory bounded.
_DEFAULT_RECON_CHUNK_SIZE = 50_000


def _vmap_in_chunks(model: eqx.Module, inputs: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """jax.vmap(model) over inputs in chunks of chunk_size, concatenated back together."""
    total = inputs.shape[0]
    chunk_size = min(chunk_size, total)
    outputs = [jax.vmap(model)(inputs[i:i + chunk_size]) for i in range(0, total, chunk_size)]
    return jnp.concatenate(outputs, axis=0)


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
        lr_decay_alpha: float = 0.01,
        max_iters: int = 2000,
        warm_max_iters: Optional[int] = None,
        batch_size: int = 2000,
        polish_optimizer: str = "lbfgs",
        lbfgs_iters: int = 50,
        lbfgs_chunk_size: int = 200_000,
        gn_iters: int = 50,
        gn_n_map: int = 8000,
        gn_init_damping: float = 1e-2,
        gn_chunk_size: int = 2000,
        clear_cache_every: int = 5,
        threshold: float = 1e-8,
        seed: int = 42,
        warm_start_payload: Optional[str] = None,
        verbose: bool = True,
    ):
        if arch not in AVAILABLE_INR_ARCHS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {AVAILABLE_INR_ARCHS}")
        if polish_optimizer not in AVAILABLE_POLISH_OPTIMIZERS:
            raise ValueError(
                f"Unknown polish_optimizer {polish_optimizer!r}. Available: {AVAILABLE_POLISH_OPTIMIZERS}"
            )
        if polish_optimizer == "gauss_newton" and "deep_128" in arch:
            raise ValueError(
                f"polish_optimizer='gauss_newton' is not usable with arch={arch!r}: its dense "
                "Gauss-Newton matrix (n_params x n_params) needs ~36G  and reliably OOMs "
                "on this architecture, independent of batch size (gn_n_map). Use a "
                "'_small_32' architecture, or polish_optimizer='lbfgs' instead."
            )

        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.vx_min, self.vx_max = float(vx_min), float(vx_max)
        self.vy_min, self.vy_max = float(vy_min), float(vy_max)

        self.arch = arch
        self.lr = float(lr)
        self.lr_decay_alpha = float(lr_decay_alpha)
        self.max_iters = int(max_iters)
        self.warm_max_iters = int(warm_max_iters) if warm_max_iters is not None else self.max_iters
        self.batch_size = int(batch_size)
        self.polish_optimizer = polish_optimizer
        self.lbfgs_iters = int(lbfgs_iters)
        self.lbfgs_chunk_size = int(lbfgs_chunk_size)
        self.gn_iters = int(gn_iters)
        self.gn_n_map = int(gn_n_map)
        self.gn_init_damping = float(gn_init_damping)
        self.gn_chunk_size = int(gn_chunk_size)
        # Built once and reused across every _fit_on_species call (all species,
        # all timesteps): same python function object each time -> jax.jit's
        # compilation cache is hit after the first call instead of retracing.
        self._chunked_losses_fn = make_chunked_losses_function(self.lbfgs_chunk_size)

        self.clear_cache_every = int(clear_cache_every)
        self._n_compress_calls = 0

        self.threshold = float(threshold)
        self.seed = int(seed)
        self.warm_start_payload = warm_start_payload
        self.verbose = bool(verbose)

        super().__init__(
            method_name=self.method_name,
            arch=self.arch,
            lr=self.lr,
            lr_decay_alpha=self.lr_decay_alpha,
            max_iters=self.max_iters,
            warm_max_iters=self.warm_max_iters,
            batch_size=self.batch_size,
            polish_optimizer=self.polish_optimizer,
            lbfgs_iters=self.lbfgs_iters,
            lbfgs_chunk_size=self.lbfgs_chunk_size,
            gn_iters=self.gn_iters,
            gn_n_map=self.gn_n_map,
            gn_init_damping=self.gn_init_damping,
            gn_chunk_size=self.gn_chunk_size,
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
        is_warm = warm_model is not None
        if is_warm:
            model = warm_model
        else:
            key, subkey = jax.random.split(key)
            model = get_inr_model(self.arch, subkey)
        
        total_points = inputs.shape[0]
        loss_history = []
        tag = "warm" if is_warm else "cold"
        
        best_model, best_loss = model, float("inf")

        if is_warm and self.polish_optimizer == "gauss_newton":
            n_iters_adam = self.warm_max_iters
        else:
            n_iters_adam = self.max_iters

        #Phase 1: ADAM, mini-batches. On a warm-started species only, lr decays over the run instead of staying constant
        if is_warm:
            learning_rate = optax.cosine_decay_schedule(
                init_value=self.lr, decay_steps=n_iters_adam, alpha=self.lr_decay_alpha
            )
        else:
            learning_rate = self.lr
        adam_opt = ScimbaAdam(model, _losses_function, learning_rate=learning_rate)
        pbar = _training_progress(n_iters_adam, f"[INR/{self.arch}][ADAM]", self.verbose)
        for i in pbar:
            key, subkey = jax.random.split(key)
            batch_idx = jax.random.choice(subkey, total_points, shape=(self.batch_size,), replace=False)
            batch = (inputs[batch_idx], targets[batch_idx])
            
            
            loss_dict, model, adam_opt = adam_opt.update(model, batch)
            loss_val = float(loss_dict["total"])
            loss_history.append(loss_val)

            if loss_val < best_loss:
                best_model, best_loss = model, loss_val

            if self.verbose and i % 100 == 0:
                 print(f"  [INR/{self.arch}][ADAM/{tag}] iter {i:4d} - loss: {loss_val:.2e}")
            if loss_val < self.threshold:
                pbar.write(f"  [INR/{self.arch}][ADAM] early convergence at iter {i}")
                break

        #Phase 2: polish -- L-BFGS (full-batch) or Gauss-Newton (mini-batch), per self.polish_optimizer
        #Resumes from ADAM's best point (best_model), not wherever ADAM happened to end up
        model = best_model

        if self.polish_optimizer == "lbfgs":
            full_batch = (inputs, targets)
            lbfgs_opt = ScimbaLBfgs(model, self._chunked_losses_fn)
            pbar = _training_progress(self.lbfgs_iters, f"[INR/{self.arch}][L-BFGS]", self.verbose)
            for i in pbar:
                loss_dict, model, lbfgs_opt = lbfgs_opt.update(model, full_batch)
                loss_val = float(loss_dict["total"])
                loss_history.append(loss_val)
                pbar.set_postfix(loss=f"{loss_val:.2e}")

                if loss_val < best_loss:
                    best_model, best_loss = model, loss_val

                if self.verbose and i % 10 == 0:
                    print(f"  [INR/{self.arch}][L-BFGS/{tag}] iter {i:4d} - loss: {loss_val:.2e}")

                if loss_val < self.threshold:
                    pbar.write(f"  [INR/{self.arch}][L-BFGS] convergence at iter {i}")
                    break

        else:  # "gauss_newton"
            def make_data(n_map, k):
                k, subkey = jax.random.split(k)
                idx = jax.random.choice(subkey, total_points, shape=(n_map,), replace=(n_map > total_points))
                return inputs[idx], targets[idx], k

            if self.verbose:
                print(f"  [INR/{self.arch}][GaussNewton/{tag}] running {self.gn_iters} iterations "
                      f"(n_map={self.gn_n_map}, chunk_size={self.gn_chunk_size})...")
            # model returned here is already train_map_gn's own best-seen iterate
            # (not necessarily its last), so best_gn_loss below (the historical
            # min of gn_loss_hist) is exactly the loss of that returned model.
            model, gn_loss_hist = train_map_gn(
                self.gn_n_map, make_data, model,
                n_iterations=self.gn_iters, init_damping=self.gn_init_damping,
                chunk_size=self.gn_chunk_size,
            )
            gn_loss_hist = np.asarray(gn_loss_hist)
            loss_history.extend(gn_loss_hist.tolist())

            best_gn_loss = float(gn_loss_hist.min()) if gn_loss_hist.size else best_loss
            if best_gn_loss < best_loss:
                best_model, best_loss = model, best_gn_loss

            if self.verbose:
                print(f"  [INR/{self.arch}][GaussNewton/{tag}] done, best loss {best_gn_loss:.2e} "
                      f"(last loss {gn_loss_hist[-1]:.2e})")

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
        
        #periodic cache clearing to prevent llvm oom due to repeated jit compilatins
        self._n_compress_calls += 1
        if self.clear_cache_every > 0 and self._n_compress_calls % self.clear_cache_every == 0: # si clear_cache_every est supérieur à 0 et que le nombre d'appels de compression est un multiple de clear_cache_every, on efface les caches de compilation JAX pour éviter les problèmes de mémoire
            if self.verbose:
                print(f"[INR/{self.arch}] Clearing JAX compilation caches (compress call #{self._n_compress_calls})")
            jax.clear_caches()
        
        inputs = self._build_inputs(nx, ny, nvx, nvy)
        key = jax.random.PRNGKey(self.seed)
        
        if self.models:
            warm_models = self.models 
        else:
            warm_models = self._load_warm_start_models(n_species, key)
            
        new_models = []
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
            
            new_models.append(model)
            self.loss_histories.append(loss_hist)
        
        self.models = new_models
        return {"models": self.models, "grid_shape": (nx, ny, nvx, nvy)}
    
    def save_loss_histories(self, out_dir, timestep):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for isp, hist in enumerate(self.loss_histories):
            p = out_dir / f"loss_iter{timestep:05d}_sp{isp}_{self.arch}.npy"
            np.save(p, np.asarray(hist))
            paths.append(p)
        return paths
    
    def decompress_array(self, compressed: dict) -> jnp.ndarray:
        models = compressed["models"]
        nx, ny, nvx, nvy = compressed["grid_shape"]
        t0 = time.perf_counter()
        print(
            f"[INR/{self.arch}] Decompression started "
            f"(Number of species={len(models)}, grid=({nx}, {ny}, {nvx}, {nvy}))...",
            flush=True,
        )
        inputs = self._build_inputs(nx, ny, nvx, nvy)
        
        species_out = []
        for model in models:
            pred = _vmap_in_chunks(model, inputs, self.batch_size).reshape(nx, ny, nvx, nvy)
            species_out.append(pred)

        print(
            f"[INR/{self.arch}] Decompression finished in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
            
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
        polish_optimizer: str = "lbfgs",
        warm_iters_gn: int = 150,
        refine_iters_gn: int = 30,
        gn_n_map: int = 16000,
        gn_init_damping: float = 1e-2,
        gn_chunk_size: int = 2000,
        batch_size: Optional[int] = None,
        threshold: float = 1e-8,
        seed: int = 42,
        verbose: bool = False,
        debug_plot: bool = False,
        save_target: bool = False,
    ):
        if arch not in AVAILABLE_INR_ARCHS:
            raise ValueError(f"Unknown arch {arch!r}. Available: {AVAILABLE_INR_ARCHS}")
        
        if polish_optimizer not in AVAILABLE_POLISH_OPTIMIZERS:
            raise ValueError(
                f"Unknown polish_optimizer {polish_optimizer!r}. Available: {AVAILABLE_POLISH_OPTIMIZERS}"
            )
        if polish_optimizer == "gauss_newton" and "deep_128" in arch:
            raise ValueError(
                f"polish_optimizer='gauss_newton' is not usable with arch={arch!r}: its dense "
                "Gauss-Newton matrix (n_params x n_params) needs ~36G  and reliably OOMs "
                "on this architecture, independent of batch size (gn_n_map). Use a "
                "'_small_32' architecture, or polish_optimizer='lbfgs' instead."
            )

        self.arch = arch
        self.lr = float(lr)
        self.warm_iters_adam = int(warm_iters_adam)
        self.warm_iters_lbfgs = int(warm_iters_lbfgs)
        self.refine_iters_adam = int(refine_iters_adam)
        self.refine_iters_lbfgs = int(refine_iters_lbfgs)
        self.polish_optimizer = polish_optimizer
        self.warm_iters_gn = int(warm_iters_gn)
        self.refine_iters_gn = int(refine_iters_gn)
        self.gn_n_map = int(gn_n_map)
        self.gn_init_damping = float(gn_init_damping)
        self.gn_chunk_size = int(gn_chunk_size)
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
            f"warm_iters_adam={self.warm_iters_adam}, warm_iters_lbfgs={self.warm_iters_lbfgs}, warm_iters_gn={self.warm_iters_gn}, "
            f"refine_iters_adam={self.refine_iters_adam}, refine_iters_lbfgs={self.refine_iters_lbfgs}, refine_iters_gn={self.refine_iters_gn}, "
            f"polish_optimizer={self.polish_optimizer}, "
            f"gn_n_map={self.gn_n_map}, gn_init_damping={self.gn_init_damping}, gn_chunk_size={self.gn_chunk_size} )"
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
        self, model, opt, inputs: jnp.ndarray, targets: jnp.ndarray, n_iters_adam: int, n_iters_lbfgs: int, n_iters_gn: int,
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

        # Phase 2: L-BFGS or Gauss-Newton
        n_iters_polish = n_iters_lbfgs if self.polish_optimizer == "lbfgs" else n_iters_gn
        if n_iters_polish > 0 and (loss_val is None or loss_val >= self.threshold):
            if self.polish_optimizer == "lbfgs":
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
            
            else: #gauss_newton
                def make_data(n_map,k):
                    k, subkey = jax.random.split(k)
                    idx = jax.random.choice(subkey, total_points, shape=(n_map,), replace=(n_map > total_points))
                    return inputs[idx], targets[idx], k
                
                model, gn_loss_hist = train_map_gn(
                    self.gn_n_map, make_data, model,
                    n_iterations=n_iters_gn, init_damping=self.gn_init_damping, 
                    chunk_size=self.gn_chunk_size,
                )
                gn_loss_hist = np.asarray(gn_loss_hist)
                if gn_loss_hist.size:
                    best_gn_loss = float(gn_loss_hist.min())
                    loss_val = float(gn_loss_hist[-1])
                    if best_gn_loss < best_loss :
                        best_model, best_loss = model, best_gn_loss

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
        n_iters_gn = self.warm_iters_gn if cold_start else self.refine_iters_gn

        t0 = time.perf_counter()
        final_losses = []
        for isp in range(n_species):
            targets = f[isp].reshape(-1, 1)
            model, opt, loss_val = self._fit_one_species(
                self.models[isp], self._opts[isp], inputs, targets, n_iters_adam, n_iters_lbfgs, n_iters_gn,
                desc=f"[OnlineINR/{self.arch}][rank={rank}][sp={isp}]",
            )
            self.models[isp] = model
            self._opts[isp] = opt
            final_losses.append(loss_val)
        t1 = time.perf_counter()

        recon_chunk_size = self.batch_size or _DEFAULT_RECON_CHUNK_SIZE
        recon_species = [
            _vmap_in_chunks(self.models[isp], inputs, recon_chunk_size).reshape(nx, ny, nvx, nvy)
            for isp in range(n_species)
        ]
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
                "polish_optimizer": self.polish_optimizer,
                "warm_iters_gn": self.warm_iters_gn,
                "refine_iters_gn": self.refine_iters_gn,
                "gn_n_map": self.gn_n_map,
                "gn_init_damping": self.gn_init_damping,
                "gn_chunk_size": self.gn_chunk_size,
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
    chunk_size: int = _DEFAULT_RECON_CHUNK_SIZE,
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
        chunk_size: points per vmap call, to bound peak memory (query_points
            can span the entire global grid).

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
            pred = _vmap_in_chunks(local_models[irank][isp], coords, chunk_size).squeeze(-1)
            out = out.at[isp].add(jnp.where(mask, pred, 0.0))

    return out