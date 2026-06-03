"""Neural networks in JAX (float64) for 4D fdistribu regression."""

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm, trange
from jax import value_and_grad
from jax.scipy.optimize import minimize
from jax.tree_util import tree_flatten, tree_unflatten

jax.config.update("jax_enable_x64", True)

ARCHITECTURES = ("siren", "mlp")
MLP_ACTIVATIONS = ("sin", "tanh")


def init_model(
    key,
    layer_sizes,
    lx,
    ly,
    arch="siren",
    activation="tanh",
    omega_0=30.0,
):
    """Build model: raw (x, y, vx, vy) input; Lx, Ly stored on the model."""
    arch = arch.lower()
    activation = activation.lower()

    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {ARCHITECTURES}, got {arch!r}")
    if arch == "mlp" and activation not in MLP_ACTIVATIONS:
        raise ValueError(f"activation must be one of {MLP_ACTIVATIONS} for mlp, got {activation!r}")

    if arch == "siren":
        layers = _init_siren_layers(key, layer_sizes, omega_0)
    else:
        layers = _init_mlp_layers(key, layer_sizes)

    return {
        "arch": arch,
        "activation": activation if arch == "mlp" else "sin",
        "lx": jnp.asarray(lx, dtype=jnp.float64),
        "ly": jnp.asarray(ly, dtype=jnp.float64),
        "omega_0": omega_0,
        "layers": layers,
    }


def _init_siren_layers(key, layer_sizes, omega_0=30.0):
    """SIREN init: first layer scaled by omega_0."""
    layers = []
    keys = jax.random.split(key, len(layer_sizes) - 1)

    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, b_key = jax.random.split(k)
        bound = jnp.sqrt(6.0 / n_in) / omega_0 if len(layers) == 0 else jnp.sqrt(6.0 / n_in)
        w = jax.random.uniform(w_key, (n_in, n_out), minval=-bound, maxval=bound)
        b = jax.random.uniform(b_key, (n_out,), minval=-bound, maxval=bound)
        layers.append({"w": w, "b": b})

    return layers


def _init_mlp_layers(key, layer_sizes):
    """Standard Xavier uniform init for a plain MLP."""
    layers = []
    keys = jax.random.split(key, len(layer_sizes) - 1)

    for k, (n_in, n_out) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        bound = jnp.sqrt(6.0 / (n_in + n_out))
        w = jax.random.uniform(k, (n_in, n_out), minval=-bound, maxval=bound)
        b = jnp.zeros((n_out,), dtype=jnp.float64)
        layers.append({"w": w, "b": b})

    return layers


def _hidden_activation(x, model, layer_index):
    arch = model["arch"]

    if arch == "siren":
        omega = model["omega_0"] if layer_index == 0 else 1.0
        return jnp.sin(omega * x)

    if model["activation"] == "sin":
        return jnp.sin(x)

    return jnp.tanh(x)


def forward(model, coords):
    x = jnp.asarray(coords, dtype=jnp.float64)
    layers = model["layers"]

    for i, layer in enumerate(layers):
        x = x @ layer["w"] + layer["b"]
        if i < len(layers) - 1:
            x = _hidden_activation(x, model, i)

    return x[..., 0]


def mse_loss(model, coords, targets):
    pred = forward(model, coords)
    return jnp.mean((pred - targets) ** 2)


def _loss_for_layers(layers, model, coords, targets):
    return mse_loss({**model, "layers": layers}, coords, targets)


def train_adam(model, coords, targets, n_steps, lr=1e-4, verbose=True):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model["layers"])
    layers = model["layers"]
    losses = []

    steps = trange(n_steps, desc="Adam", disable=not verbose)
    for _ in steps:
        loss, grads = value_and_grad(_loss_for_layers)(layers, model, coords, targets)
        updates, opt_state = optimizer.update(grads, opt_state, layers)
        layers = optax.apply_updates(layers, updates)
        loss_val = float(loss)
        losses.append(loss_val)
        if verbose:
            steps.set_postfix(loss=f"{loss_val:.4e}")

    if verbose:
        print(f"  Adam done, loss = {losses[-1]:.6e}")

    return {**model, "layers": layers}, losses


def train_bfgs(model, coords, targets, n_steps, verbose=True):
    layers = model["layers"]
    flat0, treedef = tree_flatten(layers)
    x0 = jnp.concatenate([leaf.ravel() for leaf in flat0])
    loss0 = float(_loss_for_layers(layers, model, coords, targets))
    bfgs_losses = []
    bfgs_bar = None

    def _record_loss(loss_val):
        val = float(loss_val)
        bfgs_losses.append(val)
        if bfgs_bar is not None:
            if bfgs_bar.n >= bfgs_bar.total:
                bfgs_bar.total = bfgs_bar.n + 1
            bfgs_bar.update(1)
            bfgs_bar.set_postfix(loss=f"{val:.4e}")

    if verbose:
        print(f"BFGS (max {n_steps} iterations), initial loss = {loss0:.6e} ...")
        bfgs_bar = tqdm(total=n_steps, desc="BFGS", unit="eval", dynamic_ncols=True)

    def loss_flat(p):
        new_layers = tree_unflatten(treedef, _unflatten_vector(p, flat0))
        loss = _loss_for_layers(new_layers, model, coords, targets)
        jax.debug.callback(_record_loss, loss)
        return loss

    result = minimize(
        loss_flat,
        x0,
        method="BFGS",
        options={"maxiter": n_steps},
    )

    if bfgs_bar is not None:
        bfgs_bar.close()

    layers = tree_unflatten(treedef, _unflatten_vector(result.x, flat0))
    loss_final = float(result.fun)

    if verbose:
        nit = getattr(result, "nit", None)
        nfev = getattr(result, "nfev", len(bfgs_losses))
        print(
            f"  BFGS done, loss = {loss_final:.6e}"
            f", nit = {nit}, loss evaluations = {nfev}"
        )

    if not bfgs_losses:
        bfgs_losses = [loss0, loss_final]
    elif abs(bfgs_losses[-1] - loss_final) > 1e-15 * max(1.0, abs(loss_final)):
        bfgs_losses.append(loss_final)

    return {**model, "layers": layers}, bfgs_losses


def _unflatten_vector(vector, template_leaves):
    flat_params = []
    offset = 0
    for leaf in template_leaves:
        size = leaf.size
        flat_params.append(vector[offset : offset + size].reshape(leaf.shape))
        offset += size
    return flat_params


def train(model, coords, targets, n_adam, n_bfgs, lr=1e-4, verbose=True):
    """Adam pre-training, then BFGS fine-tuning from the Adam result."""
    if verbose:
        label = model_label(model)
        print(f"Adam: {n_adam} steps (lr={lr:g}), architecture = {label}")
    model, adam_losses = train_adam(model, coords, targets, n_adam, lr=lr, verbose=verbose)
    model, bfgs_losses = train_bfgs(model, coords, targets, n_bfgs, verbose=verbose)
    if bfgs_losses and adam_losses and abs(bfgs_losses[0] - adam_losses[-1]) < 1e-12:
        bfgs_losses = bfgs_losses[1:]
    return model, adam_losses + bfgs_losses


def model_label(model):
    if model["arch"] == "siren":
        return f"siren (omega_0={model['omega_0']})"
    return f"mlp ({model['activation']})"


def predict(model, coords):
    return forward(model, jnp.asarray(coords))
