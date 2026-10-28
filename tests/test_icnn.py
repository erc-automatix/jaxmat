"""Input convex neural networks."""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxmat.nn import ICNN, positive_param, shifted_softplus
from jaxmat.nn.icnn import ICNNSkip

jax.config.update("jax_enable_x64", True)

KEY = jax.random.PRNGKey(0)
MONOTONE = {"monotone": True, "free": False, "mixed": (True, False, True)}


def randomized(model, seed, std=1.5):
    """The same network with arbitrary raw parameters: convexity and monotonicity are
    structural, they must not depend on the initialization."""
    params, static = eqx.partition(model, eqx.is_array)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    leaves = [std * jax.random.normal(k, leaf.shape) for k, leaf in zip(keys, leaves, strict=True)]
    return eqx.combine(jax.tree.unflatten(treedef, leaves), static)


def unit_directions(n, dim, seed=9):
    x = jax.random.normal(jax.random.PRNGKey(seed), (n, dim))
    return x / jnp.linalg.norm(x, axis=1, keepdims=True)


@pytest.mark.parametrize("positive", ["softplus", "exp", "square"])
@pytest.mark.parametrize("monotone", MONOTONE.values(), ids=MONOTONE)
def test_convex_for_any_parameters(monotone, positive):
    xs = 2 * jax.random.normal(jax.random.PRNGKey(1), (64, 3))
    for seed in range(3):
        model = randomized(ICNN(3, [8, 8, 8], KEY, monotone=monotone, positive=positive), seed)
        eigenvalues = jax.vmap(lambda x: jnp.linalg.eigvalsh(jax.hessian(model)(x)))(xs)
        assert float(eigenvalues.min()) >= -1e-10 * max(1.0, float(jnp.abs(eigenvalues).max()))


@pytest.mark.parametrize(
    "activation", [shifted_softplus, jax.nn.softplus, jax.nn.relu], ids=lambda f: f.__name__
)
def test_midpoint_convexity(activation):
    """f((a + b)/2) <= (f(a) + f(b))/2, which also holds for non-smooth activations."""
    a, b = (3 * jax.random.normal(jax.random.PRNGKey(s), (200, 3)) for s in (2, 3))
    for seed in range(3):
        model = randomized(ICNN(3, [16, 16], KEY, monotone=False, activation=activation), seed)
        gap = model((a + b) / 2) - (model(a) + model(b)) / 2
        assert float(gap.max()) <= 1e-10 * float(jnp.abs(model(a)).max())


@pytest.mark.parametrize("monotone", MONOTONE.values(), ids=MONOTONE)
def test_monotone_inputs_have_non_negative_derivatives(monotone):
    xs = 2 * jax.random.normal(jax.random.PRNGKey(4), (128, 3))
    mask = np.broadcast_to(np.asarray(monotone), (3,))
    for seed in range(3):
        model = randomized(ICNN(3, [8, 8], KEY, monotone=monotone), seed)
        derivatives = np.asarray(jax.vmap(jax.grad(model))(xs))
        assert np.all(derivatives[:, mask] >= -1e-12)
        if not mask.all():
            assert np.any(derivatives[:, ~mask] < 0)  # free inputs are not constrained


@pytest.mark.parametrize("monotone", MONOTONE.values(), ids=MONOTONE)
@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_zero_at_origin_at_initialization(monotone, depth):
    for seed in range(3):
        model = ICNN(3, [16] * depth, jax.random.PRNGKey(seed), monotone=monotone)
        assert float(model(jnp.zeros(3))) == 0.0


@pytest.mark.parametrize("monotone", MONOTONE.values(), ids=MONOTONE)
def test_output_scale_is_stable_with_depth(monotone):
    """For inputs of unit norm, the output is of order `scale` at initialization, whatever
    the depth (the previous initialization grew about ten-fold per layer)."""
    xs = unit_directions(256, 3)

    def rms(depth, scale=1.0):
        values = [
            jnp.sqrt(
                jnp.mean(ICNN(3, [32] * depth, jax.random.PRNGKey(s), scale, monotone)(xs) ** 2)
            )
            for s in range(5)
        ]
        return float(np.mean(values))

    magnitudes = [rms(depth) for depth in (1, 2, 4, 8)]
    assert all(0.3 < m < 1.0 for m in magnitudes)
    assert max(magnitudes) / min(magnitudes) < 2  # no trend with depth, only seed noise
    assert 0.005 < rms(4, scale=0.01) / magnitudes[2] < 0.015


@pytest.mark.parametrize("positive", ["softplus", "exp", "square"])
def test_initial_weights_hit_their_targets(positive):
    model = ICNN(4, [64, 64], KEY, positive=positive)
    assert float(positive_param(model.Ws[0], positive).mean()) == pytest.approx(1 / 64, rel=0.05)
    assert float(positive_param(model.final_W, positive).mean()) == pytest.approx(1 / 64, rel=0.05)


def fit(model, xs, ys, iters=2000, lr=2e-2):
    optimizer = optax.adam(lr)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def step(carry, _):
        params, state = carry
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean((eqx.combine(p, static)(xs) - ys) ** 2)
        )(params)
        updates, state = optimizer.update(grads, state, params)
        return (optax.apply_updates(params, updates), state), loss

    # all the iterations in a single compiled loop, instead of one dispatch per step
    _, losses = jax.jit(lambda p, s: jax.lax.scan(step, (p, s), length=iters))(params, state)
    return float(losses[-1])


def test_free_inputs_represent_non_monotone_convex_functions():
    """x^2 on [-2, 2] (variance 1.42): fitted with a free input, not with a monotone one."""
    xs = jnp.linspace(-2.0, 2.0, 81)
    assert fit(ICNN(0, [16, 16], KEY, monotone=False), xs, xs**2) < 1e-2
    assert fit(ICNN(0, [16, 16], KEY, monotone=True), xs, xs**2) > 0.3


def test_input_shapes():
    scalar = ICNN(0, [8], KEY)
    assert scalar(jnp.array(0.3)).shape == ()
    assert scalar(jnp.linspace(0.0, 1.0, 5)).shape == (5,)
    assert float(jax.grad(scalar)(0.3)) > 0
    vector = ICNN(3, [8, 8], KEY, monotone=False)
    xs = jax.random.normal(KEY, (7, 3))
    assert vector(xs[0]).shape == ()
    assert jnp.allclose(vector(xs), jax.vmap(vector)(xs))
    assert ICNN(1, [8], KEY)(jnp.array(0.3)).shape == ()


def test_icnnskip_is_a_deprecated_alias():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = ICNNSkip(3, [8], KEY)
    assert any(issubclass(w.category, FutureWarning) for w in caught)
    assert isinstance(alias, ICNN)
    x = jnp.array([0.1, -0.2, 0.3])
    assert float(alias(x)) == float(ICNN(3, [8], KEY)(x))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(in_dim=3, hidden_dims=[]),
        dict(in_dim=3, hidden_dims=[8], monotone=(True, False)),
        dict(in_dim=3, hidden_dims=[8], positive="relu"),
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ICNN(key=KEY, **kwargs)


def test_activation_and_reparametrization():
    z = jnp.linspace(-5.0, 5.0, 101)
    assert float(shifted_softplus(0.0)) == 0.0
    slopes = jax.vmap(jax.grad(shifted_softplus))(z)
    assert jnp.all(slopes > 0) and jnp.all(jnp.diff(slopes) > 0)  # increasing, convex
    assert jnp.allclose(positive_param(z), jax.nn.softplus(z))  # default unchanged
