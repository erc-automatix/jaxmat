"""Input convex neural networks (ICNN)."""

import itertools
import math
import warnings
from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

_LOG2 = math.log(2.0)


def shifted_softplus(x: jax.Array) -> jax.Array:
    r"""Softplus shifted to vanish at the origin, $\log(1+e^x) - \log 2$.

    Convex and non-decreasing like the softplus, hence a valid ICNN activation (the hidden
    units of an ICNN must be convex, not non-negative). With zero biases, a zero input gives
    zero hidden units, so an ICNN using it satisfies $f(\boldsymbol{0}) = 0$ at
    initialization, whatever its depth.
    """
    return jax.nn.softplus(x) - _LOG2


_POSITIVE = {"softplus": jax.nn.softplus, "exp": jnp.exp, "square": jnp.square}
_INVERSE = {
    # inverse softplus log(exp(y) - 1), written to stay accurate for large y
    "softplus": lambda y: y + jnp.log(-jnp.expm1(-y)),
    "exp": jnp.log,
    "square": jnp.sqrt,
}


def positive_param(W: jax.Array, method: str = "softplus") -> jax.Array:
    """Non-negative reparametrization of raw weights: ``softplus`` (default), ``exp`` or
    ``square``."""
    return _POSITIVE[method](W)


def _init_positive(shape, key, target, method, std):
    """Raw weights whose non-negative values are log-normal around ``target``.

    Drawing the raw weights directly (``normal * scale``) does not work for the softplus:
    ``softplus(0) = log 2``, so a small scale cannot make the weights small, and the output
    grows with the width and the depth.
    """
    values = target * jnp.exp(std * jax.random.normal(key, shape))
    return _INVERSE[method](values)


def _init_free(shape, key, target):
    return target * jax.random.normal(key, shape)


class ICNN(eqx.Module):
    r"""Input convex neural network, convex in its inputs and optionally non-decreasing in
    some of them.

    An ICNN (Amos et al., 2017) is a scalar network $f(\bx)$ that is convex in its inputs by
    construction. With $\sigma$ a convex and non-decreasing activation:

    $$
    \begin{aligned}
    \bz_1 &= \sigma(\bU_0\bx + \bb_0)\\
    \bz_{k+1} &= \sigma(\bW_k^{+}\bz_k + \bU_k\bx + \bb_k), \quad k=1,\ldots,n-1\\
    f(\bx) &= \bw^{+}\cdot\bz_n + \bu\cdot\bx
    \end{aligned}
    $$

    where the weights $\bW_k^{+}$ and $\bw^{+}$ acting on the hidden units are non-negative.
    Each layer is then a non-decreasing convex function of convex functions plus an affine
    function of $\bx$, hence convex. The input skips $\bU_k$ and $\bu$ are unconstrained, except
    on *monotone* inputs, where they are non-negative too, so that $f$ is also non-decreasing
    in these inputs.

    .. admonition:: References
        :class: seealso

        - Amos, B., Xu, L., & Kolter, J. Z. (2017). Input convex neural networks. International
          Conference on Machine Learning, 146-155.

    Parameters
    ----------
    in_dim : int
        Input dimension. ``0`` declares a scalar input, as for ``in_dim=1`` but called on a
        scalar.
    hidden_dims : list[int]
        Widths of the hidden layers (at least one).
    key : jax.Array
        Random key for the initialization.
    scale : float, optional
        Order of magnitude of the output at initialization for inputs of unit norm (default
        1.0). It stays roughly constant with the width and the depth.
    monotone : bool or sequence of bool, optional
        Whether $f$ is non-decreasing in each input (default ``True``, for all inputs).
        Monotone inputs have non-negative skip weights. Non-monotone inputs make the
        network able to represent any convex function, e.g. $x^2$; monotone ones are
        required for instance for polyconvex energies of invariants, or for a hardening
        potential whose derivative must be non-decreasing.
    positive : {"softplus", "exp", "square"}, optional
        Reparametrization of the non-negative weights (default ``"softplus"``).
    activation : Callable, optional
        Convex and non-decreasing activation (it need not be non-negative). The default,
        :func:`shifted_softplus`, is smooth, so that second derivatives (e.g. a tangent
        operator) are meaningful, and gives $f(\boldsymbol{0}) = 0$ at initialization. With
        ``jax.nn.softplus``, the output has an offset of about $\log 2$ per layer at the
        origin. ``jax.nn.relu`` gives a piecewise linear $f$, whose Hessian vanishes almost
        everywhere.
    init_std : float, optional
        Relative (log-normal) spread of the initial non-negative weights around their
        targets (default 0.1).

    Notes
    -----
    The hidden weights are initialized around ``1/width`` of the previous layer, so that
    each layer averages the previous one with unit gain. The skips are initialized around
    ``scale/sqrt(in_dim)``: zero-mean on non-monotone inputs, where averaging cancels their
    signs through the depth, and non-negative on monotone inputs, where they accumulate,
    hence an additional ``1/sqrt(depth)`` factor on the hidden skips.

    Calling the network on an array of shape ``(..., in_dim)`` (``(...)`` for a scalar
    input) returns an array of shape ``(...)``.
    """

    Us: list[jax.Array]
    """Input skips of each hidden layer, non-negative on monotone inputs."""
    Ws: list[jax.Array]
    """Raw hidden-to-hidden weights (non-negative after reparametrization), from the second
    hidden layer on."""
    bs: list[jax.Array]
    """Biases of each hidden layer."""
    final_W: jax.Array
    """Raw output weights on the last hidden layer (non-negative after reparametrization)."""
    final_U: jax.Array
    """Output skip, non-negative on monotone inputs."""
    monotone: tuple[bool, ...] = eqx.field(static=True)
    positive: str = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    scalar_input: bool = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        key: jax.Array,
        scale: float = 1.0,
        monotone: bool | Sequence[bool] = True,
        positive: str = "softplus",
        activation: Callable = shifted_softplus,
        init_std: float = 0.1,
    ):
        if len(hidden_dims) == 0:
            raise ValueError("An ICNN needs at least one hidden layer.")
        if positive not in _POSITIVE:
            raise ValueError(f"positive must be one of {sorted(_POSITIVE)}, got {positive!r}.")
        self.scalar_input = in_dim == 0
        in_dim = max(in_dim, 1)
        if isinstance(monotone, bool):
            monotone = (monotone,) * in_dim
        if len(monotone) != in_dim:
            raise ValueError(f"monotone has {len(monotone)} entries for {in_dim} inputs.")
        self.monotone = tuple(bool(m) for m in monotone)
        self.positive = positive
        self.activation = activation

        n = len(hidden_dims)
        keys = jax.random.split(key, 2 * n + 2)
        u_free = scale / math.sqrt(in_dim)
        u_mono = u_free / math.sqrt(n)
        mono = jnp.asarray(self.monotone)

        def skip(shape, key, target):
            positive_ = _init_positive(shape, key, target, positive, init_std)
            return jnp.where(mono, positive_, _init_free(shape, key, u_free))

        self.Us = [skip((h, in_dim), keys[2 * i], u_mono) for i, h in enumerate(hidden_dims)]
        self.Ws = [
            _init_positive((h, prev), keys[2 * i + 1], 1.0 / prev, positive, init_std)
            for i, (prev, h) in enumerate(itertools.pairwise(hidden_dims), start=1)
        ]
        self.bs = [jnp.zeros((h,)) for h in hidden_dims]
        last = hidden_dims[-1]
        self.final_W = _init_positive((last,), keys[-2], 1.0 / last, positive, init_std)
        self.final_U = skip((in_dim,), keys[-1], u_free)

    def _skip(self, U):
        """Skip weights, non-negative on monotone inputs."""
        return jnp.where(jnp.asarray(self.monotone), positive_param(U, self.positive), U)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        if self.scalar_input or x.ndim == 0:
            x = x[..., None]
        z = self.activation(x @ self._skip(self.Us[0]).T + self.bs[0])
        for W, U, b in zip(self.Ws, self.Us[1:], self.bs[1:], strict=True):
            z = self.activation(z @ positive_param(W, self.positive).T + x @ self._skip(U).T + b)
        return z @ positive_param(self.final_W, self.positive) + x @ self._skip(self.final_U)


class ICNNSkip(ICNN):
    """Deprecated alias of :class:`ICNN`, which now has input skip connections."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "'ICNNSkip' is deprecated and will be removed in v0.2: 'ICNN' now has input skip "
            "connections, non-negative on monotone inputs (the default).",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class HomogeneousICNN(eqx.Module):
    """ICNN constrained to represent a convex, 1-homogeneous gauge function."""

    Ws: list[jnp.ndarray]  # hidden-to-hidden (constrained nonneg)
    Us: list[jnp.ndarray]  # input-to-hidden skip matrices (can be any real)
    final_W: jnp.ndarray

    def __init__(self, in_dim, hidden_dims, key, scale=1.0):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.Ws = []
        self.Us = []
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            W = jax.random.normal(keys[i], (h, prev)) * scale
            U = jax.random.normal(keys[i], (h, in_dim)) * scale
            self.Ws.append(W)
            self.Us.append(U)
            prev = h
        self.final_W = jax.random.normal(keys[-1], (prev,)) * scale

    def __call__(self, x):
        z = x
        for W, U in zip(self.Ws, self.Us):
            W_pos = positive_param(W)  # enforce W_z >= 0 for convexity
            z = jax.nn.relu(z @ W_pos.T + x @ U.T)  # no biases, ReLU is 1-homog
        return jnp.dot(z, positive_param(self.final_W))
