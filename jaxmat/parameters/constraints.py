r"""Constrained parameter mappings for calibration (backed by ``paramax``).

A material parameter $\theta$ with physical constraints (e.g. positive) can be reparameterized
so that gradient-based calibration stays in a feasible region by construction. This typically
works by introducing a non-linear function $f$ whose codomain (output domain) is the wanted
constrained region, e.g. $f: \mathbb{R} \mapsto \mathbb{R}^+$. A latent variable representation
$z_\theta$ is used under-the-hood such that $\theta = f(z_{\theta})$ naturally satisfies the
constraints.

**Usage**: Declare a field with ``Positive()`` or ``Interval(lo, hi)``; construct the model with
ordinary physical values, and call ``paramax.unwrap(model)`` once at the top of the loss. Every
read then sees the constrained value while the optimiser works on an unconstrained latent
representation $z_\theta$.

.. code-block:: python

    class Mat(eqx.Module):
        E:  jax.Array = Positive()
        nu: jax.Array = Interval(-1.0, 0.5)

    m = Mat(E=210e3, nu=0.3)          # raw values; auto-wrapped
    def loss(m, target):
        l = paramax.unwrap(m)          # l is a model with latent values
        ...
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
from jax.scipy.special import logit


def _is_wrapped(v):
    return isinstance(v, paramax.AbstractUnwrappable)


def Unconstrained():
    """Unconstrained field."""

    def convert(v):
        if _is_wrapped(v):
            return v
        return paramax.Parameterize(lambda x: x, jnp.asarray(v, float))

    return eqx.field(converter=convert)


def Scaled(ref: float = 1.0):
    r"""
    Rescaled field storing $\theta/\text{ref}$ (an $O(1)$ latent variable); recover
    $\theta = \text{ref} * z_{\theta}$.

    Unconstrained value.
    """

    def convert(v):
        if isinstance(v, paramax.AbstractUnwrappable):
            return v
        return paramax.Parameterize(lambda u: ref * u, jnp.asarray(v, float) / ref)

    return eqx.field(converter=convert)


def Positive():
    r"""Field constrained to be strictly positive ($\theta > 0$), via $f(z)=\exp(z)$."""

    def convert(v):
        if _is_wrapped(v):
            return v
        return paramax.Parameterize(jnp.exp, jnp.log(jnp.asarray(v, float)))

    return eqx.field(converter=convert)


def Interval(lo: float, hi: float):
    r"""Field constrained to an open interval ($\text{lo} < \theta < \text{hi}$),
    via a scaled sigmoid."""

    def convert(v):
        if _is_wrapped(v):
            return v
        z = (jnp.asarray(v, float) - lo) / (hi - lo)
        return paramax.Parameterize(lambda u: lo + (hi - lo) * jax.nn.sigmoid(u), logit(z))

    return eqx.field(converter=convert)


def Bounded(lo: float, hi: float):
    """Alias for :func:`Interval`."""
    return Interval(lo, hi)


def _cholesky_factor(z):
    """Lower-triangular factor with diagonal exp(z_ii) and strict lower part z_ij."""
    diagonal = jnp.eye(z.shape[-1], dtype=bool)
    return jnp.where(diagonal, jnp.exp(z), jnp.tril(z, -1))


def _spd_from_latent(z):
    L = _cholesky_factor(z)
    return L @ jnp.swapaxes(L, -1, -2)


def _latent_from_spd(M):
    M = jnp.asarray(M, float)
    if M.ndim < 2 or M.shape[-1] != M.shape[-2]:
        raise ValueError(f"PositiveDefinite expects square matrices, got shape {M.shape}.")
    L = jnp.linalg.cholesky(0.5 * (M + jnp.swapaxes(M, -1, -2)))
    L = eqx.error_if(
        L, ~jnp.all(jnp.isfinite(L)), "PositiveDefinite expects a positive-definite matrix."
    )
    diagonal = jnp.eye(M.shape[-1], dtype=bool)
    return jnp.where(diagonal, jnp.log(jnp.where(diagonal, L, 1.0)), jnp.tril(L, -1))


def PositiveDefinite():
    r"""Field constrained to be a symmetric positive-definite matrix, via its Cholesky factor.

    The matrix is $\bM = \bL\bL\T$, where $\bL$ is lower triangular with a positive diagonal,
    $L_{ii} = \exp(z_{ii})$, and a free strict lower part, $L_{ij} = z_{ij}$ for $i > j$. This
    is a smooth bijection between $\mathbb{R}^{d(d+1)/2}$ and the cone of positive-definite
    matrices. Semi-definite matrices are its boundary, approached as some $z_{ii} \to -\infty$.

    The value is given as a ``(..., d, d)`` array (only its symmetric part is used). The latent
    array has the same shape; its strict upper triangle is unused and has a zero gradient.

    .. code-block:: python

        class Anisotropic(eqx.Module):
            M: jax.Array = PositiveDefinite()

        m = Anisotropic(M=jnp.diag(jnp.array([1.0, 2.0, 0.5])))
    """

    def convert(v):
        if _is_wrapped(v):
            return v
        return paramax.Parameterize(_spd_from_latent, _latent_from_spd(v))

    return eqx.field(converter=convert)
