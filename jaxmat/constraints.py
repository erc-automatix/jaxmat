"""Constrained parameter mappings for calibration (backed by ``paramax``).

A material parameter can be reparameterized so that gradient-based calibration
stays in a feasible region by construction. Declare a field with ``Positive()`` or
``Interval(lo, hi)``; construct the model with ordinary physical values, and call
``paramax.unwrap(model)`` once at the top of the loss. Every read then sees the
constrained value while the optimiser works on an unconstrained latent.

    class Mat(eqx.Module):
        E:  jax.Array = Positive()
        nu: jax.Array = Interval(0.0, 0.5)

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


def Positive():
    """Field constrained to be strictly positive (x > 0), via exp/log."""

    def convert(v):
        if _is_wrapped(v):
            return v
        return paramax.Parameterize(jnp.exp, jnp.log(jnp.asarray(v, float)))

    return eqx.field(converter=convert)


def Interval(lo: float, hi: float):
    """Field constrained to an open interval (lo < x < hi), via a scaled sigmoid."""

    def convert(v):
        if _is_wrapped(v):
            return v
        z = (jnp.asarray(v, float) - lo) / (hi - lo)
        return paramax.Parameterize(lambda u: lo + (hi - lo) * jax.nn.sigmoid(u), logit(z))

    return eqx.field(converter=convert)


def Bounded(lo: float, hi: float):
    """Alias for :func:`Interval`."""
    return Interval(lo, hi)
