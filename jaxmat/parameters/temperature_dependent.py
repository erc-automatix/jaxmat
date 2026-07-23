"""Temperature- (or input-) dependent material parameters.

A parameter is either a plain constant or an expression ``p(T)`` carrying its own
fitting parameters (subclass :class:`AbstractParameter`, implement ``evaluate``).
``evaluate_parameters(model, T)`` returns a copy of any model in which every
``AbstractParameter`` field has been replaced by its scalar value ``p.evaluate(T)``
-- constant fields untouched -- so all existing model methods work unchanged.
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from .generic_parameters import parameter

DEFAULT_TEMPERATURE = 273.15


class AbstractParameter(eqx.Module):
    """A scalar material parameter that depends on temperature (or any input)."""

    @abstractmethod
    def evaluate(self, T):
        raise NotImplementedError

    # Using an unresolved parameter is a usage error: raise something actionable
    # instead of a cryptic "unsupported operand type".
    def _unresolved(self, *a, **k):
        raise TypeError(
            f"{type(self).__name__} is a temperature-dependent parameter that has "
            "not been resolved. Call `evaluate_parameters(model, T)` first, or wrap "
            "the behavior in `TemperatureDependent(...)`."
        )

    __add__ = __radd__ = __sub__ = __rsub__ = _unresolved
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _unresolved
    __pow__ = __rpow__ = __neg__ = __array__ = _unresolved


class Constant(AbstractParameter):
    """A constant value (explicitly temperature-independent)."""

    value: jax.Array = parameter()

    def evaluate(self, T):
        return self.value


class Affine(AbstractParameter):
    r"""Linear law

    $$p(T) = p_{\textrm{ref}} + s (T-T_{\textrm{ref}})$$
    """

    ref: jax.Array = parameter()
    r"""Reference value $p_{\textrm{ref}}$"""
    slope: jax.Array = parameter()
    r"""Slope $s$"""
    T_ref: jax.Array = parameter(default=DEFAULT_TEMPERATURE)
    r"""Reference temperature $T_{\textrm{ref}}$"""

    def evaluate(self, T):
        return self.ref + self.slope * (T - self.T_ref)


class Polynomial(AbstractParameter):
    r"""
    Polynomial law

    $$p(T) = \sum_k c_k (T - T_{\textrm{ref}})^k$$
    """

    coeffs: jax.Array = parameter()
    """Polynomial coefficients, ordered from low to high order."""
    T_ref: jax.Array = parameter(default=DEFAULT_TEMPERATURE)

    def evaluate(self, T):
        return jnp.polyval(self.coeffs[::-1], T - self.T_ref)


class Arrhenius(AbstractParameter):
    r"""Arrhenius activation energy law

    $$p(T) = A\,\exp(-Q / (R\,T))$$
    """

    A: jax.Array = parameter()
    """Arrhenius factor."""
    Q: jax.Array = parameter()
    """Activation energy"""
    R: jax.Array = parameter(default=8.314462618)
    r"""Universal gas constant $R=8.314462618 \textrm{J}.\textrm{K}^{-1}.\textrm{mol}^{-1}."""

    def evaluate(self, T):
        return self.A * jnp.exp(-self.Q / (self.R * T))


class Expression(AbstractParameter):
    """Wrap any ``callable`` / ``eqx.Module`` with signature ``fn(T) -> scalar`` as a parameter."""

    fn: eqx.Module

    def evaluate(self, T):
        return self.fn(T)


def _is_parameter(x):
    return isinstance(x, AbstractParameter)


def has_parameter_dependence(model):
    """True if ``model`` contains any unresolved ``AbstractParameter``."""
    return any(_is_parameter(x) for x in jax.tree.leaves(model, is_leaf=_is_parameter))


def evaluate_parameters(model, T):
    """Materialize every ``AbstractParameter`` in ``model`` at temperature ``T``.

    For an isothermal analysis, resolve once and use the result as an ordinary
    constant model:

    .. code-block:: python

        isothermal = evaluate_parameters(model, T_fixed)
    """
    if T is None:
        if has_parameter_dependence(model):
            raise ValueError(
                "This model has temperature-dependent parameters but no temperature "
                "was given. Pass a fixed value (isothermal), or wrap the behavior in "
                "`TemperatureDependent(...)`."
            )
        return model

    def resolve(leaf):
        return leaf.evaluate(T) if _is_parameter(leaf) else leaf

    return jax.tree.map(resolve, model, is_leaf=_is_parameter)
