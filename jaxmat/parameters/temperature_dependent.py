"""Temperature- (or any external variable-) dependent material parameters.

A parameter is either a plain constant or an expression ``p(T)`` carrying its own
fitting parameters (subclass :class:`AbstractParameter`, implement ``evaluate``).
``evaluate_parameters(model, T)`` returns a copy of any model in which every
``AbstractParameter`` field has been replaced by its scalar value ``p.evaluate(T)``
-- constant fields untouched -- so all existing model methods work unchanged.

Temperature is only the default: a parameter may depend on any named external state
variable(s) through ``depends_on``, resolved with keyword arguments,

.. code-block:: python

    E = Affine(ref=210e3, slope=-5e3, T_ref=0.0, depends_on="phi")  # porosity-dependent
    k = Expression(fn=law, depends_on=("T", "phi"))                  # law(T, phi)
    resolved = evaluate_parameters(model, T=400.0, phi=0.1)
"""

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from .generic_parameters import parameter

DEFAULT_TEMPERATURE = 273.15


class AbstractParameter(eqx.Module):
    """A scalar material parameter that depends on temperature, or on other external state
    variables named by ``depends_on``."""

    depends_on: str | tuple[str, ...] = eqx.field(static=True, default="T", kw_only=True)
    """Name(s) of the external state variable(s) the parameter depends on; ``evaluate``
    receives their values in this order. Defaults to temperature ``"T"``."""

    @property
    def variables(self) -> tuple[str, ...]:
        """``depends_on`` as a tuple of names."""
        return (self.depends_on,) if isinstance(self.depends_on, str) else tuple(self.depends_on)

    @abstractmethod
    def evaluate(self, *values):
        raise NotImplementedError

    # Using an unresolved parameter is a usage error: raise something actionable
    # instead of a cryptic "unsupported operand type".
    def _unresolved(self, *a, **k):
        raise TypeError(
            f"{type(self).__name__} is a variable-dependent parameter that has "
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
    """Wrap any ``callable`` / ``eqx.Module`` with signature ``fn(T) -> scalar`` as a parameter.

    With several variables, ``fn`` receives them in the order of ``depends_on``, e.g.
    ``Expression(fn=law, depends_on=("T", "phi"))`` evaluates ``law(T, phi)``.
    """

    fn: eqx.Module

    def evaluate(self, *values):
        return self.fn(*values)


def _is_parameter(x):
    return isinstance(x, AbstractParameter)


def has_parameter_dependence(model):
    """True if ``model`` contains any unresolved ``AbstractParameter``."""
    return any(_is_parameter(x) for x in jax.tree.leaves(model, is_leaf=_is_parameter))


def parameter_dependencies(model) -> set[str]:
    """Names of the external variables on which the parameters of ``model`` depend."""
    leaves = jax.tree.leaves(model, is_leaf=_is_parameter)
    return {name for leaf in leaves if _is_parameter(leaf) for name in leaf.variables}


def evaluate_parameters(model, T=None, /, **variables):
    """Materialize every ``AbstractParameter`` in ``model`` at the given external variables.

    Temperature can be passed positionally, other variables by name:
    ``evaluate_parameters(model, 400.0)``, ``evaluate_parameters(model, T=400.0, phi=0.1)``.
    A model without parameter laws is returned unchanged.

    For an isothermal analysis, resolve once and use the result as an ordinary
    constant model:

    .. code-block:: python

        isothermal = evaluate_parameters(model, T_fixed)
    """
    if T is not None:
        if "T" in variables:
            raise ValueError("Temperature given both positionally and as `T=`.")
        variables = {"T": T, **variables}
    needed = parameter_dependencies(model)
    if not needed:
        return model
    missing = needed - variables.keys()
    if missing:
        raise ValueError(
            f"This model has parameters depending on {sorted(needed)} but no value was given "
            f"for {sorted(missing)}. Pass them to `evaluate_parameters`, wrap the behavior in "
            "`TemperatureDependent(...)` for temperature, or declare them as external variables "
            "of a multiphysics behavior."
        )

    def resolve(leaf):
        if _is_parameter(leaf):
            return leaf.evaluate(*(variables[name] for name in leaf.variables))
        return leaf

    return jax.tree.map(resolve, model, is_leaf=_is_parameter)
