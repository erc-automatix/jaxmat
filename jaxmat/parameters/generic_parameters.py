"""Material parameters base module"""

import equinox as eqx
import jax.numpy as jnp

_UNSET = object()


def parameter(default=_UNSET, *, dtype=jnp.float64, **kwargs):
    """Declare a *material coefficient* field.

    A coefficient may be supplied either as

    * a constant (``int``/``float``/array), cast to ``dtype``; or
    * an :class:`~jaxmat.parameters.AbstractParameter` law (e.g. ``Arrhenius``,
      ``Affine``, ``Expression``), which is stored **unchanged** and resolved later
      by a wrapper such as :class:`~jaxmat.materials.thermomechanics.TemperatureDependent`.

    ``None`` is also passed through, so optional coefficients work. Use this for every
    quantity a ``constitutive_update`` reads as a material property (moduli, yield
    stresses, relaxation times, hardening coefficients, shape exponents, ...). This is
    what makes a behavior composable with input-dependent wrappers.

    Examples
    --------
    ``E: float = parameter(default=210e3)`` accepts ``MyMat(E=210e3)`` (isothermal) or
    ``MyMat(E=Arrhenius(A=..., Q=...))`` (temperature-dependent), with no other change.
    """

    def convert(x):
        if x is None or isinstance(x, eqx.Module):
            return x
        return jnp.asarray(x, dtype=dtype)

    if default is not _UNSET:
        kwargs["default"] = default
    return eqx.field(converter=convert, **kwargs)
