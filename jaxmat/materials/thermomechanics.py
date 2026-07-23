r"""Temperature-dependent behaviors, added by composition.

``TemperatureDependent`` wraps *any* existing behavior and makes it consume a
temperature input alongside the mechanical input, without modifying the wrapped
behavior. At each update it materializes the wrapped behavior's parameter
expressions at the current temperature (via ``evaluate_parameters``) and delegates
to the wrapped behavior's own ``constitutive_update``.

Inputs are ``(mechanical_input, T)`` where temperature $T$ is treated as an input known at
$t_{n+1}$. If ``default_temperature`` is set, a bare mechanical input is also accepted
(useful for isothermal drivers that do not thread ``T``).
"""

import equinox as eqx
import jax.numpy as jnp

from jaxmat.parameters import evaluate_parameters, parameter

from .behavior import AbstractBehavior


class TemperatureDependent(AbstractBehavior):
    """Make a behavior temperature-dependent by resolving its parameters at ``T``."""

    behavior: AbstractBehavior
    """The wrapped behavior, whose parameters may be temperature expressions."""
    default_temperature: jnp.ndarray = parameter(default=None)
    """Optional fallback temperature used when the input carries no ``T``."""

    def make_internal_state(self):
        return self.behavior.make_internal_state()

    def init_state(self, Nbatch=None):
        return self.behavior.init_state(Nbatch)

    def _split_inputs(self, inputs):
        if isinstance(inputs, tuple) and len(inputs) == 2:
            mech, T = inputs
            return mech, jnp.asarray(T, dtype=jnp.float64)  # strong -> no retrace
        if self.default_temperature is None:
            raise ValueError(
                "TemperatureDependent expected inputs `(mechanical, T)`. Provide a "
                "temperature, or set `default_temperature=` to allow a bare input."
            )
        return inputs, self.default_temperature

    @eqx.filter_jit
    def constitutive_update(self, inputs, state, dt):
        mech, T = self._split_inputs(inputs)
        resolved = evaluate_parameters(self.behavior, T)
        return resolved.constitutive_update(mech, state, dt)
