from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

import jaxmat.materials as jm
from jaxmat.materials.behavior import AbstractBehavior
from jaxmat.state import (
    SmallStrainState,
)
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor2
from jaxmat.utils import default_value, enforce_dtype


class Constant(eqx.Module):
    value: float | jax.Array = enforce_dtype()

    def __call__(self, **kwargs):
        return self.value

    def __repr__(self):
        return str(self.value)


def as_module(x):
    """Enforce a field to behave like a callable module. If not, wrap as a Constant."""
    if not callable(x):
        return Constant(x)
    else:
        return x


class LinearElasticIsotropic(jm.AbstractLinearElastic):
    """An isotropic linear elastic model."""

    E: eqx.Module = eqx.field(converter=as_module)
    r"""Young modulus $E$"""
    nu: eqx.Module = eqx.field(converter=as_module)
    r"""Poisson ratio $\nu$"""

    def C(self, **kwargs):
        return IsotropicTensor4(self.kappa(**kwargs), self.mu(**kwargs))

    def kappa(self, **kwargs):
        E = self.E(**kwargs)
        nu = self.nu(**kwargs)
        return E / (3 * (1 - 2 * nu))

    def mu(self, **kwargs):
        E = self.E(**kwargs)
        nu = self.nu(**kwargs)
        return E / (2 * (1 + nu))


class SmallStrainThermoMechanicalState(SmallStrainState):
    temperature: float = default_value(0.0)


class SmallStrainThermoMechanicalBehavior(AbstractBehavior):
    """Abstract small strain thermo-mechanical behavior."""

    def init_state(self, Nbatch=None):
        """Initialize the mechanical small strain state."""
        return self._init_state(SmallStrainThermoMechanicalState, Nbatch)

    @abstractmethod
    def constitutive_update(self, inputs, state, dt):
        _eps, _T = inputs
        pass


class ElasticBehavior(SmallStrainThermoMechanicalBehavior):
    """A small strain linear elastic behavior."""

    elasticity: eqx.Module
    """The corresponding linear elastic model."""
    internal = None

    def constitutive_update(self, inputs, state, dt):
        eps, T = inputs
        sig = self.elasticity.C(T=T) @ eps
        new_state = state.update(strain=eps, stress=sig, temperature=T)
        return sig, new_state


class YoungModulus(eqx.Module):
    E0 = 200e3
    a = 20.0

    def __call__(self, T):
        return self.E0 * (1 + self.a * T)


elasticity1 = LinearElasticIsotropic(E=200e3, nu=0.3)
elasticity2 = LinearElasticIsotropic(E=lambda T: 200e3 * (1 + 0.1 * T), nu=lambda T: 0.3)
elasticity3 = LinearElasticIsotropic(E=YoungModulus(), nu=jnp.asarray(0.3))

for elasticity in [elasticity1, elasticity2, elasticity3]:
    material = ElasticBehavior(elasticity=elasticity)

    state = material.init_state()
    print(state.strain)
    print(state.temperature)

    eps = SymmetricTensor2.identity()
    inputs = eps, 1.5
    stress, new_state = material.constitutive_update(inputs, state, 0.0)
    print(stress.array)

# Batch along strain path and temperatures
