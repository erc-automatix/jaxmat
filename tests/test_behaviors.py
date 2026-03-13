import equinox as eqx
import jax.numpy as jnp

import jaxmat.materials as jm
from jaxmat.state import (
    make_batched,
)


class YieldStress(eqx.Module):
    sig0: float

    def __call__(self, p):
        return self.sig0


def test_small_strain_behavior():
    elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
    hardening = YieldStress(sig0=300.0)

    material = jm.vonMisesIsotropicHardening(elasticity=elasticity, yield_stress=hardening)
    state = material.init_state()
    assert hasattr(state, "strain")
    assert hasattr(state, "stress")
    assert hasattr(state.internal, "p")
    assert hasattr(state.internal, "epsp")

    Nbatch = 10
    batched_state = material.init_state(Nbatch)
    assert jnp.array_equal(batched_state.strain, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.stress, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.internal.epsp, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.internal.p, jnp.zeros((Nbatch,)))

    material2 = jm.vonMisesIsotropicHardening(elasticity=elasticity, yield_stress=hardening)
    batched_material = make_batched(material2, Nbatch)
    assert jnp.array_equal(batched_material.elasticity.E, jnp.full((Nbatch,), elasticity.E))
    state = batched_material.init_state()
    assert jnp.array_equal(state.strain, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.stress, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.internal.epsp, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.internal.p, jnp.zeros((Nbatch,)))


def test_finite_strain_behavior():
    elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
    hardening = YieldStress(sig0=300.0)

    material = jm.FeFpJ2Plasticity(elasticity=elasticity, yield_stress=hardening)
    state = material.init_state()
    assert hasattr(state, "F")
    assert hasattr(state, "PK1")
    assert hasattr(state, "Cauchy")
    assert hasattr(state.internal, "p")
    assert hasattr(state.internal, "be_bar")

    Nbatch = 10
    batched_state = material.init_state(Nbatch)
    assert hasattr(batched_state, "F")
    assert hasattr(batched_state, "PK1")
    assert hasattr(batched_state, "Cauchy")
    assert hasattr(batched_state.internal, "p")
    assert hasattr(batched_state.internal, "be_bar")
    assert jnp.array_equal(batched_state.F, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.PK1, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(
        batched_state.internal.be_bar, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3))
    )
    assert jnp.array_equal(batched_state.internal.p, jnp.zeros((Nbatch,)))

    material2 = jm.FeFpJ2Plasticity(elasticity=elasticity, yield_stress=hardening)
    batched_material = make_batched(material2, Nbatch)
    assert jnp.array_equal(batched_material.elasticity.E, jnp.full((Nbatch,), elasticity.E))
    state = batched_material.init_state()
    assert jnp.array_equal(state.F, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3)))
    assert jnp.array_equal(state.PK1, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.internal.be_bar, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3)))
    assert jnp.array_equal(state.internal.p, jnp.zeros((Nbatch,)))
