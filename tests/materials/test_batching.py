import equinox as eqx
import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.state import make_batched
from jaxmat.tensors import SymmetricTensor2

elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
hardening = jm.VoceHardening(sig0=350.0, sigu=500.0, b=1e3)

elastic = jm.ElasticBehavior(elasticity)
elastoplastic = jm.vonMisesIsotropicHardening(
    elasticity=elasticity, yield_stress=hardening
)


@pytest.mark.parametrize("material", [elastic, elastoplastic])
def test_material_batching(material):
    def eval_stress(material, eps):
        state = material.init_state()
        sig, _ = material.constitutive_update(eps, state, 0.0)
        return sig

    Nbatch = 10
    batched_material = make_batched(material, Nbatch)
    eps = SymmetricTensor2(array=jnp.array([1, 0, 0, 0, 0, 0]))
    sig_batched = eqx.filter_vmap(eval_stress, in_axes=(0, None))(batched_material, eps)

    sig = eval_stress(material, eps)
    assert jnp.allclose(
        sig_batched.array, jnp.broadcast_to(sig.array, sig_batched.shape)
    )
