import jax
import jax.numpy as jnp

import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2, SymmetricTensor4, utils


def test_small_strain_orthotropic_rotation():
    """This test instantiates an orthotropic elastic material,
    checks the rotation mechanism.
    We then check that the rotated tensor of angle pi/2 matches with
    an orthotropic material instance for which  (L-T) material parameters are
    exchanged. Finally, we verify that the stress computed with the rotated
    material matches with the one obtained by: (a) rotation the strain, (b)
    computing the stress in the local axis and then rotate back to the
    original frame."""
    # Define orthotropic elastic material
    EL = 12.0e3
    ET = 0.8e3
    EN = 1.0e3
    nuLT = 0.43
    nuLN = 0.47
    nuTN = 0.292
    muLT = 0.7e3
    muLN = 0.9e3
    muTN = 0.2e3
    elasticity = jm.LinearElasticOrthotropic(
        EL=EL,
        ET=ET,
        EN=EN,
        nuLT=nuLT,
        nuLN=nuLN,
        nuTN=nuTN,
        muLT=muLT,
        muLN=muLN,
        muTN=muTN,
    )

    angle = jnp.pi / 2
    axis = jnp.array([0, 0, 1])
    R = utils.rotation_matrix_direct(angle, axis)

    material = jm.ElasticBehavior(elasticity=elasticity)
    mat_state = material.init_state()

    # Create deformation gradient tensor
    key = jax.random.PRNGKey(42)
    N = 3
    eps_ = jax.random.normal(key, (N, 3, 3))
    eps = SymmetricTensor2(tensor=eps_)

    # Rotate stiffness tensor: L<->T permutation (90° around z-axis)
    C = elasticity.C

    C_rotated = C.rotate(R)
    assert isinstance(C_rotated, SymmetricTensor4)
    assert jnp.allclose(C_rotated.rotate(R), C)

    elasticity_rotated = jm.LinearElasticOrthotropic(
        EL=ET,
        ET=EL,
        EN=EN,
        nuLT=nuLT * ET / EL,  # nuTN = ET*nuLT/EL
        nuLN=nuTN,
        nuTN=nuLN,
        muLT=muLT,
        muLN=muTN,
        muTN=muLN,
    )

    elasticity_C_rotated = jm.LinearElastic(stiffness=C_rotated)
    assert jnp.allclose(elasticity_rotated.C, elasticity_C_rotated.C)

    material_rotated = jm.ElasticBehavior(elasticity=elasticity_rotated)

    # Compute stress for rotated material
    def compute_stress_rotated(eps_single):
        sig, _ = material_rotated.constitutive_update(eps_single, mat_state, dt=0.0)
        return sig

    def rotate_stress_strain(eps_single):
        eps_ = eps_single.rotate(R)
        sig, _ = material.constitutive_update(eps_, mat_state, dt=0.0)
        return sig.rotate(R.T)

    sig_C_rotated = jax.vmap(compute_stress_rotated)(eps)
    sig_rotated = jax.vmap(rotate_stress_strain)(eps)

    assert jnp.allclose(sig_rotated, sig_C_rotated)
