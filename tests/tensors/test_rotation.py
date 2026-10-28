import jax
import jax.numpy as jnp
import pytest

from jaxmat.tensors import (
    IsotropicTensor4,
    SymmetricTensor2,
    SymmetricTensor4,
    Tensor2,
    Tensor4,
    rotation,
)


def random_rotation(key):
    """Generate a proper orthogonal 3x3 rotation matrix."""
    A = jax.random.normal(key, (3, 3))
    Q, _ = jnp.linalg.qr(A)
    # enforce det = +1
    Q = Q * jnp.sign(jnp.linalg.det(Q))
    return Q


def test_rotate_identity_rank2():
    T = Tensor2(tensor=jnp.arange(9.0).reshape(3, 3))
    Id = jnp.eye(3)

    T_rot = T.rotate(Id)

    assert jnp.allclose(T_rot, T)


def test_rotate_identity_rank4():
    T = Tensor4(tensor=jnp.arange(81.0).reshape(3, 3, 3, 3))
    Id = jnp.eye(3)

    T_rot = T.rotate(Id)
    assert jnp.allclose(T_rot, T)

    T_ = jnp.arange(36.0).reshape(6, 6)
    T = SymmetricTensor4(array=T_)

    T_rot = T.rotate(Id)

    assert jnp.allclose(T_rot, T)


def test_rotate_rank2_matches_matrix_formula():
    key = jax.random.PRNGKey(0)
    R = random_rotation(key)

    A = jax.random.normal(key, (3, 3))
    T = Tensor2(tensor=A)

    T_rot = T.rotate(R)

    expected = R @ A @ R.T

    assert jnp.allclose(T_rot, expected, atol=1e-6)


def test_trace_invariant():
    key = jax.random.PRNGKey(1)
    R = random_rotation(key)

    A = jax.random.normal(key, (3, 3))
    A = 0.5 * (A + A.T)

    T = SymmetricTensor2(tensor=A)

    T_rot = T.rotate(R)

    assert jnp.allclose(T.tr, T_rot.tr)


def test_isotropic_tensor_rotation_invariant():
    C_ = IsotropicTensor4(kappa=10.0, mu=3.0)
    C = SymmetricTensor4(array=C_.array)

    key = jax.random.PRNGKey(6)
    R = random_rotation(key)

    C_rot = C.rotate(R)

    assert jnp.allclose(C, C_rot, atol=1e-6)


@pytest.mark.parametrize("angles", [(0.3, 0.7, -1.1), (1.2, -0.4, 2.5), (0.0, 0.0, 0.9)])
def test_euler_bunge_is_the_documented_product(angles):
    """R = Rz(phi1) Rx(Phi) Rz(phi2) with active rotations (it used to be the transpose)."""
    phi1, Phi, phi2 = angles
    ez, ex = jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0])
    expected = (
        rotation.from_axis_angle(ez, phi1)
        @ rotation.from_axis_angle(ex, Phi)
        @ rotation.from_axis_angle(ez, phi2)
    )
    R = rotation.from_euler_bunge(*angles)
    assert jnp.allclose(R, expected, atol=1e-14)
    assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-14)
    assert jnp.isclose(jnp.linalg.det(R), 1.0)


def test_euler_bunge_crystal_axes():
    """phi1 = 90 deg: the crystal x axis is the sample y axis, and the stiffness of a crystal
    is expressed in the sample frame by C.rotate(R)."""
    R = rotation.from_euler_bunge(jnp.pi / 2, 0.0, 0.0)
    assert jnp.allclose(R[:, 0], jnp.array([0.0, 1.0, 0.0]), atol=1e-14)
    e1 = SymmetricTensor2(tensor=jnp.diag(jnp.array([1.0, 0.0, 0.0])))  # uniaxial along x_c
    assert jnp.allclose(e1.rotate(R).tensor, jnp.diag(jnp.array([0.0, 1.0, 0.0])), atol=1e-14)


def test_euler_bunge_batched():
    phi1, Phi, phi2 = jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4]), jnp.array([0.5, 0.6])
    R = rotation.from_euler_bunge(phi1, Phi, phi2)
    assert R.shape == (2, 3, 3)
    assert jnp.allclose(R[1], rotation.from_euler_bunge(0.2, 0.4, 0.6))
