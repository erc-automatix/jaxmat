import jax
import jax.numpy as jnp

from jaxmat.tensors import (
    IsotropicTensor4,
    SymmetricTensor2,
    SymmetricTensor4,
    Tensor2,
    Tensor4,
)


def random_rotation(key):
    """Generate a proper orthogonal 3×3 rotation matrix."""
    A = jax.random.normal(key, (3, 3))
    Q, R = jnp.linalg.qr(A)
    # enforce det = +1
    Q = Q * jnp.sign(jnp.linalg.det(Q))
    return Q


def test_rotate_identity_rank2():
    T = Tensor2(tensor=jnp.arange(9.0).reshape(3, 3))
    I = jnp.eye(3)

    T_rot = T.rotate(I)

    assert jnp.allclose(T_rot, T)


def test_rotate_identity_rank4():
    T = Tensor4(tensor=jnp.arange(81.0).reshape(3, 3, 3, 3))
    I = jnp.eye(3)

    T_rot = T.rotate(I)
    assert jnp.allclose(T_rot, T)

    T_ = jnp.arange(36.0).reshape(6, 6)
    T = SymmetricTensor4(array=T_)

    T_rot = T.rotate(I)

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
