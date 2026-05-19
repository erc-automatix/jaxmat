import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl

from jaxmat.tensors.linear_algebra import (
    eig33,
    inv33,
    inv_sqrtm,
    isotropic_function,
    sqrtm,
)
from jaxmat.tensors.rotation import random as random_rotations


def build_matrix_from_diag_and_rot(diag, R):
    """
    Construct symmetric matrix with prescribed eigenvalues:

        A = Rᵀ D R
    """
    D = jnp.diag(diag)
    return R.T @ D @ R


batch_build_A = jax.jit(jax.vmap(build_matrix_from_diag_and_rot, in_axes=(None, 0)))

batch_eigvals = jax.jit(jax.vmap(eig33))


diags_rand = jnp.array(np.random.default_rng().random((3, 3)))
diags_two = jnp.array([[1, -0.5 + eps / 2, -0.5 - eps / 2] for eps in np.logspace(-3, -15, num=10)])
diags_two = jnp.array([[1, -0.5 + eps / 2, -0.5 - eps / 2] for eps in np.logspace(-3, -15, num=10)])
diags_three = jnp.array([[1, 1, 1]])  # , [0, 0, 0]])
diags = np.vstack((diags_rand, diags_two, diags_three))


@pytest.fixture(name="diagonal", params=diags)
def fixture_diagonal(request):
    return request.param


@pytest.fixture(name="rotations")
def fixture_rotations():
    key = jax.random.PRNGKey(0)
    return random_rotations(key, shape=(10,))


def test_eig33_reconstruction_fast(diagonal, rotations):
    # Build batch of symmetric matrices
    A_batch = batch_build_A(diagonal, rotations)  # shape (B, 3, 3)

    # Vectorized eigen-decomposition
    batch_eig33 = jax.jit(jax.vmap(eig33))
    eigvals_batch, dyads_batch = batch_eig33(A_batch)

    # Reconstruct all matrices in one shot
    A_reconstructed = jnp.einsum("bi,bijk->bjk", eigvals_batch, dyads_batch)

    # Sort eigenvalues for comparison
    diagonal_sorted = jnp.sort(diagonal)

    # Check eigenvalues
    assert jnp.allclose(jnp.sort(eigvals_batch, axis=-1), diagonal_sorted, atol=1e-6)

    # Check reconstruction
    assert jnp.allclose(A_batch, A_reconstructed, atol=1e-6)


@pytest.mark.parametrize(
    ("matrix_fun", "scalar_fun"),
    [(sl.expm, jnp.exp), (sl.logm, jnp.log)],
)
def test_isotropic_function(matrix_fun, scalar_fun, diagonal, rotations):
    A_batch = batch_build_A(jnp.abs(diagonal), rotations)
    for A in A_batch:
        fA = matrix_fun(A)
        fA_ = jax.jit(isotropic_function, static_argnums=0)(scalar_fun, A)
        assert jnp.allclose(fA, fA_)


def isqrt(x):
    return 1 / jnp.sqrt(x)


def test_sqrtm(diagonal, rotations):
    A_batch = batch_build_A(jnp.abs(diagonal), rotations)
    for A in A_batch:
        fA = jax.jit(sqrtm)(A)
        fA_ = jax.jit(isotropic_function, static_argnums=0)(jnp.sqrt, A)
        assert jnp.allclose(fA, fA_)
        fA = jax.jit(inv_sqrtm)(A)

        fA_ = jax.jit(isotropic_function, static_argnums=0)(isqrt, A)
        assert jnp.allclose(fA, fA_)


def test_inv33():
    A_batch = [jnp.array(np.random.default_rng().random((3, 3))) for _ in range(3)]
    for A in A_batch:
        iA = jax.jit(inv33)(A)
        iA_ = jnp.linalg.inv(A)
        assert jnp.allclose(iA, iA_)


def test_positive_part_tensor(rotations):
    diag = jnp.asarray([2.99999907618e-4, -1.499999988237e-4, 0.0])
    A_batch = batch_build_A(diag, rotations)
    for A in A_batch:
        A_pos = isotropic_function(lambda x: jnp.maximum(x, 0.0), A)
        A_neg = isotropic_function(lambda x: jnp.minimum(x, 0.0), A)
        assert jnp.all(eig33(A_pos)[0] >= -1e-16)
        assert jnp.allclose(A_pos + A_neg, A)
        assert jnp.isclose(jnp.tensordot(A_pos, A_neg), 0.0)
