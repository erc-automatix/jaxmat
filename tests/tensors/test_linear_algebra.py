import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl

import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2, Tensor2
from jaxmat.tensors.linear_algebra import (
    eig33,
    eigenvalue_function,
    expm,
    inv33,
    inv_sqrtm,
    isotropic_function,
    logm,
    powm,
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


# --------------------------------------------------------------------------------------------
#  Derivatives of spectral functions, also at repeated eigenvalues
# --------------------------------------------------------------------------------------------

Q = jnp.array(random_rotations(jax.random.PRNGKey(3)))
SPECTRA = {
    "distinct": [2.0, 1.0, 0.5],
    "two equal": [2.0, 1.0, 1.0],
    "nearly two equal": [2.0, 1.0, 1.0 + 1e-7],
    "all equal": [1.3, 1.3, 1.3],
}
U = jnp.array([[0.2, 0.5, 0.1], [0.5, -0.3, 0.6], [0.1, 0.6, 0.4]])
V = jnp.array([[-0.4, 0.1, 0.3], [0.1, 0.2, -0.5], [0.3, -0.5, 0.1]])


def rotated(eigenvalues):
    A = Q @ jnp.diag(jnp.array(eigenvalues)) @ Q.T
    return 0.5 * (A + A.T)


def directional(fun, x, dx, h):
    """Central finite difference of fun at x along dx."""
    return (fun(x + h * dx) - fun(x - h * dx)) / (2 * h)


def relative_error(a, b):
    return float(jnp.abs(a - b).max() / jnp.abs(b).max())


def power(A):
    return powm(A, 1.7)


SECOND_DERIVATIVES = {  # f'' of the scalar functions
    logm: lambda x: -1 / x**2,
    expm: jnp.exp,
    power: lambda x: 1.7 * 0.7 * x ** (1.7 - 2),
}


@pytest.mark.parametrize("spectrum", SPECTRA)
@pytest.mark.parametrize("fun", [logm, expm, power], ids=["logm", "expm", "powm"])
def test_matrix_function_derivatives(fun, spectrum):
    """First and second derivatives of f(A) against finite differences of the values and of
    the first derivative. Differentiating through the eigendecomposition fails at repeated
    eigenvalues, e.g. by 100% for the derivative of logm at a multiple of the identity."""
    A = rotated(SPECTRA[spectrum])

    def first(A_):
        return jax.jvp(fun, (A_,), (U,))[1]

    assert relative_error(first(A), directional(fun, A, U, 1e-6)) < 1e-7
    second = jax.jvp(first, (A,), (V,))[1]
    if spectrum == "all equal":
        # exact: D2f(a I)[U, V] = f''(a) / 2 (U V + V U)
        a = SPECTRA[spectrum][0]
        expected = 0.5 * SECOND_DERIVATIVES[fun](a) * (U @ V + V @ U)
        assert relative_error(second, expected) < 1e-12
    else:
        assert relative_error(second, directional(first, A, V, 1e-5)) < 1e-7
    # the same second derivative by reverse mode, as for a stress defined by a gradient
    hessian = jax.hessian(lambda A_: jnp.sum(U * fun(A_)))(A)
    assert relative_error(jnp.einsum("ijkl,kl->ij", hessian, V), second) < 1e-9


@pytest.mark.parametrize("spectrum", SPECTRA)
def test_matrix_function_derivatives_with_respect_to_arguments(spectrum):
    """Derivatives of A**m with respect to m, alone and mixed with A."""
    A = rotated(SPECTRA[spectrum])
    d_dm = jax.jvp(lambda m: powm(A, m), (1.7,), (1.0,))[1]
    assert relative_error(d_dm, directional(lambda m: powm(A, m), 1.7, 1.0, 1e-6)) < 1e-7

    def d_dA(m):
        return jax.jvp(lambda A_: powm(A_, m), (A,), (U,))[1]

    mixed = jax.jvp(d_dA, (1.7,), (1.0,))[1]
    assert relative_error(mixed, directional(d_dA, 1.7, 1.0, 1e-5)) < 1e-7


def smooth_symmetric(eigenvalues, c):
    """A symmetric, non-separable function of the eigenvalues, with a parameter."""
    l1, l2, l3 = eigenvalues
    spread = (l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2
    return c * jnp.sum(eigenvalues**3) + l1 * l2 * l3 + spread**2


@pytest.mark.parametrize("spectrum", SPECTRA)
def test_eigenvalue_function_derivatives(spectrum):
    A = rotated(SPECTRA[spectrum])
    c = 0.7

    def grad(A_, c_=c):
        return jax.grad(eigenvalue_function, argnums=1)(smooth_symmetric, A_, c_)

    assert jnp.isclose(
        jnp.sum(grad(A) * U),
        directional(lambda A_: eigenvalue_function(smooth_symmetric, A_, c), A, U, 1e-6),
        rtol=1e-7,
    )
    second = jax.jvp(grad, (A,), (V,))[1]
    assert relative_error(second, directional(grad, A, V, 1e-5)) < 1e-7
    mixed = jax.jvp(lambda c_: grad(A, c_), (c,), (1.0,))[1]
    assert relative_error(mixed, directional(lambda c_: grad(A, c_), c, 1.0, 1e-5)) < 1e-7


@pytest.mark.parametrize("stretches", [[1.0, 1.0, 1.0], [1.2, 0.95, 0.95], [1.2, 0.95, 0.9]])
def test_ogden_tangent(stretches):
    """The Ogden tangent at repeated principal stretches (it was off by 1% at F = I)."""
    mu, alpha = jnp.array([4.0, 1.0]), jnp.array([2.0, -2.0])
    material = jm.CompressibleOgden(mu=mu, alpha=alpha, kappa=1e3)
    F = (Q @ jnp.diag(jnp.array(stretches))).ravel()

    def PK1(F_):
        return material.PK1(Tensor2(tensor=F_.reshape(3, 3))).tensor.ravel()

    tangent = jax.jacfwd(PK1)(F)
    fd = jnp.stack([directional(PK1, F, e, 1e-6) for e in jnp.eye(9)], axis=-1)
    assert relative_error(tangent, fd) < 1e-7


@pytest.mark.parametrize("principal", [[300.0, 0.0, 0.0], [300.0, 100.0, -50.0]])
def test_hosford_normal_derivative(principal):
    """The derivative of the Hosford normal (consistent tangent) at a uniaxial stress, with two
    equal principal stresses (it was off by 50%)."""
    surface = jm.Hosford(a=8.0)
    sig = SymmetricTensor2(tensor=rotated(principal)).array

    def normal(s):
        return jax.grad(lambda x: surface(SymmetricTensor2(array=x)))(s)

    derivative = jax.jacfwd(normal)(sig)
    fd = jnp.stack([directional(normal, sig, e, 1e-4) for e in jnp.eye(6)], axis=-1)
    assert relative_error(derivative, fd) < 1e-7
