"""
tests/tensors/test_math_utils.py

Tests for
    jaxmat.tensors.utils          — safe_sqrt, safe_norm, safe_fun, FischerBurmeister
    jaxmat.tensors.linear_algebra — det33, inv33, eig33, principal/main invariants,
                                    isotropic_function, sqrtm, inv_sqrtm, expm, logm, powm
    jaxmat.tensors.tensor_utils   — sym, skw, vol, dev, tr, norm, von_mises, axl,
                                    polar, stretch_tensor, eigenvalues
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sl

from jaxmat.tensors import (
    SymmetricTensor2,
    Tensor2,
    axl,
    dev,
    eigenvalues,
    inv_sqrtm,
    norm,
    polar,
    skw,
    sqrtm,
    stretch_tensor,
    sym,
    tr,
    vol,
    von_mises,
)
from jaxmat.tensors.linear_algebra import (
    det33,
    eig33,
    expm,
    inv33,
    isotropic_function,
    logm,
    main_invariants,
    powm,
    principal_invariants,
)
from jaxmat.tensors.rotation import random as random_rotations
from jaxmat.tensors.utils import FischerBurmeister, safe_fun, safe_norm, safe_sqrt


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rotations():
    key = jax.random.PRNGKey(0)
    return random_rotations(key, shape=(12,))


def _spd(key, n=3):
    A = jax.random.normal(key, (n, n))
    return A @ A.T + jnp.eye(n)


def _sym(key, n=3):
    A = jax.random.normal(key, (n, n))
    return 0.5 * (A + A.T)


def _build(eigvals, R):
    """Symmetric matrix with prescribed eigenvalues: A = Rᵀ diag(λ) R."""
    return R.T @ jnp.diag(eigvals) @ R


# ─────────────────────────────────────────────────────────────────────────────
# safe_sqrt
# ─────────────────────────────────────────────────────────────────────────────


def test_safe_sqrt_positive():
    assert jnp.isclose(safe_sqrt(4.0), 2.0)


def test_safe_sqrt_at_zero_returns_eps():
    eps = 1e-16
    assert jnp.isclose(safe_sqrt(0.0, eps=eps), eps)


def test_safe_sqrt_below_eps_returns_eps():
    eps = 1e-8
    assert jnp.isclose(safe_sqrt(eps / 10, eps=eps), eps)


def test_safe_sqrt_gradient_at_zero_is_finite():
    assert jnp.isfinite(jax.grad(safe_sqrt)(0.0))


def test_safe_sqrt_gradient_at_zero_is_zero():
    assert jnp.isclose(jax.grad(safe_sqrt)(0.0), 0.0)


def test_safe_sqrt_gradient_positive():
    x = 4.0
    assert jnp.isclose(jax.grad(safe_sqrt)(x), 0.5 / jnp.sqrt(x))


def test_safe_sqrt_batch():
    x = jnp.array([0.0, 1.0, 4.0, 9.0])
    result = jax.vmap(safe_sqrt)(x)
    assert jnp.allclose(result[1:], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.isfinite(result[0])


# ─────────────────────────────────────────────────────────────────────────────
# safe_fun
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("x", [0.0, 1e-20, 1e-16, 1e-8, 1.0, 4.0])
def test_safe_fun_consistent_with_safe_sqrt_values(x):
    """safe_fun(jnp.sqrt, x) must match safe_sqrt(x) for all x."""
    assert jnp.isclose(safe_fun(jnp.sqrt, x), safe_sqrt(x), atol=1e-10)


@pytest.mark.parametrize("x", [0.0, 1e-20, 1.0, 4.0])
def test_safe_fun_consistent_with_safe_sqrt_gradients(x):
    gf = jax.grad(lambda x: safe_fun(jnp.sqrt, x))(x)
    gs = jax.grad(safe_sqrt)(x)
    assert jnp.isclose(gf, gs, atol=1e-10)


def test_safe_fun_returns_zero_below_eps():
    assert jnp.isclose(safe_fun(jnp.sqrt, 0.0), 0.0)


def test_safe_fun_positive():
    assert jnp.isclose(safe_fun(jnp.sqrt, 9.0), 3.0)


def test_safe_fun_gradient_at_zero_is_finite():
    assert jnp.isfinite(jax.grad(lambda x: safe_fun(jnp.sqrt, x))(0.0))


def test_safe_fun_custom_norm_small():
    v = jnp.array([1e-20, 1e-20, 1e-20])
    assert jnp.isclose(safe_fun(jnp.linalg.norm, v, norm=jnp.linalg.norm), 0.0)


def test_safe_fun_custom_norm_large():
    v = jnp.array([3.0, 4.0, 0.0])
    assert jnp.isclose(safe_fun(jnp.linalg.norm, v, norm=jnp.linalg.norm), 5.0)


def test_safe_fun_custom_eps():
    assert jnp.isclose(safe_fun(jnp.sqrt, 1e-5, eps=1e-4), 0.0)
    assert jnp.isclose(safe_fun(jnp.sqrt, 1.0, eps=1e-4), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# safe_norm
# ─────────────────────────────────────────────────────────────────────────────


def test_safe_norm_zero_is_finite_and_nonneg():
    r = safe_norm(jnp.zeros(3))
    assert jnp.isfinite(r) and r >= 0.0


def test_safe_norm_gradient_at_zero_finite():
    assert jnp.all(jnp.isfinite(jax.grad(safe_norm)(jnp.zeros(3))))


def test_safe_norm_unit_vector():
    assert jnp.isclose(safe_norm(jnp.array([1.0, 0.0, 0.0])), 1.0)


def test_safe_norm_known():
    assert jnp.isclose(safe_norm(jnp.array([3.0, 4.0, 0.0])), 5.0)


def test_safe_norm_axis_kwarg():
    M = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    assert jnp.allclose(safe_norm(M, axis=1), jnp.array([5.0, 5.0]))


# ─────────────────────────────────────────────────────────────────────────────
# FischerBurmeister
# ─────────────────────────────────────────────────────────────────────────────


def test_fischer_burmeister():
    assert jnp.isclose(FischerBurmeister(0.0, 0.0), 0.0, atol=1e-6)
    assert jnp.isclose(FischerBurmeister(1.0, 1.0), 2.0 - jnp.sqrt(2.0))
    assert jnp.isclose(FischerBurmeister(0.0, 1.0), 0.0, atol=1e-6)
    assert jnp.isclose(FischerBurmeister(1.0, 0.0), 0.0, atol=1e-6)
    assert jnp.isfinite(jax.grad(lambda x: FischerBurmeister(x, 1.0))(0.0))
    assert jnp.isfinite(jax.grad(lambda y: FischerBurmeister(1.0, y))(0.0))


# ─────────────────────────────────────────────────────────────────────────────
# det33 / inv33
# ─────────────────────────────────────────────────────────────────────────────


def test_det33():
    assert jnp.isclose(det33(jnp.eye(3)), 1.0)
    assert jnp.isclose(det33(jnp.diag(jnp.array([2.0, 3.0, 4.0]))), 24.0)
    A = jax.random.normal(jax.random.PRNGKey(0), (3, 3))
    assert jnp.isclose(det33(A), jnp.linalg.det(A), rtol=1e-5)
    assert jnp.isclose(det33(jnp.zeros((3, 3))), 0.0)


def test_inv33():
    assert jnp.allclose(inv33(jnp.eye(3)), jnp.eye(3))
    A = jax.random.normal(jax.random.PRNGKey(1), (3, 3)) + 3.0 * jnp.eye(3)
    assert jnp.allclose(A @ inv33(A), jnp.eye(3), atol=1e-5)
    assert jnp.allclose(inv33(A), jnp.linalg.inv(A), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# eig33
# ─────────────────────────────────────────────────────────────────────────────


def test_eig33_identity():
    eigvals, _ = eig33(jnp.eye(3))
    assert jnp.allclose(jnp.sort(eigvals), jnp.ones(3))


def test_eig33_diagonal():
    diag = jnp.array([3.0, 1.0, 2.0])
    eigvals, _ = eig33(jnp.diag(diag))
    assert jnp.allclose(jnp.sort(eigvals), jnp.sort(diag), atol=1e-6)


def test_eig33_reconstruction_distinct(rotations):
    diag = jnp.array([1.0, 2.0, 3.0])
    for R in rotations:
        A = _build(diag, R)
        eigvals, dyads = eig33(A)
        assert jnp.allclose(A, jnp.einsum("i,ijk->jk", eigvals, dyads), atol=1e-5)


@pytest.mark.parametrize("eps", [1e-3, 1e-6, 1e-10])
def test_eig33_reconstruction_two_equal(rotations, eps):
    diag = jnp.array([1.0, -0.5 + eps / 2, -0.5 - eps / 2])
    for R in rotations[:3]:
        A = _build(diag, R)
        eigvals, dyads = eig33(A)
        assert jnp.allclose(A, jnp.einsum("i,ijk->jk", eigvals, dyads), atol=1e-5)


def test_eig33_reconstruction_isotropic(rotations):
    diag = jnp.array([2.0, 2.0, 2.0])
    for R in rotations[:3]:
        A = _build(diag, R)
        eigvals, dyads = eig33(A)
        assert jnp.allclose(A, jnp.einsum("i,ijk->jk", eigvals, dyads), atol=1e-5)


def test_eig33_gradient_finite():
    g = jax.jacobian(lambda A: eig33(A)[0])(_spd(jax.random.PRNGKey(3)))
    assert jnp.all(jnp.isfinite(g))


def test_eig33_gradient_finite_repeated():
    g = jax.jacobian(lambda A: eig33(A)[0])(2.0 * jnp.eye(3))
    assert jnp.all(jnp.isfinite(g))


def test_eig33_vmap(rotations):
    diag = jnp.array([1.0, 2.0, 3.0])
    A_batch = jax.vmap(_build, in_axes=(None, 0))(diag, rotations)
    eigvals_batch, _ = jax.vmap(eig33)(A_batch)
    expected = jnp.broadcast_to(jnp.sort(diag), eigvals_batch.shape)
    assert jnp.allclose(jnp.sort(eigvals_batch, axis=-1), expected, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_principal_invariants():
    I1, I2, I3 = principal_invariants(jnp.eye(3))
    assert jnp.isclose(I1, 3.0) and jnp.isclose(I2, 3.0) and jnp.isclose(I3, 1.0)
    A = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    I1, I2, I3 = principal_invariants(A)
    assert jnp.isclose(I1, 6.0) and jnp.isclose(I2, 11.0) and jnp.isclose(I3, 6.0)


def test_cayley_hamilton():
    """A³ - I1 A² + I2 A - I3 I = 0."""
    A = _sym(jax.random.PRNGKey(4))
    I1, I2, I3 = principal_invariants(A)
    CH = A @ A @ A - I1 * A @ A + I2 * A - I3 * jnp.eye(3)
    assert jnp.allclose(CH, jnp.zeros((3, 3)), atol=1e-5)


def test_main_invariants():
    J1, J2, J3 = main_invariants(jnp.eye(3))
    assert jnp.isclose(J1, 3.0) and jnp.isclose(J2, 3.0) and jnp.isclose(J3, 3.0)
    A = jnp.diag(jnp.array([1.0, 2.0, 3.0]))
    J1, J2, J3 = main_invariants(A)
    assert jnp.isclose(J1, 6.0) and jnp.isclose(J2, 14.0) and jnp.isclose(J3, 36.0)


# ─────────────────────────────────────────────────────────────────────────────
# Isotropic functions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "scalar_fun,scipy_fun",
    [
        (jnp.exp, sl.expm),
        (jnp.log, sl.logm),
    ],
)
def test_isotropic_function_matches_scipy(scalar_fun, scipy_fun, rotations):
    diag = jnp.array([1.0, 2.0, 3.0])
    for R in rotations[:4]:
        A = _build(diag, R)
        assert jnp.allclose(
            isotropic_function(scalar_fun, A),
            jnp.array(np.real(scipy_fun(np.array(A)))),
            atol=1e-5,
        )


def test_isotropic_function_diagonal():
    A = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
    assert jnp.allclose(
        isotropic_function(jnp.sqrt, A),
        jnp.diag(jnp.array([1.0, 2.0, 3.0])),
        atol=1e-5,
    )


def test_isotropic_function_accepts_tensor():
    """SymmetricTensor2 accepted via __jax_array__."""
    result = isotropic_function(jnp.exp, SymmetricTensor2(tensor=jnp.eye(3)))
    assert jnp.allclose(result, jnp.diag(jnp.full(3, jnp.e)), atol=1e-5)


def test_sqrtm_identity():
    assert jnp.allclose(sqrtm(jnp.eye(3)), jnp.eye(3), atol=1e-5)


def test_sqrtm_round_trip(rotations):
    diag = jnp.array([1.0, 2.0, 4.0])
    for R in rotations[:4]:
        A = _build(diag, R)
        U = sqrtm(A)
        assert jnp.allclose(U @ U, A, atol=1e-5)


def test_sqrtm_consistent_with_isotropic_function(rotations):
    A = _build(jnp.array([1.0, 2.0, 3.0]), rotations[0])
    assert jnp.allclose(sqrtm(A), isotropic_function(jnp.sqrt, A), atol=1e-5)


def test_sqrtm_accepts_tensor():
    assert jnp.allclose(sqrtm(SymmetricTensor2(tensor=jnp.eye(3))), jnp.eye(3), atol=1e-5)


def test_inv_sqrtm_round_trip(rotations):
    """A^{-1/2} A A^{-1/2} = I."""
    for R in rotations[:4]:
        A = _build(jnp.array([1.0, 2.0, 4.0]), R)
        Anh = inv_sqrtm(A)
        assert jnp.allclose(Anh @ A @ Anh, jnp.eye(3), atol=1e-5)


def test_sqrtm_inv_sqrtm_product(rotations):
    A = _build(jnp.array([1.0, 2.0, 3.0]), rotations[0])
    assert jnp.allclose(sqrtm(A) @ inv_sqrtm(A), jnp.eye(3), atol=1e-5)


def test_expm_logm_round_trip(rotations):
    diag = jnp.array([1.0, 2.0, 3.0])
    for R in rotations[:4]:
        assert jnp.allclose(expm(logm(_build(diag, R))), _build(diag, R), atol=1e-5)


def test_powm_half_matches_sqrtm(rotations):
    A = _build(jnp.array([1.0, 4.0, 9.0]), rotations[0])
    assert jnp.allclose(powm(A, 0.5), sqrtm(A), atol=1e-5)


def test_powm_one_is_identity(rotations):
    A = _build(jnp.array([1.0, 2.0, 3.0]), rotations[0])
    assert jnp.allclose(powm(A, 1.0), A, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# tensor_utils — sym / skw / axl
# ─────────────────────────────────────────────────────────────────────────────


def test_sym_returns_symmetric_type():
    A = Tensor2(tensor=_sym(jax.random.PRNGKey(10)))
    assert isinstance(sym(A), SymmetricTensor2)


def test_sym_skw_partition():
    A = Tensor2(tensor=jax.random.normal(jax.random.PRNGKey(11), (3, 3)))
    assert jnp.allclose(jnp.asarray(sym(A)) + jnp.asarray(skw(A)), jnp.asarray(A), atol=1e-6)


def test_skw_is_antisymmetric():
    A = Tensor2(tensor=jax.random.normal(jax.random.PRNGKey(12), (3, 3)))
    W = skw(A)
    assert jnp.allclose(W.tensor + W.tensor.T, jnp.zeros((3, 3)), atol=1e-6)


def test_axl_round_trip():
    w = jnp.array([1.0, 2.0, 3.0])
    W = Tensor2(
        tensor=jnp.array(
            [
                [0.0, -w[2], w[1]],
                [w[2], 0.0, -w[0]],
                [-w[1], w[0], 0.0],
            ]
        )
    )
    assert jnp.allclose(axl(W), w, atol=1e-6)


def test_axl_finite():
    A = Tensor2(tensor=jax.random.normal(jax.random.PRNGKey(13), (3, 3)))
    assert jnp.all(jnp.isfinite(axl(A)))


# ─────────────────────────────────────────────────────────────────────────────
# tensor_utils — tr / vol / dev
# ─────────────────────────────────────────────────────────────────────────────


def test_tr_identity():
    assert jnp.isclose(tr(SymmetricTensor2.identity()), 3.0)


def test_tr_diagonal():
    A = SymmetricTensor2(tensor=jnp.diag(jnp.array([1.0, 2.0, 3.0])))
    assert jnp.isclose(tr(A), 6.0)


def test_vol_plus_dev_equals_input():
    A = SymmetricTensor2(tensor=_sym(jax.random.PRNGKey(20)))
    assert jnp.allclose((vol(A) + dev(A)).tensor, A.tensor, atol=1e-6)


def test_dev_is_traceless():
    A = SymmetricTensor2(tensor=_sym(jax.random.PRNGKey(21)))
    assert jnp.isclose(tr(dev(A)), 0.0, atol=1e-6)


def test_vol_tr_consistent():
    A = SymmetricTensor2(tensor=_sym(jax.random.PRNGKey(22)))
    assert jnp.isclose(tr(vol(A)), tr(A), atol=1e-6)


def test_dev_returns_sym2():
    assert isinstance(dev(SymmetricTensor2.identity()), SymmetricTensor2)


# ─────────────────────────────────────────────────────────────────────────────
# tensor_utils — norm / von_mises
# ─────────────────────────────────────────────────────────────────────────────


def test_norm_identity():
    assert jnp.isclose(norm(SymmetricTensor2.identity()), jnp.sqrt(3.0), atol=1e-6)


def test_norm_zero():
    assert jnp.isclose(norm(SymmetricTensor2()), 0.0, atol=1e-6)


def test_norm_nonneg():
    A = SymmetricTensor2(tensor=_sym(jax.random.PRNGKey(30)))
    assert norm(A) >= 0.0


def test_norm_gradient_finite():
    A = SymmetricTensor2(tensor=_sym(jax.random.PRNGKey(31)))
    g = jax.grad(lambda a: norm(SymmetricTensor2(array=a)))(A.array)
    assert jnp.all(jnp.isfinite(g))


def test_von_mises_hydrostatic_is_zero():
    sig = SymmetricTensor2(tensor=3.0 * jnp.eye(3))
    assert jnp.isclose(von_mises(sig), 0.0, atol=1e-6)


def test_von_mises_uniaxial():
    """σ_VM = s for uniaxial tension σ_11 = s."""
    s = 2.0
    sig = SymmetricTensor2(tensor=jnp.diag(jnp.array([s, 0.0, 0.0])))
    assert jnp.isclose(von_mises(sig), s, atol=1e-5)


def test_von_mises_pure_shear():
    """σ_VM = √3 τ for pure shear τ_12 = τ."""
    t = 1.5
    sig = SymmetricTensor2(tensor=jnp.array([[0.0, t, 0.0], [t, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert jnp.isclose(von_mises(sig), jnp.sqrt(3.0) * t, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# tensor_utils — polar / stretch_tensor
# ─────────────────────────────────────────────────────────────────────────────


def _simple_F(gamma=0.5):
    return SymmetricTensor2.identity() + Tensor2(
        tensor=jnp.array([[0.0, gamma, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )


def test_polar_RU_reconstruction():
    R, U = polar(_simple_F(), mode="RU")
    assert jnp.allclose(jnp.asarray(R @ U), jnp.asarray(_simple_F()), atol=1e-5)


def test_polar_VR_reconstruction():
    V, R = polar(_simple_F(), mode="VR")
    assert jnp.allclose(jnp.asarray(V @ R), jnp.asarray(_simple_F()), atol=1e-5)


def test_polar_R_orthogonal():
    R, _ = polar(_simple_F())
    assert jnp.allclose(jnp.asarray(R).T @ jnp.asarray(R), jnp.eye(3), atol=1e-5)


def test_polar_U_is_symmetric():
    _, U = polar(_simple_F())
    assert isinstance(U, SymmetricTensor2)


def test_polar_U_squared_equals_C():
    F = _simple_F()
    _, U = polar(F)
    C = jnp.asarray((F.T @ F).sym)
    assert jnp.allclose(jnp.asarray(U) @ jnp.asarray(U), C, atol=1e-5)


def test_polar_same_R_both_modes():
    F = _simple_F()
    R1, _ = polar(F, mode="RU")
    _, R2 = polar(F, mode="VR")
    assert jnp.allclose(jnp.asarray(R1), jnp.asarray(R2), atol=1e-5)


def test_stretch_tensor_matches_polar():
    F = _simple_F()
    _, U = polar(F)
    assert jnp.allclose(jnp.asarray(stretch_tensor(F)), jnp.asarray(U), atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# tensor_utils — eigenvalues (custom JVP)
# ─────────────────────────────────────────────────────────────────────────────


def test_eigenvalues_known_spectrum():
    A = SymmetricTensor2(tensor=jnp.diag(jnp.array([3.0, 1.0, 2.0])))
    ev = eigenvalues(A)
    assert jnp.allclose(jnp.sort(ev), jnp.array([1.0, 2.0, 3.0]), atol=1e-6)


def test_eigenvalues_jvp_finite_distinct():
    A = SymmetricTensor2(tensor=_spd(jax.random.PRNGKey(50)))
    _, g = jax.jvp(
        lambda a: eigenvalues(SymmetricTensor2(array=a)),
        (A.array,),
        (jnp.ones_like(A.array),),
    )
    assert jnp.all(jnp.isfinite(g))


def test_eigenvalues_jvp_finite_repeated():
    A = SymmetricTensor2(tensor=2.0 * jnp.eye(3))
    _, g = jax.jvp(
        lambda a: eigenvalues(SymmetricTensor2(array=a)),
        (A.array,),
        (jnp.ones_like(A.array),),
    )
    assert jnp.all(jnp.isfinite(g))


def test_eigenvalues_grad_equals_kelvin_identity():
    """d/dA Σᵢ λᵢ = {I} in Kelvin space (since Σᵢ λᵢ = tr A)."""
    A = SymmetricTensor2(tensor=_spd(jax.random.PRNGKey(51)))
    g = jax.grad(lambda a: jnp.sum(eigenvalues(SymmetricTensor2(array=a))))(A.array)
    assert jnp.allclose(g, SymmetricTensor2.identity().array, atol=1e-5)


def test_eigenvalues_rotation_invariant(rotations):
    A = SymmetricTensor2(tensor=_spd(jax.random.PRNGKey(52)))
    ev_ref = jnp.sort(eigenvalues(A))
    for R in rotations[:4]:
        A_rot = SymmetricTensor2(tensor=R @ jnp.asarray(A) @ R.T)
        assert jnp.allclose(jnp.sort(eigenvalues(A_rot)), ev_ref, atol=1e-5)
