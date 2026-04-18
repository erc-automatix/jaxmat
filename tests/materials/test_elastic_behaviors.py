"""
tests/materials/test_elastic_behaviors.py

Unit tests for all linear elastic constitutive behaviors:

    LinearElasticIsotropic
    LinearElasticTransverseIsotropic
    LinearElasticOrthotropic
    LinearElastic (generic)
    ElasticBehavior (SmallStrainBehavior wrapper)

Checks:
- Elastic moduli formulas
- Compliance matrix entries
- Voigt symmetry (C = Cᵀ, major and minor)
- Positive definiteness of C
- C : S = I (inversion round-trip)
- Stress = C : ε consistency
- Isotropy / material symmetry under rotation
- constitutive_update signature, state update, AD (consistent tangent)
"""

import jax
import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2, SymmetricTensor4, rotation

jax.config.update("jax_enable_x64", True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared material parameters
# ─────────────────────────────────────────────────────────────────────────────

E, nu = 200e3, 0.3
EL = 12.0e3
ET = 0.8e3
EN = 1.0e3
nuLT = 0.43
nuLN = 0.47
nuTN = 0.292
muLT = 0.7e3
muLN = 0.9e3
muTN = 0.2e3

nuT_ti = 0.43
nuL_ti = 0.292
muL_ti = 0.7e3

AXIS_Z = jnp.array([0.0, 0.0, 1.0])
AXIS_X = jnp.array([1.0, 0.0, 0.0])

ATOL = 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def isotropic():
    return jm.LinearElasticIsotropic(E=E, nu=nu)


@pytest.fixture
def transverse_isotropic():
    return jm.LinearElasticTransverseIsotropic(
        axis=AXIS_Z, EL=EL, ET=ET, nuT=nuT_ti, nuL=nuL_ti, muL=muL_ti
    )


@pytest.fixture
def orthotropic():
    return jm.LinearElasticOrthotropic(
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


@pytest.fixture
def random_eps():
    key = jax.random.PRNGKey(42)
    arr = jax.random.normal(key, (3, 3))
    return SymmetricTensor2(tensor=0.5 * (arr + arr.T) * 1e-3)


@pytest.fixture
def random_eps_batch():
    key = jax.random.PRNGKey(0)
    arrs = jax.random.normal(key, (5, 3, 3)) * 1e-3
    return SymmetricTensor2(tensor=0.5 * (arrs + arrs.swapaxes(-1, -2)))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _kelvin_identity(n=6):
    """Identity in Kelvin-Mandel space (n×n)."""
    return jnp.eye(n)


def _is_positive_definite(M):
    """True if all eigenvalues of M are positive."""
    eigvals = jnp.linalg.eigvalsh(M)
    return bool(jnp.all(eigvals > 0))


# ─────────────────────────────────────────────────────────────────────────────
# LinearElasticIsotropic — moduli
# ─────────────────────────────────────────────────────────────────────────────


def test_isotropic_shear_modulus(isotropic):
    """μ = E / (2(1+ν))."""
    assert jnp.isclose(isotropic.mu, E / (2 * (1 + nu)), rtol=1e-10)


def test_isotropic_bulk_modulus(isotropic):
    """κ = E / (3(1-2ν))."""
    assert jnp.isclose(isotropic.kappa, E / (3 * (1 - 2 * nu)), rtol=1e-10)


def test_isotropic_lame_modulus(isotropic):
    """λ = E ν / ((1+ν)(1-2ν))."""
    lmbda_expected = E * nu / ((1 + nu) * (1 - 2 * nu))
    assert jnp.isclose(isotropic.lmbda, lmbda_expected, rtol=1e-10)


def test_isotropic_C_symmetry(isotropic):
    """Stiffness tensor must be symmetric (major and minor symmetries)."""
    C = isotropic.C.array
    assert jnp.allclose(C, C.T, atol=ATOL)


def test_isotropic_C_positive_definite(isotropic):
    assert _is_positive_definite(isotropic.C.array)


def test_isotropic_compliance_roundtrip(isotropic):
    """C : S = I in Kelvin space."""
    CS = isotropic.C.array @ isotropic.S.array
    assert jnp.allclose(CS, _kelvin_identity(), atol=ATOL)


def test_isotropic_uniaxial_stress(isotropic):
    """Uniaxial strain ε₁₁=ε → σ₁₁ = λ ε (1-2ν)/(1-ν)·E·... use closed form."""
    eps_val = 1e-3
    eps = SymmetricTensor2(tensor=jnp.diag(jnp.array([eps_val, 0.0, 0.0])))
    sig = isotropic.C @ eps
    # σ = C:ε → σ₁₁ = (λ + 2μ) ε₁₁
    lam, mu = isotropic.lmbda, isotropic.mu
    expected_11 = (lam + 2 * mu) * eps_val
    expected_22 = lam * eps_val
    assert jnp.isclose(jnp.asarray(sig)[0, 0], expected_11, rtol=1e-8)
    assert jnp.isclose(jnp.asarray(sig)[1, 1], expected_22, rtol=1e-8)
    assert jnp.isclose(jnp.asarray(sig)[2, 2], expected_22, rtol=1e-8)


def test_isotropic_hydrostatic_stress(isotropic):
    """Hydrostatic strain ε = δ I → σ = 3κ δ I."""
    delta = 1e-3
    eps = SymmetricTensor2(tensor=delta * jnp.eye(3))
    sig = isotropic.C @ eps
    assert jnp.allclose(jnp.asarray(sig), 3 * isotropic.kappa * delta * jnp.eye(3), atol=ATOL)


def test_isotropic_shear_stress(isotropic):
    """Pure shear ε₁₂ = γ/2 → σ₁₂ = μ γ."""
    gamma = 1e-3
    eps = SymmetricTensor2(tensor=jnp.array([[0, gamma / 2, 0], [gamma / 2, 0, 0], [0, 0, 0]]))
    sig = isotropic.C @ eps
    assert jnp.isclose(jnp.asarray(sig)[0, 1], isotropic.mu * gamma, rtol=1e-8)


def test_isotropic_rotation_invariance(isotropic):
    """Isotropic C must be invariant under any rotation: C.rotate(R) = C."""
    key = jax.random.PRNGKey(1)
    R = rotation.random(key)
    C_rotated = isotropic.C.rotate(R)
    assert jnp.allclose(C_rotated.array, isotropic.C.array, atol=ATOL)


# ─────────────────────────────────────────────────────────────────────────────
# LinearElasticTransverseIsotropic — compliance entries
# ─────────────────────────────────────────────────────────────────────────────


def test_transverse_isotropic_compliance_matrix(transverse_isotropic):
    """
    Compliance matrix S in Kelvin-Mandel ordering [xx, yy, zz, xy, xz, yz]
    with axis = z must match the analytical form.
    """
    elas = transverse_isotropic
    S_expected = jnp.array(
        [
            [1 / ET, -nuT_ti / ET, -nuL_ti / EL, 0, 0, 0],
            [-nuT_ti / ET, 1 / ET, -nuL_ti / EL, 0, 0, 0],
            [-nuL_ti / EL, -nuL_ti / EL, 1 / EL, 0, 0, 0],
            [0, 0, 0, (1 + nuT_ti) / ET, 0, 0],
            [0, 0, 0, 0, 1 / (2 * muL_ti), 0],
            [0, 0, 0, 0, 0, 1 / (2 * muL_ti)],
        ]
    )
    assert jnp.allclose(S_expected, elas.S.array, atol=ATOL)


def test_transverse_isotropic_C_symmetry(transverse_isotropic):
    C = transverse_isotropic.C.array
    assert jnp.allclose(C, C.T, atol=ATOL)


def test_transverse_isotropic_C_positive_definite(transverse_isotropic):
    assert _is_positive_definite(transverse_isotropic.C.array)


def test_transverse_isotropic_compliance_roundtrip(transverse_isotropic):
    CS = transverse_isotropic.C.array @ transverse_isotropic.S.array
    assert jnp.allclose(CS, _kelvin_identity(), atol=ATOL)


def test_transverse_isotropic_symmetry_around_axis(transverse_isotropic):
    """C must be invariant under rotation about the symmetry axis."""
    angle = jnp.pi / 3
    R = rotation.from_axis_angle(AXIS_Z, angle)
    C_dense = transverse_isotropic.C.to_symmetric()
    C_rotated = SymmetricTensor4(array=transverse_isotropic.C.array).rotate(R)
    assert jnp.allclose(C_dense.array, C_rotated.array, atol=ATOL)


def test_transverse_isotropic_breaks_symmetry_off_axis(transverse_isotropic):
    """Rotation about a non-symmetry axis must change C."""
    angle = jnp.pi / 4
    R = rotation.from_axis_angle(AXIS_X, angle)
    C_rotated = transverse_isotropic.C.to_symmetric().rotate(R)
    assert not jnp.allclose(C_rotated.array, transverse_isotropic.C.array, atol=ATOL)


def test_transverse_isotropic_isotropic_limit():
    """When EL=ET, nuL=nuT, muL=E/(2(1+nuT)), should recover isotropic C."""
    E_iso = 10e3
    nu_iso = 0.3
    mu_iso = E_iso / (2 * (1 + nu_iso))
    elas_ti = jm.LinearElasticTransverseIsotropic(
        axis=AXIS_Z, EL=E_iso, ET=E_iso, nuT=nu_iso, nuL=nu_iso, muL=mu_iso
    )
    elas_iso = jm.LinearElasticIsotropic(E=E_iso, nu=nu_iso)
    assert jnp.allclose(elas_ti.C.array, elas_iso.C.array, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# LinearElasticOrthotropic — compliance, rotation, symmetry
# ─────────────────────────────────────────────────────────────────────────────


def test_orthotropic_compliance_roundtrip(orthotropic):
    CS = orthotropic.C.array @ orthotropic.S.array
    assert jnp.allclose(CS, _kelvin_identity(), atol=ATOL)


def test_orthotropic_C_symmetry(orthotropic):
    C = orthotropic.C.array
    assert jnp.allclose(C, C.T, atol=ATOL)


def test_orthotropic_C_positive_definite(orthotropic):
    assert _is_positive_definite(orthotropic.C.array)


def test_orthotropic_rotation_LT_swap():
    """
    90° rotation about z-axis swaps L and T directions.
    The rotated C must equal the C of the L↔T-swapped material
    (with appropriate Poisson ratio reciprocity).
    """
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
    R = rotation.from_axis_angle(AXIS_Z, jnp.pi / 2)
    C_rotated = elasticity.C.rotate(R)
    assert isinstance(C_rotated, SymmetricTensor4)

    # Two 90° rotations = identity
    assert jnp.allclose(C_rotated.rotate(R).array, elasticity.C.array, atol=ATOL)

    # Rotated C matches the L↔T swapped material
    elasticity_swapped = jm.LinearElasticOrthotropic(
        EL=ET,
        ET=EL,
        EN=EN,
        nuLT=nuLT * ET / EL,  # reciprocity: nuTL = ET * nuLT / EL
        nuLN=nuTN,
        nuTN=nuLN,
        muLT=muLT,
        muLN=muTN,
        muTN=muLN,
    )
    assert jnp.allclose(C_rotated.array, elasticity_swapped.C.array, atol=ATOL)


def test_orthotropic_stress_rotation_equivalence(orthotropic, random_eps_batch):
    """
    Test objectivity of the constitutive law.
    The test computes the left-hand side directly via the original material and compares
    it against the right-hand side: rotate the strain into the material frame, apply the
    rotated stiffness, then rotate the resulting stress back.
    Tested for a batch of random strains.
    """
    R = rotation.from_axis_angle(AXIS_Z, jnp.pi / 3)
    mat = jm.ElasticBehavior(elasticity=orthotropic)
    st = mat.init_state()

    elasticity_rotated = jm.LinearElastic(stiffness=orthotropic.C.rotate(R))
    mat_rotated = jm.ElasticBehavior(elasticity=elasticity_rotated)

    def stress(eps_single):
        sig, _ = mat.constitutive_update(eps_single, st, dt=0.0)
        return sig

    def rotate_stress_strain(eps_single):
        # Rotate strain into material frame, apply rotated stiffness, rotate stress back
        eps_local = eps_single.rotate(R)
        sig_local, _ = mat_rotated.constitutive_update(eps_local, st, dt=0.0)
        return sig_local.rotate(R.T)

    sig_a = jax.vmap(stress)(random_eps_batch)
    sig_b = jax.vmap(rotate_stress_strain)(random_eps_batch)
    assert jnp.allclose(jnp.asarray(sig_a), jnp.asarray(sig_b), atol=ATOL)


def test_orthotropic_transverse_isotropy_limit():
    """
    When ET=EN, nuLT=nuLN, muLT=muLN, the orthotropic model should give the
    same C as the transversely isotropic model (axis = L = x direction).
    """
    E_L = 12e3
    E_T = 2e3
    nu_L = 0.3
    nu_T = 0.25
    mu_L = 1e3
    mu_T = E_T / (2 * (1 + nu_T))

    elas_orth = jm.LinearElasticOrthotropic(
        EL=E_L,
        ET=E_T,
        EN=E_T,
        nuLT=nu_L,
        nuLN=nu_L,
        nuTN=nu_T,
        muLT=mu_L,
        muLN=mu_L,
        muTN=mu_T,
    )
    elas_ti = jm.LinearElasticTransverseIsotropic(
        axis=AXIS_X, EL=E_L, ET=E_T, nuT=nu_T, nuL=nu_L, muL=mu_L
    )
    # Compare via SymmetricTensor4 dense form
    C_orth = elas_orth.C.array
    C_ti = elas_ti.C.to_symmetric().array
    assert jnp.allclose(C_orth, C_ti, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# LinearElastic (generic, stiffness-first)
# ─────────────────────────────────────────────────────────────────────────────


def test_generic_elastic_roundtrip_from_isotropic(isotropic):
    """LinearElastic built from C of LinearElasticIsotropic must give same stress."""
    elas_generic = jm.LinearElastic(stiffness=isotropic.C)
    eps = SymmetricTensor2(tensor=jnp.diag(jnp.array([1e-3, -3e-4, -3e-4])))
    sig_iso = isotropic.C @ eps
    sig_generic = elas_generic.C @ eps
    assert jnp.allclose(jnp.asarray(sig_iso), jnp.asarray(sig_generic), atol=ATOL)


def test_generic_elastic_strain_energy(isotropic):
    """ψ = ½ ε:C:ε must be positive for any non-zero strain."""
    eps = SymmetricTensor2(tensor=jnp.diag(jnp.array([1e-3, 0.0, 0.0])))
    psi = isotropic.strain_energy(eps)
    assert float(psi) > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ElasticBehavior — constitutive_update interface
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("elasticity_name", ["isotropic", "transverse_isotropic", "orthotropic"])
def test_elastic_behavior_stress_equals_C_eps(elasticity_name, request, random_eps):
    """constitutive_update must return σ = C:ε for all elastic models."""
    elas = request.getfixturevalue(elasticity_name)
    mat = jm.ElasticBehavior(elasticity=elas)
    st = mat.init_state()
    sig, _ = mat.constitutive_update(random_eps, st, dt=0.0)
    sig_direct = elas.C @ random_eps
    assert jnp.allclose(jnp.asarray(sig), jnp.asarray(sig_direct), atol=ATOL)


@pytest.mark.parametrize("elasticity_name", ["isotropic", "transverse_isotropic", "orthotropic"])
def test_elastic_behavior_state_strain_updated(elasticity_name, request, random_eps):
    """new_state.strain must equal the input strain after the update."""
    elas = request.getfixturevalue(elasticity_name)
    mat = jm.ElasticBehavior(elasticity=elas)
    st = mat.init_state()
    _, new_state = mat.constitutive_update(random_eps, st, dt=0.0)
    assert jnp.allclose(jnp.asarray(new_state.strain), jnp.asarray(random_eps), atol=ATOL)


@pytest.mark.parametrize("elasticity_name", ["isotropic", "transverse_isotropic", "orthotropic"])
def test_elastic_behavior_state_stress_updated(elasticity_name, request, random_eps):
    """new_state.stress must equal σ returned by constitutive_update."""
    elas = request.getfixturevalue(elasticity_name)
    mat = jm.ElasticBehavior(elasticity=elas)
    st = mat.init_state()
    sig, new_state = mat.constitutive_update(random_eps, st, dt=0.0)
    assert jnp.allclose(jnp.asarray(new_state.stress), jnp.asarray(sig), atol=ATOL)


def test_elastic_behavior_zero_strain_zero_stress(isotropic):
    """Zero strain must produce zero stress."""
    mat = jm.ElasticBehavior(elasticity=isotropic)
    st = mat.init_state()
    eps = SymmetricTensor2()
    sig, _ = mat.constitutive_update(eps, st, dt=0.0)
    assert jnp.allclose(jnp.asarray(sig), 0.0, atol=ATOL)


def test_elastic_behavior_consistent_tangent_equals_C(isotropic, random_eps):
    """
    ∂σ/∂ε (consistent tangent) must equal C for a linear elastic model.
    Computed via jax.jacfwd through constitutive_update.
    """
    mat = jm.ElasticBehavior(elasticity=isotropic)
    st = mat.init_state()

    def sig_array(eps_arr):
        eps = SymmetricTensor2(array=eps_arr)
        sig, _ = mat.constitutive_update(eps, st, dt=0.0)
        return sig.array

    C_ad = jax.jit(jax.jacfwd(sig_array))(random_eps.array)
    assert jnp.allclose(C_ad, isotropic.C.array, atol=1e-6)


def test_elastic_behavior_tangent_finite_for_anisotropic(orthotropic, random_eps):
    """Consistent tangent must be finite for orthotropic model."""
    mat = jm.ElasticBehavior(elasticity=orthotropic)
    st = mat.init_state()

    def sig_array(eps_arr):
        sig, _ = mat.constitutive_update(SymmetricTensor2(array=eps_arr), st, dt=0.0)
        return sig.array

    C_ad = jax.jit(jax.jacfwd(sig_array))(random_eps.array)
    assert jnp.all(jnp.isfinite(C_ad))


def test_elastic_behavior_grad_wrt_material_params(random_eps):
    """Gradient of σ₁₁ w.r.t. Young's modulus must be finite and non-zero."""
    st = jm.ElasticBehavior(elasticity=jm.LinearElasticIsotropic(E=200e3, nu=0.3)).init_state()

    def sig11(E):
        mat = jm.ElasticBehavior(elasticity=jm.LinearElasticIsotropic(E=E, nu=0.3))
        sig, _ = mat.constitutive_update(random_eps, st, dt=0.0)
        return jnp.asarray(sig)[0, 0]

    g = jax.grad(sig11)(jnp.array(200e3))
    assert jnp.isfinite(g)
    assert not jnp.isclose(g, 0.0)


def test_elastic_behavior_vmap_batch(isotropic, random_eps_batch):
    """constitutive_update vmapped over a batch of strains must give σ = C:ε."""
    mat = jm.ElasticBehavior(elasticity=isotropic)
    st = mat.init_state()

    def update(eps_single):
        sig, _ = mat.constitutive_update(eps_single, st, dt=0.0)
        return sig

    sig_batch = jax.vmap(update)(random_eps_batch)
    sig_expected = jax.vmap(lambda e: isotropic.C @ e)(random_eps_batch)
    assert jnp.allclose(jnp.asarray(sig_batch), jnp.asarray(sig_expected), atol=ATOL)
