"""Finite-strain behaviors obtained from small-strain ones in the logarithmic strain space.

The wrapper is tested on four small-strain behaviors: isotropic and transversely isotropic
elasticity, von Mises plasticity with Voce hardening, and a standard linear solid. Generic
properties (stress-free states, symmetry, objectivity, small-strain limit) are checked for
all of them, then behavior-specific ones against exact or finite-difference references.
"""

import jax
import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2, Tensor2, logm, polar, rotation, tr, von_mises

jax.config.update("jax_enable_x64", True)

DT = jnp.array(1.0)  # an array, so that changing dt does not recompile the update

ELASTICITY = jm.LinearElasticIsotropic(E=210e3, nu=0.3)
TRANSVERSE = jm.LinearElasticTransverseIsotropic(
    axis=jnp.array([1.0, 2.0, 2.0]) / 3, EL=150e3, ET=10e3, nuL=0.3, nuT=0.4, muL=5e3
)
SIG0, SIGU, B = 250.0, 350.0, 20.0

SMALL = {
    "elastic": jm.ElasticBehavior(elasticity=ELASTICITY),
    "anisotropic": jm.ElasticBehavior(elasticity=TRANSVERSE),
    "plastic": jm.vonMisesIsotropicHardening(
        elasticity=ELASTICITY, yield_stress=jm.VoceHardening(sig0=SIG0, sigu=SIGU, b=B)
    ),
    "viscoelastic": jm.StandardLinearSolid(
        elasticity=jm.LinearElasticIsotropic(E=100e3, nu=0.3),
        maxwell_stiffness=jm.LinearElasticIsotropic(E=110e3, nu=0.3),
        maxwell_viscosity=5e4,
    ),
}
FINITE = {name: jm.LogarithmicStrain(small) for name, small in SMALL.items()}

# A general deformation gradient: stretch, shear and rotation
F_GENERAL = Tensor2(tensor=[[1.15, 0.06, -0.02], [0.03, 0.95, 0.04], [-0.01, 0.02, 1.08]])


# --------------------------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------------------------


def frobenius(A):
    """Frobenius norm sqrt(A:A).

    ``jaxmat.tensors.norm`` is not used: being safe for differentiation, it returns 1e-16 for
    any tensor of norm below 1e-8, which would make the tighter tolerances below vacuous.
    """
    return jnp.sqrt(A.double_contract(A))


def diagonal(*values):
    """Diagonal second-rank tensor."""
    return Tensor2(tensor=jnp.diag(jnp.array(values, dtype=jnp.float64)))


def stretch_shear_path(n=7):
    """Increasing stretch and shear, F_k = I + k G."""
    G = Tensor2(tensor=[[0.01, 0.006, 0.0], [0.0, -0.003, 0.0], [0.0, 0.0, 0.0]])
    return [Tensor2.identity() + k * G for k in range(1, n + 1)]


def run_path(material, Fs, dt=DT):
    """States along a path of deformation gradients."""
    state, states = material.init_state(), []
    for F in Fs:
        _, state = material.constitutive_update(F, state, dt)
        states.append(state)
    return states


def tangents(material, F, state, h=1e-6):
    """dPK1/dF by forward AD and by central finite differences, as 9x9 matrices acting on
    the component arrays of ``Tensor2``."""

    def PK1(F_array):
        return material.constitutive_update(Tensor2(array=F_array), state, DT)[0].array

    dF = h * jnp.eye(9)
    A_fd = jnp.stack([(PK1(F.array + d) - PK1(F.array - d)) / (2 * h) for d in dF], axis=-1)
    return jax.jacfwd(PK1)(F.array), A_fd


# --------------------------------------------------------------------------------------------
#  Generic properties, for every small-strain behavior
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FINITE)
def test_zero_stress_at_identity(name):
    material = FINITE[name]
    P, state = material.constitutive_update(Tensor2.identity(), material.init_state(), DT)
    assert frobenius(P) < 1e-9
    assert frobenius(state.PK2) < 1e-9


@pytest.mark.parametrize("name", FINITE)
def test_zero_stress_under_rotation(name):
    """A rigid rotation F = Q has C = I: no stress."""
    material = FINITE[name]
    Q = Tensor2(tensor=rotation.random(jax.random.PRNGKey(0)))
    P, state = material.constitutive_update(Q, material.init_state(), DT)
    assert frobenius(P) < 1e-8
    assert frobenius(state.sig) < 1e-8


@pytest.mark.parametrize("name", FINITE)
def test_pk2_symmetry_along_path(name):
    """F^-1 P is symmetric. It is checked before ``state.PK2``, which is symmetrized."""
    for state in run_path(FINITE[name], stretch_shear_path()):
        assert frobenius((state.F.inv @ state.PK1).skw) < 1e-9


@pytest.mark.parametrize("name", FINITE)
def test_state(name):
    """``state.internal`` has the internal variables of the small-strain behavior, and
    ``state.log_strain`` is the Hencky strain of ``state.F``."""
    material, small = FINITE[name], SMALL[name]
    structure = jax.tree.structure(material.init_state().internal)
    assert structure == jax.tree.structure(small.init_state().internal)
    for state in run_path(material, stretch_shear_path()):
        assert jax.tree.structure(state.internal) == structure
        assert frobenius(state.log_strain - jm.LogarithmicStrain.log_strain(state.F)) < 1e-12


@pytest.mark.parametrize("name", FINITE)
def test_objectivity(name):
    """Under F -> QF along a path, PK2, the log-space stress and the internal variables are
    unchanged, and the Cauchy stress rotates."""
    material = FINITE[name]
    Q = Tensor2(tensor=rotation.from_euler_bunge(0.5, 0.3, -0.4))
    Fs = stretch_shear_path()
    for a, b in zip(run_path(material, Fs), run_path(material, [Q @ F for F in Fs]), strict=True):
        assert frobenius(a.PK2 - b.PK2) < 1e-8
        assert frobenius(a.log_stress - b.log_stress) < 1e-8
        assert frobenius(a.sig.rotate(Q) - b.sig) < 1e-8
        for x, y in zip(jax.tree.leaves(a.internal), jax.tree.leaves(b.internal), strict=True):
            assert jnp.abs(x - y).max() < 1e-8


@pytest.mark.parametrize("name", FINITE)
def test_small_strain_limit(name):
    """For F = I + eps H, the Cauchy stress tends to the small-strain response at sym(eps H),
    with an error of order eps."""
    material, small = FINITE[name], SMALL[name]
    H = Tensor2(tensor=[[1.0, 0.4, 0.0], [0.4, -0.5, 0.2], [0.0, 0.2, 0.3]])

    def relative_error(eps):
        _, state = material.constitutive_update(
            Tensor2.identity() + eps * H, material.init_state(), DT
        )
        reference, _ = small.constitutive_update((eps * H).sym, small.init_state(), DT)
        return frobenius(state.sig - reference) / frobenius(reference)

    large, small_ = relative_error(1e-3), relative_error(1e-5)
    assert large < 3e-3
    assert small_ < 3e-5
    assert small_ < 0.05 * large  # first order


@pytest.mark.parametrize("stretches", [(1.3, 0.8, 1.1), (1.05, 1.05, 1.05), (0.7, 1.4, 1.0)])
def test_hencky_strain(stretches):
    """E = diag(log l_i) for F = diag(l_i), and E does not see a superposed rotation."""
    F = diagonal(*stretches)
    expected = diagonal(*jnp.log(jnp.array(stretches))).sym
    Q = Tensor2(tensor=rotation.from_axis_angle(jnp.array([1.0, -1.0, 2.0]), 0.7))
    for F_ in (F, Q @ F):
        assert frobenius(jm.LogarithmicStrain.log_strain(F_) - expected) < 1e-12


def test_hencky_strain_from_polar_decomposition():
    """With F = RU = VR, E = log U, and its push-forward R E R^T is log V."""
    R, U = polar(F_GENERAL)
    V, _ = polar(F_GENERAL, mode="VR")
    E = jm.LogarithmicStrain.log_strain(F_GENERAL)
    assert frobenius(E - logm(U)) < 1e-12
    assert frobenius(E.rotate(R) - logm(V)) < 1e-12


def test_batched_update_matches_pointwise():
    material = FINITE["plastic"]
    G = Tensor2(tensor=[[0.1, 0.02, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    Fs = [Tensor2.identity() + i * G for i in range(4)]
    F_batch = Tensor2(array=jnp.stack([F.array for F in Fs]))
    P, state = material.batched_constitutive_update(F_batch, material.init_state(4), DT)
    for i, F in enumerate(Fs):
        P_i, state_i = material.constitutive_update(F, material.init_state(), DT)
        assert frobenius(P_i - P[i]) < 1e-8
        assert jnp.abs(state.internal.p[i] - state_i.internal.p) < 1e-12


# --------------------------------------------------------------------------------------------
#  Elasticity
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["elastic", "anisotropic"])
def test_elastic_is_hyperelastic(name):
    """PK1 is the derivative of the energy psi(F) = 1/2 E:C:E, also for anisotropy."""
    material, stiffness = FINITE[name], SMALL[name].elasticity.C

    def psi(F):
        E = SymmetricTensor2(tensor=0.5 * logm((F.T @ F).sym))
        return 0.5 * E.double_contract(stiffness @ E)

    P_energy = jax.grad(psi)(F_GENERAL)
    P, _ = material.constitutive_update(F_GENERAL, material.init_state(), DT)
    assert frobenius(P - P_energy) / frobenius(P_energy) < 1e-10


@pytest.mark.parametrize("stretch", [0.6, 1.5, 3.0])
def test_uniaxial_stress_hencky_elasticity(stretch):
    """Uniaxial stress: F = diag(l, l^-nu, l^-nu) gives the uniaxial Kirchhoff stress
    tau = E log(l) e1 x e1, exactly, at any stretch."""
    material = FINITE["elastic"]
    F = diagonal(stretch, stretch**-0.3, stretch**-0.3)
    P, _ = material.constitutive_update(F, material.init_state(), DT)
    tau = P @ F.T
    assert frobenius(tau - diagonal(210e3 * jnp.log(stretch), 0.0, 0.0)) < 1e-9 * 210e3


def test_elastic_tangent_matches_finite_differences():
    material = FINITE["elastic"]
    F = Tensor2(tensor=[[1.10, 0.05, 0.0], [-0.03, 0.98, 0.02], [0.0, 0.01, 1.03]])
    A, A_fd = tangents(material, F, material.init_state())
    assert jnp.abs(A - A_fd).max() / jnp.abs(A).max() < 1e-6


# repeated principal stretches: F = I, and two equal stretches off the material axes
REPEATED = {
    "F = I": Tensor2.identity(),
    "two equal stretches": Tensor2(tensor=rotation.from_euler_bunge(0.4, 1.1, -0.6))
    @ diagonal(1.2, 0.95, 0.95),
}


@pytest.mark.parametrize("F", REPEATED.values(), ids=REPEATED.keys())
@pytest.mark.parametrize("name", ["elastic", "anisotropic"])
def test_tangent_at_repeated_stretches(name, F):
    """At repeated eigenvalues of C the tangent involves the second derivative of logm
    there (it was off by 38% at F = I)."""
    material = FINITE[name]
    A, A_fd = tangents(material, F, material.init_state())
    assert jnp.abs(A - A_fd).max() / jnp.abs(A).max() < 1e-6


@pytest.mark.parametrize("name", ["elastic", "anisotropic"])
def test_tangent_at_identity_is_the_small_strain_stiffness(name):
    """dP/dF[H] = C : sym(H) at F = I."""
    material, stiffness = FINITE[name], SMALL[name].elasticity.C
    H = Tensor2(tensor=[[0.3, -0.2, 0.5], [0.1, 0.4, -0.3], [0.2, 0.6, -0.1]])

    def PK1(F):
        return material.constitutive_update(F, material.init_state(), DT)[0]

    _, dP = jax.jvp(PK1, (Tensor2.identity(),), (H,))
    expected = stiffness @ H.sym
    assert frobenius(dP - expected) < 1e-9 * frobenius(expected)


def test_anisotropic_stress_at_repeated_stretches():
    """With anisotropy, the log-space stress is not coaxial with C, and the stress involves
    the derivative of logm at repeated eigenvalues (it was off by 12%). Reference: finite
    differences of the energy, which only uses values of logm."""
    material, stiffness = FINITE["anisotropic"], SMALL["anisotropic"].elasticity.C
    F = REPEATED["two equal stretches"]

    def psi(F_array):
        F_ = Tensor2(array=F_array)
        E = SymmetricTensor2(tensor=0.5 * logm((F_.T @ F_).sym))
        return 0.5 * E.double_contract(stiffness @ E)

    h = 1e-6
    P_fd = jnp.array([(psi(F.array + h * e) - psi(F.array - h * e)) / (2 * h) for e in jnp.eye(9)])
    P, _ = material.constitutive_update(F, material.init_state(), DT)
    assert jnp.abs(P.array - P_fd).max() / jnp.abs(P_fd).max() < 1e-8


def test_elastic_path_independence():
    material = FINITE["elastic"]
    F_end = Tensor2(tensor=[[1.2, 0.1, 0.0], [0.05, 0.9, 0.03], [0.0, 0.0, 1.05]])
    P_direct, _ = material.constitutive_update(F_end, material.init_state(), DT)
    state = material.init_state()
    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        F = Tensor2.identity() + t * (F_end - Tensor2.identity())
        P_path, state = material.constitutive_update(F, state, DT)
    assert frobenius(P_direct - P_path) < 1e-9


# --------------------------------------------------------------------------------------------
#  Plasticity
# --------------------------------------------------------------------------------------------


def test_plastic_consistency_and_hardening():
    """The log-space stress stays on the yield surface, p never decreases."""
    material = FINITE["plastic"]
    state = material.init_state()
    p_old, yielded = 0.0, False
    for k in range(1, 12):
        _, state = material.constitutive_update(diagonal(1 + 0.02 * k, 1, 1), state, DT)
        p = state.internal.p
        assert p >= p_old - 1e-12
        if p > 1e-10:
            yielded = True
            sig_y = SIG0 + (SIGU - SIG0) * (1 - jnp.exp(-B * p))
            assert jnp.abs(von_mises(state.log_stress) - sig_y) < 1e-6
        p_old = p
    assert yielded


def test_plastic_flow_is_isochoric():
    """tr(log-space plastic strain) = 0, i.e. det F^p = 1."""
    material = FINITE["plastic"]
    state = material.init_state()
    for k in range(1, 8):
        F = diagonal(1 + 0.03 * k, 1 - 0.005 * k, 1)
        _, state = material.constitutive_update(F, state, DT)
    assert state.internal.p > 0
    assert jnp.abs(tr(state.internal.epsp)) < 1e-10


def test_plastic_volume_response_stays_elastic():
    """With isochoric flow, the mean Kirchhoff stress is K log J at any plastic strain."""
    material = FINITE["plastic"]
    K = ELASTICITY.kappa
    state = material.init_state()
    for k in range(1, 15):
        F = diagonal(1 + 0.05 * k, 1 - 0.01 * k, 1)
        P, state = material.constitutive_update(F, state, DT)
        tau_mean = tr(P @ F.T) / 3
        log_J = jnp.log(F.det)
        assert jnp.abs(tau_mean - K * log_J) < 1e-12 * K * jnp.abs(log_J)
    assert state.internal.p > 0.3


def test_plastic_tangent_matches_finite_differences():
    """Consistent tangent of a plastic step, from a plastic state, along a general F."""
    material = FINITE["plastic"]
    F_old = Tensor2(tensor=[[1.06, 0.02, 0.0], [0.01, 0.99, 0.0], [0.0, 0.0, 1.0]])
    _, state = material.constitutive_update(F_old, material.init_state(), DT)
    assert state.internal.p > 0
    F = Tensor2(tensor=[[1.08, 0.04, 0.01], [0.015, 0.985, 0.005], [0.0, 0.01, 1.002]])
    A, A_fd = tangents(material, F, state)
    assert jnp.abs(A - A_fd).max() / jnp.abs(A).max() < 1e-7


# --------------------------------------------------------------------------------------------
#  Viscoelasticity
# --------------------------------------------------------------------------------------------


def test_viscoelastic_relaxation():
    """At fixed F, the stress relaxes monotonically."""
    material = FINITE["viscoelastic"]
    F = diagonal(1.05, 1, 1)
    _, state = material.constitutive_update(F, material.init_state(), jnp.array(0.01))
    q = [von_mises(state.sig)]
    for _ in range(10):
        _, state = material.constitutive_update(F, state, jnp.array(20.0))
        q.append(von_mises(state.sig))
    assert jnp.all(jnp.diff(jnp.array(q)) <= 1e-7)
    assert q[-1] < 0.9 * q[0]


def test_viscoelastic_rate_dependence():
    """The same stretch reached faster gives a higher stress."""
    material = FINITE["viscoelastic"]
    F = diagonal(1.03, 1, 1)
    _, fast = material.constitutive_update(F, material.init_state(), jnp.array(1e-3))
    _, slow = material.constitutive_update(F, material.init_state(), jnp.array(1e3))
    assert von_mises(fast.sig) > 1.001 * von_mises(slow.sig)
