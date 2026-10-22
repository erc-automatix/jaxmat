"""Generic multiphysics states and behaviors."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.parameters import (
    Affine,
    Expression,
    evaluate_parameters,
    parameter,
    parameter_dependencies,
)
from jaxmat.state import (
    AbstractState,
    FiniteStrainState,
    SmallStrainState,
    Vector,
    make_batched,
    make_physics,
)
from jaxmat.tensors import SymmetricTensor2, Tensor2

jax.config.update("jax_enable_x64", True)

Id = SymmetricTensor2.identity()
elasticity = jm.LinearElasticIsotropic(E=20e3, nu=0.25)
B, M = 0.8, 5e3


def _eps(exx=1e-3, eyy=0.0):
    return SymmetricTensor2(array=jnp.array([exx, eyy, 0.0, 0.0, 0.0, 0.0]))


POROELASTICITY = make_physics(
    "Poroelasticity",
    gradients={"strain": SymmetricTensor2, "p": 0.0},
    fluxes={"stress": SymmetricTensor2, "zeta": 0.0},
    aliases={"eps": "strain", "sig": "stress"},
)


class SkeletonBiot(jm.MultiphysicsBehavior):
    """Biot coupling around a small-strain skeleton, whose full state is the internal state:
    exercises nested internal states, plasticity, scan carries and gradients."""

    physics = POROELASTICITY
    skeleton: jm.SmallStrainBehavior
    b: jax.Array = parameter()
    M: jax.Array = parameter()

    def make_internal_state(self):
        return self.skeleton.init_state()

    def response(self, x, state, dt):
        sig_eff, skeleton_state = self.skeleton.constitutive_update(x.strain, state.internal, dt)
        sig = sig_eff - self.b * x.p * Id
        zeta = self.b * x.strain.tr + x.p / self.M
        return (sig, zeta), skeleton_state


def _biot(skeleton=None):
    skeleton = skeleton or jm.ElasticBehavior(elasticity=elasticity)
    return SkeletonBiot(skeleton=skeleton, b=B, M=M)


def _von_mises():
    hardening = jm.VoceHardening(sig0=50.0, sigu=80.0, b=200.0)
    return jm.vonMisesIsotropicHardening(elasticity=elasticity, yield_stress=hardening)


# The mechanical states, declared with the factory. The hand-written classes are kept: these
# declarations only check that the factory reproduces them.
SMALL_STRAIN = make_physics(
    "SmallStrain",
    gradients={"strain": SymmetricTensor2},
    fluxes={"stress": SymmetricTensor2},
    aliases={"eps": "strain", "sig": "stress"},
)
FINITE_STRAIN = make_physics(
    "FiniteStrain", gradients={"F": Tensor2.identity}, fluxes={"PK1": Tensor2}
)


class LinearBiot(jm.MultiphysicsBehavior):
    """Biot poroelasticity written from scratch: no state class, no state.update."""

    physics = POROELASTICITY
    elasticity: jm.AbstractLinearElastic
    b: jax.Array = parameter()
    M: jax.Array = parameter()

    def response(self, x, state, dt):
        sig = self.elasticity.C @ x.strain - self.b * x.p * Id
        zeta = self.b * x.strain.tr + x.p / self.M
        return (sig, zeta), state.internal


THM = make_physics(
    "ThermoPoromechanics",
    gradients={"strain": SymmetricTensor2, "p": 0.0},
    fluxes={"stress": SymmetricTensor2, "zeta": 0.0},
    external={"T": 293.15},
)


class LinearTHM(jm.MultiphysicsBehavior):
    """Thermo-poroelasticity, declared as one physics rather than composed."""

    physics = THM
    elasticity: jm.LinearElasticIsotropic
    b: jax.Array = parameter()
    M: jax.Array = parameter()
    alpha: jax.Array = parameter()
    """Drained thermal expansion coefficient."""
    alpha_m: jax.Array = parameter()
    """Thermal coefficient of the fluid content."""
    T0: jax.Array = parameter(default=293.15)

    def response(self, x, state, dt):
        dT = x.T - self.T0
        sig = self.elasticity.C @ (x.strain - self.alpha * dT * Id) - self.b * x.p * Id
        zeta = self.b * x.strain.tr + x.p / self.M - 3 * self.alpha_m * dT
        return (sig, zeta), None


CONDUCTION = make_physics(
    "PorousConduction",
    gradients={"grad_T": Vector(3)},
    fluxes={"q": Vector(3)},
    external={"T": 293.15, "phi": 0.0},
)


class Conductivity(eqx.Module):
    k0: jax.Array = parameter()

    def __call__(self, T, phi):
        return self.k0 * (1 - phi) * (300.0 / T)


class Fourier(jm.MultiphysicsBehavior):
    """Heat conduction: vector variables, two external variables."""

    physics = CONDUCTION
    k: jax.Array = parameter()

    def response(self, x, state, dt):
        return (-self.k * x.grad_T,), None


class TestPhysics:
    def test_equal_declarations_give_the_same_classes(self):
        P = make_physics(
            "Poroelasticity",
            gradients={"strain": SymmetricTensor2, "p": 0.0},
            fluxes={"stress": SymmetricTensor2, "zeta": 0.0},
            aliases={"eps": "strain", "sig": "stress"},
        )
        assert P == POROELASTICITY
        assert P.State is POROELASTICITY.State
        assert P.Inputs is POROELASTICITY.Inputs

    def test_generated_state(self):
        state = POROELASTICITY.State()
        assert isinstance(state, AbstractState)
        assert type(state).__name__ == "PoroelasticityState"
        assert state.internal is None and float(state.p) == 0.0
        new = state.update(sig=Id, p=2.0).add(zeta=1.0)
        assert jnp.allclose(new.stress.array, Id.array) and new.sig is new.stress
        assert float(new.p) == 2.0 and float(new.zeta) == 1.0
        assert make_batched(state, 5).strain.array.shape == (5, 6)

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(gradients={"strain": SymmetricTensor2}, fluxes={"strain": SymmetricTensor2}),
            dict(gradients={"internal": 0.0}, fluxes={"q": 0.0}),
            dict(gradients={"x": 0.0}, fluxes={}),
            dict(gradients={"x": 0.0}, fluxes={"y": 0.0}, aliases={"z": "w"}),
            dict(gradients={"x": True}, fluxes={"y": 0.0}),
        ],
    )
    def test_invalid_declarations(self, kwargs):
        with pytest.raises((ValueError, TypeError)):
            make_physics("Bad", **kwargs)

    def test_inputs_are_always_tuples_or_dicts(self):
        x = SMALL_STRAIN.make_inputs((_eps(),))
        assert x.strain is not None and x == SMALL_STRAIN.make_inputs({"strain": x.strain})
        with pytest.raises(TypeError):
            SMALL_STRAIN.make_inputs(_eps())  # no bare single value
        with pytest.raises(ValueError):
            POROELASTICITY.make_inputs((_eps(),))
        with pytest.raises(ValueError):
            POROELASTICITY.make_inputs({"strain": _eps(), "q": 1.0})


class TestMechanicalStatesMatch:
    """The factory reproduces the hand-written mechanical states."""

    @pytest.mark.parametrize(
        ("physics", "original"),
        [(SMALL_STRAIN, SmallStrainState), (FINITE_STRAIN, FiniteStrainState)],
    )
    def test_same_fields_values_and_aliases(self, physics, original):
        gen, ref = physics.State(), original()
        assert [f.name for f in dataclasses.fields(gen)] == [
            f.name for f in dataclasses.fields(ref)
        ]
        for a, b in zip(jax.tree.leaves(gen), jax.tree.leaves(ref), strict=True):
            assert a.shape == b.shape and a.dtype == b.dtype and jnp.array_equal(a, b)
        aliases = getattr(original, "__alias_targets__", {})
        assert gen.__alias_targets__ == aliases
        for alias in aliases:
            assert jnp.array_equal(getattr(gen, alias).array, getattr(ref, alias).array)
        batched = (jax.tree.leaves(make_batched(s, 3)) for s in (gen, ref))
        for a, b in zip(*batched, strict=True):
            assert a.shape == b.shape

    @pytest.mark.parametrize("hypothesis", ["small", "finite"])
    def test_behaviors_give_the_same_update(self, hypothesis):
        if hypothesis == "small":
            beh, physics, inputs = _von_mises(), SMALL_STRAIN, _eps(5e-3, -1e-3)
        else:
            beh = jm.Hyperelasticity(jm.CompressibleNeoHookean(mu=1e3, kappa=3e3))
            physics = FINITE_STRAIN
            inputs = Tensor2(tensor=jnp.eye(3) + 0.1 * jnp.arange(9.0).reshape(3, 3) / 9)
        ref_state = beh.init_state()
        gen_state = physics.State(internal=ref_state.internal)
        out_ref, new_ref = beh.constitutive_update(inputs, ref_state, 0.0)
        out_gen, new_gen = beh.constitutive_update(inputs, gen_state, 0.0)
        assert jnp.array_equal(out_ref.array, out_gen.array)
        for a, b in zip(jax.tree.leaves(new_ref), jax.tree.leaves(new_gen), strict=True):
            assert jnp.array_equal(a, b)


class TestBiot:
    def test_fluxes_and_recorded_state(self):
        beh = _biot()
        state = beh.init_state()
        assert type(state.internal).__name__ == "SmallStrainState"
        eps, p = _eps(), 2.0
        (sig, zeta), new = beh.constitutive_update((eps, p), state, 1.0)
        sig_eff = elasticity.C @ eps
        assert jnp.allclose(sig.array, (sig_eff - B * p * Id).array)
        assert float(zeta) == pytest.approx(B * 1e-3 + p / M)
        assert float(new.p) == p and float(new.zeta) == pytest.approx(float(zeta))
        assert jnp.allclose(new.internal.stress.array, sig_eff.array)  # effective stress

    def test_tangent_blocks(self):
        T = _biot().tangent((_eps(), 2.0), _biot().init_state(), 1.0)
        assert jnp.allclose(T[("stress", "strain")], elasticity.C.array)
        assert jnp.allclose(T[("stress", "p")], -B * Id.array)
        assert jnp.allclose(T[("zeta", "strain")], B * Id.array)  # = -d sigma / d p
        assert float(T[("zeta", "p")]) == pytest.approx(1 / M)

    def test_from_scratch_law_matches_wrapper(self):
        a, b = LinearBiot(elasticity=elasticity, b=B, M=M), _biot()
        x = {"strain": _eps(1e-3, -4e-4), "p": 3.0}  # inputs may also be a dict
        (sa, za), _ = a.constitutive_update(x, a.init_state(), 1.0)
        (sb, zb), _ = b.constitutive_update(x, b.init_state(), 1.0)
        assert jnp.allclose(sa.array, sb.array) and jnp.allclose(za, zb)
        Ta, Tb = a.tangent(x, a.init_state(), 1.0), b.tangent(x, b.init_state(), 1.0)
        assert all(jnp.allclose(Ta[k], Tb[k]) for k in Ta)

    def test_plastic_skeleton_scan_and_grad(self):
        """von Mises skeleton: drive a load path with lax.scan, the state is the carry."""
        beh = _biot(_von_mises())
        exx = jnp.linspace(0.0, 1e-2, 11)
        ps = jnp.linspace(0.0, 5.0, 11)

        def run(scale):
            def step(state, xs):
                e, p = xs
                inputs = (_eps(e, -0.5 * e), scale * p)
                (sig, zeta), state = beh.constitutive_update(inputs, state, 1.0)
                return state, (sig.array[0], zeta)

            return jax.lax.scan(step, beh.init_state(), (exx, ps))

        final, (sxx, zeta) = run(1.0)
        assert float(final.internal.internal.p) > 0  # skeleton plasticity, nested state
        # effective stress principle: the skeleton sees the strain only, not the pressure
        assert jnp.allclose(final.stress.array, (final.internal.stress - B * 5.0 * Id).array)
        assert jnp.allclose(zeta, B * 0.5 * exx + ps / M)  # tr(eps) = exx - 0.5 exx
        dsig_dscale = jax.grad(lambda s: run(s)[1][0][-1])(1.0)
        assert float(dsig_dscale) == pytest.approx(-B * 5.0)

    def test_batched(self):
        beh = _biot()
        N = 4
        eps = SymmetricTensor2(array=jnp.zeros((N, 6)).at[:, 0].set(jnp.linspace(0, 1e-3, N)))
        p = jnp.linspace(0.0, 1.0, N)
        (sig, zeta), state = beh.batched_constitutive_update((eps, p), beh.init_state(N), 1.0)
        assert sig.array.shape == (N, 6) and zeta.shape == (N,)
        assert state.internal.stress.array.shape == (N, 6)


class TestDeclaredCoupledPhysics:
    def test_thermo_poroelasticity(self):
        E = Affine(ref=20e3, slope=-10.0, T_ref=293.15)
        el = jm.LinearElasticIsotropic(E=E, nu=0.25)
        beh = LinearTHM(elasticity=el, b=B, M=M, alpha=1e-5, alpha_m=3e-5)
        state = beh.init_state()
        T, dT = 393.15, 100.0
        (sig, zeta), new = beh.constitutive_update((1e-5 * dT * Id, 0.0, T), state, 1.0)
        assert jnp.allclose(sig.array, 0.0, atol=1e-12)  # free thermal expansion
        assert float(zeta) == pytest.approx(3 * dT * (B * 1e-5 - 3e-5))
        assert float(new.T) == T
        (sig, _), _ = beh.constitutive_update((SymmetricTensor2(), 0.0, T), state, 1.0)
        E_T = 20e3 - 10.0 * dT
        assert jnp.allclose(sig.array, -E_T / (1 - 2 * 0.25) * 1e-5 * dT * Id.array)
        tangent = beh.tangent((SymmetricTensor2(), 0.0, T), state, 1.0)
        assert set(tangent) == {(f, g) for f in ("stress", "zeta") for g in ("strain", "p")}
        C_T = jm.LinearElasticIsotropic(E=E_T, nu=0.25).C.array
        assert jnp.allclose(tangent[("stress", "strain")], C_T)


class TestVariableDependence:
    def test_multi_variable_expression_in_a_behavior(self):
        beh = Fourier(k=Expression(fn=Conductivity(k0=2.0), depends_on=("T", "phi")))
        assert parameter_dependencies(beh) == {"T", "phi"}
        g = jnp.array([1.0, -2.0, 0.5])
        (q,), state = beh.constitutive_update((g, 400.0, 0.25), beh.init_state(), 1.0)
        k = 2.0 * 0.75 * 300.0 / 400.0
        assert jnp.allclose(q, -k * g) and float(state.phi) == 0.25
        tangent = beh.tangent((g, 400.0, 0.25), state, 1.0)
        assert jnp.allclose(tangent[("q", "grad_T")], -k * jnp.eye(3))

    def test_undeclared_variable_raises(self):
        el = jm.LinearElasticIsotropic(E=Affine(ref=20e3, slope=-10.0), nu=0.25)
        beh = _biot(jm.ElasticBehavior(elasticity=el))  # poromechanics declares no T
        with pytest.raises(ValueError, match="T"):
            beh.constitutive_update((_eps(), 0.0), beh.init_state(), 1.0)

    def test_temperature_dependent_wrapper_is_unchanged(self):
        el = jm.LinearElasticIsotropic(E=Affine(ref=20e3, slope=-10.0, T_ref=293.15), nu=0.25)
        beh = jm.TemperatureDependent(jm.ElasticBehavior(elasticity=el))
        sig, _ = beh.constitutive_update((_eps(), 393.15), beh.init_state(), 1.0)
        expected = evaluate_parameters(el, 393.15).C @ _eps()
        assert jnp.allclose(sig.array, expected.array)
