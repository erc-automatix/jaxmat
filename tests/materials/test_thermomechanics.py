"""Tests for temperature-dependent parameters and behaviors."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxmat.materials.elasticity import ElasticBehavior, LinearElasticIsotropic
from jaxmat.materials.thermomechanics import TemperatureDependent
from jaxmat.parameters import (
    Affine,
    Arrhenius,
    Constant,
    Expression,
    Polynomial,
    evaluate_parameters,
    has_parameter_dependence,
    parameter,
)
from jaxmat.tensors import SymmetricTensor2

jax.config.update("jax_enable_x64", True)


def _eps(exx=1e-3):
    return SymmetricTensor2(array=jnp.array([exx, 0.0, 0.0, 0.0, 0.0, 0.0]))


class TestExpressions:
    def test_constant(self):
        assert float(Constant(value=3.0).evaluate(999.0)) == pytest.approx(3.0)

    def test_affine(self):
        p = Affine(ref=210e3, slope=-50.0, T_ref=293.0)
        print(p.evaluate(500.0))
        assert float(p.evaluate(500.0)) == pytest.approx(210e3 - 50 * (500 - 293))

    def test_polynomial(self):
        T_ref = 273.15
        T_ = 300.0
        p = Polynomial(coeffs=jnp.array([1.0, 2.0, 3.0]))
        assert float(p.evaluate(T_)) == pytest.approx(1 + 2 * (T_ - T_ref) + 3 * (T_ - T_ref) ** 2)

    def test_arrhenius(self):
        p = Arrhenius(A=1.0, Q=1000.0)
        assert float(p.evaluate(300.0)) == pytest.approx(
            float(jnp.exp(-1000.0 / (8.314462618 * 300.0)))
        )

    def test_expression(self):
        class Law(eqx.Module):
            a: jax.Array = parameter()

            def __call__(self, T):
                return self.a * jnp.sqrt(T)

        assert float(Expression(fn=Law(a=2.0)).evaluate(9.0)) == pytest.approx(6.0)


class TestEvaluateParameters:
    def test_resolves_and_leaves_constants(self):
        el = LinearElasticIsotropic(E=Affine(ref=210e3, slope=-50.0, T_ref=293.0), nu=0.3)
        res = evaluate_parameters(el, 500.0)
        assert float(res.E) == pytest.approx(199650.0)
        assert float(res.nu) == pytest.approx(0.3)

    def test_resolved_uses_existing_properties(self):
        el = LinearElasticIsotropic(E=Affine(ref=210e3, slope=-50.0, T_ref=293.0), nu=0.3)
        res = evaluate_parameters(el, 500.0)
        assert float(res.kappa) == pytest.approx(199650.0 / (3 * (1 - 2 * 0.3)))

    def test_constant_model_noop(self):
        el = LinearElasticIsotropic(E=210e3, nu=0.3)
        assert evaluate_parameters(el, None) is el

    def test_has_parameter_dependence(self):
        assert has_parameter_dependence(
            LinearElasticIsotropic(E=Affine(ref=1.0, slope=1.0), nu=0.3)
        )
        assert not has_parameter_dependence(LinearElasticIsotropic(E=1.0, nu=0.3))

    def test_none_errors_with_expressions(self):
        with pytest.raises(ValueError):
            evaluate_parameters(LinearElasticIsotropic(E=Affine(ref=1.0, slope=1.0), nu=0.3), None)


class TestUnresolvedFailsClearly:
    def test_property_access_raises(self):
        el = LinearElasticIsotropic(E=Affine(ref=210e3, slope=-50.0), nu=0.3)
        with pytest.raises(TypeError, match="not been resolved"):
            _ = el.kappa

    def test_arithmetic_raises(self):
        p = Affine(ref=1.0, slope=1.0)
        for op in (lambda: p + 1, lambda: 1 + p, lambda: p * 2, lambda: -p):
            with pytest.raises(TypeError, match="not been resolved"):
                op()


class TestTemperatureDependentBehavior:
    def _beh(self):
        el = LinearElasticIsotropic(E=Affine(ref=210e3, slope=-50.0, T_ref=293.0), nu=0.3)
        return TemperatureDependent(ElasticBehavior(elasticity=el))

    def test_stress_decreases_with_T(self):
        beh = self._beh()
        st = beh.init_state()
        s = [
            float(beh.constitutive_update((_eps(), T), st, 1.0)[0].array[0])
            for T in [293.0, 500.0, 800.0]
        ]
        assert s[0] > s[1] > s[2]

    def test_constant_wrapped_ignores_T(self):
        beh = TemperatureDependent(
            ElasticBehavior(elasticity=LinearElasticIsotropic(E=210e3, nu=0.3))
        )
        st = beh.init_state()
        s5 = beh.constitutive_update((_eps(), 500.0), st, 1.0)[0].array[0]
        s8 = beh.constitutive_update((_eps(), 800.0), st, 1.0)[0].array[0]
        assert float(s5) == pytest.approx(float(s8))

    def test_default_temperature_bare_input(self):
        el = LinearElasticIsotropic(E=Affine(ref=210e3, slope=-50.0, T_ref=293.0), nu=0.3)
        beh = TemperatureDependent(ElasticBehavior(elasticity=el), default_temperature=500.0)
        sig, _ = beh.constitutive_update(_eps(), beh.init_state(), 1.0)
        assert jnp.isfinite(sig.array[0])

    def test_missing_T_without_default_raises(self):
        with pytest.raises(ValueError, match="expected inputs"):
            self._beh().constitutive_update(_eps(), self._beh().init_state(), 1.0)

    def test_batched_over_temperatures(self):
        beh = self._beh()
        Ts = jnp.linspace(293.0, 800.0, 10)
        st = beh.init_state()
        f = jax.jit(jax.vmap(lambda T: beh.constitutive_update((_eps(), T), st, 1.0)[0].array[0]))
        s = f(Ts)
        assert s.shape == (10,)
        assert jnp.all(jnp.diff(s) < 0)  # monotonically decreasing

    def test_differentiable_wrt_param_and_T(self):
        st = self._beh().init_state()

        def stress(refE, T):
            beh = TemperatureDependent(
                ElasticBehavior(
                    elasticity=LinearElasticIsotropic(
                        E=Affine(ref=refE, slope=-50.0, T_ref=293.0), nu=0.3
                    )
                )
            )
            return beh.constitutive_update((_eps(), T), st, 1.0)[0].array[0]

        assert jnp.isfinite(jax.grad(stress, 0)(210e3, 500.0))
        assert float(jax.grad(stress, 1)(210e3, 500.0)) < 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
