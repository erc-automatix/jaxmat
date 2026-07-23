"""Tests for paramax-backed constrained parameter fields."""

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
import pytest

from jaxmat.parameters import Bounded, Interval, Positive, Scaled, Unconstrained

jax.config.update("jax_enable_x64", True)

RTOL = 1e-9


class PositiveHolder(eqx.Module):
    x: jax.Array = Positive()


class IntervalHolder(eqx.Module):
    x: jax.Array = Interval(0.0, 0.5)


class U(eqx.Module):
    x: jax.Array = Unconstrained()


def S(ref):
    class _S(eqx.Module):
        x: jax.Array = Scaled(ref)

    return _S


# --------------------------------------------------------------------------- #
# Unconstrained
# --------------------------------------------------------------------------- #
class TestUnconstrained:
    @pytest.mark.parametrize("v", [-5.0, -1e-3, 0.0, 2.5, 7.0e4])
    def test_roundtrip_including_negatives(self, v):
        assert float(paramax.unwrap(U(x=v)).x) == pytest.approx(v, abs=1e-9, rel=1e-9)

    def test_latent_equals_value(self):
        # identity map: the stored latent is the value itself
        m = U(x=3.5)
        assert float(m.x.args[0]) == pytest.approx(3.5)

    def test_strong_float_dtype(self):
        assert paramax.unwrap(U(x=1.0)).x.dtype == jnp.float64

    def test_gradient_is_physical(self):
        # identity reparam => d/dlatent (x^2) = 2 x
        g = eqx.filter_grad(lambda m: paramax.unwrap(m).x ** 2)(U(x=3.0))
        (leaf,) = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
        assert float(leaf) == pytest.approx(6.0)

    def test_idempotent_on_wrapped(self):
        m1 = U(x=2.0)
        m2 = U(x=m1.x)  # feeding an already-wrapped value through
        assert m2.x is m1.x

    def test_survives_flatten_unflatten(self):
        m = U(x=-4.0)
        leaves, td = jax.tree_util.tree_flatten(m)
        r = jax.tree_util.tree_unflatten(td, leaves)
        assert float(paramax.unwrap(r).x) == pytest.approx(-4.0)


# --------------------------------------------------------------------------- #
# Scaled
# --------------------------------------------------------------------------- #
class TestScaled:
    @pytest.mark.parametrize("v", [-4e5, -1.0, 0.0, 2.5e5, 1e8])
    def test_roundtrip_including_negatives(self, v):
        cls = S(1e5)
        assert float(paramax.unwrap(cls(x=v)).x) == pytest.approx(v, rel=1e-9, abs=1e-9)

    def test_latent_is_order_one(self):
        # storing x/ref brings the optimised latent to O(1)
        cls = S(1e5)
        m = cls(x=2.5e5)
        assert float(m.x.args[0]) == pytest.approx(2.5)

    def test_default_ref_is_identity(self):
        cls = S(1.0)
        m = cls(x=7.0)
        assert float(m.x.args[0]) == pytest.approx(7.0)
        assert float(paramax.unwrap(m).x) == pytest.approx(7.0)

    def test_gradient_scales_with_ref(self):
        # d/dlatent (x^2) with x = ref*u  =>  2 x * ref
        ref = 1e5
        cls = S(ref)
        g = eqx.filter_grad(lambda m: paramax.unwrap(m).x ** 2)(cls(x=3e5))
        (leaf,) = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
        assert float(leaf) == pytest.approx(2 * 3e5 * ref)

    def test_idempotent_on_wrapped(self):
        cls = S(10.0)
        m1 = cls(x=50.0)
        m2 = cls(x=m1.x)
        assert m2.x is m1.x

    def test_survives_flatten_unflatten(self):
        cls = S(1e3)
        m = cls(x=-2e3)
        leaves, td = jax.tree_util.tree_flatten(m)
        r = jax.tree_util.tree_unflatten(td, leaves)
        assert float(paramax.unwrap(r).x) == pytest.approx(-2e3)


class TestPositive:
    @pytest.mark.parametrize("v", [1e-6, 1, 210e3, 1e8])
    def test_roundtrip(self, v):
        assert float(paramax.unwrap(PositiveHolder(x=v)).x) == pytest.approx(v, rel=1e-5)

    def test_positive_and_strong_typed(self):
        out = paramax.unwrap(PositiveHolder(x=1e-4)).x
        assert float(out) > 0.0

    def test_large_init_no_overflow(self):
        assert jnp.isfinite(paramax.unwrap(PositiveHolder(x=1e8)).x)

    def test_negative_init_nan(self):
        assert jnp.isnan(paramax.unwrap(PositiveHolder(x=-1.0)).x)

    def test_gradient_flows_through_latent(self):
        # Positive() stores u = log(x); grad is d/du (exp(u))^2 = 2 x^2
        g = eqx.filter_grad(lambda m: paramax.unwrap(m).x ** 2)(PositiveHolder(x=3.0))
        (leaf,) = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
        assert float(leaf) == pytest.approx(2 * 3.0**2)  # 18, w.r.t. the latent

    def test_physical_gradient(self):
        g = jax.grad(lambda x: x**2)(paramax.unwrap(PositiveHolder(x=3.0)).x)
        assert float(g) == pytest.approx(6.0)


class TestInterval:
    @pytest.mark.parametrize("v", [0.01, 0.25, 0.49])
    def test_roundtrip(self, v):
        assert float(paramax.unwrap(IntervalHolder(x=v)).x) == pytest.approx(v, rel=1e-8)

    def test_stays_in_bounds(self):
        for v in [1e-4, 0.25, 0.4999]:
            assert 0.0 < float(paramax.unwrap(IntervalHolder(x=v)).x) < 0.5

    def test_bounded_alias(self):
        class A(eqx.Module):
            x: jax.Array = Interval(1.0, 3.0)

        class B(eqx.Module):
            x: jax.Array = Bounded(1.0, 3.0)

        assert float(paramax.unwrap(A(x=2.0)).x) == pytest.approx(float(paramax.unwrap(B(x=2.0)).x))

    def test_extreme_latent_clamped(self):
        m = IntervalHolder(x=0.25)
        w = m.x
        for big in (1e4, -1e4):
            forced = eqx.tree_at(lambda z: z.args, w, (jnp.asarray(big),))
            assert 0.0 <= float(paramax.unwrap(forced)) <= 0.5


class TestPyTreeBehaviour:
    def test_idempotent_on_wrapped(self):
        m1 = PositiveHolder(x=7.0)
        m2 = PositiveHolder(x=m1.x)
        assert m2.x is m1.x

    def test_survives_flatten_unflatten(self):
        m = PositiveHolder(x=5.0)
        leaves, td = jax.tree_util.tree_flatten(m)
        assert float(paramax.unwrap(jax.tree_util.tree_unflatten(td, leaves)).x) == pytest.approx(
            5.0
        )

    def test_optimisation_keeps_constraints(self):
        optax = pytest.importorskip("optax")

        class Mat(eqx.Module):
            E: jax.Array = Positive()
            nu: jax.Array = Interval(0.0, 0.5)

            def stiffness(self):
                return self.E / (1 - 2 * self.nu)

        m = Mat(E=210e3, nu=0.30)

        def loss(m, t):
            return (paramax.unwrap(m).stiffness() - t) ** 2

        opt = optax.adam(1e-2)
        st = opt.init(eqx.filter(m, eqx.is_inexact_array))

        @eqx.filter_jit
        def step(m, st):
            _, g = eqx.filter_value_and_grad(loss)(m, 6e5)
            u, st = opt.update(g, st, m)
            return eqx.apply_updates(m, u), st

        for _ in range(2000):
            m, st = step(m, st)
            mu = paramax.unwrap(m)
            assert float(mu.E) > 0.0 and 0.0 < float(mu.nu) < 0.5
        assert float(paramax.unwrap(m).stiffness()) == pytest.approx(6e5, rel=1e-3)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
