"""Global configuration and the switchable trace assertion."""

import os
import subprocess
import sys

import equinox as eqx
import jax.numpy as jnp
import pytest

import jaxmat
import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2
from jaxmat.utils import assert_max_traces


def _traced_twice():
    """A fresh jitted function, called with two input dtypes, i.e. traced twice."""

    @eqx.filter_jit
    @assert_max_traces(max_traces=1)
    def f(x):
        return 2 * x

    f(jnp.array(1.0))
    f(jnp.array(1))


class TestOptions:
    def test_default_is_off(self):
        assert jaxmat.config.check_max_traces is False

    def test_update_and_override(self):
        with jaxmat.config.override(check_max_traces=True):
            assert jaxmat.config.check_max_traces is True
        assert jaxmat.config.check_max_traces is False
        jaxmat.config.update("check_max_traces", True)
        try:
            assert jaxmat.config.check_max_traces is True
        finally:
            jaxmat.config.update("check_max_traces", False)

    def test_override_restores_after_an_error(self):
        with pytest.raises(KeyError), jaxmat.config.override(check_max_traces=True):
            raise KeyError
        assert jaxmat.config.check_max_traces is False

    def test_invalid_options(self):
        with pytest.raises(ValueError):
            jaxmat.config.update("check_max_tracing", True)
        with pytest.raises(TypeError):
            jaxmat.config.update("check_max_traces", 1)

    @pytest.mark.parametrize(("value", "expected"), [("1", "True"), ("off", "False")])
    def test_environment_variable(self, value, expected):
        env = {**os.environ, "JAXMAT_CHECK_MAX_TRACES": value}
        code = "import jaxmat; print(jaxmat.config.check_max_traces)"
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert out.stdout.strip() == expected

    def test_invalid_environment_variable(self):
        env = {**os.environ, "JAXMAT_CHECK_MAX_TRACES": "maybe"}
        out = subprocess.run([sys.executable, "-c", "import jaxmat"], env=env, capture_output=True)
        assert out.returncode != 0 and b"JAXMAT_CHECK_MAX_TRACES" in out.stderr


class TestAssertMaxTraces:
    def test_off_allows_retracing(self):
        _traced_twice()

    def test_on_raises_on_retracing(self):
        with jaxmat.config.override(check_max_traces=True), pytest.raises(RuntimeError):
            _traced_twice()

    def test_wraps_the_function(self):
        update = jm.vonMisesIsotropicHardening.constitutive_update
        assert update.__wrapped__.__wrapped__.__name__ == "constitutive_update"


class _LinearHardening(eqx.Module):
    sig0: float
    H: float

    def __call__(self, p):
        return self.sig0 + self.H * p


class _PowerHardening(eqx.Module):
    sig0: float
    n: float

    def __call__(self, p):
        return self.sig0 * (1 + p) ** self.n


def _update_with(hardening):
    el = jm.LinearElasticIsotropic(E=70e3, nu=0.3)
    mat = jm.vonMisesIsotropicHardening(elasticity=el, yield_stress=hardening)
    eps = SymmetricTensor2(array=jnp.array([1e-2, 0.0, 0.0, 0.0, 0.0, 0.0]))
    return mat.constitutive_update(eps, mat.init_state(), 0.0)[0]


def test_two_materials_of_the_same_class():
    """Different hardening laws retrace the same update: allowed unless checking is on."""
    sigs = [_update_with(h) for h in (_LinearHardening(350.0, 1e3), _PowerHardening(350.0, 0.1))]
    assert float(sigs[0].array[0]) != float(sigs[1].array[0])

    # new hardening types: compiled signatures do not exist yet, so both calls trace
    class Linear(_LinearHardening):
        pass

    class Power(_PowerHardening):
        pass

    with jaxmat.config.override(check_max_traces=True), pytest.raises(RuntimeError):
        _update_with(Linear(350.0, 1e3))
        _update_with(Power(350.0, 0.1))
