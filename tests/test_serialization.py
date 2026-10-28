"""Saving and loading models and states with save_model and load_model."""

import equinox as eqx
import jax
import jax.numpy as jnp
import paramax
import pytest

import jaxmat.materials as jm
from jaxmat.parameters import Affine, Interval, Positive, Scaled, Unconstrained, parameter
from jaxmat.tensors import SymmetricTensor2
from jaxmat.utils import load_model, save_model


class Module1(eqx.Module):
    A: float = parameter()
    B: float = parameter()


class Module2(eqx.Module):
    len_A: int = eqx.field(static=True)
    A: jax.Array

    def __init__(self, len_A: int, A: jax.Array | None = None):
        self.len_A = len_A
        if A is None:
            self.A = jnp.zeros(len_A)
        else:
            assert len(A) == len_A
            self.A = jnp.asarray(A, dtype=jnp.float64)


def von_mises(sig0=250.0, E=210e3):
    return jm.vonMisesIsotropicHardening(
        elasticity=jm.LinearElasticIsotropic(E=E, nu=0.3),
        yield_stress=jm.VoceHardening(sig0=sig0, sigu=350.0, b=20.0),
    )


def maxwell(Nbranch, scale=1.0):
    return jm.GeneralizedMaxwell(
        elasticity=jm.LinearElasticIsotropic(E=scale * 1e3, nu=0.3),
        Nbranch=Nbranch,
        viscous_branches=jm.LinearElasticIsotropic(
            E=scale * jnp.arange(1.0, Nbranch + 1), nu=jnp.full(Nbranch, 0.3)
        ),
        relaxation_times=scale * jnp.logspace(-1, 1, Nbranch),
    )


def test_save_load_without_hyperparams(tmp_path):
    model = Module1(A=11, B=12)
    save_model(tmp_path / "model.eqx", model)
    loaded_model = load_model(tmp_path / "model.eqx", Module1(A=0, B=0))
    assert eqx.tree_equal(model, loaded_model)


def test_save_load_with_hyperparams(tmp_path):
    model = Module2(len_A=3, A=[1, 2, 5])
    save_model(tmp_path / "model.eqx", model, {"len_A": model.len_A})
    loaded_model = load_model(tmp_path / "model.eqx", lambda len_A: Module2(len_A))
    assert eqx.tree_equal(model, loaded_model)


@pytest.mark.parametrize(
    ("model", "template"),
    [
        (von_mises(sig0=312.5, E=195e3), von_mises()),
        (
            jm.TemperatureDependent(von_mises(sig0=Affine(ref=300.0, slope=-0.2))),
            jm.TemperatureDependent(von_mises(sig0=Affine(ref=0.0, slope=0.0))),
        ),
        (maxwell(3, scale=2.0), maxwell(3)),
    ],
    ids=["von Mises", "temperature dependent", "generalized Maxwell"],
)
def test_material_round_trip(tmp_path, model, template):
    save_model(tmp_path / "material.eqx", model)
    assert eqx.tree_equal(load_model(tmp_path / "material.eqx", template), model)


class Constrained(eqx.Module):
    E: jax.Array = Positive()
    nu: jax.Array = Interval(-1.0, 0.5)
    sig0: jax.Array = Scaled(ref=100.0)
    a: jax.Array = Unconstrained()


VALUES = dict(E=210e3, nu=0.3, sig0=251.25, a=-1.5)
TEMPLATE = dict(E=1.0, nu=0.0, sig0=1.0, a=0.0)


def test_constrained_parameters_round_trip(tmp_path):
    """Constrained parameters are saved in their latent form and recovered exactly. The
    transforms come from the template (Interval and Scaled build a new function for each
    instance, so only the arrays compare equal)."""
    model = Constrained(**VALUES)
    save_model(tmp_path / "law.eqx", model)
    loaded = load_model(tmp_path / "law.eqx", Constrained(**TEMPLATE))
    assert eqx.tree_equal(eqx.filter(loaded, eqx.is_array), eqx.filter(model, eqx.is_array))
    physical = paramax.unwrap(loaded)
    for name, value in VALUES.items():
        assert float(getattr(physical, name)) == pytest.approx(value, rel=1e-14)


def test_unwrapped_model_into_constrained_template_raises(tmp_path):
    """A model saved with unwrapped parameters has the same class and number of leaves as the
    constrained template: without a structure check, physical values would be read as latent
    ones."""
    save_model(tmp_path / "law.eqx", paramax.unwrap(Constrained(**VALUES)))
    with pytest.raises(ValueError, match=r"leaf 0 is \.E in the file and \.E\.args\[0\]"):
        load_model(tmp_path / "law.eqx", Constrained(**TEMPLATE))


def test_other_structure_raises(tmp_path):
    """A constant replaced by a temperature law changes the leaves: clear error."""
    save_model(tmp_path / "material.eqx", jm.TemperatureDependent(von_mises()))
    with pytest.raises(ValueError, match="does not match the structure"):
        load_model(
            tmp_path / "material.eqx",
            jm.TemperatureDependent(von_mises(sig0=Affine(ref=0.0, slope=0.0))),
        )


def test_batched_state_round_trip(tmp_path):
    """A state, e.g. for restarting a computation, is saved like a model."""
    material = von_mises()
    eps = SymmetricTensor2(array=jnp.zeros((4, 6)).at[:, 0].set(jnp.linspace(0.0, 1e-2, 4)))
    _, state = material.batched_constitutive_update(eps, material.init_state(4), 0.0)
    save_model(tmp_path / "state.eqx", state)
    loaded = load_model(tmp_path / "state.eqx", material.init_state(4))
    assert eqx.tree_equal(loaded, state)
    assert float(loaded.internal.p[-1]) > 0


def test_hyperparameters_are_optional_on_both_sides(tmp_path):
    """A file with hyperparameters loads with a template module, and a file without
    hyperparameters loads with a factory, which is called without arguments."""
    save_model(tmp_path / "with.eqx", maxwell(3, scale=2.0), {"Nbranch": 3})
    assert eqx.tree_equal(load_model(tmp_path / "with.eqx", maxwell(3)), maxwell(3, scale=2.0))
    save_model(tmp_path / "without.eqx", von_mises(sig0=312.5))
    assert eqx.tree_equal(load_model(tmp_path / "without.eqx", von_mises), von_mises(sig0=312.5))


def test_hyperparameters_build_the_template(tmp_path):
    save_model(tmp_path / "maxwell.eqx", maxwell(5, scale=2.0), {"Nbranch": 5})
    loaded = load_model(tmp_path / "maxwell.eqx", lambda Nbranch: maxwell(Nbranch))
    assert loaded.Nbranch == 5
    assert eqx.tree_equal(loaded, maxwell(5, scale=2.0))


def test_wrong_model_class_raises(tmp_path):
    save_model(tmp_path / "material.eqx", von_mises())
    with pytest.raises(ValueError, match="vonMisesIsotropicHardening.*ElasticBehavior"):
        load_model(
            tmp_path / "material.eqx",
            jm.ElasticBehavior(elasticity=jm.LinearElasticIsotropic(E=1.0, nu=0.3)),
        )


def test_wrong_shapes_raise(tmp_path):
    save_model(tmp_path / "maxwell.eqx", maxwell(3))
    with pytest.raises(RuntimeError, match="changed shape"):
        load_model(tmp_path / "maxwell.eqx", maxwell(5))


def test_not_a_model_file_raises(tmp_path):
    with open(tmp_path / "leaves.eqx", "wb") as file:
        eqx.tree_serialise_leaves(file, von_mises())  # leaves only, no header
    with pytest.raises(ValueError, match="not a model file"):
        load_model(tmp_path / "leaves.eqx", von_mises())
