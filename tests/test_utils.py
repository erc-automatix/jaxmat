"""Pretty printing of models with print_eqx_fields."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.parameters import Positive
from jaxmat.utils import partition_by_node_names, print_eqx_fields


class Hardening(eqx.Module):
    """A calibrated law with constrained parameters."""

    sig0: jax.Array = Positive()
    H: jax.Array = Positive()

    def __call__(self, p):
        return self.sig0 + self.H * p


class Fields(eqx.Module):
    vector: jax.Array
    flags: jax.Array
    pair: tuple
    items: list
    label: str
    active: bool


def material():
    return jm.vonMisesIsotropicHardening(
        elasticity=jm.LinearElasticIsotropic(E=210e3, nu=0.3),
        yield_stress=Hardening(sig0=251.25, H=1234.5),
    )


def printed(capsys, obj, **kwargs):
    print_eqx_fields(obj, **kwargs)
    return capsys.readouterr().out.splitlines()


def test_default_printing(capsys):
    lines = printed(capsys, material(), fields=["elasticity"])
    assert lines == [
        "vonMisesIsotropicHardening:",
        "  elasticity:",
        "    LinearElasticIsotropic:",
        "      E = 210000.0",
        "      nu = 0.3",
    ]


def test_nested_field_selection(capsys):
    lines = printed(capsys, material(), fields=["elasticity.nu", "yield_stress"])
    assert "      nu = 0.3" in lines
    assert not any("E =" in line for line in lines)
    assert any("sig0" in line for line in lines)
    assert not any("solver" in line or "plastic_surface" in line for line in lines)


def test_constrained_parameters_show_physical_values(capsys):
    lines = printed(capsys, material(), fields=["yield_stress"], format=".2f")
    assert "      sig0 = 251.25" in lines
    assert "      H = 1234.50" in lines


def test_format_whole_model(capsys):
    """A format applies to numbers only: functions and booleans of solvers print as is."""
    lines = printed(capsys, material(), format=".3e")
    assert "      E = 2.100e+05" in lines
    assert "      rtol = 1.000e-08" in lines
    assert any(line.startswith("      norm = <function") for line in lines)
    assert "      cauchy_termination = True" in lines


def test_format_per_field(capsys):
    lines = printed(
        capsys,
        material(),
        fields=["elasticity", "yield_stress"],
        format={"elasticity": {"E": ".1e"}, "yield_stress": ".1f"},
    )
    assert "      E = 2.1e+05" in lines
    assert "      nu = 0.3" in lines  # no specification
    assert "      sig0 = 251.2" in lines


def test_arrays_sequences_and_other_values(capsys):
    obj = Fields(
        vector=jnp.array([1.0, 2.5]),
        flags=jnp.array([True, False]),
        pair=(1.0, 2.0),
        items=[0.5, eqx.nn.Identity()],
        label="steel",
        active=True,
    )
    lines = printed(capsys, obj, format={"vector": ".2f", "flags": ".2f", "pair": [".1f", ".2e"]})
    assert "  vector = [1.00 2.50]" in lines
    assert "  flags = [ True False]" in lines
    assert "  pair = (1.0, 2.00e+00)" in lines
    lines = printed(capsys, obj, format=".1f")
    assert "  items = [0.5, Identity()]" in lines
    assert "  label = steel" in lines
    assert "  active = True" in lines


def test_top_level_sequence(capsys):
    assert printed(capsys, [1.0, 2.0], format=[".1f", ".2f"]) == ["[0]: 1.0", "[1]: 2.00"]
    with pytest.raises(ValueError):
        print_eqx_fields([1.0, 2.0], format=[".1f"])


def test_rebuilt_module_prints_like_the_original(capsys):
    """A module rebuilt by a tree map, e.g. after a calibration, prints the same fields."""
    model = material()
    trainable, static = partition_by_node_names(model, ["elasticity"])
    assert printed(capsys, eqx.combine(trainable, static)) == printed(capsys, model)
    assert not any("Missing" in line or "__" in line for line in printed(capsys, model))
