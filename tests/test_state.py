import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from optax.tree_utils import tree_add, tree_scale, tree_zeros_like

from jaxmat.state import (
    FiniteStrainState,
    SmallStrainState,
    make_batched,
)
from jaxmat.tensors import SymmetricTensor2, Tensor
from jaxmat.utils import default_value


class MyState(SmallStrainState):
    other_attribute: jax.Array = None


class OtherState(SmallStrainState):
    float_attribute: float = default_value(0.0)
    array_attribute: jax.Array = eqx.field(default_factory=lambda: jnp.ones((3,)))
    tensor_attribute: SymmetricTensor2 = SymmetricTensor2()


def test_state():
    state = MyState(internal=jnp.zeros((3,)))
    assert hasattr(state, "strain")
    assert hasattr(state, "stress")
    assert hasattr(state, "internal")
    assert hasattr(state, "other_attribute")

    ones = jnp.ones((3, 3))
    state = state.update(strain=ones)
    assert jnp.allclose(state.strain, ones)
    state = state.add(stress=2 * ones, strain=0.5 * ones)
    assert jnp.allclose(state.strain, 1.5 * ones)
    assert jnp.allclose(state.stress, 2 * ones)

    state = state.add(foobar=0)
    assert jnp.allclose(state.internal, jnp.zeros((3,)))
    assert state.other_attribute is None
    assert not hasattr(state, "foobar")


def test_state_tensor_conversion():
    state = OtherState(tensor_attribute=SymmetricTensor2.identity())
    new_state = jax.tree.map(
        lambda x: x.array if isinstance(x, Tensor) else x,
        state,
        is_leaf=lambda x: isinstance(x, Tensor),
    )
    assert new_state.strain.shape == (6,)
    assert new_state.stress.shape == (6,)
    assert new_state.float_attribute.shape == ()
    assert new_state.array_attribute.shape == (3,)
    assert new_state.tensor_attribute.shape == (6,)


@pytest.mark.parametrize("Nbatch", [1, 10, 100])
def test_batching(Nbatch):
    state = MyState(internal=jnp.zeros((3,)))
    batched_state = make_batched(state, Nbatch)
    assert batched_state.strain.tensor.shape == (Nbatch, 3, 3)
    assert batched_state.stress.tensor.shape == (Nbatch, 3, 3)
    assert batched_state.internal.shape == (Nbatch, 3)
    assert batched_state.other_attribute is None

    state = MyState(
        internal=jnp.zeros((3, 3)),
    )
    batched_state = make_batched(state, Nbatch)
    assert batched_state.internal.shape == (Nbatch, 3, 3)


def test_tree_utils():
    class MyModule(eqx.Module):
        var: float = 0

    m1 = MyModule(var=1.0)
    m2 = MyModule(var=jnp.eye(3))
    m = tree_add(m1, m2)
    assert jnp.allclose(m.var, jnp.eye(3) + 1)
    m = tree_zeros_like(m2)
    assert jnp.allclose(m.var, jnp.zeros_like(m2.var))
    m = tree_scale(3.0, m1)
    assert jnp.allclose(m.var, 3 * m1.var)


def test_small_strain():
    state = SmallStrainState(strain=SymmetricTensor2.identity())
    state = state.update(stress=2 * state.strain)
    assert jnp.allclose(state.strain, jnp.eye(3))
    assert jnp.allclose(state.eps, jnp.eye(3))
    assert jnp.allclose(state.stress, 2 * jnp.eye(3))
    assert jnp.allclose(state.sig, 2 * jnp.eye(3))
    state = state.update(sig=3 * state.strain)
    assert jnp.allclose(state.stress, 3 * jnp.eye(3))
    assert jnp.allclose(state.sig, 3 * jnp.eye(3))


def test_finite_strain():
    state = FiniteStrainState()
    assert jnp.allclose(state.F, jnp.eye(3))
    state = state.update(PK1=2 * state.F)
    assert jnp.allclose(state.PK1, 2 * jnp.eye(3))
    state = state.update(PK1=3 * state.F)
    assert jnp.allclose(state.PK1, 3 * jnp.eye(3))
