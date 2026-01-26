import jax
import jax.numpy as jnp
import equinox as eqx
from optax.tree_utils import tree_add, tree_zeros_like, tree_scale
from jaxmat.state import (
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)
from jaxmat.utils import default_value
from jaxmat.tensors import SymmetricTensor2, Tensor
import pytest


class State(SmallStrainState):
    float_attribute: float = default_value(0.0)
    array_attribute: jax.Array = eqx.field(default_factory=lambda: jnp.ones((3,)))
    tensor_attribute: SymmetricTensor2 = SymmetricTensor2()
    batched_tensor_attribute: SymmetricTensor2 = eqx.field(
        default_factory=lambda: make_batched(SymmetricTensor2(tensor=jnp.eye(3)), 10)
    )


N = 100000
state = make_batched(State(), N)


def flatten(x):
    if isinstance(x, Tensor):
        return x.array
    else:
        return x


@eqx.filter_jit
def convert(state):
    return jax.tree.map(flatten, state, is_leaf=lambda x: isinstance(x, Tensor))


convert(state)

from time import time

t0 = time()
convert(state)
t_exec = time() - t0

print(f"Execution time: {t_exec*1000:.3f} ms")
