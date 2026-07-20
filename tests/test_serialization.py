import jax
import jax.numpy as jnp
import equinox as eqx
import tempfile

from jaxmat.utils import save_model, load_model, enforce_dtype


class Module1(eqx.Module):
    A: float = enforce_dtype()
    B: float = enforce_dtype()


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


def test_save_load_without_hyperparams():
    model = Module1(A=11, B=12)
    file = tempfile.NamedTemporaryFile()
    save_model(file.name, model)
    loaded_model = load_model(file.name, Module1(A=0, B=0))
    assert eqx.tree_equal(model, loaded_model)


def test_save_load_with_hyperparams():
    model = Module2(len_A=3, A=[1, 2, 5])
    file = tempfile.NamedTemporaryFile()
    save_model(file.name, model, {"len_A": model.len_A})
    loaded_model = load_model(file.name, lambda len_A: Module2(len_A))
    assert eqx.tree_equal(model, loaded_model)


if __name__ == "__main__":
    test_save_load_without_hyperparams()
    test_save_load_with_hyperparams()
