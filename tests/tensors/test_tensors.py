import jax
import jax.numpy as jnp
import pytest

from jaxmat.state import make_batched
from jaxmat.tensors import (
    IsotropicTensor4,
    SymmetricTensor2,
    SymmetricTensor4,
    Tensor2,
    dev,
    polar,
    stretch_tensor,
    sym,
)


def _tensor2_init(tensor_type, T_, T_vect_):
    T = tensor_type(tensor=T_)
    assert jnp.allclose(T, T_)
    assert jnp.allclose(T.array, T_vect_)
    T2 = tensor_type(array=T_vect_)
    assert jnp.allclose(T2, T_)
    assert jnp.allclose(T.T, T_.T)
    assert jnp.allclose(
        (T + T).array,
        2 * T_vect_,
    )
    assert type((T @ T.T * jnp.linalg.det(T)).sym) is SymmetricTensor2
    assert jnp.allclose(
        (3 * T - T).array,
        2 * T_vect_,
    )
    assert jnp.allclose(
        (-T).array,
        -T_vect_,
    )
    assert jnp.allclose(
        (T / 2).array,
        0.5 * T_vect_,
    )
    assert jnp.allclose(
        T @ T,
        T_ @ T_,
    )


def test_tensor2_init():
    T_ = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=jnp.float64)
    T_vect_ = jnp.array([0, 4, 8, 1, 3, 2, 6, 5, 7], dtype=jnp.float64)
    _tensor2_init(Tensor2, T_, T_vect_)
    # check wrong size on initialization
    with pytest.raises(ValueError):
        SymmetricTensor2(array=T_vect_)
    # # Warning: we don't check for symmetry upon initialization
    # # But symmetry can be checked
    # assert not SymmetricTensor2(tensor=T_).is_symmetric()
    assert jnp.allclose(Tensor2.identity(), jnp.eye(3))
    T = Tensor2(tensor=jnp.ones((4, 3, 3)))
    assert T.tensor.shape == (4, 3, 3)
    T = Tensor2(tensor=jnp.ones((4, 5, 3, 3)))
    assert T.tensor.shape == (4, 5, 3, 3)
    with pytest.raises(ValueError):
        Tensor2(tensor=jnp.ones((4, 4, 3)))


def test_sym_tensor2_init():
    S_ = jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]], dtype=jnp.float64)
    S_vect_ = jnp.array(
        [0, 3, 5, jnp.sqrt(2) * 1, jnp.sqrt(2) * 2, jnp.sqrt(2) * 4], dtype=jnp.float64
    )
    # this passes
    Tensor2(tensor=S_)
    # this does not
    with pytest.raises(ValueError):
        Tensor2(array=S_vect_)

    S2_ = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.float64)
    S = SymmetricTensor2(tensor=S_)
    S2 = SymmetricTensor2(tensor=S2_)
    assert isinstance(S @ S2, Tensor2)
    assert not jnp.allclose(S @ S2, S2 @ S)
    assert isinstance((S @ S2).sym, SymmetricTensor2)
    assert jnp.allclose((S @ S2).sym, (S2 @ S).sym)


def test_symmetries():
    gamma = 0.75
    Id = SymmetricTensor2.identity()
    F = Tensor2(tensor=jnp.array([[0, gamma, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.float64))
    f1 = F + Id
    f2 = Id + F
    g1 = F @ Id
    g2 = Id @ F
    h1 = F @ Id.tensor
    # h2 does not inherit from F since @ is left-dominated
    h2 = Id.tensor @ F
    assert type(f1) is Tensor2
    assert type(f2) is Tensor2
    assert type(g1) is Tensor2
    assert type(g2) is Tensor2
    assert type(h1) is Tensor2
    assert type(h2) is not Tensor2
    assert type(2 * Id) is SymmetricTensor2
    assert type(Id + Id) is SymmetricTensor2
    assert type(Id - Id) is SymmetricTensor2


def test_stretch_tensor():
    gamma = 0.75
    Id = SymmetricTensor2.identity()
    F = Id + Tensor2(tensor=jnp.array([[0, gamma, 0], [0, 0, 0], [0, 0, 0]], dtype=jnp.float64))
    R, U = polar(F)
    C = (F.T @ F).sym
    B = (F @ F.T).sym
    # print(C, B)
    assert jnp.allclose(F, R @ U)
    assert jnp.allclose(C, U @ U)
    assert jnp.allclose(Tensor2.identity(), R.T @ R)
    V, R_ = polar(F, mode="VR")
    assert jnp.allclose(R, R_)
    assert jnp.allclose(B, V @ V)
    U_ = stretch_tensor(F)
    assert jnp.allclose(U, U_)
    V, R_ = polar(F.tensor, mode="VR")


def test_tensor4():
    Id = SymmetricTensor4.identity()
    Id2 = SymmetricTensor2.identity()
    key = jax.random.PRNGKey(0)
    A_ = jax.random.normal(key, (6, 6))
    A_ = 0.5 * (A_ + A_.T)
    b_ = jax.random.normal(key, (3, 3))
    b_ = 0.5 * (b_ + b_.T)
    A = SymmetricTensor4(array=A_)
    B = SymmetricTensor2(tensor=b_)
    assert jnp.allclose(A @ Id, A)
    assert jnp.allclose((A @ Id).array, A_)
    assert jnp.allclose(Id @ B, B)

    J = SymmetricTensor4.J()
    K = SymmetricTensor4.K()
    assert type(2.0 * J + 2.0 * K) is SymmetricTensor4
    assert jnp.allclose(2 * J + 2 * K, 2 * Id)
    assert jnp.allclose(J @ B, jnp.trace(B) / 3 * Id2)
    assert jnp.allclose(K @ B, dev(B))
    assert jnp.allclose(J @ J, J)
    assert jnp.allclose(K @ K, K)
    assert jnp.allclose(J @ K, 0)


def test_isotropic_tensor():
    kappa = 1.0
    mu = 1.0
    lmbda = kappa - 2 / 3 * mu
    C = IsotropicTensor4(kappa, mu)
    assert lmbda + 2 * mu == C.array[2, 2]
    assert lmbda == C.array[0, 1]
    assert 2 * mu == C.array[4, 4]
    C_ = SymmetricTensor4(array=C.array)
    S = IsotropicTensor4(1 / 9 / kappa, 1 / 4 / mu)
    assert jnp.allclose(C_.inv, S)
    assert jnp.allclose(C_.inv, C.inv)


def test_operator_symmetry():
    kappa = 1.0
    mu = 1.0
    C = IsotropicTensor4(kappa, mu)
    K = SymmetricTensor4.K()
    eps = SymmetricTensor2.identity()
    assert type(sym(eps)) is SymmetricTensor2
    assert type(dev(eps)) is SymmetricTensor2
    assert type(K @ eps) is SymmetricTensor2
    assert type(C @ eps) is SymmetricTensor2
    assert type(C @ K @ eps) is SymmetricTensor2


@pytest.mark.parametrize("cls", [Tensor2, SymmetricTensor2])
def test_batch_tensors(cls):
    Nbatch = 5
    val = 0.5 * jnp.eye(3)
    A = make_batched(cls(tensor=val), Nbatch=Nbatch)
    assert type(A) is cls
    assert A.tensor.shape == (Nbatch, 3, 3)
    if cls == Tensor2:
        d = 9
    elif cls == SymmetricTensor2:
        d = 6
    assert A.array.shape == (Nbatch, d)
    assert jnp.allclose(A[1], val)
    assert type(A + A) is cls
    assert jnp.allclose(A + A, jnp.broadcast_to(2 * val, (Nbatch, 3, 3)))
    # matmult doc: If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    assert type(A @ A) is cls if cls == Tensor2 else Tensor2
    # matmult doc: If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
    assert jnp.allclose(A @ A, jnp.broadcast_to(val @ val, (Nbatch, 3, 3)))


@pytest.mark.parametrize("cls", [Tensor2, SymmetricTensor2])
def test_double_batch_tensors(cls):
    Nbatch1, Nbatch2 = 4, 5
    val = 0.5 * jnp.eye(3)
    A_ = make_batched(cls(tensor=val), Nbatch=Nbatch2)
    A = make_batched(A_, Nbatch=Nbatch1)
    assert type(A) is cls
    assert A.tensor.shape == (Nbatch1, Nbatch2, 3, 3)
    if cls == Tensor2:
        d = 9
    elif cls == SymmetricTensor2:
        d = 6
    assert A.array.shape == (Nbatch1, Nbatch2, d)
    assert jnp.allclose(A[1], val)
    assert type(A + A) is cls
    assert jnp.allclose(A + A, jnp.broadcast_to(2 * val, (Nbatch1, Nbatch2, 3, 3)))
    assert type(A @ A) is cls if cls == Tensor2 else Tensor2
    assert jnp.allclose(A @ A, jnp.broadcast_to(val @ val, (Nbatch1, Nbatch2, 3, 3)))


# FIXME: should better handle views and array operations on tensors,
# see https://github.com/bleyerj/jaxmat/issues/16
@pytest.mark.xfail(reason="See issue #16")
def test_symmetry_preserving():
    N = 3
    sig = make_batched(SymmetricTensor2.identity(), N)
    sig2 = SymmetricTensor2()
    assert isinstance(sig2 + jnp.sum(sig, axis=0), SymmetricTensor2)
    assert isinstance(jnp.sum(sig, axis=0), SymmetricTensor2)
    assert isinstance(sig[0] + sig[1] + sig[2], SymmetricTensor2)
