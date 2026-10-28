"""Tensor operations on batches give the same results as pointwise.

Batches of 6 tensors are used where a batch dimension could be confused with a storage
dimension (6 Kelvin-Mandel components), which used to give silently wrong results, and a
batch shape (2, 3) for nested batches. Each operation on a batch is compared with the same
operation applied to each tensor.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmat.tensors import (
    IsotropicTensor4,
    SymmetricTensor2,
    SymmetricTensor4,
    Tensor2,
    Tensor4,
    TransverseIsotropicTensor4,
    axl,
    det33,
    dev,
    eig33,
    eigenvalue_function,
    eigenvalues,
    expm,
    inv_sqrtm,
    isotropic_function,
    logm,
    main_invariants,
    norm,
    polar,
    powm,
    pq_invariants,
    principal_invariants,
    rotation,
    sqrtm,
    stretch_tensor,
    tr,
    vol,
    von_mises,
)

jax.config.update("jax_enable_x64", True)

BATCH_SHAPES = [(6,), (2, 3)]


def leaves_close(a, b, rtol=1e-10, atol=1e-12):
    a, b = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape
        assert jnp.allclose(x, y, rtol=rtol, atol=atol)


def pointwise(batch_shape, fun, *batched):
    """``fun`` applied to each element of pytrees (tensors or arrays) with leading
    ``batch_shape``, and the results stacked back to that batch shape."""
    k = len(batch_shape)
    flat = [jax.tree.map(lambda x: x.reshape((-1, *x.shape[k:])), b) for b in batched]
    n = jax.tree.leaves(flat[0])[0].shape[0]
    results = [fun(*[jax.tree.map(lambda x, i=i: x[i], f) for f in flat]) for i in range(n)]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *results)
    return jax.tree.map(lambda x: x.reshape(batch_shape + x.shape[1:]), stacked)


def general(batch_shape, key=0):
    A = jax.random.normal(jax.random.PRNGKey(key), (*batch_shape, 3, 3))
    return Tensor2(tensor=A + 3 * jnp.eye(3))


def spd(batch_shape, key=1):
    """Symmetric positive definite, some with repeated eigenvalues."""
    n = 1
    for d in batch_shape:
        n *= d
    R = rotation.random(jax.random.PRNGKey(key), (n,))
    lam = jax.random.uniform(jax.random.PRNGKey(key + 1), (n, 3), minval=0.5, maxval=2.0)
    lam = lam.at[::2, 2].set(lam[::2, 1])  # two equal eigenvalues for every other one
    A = jnp.einsum("nij,nj,nkj->nik", R, lam, R)
    return SymmetricTensor2(tensor=A.reshape((*batch_shape, 3, 3)))


@pytest.fixture(params=BATCH_SHAPES, ids=str)
def batch_shape(request):
    return request.param


@pytest.mark.parametrize(
    "fun",
    [
        lambda T: T.det,
        lambda T: T.inv,
        lambda T: T.tr,
        lambda T: T.T,
        lambda T: T.sym,
        lambda T: T.skw,
        norm,
        tr,
        dev,
        vol,
        von_mises,
        axl,
        lambda T: T.double_contract(T.T),
        lambda T: T.matvec(jnp.array([1.0, -2.0, 0.5])),
        lambda T: polar(T),
        lambda T: polar(T, mode="VR"),
        stretch_tensor,
        lambda T: T @ T,
    ],
    ids=[
        "det", "inv", "tr property", "transpose", "sym", "skw", "norm", "tr", "dev", "vol",
        "von_mises", "axl", "double_contract", "matvec", "polar RU", "polar VR",
        "stretch_tensor", "matmul",
    ],
)  # fmt: skip
def test_tensor2_operations(fun, batch_shape):
    T = general(batch_shape)
    leaves_close(fun(T), pointwise(batch_shape, fun, T))


@pytest.mark.parametrize(
    "fun",
    [
        det33,
        lambda A: principal_invariants(A),
        lambda A: main_invariants(A),
        lambda A: pq_invariants(A),
        lambda A: eig33(A)[0],
        lambda A: logm(A),
        lambda A: expm(A),
        lambda A: sqrtm(A),
        lambda A: inv_sqrtm(A),
        lambda A: powm(A, 1.7),
        lambda A: isotropic_function(jnp.power, A, 0.5),
        lambda A: eigenvalue_function(lambda lam: jnp.sum(lam**3), A),
        lambda A: eigenvalues(SymmetricTensor2(tensor=A)),
    ],
    ids=[
        "det33", "principal_invariants", "main_invariants", "pq_invariants", "eig33", "logm",
        "expm", "sqrtm", "inv_sqrtm", "powm", "isotropic_function", "eigenvalue_function",
        "eigenvalues",
    ],
)  # fmt: skip
def test_array_functions(fun, batch_shape):
    A = spd(batch_shape).tensor
    leaves_close(fun(A), pointwise(batch_shape, fun, A))


def test_derivatives_on_batches(batch_shape):
    A = spd(batch_shape).tensor
    U = jnp.broadcast_to(jnp.array([[0.2, 0.5, 0.1], [0.5, -0.3, 0.6], [0.1, 0.6, 0.4]]), A.shape)

    def first(A_, U_):
        return jax.jvp(logm, (A_,), (U_,))[1]

    leaves_close(first(A, U), pointwise(batch_shape, first, A, U), rtol=1e-9)
    grad = jax.grad(lambda A_: jnp.sum(eigenvalue_function(lambda lam: jnp.sum(lam**3), A_)))
    leaves_close(grad(A), pointwise(batch_shape, grad, A), rtol=1e-9)


def test_eigenvalues_jvp_on_batches(batch_shape):
    S = spd(batch_shape)
    dS = SymmetricTensor2(array=jnp.ones_like(S.array))
    jvp = jax.jvp(eigenvalues, (S,), (dS,))[1]
    expected = pointwise(batch_shape, lambda s, ds: jax.jvp(eigenvalues, (s,), (ds,))[1], S, dS)
    leaves_close(jvp, expected)


def test_scalar_times_tensor_on_batches(batch_shape):
    """A tensor times a batch of scalars multiplies tensor by tensor. (A JAX array on the left,
    ``scalars * T``, is converted by JAX to a dense array before ``T.__rmul__`` is tried.)"""
    scalars = jnp.arange(1.0, 7.0).reshape(batch_shape)
    T, S = general(batch_shape), spd(batch_shape)
    C = SymmetricTensor4(array=jnp.broadcast_to(jnp.eye(6), (*batch_shape, 6, 6)))
    for X in (T, S, C):
        leaves_close(X * scalars, pointwise(batch_shape, lambda s, x: x * s, scalars, X))
        leaves_close(X / scalars, pointwise(batch_shape, lambda s, x: x / s, scalars, X))
    single = SymmetricTensor2.identity()
    assert (single * scalars).array.shape == (*batch_shape, 6)
    leaves_close(dev(S), S - SymmetricTensor2.identity() * (S.tr / 3))


def test_structured_scalar_multiplication_on_batches():
    C = IsotropicTensor4(kappa=jnp.ones(6), mu=jnp.ones(6))
    scalars = jnp.arange(1.0, 7.0)
    assert jnp.allclose((C * scalars).coeffs, scalars[:, None] * C.coeffs)


def test_kelvin_products_on_batches(batch_shape):
    """A single or batched stiffness applied to a batch of strains."""
    eps = spd(batch_shape)
    C = IsotropicTensor4(kappa=1.0, mu=2.0).to_symmetric()
    C_batch = SymmetricTensor4(array=jnp.broadcast_to(C.array, (*batch_shape, 6, 6)))
    expected = pointwise(batch_shape, lambda e: C @ e, eps)
    leaves_close(C @ eps, expected)
    leaves_close(C_batch @ eps, expected)
    leaves_close(IsotropicTensor4(kappa=1.0, mu=2.0) @ eps, expected)
    leaves_close(C @ eps.array, expected.array)  # plain Kelvin arrays
    TI = TransverseIsotropicTensor4(
        jnp.array([1.0, 2.0, 2.0]), jnp.array([5.0, 3.0, 1.0, 1.0, 2.0, 4.0])
    )
    leaves_close(TI @ eps, pointwise(batch_shape, lambda e: TI @ e, eps))
    D = Tensor4(tensor=jax.random.normal(jax.random.PRNGKey(4), (3, 3, 3, 3)))
    T = general(batch_shape)
    leaves_close(D @ T, pointwise(batch_shape, lambda t: D @ t, T))
    leaves_close(D @ eps, pointwise(batch_shape, lambda e: D @ e, eps))
    leaves_close(
        TI.fourth_contract(C_batch.array),
        pointwise(batch_shape, lambda c: TI.fourth_contract(c), C_batch),
    )


def test_rotations_on_batches(batch_shape):
    n = 6
    R = rotation.random(jax.random.PRNGKey(5), (n,)).reshape((*batch_shape, 3, 3))
    T = general(batch_shape)
    C = SymmetricTensor4(tensor=jax.random.normal(jax.random.PRNGKey(6), (3, 3, 3, 3)))
    C = SymmetricTensor4(array=0.5 * (C.array + C.array.T))
    C_batch = SymmetricTensor4(array=jnp.broadcast_to(C.array, (*batch_shape, 6, 6)))
    leaves_close(T.rotate(R), pointwise(batch_shape, lambda t, r: t.rotate(r), T, R))
    leaves_close(C_batch.rotate(R), pointwise(batch_shape, lambda c, r: c.rotate(r), C_batch, R))
    single_R = rotation.from_axis_angle(jnp.array([1.0, 2.0, 3.0]), 0.4)
    leaves_close(
        C_batch.rotate(single_R), pointwise(batch_shape, lambda c: c.rotate(single_R), C_batch)
    )
