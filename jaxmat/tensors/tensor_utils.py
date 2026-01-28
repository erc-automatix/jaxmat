from functools import partial

import jax
import jax.numpy as jnp

from . import SymmetricTensor2, SymmetricTensor4, Tensor2
from .linear_algebra import _sqrtm, eig33


@partial(jax.jit, static_argnums=1)
def polar(F, mode="RU"):
    """Computes the 'RU' or 'VR' polar decomposition of F."""
    C = F.T @ F
    U, U_inv = _sqrtm(jnp.asarray(C))
    R = F @ U_inv
    if mode == "RU":
        return R, SymmetricTensor2(tensor=U)
    elif mode == "VR":
        V = (R @ U @ R.T).sym
        return V, R


def sym(A):
    """Computes the symmetric part of a tensor."""
    return 0.5 * (A + A.T)


def skew(A):
    """Computes the skew part of a tensor."""
    return 0.5 * (A - A.T)


def axl(A):
    """Computes the axial part of a skew symmetric tensor.
    If not skew-symmetric, this function takes the skew-symmetric part first."""
    As = skew(A)
    return jnp.array([-As[1, 2], As[0, 2], -As[0, 1]])


def tr(A):
    """Trace of a n-dim 2nd-rank tensor."""
    return jnp.trace(A)


def dev(A):
    Id = SymmetricTensor2.identity()
    return A - Id * (tr(A) / A.dim)


def stretch_tensor(F):
    """Computes the strech tensor U = sqrtm(F.T @ F)."""
    return polar(F)[1]


@jax.custom_jvp
def eigenvalues(sig):
    eigvals, _eigendyads = eig33(sig)
    return eigvals


@eigenvalues.defjvp
def eigenvalues_jvp(primals, tangents):
    (sig,) = primals
    (dsig,) = tangents
    eigvals, eigendyads = eig33(sig)
    deig = jnp.tensordot(eigendyads, dsig)
    return eigvals, deig
