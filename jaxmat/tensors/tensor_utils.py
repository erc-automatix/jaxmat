from functools import partial

import jax
import jax.numpy as jnp

from jaxmat.tensors import SymmetricTensor2
from jaxmat.tensors.linear_algebra import _sqrtm, eig33


@partial(jax.jit, static_argnums=1)
def polar(F, mode="RU"):
    r"""
    Polar decomposition of a second-rank tensor.

    Computes either the right or left polar decomposition

    $$\bF = \bR \bU \quad \text{or}\quad   \bF = \bV \bR$$,

    where $\bR$ is orthogonal and $\bU, \bV$ are symmetric positive definite stretch
    tensors.

    Parameters
    ----------
    F : Tensor2 or array_like, shape (..., 3, 3)
        Deformation gradient.
    mode : {"RU", "VR"}, optional
        Type of decomposition. ``"RU"`` returns $(\bR, \bU)$, ``"VR"`` returns $(\bV, \bR)$.

    Returns
    -------
    tuple
        Rotation and stretch tensors according to ``mode``.
    """
    C = F.T @ F
    U, U_inv = _sqrtm(jnp.asarray(C))
    R = F @ U_inv
    if mode == "RU":
        return R, SymmetricTensor2(tensor=U)
    elif mode == "VR":
        V = (R @ U @ R.T).sym
        return V, R


def sym(A):
    r"""
    Symmetric part of a second-rank tensor $\bA$.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    SymmetricTensor2
        $(\bA + \bA\T) / 2$.
    """
    return A.sym


def skew(A):
    r"""
    Skew-symmetric part of a second-rank tensor $\bA$.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    Tensor2
        $(\bA - \bA\T) / 2$.
    """
    return 0.5 * (A - A.T)


def axl(A):
    """
    Axial vector associated with a skew-symmetric tensor.

    Parameters
    ----------
    A : Tensor2
        Skew-symmetric or general tensor. The skew part is used.

    Returns
    -------
    jax.Array, shape (3,)
        Axial (dual) vector.
    """
    As = skew(A)
    return jnp.array([-As[1, 2], As[0, 2], -As[0, 1]])


def tr(A):
    r"""
    Trace of a second-rank tensor.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    jax.Array
        Sum of diagonal components $\tr(\bA)=A_{ii}$.
    """
    return A.tr


def dev(A):
    r"""
    Deviatoric part of a second-rank tensor $\bA$.

    Parameters
    ----------
    A : Tensor2 or SymmetricTensor2

    Returns
    -------
    Tensor2
        $\dev(\bA) = \bA - (\tr(\bA) / dim) \bI$.
    """
    Id = SymmetricTensor2.identity()
    return A - Id * (tr(A) / A.dim)


def stretch_tensor(F):
    r"""
    Right stretch tensor from polar decomposition.

    Parameters
    ----------
    F : Tensor2 or array_like

    Returns
    -------
    SymmetricTensor2
        $\bU = (\bF\T\bF)^{1/2}$.
    """
    return polar(F)[1]


@jax.custom_jvp
def eigenvalues(sig):
    """
    Eigenvalues of a symmetric second-rank tensor.

    This function defines a custom JVP rule to provide stable and efficient
    differentiation of eigenvalues.

    Parameters
    ----------
    sig : SymmetricTensor2 or array_like, shape (..., 3, 3)

    Returns
    -------
    jax.Array, shape (..., 3)
        Eigenvalues.
    """
    eigvals, _ = eig33(sig)
    return eigvals


@eigenvalues.defjvp
def eigenvalues_jvp(primals, tangents):
    (sig,) = primals
    (dsig,) = tangents
    eigvals, eigendyads = eig33(sig)
    deig = jnp.tensordot(eigendyads, dsig)
    return eigvals, deig
