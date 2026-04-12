"""
jaxmat/tensors/tensor_utils.py

Functional utilities for second-rank tensor algebra.

All functions here either return :class:`~jaxmat.tensors.Tensor2` /
:class:`~jaxmat.tensors.SymmetricTensor2` objects, or require tensor-specific
knowledge (e.g. the Kelvin dot product in :func:`norm`, the custom JVP in
:func:`eigenvalues`).

Array-level operations — including isotropic matrix functions
(:func:`~jaxmat.tensors.linear_algebra.expm`,
:func:`~jaxmat.tensors.linear_algebra.logm`, etc.) — live in
:mod:`jaxmat.tensors.linear_algebra` and accept tensor objects directly via
``__jax_array__``, so no explicit wrapping is needed.
"""

from functools import partial

import jax
import jax.numpy as jnp

from jaxmat.tensors.generic_tensors import SymmetricTensor2, Tensor2
from jaxmat.tensors.linear_algebra import _sqrtm, eig33
from jaxmat.tensors.utils import safe_sqrt

# ─────────────────────────────────────────────────────────────────────────────
# Decompositions — return tensor objects
# ─────────────────────────────────────────────────────────────────────────────


def sym(A: Tensor2) -> SymmetricTensor2:
    r"""
    Symmetric part $(\mathbf{A} + \mathbf{A}^{\mathsf{T}}) / 2$.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    SymmetricTensor2
    """
    return A.sym


def skw(A: Tensor2) -> Tensor2:
    r"""
    Skew-symmetric part $(\mathbf{A} - \mathbf{A}^{\mathsf{T}}) / 2$.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    Tensor2
    """
    return A.skw


def vol(A) -> SymmetricTensor2:
    r"""
    Volumetric (spherical) part $\tfrac{1}{3}\operatorname{tr}(\mathbf{A})\,\mathbf{I}$.

    Complement of :func:`dev`: ``vol(A) + dev(A) == A`` for symmetric ``A``.

    Parameters
    ----------
    A : Tensor2 or SymmetricTensor2

    Returns
    -------
    SymmetricTensor2
    """
    return SymmetricTensor2.identity() * (tr(A) / 3)


def dev(A) -> SymmetricTensor2:
    r"""
    Deviatoric part $\mathbf{A} - \tfrac{1}{3}\operatorname{tr}(\mathbf{A})\,\mathbf{I}$.

    Parameters
    ----------
    A : Tensor2 or SymmetricTensor2

    Returns
    -------
    SymmetricTensor2
    """
    return A - SymmetricTensor2.identity() * (tr(A) / 3)


# ─────────────────────────────────────────────────────────────────────────────
# Scalar invariants
# ─────────────────────────────────────────────────────────────────────────────


def tr(A: Tensor2) -> jax.Array:
    r"""
    Trace $\operatorname{tr}(\mathbf{A}) = A_{ii}$.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    jax.Array
        Scalar (or batch of scalars).
    """
    return A.tr


def norm(A: Tensor2) -> jax.Array:
    r"""
    Frobenius norm $\|\mathbf{A}\| = \sqrt{\mathbf{A}:\mathbf{A}}$.

    For :class:`SymmetricTensor2` operands the double contraction is evaluated
    as a Kelvin dot product with no dense intermediate.

    Parameters
    ----------
    A : Tensor2 or SymmetricTensor2

    Returns
    -------
    jax.Array
        Scalar (or batch of scalars).
    """
    return safe_sqrt(A.double_contract(A))


def von_mises(sig) -> jax.Array:
    r"""
    Von Mises equivalent stress.

    .. math::

        \sigma_\text{VM} = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}, \qquad
        \mathbf{s} = \operatorname{dev}(\boldsymbol{\sigma})

    Parameters
    ----------
    sig : Tensor2 or SymmetricTensor2
        Cauchy stress tensor.

    Returns
    -------
    jax.Array
        Scalar (or batch of scalars).
    """
    s = dev(sig)
    return safe_sqrt(1.5 * s.double_contract(s))


def axl(A: Tensor2) -> jax.Array:
    r"""
    Axial vector of a skew-symmetric tensor.

    The axial vector $\mathbf{w}$ associated with $\mathbf{W} = \operatorname{skw}(\mathbf{A})$
    satisfies $\mathbf{W}\mathbf{v} = \mathbf{w} \times \mathbf{v}$ and is given by
    $w_i = -\tfrac{1}{2}\,\varepsilon_{ijk}\,W_{jk}$.

    Parameters
    ----------
    A : Tensor2

    Returns
    -------
    jax.Array
        Shape ``(..., 3)``.
    """
    W = A.skw
    return jnp.array([-W[1, 2], W[0, 2], -W[0, 1]])


# ─────────────────────────────────────────────────────────────────────────────
# Polar decomposition
# ─────────────────────────────────────────────────────────────────────────────


@partial(jax.jit, static_argnums=1)
def polar(F: Tensor2, mode: str = "RU") -> tuple:
    r"""
    Polar decomposition $\mathbf{F} = \mathbf{R}\mathbf{U}$ or
    $\mathbf{F} = \mathbf{V}\mathbf{R}$.

    Parameters
    ----------
    F : Tensor2
        Deformation gradient.
    mode : {"RU", "VR"}, optional
        Selects the right polar decomposition (``"RU"``, default) or the
        left polar decomposition (``"VR"``).

    Returns
    -------
    tuple
        ``(R, U)`` for ``mode="RU"`` where ``R`` is a :class:`Tensor2`
        (rotation) and ``U`` a :class:`SymmetricTensor2` (right stretch), or
        ``(V, R)`` for ``mode="VR"`` where ``V`` is a :class:`SymmetricTensor2`
        (left stretch).
    """
    C = (F.T @ F).sym
    U_dense, U_inv_dense = _sqrtm(jnp.asarray(C))
    U = SymmetricTensor2(tensor=U_dense)
    R = F @ Tensor2(tensor=U_inv_dense)
    if mode == "RU":
        return R, U
    V = (R @ U @ R.T).sym
    return V, R


def stretch_tensor(F: Tensor2) -> SymmetricTensor2:
    r"""
    Right stretch tensor $\mathbf{U} = (\mathbf{F}^{\mathsf{T}}\mathbf{F})^{1/2}$.

    Convenience wrapper around :func:`polar`.

    Parameters
    ----------
    F : Tensor2
        Deformation gradient.

    Returns
    -------
    SymmetricTensor2
    """
    return polar(F)[1]


# ─────────────────────────────────────────────────────────────────────────────
# Spectral analysis
# ─────────────────────────────────────────────────────────────────────────────


@jax.custom_jvp
def eigenvalues(sig: SymmetricTensor2) -> jax.Array:
    r"""
    Eigenvalues $\lambda_1 \leq \lambda_2 \leq \lambda_3$ of a symmetric tensor.

    Wraps :func:`~jaxmat.tensors.linear_algebra.eig33` and provides a
    stable custom JVP via the eigenprojectors, correct even for repeated
    or near-repeated eigenvalues.

    Parameters
    ----------
    sig : SymmetricTensor2

    Returns
    -------
    jax.Array
        Shape ``(..., 3)``.

    Notes
    -----
    The custom JVP computes
    $\dot{\lambda}_i = (\mathbf{n}_i \otimes \mathbf{n}_i) : \dot{\mathbf{A}}$
    from the eigenprojectors returned by ``eig33``.
    """
    eigvals, _ = eig33(jnp.asarray(sig))
    return eigvals


@eigenvalues.defjvp
def _eigenvalues_jvp(primals, tangents):
    (sig,) = primals
    (dsig,) = tangents
    eigvals, eigendyads = eig33(jnp.asarray(sig))
    deig = jnp.tensordot(eigendyads, jnp.asarray(dsig), axes=([-2, -1], [-2, -1]))
    return eigvals, deig
