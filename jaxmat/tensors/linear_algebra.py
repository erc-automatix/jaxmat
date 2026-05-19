"""
jaxmat/tensors/linear_algebra.py

Linear algebra operations on plain JAX arrays.

All functions operate on ``jax.Array`` objects of shape ``(3, 3)`` (or
batched equivalents).  Because tensor wrapper classes implement
``__jax_array__``, any function here accepts a :class:`~jaxmat.tensors.Tensor2`
or :class:`~jaxmat.tensors.SymmetricTensor2` directly — no explicit
conversion is needed.

Public API
----------
det33, inv33, eig33,
principal_invariants, main_invariants, pq_invariants,
isotropic_function, sqrtm, inv_sqrtm, expm, logm, powm

Private (internal helpers)
--------------------------
_dim, _tr, _dev, _sqrtm
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .utils import safe_norm, safe_sqrt

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _dim(A) -> int:
    r"""Spatial dimension of a square matrix $\mathbf{A}$, inferred from its shape."""
    return jnp.asarray(A).shape[0]


def _tr(A) -> jax.Array:
    r"""Trace $\operatorname{tr}(\mathbf{A}) = A_{ii}$."""
    return jnp.trace(A)


def _dev(A) -> jax.Array:
    r"""
    Deviatoric part of a $d \times d$ matrix $\mathbf{A}$.

    .. math:: \operatorname{dev}(\mathbf{A}) = \mathbf{A} -
        \frac{1}{d}\operatorname{tr}(\mathbf{A})\,\mathbf{I}
    """
    d = _dim(A)
    return A - _tr(A) / d * jnp.eye(d)


# ─────────────────────────────────────────────────────────────────────────────
# Public array-level operations
# ─────────────────────────────────────────────────────────────────────────────


def det33(A) -> jax.Array:
    r"""
    Determinant $\det(\mathbf{A})$ of a $3 \times 3$ matrix.

    Evaluated via the explicit Sarrus formula, avoiding ``jnp.linalg.det``
    overhead for the fixed-size case.

    Parameters
    ----------
    A : array_like, shape (3, 3)

    Returns
    -------
    jax.Array
        Scalar determinant.
    """
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]
    return (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )


def inv33(A) -> jax.Array:
    r"""
    Inverse $\mathbf{A}^{-1}$ of a $3 \times 3$ matrix.

    Computed via the explicit cofactor formula (adjugate divided by
    determinant), avoiding ``jnp.linalg.solve`` overhead for the fixed-size
    case.

    Parameters
    ----------
    A : array_like, shape (3, 3)

    Returns
    -------
    jax.Array
        Shape (3, 3).
    """
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]
    cof = jnp.array(
        [
            [a22 * a33 - a23 * a32, a13 * a32 - a12 * a33, a12 * a23 - a13 * a22],
            [a23 * a31 - a21 * a33, a11 * a33 - a13 * a31, a13 * a21 - a11 * a23],
            [a21 * a32 - a22 * a31, a12 * a31 - a11 * a32, a11 * a22 - a12 * a21],
        ]
    )
    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    return cof / det


def principal_invariants(A) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Principal invariants $(I_1, I_2, I_3)$ of a $3 \times 3$ matrix $\mathbf{A}$.

    .. math::

        I_1 = \operatorname{tr}(\mathbf{A}), \quad
        I_2 = \tfrac{1}{2}\bigl(\operatorname{tr}(\mathbf{A})^2
              - \operatorname{tr}(\mathbf{A}^2)\bigr), \quad
        I_3 = \det(\mathbf{A})

    Parameters
    ----------
    A : array_like, shape (3, 3)

    Returns
    -------
    I1, I2, I3 : jax.Array
        Three scalar invariants.
    """
    i1 = jnp.trace(A)
    i2 = (jnp.trace(A) ** 2 - jnp.trace(A @ A)) / 2
    i3 = det33(A)
    return i1, i2, i3


def main_invariants(A) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Main (trace-power) invariants $(J_1, J_2, J_3)$ of a $3 \times 3$ matrix.

    .. math::

        J_k = \operatorname{tr}(\mathbf{A}^k), \quad k = 1, 2, 3

    Parameters
    ----------
    A : array_like, shape (3, 3)

    Returns
    -------
    J1, J2, J3 : jax.Array
        Three scalar invariants.
    """
    j1 = jnp.trace(A)
    j2 = jnp.trace(A @ A)
    j3 = jnp.trace(A @ A @ A)
    return j1, j2, j3


def pq_invariants(sig) -> tuple[jax.Array, jax.Array]:
    r"""
    Hydrostatic pressure $p$ and deviatoric equivalent stress $q$.

    Commonly used in soil mechanics and pressure-sensitive plasticity.

    .. math::

        p = -\tfrac{1}{3}\operatorname{tr}(\boldsymbol{\sigma}), \qquad
        q = \sqrt{\tfrac{3}{2}\,\mathbf{s}:\mathbf{s}}

    where $\mathbf{s} = \operatorname{dev}(\boldsymbol{\sigma})$.

    Parameters
    ----------
    sig : array_like, shape (3, 3)
        Cauchy stress tensor.

    Returns
    -------
    p : jax.Array
        Mean pressure (positive in compression).
    q : jax.Array
        Von Mises equivalent stress.
    """
    p = -jnp.trace(sig) / 3
    s = _dev(sig)
    q = safe_sqrt(3.0 / 2.0 * jnp.vdot(s, s))
    return p, q


def eig33_HA(A, rtol=1e-16) -> tuple[jax.Array, jax.Array]:
    r"""
    Eigenvalues and eigenvalue dyads of a $3 \times 3$ real symmetric matrix.

    Implements the numerically stable method of Harari & Albocher (2023),
    which avoids catastrophic cancellation when two or more eigenvalues are
    nearly equal.  Eigenvalue dyads (derivatives of eigenvalues with respect
    to $\mathbf{A}$) are obtained via ``jax.jacfwd``.

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Real symmetric matrix.
    rtol : float, optional
        Relative tolerance for the near-isotropic and two-equal-eigenvalue
        branches.  Defaults to ``1e-16``.

    Returns
    -------
    eigvals : jax.Array, shape (3,)
        Eigenvalues in ascending order.
    eigendyads : jax.Array, shape (3, 3, 3)
        Rank-1 projectors $\mathbf{n}_i \otimes \mathbf{n}_i$ for each
        eigenvalue, forming the derivative
        $\partial \lambda_i / \partial \mathbf{A}$.

    Notes
    -----
    The input must be symmetric; asymmetric components are silently ignored
    by the algorithm.

    .. admonition:: References
        :class: seealso

        Harari, I., & Albocher, U. (2023). Computation of eigenvalues of a
        real, symmetric 3x3 matrix with particular reference to the
        pernicious case of two nearly equal eigenvalues. *International
        Journal for Numerical Methods in Engineering*, 124(5), 1089-1110.
    """

    def _compute(A):
        A = jnp.asarray(A)
        norm = safe_norm(A)
        Id = jnp.eye(_dim(A))
        I1 = jnp.trace(A)
        S = _dev(A)
        J2 = _tr(S.T @ S) / 2
        s = safe_sqrt(J2 / 3)

        def _near_iso(_):
            ev = jnp.ones((3,)) * I1 / 3
            return ev, ev

        def _general(_):
            T = S @ S - 2 * J2 / 3 * Id
            d = safe_norm(T - s * S) / safe_norm(T + s * S)
            sj = jnp.sign(1 - d)
            cond = sj * (1 - d) < rtol * norm

            def _two(_):
                lm = jnp.sqrt(3) * s
                ev = jnp.array([lm, 0.0, -lm]) + I1 / 3
                return ev, ev

            def _three(_):
                alpha = 2 / 3 * jnp.arctan2(safe_norm(T - s * S) ** sj, safe_norm(T + s * S) ** sj)
                ld = 2 * sj * s * jnp.cos(alpha)
                sd = jnp.sqrt(3) * s * jnp.sin(alpha)
                ev_dev = jnp.array([-ld / 2 - sd, -ld / 2 + sd, ld])
                return ev_dev + I1 / 3, ev_dev + I1 / 3

            return lax.cond(cond, _two, _three, operand=None)

        return lax.cond(s < rtol * norm, _near_iso, _general, operand=None)

    eigendyads, eigvals = jax.jacfwd(_compute, has_aux=True)(A)
    order = jnp.argsort(eigvals)
    eigvals = eigvals[order]
    eigendyads = eigendyads[order]
    eigendyads = 0.5 * (eigendyads + jnp.swapaxes(eigendyads, -1, -2))
    return eigvals, eigendyads


@partial(jax.jit, static_argnums=1)
def eig33(A, rtol=1e-16):
    norm_A = jnp.linalg.norm(A)

    def J2s(A):
        d0 = A[0, 0] - A[1, 1]
        d1 = A[0, 0] - A[2, 2]
        d2 = A[1, 1] - A[2, 2]
        offdiag = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
        diag = (d0**2 + d1**2 + d2**2) / 6.0
        return offdiag + diag

    def J3s(A):
        d0 = A[0, 0] - A[1, 1]
        d1 = A[0, 0] - A[2, 2]
        d2 = A[1, 1] - A[2, 2]
        t1 = d1 + d2
        t2 = d0 - d2
        t3 = -d0 - d1
        offdiag = 2.0 * A[0, 1] * A[1, 2] * A[0, 2]
        mixed = (A[0, 1] ** 2 * t1 + A[0, 2] ** 2 * t2 + A[1, 2] ** 2 * t3) / 3.0
        diag = (t1 * t2 * t3) / 27.0
        return offdiag + mixed - diag

    def dxs(A):
        d0 = A[0, 0] - A[1, 1]
        d1 = A[0, 0] - A[2, 2]
        d2 = A[1, 1] - A[2, 2]

        w = A[0, 1]
        v = A[0, 2]
        u = A[1, 2]

        alpha = d2
        beta = -d1
        gamma = d0

        return jnp.asarray(
            [
                3.0 * jnp.sqrt(3.0) * (v * w * alpha + u * (v * v - w * w)),
                alpha * beta * gamma + alpha * u * u + beta * v * v + gamma * w * w,
                2.0 * u * beta * gamma - v * w * (beta - gamma) + u * (2.0 * u * u - v * v - w * w),
                2.0
                * (v * alpha * gamma + u * w * (beta - gamma) + v * (v * v + w * w - 2.0 * u * u)),
                2.0
                * (w * alpha * beta + u * v * (beta - gamma) + w * (v * v + w * w - 2.0 * u * u)),
            ],
            dtype=A.dtype,
        )

    def discs(A):
        terms = dxs(A)
        return jnp.sum(terms * terms)

    def compute_eigvals(A):
        A = jnp.asarray(A)
        I1 = jnp.trace(A)
        j2 = J2s(A)
        j3 = J3s(A)
        discriminant = discs(A)
        normA = safe_norm(A)

        def branch_near_iso(_):
            eigvals = jnp.ones((3,), dtype=A.dtype) * I1 / 3.0
            return eigvals, eigvals

        def branch_general(_):
            phi = jnp.arctan2(safe_sqrt(27.0 * discriminant), 27.0 * j3)
            amplitude = 2.0 * safe_sqrt(3.0 * j2)
            shifts = 2.0 * jnp.pi * jnp.asarray([1.0, 2.0, 3.0], dtype=A.dtype)
            eigvals = (amplitude * jnp.cos((phi + shifts) / 3.0) + I1) / 3.0
            return eigvals, eigvals

        return lax.cond(j2 < rtol * normA, branch_near_iso, branch_general, operand=None)

    eigendyads, eigvals = jax.jacfwd(compute_eigvals, has_aux=True)(A / norm_A)
    order = jnp.argsort(eigvals)
    eigvals = norm_A * eigvals[order]
    eigendyads = eigendyads[order]
    eigendyads = 0.5 * (eigendyads + jnp.swapaxes(eigendyads, -1, -2))
    return eigvals, eigendyads


# ─────────────────────────────────────────────────────────────────────────────
# Private (implementation detail)
# ─────────────────────────────────────────────────────────────────────────────


def _sqrtm(C) -> tuple[jax.Array, jax.Array]:
    r"""
    Square root $\mathbf{U} = \mathbf{C}^{1/2}$ and inverse square root
    $\mathbf{U}^{-1}$ of a symmetric positive definite $3 \times 3$ matrix.

    Uses the closed-form expression due to Simo & Hughes (1998), p. 244,
    based on the principal invariants of $\mathbf{U}$.

    Parameters
    ----------
    C : array_like, shape (3, 3)
        Symmetric positive definite matrix (typically the right Cauchy-Green
        deformation tensor $\mathbf{C} = \mathbf{F}^{\mathsf{T}}\mathbf{F}$).

    Returns
    -------
    U : jax.Array, shape (3, 3)
        Matrix square root $\mathbf{C}^{1/2}$.
    U_inv : jax.Array, shape (3, 3)
        Inverse square root $\mathbf{C}^{-1/2}$.

    .. admonition:: References
        :class: seealso

        Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*.
        Springer. p. 244.
    """
    Id = jnp.eye(3)
    C2 = C @ C
    eigvals, _ = eig33(C)
    lamb = safe_sqrt(eigvals)
    i1 = jnp.sum(lamb)
    i2 = lamb[0] * lamb[1] + lamb[1] * lamb[2] + lamb[0] * lamb[2]
    i3 = jnp.prod(lamb)
    D = i1 * i2 - i3
    U = 1 / D * (-C2 + (i1**2 - i2) * C + i1 * i3 * Id)
    U_inv = 1 / i3 * (C - i1 * U + i2 * Id)
    return U, U_inv


def isotropic_function(fun, A) -> jax.Array:
    r"""
    Isotropic matrix function $f(\mathbf{A})$ of a symmetric $3 \times 3$ matrix.

    Evaluates the spectral decomposition
    $f(\mathbf{A}) = \sum_{i=1}^{3} f(\lambda_i)\,\mathbf{n}_i \otimes \mathbf{n}_i$
    where $\lambda_i$ are the eigenvalues and $\mathbf{n}_i$ the corresponding
    eigenvectors of $\mathbf{A}$.

    Parameters
    ----------
    fun : callable
        Scalar function $f : \mathbb{R} \to \mathbb{R}$ applied to each eigenvalue.
    A : array_like, shape (3, 3)
        Real symmetric matrix.

    Returns
    -------
    jax.Array
        Shape (3, 3).
    """
    eigvals, projectors = eig33(jnp.asarray(A))
    # The projectors P_i = ∂λ_i/∂A come from jacfwd, which perturbs each
    # element of A independently (without enforcing A_{ij} = A_{ji}).  This
    # means P_i may have a small antisymmetric component.  Symmetrising before
    # the spectral reconstruction removes that noise and ensures f(A) is
    # symmetric, consistent with the reference implementation.
    projectors = 0.5 * (projectors + jnp.swapaxes(projectors, -1, -2))
    return jnp.einsum("a,aij->ij", fun(eigvals), projectors)


def sqrtm(A) -> jax.Array:
    r"""
    Matrix square root $\mathbf{A}^{1/2}$ of a symmetric positive definite
    $3 \times 3$ matrix.

    Uses the closed-form expression of Simo & Hughes (1998) based on the
    principal invariants of $\mathbf{A}^{1/2}$.  Accepts any object that
    implements ``__jax_array__`` (e.g. a :class:`~jaxmat.tensors.SymmetricTensor2`).

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Symmetric positive definite matrix.

    Returns
    -------
    jax.Array
        Shape (3, 3).

    .. admonition:: References
        :class: seealso

        Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*.
        Springer. p. 244.
    """
    return _sqrtm(jnp.asarray(A))[0]


def inv_sqrtm(A) -> jax.Array:
    r"""
    Inverse square root $\mathbf{A}^{-1/2}$ of a symmetric positive definite
    $3 \times 3$ matrix.

    Computed jointly with :func:`sqrtm` via the same closed-form expression,
    so both are available at identical cost.

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Symmetric positive definite matrix.

    Returns
    -------
    jax.Array
        Shape (3, 3).

    .. admonition:: References
        :class: seealso

        Simo, J. C., & Hughes, T. J. R. (1998). *Computational Inelasticity*.
        Springer. p. 244.
    """
    return _sqrtm(jnp.asarray(A))[1]


def expm(A) -> jax.Array:
    r"""
    Matrix exponential $\exp(\mathbf{A})$ of a symmetric $3 \times 3$ matrix.

    Computed via the spectral decomposition; see :func:`isotropic_function`.
    Accepts any object that implements ``__jax_array__``.

    Parameters
    ----------
    A : array_like, shape (3, 3)

    Returns
    -------
    jax.Array
        Shape (3, 3).
    """
    return isotropic_function(jnp.exp, jnp.asarray(A))


def logm(A) -> jax.Array:
    r"""
    Matrix logarithm $\log(\mathbf{A})$ of a symmetric positive definite
    $3 \times 3$ matrix.

    Computed via the spectral decomposition; see :func:`isotropic_function`.
    Accepts any object that implements ``__jax_array__``.

    Parameters
    ----------
    A : array_like, shape (3, 3)
        Symmetric positive definite matrix.

    Returns
    -------
    jax.Array
        Shape (3, 3).
    """
    return isotropic_function(jnp.log, jnp.asarray(A))


def powm(A, m) -> jax.Array:
    r"""
    Matrix power $\mathbf{A}^m$ of a symmetric $3 \times 3$ matrix.

    Computed via the spectral decomposition; see :func:`isotropic_function`.
    Accepts any object that implements ``__jax_array__``.

    Parameters
    ----------
    A : array_like, shape (3, 3)
    m : float
        Exponent.

    Returns
    -------
    jax.Array
        Shape (3, 3).
    """
    return isotropic_function(lambda x: jnp.power(x, m), jnp.asarray(A))
