"""
This module implements various linear algebra operations on matrices.
Other linear algebra utilities working on jaxmat tensors are available in tensor_utils.py
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from .utils import safe_norm, safe_sqrt


def dim(A):
    r"""Dimension ``dim`` of a n-rank matrix $\bA$, assuming ``shape=(dim, dim, ..., dim)``."""
    return A.shape[0]


def tr(A):
    r"""
    Trace of a matrix $\bA$.
    $$\tr(\bA)=A_{ii}$$
    """
    return jnp.trace(A)


def dev(A):
    r"""
    Deviatoric part of a $d\times d$ matrix $\bA$.
    $$\dev(\bA) = \bA - \dfrac{1}{d}\bI$$
    """
    d = dim(A)
    Id = jnp.eye(d)
    return A - tr(A) / d * Id


def det33(A):
    r"""Determinant $\det(\bA)$ of a 3x3 matrix $\bA$, computed using explicit formula."""
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]
    return (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )


def inv33(A):
    r"""Inverse $\bA^{-1}$ of a 3x3 matrix $\bA$, explicitly computed using cofactor formula."""
    # Minors and cofactors
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]

    # Cofactor matrix (transposed for adjugate directly)
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

    invA = cof / det
    return invA


def principal_invariants(A):
    r"""Principal invariants of a 3x3 matrix $\bA$.
    $$\begin{align*}
    I_1 &= \tr(\bA)\\
    I_2 &= \frac{1}{2}(\tr(\bA)^2-\tr(\bA^2))\\
    I_3 &= \det(\bA)
    \end{align*}$$
    """
    i1 = jnp.trace(A)
    i2 = (jnp.trace(A) ** 2 - jnp.trace(A @ A)) / 2
    i3 = det33(A)
    return i1, i2, i3


def main_invariants(A):
    r"""Main invariants of a 3x3 matrix $\bA$:
    $$\tr(\bA),\: \tr(\bA^2),\: \tr(\bA^3)$$.
    """
    j1 = jnp.trace(A)
    j2 = jnp.trace(A @ A)
    j3 = jnp.trace(A @ A @ A)
    return j1, j2, j3


def pq_invariants(sig):
    r"""Hydrostatic/deviatoric equivalent stresses $(p,q)$. Typically used in soil mechanics.

    $$p = - \tr(\bsig)/3 = -I_1/3$$
    $$q = \sqrt{\frac{3}{2}\bs:\bs} = \sqrt{3 J_2}$$
    """
    p = -jnp.trace(sig) / 3
    s = dev(sig)
    q = safe_sqrt(3.0 / 2.0 * jnp.vdot(s, s))
    return p, q


# @partial(jax.jit, static_argnums=1)
def eig33(A, rtol=1e-16):
    r"""
    Computes the eigenvalues and eigenvalue derivatives of a 3 x 3 real symmetric matrix.

    This function implements a numerically stable eigendecomposition for 3 x 3 symmetric
    matrices based on the method by Harari & Albocher (2023)

    The implementation avoids catastrophic cancellation and loss of precision in
    cases where two or more eigenvalues are nearly equal.

    Parameters
    ----------
    A : array_like of shape (3, 3)
        Real symmetric matrix whose eigenvalues (and optionally eigenvalue dyads)
        are to be computed.
    rtol : float, optional
        Relative tolerance used to determine near-isotropic or nearly repeated
        eigenvalue cases. Default is `1e-16`.

    Returns
    -------
    eigvals : jax.Array of shape (3,)
        Eigenvalues of ``A\`, ordered in a consistent but unspecified order.
    eigendyads : jax.Array of shape (3, 3, 3)
        Derivatives of the eigenvalues with respect to the components of ``A\`,
        obtained via forward-mode automatic differentiation (`jax.jacfwd`).


    Notes
    -----
    - The method distinguishes three cases:
        1. Near-isotropic case (``s < rtol * ||A||``): all eigenvalues are nearly equal.
        2. Two nearly equal eigenvalues: handled by a special branch to ensure stability.
        3. Three distinct eigenvalues: computed via trigonometric relations.
    - The implementation uses ``safe_norm`` and ``safe_sqrt`` for numerical safety.
    - Input ``A`` must be symmetric; asymmetry may lead to inaccurate results.


    .. admonition:: References
        :class: seealso

        Harari, I., & Albocher, U. (2023). Computation of eigenvalues of a real,
        symmetric 3 x 3 matrix with particular reference to the pernicious case of
        two nearly equal eigenvalues. *International Journal for Numerical Methods in
        Engineering*, 124(5), 1089-1110.
    """

    def compute_eigvals_HarariAlbocher(A):
        """
        Eigendecomposition of 3x3 symmetric matrix based on Harari, I., & Albocher, U. (2023).
        """
        A = jnp.asarray(A)
        norm = safe_norm(A)
        Id = jnp.eye(dim(A))
        I1 = jnp.trace(A)
        S = dev(A)
        J2 = tr(S.T @ S) / 2
        s = safe_sqrt(J2 / 3)

        def branch_near_iso(_):
            eigvals = jnp.ones((3,)) * I1 / 3
            return eigvals, eigvals

        def branch_general(_):
            T = S @ S - 2 * J2 / 3 * Id
            d = safe_norm(T - s * S) / safe_norm(T + s * S)
            sj = jnp.sign(1 - d)
            cond = sj * (1 - d) < rtol * norm

            def branch_two_eigvals(_):
                lamb_max = jnp.sqrt(3) * s
                eigvals_dev = jnp.array([lamb_max, 0.0, -lamb_max])
                eigvals = eigvals_dev + I1 / 3
                return eigvals, eigvals

            def branch_three_eigvals(_):
                alpha = 2 / 3 * jnp.arctan(d**sj)
                lambda_d = 2 * sj * s * jnp.cos(alpha)
                sd = jnp.sqrt(3) * s * jnp.sin(alpha)

                eigvals_dev = lax.cond(
                    lambda_d > 0,
                    lambda _: jnp.array([-lambda_d / 2 - sd, -lambda_d / 2 + sd, lambda_d]),
                    lambda _: jnp.array([lambda_d, -lambda_d / 2 - sd, -lambda_d / 2 + sd]),
                    operand=None,
                )
                eigvals = eigvals_dev + I1 / 3
                return eigvals, eigvals

            return lax.cond(cond, branch_two_eigvals, branch_three_eigvals, operand=None)

        return lax.cond(s < rtol * norm, branch_near_iso, branch_general, operand=None)

    eigendyads, eigvals = jax.jacfwd(compute_eigvals_HarariAlbocher, has_aux=True)(A)

    return eigvals, eigendyads


def _sqrtm(C):
    r"""
    Unified expression for sqrt and inverse sqrt of a symmetric matrix $\bC$
    Simo, J. C., & Hughes, T. J. (1998). Computational inelasticity., p.244
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


def sqrtm(A):
    r"""
    Matrix square-root $\bA^{1/2}$ of a symmetric 3x3 matrix $\bA$.
    Computed using the unified square root and inverse square root
    formula, see Simo & Hughes, 1998.

    .. admonition:: References
        :class: seealso

        Simo, J. C., & Hughes, T. J. (1998). Computational inelasticity., p.244
    """
    return _sqrtm(A)[0]


def inv_sqrtm(A):
    r"""
    Matrix inverse square-root $\bA^{-1/2}$ of a symmetric 3x3 matrix $\bA$.

    Computed using the unified square root and inverse square root
    formula, see Simo & Hughes, 1998.

    .. admonition:: References
        :class: seealso

        Simo, J. C., & Hughes, T. J. (1998). Computational inelasticity., p.244
    """
    return _sqrtm(A)[1]


def isotropic_function(fun, A):
    r"""Computes an isotropic function of a symmetric 3x3 matrix $\bA$.

    Parameters
    ----------
    fun : callable
        A scalar function $f(x)$
    A : jax.Array
        A symmetric 3x3 matrix

    Returns
    -------
    jax.Array
        A new 3x3 matrix $f_{\bA}$ such that
        $$f_{\bA} = \sum_{i=1}^3 f(\lambda_i) \bn_i \otimes \bn_i$$
        where $\lambda_i$ and $\bn_i$ are the eigenvalues and eigenvectors of $\bA$.
    """
    eigvals, eigendyads = eig33(A)
    f = fun(eigvals)
    return sum([fi * Ni for fi, Ni in zip(f, eigendyads)])


def expm(A):
    r"""Matrix exponential $\exp(\bA)$ of a symmetric 3x3 matrix $\bA$."""
    return isotropic_function(jnp.exp, A)


def logm(A):
    r"""Matrix logarithm $\log(\bA)$ of a symmetric 3x3 matrix $\bA$."""
    return isotropic_function(jnp.log, A)


def powm(A, m):
    r"""Matrix power $\bA^m$ of exponent $m$ of a symmetric 3x3 matrix $\bA$."""
    return isotropic_function(lambda x: jnp.power(x, m), A)
