import jax.numpy as jnp
import optax


def safe_fun(fun, x, norm=None, eps=1e-16):
    """
    Safely applies a function to an input, avoiding numerical issues near zero.

    This function applies ``fun(x)`` only when the norm of ``x`` exceeds a small
    tolerance ``eps``. Otherwise, it returns zero. This is useful for ensuring
    numerical stability in cases where evaluating ``fun`` at or near zero could
    result in undefined or unstable behavior (e.g., division by zero).

    Parameters
    ----------
    fun : Callable
        The function to apply safely.
    x : array-like
        Input array or tensor.
    norm : Callable, optional
        A norm or magnitude function used to test whether ``x`` is sufficiently
        large. Defaults to the identity function.
    eps : float, optional
        Small threshold to determine whether ``x`` is treated as nonzero.
        Defaults to 1e-16.

    Returns
    -------
    array-like
        ``fun(x)`` if ``norm(x) > eps``, otherwise ``0`` (of the same shape as ``x``).
    """
    if norm is None:

        def norm(x):
            return x

    nonzero_x = jnp.where(norm(x) > eps, x, 0 * x)
    return jnp.where(norm(x) > eps, fun(nonzero_x), 0)


def safe_sqrt(x, eps=1e-16):
    """
    Computes a numerically safe square root.

    Ensures the argument to the square root is greater than `eps`
    to avoid taking the square root of zero or negative values,
    which could cause instability or NaNs.

    Parameters
    ----------
    x : array-like
        Input array or tensor.
    eps : float, optional
        Minimum threshold for `x` before taking the square root. Defaults to 1e-16.

    Returns
    --------
    array-like
        The square root of `x` for `x > eps`, otherwise 0.
    """
    nonzero_x = jnp.where(x > eps, x, 0.0)
    return jnp.where(x > eps, jnp.sqrt(nonzero_x), 0.0)


def safe_norm(x, eps=1e-16, **kwargs):
    """
    Wrapper around ``optax.safe_norm`` that computes a numerically stable norm.

    This function prevents numerical instability when computing vector norms
    for small magnitudes by internally applying a stability threshold.

    Parameters
    ----------
    x : array-like
        Input vector or tensor.
    eps : float, optional
        Small constant added for numerical stability. Defaults to ``1e-16``.
    **kwargs:
        Additional arguments passed to ``optax.safe_norm``.

    Returns
    -------
    array-like
        The numerically stable norm of ``x``.
    """
    return optax.safe_norm(x, eps, **kwargs)


def FischerBurmeister(x, y):
    r"""
    Computes the scalar Fischer-Burmeister function.

    The Fischer-Burmeister function is defined as:
    $$\Phi(x, y) = x + y - \sqrt{x^2 + y^2}$$

    and is commonly used in complementarity problem formulations to provide
    a semi-smooth reformulation of the complementarity conditions
    $$x \geq 0, y \geq 0, xy = 0$$.
    """
    return x + y - safe_sqrt(x**2 + y**2)


def rotation_matrix_direct(theta, axis):
    """
    Calculate the rotation matrix for rotating around an arbitrary axis by angle theta.

    """
    x, y, z = axis[0], axis[1], axis[2]

    c = jnp.cos(theta)
    s = jnp.sin(theta)
    C = 1 - c

    # Compute the matrix elements directly
    R = jnp.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )

    return R


def euler_to_rotation(phi1, Phi, phi2):
    """Euler's angle → rotation matrix (Bunge convention  ZXZ)"""
    c1, s1 = jnp.cos(phi1), jnp.sin(phi1)
    cP, sP = jnp.cos(Phi), jnp.sin(Phi)
    c2, s2 = jnp.cos(phi2), jnp.sin(phi2)

    R = jnp.array(
        [
            [c1 * c2 - s1 * s2 * cP, -c1 * s2 - s1 * c2 * cP, s1 * sP],
            [s1 * c2 + c1 * s2 * cP, -s1 * s2 + c1 * c2 * cP, -c1 * sP],
            [s2 * sP, c2 * sP, cP],
        ]
    )
    return R
