import jax.numpy as jnp
import optax


def safe_fun(fun, x, norm=None, eps=1e-16):
    r"""
    Apply a function safely, avoiding evaluation at or near zero.

    The input ``x`` is replaced by a small positive sentinel ``eps`` whenever
    ``norm(x) <= eps`` before calling ``fun``.  The final result is then
    masked to return zero in that case.  This sentinel strategy ensures that
    ``fun`` is always evaluated at a numerically safe point, so that gradients
    through ``fun`` remain finite under automatic differentiation.

    This is consistent with :func:`safe_sqrt`:
    ``safe_fun(jnp.sqrt, x)`` produces the same values and gradients as
    ``safe_sqrt(x)``.

    Parameters
    ----------
    fun : callable
        Scalar or array function to apply safely.
    x : array_like
        Input value.
    norm : callable, optional
        Scalar-valued function of ``x`` used to test proximity to zero.
        Defaults to the identity (i.e. ``x`` itself is the magnitude).
    eps : float, optional
        Threshold below which ``x`` is considered zero.  Defaults to ``1e-16``.

    Returns
    -------
    jax.Array
        ``fun(x)`` where ``norm(x) > eps``, otherwise ``0``.

    Notes
    -----
    The key property is that the *sentinel-substituted* input ``eps`` (not
    ``0``) is passed to ``fun`` in the masked branch.  This prevents
    ``jax.grad`` from encountering undefined derivatives (e.g.
    ``1 / (2 sqrt(0))`` for ``fun = jnp.sqrt``).
    """
    if norm is None:

        def norm(x):
            return x

    is_nonzero = norm(x) > eps
    # Use eps as sentinel (not 0) so fun is always evaluated at a safe point.
    safe_x = jnp.where(is_nonzero, x, eps)
    return jnp.where(is_nonzero, fun(safe_x), 0 * fun(safe_x))


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
        The square root of `x` for `x > eps`, otherwise `eps`.
    """
    nonzero_x = jnp.where(x > eps, x, eps)
    return jnp.where(x > eps, jnp.sqrt(nonzero_x), eps)


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
