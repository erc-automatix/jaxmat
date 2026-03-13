import jax
import jax.numpy as jnp

from jaxmat.tensors import safe_norm


def from_axis_angle(axis, theta):
    """
    Construct rotation matrix/matrices from an axis-angle representation.

    Uses Rodrigues' rotation formula:

    .. math::
        R = I + \\sin\\theta \\, [\\hat{a}]_\\times
            + (1-\\cos\\theta) [\\hat{a}]_\\times^2

    where :math:`\\hat{a}` is the unit rotation axis.

    Parameters
    ----------
    axis : array_like, shape (..., 3)
        Rotation axis vectors. They do not need to be normalized.
        The last dimension must be 3.
    theta : array_like, shape (...)
        Rotation angle(s) in radians. Must be broadcast-compatible
        with ``axis[..., 0]``.

    Returns
    -------
    R : jax.Array, shape (..., 3, 3)
        Rotation matrix/matrices in SO(3).
    """
    axis = axis / safe_norm(axis, axis=-1, keepdims=True)

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    C = 1.0 - c

    R = jnp.stack(
        [
            c + x * x * C,
            x * y * C - z * s,
            x * z * C + y * s,
            y * x * C + z * s,
            c + y * y * C,
            y * z * C - x * s,
            z * x * C - y * s,
            z * y * C + x * s,
            c + z * z * C,
        ],
        axis=-1,
    )

    return R.reshape(axis.shape[:-1] + (3, 3))


# ---------------------------------------------------------


def from_quaternion(q):
    """
    Construct rotation matrix/matrices from quaternion(s).

    The quaternion is assumed to follow the scalar-first convention:

    .. math::
        q = (w, x, y, z)

    Parameters
    ----------
    q : array_like, shape (..., 4)
        Quaternion(s) in scalar-first format ``(w, x, y, z)``.
        The quaternion(s) need not be normalized.

    Returns
    -------
    R : jax.Array, shape (..., 3, 3)
        Rotation matrix/matrices in SO(3).
    """
    q = q / safe_norm(q, axis=-1, keepdims=True)

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = jnp.stack(
        [
            jnp.stack(
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
                axis=-1,
            ),
        ],
        axis=-2,
    )

    return R


def from_euler_bunge(phi1, Phi, phi2):
    """
    Construct rotation matrix/matrices from Bunge Euler angles (ZXZ convention).

    The rotation is defined as:

    .. math::
        R = R_z(\\phi_1) R_x(\\Phi) R_z(\\phi_2)

    Parameters
    ----------
    phi1 : array_like, shape (...)
        First rotation about the z-axis (radians).
    Phi : array_like, shape (...)
        Second rotation about the x-axis (radians).
    phi2 : array_like, shape (...)
        Third rotation about the z-axis (radians).

    Returns
    -------
    R : jax.Array, shape (..., 3, 3)
        Rotation matrix/matrices in SO(3).
    """
    c1, s1 = jnp.cos(phi1), jnp.sin(phi1)
    c, s = jnp.cos(Phi), jnp.sin(Phi)
    c2, s2 = jnp.cos(phi2), jnp.sin(phi2)

    R = jnp.stack(
        [
            c1 * c2 - s1 * s2 * c,
            s1 * c2 + c1 * s2 * c,
            s2 * s,
            -c1 * s2 - s1 * c2 * c,
            -s1 * s2 + c1 * c2 * c,
            c2 * s,
            s1 * s,
            -c1 * s,
            c,
        ],
        axis=-1,
    )

    return R.reshape(jnp.shape(phi1) + (3, 3))


def random(key, shape=()):
    """
    Uniform random rotation(s) in SO(3).

    Uses the Shoemake quaternion sampling method to generate
    rotations uniformly distributed with respect to the Haar
    measure on SO(3).

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key.
    shape : tuple of int, optional
        Batch shape.

    Returns
    -------
    R : jax.Array, shape (*shape, 3, 3)
        Random rotation matrix/matrices.
    """
    u = jax.random.uniform(key, shape + (3,))
    u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]

    q = jnp.stack(
        [
            jnp.sqrt(1 - u1) * jnp.cos(2 * jnp.pi * u2),
            jnp.sqrt(1 - u1) * jnp.sin(2 * jnp.pi * u2),
            jnp.sqrt(u1) * jnp.cos(2 * jnp.pi * u3),
            jnp.sqrt(u1) * jnp.sin(2 * jnp.pi * u3),
        ],
        axis=-1,
    )

    return from_quaternion(q)
