import jax.numpy as jnp
from jaxmat.tensors.generic_tensors import (
    SymmetricTensor4,
    SymmetricTensor2,
    AbstractProjectedTensor4,
)
from jaxmat.tensors.mappings import kelvin_rank4_map


def isotropic_projectors():
    """Isotropic projectors J and K in (6, 6) Kelvin-Mandel format."""
    id = SymmetricTensor2.identity()
    Id = SymmetricTensor4.identity()

    Ibar = jnp.einsum("ij,kl", id, id)
    J4 = SymmetricTensor4(tensor=Ibar / 3.0)
    K4 = Id - J4

    return J4, K4


def cubic_projectors(d=3):
    """
    Cubic symmetry projectors in 6x6 Kelvin-Mandel format.

    Returns
    -------
    J, Ka, Kb : jax.Array
        6x6 Kelvin-Mandel projectors
    J4, Ka4, Kb4 : jax.Array
        4th-rank tensor projectors (d,d,d,d)
    """
    I = jnp.eye(d)
    I4 = jnp.einsum("ij,kl->ijkl", I, I)
    I4s = 0.5 * (jnp.einsum("ik,jl->ijkl", I, I) + jnp.einsum("il,jk->ijkl", I, I))

    # 1. Spherical projector
    J4 = (1.0 / d) * I4

    # 2. Cubic invariant tensor
    Lambda = jnp.zeros((d, d, d, d))
    for i in range(d):
        Lambda = Lambda.at[i, i, i, i].set(1.0)

    # 3. Cubic decomposition of deviatoric part
    # Ka = Lambda - J = deviatoric projector of diagonal part
    Ka4 = Lambda - J4

    # Remaining part is off-diagonal shear projector
    Kb4 = I4s - Lambda

    # Kelvin mapping
    (i, j, k, l), W = kelvin_rank4_map(d)

    J = W * J4[i, j, k, l]
    Ka = W * Ka4[i, j, k, l]
    Kb = W * Kb4[i, j, k, l]

    return J, Ka, Kb, J4, Ka4, Kb4


class IsotropicTensor4(AbstractProjectedTensor4):
    """
    Symmetric 4th-rank isotropic tensor with compressed storage.
    """

    # ---- static projectors ----
    J, K = isotropic_projectors()

    _kelvin_basis = jnp.stack([J.array, K.array], axis=0)
    _tensor_basis = jnp.stack([J.tensor, K.tensor], axis=0)

    # ----------------------------

    def __init__(self, *, coeffs=None, kappa=None, mu=None):

        if coeffs is None:
            if (kappa is None) or (mu is None):
                raise ValueError("Provide either coeffs or (kappa, mu)")
            coeffs = jnp.stack([3.0 * kappa, 2.0 * mu], axis=-1)

        super().__init__(coeffs=coeffs)


class CubicTensor4(AbstractProjectedTensor4):
    """
    Cubic-symmetric 4th-rank tensor.
    """

    Jk, Kak, Kbk, J4, Ka4, Kb4 = cubic_projectors()

    _kelvin_basis = jnp.stack([Jk, Kak, Kbk], axis=0)
    _tensor_basis = jnp.stack([J4, Ka4, Kb4], axis=0)

    def __init__(self, *, coeffs=None, kappa=None, mua=None, mub=None):

        if coeffs is None:
            if None in (kappa, mua, mub):
                raise ValueError("Provide either coeffs or (c1, c2, c3)")
            coeffs = jnp.stack([3 * kappa, 2 * mua, 2 * mub], axis=-1)

        super().__init__(coeffs=coeffs)
