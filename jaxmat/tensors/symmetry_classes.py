from abc import abstractmethod
import jax
import jax.numpy as jnp
from jaxmat.tensors.generic_tensors import (
    SymmetricTensor4,
    SymmetricTensor2,
)
from jaxmat.tensors.mappings import kelvin_rank4_map


class AbstractProjectedTensor4(SymmetricTensor4):
    """
    Base class for symmetry-reduced 4th-rank tensors.
    """

    _coeffs: jax.Array  # (...,2) → [a_J, a_K]

    _kelvin_basis: jax.Array
    _tensor_basis: jax.Array

    def __init__(self, coeffs):
        self._coeffs = jnp.asarray(coeffs)
        self._array = self.array

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def array(self):
        return jnp.tensordot(self._coeffs, self._kelvin_basis, axes=1)

    @property
    def tensor(self):
        return jnp.tensordot(self._coeffs, self._tensor_basis, axes=1)

    @property
    def inv(self):
        return type(self)(coeffs=1.0 / self._coeffs)

    @classmethod
    @abstractmethod
    def project(cls, C):
        pass


def isotropic_projectors():
    """Isotropic projectors J and K in (6, 6) Kelvin-Mandel format."""
    id = SymmetricTensor2.identity()
    Id = SymmetricTensor4.identity()

    Ibar = jnp.einsum("ij,kl", id, id)
    J4 = SymmetricTensor4(tensor=Ibar / 3.0)
    K4 = Id - J4

    return J4, K4


def cubic_projectors():
    """
    Cubic symmetry projectors in 6x6 Kelvin-Mandel format.

    Returns
    -------
    J, Ka, Kb : jax.Array
        6x6 Kelvin-Mandel projectors
    J4, Ka4, Kb4 : jax.Array
        4th-rank tensor projectors (d,d,d,d)
    """
    J, _ = isotropic_projectors()

    # 2. Cubic invariant tensor
    d = 3
    Lambda = jnp.zeros((d, d, d, d))
    for i in range(d):
        Lambda = Lambda.at[i, i, i, i].set(1.0)
    Lambda = SymmetricTensor4(tensor=Lambda)

    # 3. Cubic decomposition of deviatoric part
    # Ka = Lambda - J = deviatoric projector of diagonal part
    Ka = Lambda - J

    # Remaining part is off-diagonal shear projector
    Kb = SymmetricTensor4.identity() - Lambda

    return J, Ka, Kb


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

    @classmethod
    def project(cls, C):
        """Projects a 4th rank tensor onto the isotropic symmetry class."""
        kappa = C.fourth_contract(cls.J) / 3.0
        mu = C.fourth_contract(cls.K) / 10.0
        return IsotropicTensor4(kappa=kappa, mu=mu)


class CubicTensor4(AbstractProjectedTensor4):
    """
    Cubic-symmetric 4th-rank tensor.
    """

    J, Ka, Kb = cubic_projectors()

    _kelvin_basis = jnp.stack([J.array, Ka.array, Kb.array], axis=0)
    _tensor_basis = jnp.stack([J.tensor, Ka.tensor, Kb.tensor], axis=0)

    def __init__(self, *, coeffs=None, kappa=None, mua=None, mub=None):

        if coeffs is None:
            if None in (kappa, mua, mub):
                raise ValueError("Provide either coeffs or (c1, c2, c3)")
            coeffs = jnp.stack([3 * kappa, 2 * mua, 2 * mub], axis=-1)

        super().__init__(coeffs=coeffs)

    @classmethod
    def project(cls, C):
        """Projects a 4th rank tensor onto the cubic symmetry class."""
        kappa = C.fourth_contract(cls.J) / 3.0
        mua = C.fourth_contract(cls.Ka) / 4.0
        mub = C.fourth_contract(cls.Kb) / 6.0
        return CubicTensor4(kappa=kappa, mua=mua, mub=mub)
