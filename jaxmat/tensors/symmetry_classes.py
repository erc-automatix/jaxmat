from abc import abstractmethod
import jax
import jax.numpy as jnp
from jaxmat.tensors.generic_tensors import (
    SymmetricTensor4,
    SymmetricTensor2,
)


class AbstractProjectedTensor4(SymmetricTensor4):
    """
    Base class for symmetry-reduced 4th-rank tensors.
    """

    _coeffs: jax.Array  # (...,2) → [a_J, a_K]

    _basis: jax.Array

    def __init__(self, coeffs):
        self._coeffs = jnp.asarray(coeffs)
        self._array = self.array

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def array(self):
        return jnp.tensordot(self._coeffs, self._basis.array, axes=1)

    @property
    def tensor(self):
        return jnp.tensordot(self._coeffs, self._basis.tensor, axes=1)

    @property
    def inv(self):
        return type(self)(coeffs=1.0 / self._coeffs)

    @classmethod
    @abstractmethod
    def project(cls, C):
        pass


def isotropic_projectors():
    """Isotropic symmetry projectors.

    Returns
    -------
    J, K : SymmetricTensor4
        The two isotropic projectors
    """
    id = SymmetricTensor2.identity()
    Id = SymmetricTensor4.identity()

    # Ibar = jnp.einsum("ij,kl", id, id)
    Ibar = jnp.outer(id.array, id.array)
    J = SymmetricTensor4(array=Ibar / 3.0)
    K = Id - J
    return J, K


def cubic_projectors():
    """
    Cubic symmetry projectors.

    Returns
    -------
    J, Ka, Kb : SymmetricTensor4
        The three cubic projectors
    """
    J, _ = isotropic_projectors()

    # 2. Cubic invariant tensor
    Lambda = SymmetricTensor4(array=jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])))

    # 3. Cubic decomposition of deviatoric part
    # Ka = Lambda - J = deviatoric projector of diagonal part
    Ka = Lambda - J

    # Remaining part is off-diagonal shear projector
    Kb = SymmetricTensor4.identity() - Lambda

    return J, Ka, Kb


def transverse_isotropic_projectors(axis):
    """
    Transverse isotropic symmetry projectors around `axis`.

    Returns
    -------
    E1, E2, E3, E4, F, G : SymmetricTensor4
        The 6 Walpole basis tensors
    """
    P = SymmetricTensor2(tensor=jnp.outer(axis, axis))
    Q = SymmetricTensor2(tensor=(jnp.eye(3) - P) / jnp.sqrt(2.0))
    E1 = SymmetricTensor4(array=jnp.outer(P.array, P.array))
    E2 = SymmetricTensor4(array=jnp.outer(Q.array, Q.array))
    E3 = SymmetricTensor4(array=jnp.outer(P.array, Q.array))
    E4 = SymmetricTensor4(array=jnp.outer(Q.array, P.array))

    def sym_cross_prod(A, B):
        return 0.5 * (jnp.einsum("ik,jl->ijkl", A, B) + jnp.einsum("il,jk->ijkl", A, B))

    F = SymmetricTensor4(tensor=2 * sym_cross_prod(Q, Q)) - E2
    G = SymmetricTensor4(
        tensor=jnp.sqrt(2) * (sym_cross_prod(P, Q) + sym_cross_prod(Q, P))
    )
    return E1, E2, E3, E4, F, G


class IsotropicTensor4(AbstractProjectedTensor4):
    """
    Symmetric 4th-rank isotropic tensor with compressed storage.
    """

    # ---- static projectors ----
    J, K = isotropic_projectors()
    _basis = SymmetricTensor4(array=jnp.stack([J.array, K.array], axis=0))

    # _kelvin_basis = jnp.stack([J.array, K.array], axis=0)
    # _tensor_basis = jnp.stack([J.tensor, K.tensor], axis=0)

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

    _basis = SymmetricTensor4(array=jnp.stack([J.array, Ka.array, Kb.array], axis=0))
    # _kelvin_basis = jnp.stack([J.array, Ka.array, Kb.array], axis=0)
    # _tensor_basis = jnp.stack([J.tensor, Ka.tensor, Kb.tensor], axis=0)

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


class TransverseIsotropicTensor4(AbstractProjectedTensor4):
    """
    Transverse isotropic-symmetric 4th-rank tensor about `axis`.
    """

    axis: jax.Array
    E1: jax.Array
    E2: jax.Array
    E3: jax.Array
    E4: jax.Array
    F: jax.Array
    G: jax.Array

    def __init__(self, axis, coeffs):
        self.axis = axis
        basis = transverse_isotropic_projectors(self.axis)
        self.E1, self.E2, self.E3, self.E4, self.F, self.G = basis
        self._basis = SymmetricTensor4(
            array=jnp.stack([b.array for b in basis], axis=0)
        )
        super().__init__(coeffs=coeffs)

    @property
    def inv(self):
        a = self.coeffs[:4]
        A = jnp.asarray([[a[0], a[2]], [a[3], a[1]]])
        invA = jnp.linalg.inv(A)
        inv_coeffs = 1 / self.coeffs
        inv_coeffs = inv_coeffs.at[:4].set(
            jnp.asarray([invA[0, 0], invA[1, 1], invA[0, 1], invA[1, 0]])
        )
        return type(self)(self.axis, coeffs=inv_coeffs)

    @classmethod
    def project(cls, axis, C):
        """Projects a 4th rank tensor onto the transverse isotropy symmetry class for a given axis."""
        E1, E2, E3, E4, F, G = transverse_isotropic_projectors(axis)
        c1 = C.fourth_contract(E1)
        c2 = C.fourth_contract(E2)
        c3 = C.fourth_contract(E3)
        c4 = C.fourth_contract(E4)
        c5 = C.fourth_contract(F) / 2.0
        c6 = C.fourth_contract(G) / 2.0
        return TransverseIsotropicTensor4(
            axis, coeffs=jnp.asarray([c1, c2, c3, c4, c5, c6])
        )
