from abc import abstractmethod

import jax
import jax.numpy as jnp

from jaxmat.tensors.generic_tensors import (
    SymmetricTensor2,
    SymmetricTensor4,
)


class AbstractProjectedTensor4(SymmetricTensor4):
    r"""
    Base class for symmetry-reduced fourth-rank tensors.

    The tensor is represented as a linear combination of a small number of
    symmetry-projector basis tensors,

    $$\mathbb{C} = \sum_{i=1}^{n_{\text{basis}}} c_i \mathbb{B}_i$$

    where the coefficients $c_i$ are the minimal stored data. The basis
    $(\mathbb{B}_i)$ is stored as a stacked ``SymmetricTensor4``.

    Parameters
    ----------
    coeffs : array_like, shape ``(..., n_basis)``
        Expansion coefficients in the chosen symmetry basis.

    Notes
    -----
    Both ``array`` and ``tensor`` representations are constructed lazily from
    the coefficients. Subclasses must define ``_basis`` and implement
    ``project``.
    """

    _coeffs: jax.Array

    _basis: jax.Array

    def __init__(self, coeffs):
        self._coeffs = jnp.asarray(coeffs)
        self._array = self.array

    @property
    def coeffs(self):
        """
        Expansion coefficients in the symmetry basis.

        Returns
        -------
        jax.Array
            Shape ``(..., n_basis)``.
        """
        return self._coeffs

    @property
    def n_basis(self):
        """Number of basis elements."""
        return len(self._coeffs)

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
        """
        Project a fourth-rank tensor onto the symmetry class.

        Parameters
        ----------
        C : SymmetricTensor4 or Tensor4

        Returns
        -------
        AbstractProjectedTensor4
        """
        pass


def isotropic_projectors():
    """
    Construct isotropic fourth-rank projectors.

    Returns
    -------
    J, K : SymmetricTensor4
        Volumetric and deviatoric projectors forming an orthogonal
        decomposition of the identity.
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
    Construct cubic symmetry projectors.

    Returns
    -------
    J, Ka, Kb : SymmetricTensor4
        Projectors onto volumetric, diagonal deviatoric, and shear
        subspaces, respectively.
    """
    J, _ = isotropic_projectors()

    # Cubic invariant tensor
    Lambda = SymmetricTensor4(array=jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])))

    # Cubic decomposition of deviatoric part
    # Ka = Lambda - J = deviatoric projector of diagonal part
    Ka = Lambda - J

    # Remaining part is off-diagonal shear projector
    Kb = SymmetricTensor4.identity() - Lambda

    return J, Ka, Kb


def transverse_isotropic_projectors(axis):
    """
    Construct transverse isotropic (Walpole) projectors.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Symmetry axis (unit vector).

    Returns
    -------
    E1, E2, E3, E4, F, G : SymmetricTensor4
        Six basis tensors spanning the transverse isotropic subspace.
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
    r"""
    Isotropic fourth-rank tensor.

    Parameterized by bulk and shear moduli and expressed in the basis
    of volumetric ($\mathbb{J}$) and deviatoric ($\mathbb{K}$) projectors.

    Parameters
    ----------
    coeffs : array_like, optional
        Direct basis coefficients.
    kappa : float or array_like, optional
        Bulk modulus $\kappa$.
    mu : float or array_like, optional
        Shear modulus $\mu$.

    Notes
    -----
    Coefficients are internally stored as $\{3\kappa,2\mu\}$.
    """

    J, K = isotropic_projectors()
    _basis = SymmetricTensor4(array=jnp.stack([J.array, K.array], axis=0))

    def __init__(self, *, coeffs=None, kappa=None, mu=None):

        if coeffs is None:
            if (kappa is None) or (mu is None):
                raise ValueError("Provide either coeffs or (kappa, mu)")
            coeffs = jnp.stack([3.0 * kappa, 2.0 * mu], axis=-1)

        super().__init__(coeffs=coeffs)

    @classmethod
    def project(cls, C):
        """
        Project a tensor onto the isotropic symmetry class.

        Parameters
        ----------
        C : SymmetricTensor4

        Returns
        -------
        IsotropicTensor4
        """
        kappa = C.fourth_contract(cls.J) / 3.0
        mu = C.fourth_contract(cls.K) / 10.0
        return IsotropicTensor4(kappa=kappa, mu=mu)


class CubicTensor4(AbstractProjectedTensor4):
    r"""
    Cubic-symmetric fourth-rank tensor.

    Represented in the basis of three orthogonal projectors corresponding to
    volumetric, diagonal deviatoric, and shear parts.

    Parameters
    ----------
    coeffs : array_like, optional
        Direct basis coefficients.
    kappa : float or array_like, optional
        Cubic bulk modulus $\kappa$
    mua : float or array_like, optional
        Diagonal deviatoric modulus $\mu_a$
    mub : float or array_like, optional
        Shear modulus $\mu_b$

    Notes
    -----
    Coefficients are internally stored as $\{3\kappa,2\mu_a,2\mu_b\}$.
    """

    J, Ka, Kb = cubic_projectors()

    _basis = SymmetricTensor4(array=jnp.stack([J.array, Ka.array, Kb.array], axis=0))

    def __init__(self, *, coeffs=None, kappa=None, mua=None, mub=None):

        if coeffs is None:
            if None in (kappa, mua, mub):
                raise ValueError("Provide either coeffs or (c1, c2, c3)")
            coeffs = jnp.stack([3 * kappa, 2 * mua, 2 * mub], axis=-1)

        super().__init__(coeffs=coeffs)

    @classmethod
    def project(cls, C):
        """
        Project a tensor onto the cubic symmetry class.

        Parameters
        ----------
        C : SymmetricTensor4

        Returns
        -------
        CubicTensor4
        """
        kappa = C.fourth_contract(cls.J) / 3.0
        mua = C.fourth_contract(cls.Ka) / 4.0
        mub = C.fourth_contract(cls.Kb) / 6.0
        return CubicTensor4(kappa=kappa, mua=mua, mub=mub)


class TransverseIsotropicTensor4(AbstractProjectedTensor4):
    """
    Transversely isotropic fourth-rank tensor.

    Defined with respect to a symmetry axis and expanded in the six Walpole
    basis tensors.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Symmetry axis.
    coeffs : array_like, shape (6,)
        Basis coefficients.

    Notes
    -----
    The inverse is computed analytically in the Walpole basis.
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
        """
        Inverse operator within the transverse isotropic subspace.

        Returns
        -------
        TransverseIsotropicTensor4
        """
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
        """
        Project a tensor onto the transverse isotropic symmetry class.

        Parameters
        ----------
        axis : array_like, shape (3,)
            Symmetry axis.
        C : SymmetricTensor4

        Returns
        -------
        TransverseIsotropicTensor4
        """
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
