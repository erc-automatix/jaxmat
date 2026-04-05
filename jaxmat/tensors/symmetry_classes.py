"""
jaxmat/tensors/symmetry_classes.py

Symmetry-reduced fourth-rank tensor classes for solid mechanics.

Each class stores only its **coefficient vector** (the unique JAX leaf).
No materialised (6,6) Kelvin matrix is stored as a leaf.

All shared behaviour lives in :class:`AbstractStructuredTensor4`:

- ``array`` property via tensordot with the precomputed basis stack
- :meth:`to_symmetric`, :meth:`fourth_contract`, :meth:`rotate` (materialise fallback)
- All scalar arithmetic via :meth:`_rebuild`
- Binary ``+`` / ``-``: same-type stays same-type, cross-type materialises
- ``@`` default: ``(6,6) @ (6,)`` on the materialised Kelvin matrix
- ``__rmatmul__`` fallback

Each subclass provides only:

- ``__init__``  (parameter parsing and ``_basis_arrays`` assignment)
- :meth:`_rebuild` — factory for a new same-type instance with updated coefficients
- :attr:`inv` — class-specific inversion
- ``__matmul__`` override for same-class composition (coefficient-space algebra)
- :meth:`project` classmethod

Projector algebra
-----------------
Every $G$-invariant fourth-rank tensor decomposes as
$\\mathbb{C} = \\sum_\\alpha c_\\alpha \\mathbb{P}_\\alpha$ where the
$\\mathbb{P}_\\alpha$ are fixed orthogonal (or block-orthogonal) projectors
determined by the symmetry group $G$.

======================  ========  ==========  ===========================
Symmetry class          Coeffs    Projectors  Composition rule
======================  ========  ==========  ===========================
Isotropic               2         J, K        elementwise (orthogonal)
Cubic                   3         J, Kₐ, K_b  elementwise (orthogonal)
Transverse isotropic    6         Walpole     2×2 block + scalar pairs
======================  ========  ==========  ===========================
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.tensors.generic_tensors import (
    SymmetricTensor2,
    SymmetricTensor4,
    _array4,
)


# ─────────────────────────────────────────────────────────────────────────────
# Projector constructors
# ─────────────────────────────────────────────────────────────────────────────


def _isotropic_projectors():
    r"""
    Construct isotropic fourth-rank projectors.

    Returns
    -------
    J : :class:`SymmetricTensor4`
        Volumetric projector $\mathbb{J}_{ijkl} = \frac{1}{3}\delta_{ij}\delta_{kl}$.
    K : :class:`SymmetricTensor4`
        Deviatoric projector $\mathbb{K} = \mathbb{I}^s - \mathbb{J}$.

    Notes
    -----
    The projectors satisfy $\mathbb{J}:\mathbb{J}=\mathbb{J}$,
    $\mathbb{K}:\mathbb{K}=\mathbb{K}$, $\mathbb{J}:\mathbb{K}=0$, and
    $\mathbb{J}+\mathbb{K}=\mathbb{I}^s$.
    """
    Id = SymmetricTensor2.identity()
    Id4 = SymmetricTensor4.identity()
    J = SymmetricTensor4(array=jnp.outer(Id.array, Id.array) / 3.0)
    K = Id4 - J
    return J, K


def _cubic_projectors():
    r"""
    Construct cubic-symmetry fourth-rank projectors.

    Returns
    -------
    J : :class:`SymmetricTensor4`
        Volumetric projector.
    Ka : :class:`SymmetricTensor4`
        Diagonal deviatoric projector (cubic anisotropic part of the diagonal).
    Kb : :class:`SymmetricTensor4`
        Off-diagonal shear projector.

    Notes
    -----
    The three projectors are mutually orthogonal and partition the identity:
    $\mathbb{J}+\mathbb{K}_a+\mathbb{K}_b = \mathbb{I}^s$.
    """
    J, _ = _isotropic_projectors()
    Lambda = SymmetricTensor4(array=jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])))
    Ka = Lambda - J
    Kb = SymmetricTensor4.identity() - Lambda
    return J, Ka, Kb


def _transverse_isotropic_projectors(axis: jax.Array):
    r"""
    Construct transverse-isotropic (Walpole) fourth-rank projectors.

    Parameters
    ----------
    axis : array_like, shape (3,)
        Unit symmetry axis $\hat{\mathbf{a}}$.

    Returns
    -------
    E1, E2, E3, E4, F, G : :class:`SymmetricTensor4`
        Six Walpole basis tensors spanning the transverse-isotropic subspace.
        E1 and E2 are projectors; E3, E4 are their cross terms; F and G
        are the remaining orthogonal complements.

    Notes
    -----
    The Walpole basis is not orthogonal in the usual sense: composition
    follows a $2\times 2$ block rule for E1..E4 and scalar inversion for
    F and G.  See :class:`TransverseIsotropicTensor4` for the inversion formula.
    """
    P = SymmetricTensor2(tensor=jnp.outer(axis, axis))
    Q = SymmetricTensor2(tensor=(jnp.eye(3) - jnp.outer(axis, axis)) / jnp.sqrt(2.0))

    E1 = SymmetricTensor4(array=jnp.outer(P.array, P.array))
    E2 = SymmetricTensor4(array=jnp.outer(Q.array, Q.array))
    E3 = SymmetricTensor4(array=jnp.outer(P.array, Q.array))
    E4 = SymmetricTensor4(array=jnp.outer(Q.array, P.array))

    def _sym4(A, B):
        return 0.5 * (jnp.einsum("ik,jl->ijkl", A, B) + jnp.einsum("il,jk->ijkl", A, B))

    Q33 = Q.tensor
    P33 = P.tensor
    F = SymmetricTensor4(tensor=2.0 * _sym4(Q33, Q33)) - E2
    G = SymmetricTensor4(tensor=jnp.sqrt(2.0) * (_sym4(P33, Q33) + _sym4(Q33, P33)))
    return E1, E2, E3, E4, F, G


# Precomputed basis stacks for orientation-independent classes (module-level constants)
_ISO_BASIS = jnp.stack([P.array for P in _isotropic_projectors()], axis=0)  # (2, 6, 6)
_CUB_BASIS = jnp.stack([P.array for P in _cubic_projectors()], axis=0)  # (3, 6, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────


class AbstractStructuredTensor4(eqx.Module):
    r"""
    Abstract base for symmetry-reduced fourth-rank tensors.

    Concrete subclasses store a coefficient vector
    ``_coeffs`` of shape ``(..., n)`` (the unique JAX leaf) and a
    precomputed basis stack ``_basis_arrays`` of shape ``(n, 6, 6)`` (static).
    Every shared operation is derived from these two objects alone.

    The tensor is represented as

    .. math::

        \mathbb{C} = \sum_{\alpha=1}^n c_\alpha \mathbb{P}_\alpha

    where $\mathbb{P}_\alpha$ are the basis projectors (stored in ``_basis_arrays``)
    and $c_\alpha$ are the coefficients (stored in ``_coeffs``).

    Subclasses must implement
    -------------------------
    :meth:`_rebuild`
        Factory: return a new instance of the same class from updated coefficients.
    :attr:`inv`
        Class-specific inversion (coefficient-wise or block-matrix).
    """

    _coeffs: jax.Array  # shape (..., n) — the unique JAX leaf
    _basis_arrays: (
        jax.Array
    )  # shape (n, 6, 6) — precomputed constant, frozen by module immutability

    # ── abstract interface ─────────────────────────────────────────────────────

    def _rebuild(self, new_coeffs: jax.Array) -> "AbstractStructuredTensor4":
        """
        Return a new same-type instance with updated coefficients.

        Parameters
        ----------
        new_coeffs : jax.Array
            Replacement coefficient vector, same shape as ``self._coeffs``.

        Returns
        -------
        AbstractStructuredTensor4
            A new instance of the concrete subclass.
        """
        raise NotImplementedError

    @property
    def inv(self) -> "AbstractStructuredTensor4":
        r"""
        Inverse of the tensor operator.

        Returns
        -------
        AbstractStructuredTensor4
            The inverse $\mathbb{C}^{-1}$ within the same symmetry class.
        """
        raise NotImplementedError

    # ── shared properties ──────────────────────────────────────────────────────

    @property
    def coeffs(self) -> jax.Array:
        r"""
        Expansion coefficients $c_\alpha$ in the symmetry projector basis.

        Returns
        -------
        jax.Array
            Shape ``(..., n)``.
        """
        return self._coeffs

    @property
    def array(self) -> jax.Array:
        r"""
        Materialised $(6, 6)$ Kelvin matrix.

        Computed as $\sum_\alpha c_\alpha \{P_\alpha\}$ via a single
        ``tensordot`` with the precomputed basis stack.  Supports arbitrary
        batch dimensions in ``_coeffs``.

        Returns
        -------
        jax.Array
            Shape ``(..., 6, 6)``.
        """
        return jnp.tensordot(self._coeffs, self._basis_arrays, axes=([-1], [0]))

    def to_symmetric(self) -> SymmetricTensor4:
        r"""
        Materialise to a general :class:`SymmetricTensor4`.

        Returns
        -------
        SymmetricTensor4
            The same tensor stored as a full $(6, 6)$ Kelvin matrix.
        """
        return SymmetricTensor4(array=self.array)

    def fourth_contract(self, other) -> jax.Array:
        r"""
        Full fourth-order contraction $\mathbb{C}::\mathbb{D} = C_{ijkl}D_{ijkl}$.

        Computed analytically as $\sum_\alpha c_\alpha (\mathbb{P}_\alpha :: \mathbb{D})$
        without materialising the full $(6,6)$ matrix.

        Parameters
        ----------
        other : :class:`SymmetricTensor4` or array_like, shape ``(..., 6, 6)``
            Second operand $\mathbb{D}$.

        Returns
        -------
        jax.Array
            Scalar (or batch of scalars).
        """
        D = _array4(other)
        PD = jnp.einsum("pij,ij->p", self._basis_arrays, D)  # (n,)
        return jnp.sum(self._coeffs * PD, axis=-1)

    def rotate(self, R: jax.Array) -> SymmetricTensor4:
        r"""
        Rotate the tensor by an orthogonal matrix $\mathbf{R}$.

        The default implementation materialises to :class:`SymmetricTensor4`
        and applies the rotation there.  Subclasses may override this when
        the class is closed under rotation (e.g. isotropic tensors).

        Parameters
        ----------
        R : array_like, shape (3, 3)
            Orthogonal rotation matrix.

        Returns
        -------
        SymmetricTensor4
        """
        return self.to_symmetric().rotate(R)

    # ── composition ``@`` ─────────────────────────────────────────────────────

    def __matmul__(self, other):
        r"""
        Double contraction or composition via the materialised Kelvin matrix.

        - ``C @ eps`` with :class:`SymmetricTensor2` $\boldsymbol\varepsilon$:
          double contraction $\mathbb{C}:\boldsymbol\varepsilon$,
          ``(6,6) @ (6,)``  →  :class:`SymmetricTensor2`.
        - ``C @ D`` with any rank-4 operand:
          composition $(\mathbb{C}:\mathbb{D})_{ijmn} = C_{ijkl}D_{klmn}$,
          ``(6,6) @ (6,6)``  →  :class:`SymmetricTensor4`.

        Subclasses override this for same-class pairs where a more efficient
        coefficient-space formula is available.
        """
        if isinstance(other, SymmetricTensor2):
            return SymmetricTensor2(array=self.array @ other._array)
        return SymmetricTensor4(array=self.array @ _array4(other))

    def __rmatmul__(self, other):
        r"""Right-multiply: ``D @ C`` → :class:`SymmetricTensor4`."""
        return SymmetricTensor4(array=_array4(other) @ self.array)

    # ── scalar arithmetic via _rebuild ─────────────────────────────────────────

    def __mul__(self, other):
        r"""Scalar multiplication $\alpha \mathbb{C}$; stays in the same symmetry class."""
        return self._rebuild(jnp.asarray(other) * self._coeffs)

    def __rmul__(self, other):
        r"""Scalar multiplication $\alpha \mathbb{C}$; stays in the same symmetry class."""
        return self._rebuild(jnp.asarray(other) * self._coeffs)

    def __truediv__(self, other):
        r"""Scalar division $\mathbb{C} / \alpha$; stays in the same symmetry class."""
        return self._rebuild(self._coeffs / jnp.asarray(other))

    def __neg__(self):
        r"""Negation $-\mathbb{C}$; stays in the same symmetry class."""
        return self._rebuild(-self._coeffs)

    # ── binary +/- ────────────────────────────────────────────────────────────

    def __add__(self, other):
        r"""
        Addition.

        Same symmetry class: coefficient-space addition, result stays in-class.
        Cross-class: materialise both to :class:`SymmetricTensor4`.
        """
        if type(other) is type(self):
            return self._rebuild(self._coeffs + other._coeffs)
        return SymmetricTensor4(array=self.array + _array4(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        r"""
        Subtraction.

        Same symmetry class: coefficient-space subtraction, result stays in-class.
        Cross-class: materialise both to :class:`SymmetricTensor4`.
        """
        if type(other) is type(self):
            return self._rebuild(self._coeffs - other._coeffs)
        return SymmetricTensor4(array=self.array - _array4(other))

    def __rsub__(self, other):
        if type(other) is type(self):
            return self._rebuild(other._coeffs - self._coeffs)
        return SymmetricTensor4(array=_array4(other) - self.array)

    def __repr__(self):
        return f"{type(self).__name__}(coeffs={self._coeffs})"


# ─────────────────────────────────────────────────────────────────────────────
# IsotropicTensor4
# ─────────────────────────────────────────────────────────────────────────────


class IsotropicTensor4(AbstractStructuredTensor4):
    r"""
    Isotropic fourth-rank tensor.

    Parameterised by bulk and shear moduli and expressed in the orthogonal
    projector basis $(\mathbb{J}, \mathbb{K})$:

    .. math::

        \mathbb{C} = 3\kappa\,\mathbb{J} + 2\mu\,\mathbb{K}

    Parameters
    ----------
    coeffs : array_like, shape ``(..., 2)``, optional
        Direct basis coefficients $[3\kappa, 2\mu]$.
    kappa : float or array_like, optional
        Bulk modulus $\kappa$.
    mu : float or array_like, optional
        Shear modulus $\mu$.

    Notes
    -----
    Exactly one of ``coeffs`` or the pair ``(kappa, mu)`` must be provided.
    Coefficients are stored as $\{3\kappa, 2\mu\}$ so that the projector
    expansion $c_1\mathbb{J}+c_2\mathbb{K}$ uses the standard elasticity
    identities $\mathbb{J}:\mathbf{A} = \tfrac{1}{3}\operatorname{tr}(\mathbf{A})\mathbf{I}$
    and $\mathbb{K}:\mathbf{A} = \operatorname{dev}(\mathbf{A})$.

    Batched use (one tensor per material point) is supported by passing
    array-valued ``kappa`` and ``mu`` of shape ``(...,)``.
    """

    # Class-level projectors — accessible as IsotropicTensor4.J and .K
    J, K = _isotropic_projectors()

    def __init__(self, *, coeffs=None, kappa=None, mu=None):
        if coeffs is None:
            if kappa is None or mu is None:
                raise ValueError("Provide either coeffs or (kappa, mu)")
            coeffs = jnp.stack(
                [3.0 * jnp.asarray(kappa), 2.0 * jnp.asarray(mu)], axis=-1
            )
        object.__setattr__(self, "_coeffs", jnp.asarray(coeffs))
        object.__setattr__(self, "_basis_arrays", _ISO_BASIS)

    def _rebuild(self, new_coeffs: jax.Array) -> "IsotropicTensor4":
        return IsotropicTensor4(coeffs=new_coeffs)

    @property
    def kappa(self) -> jax.Array:
        r"""Bulk modulus $\kappa = c_1 / 3$."""
        return self._coeffs[..., 0] / 3.0

    @property
    def mu(self) -> jax.Array:
        r"""Shear modulus $\mu = c_2 / 2$."""
        return self._coeffs[..., 1] / 2.0

    @property
    def inv(self) -> "IsotropicTensor4":
        r"""
        Inverse $\mathbb{C}^{-1}$.

        Because $\mathbb{J}$ and $\mathbb{K}$ are orthogonal projectors,
        the inverse is obtained by inverting each coefficient:
        $\mathbb{C}^{-1} = \tfrac{1}{3\kappa}\mathbb{J} + \tfrac{1}{2\mu}\mathbb{K}$.

        Returns
        -------
        IsotropicTensor4
        """
        return IsotropicTensor4(coeffs=1.0 / self._coeffs)

    def rotate(self, R: jax.Array) -> "IsotropicTensor4":
        r"""
        Isotropic tensors are invariant under all rotations.

        Parameters
        ----------
        R : array_like, shape (3, 3)
            Orthogonal rotation matrix (unused).

        Returns
        -------
        IsotropicTensor4
            ``self`` unchanged.
        """
        return self

    def __matmul__(self, other):
        r"""
        Double contraction $\mathbb{C}:\mathbb{D}$.

        - :class:`IsotropicTensor4` : coefficient elementwise product
          (orthogonal projector algebra), result stays isotropic.
        - Anything else: delegates to base-class materialised path.
        """
        if isinstance(other, IsotropicTensor4):
            # (c₁J + c₂K):(d₁J + d₂K) = c₁d₁J + c₂d₂K
            return IsotropicTensor4(coeffs=self._coeffs * other._coeffs)
        return super().__matmul__(other)

    @classmethod
    def project(cls, C: SymmetricTensor4) -> "IsotropicTensor4":
        r"""
        Project a :class:`SymmetricTensor4` onto the isotropic subspace.

        Extracts the isotropic part by computing
        $\kappa = \tfrac{1}{3}\mathbb{C}::\mathbb{J}$ and
        $\mu = \tfrac{1}{10}\mathbb{C}::\mathbb{K}$.

        Parameters
        ----------
        C : :class:`SymmetricTensor4`

        Returns
        -------
        IsotropicTensor4
        """
        J, K = _isotropic_projectors()
        kappa = C.fourth_contract(J) / 3.0
        mu = C.fourth_contract(K) / 10.0
        return cls(kappa=kappa, mu=mu)


# ─────────────────────────────────────────────────────────────────────────────
# CubicTensor4
# ─────────────────────────────────────────────────────────────────────────────


class CubicTensor4(AbstractStructuredTensor4):
    r"""
    Cubic-symmetric fourth-rank tensor.

    Represented in the basis of three mutually orthogonal projectors:

    .. math::

        \mathbb{C} = 3\kappa\,\mathbb{J}
                   + 2\mu_a\,\mathbb{K}_a
                   + 2\mu_b\,\mathbb{K}_b

    where $\mathbb{J}$ is the volumetric projector, $\mathbb{K}_a$ projects
    onto the diagonal deviatoric part, and $\mathbb{K}_b$ projects onto the
    off-diagonal shear part.

    Parameters
    ----------
    coeffs : array_like, shape ``(..., 3)``, optional
        Direct basis coefficients $[3\kappa, 2\mu_a, 2\mu_b]$.
    kappa : float or array_like, optional
        Cubic bulk modulus $\kappa$.
    mua : float or array_like, optional
        Diagonal deviatoric modulus $\mu_a$.
    mub : float or array_like, optional
        Off-diagonal shear modulus $\mu_b$.

    Notes
    -----
    When $\mu_a = \mu_b = \mu$ the tensor reduces to the isotropic case
    $\mathbb{C} = 3\kappa\mathbb{J} + 2\mu\mathbb{K}$.
    Coefficients are stored as $\{3\kappa, 2\mu_a, 2\mu_b\}$.
    """

    # Class-level projectors — accessible as CubicTensor4.J, .Ka, .Kb
    J, Ka, Kb = _cubic_projectors()

    def __init__(self, *, coeffs=None, kappa=None, mua=None, mub=None):
        if coeffs is None:
            if None in (kappa, mua, mub):
                raise ValueError("Provide either coeffs or (kappa, mua, mub)")
            coeffs = jnp.stack(
                [
                    3.0 * jnp.asarray(kappa),
                    2.0 * jnp.asarray(mua),
                    2.0 * jnp.asarray(mub),
                ],
                axis=-1,
            )
        object.__setattr__(self, "_coeffs", jnp.asarray(coeffs))
        object.__setattr__(self, "_basis_arrays", _CUB_BASIS)

    def _rebuild(self, new_coeffs: jax.Array) -> "CubicTensor4":
        return CubicTensor4(coeffs=new_coeffs)

    @property
    def inv(self) -> "CubicTensor4":
        r"""
        Inverse $\mathbb{C}^{-1}$.

        Because the three projectors are mutually orthogonal, the inverse
        is coefficient-wise: $c_\alpha^{-1}$ for each $\alpha$.

        Returns
        -------
        CubicTensor4
        """
        return CubicTensor4(coeffs=1.0 / self._coeffs)

    def __matmul__(self, other):
        r"""
        Double contraction $\mathbb{C}:\mathbb{D}$.

        - :class:`CubicTensor4` : coefficient elementwise product
          (orthogonal projector algebra), result stays cubic.
        - Anything else: delegates to base-class materialised path.
        """
        if isinstance(other, CubicTensor4):
            return CubicTensor4(coeffs=self._coeffs * other._coeffs)
        return super().__matmul__(other)

    @classmethod
    def project(cls, C: SymmetricTensor4) -> "CubicTensor4":
        r"""
        Project a :class:`SymmetricTensor4` onto the cubic subspace.

        Parameters
        ----------
        C : :class:`SymmetricTensor4`

        Returns
        -------
        CubicTensor4
        """
        J, Ka, Kb = _cubic_projectors()
        kappa = C.fourth_contract(J) / 3.0
        mua = C.fourth_contract(Ka) / 4.0
        mub = C.fourth_contract(Kb) / 6.0
        return cls(kappa=kappa, mua=mua, mub=mub)


# ─────────────────────────────────────────────────────────────────────────────
# TransverseIsotropicTensor4
# ─────────────────────────────────────────────────────────────────────────────


class TransverseIsotropicTensor4(AbstractStructuredTensor4):
    r"""
    Transversely isotropic fourth-rank tensor in the Walpole basis.

    Defined with respect to a unit symmetry axis $\hat{\mathbf{a}}$ and
    expanded in the six Walpole basis tensors
    $\mathbb{E}_1, \mathbb{E}_2, \mathbb{E}_3, \mathbb{E}_4, \mathbb{F}, \mathbb{G}$:

    .. math::

        \mathbb{C} = \sum_{\alpha=1}^6 c_\alpha \mathbb{P}_\alpha

    Parameters
    ----------
    axis : array_like, shape (3,)
        Unit symmetry axis $\hat{\mathbf{a}}$.  Need not be pre-normalised.
    coeffs : array_like, shape ``(..., 6)``
        Walpole basis coefficients $[c_1, c_2, c_3, c_4, c_5, c_6]$.

    Notes
    -----
    Inversion uses the $2\times 2$ block structure of $\mathbb{E}_1\ldots\mathbb{E}_4$
    and scalar inversion for $\mathbb{F}$ and $\mathbb{G}$.
    """

    axis: jax.Array

    def __init__(self, axis: jax.Array, coeffs: jax.Array):
        axis = jnp.asarray(axis) / jnp.linalg.norm(axis)
        basis = jnp.stack(
            [P.array for P in _transverse_isotropic_projectors(axis)], axis=0
        )  # (6, 6, 6)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "_coeffs", jnp.asarray(coeffs))
        object.__setattr__(self, "_basis_arrays", basis)

    def _rebuild(self, new_coeffs: jax.Array) -> "TransverseIsotropicTensor4":
        return TransverseIsotropicTensor4(self.axis, new_coeffs)

    @property
    def inv(self) -> "TransverseIsotropicTensor4":
        r"""
        Inverse operator within the transverse-isotropic subspace.

        The sub-block $[c_1, c_3; c_4, c_2]$ corresponding to
        $\mathbb{E}_1\ldots\mathbb{E}_4$ is inverted as a $2\times 2$ matrix,
        while $c_5$ and $c_6$ (for $\mathbb{F}$ and $\mathbb{G}$) are inverted
        individually.

        Returns
        -------
        TransverseIsotropicTensor4
        """
        a = self._coeffs[..., :4]
        A = jnp.stack(
            [
                jnp.stack([a[..., 0], a[..., 2]], axis=-1),
                jnp.stack([a[..., 3], a[..., 1]], axis=-1),
            ],
            axis=-2,
        )  # (..., 2, 2)
        invA = jnp.linalg.inv(A)
        inv_c = jnp.concatenate(
            [
                invA[..., 0, 0:1],
                invA[..., 1, 1:2],
                invA[..., 0, 1:2],
                invA[..., 1, 0:1],
                1.0 / self._coeffs[..., 4:5],
                1.0 / self._coeffs[..., 5:6],
            ],
            axis=-1,
        )
        return TransverseIsotropicTensor4(self.axis, inv_c)

    @classmethod
    def project(
        cls, axis: jax.Array, C: SymmetricTensor4
    ) -> "TransverseIsotropicTensor4":
        r"""
        Project a :class:`SymmetricTensor4` onto the transverse-isotropic subspace.

        Parameters
        ----------
        axis : array_like, shape (3,)
            Unit symmetry axis $\hat{\mathbf{a}}$.
        C : :class:`SymmetricTensor4`

        Returns
        -------
        TransverseIsotropicTensor4
        """
        E1, E2, E3, E4, F, G = _transverse_isotropic_projectors(axis)
        coeffs = jnp.stack(
            [
                C.fourth_contract(E1),
                C.fourth_contract(E2),
                C.fourth_contract(E3),
                C.fourth_contract(E4),
                C.fourth_contract(F) / 2.0,
                C.fourth_contract(G) / 2.0,
            ],
            axis=-1,
        )
        return cls(axis, coeffs)


# ─────────────────────────────────────────────────────────────────────────────
# Public re-exports
# ─────────────────────────────────────────────────────────────────────────────

isotropic_projectors = _isotropic_projectors
cubic_projectors = _cubic_projectors
transverse_isotropic_projectors = _transverse_isotropic_projectors
