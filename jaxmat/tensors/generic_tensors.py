"""
jaxmat/tensors/generic_tensors.py

Second- and fourth-rank tensor classes for solid mechanics computations in JAX.

Design principles
-----------------
* The Kelvin-Mandel array is the unique JAX leaf stored per object.
  All other representations are derived properties.
* All index maps and weight arrays are module-level constants computed once
  at import time and shared across all instances.
* ``tensor`` properties use gather operations (no scatter, no intermediate
  zero allocation) for rank-2 classes.
* Each concrete class exposes exactly one equinox field: ``_array``.
* Operator ``@`` semantics:

  - :class:`Tensor2` ``@`` :class:`Tensor2` → dense composition → :class:`Tensor2`.
  - :class:`SymmetricTensor4` ``@`` :class:`SymmetricTensor2` → Kelvin matvec → :class:`SymmetricTensor2`.
  - :class:`SymmetricTensor4` ``@`` :class:`SymmetricTensor4` → Kelvin matmul → :class:`SymmetricTensor4`.

* Named methods for algebraically unambiguous operations:
  :meth:`~Tensor2.double_contract`, :meth:`~Tensor2.matvec`,
  :meth:`~_AbstractTensor4.fourth_contract`.
* :class:`_AbstractTensor4` centralizes all code shared between
  :class:`Tensor4` and :class:`SymmetricTensor4`; subclasses supply only
  ``__init__``, ``tensor``, ``__matmul__``, ``base_array_shape``, and
  ``identity``.
"""  # noqa: E501

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.tensors.mappings import (
    full_rank2_map,
    full_rank4_map,
    kelvin_rank2_map,
    kelvin_rank4_map,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level index maps — computed once at import, shared by all instances
# ─────────────────────────────────────────────────────────────────────────────

(_T2_I, _T2_J), _ = full_rank2_map(3)
(_S2_I, _S2_J), _S2_W = kelvin_rank2_map(3)
(_T4_I, _T4_J, _T4_K, _T4_L), _ = full_rank4_map(3)
(_S4_I, _S4_J, _S4_K, _S4_L), _S4_W = kelvin_rank4_map(3)

# Gather table for Tensor2.tensor: T[i,j] = _array[_T2_RECON33[i,j]]
_T2_RECON33 = jnp.array([[0, 3, 5], [4, 1, 7], [6, 8, 2]])

# Permutation for Tensor2.T: _T2_TRANSPOSE[k] = position of the (j,i) element
# given that the (i,j) element occupies position k in the Kelvin array.
_T2_TRANSPOSE = jnp.array([0, 1, 2, 4, 3, 6, 5, 8, 7])

# Gather table for SymmetricTensor2.tensor: T[i,j] = (_array/_S2_W)[_S2_RECON33[i,j]]
_S2_RECON33 = jnp.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])

# Precomputed gather for _sym2_to_full9: Kelvin-array indices and weights
_S2_TO_FULL9_IDX = _S2_RECON33[_T2_I, _T2_J]  # shape (9,)
_S2_TO_FULL9_W = _S2_W[_S2_TO_FULL9_IDX]  # shape (9,)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _check_trailing(arr, expected, name):
    """Raise ``ValueError`` if the trailing shape of ``arr`` does not match ``expected``."""
    if arr.shape[-len(expected) :] != tuple(expected):
        raise ValueError(
            f"Wrong {name} trailing shape {arr.shape[-len(expected) :]}; expected {tuple(expected)}"
        )


def _raw_array(arg):
    """
    Return the underlying JAX array from a tensor object or a plain array.

    If ``arg`` has a ``_array`` attribute (i.e. it is an equinox tensor
    module), its ``_array`` field is returned directly.  Otherwise
    ``jnp.asarray`` is applied.

    This is used in constructors and in the :attr:`SymmetricTensor2.tensor`
    property to handle the case where equinox's pytree unflatten sets
    ``_array`` to a tensor module object during automatic differentiation,
    bypassing ``__init__``.
    """
    return arg._array if hasattr(arg, "_array") else jnp.asarray(arg)


def _sym2_to_full9(s: "SymmetricTensor2") -> jax.Array:
    """
    Convert a :class:`SymmetricTensor2` Kelvin array to the full 9-component
    ordering without constructing the dense (3, 3) tensor as an intermediate.

    Parameters
    ----------
    s : SymmetricTensor2

    Returns
    -------
    jax.Array
        Shape ``(..., 9)``.
    """
    return (s._array / _S2_W)[..., _S2_TO_FULL9_IDX]


def _array4(obj) -> jax.Array:
    """
    Extract the Kelvin array from a rank-4 tensor object or a plain array.

    Parameters
    ----------
    obj : _AbstractTensor4 or array_like

    Returns
    -------
    jax.Array
        The raw Kelvin matrix, shape ``(..., N, N)``.
    """
    return obj._array if hasattr(obj, "_array") else jnp.asarray(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Marker base class
# ─────────────────────────────────────────────────────────────────────────────


class Tensor(eqx.Module):
    """
    Empty marker base class for all jaxmat tensor objects.

    Provides a single type to test against with ``isinstance(x, Tensor)``
    without coupling to any specific rank or symmetry class.

    Both rank-2 classes (:class:`Tensor2`, :class:`SymmetricTensor2`) and
    rank-4 classes (:class:`Tensor4`, :class:`SymmetricTensor4`, and the
    symmetry-reduced subclasses) inherit from this marker.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Rank-2 tensors
# ─────────────────────────────────────────────────────────────────────────────


class Tensor2(Tensor):
    r"""
    Full (non-symmetric) second-rank tensor in 3-D.

    Stored as a 9-component Kelvin array in the ordering
    $[T_{11}, T_{22}, T_{33}, T_{12}, T_{21}, T_{13}, T_{31}, T_{23}, T_{32}]$.

    Parameters
    ----------
    tensor : array_like, shape ``(..., 3, 3)``, optional
        Dense tensor representation.
    array : array_like, shape ``(..., 9)``, optional
        Pre-built Kelvin array.  Passing another :class:`Tensor2` is accepted;
        its ``_array`` field is used directly.

    Notes
    -----
    Exactly one of ``tensor`` or ``array`` may be provided.  If neither is
    given the tensor is initialized to zero.

    Batch dimensions precede the storage dimension: a batch of $N$ tensors
    has ``array.shape == (N, 9)`` and ``tensor.shape == (N, 3, 3)``.

    The ``@`` operator performs **dense matrix composition**
    $(\mathbf{T} \cdot \mathbf{S})_{ik} = T_{ij} S_{jk}$ and always
    returns a :class:`Tensor2`, regardless of the symmetry of the operands.
    Use :meth:`double_contract` for $\mathbf{T}:\mathbf{S}$ and
    :meth:`matvec` for $\mathbf{T} \cdot \mathbf{v}$.
    """

    _array: jax.Array
    """9-component Kelvin storage array — the sole JAX leaf."""

    dim = 3
    """Spatial dimension."""
    rank = 2
    """Tensor rank."""
    base_tensor_shape = (3, 3)
    """Shape of the base (unbatched) dense tensor."""
    array_rank = 1
    """Number of storage dimensions (1 for rank-2, 2 for rank-4)."""
    base_array_shape = (9,)
    """Shape of the base (unbatched) Kelvin array."""

    def __init__(self, *, tensor=None, array=None):
        if tensor is not None:
            t = jnp.asarray(tensor)
            _check_trailing(t, (3, 3), "tensor")
            object.__setattr__(self, "_array", t[..., _T2_I, _T2_J])
        elif array is not None:
            a = _raw_array(array)
            _check_trailing(a, (9,), "array")
            object.__setattr__(self, "_array", a)
        else:
            object.__setattr__(self, "_array", jnp.zeros(9))

    # ── representations ───────────────────────────────────────────────────────

    @property
    def array(self) -> jax.Array:
        """
        9-component Kelvin array.

        Returns
        -------
        jax.Array
            Shape ``(..., 9)``.
        """
        return self._array

    @property
    def tensor(self) -> jax.Array:
        """
        Dense tensor representation.

        Reconstructed via a single gather on ``_array`` followed by a
        reshape — no scatter, no zero allocation.

        Returns
        -------
        jax.Array
            Shape ``(..., 3, 3)``.
        """
        return self._array[..., _T2_RECON33].reshape((*self._array.shape[:-1], 3, 3))

    @property
    def shape(self) -> tuple:
        """Shape of the underlying Kelvin array ``(..., 9)``."""
        return self._array.shape

    @property
    def batch_shape(self) -> tuple:
        """Leading batch dimensions ``(...)``."""
        return self._array.shape[:-1]

    @property
    def tensor_shape(self) -> tuple:
        """Shape of the dense tensor ``(..., 3, 3)``."""
        return self.batch_shape + self.base_tensor_shape

    def __jax_array__(self):
        """Allow ``jnp.asarray(T)`` to return the dense (3, 3) tensor."""
        return self.tensor

    # ── scalar invariants ─────────────────────────────────────────────────────

    @property
    def tr(self) -> jax.Array:
        r"""
        Trace $\mathrm{tr}(\mathbf{T}) = T_{ii}$.

        Returns
        -------
        jax.Array
        """
        return jnp.sum(self._array[..., :3], axis=-1)

    @property
    def det(self) -> jax.Array:
        r"""
        Determinant $\det(\mathbf{T})$.

        Returns
        -------
        jax.Array
        """
        from jaxmat.tensors.linear_algebra import det33

        return det33(self.tensor)

    # ── derived tensors ───────────────────────────────────────────────────────

    @property
    def T(self) -> "Tensor2":
        r"""
        Transpose $T^{\mathsf{T}}_{ij} = T_{ji}$.

        Computed via a single gather on the Kelvin array; no dense
        intermediate is constructed.

        Returns
        -------
        Tensor2
        """
        return Tensor2(array=self._array[..., _T2_TRANSPOSE])

    @property
    def sym(self) -> "SymmetricTensor2":
        r"""
        Symmetric part $(\mathbf{T} + \mathbf{T}^{\mathsf{T}}) / 2$.

        Returns
        -------
        SymmetricTensor2
        """
        return SymmetricTensor2(tensor=0.5 * (self.tensor + jnp.swapaxes(self.tensor, -1, -2)))

    @property
    def skw(self) -> "Tensor2":
        r"""
        Skew-symmetric part $(\mathbf{T} - \mathbf{T}^{\mathsf{T}}) / 2$.

        Returns
        -------
        Tensor2
        """
        return Tensor2(tensor=0.5 * (self.tensor - jnp.swapaxes(self.tensor, -1, -2)))

    @property
    def inv(self) -> "Tensor2":
        r"""
        Inverse $\mathbf{T}^{-1}$.

        Returns
        -------
        Tensor2
        """
        from jaxmat.tensors.linear_algebra import inv33

        return Tensor2(tensor=inv33(self.tensor))

    def rotate(self, R: jax.Array) -> "Tensor2":
        r"""
        Rotate the tensor: $\mathbf{R} \mathbf{T} \mathbf{R}^{\mathsf{T}}$.

        Parameters
        ----------
        R : array_like, shape (3, 3)
            Orthogonal rotation matrix.

        Returns
        -------
        Tensor2
        """
        T = jnp.einsum("...ij,...jk->...ik", R, self.tensor)
        return type(self)(tensor=jnp.einsum("...ij,...kj->...ik", T, R))

    # ── named contractions ─────────────────────────────────────────────────────

    def double_contract(self, other: "Tensor2") -> jax.Array:
        r"""
        Double contraction $\mathbf{T} : \mathbf{S} = T_{ij} S_{ij}$.

        Parameters
        ----------
        other : Tensor2 or array_like, shape ``(..., 3, 3)``

        Returns
        -------
        jax.Array
            Scalar (or batch of scalars).
        """
        return jnp.sum(self.tensor * jnp.asarray(other), axis=(-2, -1))

    def matvec(self, v: jax.Array) -> jax.Array:
        r"""
        Matrix-vector product $(\mathbf{T} \cdot \mathbf{v})_i = T_{ij} v_j$.

        Parameters
        ----------
        v : array_like, shape ``(..., 3)``

        Returns
        -------
        jax.Array
            Shape ``(..., 3)``.
        """
        return jnp.einsum("...ij,...j->...i", self.tensor, v)

    # ── composition ``@`` ─────────────────────────────────────────────────────

    def __matmul__(self, other) -> "Tensor2":
        r"""
        Dense matrix composition $(\mathbf{T} \cdot \mathbf{S})_{ik} = T_{ij} S_{jk}$.

        Always returns :class:`Tensor2`, regardless of the symmetry of the
        operands.  Use :meth:`double_contract` for the double contraction
        $\mathbf{T} : \mathbf{S}$ and :meth:`matvec` for $\mathbf{T} \cdot
        \mathbf{v}$.
        """
        if isinstance(other, Tensor2):
            return Tensor2(tensor=self.tensor @ other.tensor)
        if isinstance(other, jax.Array) and other.shape[-2:] == (3, 3):
            return Tensor2(tensor=self.tensor @ other)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, jax.Array) and other.shape[-2:] == (3, 3):
            return Tensor2(tensor=other @ self.tensor)
        return NotImplemented

    # ── arithmetic ─────────────────────────────────────────────────────────────

    def __add__(self, other):
        if type(other) is type(self):  # same class → operate on Kelvin arrays
            return type(self)(array=self._array + other._array)
        if isinstance(other, Tensor2):  # cross-symmetry → promote to Tensor2
            return Tensor2(tensor=self.tensor + jnp.asarray(other))
        arr = jnp.asarray(other)
        if arr.shape[-2:] == (3, 3):  # plain (3,3) array → tensor path
            return type(self)(tensor=self.tensor + arr)
        return type(self)(array=self._array + arr)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is type(self):
            return type(self)(array=self._array - other._array)
        if isinstance(other, Tensor2):
            return Tensor2(tensor=self.tensor - jnp.asarray(other))
        arr = jnp.asarray(other)
        if arr.shape[-2:] == (3, 3):
            return type(self)(tensor=self.tensor - arr)
        return type(self)(array=self._array - arr)

    def __rsub__(self, other):
        if type(other) is type(self):
            return type(self)(array=other._array - self._array)
        if isinstance(other, Tensor2):
            return Tensor2(tensor=jnp.asarray(other) - self.tensor)
        arr = jnp.asarray(other)
        if arr.shape[-2:] == (3, 3):
            return type(self)(tensor=arr - self.tensor)
        return type(self)(array=arr - self._array)

    def __mul__(self, other):
        return type(self)(array=jnp.asarray(other) * self._array)

    def __rmul__(self, other):
        return type(self)(array=jnp.asarray(other) * self._array)

    def __truediv__(self, other):
        return type(self)(array=self._array / jnp.asarray(other))

    def __neg__(self):
        return type(self)(array=-self._array)

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __repr__(self):
        return f"{type(self).__name__}(shape={self._array.shape})"

    @classmethod
    def identity(cls) -> "Tensor2":
        r"""
        Second-rank identity $\delta_{ij}$.

        Returns
        -------
        Tensor2
        """
        return cls(tensor=jnp.eye(3))


class SymmetricTensor2(Tensor2):
    r"""
    Symmetric second-rank tensor in 3-D.

    Stored as a 6-component Kelvin-Mandel array

    .. math::

        \{T\} = [T_{11},\, T_{22},\, T_{33},\,
                 \sqrt{2}\,T_{12},\, \sqrt{2}\,T_{13},\, \sqrt{2}\,T_{23}]

    The $\sqrt{2}$ scaling makes the Kelvin basis orthonormal so that the
    dot product of two Kelvin vectors equals the double contraction of the
    corresponding tensors: $\{T\} \cdot \{S\} = \mathbf{T} : \mathbf{S}$.

    Parameters
    ----------
    tensor : array_like, shape ``(..., 3, 3)``, optional
        Dense symmetric tensor.
    array : array_like, shape ``(..., 6)``, optional
        Pre-built Kelvin array.

    Notes
    -----
    The ``@`` operator returns :class:`Tensor2` because the product of two
    symmetric matrices is not symmetric in general.

    :meth:`double_contract` operates directly on the Kelvin arrays
    (dot product), with no dense intermediate.

    Batch dimensions and ``jacfwd`` compatibility: equinox may set ``_array``
    directly to the tangent module during forward-mode AD, bypassing
    ``__init__``.  The :attr:`tensor` property uses :func:`_raw_array` to
    unwrap the value safely in that case.
    """

    base_array_shape = (6,)
    """Shape of the base (unbatched) Kelvin array."""

    def __init__(self, *, tensor=None, array=None):
        if tensor is not None:
            t = jnp.asarray(tensor)
            _check_trailing(t, (3, 3), "tensor")
            object.__setattr__(self, "_array", t[..., _S2_I, _S2_J] * _S2_W)
        elif array is not None:
            a = _raw_array(array)
            _check_trailing(a, (6,), "array")
            object.__setattr__(self, "_array", a)
        else:
            object.__setattr__(self, "_array", jnp.zeros(6))

    @property
    def tensor(self) -> jax.Array:
        r"""
        Dense symmetric (3, 3) tensor.

        Reconstructed by scattering the Kelvin components into the upper
        triangle and symmetrising.  :func:`_raw_array` is applied to
        ``_array`` first to handle the case where equinox sets ``_array``
        to the tangent module object during ``jacfwd`` (bypassing
        ``__init__``).

        Returns
        -------
        jax.Array
            Shape ``(..., 3, 3)``.
        """
        array = _raw_array(self._array)  # (..., 6) JAX array
        out = jnp.zeros((*array.shape[:-1], 3, 3), dtype=array.dtype)
        out = out.at[..., _S2_I, _S2_J].add(array * _S2_W)
        return 0.5 * (out + jnp.swapaxes(out, -1, -2))

    @property
    def T(self) -> "SymmetricTensor2":
        r"""
        Transpose — returns ``self`` since $\mathbf{T} = \mathbf{T}^{\mathsf{T}}$.

        Returns
        -------
        SymmetricTensor2
        """
        return self

    def double_contract(self, other) -> jax.Array:
        r"""
        Double contraction $\mathbf{T} : \mathbf{S} = T_{ij} S_{ij}$.

        When ``other`` is also a :class:`SymmetricTensor2`, the contraction
        reduces to a Kelvin dot product $\{T\} \cdot \{S\}$ with no dense
        intermediate.

        Parameters
        ----------
        other : SymmetricTensor2 or array_like, shape ``(..., 3, 3)``

        Returns
        -------
        jax.Array
            Scalar (or batch of scalars).
        """
        if isinstance(other, SymmetricTensor2):
            return jnp.sum(self._array * other._array, axis=-1)
        return jnp.sum(self.tensor * jnp.asarray(other), axis=(-2, -1))

    def __matmul__(self, other) -> Tensor2:
        r"""
        Dense composition $(\mathbf{T} \cdot \mathbf{S})_{ik} = T_{ij} S_{jk}$.

        Returns :class:`Tensor2` because the product of two symmetric tensors
        is not symmetric in general.
        """
        if isinstance(other, Tensor2):
            return Tensor2(tensor=self.tensor @ other.tensor)
        if isinstance(other, jax.Array) and other.shape[-2:] == (3, 3):
            return Tensor2(tensor=self.tensor @ other)
        return NotImplemented


# ─────────────────────────────────────────────────────────────────────────────
# Rank-4 tensors
# ─────────────────────────────────────────────────────────────────────────────


class _AbstractTensor4(Tensor):
    r"""
    Abstract base class shared by :class:`Tensor4` and :class:`SymmetricTensor4`.

    Centralises all operations that are identical for both subclasses:
    shape metadata, arithmetic (``+``, ``-``, ``*``, ``/``, negation),
    :attr:`T`, :attr:`inv`, :meth:`fourth_contract`, :meth:`rotate`,
    and ``__jax_array__``.  Every method uses ``type(self)(...)`` so that
    the correct concrete subclass is returned without any explicit dispatch.

    Subclasses must provide
    -------------------------
    ``base_array_shape``
        Class-level tuple giving the storage array shape, e.g. ``(9, 9)``
        or ``(6, 6)``.
    ``__init__``
        Constructor accepting ``tensor=`` and ``array=`` keyword arguments.
    ``tensor`` (property)
        Dense ``(3, 3, 3, 3)`` representation.
    ``__matmul__``
        Double contraction / composition dispatch.
    ``identity`` (classmethod)
        Return the identity tensor of the concrete subclass.
    """

    _array: jax.Array
    """Kelvin storage matrix — the sole JAX leaf."""

    dim = 3
    """Spatial dimension."""
    rank = 4
    """Tensor rank."""
    base_tensor_shape = (3, 3, 3, 3)
    """Shape of the base (unbatched) dense tensor."""
    array_rank = 2
    """Number of storage dimensions."""

    @property
    def array(self) -> jax.Array:
        """
        Kelvin matrix.

        Returns
        -------
        jax.Array
            Shape ``(..., N, N)`` where ``N`` is 9 for :class:`Tensor4`
            and 6 for :class:`SymmetricTensor4`.
        """
        return self._array

    @property
    def shape(self) -> tuple:
        """Shape of the Kelvin storage array ``(..., N, N)``."""
        return self._array.shape

    @property
    def batch_shape(self) -> tuple:
        """Leading batch dimensions ``(...)``."""
        return self._array.shape[:-2]

    @property
    def tensor_shape(self) -> tuple:
        """Shape of the dense tensor ``(..., 3, 3, 3, 3)``."""
        return self.batch_shape + self.base_tensor_shape

    def __jax_array__(self):
        """Allow ``jnp.asarray(C)`` to return the dense (3, 3, 3, 3) tensor."""
        return self.tensor

    @property
    def T(self) -> "_AbstractTensor4":
        r"""
        Major transpose $(\mathbb{C}^{\mathsf{T}})_{ijkl} = C_{klij}$.

        Implemented as a swap of the last two axes of the Kelvin matrix.

        Returns
        -------
        Same type as ``self``.
        """
        return type(self)(array=jnp.swapaxes(self._array, -1, -2))

    @property
    def inv(self) -> "_AbstractTensor4":
        r"""
        Inverse of the Kelvin matrix $\mathbb{C}^{-1}$.

        Returns
        -------
        Same type as ``self``.
        """
        return type(self)(array=jnp.linalg.inv(self._array))

    def fourth_contract(self, other) -> jax.Array:
        r"""
        Full fourth-order contraction $\mathbb{C} :: \mathbb{D} = C_{ijkl} D_{ijkl}$.

        Computed as the Frobenius inner product of the two Kelvin matrices.

        Parameters
        ----------
        other : _AbstractTensor4 or array_like

        Returns
        -------
        jax.Array
            Scalar (or batch of scalars).
        """
        return jnp.sum(self._array * _array4(other), axis=(-2, -1))

    def rotate(self, R: jax.Array) -> "_AbstractTensor4":
        r"""
        Rotate the tensor by applying $\mathbf{R}$ to each index.

        Parameters
        ----------
        R : array_like, shape (3, 3)
            Orthogonal rotation matrix.

        Returns
        -------
        Same type as ``self``.
        """
        T = self.tensor
        for ax in range(4):
            T = jnp.moveaxis(jnp.tensordot(R, T, (1, ax)), 0, ax)
        return type(self)(tensor=T)

    def __add__(self, other):
        return type(self)(array=self._array + _array4(other))

    def __radd__(self, other):
        return type(self)(array=_array4(other) + self._array)

    def __sub__(self, other):
        return type(self)(array=self._array - _array4(other))

    def __rsub__(self, other):
        return type(self)(array=_array4(other) - self._array)

    def __mul__(self, other):
        return type(self)(array=jnp.asarray(other) * self._array)

    def __rmul__(self, other):
        return type(self)(array=jnp.asarray(other) * self._array)

    def __truediv__(self, other):
        return type(self)(array=self._array / jnp.asarray(other))

    def __neg__(self):
        return type(self)(array=-self._array)

    def __repr__(self):
        return f"{type(self).__name__}(shape={self._array.shape})"

    def __getitem__(self, idx):
        return self.tensor[idx]


class Tensor4(_AbstractTensor4):
    r"""
    Full (non-minor-symmetric) fourth-rank tensor in 3-D.

    Stored as a $(9 \times 9)$ Kelvin matrix corresponding to the full
    double-index ordering $[T_{11}, T_{22}, T_{33}, T_{12}, T_{21},
    T_{13}, T_{31}, T_{23}, T_{32}]$.

    Parameters
    ----------
    tensor : array_like, shape ``(..., 3, 3, 3, 3)``, optional
        Dense tensor representation.
    array : array_like, shape ``(..., 9, 9)``, optional
        Pre-built Kelvin matrix.

    Notes
    -----
    The ``@`` operator denotes **double contraction**:

    - :class:`Tensor4` ``@`` :class:`Tensor2` → $C_{ijkl}\varepsilon_{kl}$ → :class:`Tensor2`.
    - :class:`Tensor4` ``@`` :class:`Tensor4` → $(C:D)_{ijmn} = C_{ijkl}D_{klmn}$ → :class:`Tensor4`.
    """  # noqa: E501

    base_array_shape = (9, 9)
    """Shape of the base (unbatched) Kelvin matrix."""

    def __init__(self, *, tensor=None, array=None):
        if tensor is not None:
            object.__setattr__(self, "_array", jnp.asarray(tensor)[..., _T4_I, _T4_J, _T4_K, _T4_L])
        elif array is not None:
            object.__setattr__(self, "_array", _raw_array(array))
        else:
            object.__setattr__(self, "_array", jnp.zeros((9, 9)))

    @property
    def tensor(self) -> jax.Array:
        """
        Dense (3, 3, 3, 3) tensor via a 2-D gather on the Kelvin matrix.

        Returns
        -------
        jax.Array
            Shape ``(..., 3, 3, 3, 3)``.
        """
        return self._array[..., _T2_RECON33[:, :, None, None], _T2_RECON33[None, None, :, :]]

    def __matmul__(self, other):
        r"""
        Double contraction $\mathbb{C} : \boldsymbol{\varepsilon}$ or
        composition $\mathbb{C} : \mathbb{D}$.

        - ``@`` :class:`SymmetricTensor2` or :class:`Tensor2` → :class:`Tensor2`.
        - ``@`` :class:`Tensor4` → :class:`Tensor4`.
        """
        if isinstance(other, SymmetricTensor2):
            return Tensor2(array=self._array @ _sym2_to_full9(other))
        if isinstance(other, Tensor2):
            return Tensor2(array=self._array @ other._array)
        if isinstance(other, Tensor4):
            return Tensor4(array=self._array @ other._array)
        return NotImplemented

    @classmethod
    def identity(cls) -> "Tensor4":
        r"""
        Fourth-rank identity $\mathbb{I}_{ijkl} = \delta_{ik}\delta_{jl}$.

        Returns
        -------
        Tensor4
        """
        return cls(array=jnp.eye(9))


class SymmetricTensor4(_AbstractTensor4):
    r"""
    Fourth-rank tensor with minor symmetries
    $C_{ijkl} = C_{jikl} = C_{ijlk} = C_{jilk}$.

    Stored as a $(6 \times 6)$ Kelvin-Mandel matrix.  The Kelvin scaling
    ensures that the double contraction $\mathbb{C} : \boldsymbol{\varepsilon}$
    is a plain matrix-vector product on the Kelvin arrays.

    Parameters
    ----------
    tensor : array_like, shape ``(..., 3, 3, 3, 3)``, optional
        Dense tensor with minor symmetries.
    array : array_like, shape ``(..., 6, 6)``, optional
        Pre-built Kelvin matrix.

    Notes
    -----
    The ``@`` operator performs Kelvin-space products:

    - ``@`` :class:`SymmetricTensor2` → $(6,6) \cdot (6,)$ → :class:`SymmetricTensor2`.
    - ``@`` :class:`SymmetricTensor4` → $(6,6) \cdot (6,6)$ → :class:`SymmetricTensor4`.

    Use :meth:`to_symmetric` on :class:`_AbstractTensor4` subclasses to
    materialise them into this form before mixed-class operations.
    """

    base_array_shape = (6, 6)
    """Shape of the base (unbatched) Kelvin matrix."""

    def __init__(self, *, tensor=None, array=None):
        if tensor is not None:
            object.__setattr__(
                self,
                "_array",
                jnp.asarray(tensor)[..., _S4_I, _S4_J, _S4_K, _S4_L] * _S4_W,
            )
        elif array is not None:
            object.__setattr__(self, "_array", _raw_array(array))
        else:
            object.__setattr__(self, "_array", jnp.zeros((6, 6)))

    @property
    def tensor(self) -> jax.Array:
        """
        Dense (3, 3, 3, 3) tensor with minor symmetries enforced.

        Reconstructed via a 2-D gather on ``_array / _S4_W``.

        Returns
        -------
        jax.Array
            Shape ``(..., 3, 3, 3, 3)``.
        """
        return (self._array / _S4_W)[
            ..., _S2_RECON33[:, :, None, None], _S2_RECON33[None, None, :, :]
        ]

    def to_symmetric(self) -> "SymmetricTensor4":
        """
        Return ``self`` — already the fully materialised symmetric form.

        Returns
        -------
        SymmetricTensor4
        """
        return self

    def __matmul__(self, other):
        r"""
        Kelvin-space double contraction or composition.

        - ``@`` :class:`SymmetricTensor2` → $(6,6)@(6,)$ → :class:`SymmetricTensor2`.
        - ``@`` :class:`SymmetricTensor4` → $(6,6)@(6,6)$ → :class:`SymmetricTensor4`.
        """
        if isinstance(other, SymmetricTensor2):
            return SymmetricTensor2(array=self._array @ other._array)
        if isinstance(other, SymmetricTensor4):
            return SymmetricTensor4(array=self._array @ other._array)
        other_arr = _array4(other)
        if other_arr.shape[-1] == 6:
            return SymmetricTensor2(array=self._array @ other_arr)
        if other_arr.shape[-2:] == (6, 6):
            return SymmetricTensor4(array=self._array @ other_arr)
        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, SymmetricTensor4):
            return SymmetricTensor4(array=other._array @ self._array)
        return NotImplemented

    @classmethod
    def identity(cls) -> "SymmetricTensor4":
        r"""
        Symmetric fourth-rank identity
        $\mathbb{I}^s_{ijkl} = \tfrac{1}{2}(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})$.

        Returns
        -------
        SymmetricTensor4
        """
        return cls(array=jnp.eye(6))
