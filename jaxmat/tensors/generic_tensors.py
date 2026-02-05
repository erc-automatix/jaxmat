import abc
from typing import ClassVar, Tuple  # noqa: UP035

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxmat.tensors import linear_algebra, mappings


def _binary_op(self, other, op):
    """
    Apply a binary elementwise operation between two tensors.

    Parameters
    ----------
    self : Tensor
        Left operand.
    other : Tensor or array_like
        Right operand. If a Tensor, representations are aligned before
        applying the operation. Otherwise interpreted as a dense tensor.
    op : callable
        Binary operation acting on array representations.

    Returns
    -------
    Tensor
        Result with appropriate tensor type and broadcasting.
    """
    if type(self) is type(other):
        return eqx.tree_at(
            lambda t: t._array,
            self,
            op(self.array, other.array),
        )

    elif isinstance(self, type(other)):  # self is a child of other
        return eqx.tree_at(
            lambda t: t._array,
            other,
            op(self._weaken().array, other.array),
        )

    elif isinstance(other, type(self)):  # other is a child of self
        return eqx.tree_at(
            lambda t: t._array,
            self,
            op(self.array, other._weaken().array),
        )

    else:
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=op(self.tensor, other_array))


class Tensor(eqx.Module):
    """
    Base class for JAX-compatible tensors with compressed storage.

    Tensors are internally stored in an ``array`` representation for efficient
    computation and converted on demand to their dense ``tensor`` form.
    Conversions are defined by precomputed index mappings and weights.

    Supports batching, arithmetic operations, contractions, and JIT compilation.

    Parameters
    ----------
    tensor : array_like, optional
        Dense tensor of shape ``(..., base_tensor_shape)``.
    array : array_like or Tensor, optional
        Compressed representation of shape ``(..., base_array_shape)``.

    Notes
    -----
    Batch dimensions always precede tensor dimensions.
    Most arithmetic operations are performed in array space.
    """

    dim: int = eqx.field(static=True)
    """Vectorial space dimension. Currently only ``dim=3`` is supported."""
    rank: int = eqx.field(static=True)
    """Rank of the tensor."""
    base_tensor_shape: Tuple[int, ...] = eqx.field(static=True)
    """Shape of the base tensor representation. ``(dim,)*rank``."""
    base_array_shape: Tuple[int, ...] = eqx.field(static=True)
    """Shape of the base array representation."""

    # index maps
    _tensor_indices: Tuple[jax.Array, ...]  # should be of shape base_array_shape
    _weights: jax.Array  # should be of shape base_array_shape

    _array: jax.Array

    def __init__(self, *, tensor=None, array=None):
        if tensor is not None:
            if tensor.shape[(-1) * self.rank :] != self.base_tensor_shape:
                raise ValueError(
                    f"Wrong tensor shape {tensor.shape[(-1)*self.rank :]} "
                    f"<> {self.base_tensor_shape}"
                )
            self._array = self._as_array(tensor)
        elif array is not None:
            if isinstance(array, Tensor):
                _array = array.array
            else:
                _array = jnp.asarray(array)
            if _array.shape[-self.array_rank :] != self.base_array_shape:
                raise ValueError(
                    f"Wrong array shape {_array.shape[-self.array_rank:]} "
                    f"<> {self.base_array_shape}"
                )
            self._array = _array
        else:
            self._array = jnp.zeros(self.base_array_shape)

    def _as_array(self, tensor):
        tensor = jnp.asarray(tensor)
        gathered = tensor[(...,) + self._tensor_indices]
        return gathered * self._weights

    def _as_tensor(self, array):
        if isinstance(
            array, Tensor
        ):  # needed when computing tangent operators from derivatives
            array = array.array
        else:
            array = jnp.asarray(array)
        out = jnp.zeros(
            array.shape[: -self.array_rank] + self.base_tensor_shape,
            dtype=array.dtype,
        )
        return out.at[(...,) + self._tensor_indices].add(array * self._weights)

    @property
    def array(self):
        """The underlying array representation."""
        return self._array

    @property
    def tensor(self):
        """The corresponding tensor representation."""
        return self._as_tensor(self.array)

    @property
    def shape(self):
        """Shape of the underlying array representation."""
        return self.array.shape

    @property
    def tensor_shape(self):
        """Shape of the corresponding tensor representation."""
        return self.batch_shape + self.base_tensor_shape

    @property
    def batch_shape(self):
        """Shape of the batch dimensions."""
        return self.shape[: -self.array_rank]

    @property
    def array_rank(self):
        """Rank of the base array representation (e.g. 1 for 2nd-rank tensors, 2 for 4th rank tensors)."""
        return len(self.base_array_shape)

    @property
    @abc.abstractmethod
    def T(self):
        """Transposed tensor."""
        return

    def __getitem__(self, idx):
        return self.tensor[idx]

    def __jax_array__(self):
        return self.tensor

    def __add__(self, other):
        return _binary_op(self, other, lambda a, b: a + b)

    def __sub__(self, other):
        return _binary_op(self, other, lambda a, b: a - b)

    def __mul__(self, other):
        return eqx.tree_at(lambda t: t._array, self, jnp.asarray(other) * self.array)

    def __truediv__(self, other):
        return eqx.tree_at(lambda t: t._array, self, self.array / jnp.asarray(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        """If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly."""
        return self.__class__(tensor=jnp.asarray(self) @ jnp.asarray(other))

    def __rmatmul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) @ self.tensor)

    def __neg__(self):
        return self.__class__(array=-self.array)

    def __repr__(self):
        return f"{self.tensor}"

    def _weaken_with(self, other):
        return self.__class__

    def rotate(self, R):
        """Rotate the tensor by applying rotation matrix `R` to each index.

        Parameters
        ----------
        R : array_like, shape (dim, dim)
            Rotation matrix.

        Returns
        -------
        Tensor
            Rotated tensor.
        """
        T = self.tensor
        for ax in range(self.rank):
            T = jnp.moveaxis(jnp.tensordot(R, T, (1, ax)), 0, ax)
        return type(self)(tensor=T)


class Tensor2(Tensor):
    """
    Full second-rank tensor in 3D.

    Stored as a flattened 9-component array corresponding to the dense
    (3, 3) tensor. Supports standard matrix algebra.

    Notes
    -----
    The ``@`` operator overloads simple contraction.
    """

    dim = 3
    rank = 2
    base_tensor_shape = (dim, dim)
    base_array_shape = (dim**rank,)

    _tensor_indices, _weights = mappings.full_rank2_map(3)

    @classmethod
    def identity(cls):
        r"""
        Identity tensor $I_{ij}=\delta_{ij}$.

        Returns
        -------
        Tensor2
            Second-rank identity.
        """
        I_ = jnp.zeros(cls.base_array_shape)
        I_ = I_.at[: cls.dim].set(1.0)
        return cls(array=I_)

    @property
    def sym(self):
        r"""
        Symmetric part of the tensor.

        Returns
        -------
        SymmetricTensor2
            $(A + A^\text{T}) / 2$
        """
        upper_indices = jnp.concatenate(
            (jnp.arange(self.dim), jnp.arange(self.dim, self.dim**2, 2))
        )
        lower_indices = jnp.concatenate(
            (jnp.arange(self.dim), jnp.arange(self.dim + 1, self.dim**2, 2))
        )
        array = self.array[..., upper_indices]
        array_T = self.array[..., lower_indices]
        return SymmetricTensor2(
            array=0.5 * (array + array_T) * SymmetricTensor2._weights
        )

    def _weaken(self):
        return self

    @property
    def tr(self):
        """
        Trace of the tensor.

        Returns
        -------
        jax.Array
        """
        return jnp.sum(self.array[: self.dim])

    @property
    def inv(self):
        """
        Inverse tensor.

        Returns
        -------
        Tensor2
        """
        return self.__class__(tensor=linear_algebra.inv33(self.tensor))

    @property
    def eigenvalues(self):
        r"""
        Eigenvalues $\lambda_i$ and eigenprojectors $\boldsymbol{n}_i\otimes\boldsymbol{n}_i$.

        Returns
        -------
        eigenvalues : jax.Array
        projectors : sequence of SymmetricTensor2
        """
        eigenvalues, eigendyads = linear_algebra.eig33(self.tensor)
        return eigenvalues, jnp.asarray(
            [SymmetricTensor2(tensor=N) for N in eigendyads]
        )

    @property
    def T(self):
        """
        Transposed tensor.

        Returns
        -------
        Tensor
            Tensor with last two indices swapped.
        """
        upper_indices = jnp.arange(self.dim, self.dim**2, 2)
        lower_indices = jnp.arange(self.dim + 1, self.dim**2, 2)
        swapped_indices = jnp.hstack(
            [lower_indices[:, jnp.newaxis], upper_indices[:, jnp.newaxis]]
        )
        new_indices = jnp.concatenate((jnp.arange(self.dim), swapped_indices.ravel()))
        # we transpose only the last two indices in case of a batched tensor
        array_T = self.array[..., new_indices]
        return self.__class__(array=array_T)

    def double_contract(self, other):
        """Double contraction between two Tensor2 objects."""
        return jnp.tensordot(self, other, axes=([-2, -1], [-2, -1]))


class SymmetricTensor2(Tensor2):
    dim = 3
    base_array_shape = (dim * (dim + 1) // 2,)

    _tensor_indices, _weights = mappings.kelvin_rank2_map(3)

    def _weaken(self):
        """Weaken to Tensor2."""
        diag_indices = jnp.arange(self.dim)
        upper_indices = jnp.arange(self.dim, self.dim**2, 2)
        lower_indices = jnp.arange(self.dim + 1, self.dim**2, 2)
        array = jnp.zeros(self.batch_shape + Tensor2.base_array_shape)
        array = array.at[diag_indices].set(self.array[diag_indices])
        array = array.at[upper_indices].set(self.array[self.dim :] / jnp.sqrt(2.0))
        array = array.at[lower_indices].set(self.array[self.dim :] / jnp.sqrt(2.0))
        return Tensor2(array=array)

    @property
    def T(self):
        """Transposed tensor."""
        return self

    def _as_tensor(self, array):
        out = super()._as_tensor(array)
        return 0.5 * (out + jnp.swapaxes(out, axis1=-1, axis2=-2))

    def __matmul__(self, other):
        # Multiplication of symmetric tensors cannot be ensured to remain symmetric
        return Tensor2(tensor=jnp.asarray(self.tensor) @ jnp.asarray(other))

    def _weaken_with(self, other):
        if isinstance(other, self.__class__):
            return self.__class__
        return Tensor2


def _enforce_minor_symmetry(C):
    C_ij = jnp.swapaxes(C, -4, -3)  # j i k l
    C_kl = jnp.swapaxes(C, -2, -1)  # i j l k
    C_ij_kl = jnp.swapaxes(C_ij, -2, -1)  # j i l k

    return 0.25 * (C + C_ij + C_kl + C_ij_kl)


class Tensor4(Tensor):
    """
    Full fourth-rank tensor.

    Stored as a matrix-like array of shape (9, 9) corresponding to double
    index contraction. Suitable for linear operators on second-rank tensors.

    Notes
    -----
    The ``@`` operator denotes double contraction.
    """

    dim = 3
    rank = 4
    array_rank = 2
    base_tensor_shape = (dim, dim, dim, dim)
    base_array_shape = (dim**2, dim**2)

    _tensor_indices, _weights = mappings.full_rank4_map(3)

    @classmethod
    def identity(cls):
        r"""
        Fourth-rank identity operator $I_{ijkl}=\delta_{ik}\delta_{jl}$

        Returns
        -------
        Tensor4
        """
        return cls(array=jnp.eye(cls.base_array_shape[0]))

    def _as_tensor(self, array):
        return super()._as_tensor(array)

    def __matmul__(self, other):
        """`@` operator is understood as double contraction for 4th-rank tensors"""
        return other.__class__(array=self.array @ other.array)

    @property
    def T(self):
        r"""
        Transposed tensor. Transposition is understood with respect to major indices i.e.

        $$(\mathbb{C}^\text{T})_{ijkl}=\mathbb{C}_{klij}$$

        Returns
        -------
        Tensor
            Tensor with index pairs swapped.
        """
        return self.__class__(array=jnp.swapaxes(self.array, axis1=-1, axis2=-2))

    @property
    def inv(self):
        """
        Inverse tensor.

        Returns
        -------
        Tensor4
        """
        return self.__class__(array=jnp.linalg.inv(self.array))

    def fourth_contract(self, other):
        """Fourth contraction between two Tensor4 objects."""
        return jnp.tensordot(self, other, axes=([-4, -3, 2, -1], [-4, -3, -2, -1]))

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.tensor.shape})"


class SymmetricTensor4(Tensor4):
    """
    Fourth-rank tensor with minor symmetries, i.e. $C_{ijkl}=C_{jikl}=C_{ijlk}=C_{jilk}$.

    Stores only independent components as a (6, 6) matrix in 3D,
    commonly used for elasticity and constitutive operators.
    """

    dim = 3
    rank = 4
    array_rank = 2
    base_tensor_shape = (dim, dim, dim, dim)
    base_array_shape = (dim * (dim + 1) // 2, dim * (dim + 1) // 2)

    _tensor_indices, _weights = mappings.kelvin_rank4_map(3)

    def _as_tensor(self, array):
        return _enforce_minor_symmetry(super()._as_tensor(array))

    @classmethod
    def identity(cls):
        r"""
        Fourth-rank symmetric identity operator $I_{ijkl}=\frac{1}{2}(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})$

        Returns
        -------
        SymmetricTensor4
        """
        return cls(array=jnp.eye(cls.base_array_shape[0]))
