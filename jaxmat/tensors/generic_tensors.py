import abc
from typing import ClassVar, Tuple  # noqa: UP035

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxmat.tensors import linear_algebra, mappings


def _binary_op(self, other, op):
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
            op(self.weaken().array, other.array),
        )

    elif isinstance(other, type(self)):  # other is a child of self
        return eqx.tree_at(
            lambda t: t._array,
            self,
            op(self.array, other.weaken().array),
        )

    else:
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=op(self.tensor, other_array))


class Tensor(eqx.Module):
    dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    base_tensor_shape: Tuple[int, ...] = eqx.field(static=True)
    base_array_shape: Tuple[int, ...] = eqx.field(static=True)

    # index maps
    tensor_indices: Tuple[jax.Array, ...]  # should be of shape base_array_shape
    weights: jax.Array  # should be of shape base_array_shape

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
        gathered = tensor[(...,) + self.tensor_indices]
        return gathered * self.weights

    def _as_tensor(self, array):
        array = jnp.asarray(array)
        out = jnp.zeros(
            array.shape[: -self.array_rank] + self.base_tensor_shape,
            dtype=array.dtype,
        )
        return out.at[(...,) + self.tensor_indices].add(array * self.weights)

    @property
    def shape(self):
        return self.array.shape

    @property
    def tensor(self):
        return self._as_tensor(self.array)

    @property
    def tensor_shape(self):
        return self.batch_shape + self.base_tensor_shape

    @property
    @abc.abstractmethod
    def T(self):
        """Transposed tensor."""
        return

    @property
    def array(self):
        return self._array

    @property
    def batch_shape(self):
        return self.shape[: -self.array_rank]

    @property
    def array_rank(self):
        return len(self.base_array_shape)

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
        """Rotate the tensor by applying rotation matrix to each index."""
        # Use different character ranges to avoid collision (works only for rank <= 13)
        # Rotation matrices: ab, cd, ef, gh, ...
        # Tensor indices: ijkl...
        # Output indices: ijkl...

        # Generate pairs of indices (a,b), (c,d), (e,f), ...
        assert self.rank <= 13
        pairs = [(chr(97 + 2 * i), chr(97 + 2 * i + 1)) for i in range(self.rank)]

        rotation_pairs = [first + second for first, second in pairs]
        output_indices = "".join([first for first, _ in pairs])
        tensor_indices = "".join([second for _, second in pairs])

        einsum_str = (
            ",".join(rotation_pairs) + "," + tensor_indices + "->" + output_indices
        )

        rotated_tensor = jnp.einsum(einsum_str, *([R] * self.rank), self.tensor)

        return self.__class__(tensor=rotated_tensor)


class Tensor2(Tensor):
    dim = 3
    rank = 2
    base_tensor_shape = (dim, dim)
    base_array_shape = (dim**rank,)

    tensor_indices, weights = mappings.full_rank2_map(3)

    @classmethod
    def identity(cls):
        I_ = jnp.zeros(cls.base_array_shape)
        I_ = I_.at[: cls.dim].set(1.0)
        return cls(array=I_)

    @property
    def sym(self):
        upper_indices = jnp.concatenate(
            (jnp.arange(self.dim), jnp.arange(self.dim, self.dim**2, 2))
        )
        lower_indices = jnp.concatenate(
            (jnp.arange(self.dim), jnp.arange(self.dim + 1, self.dim**2, 2))
        )
        array = self.array[..., upper_indices]
        array_T = self.array[..., lower_indices]
        return SymmetricTensor2(
            array=0.5 * (array + array_T) * SymmetricTensor2.weights
        )

    def weaken(self):
        return self

    @property
    def tr(self):
        return jnp.sum(self.array[: self.dim])

    @property
    def inv(self):
        return self.__class__(tensor=linear_algebra.inv33(self.tensor))

    @property
    def eigenvalues(self):
        eigenvalues, eigendyads = linear_algebra.eig33(self.tensor)
        return eigenvalues, jnp.asarray(
            [SymmetricTensor2(tensor=N) for N in eigendyads]
        )

    @property
    def T(self):
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

    tensor_indices, weights = mappings.kelvin_rank2_map(3)

    def weaken(self):
        """Weaken to Tensor2."""
        diag_indices = jnp.arange(self.dim)
        upper_indices = jnp.arange(self.dim, self.dim**2, 2)
        lower_indices = jnp.arange(self.dim + 1, self.dim**2, 2)
        array = jnp.zeros(self.batch_shape + Tensor2.base_array_shape)
        array = array.at[diag_indices].set(self.array[diag_indices])
        array = array.at[upper_indices].set(self.array[self.dim :] / jnp.sqrt(2.0))
        array = array.at[lower_indices].set(self.array[self.dim :] / jnp.sqrt(2.0))
        return Tensor2(array=array)

    def is_symmetric(self):
        return True

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


def enforce_minor_symmetry(C):
    C_ij = jnp.swapaxes(C, -4, -3)  # j i k l
    C_kl = jnp.swapaxes(C, -2, -1)  # i j l k
    C_ij_kl = jnp.swapaxes(C_ij, -2, -1)  # j i l k

    return 0.25 * (C + C_ij + C_kl + C_ij_kl)


class Tensor4(Tensor):
    dim = 3
    rank = 4
    array_rank = 2
    base_tensor_shape = (dim, dim, dim, dim)
    base_array_shape = (dim**2, dim**2)

    tensor_indices, weights = mappings.full_rank4_map(3)

    @classmethod
    def identity(cls):
        d = cls.dim
        n = d**2
        return cls(array=jnp.eye(n))

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

    def _as_tensor(self, array):
        return super()._as_tensor(array)

    def __matmul__(self, other):
        """`@` operator is understood as double contraction for 4th-rank tensors"""
        return other.__class__(array=self.array @ other.array)

    @property
    def T(self):
        return self.__class__(array=jnp.swapaxes(self.array, axis1=-1, axis2=-2))

    @property
    def inv(self):
        return self.__class__(array=jnp.linalg.inv(self.array))


class SymmetricTensor4(Tensor4):
    dim = 3
    rank = 4
    array_rank = 2
    base_tensor_shape = (dim, dim, dim, dim)
    base_array_shape = (dim * (dim + 1) // 2, dim * (dim + 1) // 2)

    tensor_indices, weights = mappings.kelvin_rank4_map(3)

    @classmethod
    def identity(cls):
        d = cls.dim
        n = d * (d + 1) // 2
        return cls(array=jnp.eye(n))

    @classmethod
    def J(cls):
        I2 = SymmetricTensor2.identity()
        J = jnp.einsum("ij,kl->ijkl", I2, I2) / cls.dim
        return cls(tensor=J)

    @classmethod
    def K(cls):
        return cls.identity() - cls.J()

    def _as_tensor(self, array):
        return enforce_minor_symmetry(super()._as_tensor(array))

    def fourth_contract(self, other):
        """Fourth contraction between two Tensor2 objects."""
        return jnp.tensordot(self, other, axes=([-4, -3, 2, -1], [-4, -3, -2, -1]))


def _eval_basis(coeffs, basis):
    return sum([c * b for (c, b) in zip(coeffs, basis)])


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


class IsotropicTensor4(AbstractProjectedTensor4):
    """
    Symmetric 4th-rank isotropic tensor with compressed storage.
    """

    # ---- static projectors ----
    Jk, Kk, J4, K4 = mappings.isotropic_projectors(3)

    _kelvin_basis = jnp.stack([Jk, Kk], axis=0)
    _tensor_basis = jnp.stack([J4, K4], axis=0)

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

    Jk, Kak, Kbk, J4, Ka4, Kb4 = mappings.cubic_projectors(3)

    _kelvin_basis = jnp.stack([Jk, Kak, Kbk], axis=0)
    _tensor_basis = jnp.stack([J4, Ka4, Kb4], axis=0)

    def __init__(self, *, coeffs=None, kappa=None, mua=None, mub=None):

        if coeffs is None:
            if None in (kappa, mua, mub):
                raise ValueError("Provide either coeffs or (c1, c2, c3)")
            coeffs = jnp.stack([3 * kappa, 2 * mua, 2 * mub], axis=-1)

        super().__init__(coeffs=coeffs)
