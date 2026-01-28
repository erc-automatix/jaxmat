import abc
from typing import ClassVar, Tuple  # noqa: UP035

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxmat.tensors import linear_algebra, mappings


class Tensor(eqx.Module):
    dim: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    base_tensor_shape: Tuple[int, ...] = eqx.field(static=True)
    base_array_shape: Tuple[int, ...] = eqx.field(static=True)

    # index maps
    tensor_indices: Tuple[jax.Array, ...] = eqx.field(
        static=True
    )  # each should be of shape base_array_shape
    weights: jax.Array = eqx.field(static=True)  # should be of shape base_array_shape

    _array: jax.Array

    def __init__(self, *, tensor=None, array=None):
        if tensor is not None:
            if tensor.shape[-self.rank :] != self.base_tensor_shape:
                raise ValueError(
                    f"Wrong tensor shape {tensor.shape[-self.rank :]} "
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
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=self.tensor + other_array)

    def __sub__(self, other):
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=self.tensor - other_array)

    def __mul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) * self.tensor)

    def __truediv__(self, other):
        return self.__class__(tensor=self.tensor / jnp.asarray(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.__class__(tensor=jnp.asarray(self) @ jnp.asarray(other))

    def __rmatmul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) @ self.tensor)

    def __neg__(self):
        return self.__class__(tensor=-self.tensor)

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
        return SymmetricTensor2(
            tensor=0.5 * (self.tensor + jnp.swapaxes(self.tensor, -1, -2))
        )

    @property
    def inv(self):
        return self.__class__(tensor=linear_algebra.inv33(self.tensor))

    @property
    def eigenvalues(self):
        eigenvalues, eigendyads = linear_algebra.eig33(self.tensor)
        return eigenvalues, jnp.asarray([SymmetricTensor2(N) for N in eigendyads])

    @property
    def T(self):
        # we transpose only the last two indices in case of a batched tensor
        return self.__class__(tensor=jnp.swapaxes(self.tensor, -1, -2))

    def double_contract(self, other):
        """Double contraction between two Tensor2 objects."""
        return jnp.tensordot(self, other, axes=([-2, -1], [-2, -1]))


class SymmetricTensor2(Tensor2):
    dim = 3
    base_array_shape = (dim * (dim + 1) // 2,)

    tensor_indices, weights = mappings.kelvin_rank2_map(3)

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

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


class SymmetricTensor4(Tensor):
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

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

    def _as_tensor(self, array):
        return enforce_minor_symmetry(super()._as_tensor(array))

    def __matmul__(self, other):
        return other.__class__(
            # tensor=jnp.tensordot(jnp.asarray(self), jnp.asarray(other).T)
            array=self.array
            @ other.array.T
        )

    @property
    def T(self):
        return self.__class__(array=jnp.swapaxes(self.array, axis1=-1, axis2=-2))

    @property
    def inv(self):
        return self.__class__(array=jnp.linalg.inv(self.array))


def _eval_basis(coeffs, basis):
    return sum([c * b for (c, b) in zip(coeffs, basis)])


class IsotropicTensor4(SymmetricTensor4):
    """
    Symmetric 4th-rank isotropic tensor with compressed storage.
    """

    _coeffs: jax.Array  # (...,2) → [a_J, a_K]

    # ---- static projectors ----
    Jk, Kk, J4, K4 = mappings.isotropic_projectors(3)

    _kelvin_basis = jnp.stack([Jk, Kk], axis=0)
    _tensor_basis = jnp.stack([J4, K4], axis=0)

    # ----------------------------

    def __init__(self, *, coeffs=None, array=None, tensor=None, kappa=None, mu=None):

        if coeffs is not None:
            coeffs = jnp.asarray(coeffs)
            if coeffs.shape[-1] != 2:
                raise ValueError("coeffs must be (...,2)")
            self._coeffs = coeffs

        elif array is not None:
            array = jnp.asarray(array)
            self._coeffs = self._project_kelvin(array)

        elif tensor is not None:
            tensor = jnp.asarray(tensor)
            self._coeffs = self._project_tensor(tensor)

        elif (kappa is not None) and (mu is not None):
            self._coeffs = jnp.stack([3 * kappa, 2 * mu], axis=-1)

        else:
            self._coeffs = jnp.zeros((2,))

        self._array = self.array

    @property
    def array(self):
        return jnp.tensordot(self._coeffs, self._kelvin_basis, axes=1)

    @property
    def tensor(self):
        return jnp.tensordot(self._coeffs, self._tensor_basis, axes=1)

    @property
    def inv(self):
        return IsotropicTensor4(coeffs=1.0 / self._coeffs)

    # ----------------------------
    # projection operators
    # ----------------------------

    def _project_kelvin(self, Ck):
        # assumes Kelvin orthonormality
        return jnp.stack(
            [
                jnp.einsum("...ij,ij->...", Ck, self.Jk),
                jnp.einsum("...ij,ij->...", Ck, self.Kk),
            ],
            axis=-1,
        )

    def _project_tensor(self, C):
        return jnp.stack(
            [
                jnp.einsum("...ijkl,ijkl->...", C, self.J4),
                jnp.einsum("...ijkl,ijkl->...", C, self.K4),
            ],
            axis=-1,
        )


# class IsotropicTensor4(SymmetricTensor4):
#     dim = 3
#     rank = 4
#     array_rank = 1
#     base_tensor_shape = (dim, dim, dim, dim)
#     base_array_shape = (2,)

#     kappa: float
#     mu: float

#     _array: jax.Array
#     _basis = jnp.stack([*mappings.isotropic_projectors(3)], axis=0)

#     def __init__(self, kappa, mu):
#         self.kappa = kappa
#         self.mu = mu
#         self._array = jnp.asarray([3 * self.kappa, 2 * self.mu])

#     @property
#     def array(self):
#         return jnp.tensordot(self._array, self._basis, axes=1)

#     def __matmul__(self, other):
#         return other.__class__(
#             tensor=jnp.tensordot(jnp.asarray(self), jnp.asarray(other).T)
#         )

#     @property
#     def inv(self):
#         return IsotropicTensor4(1 / 9 / self.kappa, 1 / 4 / self.mu)
