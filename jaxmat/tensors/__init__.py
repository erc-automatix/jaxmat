"""jaxmat.tensors — tensor algebra for solid mechanics."""

from .generic_tensors import (
    SymmetricTensor2,
    SymmetricTensor4,
    Tensor2,
    Tensor4,
)
from .linear_algebra import (
    det33,
    eig33,
    expm,
    inv_sqrtm,
    isotropic_function,
    logm,
    main_invariants,
    powm,
    pq_invariants,
    principal_invariants,
    sqrtm,
)
from .symmetry_classes import (
    CubicTensor4,
    IsotropicTensor4,
    TransverseIsotropicTensor4,
    cubic_projectors,
    isotropic_projectors,
    transverse_isotropic_projectors,
)
from .tensor_utils import (
    axl,
    dev,
    eigenvalues,
    norm,
    polar,
    skw,
    stretch_tensor,
    sym,
    tr,
    vol,
    von_mises,
)
from .utils import safe_fun, safe_norm, safe_sqrt

__all__ = [  # noqa: RUF022
    # rank-2 tensors
    "SymmetricTensor2",
    "Tensor2",
    # rank-4 tensors
    "SymmetricTensor4",
    "Tensor4",
    "CubicTensor4",
    "IsotropicTensor4",
    "TransverseIsotropicTensor4",
    # symmetry projectors
    "cubic_projectors",
    "isotropic_projectors",
    "transverse_isotropic_projectors",
    # tensor utilities — decompositions and invariants
    "axl",
    "dev",
    "eigenvalues",
    "norm",
    "polar",
    "skw",
    "stretch_tensor",
    "sym",
    "tr",
    "vol",
    "von_mises",
    # linear algebra — array-level (accept tensor objects via __jax_array__)
    "det33",
    "eig33",
    "expm",
    "inv_sqrtm",
    "isotropic_function",
    "logm",
    "main_invariants",
    "powm",
    "pq_invariants",
    "principal_invariants",
    "sqrtm",
    # numeric utilities
    "safe_fun",
    "safe_norm",
    "safe_sqrt",
]
