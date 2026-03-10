import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.tensors import SymmetricTensor2, Tensor2


class AbstractState(eqx.Module):
    """Abstract base class representing mechanical states."""

    def _resolve_aliases(self, changes):
        alias_map = getattr(self, "__alias_targets__", {})
        field_names = self.__dict__
        resolved = {}
        for k, v in changes.items():
            field_name = alias_map.get(k, k)
            if field_name in field_names:
                resolved[field_name] = v
        return resolved

    def add(self, **changes):
        """Utility method to add values such as ``new_state = state.add(stress=dsig)``."""
        valid_changes = self._resolve_aliases(changes)
        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            [getattr(self, k) + v for k, v in valid_changes.items()],
        )

    def update(self, **changes):
        """Utility method to update values such as ``new_state = state.update(stress=sig)``."""
        valid_changes = self._resolve_aliases(changes)
        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            list(valid_changes.values()),
        )


class SmallStrainState(AbstractState):
    r"""
    State representation for small-strain behaviors.

    This class stores the current strain $\beps$ and stress $\bsig$ tensors under the assumption
    of infinitesimal (small) deformations. It also supports internal state variables
    such as plastic strain, damage, or other material history quantities through
    the `internal` attribute.

    Attributes
    ----------
    internal : :class:`AbstractState`, optional
        Nested state object representing internal variables (e.g., plastic strain,
        hardening variables, etc.). Defaults to None.
    strain : :class:`SymmetricTensor2`
        Symmetric second-order strain tensor $\beps$ (small-strain assumption).
    stress : :class:`SymmetricTensor2`
        Symmetric second-order Cauchy stress tensor $\bsig$.

    Notes
    -----
    eps : :class:`SymmetricTensor2`
        Alias for `strain`, allows accessing via `state.eps`.
    sig : :class:`SymmetricTensor2`
        Alias for `stress`, allows accessing via `state.sig`.
    """

    internal: AbstractState = None
    strain: SymmetricTensor2 = eqx.field(default_factory=SymmetricTensor2)
    stress: SymmetricTensor2 = eqx.field(default_factory=SymmetricTensor2)

    # define alias targets to authorize state updates with alias names
    __alias_targets__ = {"eps": "strain", "sig": "stress"}

    @property
    def eps(self):
        """Alias for ``strain``."""
        return self.strain

    @property
    def sig(self):
        """Alias for ``stress``."""
        return self.stress


def PK1_to_PK2(F, PK1):
    """
    Convert the first Piola-Kirchhoff stress tensor (PK1) to the
    second Piola-Kirchhoff stress tensor (PK2). Enforce symmetry explicitly.
    """
    return (F.inv @ PK1).sym


def PK1_to_Cauchy(F, PK1):
    """
    Convert the first Piola-Kirchhoff stress tensor (PK1) to the
    Cauchy stress tensor. Enforce symmetry explicitly.
    """
    return (PK1 @ F.T).sym / jnp.linalg.det(F)


class FiniteStrainState(AbstractState):
    r"""
    State representation for finite-strain continuum mechanics.

    This class encapsulates the deformation gradient $\bF$ (``F``) and first Piola-Kirchhoff
    stress $\bP$ (``PK1``), along with optional internal variables. It provides convenience
    properties for converting between common stress measures: second Piola-Kirchhoff
    $\bS$ (``PK2``) and Cauchy $\bsig$ (``sig``) stresses.

    Attributes
    ----------
    internal : AbstractState, optional
        Nested internal state representing material history or additional
        constitutive information. Defaults to None.
    F : :class:`Tensor2`
        Deformation gradient $\bF$. Initialized as the identity tensor.
    PK1 : :class:`Tensor2`
        First Piola-Kirchhoff stress tensor $\bP$.

    Notes
    -----
    PK2 : :class:`SymmetricTensor2`
        Second Piola-Kirchhoff stress tensor $\bS$, computed via :func:`PK1_to_PK2`.
    sig : :class:`SymmetricTensor2`
        Cauchy stress tensor $\bsig$, computed via :func:`PK1_to_Cauchy`.
    Cauchy : :class:`SymmetricTensor2`
        Alias for ``sig``.

    """

    internal: AbstractState = None
    F: Tensor2 = eqx.field(default_factory=Tensor2.identity)
    PK1: Tensor2 = eqx.field(default_factory=Tensor2)

    @property
    def PK2(self):
        vmap_axes = 0 if self.F.tensor.ndim == 3 else None
        return eqx.filter_vmap(PK1_to_PK2, in_axes=vmap_axes, out_axes=vmap_axes)(self.F, self.PK1)

    @property
    def sig(self):
        vmap_axes = 0 if self.F.tensor.ndim == 3 else None
        return eqx.filter_vmap(PK1_to_Cauchy, in_axes=vmap_axes, out_axes=vmap_axes)(
            self.F, self.PK1
        )

    @property
    def Cauchy(self):
        return self.sig


def make_batched(module: eqx.Module, Nbatch: int) -> eqx.Module:
    """Broadcasts all leaf arrays of a single unbatched module into a batched version.

    Args:
        module: An instance of an equinox Module (e.g., `State`) with array leaves.
        Nbatch: The number of batch items to broadcast.

    Returns:
        A new instance of the same class, with each array field having shape (Nbatch, ...).
    """

    def _broadcast(x):
        x_ = jnp.asarray(x)
        return jnp.broadcast_to(x_, (Nbatch,) + x_.shape)

    batched_module = jax.tree.map(_broadcast, module)

    # Update `_batch_size` if it exists and is static
    if hasattr(module, "_batch_size"):
        if module._batch_size is None:
            batch_size = (Nbatch,)
        else:
            batch_size = (Nbatch,) + module._batch_size
        object.__setattr__(batched_module, "_batch_size", batch_size)

    return batched_module
