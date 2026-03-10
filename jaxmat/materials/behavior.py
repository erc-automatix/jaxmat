from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxmat.state import (
    AbstractState,
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)
from jaxmat.solvers import DEFAULT_SOLVERS


class AbstractBehavior(eqx.Module):
    """Abstract base class describing a mechanical behavior."""

    """Internal variables state."""
    solver: optx.AbstractRootFinder = eqx.field(static=True, init=False, default=DEFAULT_SOLVERS[0])
    """Implicit solver."""
    adjoint: optx.AbstractAdjoint = eqx.field(static=True, init=False, default=DEFAULT_SOLVERS[1])
    """Adjoint solver."""
    _batch_size: tuple = eqx.field(static=True, init=False, default=None)

    # --- Serializable internal-state class reference ---
    internal_type: type[AbstractState] = eqx.field(static=True, init=False, default=None)
    """Class (type) describing the internal-state structure (serialized with the model)."""

    # --- Required by user subclasses ---
    # @abstractmethod
    def make_internal_state(self) -> AbstractState:
        """Return a freshly constructed internal state instance."""
        pass

    # --- Unified initializer ---
    def _init_state(self, cls, Nbatch=None):
        """Initialize a full material state, optionally batched."""
        # Prefer explicitly stored internal_type (for serialization)
        if self.internal_type is not None:
            internal = self.internal_type()  # pylint: disable=E1102
        else:
            internal = self.make_internal_state()  # pylint: disable=E1111

        state = cls(internal=internal)

        if Nbatch is None and self._batch_size is None:
            return state

        Nbatch = self._batch_size[0] if Nbatch is None else Nbatch
        return make_batched(state, Nbatch)

    @abstractmethod
    def constitutive_update(self, inputs, state, dt):
        pass

    def batched_constitutive_update(self, inputs, state, dt):
        """Batched and jitted version of constitutive update along first axis of ``inputs`` and ``state``."""
        return eqx.filter_jit(eqx.filter_vmap(self.constitutive_update, in_axes=(0, 0, None)))(
            inputs, state, dt
        )


class SmallStrainBehavior(AbstractBehavior):
    """Abstract small strain behavior."""

    def init_state(self, Nbatch=None):
        """Initialize the mechanical small strain state."""
        return self._init_state(SmallStrainState, Nbatch)

    @abstractmethod
    def constitutive_update(self, eps, state, dt):
        """
        Perform the constitutive update for a given small strain increment
        for a small-strain behavior.

        This abstract method defines the interface for advancing the material
        state over a time increment based on the provided strain tensor.
        Implementations should return the updated stress tensor and internal
        variables, along with any auxiliary information required for consistent
        tangent computation or subsequent analysis.

        Parameters
        ----------
        eps : array_like
            Small strain tensor at the current integration point.
        state : PyTree
            PyTree containing the current state variables (stress, strain and internal) of the
            material.
        dt : float
            Time increment over which the update is performed.

        Returns
        -------
        stress : array_like
            Updated Cauchy stress tensor.
        new_state : PyTree
            Updated state variables after the constitutive update.

        Notes
        -----
        This method should be implemented by subclasses defining specific
        constitutive behaviors (elastic, plastic, viscoplastic, etc.).
        """
        pass


class FiniteStrainBehavior(AbstractBehavior):
    """Abstract finite strain behavior."""

    def init_state(self, Nbatch=None):
        """Initialize the mechanical finite strain state."""
        return self._init_state(FiniteStrainState, Nbatch)

    @abstractmethod
    def constitutive_update(self, F, state, dt):
        """
        Perform the constitutive update for a given deformation gradient increment
        for a finite-strain behavior.

        This abstract method defines the interface for advancing the material
        state over a time increment based on the provided strain tensor.
        Implementations should return the updated stress tensor and internal
        variables, along with any auxiliary information required for consistent
        tangent computation or subsequent analysis.

        Parameters
        ----------
        F : array_like
            Deformation gradient tensor at the current integration point.
        state : PyTree
            PyTree containing the current state variables (stress, strain and internal) of the
            material.
        dt : float
            Time increment over which the update is performed.

        Returns
        -------
        PK1 : array_like
            Updated first Piola-Kirchhoff stress tensor.
        new_state : PyTree
            Updated state variables after the constitutive update.

        Notes
        -----
        This method should be implemented by subclasses defining specific
        constitutive behaviors (elastic, plastic, viscoplastic, etc.).
        """
        pass
