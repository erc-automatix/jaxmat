import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from optax.tree_utils import tree_add, tree_zeros_like

from jaxmat.state import (
    SmallStrainState,
    make_batched,
)
from jaxmat.tensors import SymmetricTensor2, dev
from jaxmat.utils import default_value

from .behavior import SmallStrainBehavior
from .elasticity import LinearElasticIsotropic
from .plastic_surfaces import AbstractPlasticSurface, vonMises
from .viscoplastic_flows import (
    AbstractKinematicHardening,
    ArmstrongFrederickHardening,
    NortonFlow,
)


class AFInternalState(SmallStrainState):
    """Internal state for the Armstrong-Frederick model"""

    p: float = default_value(0.0)
    """Cumulated plastic strain"""
    epsp: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())
    """Plastic strain tensor"""
    X: SymmetricTensor2 = eqx.field(default_factory=lambda: make_batched(SymmetricTensor2(), 2))
    """Backstress tensors"""


class ArmstrongFrederickViscoplasticity(SmallStrainBehavior):
    """
    Small-strain viscoplastic constitutive model with Armstrong-Frederick
    kinematic hardening, Voce isotropic hardening, and Norton-type viscous flow.

    This model represents a rate-dependent J2 (von Mises) viscoplastic material
    combining isotropic and kinematic hardening mechanisms under small strain
    assumptions. The total strain is additively decomposed into elastic and
    viscoplastic parts, and the evolution of the backstress follows the
    nonlinear Armstrong-Frederick law.

    .. note::
        - The model is suitable for cyclic loading and ratcheting simulations.
        - When the viscous exponent tends to infinity, the model approaches the
          rate-independent limit.
        - The formulation assumes small strains and isotropic material symmetry.

    .. admonition:: References
        :class: seealso

        - Armstrong, P. J., & Frederick, C. O. (1966).
            "A Mathematical Representation of the Multiaxial Bauschinger Effect for
            Hardening Materials." CEGB Report RD/B/N731.
        - Norton, F. H. (1929).
            "The Creep of Steel at High Temperatures." McGraw-Hill.
    """

    elasticity: LinearElasticIsotropic
    """Linear isotropic elasticity defined by Young modulus and Poisson ratio."""
    yield_stress: eqx.Module
    """Isotropic hardening law controlling the evolution of the yield surface size."""
    viscous_flow: NortonFlow
    """Viscoplastic flow rule following Norton (power-law) viscosity formulation."""
    kinematic_hardening: ArmstrongFrederickHardening
    """
    Kinematic hardening model defining the backstress evolution rate with
    dynamic recovery (Armstrong-Frederick formulation).
    """
    plastic_surface = vonMises()
    """J2-type yield (or loading) surface based on the deviatoric stress invariant."""
    internal_type = AFInternalState
    """Internal variables associated with the accumulated plastic strain and backstress tensor."""

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def sig_eq(sig):
            return self.plastic_surface(sig)

        def eval_stress(deps, dy):
            return sig_old + self.elasticity.C @ (deps - dev(dy.epsp))

        def solve_state(deps, y_old):
            def residual(dy, args):
                y = tree_add(y_old, dy)
                sig = eval_stress(deps, dy)
                sig_eff = self.kinematic_hardening.sig_eff(sig, y.X)
                yield_criterion = sig_eq(sig_eff) - self.yield_stress(y.p)
                n = self.plastic_surface.normal(sig_eff)
                res = (
                    dy.p - dt * self.viscous_flow(yield_criterion),
                    dy.epsp - n * dy.p,
                    (dy.X - self.kinematic_hardening(y.X, dy.p, dy.epsp))
                    / jnp.sum(self.kinematic_hardening.C),
                )
                return res, y

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(residual, self.solver, dy0, has_aux=True, adjoint=self.adjoint)
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GenericInternalState(SmallStrainState):
    """Internal state for the generic elastoviscoplastic model."""

    p: float = default_value(0.0)
    """Cumulated plastic strain"""
    epsp: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())
    """Plastic strain tensor"""
    nX: int = eqx.field(static=True, default=1)
    """Number of kinematic hardening mechanisms."""
    X: SymmetricTensor2 = eqx.field(init=False)
    """Backstress tensors"""

    def __post_init__(self):
        self.X = make_batched(SymmetricTensor2(), self.nX)


class GenericViscoplasticity(SmallStrainBehavior):
    """
    Small-strain viscoplastic constitutive model with generic yield surface, isotropic
    and kinematic hardening and viscoplastic flow rule.
    """

    elasticity: LinearElasticIsotropic
    """Linear isotropic elasticity defined by Young modulus and Poisson ratio."""
    plastic_surface: AbstractPlasticSurface
    """A generic plastic yield surface."""
    yield_stress: eqx.Module
    """Isotropic hardening law controlling the evolution of the yield surface size."""
    viscous_flow: eqx.Module
    """A generic viscoplastic flow rule."""
    kinematic_hardening: AbstractKinematicHardening
    """A generic kinematic hardening law."""

    def make_internal_state(self):
        return GenericInternalState(nX=self.kinematic_hardening.nvars)

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def sig_eq(sig):
            return self.plastic_surface(sig)

        def eval_stress(deps, dy):
            return sig_old + self.elasticity.C @ (deps - dev(dy.epsp))

        def solve_state(deps, y_old):
            def residual(dy, args):
                y = tree_add(y_old, dy)
                sig = eval_stress(deps, dy)
                sig_eff = self.kinematic_hardening.sig_eff(sig, y.X)
                yield_criterion = sig_eq(sig_eff) - self.yield_stress(y.p)
                n = self.plastic_surface.normal(sig_eff)
                res = (
                    dy.p - dt * self.viscous_flow(yield_criterion),
                    dy.epsp - n * dy.p,
                    dy.X - self.kinematic_hardening(y.X, dy.p, dy.epsp),
                )
                return res, y

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(residual, self.solver, dy0, has_aux=True, adjoint=self.adjoint)
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state
