import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from optax.tree_utils import tree_add, tree_zeros_like

from jaxmat.state import AbstractState, make_batched
from jaxmat.tensors import SymmetricTensor2, dev
from jaxmat.tensors.utils import FischerBurmeister as FB
from jaxmat.utils import default_value, enforce_dtype

from .behavior import SmallStrainBehavior
from .elasticity import AbstractLinearElastic, LinearElasticIsotropic
from .plastic_surfaces import (
    AbstractPlasticSurface,
    vonMises,
)


class InternalState(AbstractState):
    """
    Internal state for hardening plasticity
    (:class:`vonMisesIsotropicHardening`, :class:`GeneralIsotropicHardening`).
    """

    p: jax.Array = default_value(0.0)
    """Cumulated plastic strain"""
    epsp: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())
    """Plastic strain tensor"""


class vonMisesIsotropicHardening(SmallStrainBehavior):
    r"""
    Small-strain rate-independent elastoplastic constitutive model with isotropic hardening
    and  J2 (von Mises) plastic surface. The model assumes isotropic elasticity.

    Return-mapping only requires solving a scalar non-linear equation in terms of $p$.
    """

    elasticity: LinearElasticIsotropic
    """Linear isotropic elasticity defined by Young modulus and Poisson ratio."""
    yield_stress: eqx.Module
    """Isotropic hardening law controlling the evolution of the yield surface size."""
    plastic_surface: AbstractPlasticSurface = vonMises()
    """von Mises plastic surface."""
    internal_type = InternalState

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def solve_state(deps, isv_old):
            mu = self.elasticity.mu
            sig_el = sig_old + self.elasticity.C @ deps
            sig_eq_el = self.plastic_surface(sig_el)
            n_el = self.plastic_surface.normal(sig_el)
            p_old = isv_old.p

            def residual(dp, args):
                p = p_old + dp
                yield_criterion = sig_eq_el - 3 * mu * dp - self.yield_stress(p)
                res = FB(-yield_criterion / self.elasticity.E, dp)
                return res

            dy0 = jnp.array(0.0)
            sol = optx.root_find(residual, self.solver, dy0, adjoint=self.adjoint)
            dp = sol.value

            depsp = n_el * dp
            sig = sig_old + self.elasticity.C @ (deps - dev(depsp))
            isv = isv_old.add(p=dp, epsp=depsp)
            return sig, isv

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GeneralIsotropicHardening(SmallStrainBehavior):
    r"""
    Small-strain rate-independent elastoplastic constitutive model with isotropic hardening
    and generic plastic surface. The model does not assume isotropic elasticity.

    Return-mapping requires solving a non-linear system in terms of $p$ and $\bepsp$.
    """

    elasticity: AbstractLinearElastic
    """Linear elastic model."""
    yield_stress: eqx.Module
    """Isotropic hardening law controlling the evolution of the yield surface size."""
    plastic_surface: AbstractPlasticSurface
    """Generic plastic surface."""
    internal_type = InternalState

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def eval_stress(deps, dy):
            return sig_old + self.elasticity.C @ (deps - dy.epsp)

        def solve_state(deps, y_old):
            p_old = y_old.p

            def residual(dy, args):
                dp, depsp = dy.p, dy.epsp
                sig = eval_stress(deps, dy)
                yield_criterion = self.plastic_surface(sig) - self.yield_stress(p_old + dp)
                n = self.plastic_surface.normal(sig)
                res = (
                    FB(-yield_criterion / self.elasticity.E, dp),
                    depsp - n * dp,
                )
                y = tree_add(y_old, dy)
                return (res, y)

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(residual, self.solver, dy0, has_aux=True, adjoint=self.adjoint)
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)
        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GeneralHardeningInternalState(AbstractState):
    """Internal state for the :class:`GeneralHardening` model."""

    p: float = default_value(0.0)
    """Cumulated plastic strain"""
    epsp: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())
    """Plastic strain tensor"""
    alpha: SymmetricTensor2 = eqx.field(init=False)
    r"""Kinematic hardening variables $\balpha_i$."""
    nvar: int = eqx.field(static=True, default=1)
    """Number of kinematic hardening variables."""

    def __post_init__(self):
        self.alpha = make_batched(SymmetricTensor2(), self.nvar)


class GeneralHardening(SmallStrainBehavior):
    r"""
    Small-strain rate-independent elastoplastic constitutive model with general
    combined isotropic and kinematic hardening and generic plastic surface.
    The model accounts for a single plastic surface but several kinematic hardening variables.

    Return-mapping requires solving a non-linear system in terms of $p$, $\bepsp$ and the $\balpha_i$.
    """

    elasticity: AbstractLinearElastic
    """Linear elastic model."""
    yield_stress: float = enforce_dtype()
    """Initial yield stress."""
    plastic_surface: AbstractPlasticSurface
    """Generic plastic surface."""
    combined_hardening: eqx.Module
    r"""
    Combined hardening module representing a hardening potential $\psi_\textrm{h}(\balpha,p)$. 
    Should provide two methods:

    - ``combined_hardening.dalpha(alpha, p)`` returning $\dfrac{\partial \psi_\textrm{h}}{\partial \balpha}(\balpha,p)$
    - ``combined_hardening.dp(alpha, p)`` returning $\dfrac{\partial \psi_\textrm{h}}{\partial p}(\balpha,p)$
    """
    nvar: int = eqx.field(static=True, default=1)

    def make_internal_state(self):
        return GeneralHardeningInternalState(nvar=self.nvar)

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def eval_stress(deps, dy):
            return sig_old + self.elasticity.C @ (deps - dy.epsp)

        def solve_state(deps, y_old):
            def residual(dy, args):
                dp, depsp, dalpha = dy.p, dy.epsp, dy.alpha
                y = tree_add(y_old, dy)
                p, alpha = y.p, y.alpha
                sig = eval_stress(deps, dy)
                X = self.combined_hardening.dalpha(alpha, p)
                yield_criterion = (
                    self.plastic_surface(sig, X)
                    - self.combined_hardening.dp(alpha, p)
                    - self.yield_stress
                )
                n = self.plastic_surface.normal(sig, X)
                res = (
                    FB(-yield_criterion / self.elasticity.E, dp),
                    depsp - n * dp,
                    dalpha + self.plastic_surface.dX(sig, X) * dp,
                )
                return (res, y)

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(
                residual,
                self.solver,
                dy0,
                has_aux=True,
                adjoint=self.adjoint,
                throw=False,
            )
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)
        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state
