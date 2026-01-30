import equinox as eqx
import jax.numpy as jnp
import optimistix as optx

from jaxmat.state import AbstractState
from jaxmat.tensors import SymmetricTensor2, Tensor2, dev, tr, safe_fun
from jaxmat.tensors.linear_algebra import det33 as det
from jaxmat.tensors.utils import FischerBurmeister as FB
from jaxmat.utils import default_value

from .behavior import FiniteStrainBehavior
from .elasticity import LinearElasticIsotropic
from .plastic_surfaces import AbstractPlasticSurface, vonMises


class InternalState(AbstractState):
    """Internal state for :class:`FeFpJ2Plasticity`."""

    p: float = default_value(0.0)
    """Cumulated plastic strain $p$."""
    be_bar: SymmetricTensor2 = SymmetricTensor2.identity()
    r"""Isochoric elastic left Cauchy-Green strain $\bar{\bb}^\text{e}$."""


class FeFpJ2Plasticity(FiniteStrainBehavior):
    """Material model based on https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6843"""

    elasticity: LinearElasticIsotropic
    """Isotropic elastic model."""
    yield_stress: eqx.Module
    """Isotropic hardening law controlling the evolution of the yield surface size."""
    plastic_surface: AbstractPlasticSurface = vonMises()
    """von Mises plastic surface."""
    internal_type = InternalState

    def constitutive_update(self, F, state, dt):
        F_old = state.F
        isv_old = state.internal
        be_bar_old = isv_old.be_bar
        p_old = isv_old.p
        Id = SymmetricTensor2.identity()

        def solve_state(F):
            # relative strain and elastic predictor
            f = F @ F_old.inv
            f_bar = f * safe_fun(lambda J: J ** (-1 / 3), det(f))
            be_bar_trial = (f_bar.T @ be_bar_old @ f_bar).sym

            def residual(dy, args):
                dp, be_bar = dy  # .p, dy.be_bar
                s = self.elasticity.mu * dev(be_bar)
                yield_criterion = self.plastic_surface(s) - self.yield_stress(p_old + dp)
                n = self.plastic_surface.normal(s)
                res = (
                    FB(-yield_criterion / self.elasticity.E, dp),
                    (
                        dev(be_bar - be_bar_trial)
                        + 2 * dp * tr(be_bar) / 3 * n
                        + Id * (det(be_bar) - 1)
                    ).sym,
                )
                # import jax

                # jax.debug.print("Residual = {}", res)
                return res

            # dy0 = isv_old.update(p=0, be_bar=be_bar_trial)
            dy0 = 0.0, be_bar_trial
            sol = optx.root_find(
                residual, self.solver, dy0, adjoint=self.adjoint, throw=False
            )
            return sol.value, be_bar_trial

        dy, _ = solve_state(F)
        dp, be_bar = dy
        # be_bar = dy.be_bar.sym  # enforce symmetry
        # dp = dy.p
        y = isv_old.update(p=isv_old.p + dp, be_bar=be_bar)

        s = self.elasticity.mu * dev(be_bar)
        J = det(F)
        tau = s + self.elasticity.kappa / 2 * (J**2 - 1) * Id
        P = Tensor2(tensor=tau @ (F.T).inv)

        new_state = state.update(PK1=P, internal=y)
        return P, new_state
