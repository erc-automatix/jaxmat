from abc import abstractmethod

import equinox as eqx
import jax
import lineax as lx
import optimistix as optx
from optax.tree_utils import tree_add, tree_scale, tree_zeros_like

import jaxmat.materials as jm
from jaxmat.solvers import NewtonTrustRegion


class GeneralizedStandardMaterial(jm.SmallStrainBehavior):
    r"""Generalized Standard Material (GSM) in small strain formulation.

    The GSM framework provides a unified thermodynamic formulation for
    viscoelastic, viscoplastic, and other dissipative material behaviors.
    It is characterized by two potentials:

    - A **free energy** $\Psi(\beps,\balpha)$ defining the
      elastic (reversible) response.
    - A **dissipation pseudo-potential** $\Phi(\dot{\balpha})$ defining the
      evolution of the internal variables (irreversible processes).

    The state variable evolution is obtained by minimizing the **incremental potential**:

    $$\min_{\Delta \balpha} J(\Delta\balpha ;\beps_{n+1},\balpha_n) = \min_{\Delta \balpha} \Psi(\beps_{n+1}, \balpha_n + \Delta \balpha)
    + \Delta t \Phi\left(\dfrac{\Delta \balpha}{\Delta t}\right)$$

    subject to the current strain $\beps_{n+1} = \beps_n+\Delta\beps$ and time increment $\Delta t$.

    The minimization is performed numerically using a Newton line-search solver.

    Important
    ---------
    This is an abstract behavior which must concretely implement ``make_internal_state`` as the structure of internal state variables is not
    known before hand.

    Notes
    -----
    The stress is automatically computed via
        $$\bsig = \dfrac{\partial \Psi}{\partial \beps}$$


    .. admonition:: References
        :class: seealso

        - Halphen, B., & Nguyen, Q. S. (1975). Sur les matériaux standard généralisés.
          Journal de mécanique, 14(1), 39-63.
        - Ortiz, M., & Stainier, L. (1999). The variational formulation of viscoplastic
          constitutive updates. Computer methods in applied mechanics and engineering,
          171(3-4), 419-444.
    """

    free_energy: eqx.Module
    r"""Module defining the Helmholtz free energy $\Psi(\beps,\balpha)"""
    dissipation_potential: eqx.Module
    r"""Module defining the dissipation pseudo-potential $\Phi(\dot\balpha)$"""
    minimisation_solver = NewtonTrustRegion(
        rtol=1e-6, atol=1e-6, linear_solver=lx.AutoLinearSolver(well_posed=False)
    )
    """Minimisation solver used to minimize the incremental potential."""

    @abstractmethod
    def make_internal_state(self):
        """Create internal state variables."""
        pass

    def incremental_potential(self, d_isv, args):
        r"""Compute the incremental potential for a given internal variable increment.

        Parameters
        ----------
        d_isv : PyTree
            Increment of internal state variables $\Delta \balpha$.
        args : tuple
            Tuple containing ``(eps, state, dt)``:
              - ``eps`` : total strain tensor.
              - ``state`` : current material state.
              - ``dt`` : time increment.
        """
        eps, state, dt = args
        isv_old = state.internal
        isv = tree_add(isv_old, d_isv)  # compute alpha_{n+1} = alpha_n + Delta alpha
        isv_dot = tree_scale(1 / dt, d_isv)  # compute \dot alpha = Delta alpha/Delta t
        free_eng = self.free_energy(eps, isv)
        diss_eng = dt * self.dissipation_potential(isv_dot)
        return free_eng + diss_eng

    def constitutive_update(self, eps, state, dt):
        isv_old = state.internal
        d_isv0 = tree_zeros_like(isv_old)
        args = eps, state, dt
        sol = optx.minimise(
            self.incremental_potential,
            self.minimisation_solver,
            d_isv0,
            args,
            throw=False,
        )
        d_isv = sol.value
        isv = tree_add(isv_old, d_isv)
        sig = jax.jacfwd(self.free_energy, argnums=0)(eps, isv)
        new_state = state.update(stress=sig, internal=isv)
        return sig, new_state
