import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    ODETerm,
    SaveAt,
    Tsit5,
    Kvaerno3,
    Dopri5,
    Heun,
    Euler,
    diffeqsolve,
    PIDController,
)

from jaxmat.state import AbstractState, SmallStrainState
from jaxmat.tensors import SymmetricTensor2
from jaxmat.utils import enforce_dtype, default_value

from .behavior import SmallStrainBehavior


def ppart(x):
    r"""Positive part :math:`\langle x \rangle_+ = \max(x, 0)`."""
    return jnp.maximum(x, 0.0)


_SOLVERS = {
    "Tsit5": Tsit5,
    "Kvaerno3": Kvaerno3,
    "Dopri5": Dopri5,
    "Euler": Euler,
    "Heun":Heun
}


# ---------------------------------------------------------------------------
#  Internal states
# ---------------------------------------------------------------------------

class StaticPorosityInternalState(AbstractState):
    """Internal state for the static Johnson porosity model."""

    f: jax.Array = default_value(1e-4)
    """Current porosity."""
    base_state: SmallStrainState = eqx.field(default_factory=SmallStrainState)
    """State of the underlying (undegraded) behavior."""


class DynamicPorosityInternalState(AbstractState):
    r"""Internal state for the dynamic Johnson porosity model.

    Attributes
    ----------
    f : float
        Current porosity.
    a_tilde : float
        Normalised pore radius :math:`\tilde a = a / \ell_{\mathrm{dyn}}`.
    dot_a_tilde : float
        Rate of normalised pore radius.
    base_state : SmallStrainState
        Full state of the wrapped base behavior.
    """

    f: jax.Array = default_value(1e-4)
    """Current porosity."""
    a_tilde: jax.Array = default_value(0.0)
    """Normalised pore radius."""
    dot_a_tilde: jax.Array = default_value(0.0)
    """Rate of normalised pore radius."""
    base_state: SmallStrainState = eqx.field(default_factory=SmallStrainState)
    """State of the underlying (undegraded) behavior."""


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class AbstractJohnsonPorosity(SmallStrainBehavior):
    r"""Abstract base class for Johnson porosity-based damage behaviors."""

    base_behavior: SmallStrainBehavior
    """Underlying (undamaged) constitutive behavior."""
    sigma_0: float = enforce_dtype()
    r"""Reference stress :math:`\sigma_0`."""
    f0: float = enforce_dtype()
    """Initial porosity."""
    f_max: float = enforce_dtype()
    """Maximum admissible porosity."""

    def _get_solver(self):
        return _SOLVERS[self.solver_type]()

    def _solve_ode(self, term, dt, y0, args):
        """Integrate an ODE over ``[0, dt]``."""
        solver = self._get_solver()
        if self.solver_type == "Euler":
            sol = diffeqsolve(
                term, solver, t0=0.0, t1=dt, dt0=dt, y0=y0,
                args=args, saveat=SaveAt(t1=True), max_steps=1,
            )
        else:
            sol = diffeqsolve(
                term, solver, t0=0.0, t1=dt, dt0=dt, y0=y0,
                args=args, saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-6, atol=1e-6),
                max_steps=int(1e5),
            )
        return sol.ys[-1]

    def _clamp(self, f):
        """Clamp porosity to ``[f0, f_max]``."""
        return jnp.clip(f, self.f0, self.f_max)

    @staticmethod
    def degradation(f):
        r"""Degradation function :math:`g(f) = 1 - f`."""
        return 1.0 - f

class StaticJohnsonPorosity(AbstractJohnsonPorosity):
    r"""Static (quasi-static, viscous) Johnson porosity model.

    .. math::

        \dot{f} = \left\langle \frac{3\,f\,\sigma_0\,F_{\mathrm{mot}}}
        {4\,\eta} \right\rangle_+
        \quad\text{with}\quad
        F_{\mathrm{mot}} = \tfrac{2}{3}\ln f - (1 - f)\,\frac{p}{\sigma_0}
    """

    eta: float = enforce_dtype()
    r"""Viscosity parameter :math:`\eta`."""

    solver_type: str = eqx.field(static=True, default="Euler")
    """Diffrax ODE solver name."""

    # # internal_type = StaticPorosityInternalState
    # """Internal state type."""

    @staticmethod
    def driving_force(f, p, sigma_0):
        r"""Driving force :math:`F_{\mathrm{mot}} = \frac{2}{3}\ln f - (1 - f)\frac{p}{\sigma_0}`."""
        return 2.0 / 3.0 * jnp.log(f) - (1.0 - f) * p / sigma_0

    @staticmethod
    def porosity_rhs(t, f, args):
        r"""RHS of the porosity ODE: :math:`\dot{f} = \langle \frac{3 f \sigma_0 F_{\mathrm{mot}}}{4\eta} \rangle_+`."""
        p, sigma_0, eta = args
        F_mot = StaticJohnsonPorosity.driving_force(f, p, sigma_0)
        return ppart(3.0 * f * sigma_0 * F_mot / (4.0 * eta))

    def make_internal_state(self):
        base_state = self.base_behavior.init_state()
        return StaticPorosityInternalState(
            f=jnp.array(self.f0),
            base_state=base_state,
        )

    def constitutive_update(self, eps, state, dt):
        isv = state.internal

        # --- base behavior (effective, undegraded)
        sig_eff, new_base_state = self.base_behavior.constitutive_update(
            eps, isv.base_state, dt
        )

        # --- pressure (positive in tension)
        p = -jnp.trace(sig_eff) / 3.0

        # --- porosity evolution ODE
        f_new = self._solve_ode(
            ODETerm(self.porosity_rhs), dt, isv.f, (p, self.sigma_0, self.eta)
        )
        f_new = self._clamp(f_new)

        # --- degraded stress
        g = self.degradation(f_new)
        sig = SymmetricTensor2(tensor=g * sig_eff.tensor)

        new_isv = isv.update(f=f_new, base_state=new_base_state)
        new_state = state.update(strain=eps, stress=sig, internal=new_isv)
        return sig, new_state


# ---------------------------------------------------------------------------
#  Dynamic Johnson
# ---------------------------------------------------------------------------

class DynamicJohnsonPorosity(AbstractJohnsonPorosity):
    r"""Dynamic Johnson porosity model with viscous pore dynamics.

    Porosity is derived from the normalised pore radius
    :math:`\tilde a = a / \ell_{\mathrm{dyn}}` which satisfies a second-order
    ODE in dimensionless time :math:`\tilde t = t / \tau`.

    Parameters
    ----------
    eta : float
        Viscosity parameter :math:`\eta`.
    b : float
        Initial inter-pore distance.
    rho_0 : float
        Reference density :math:`\rho_0`.
    """

    eta: float = enforce_dtype()
    r"""Viscosity parameter :math:`\eta`."""
    b: float = enforce_dtype()
    """Initial inter-pore distance."""
    rho_0: float = enforce_dtype()
    r"""Reference density :math:`\rho_0`."""

    solver_type: str = eqx.field(static=True, default="Tsit5")
    """ODE solver — defaults to Tsit5 (adaptive) for the 2nd-order pore ODE."""

    # internal_type = DynamicPorosityInternalState
    # """Internal state type."""

    # ---- derived quantities -----------------------------------------------

    @property
    def tau(self):
        r"""Characteristic viscous time :math:`\tau = \eta / \sigma_0`."""
        return self.eta / self.sigma_0

    @property
    def v_0(self):
        r"""Characteristic velocity :math:`v_0 = \sqrt{\sigma_0 / \rho_0}`."""
        return jnp.sqrt(self.sigma_0 / self.rho_0)

    @property
    def l_dyn(self):
        r"""Dynamic length :math:`\ell_{\mathrm{dyn}} = \tau\, v_0`."""
        return self.tau * self.v_0

    # ---- helpers ----------------------------------------------------------

    def _f_from_a_tilde(self, a_tilde):
        """Compute porosity from normalised pore radius."""
        a0 = self.b * self.f0 ** (1.0 / 3.0)
        a_tilde_0 = a0 / self.l_dyn
        b_tilde = self.b / self.l_dyn
        return a_tilde ** 3 / (a_tilde ** 3 - a_tilde_0 ** 3 + b_tilde ** 3)

    # ---- state ------------------------------------------------------------

    def make_internal_state(self):
        base_state = self.base_behavior.init_state()
        a0 = self.b * self.f0 ** (1.0 / 3.0)
        a_tilde_0 = a0 / self.l_dyn
        return DynamicPorosityInternalState(
            f=jnp.array(self.f0),
            a_tilde=jnp.array(a_tilde_0),
            dot_a_tilde=jnp.array(0.0),
            base_state=base_state,
        )

    # ---- ODE RHS ----------------------------------------------------------

    @staticmethod
    def pore_dynamics_rhs(t, y, args):
        r"""RHS of the pore dynamics ODE (dimensionless time).

        State vector ``y = [a_tilde, dot_a_tilde]``.
        """
        p, sigma_0, f = args
        a_t, dot_a_t = y
        eps_ = 1e-10
        a_safe = jnp.maximum(a_t, eps_)

        term1 = (1.0 - f ** (1.0 / 3.0)) * a_safe
        F_steady = (
            (1.5 - 2.0 * f ** (1.0 / 3.0) + 0.5 * f ** (4.0 / 3.0))
            * dot_a_t ** 2
        )
        F_mot = (
            2.0 / 3.0 * jnp.log(f) - (1.0 - f) ** 2 * p / sigma_0
        )
        F_visc = 4.0 * ppart(dot_a_t) / a_safe * (1.0 - f)

        ddot_a_t = jnp.maximum(F_mot - F_visc - F_steady, 0.0) / term1
        return jnp.array([dot_a_t, ddot_a_t])

    # ---- constitutive update ----------------------------------------------

    def constitutive_update(self, eps, state, dt):
        isv = state.internal

        # --- base behavior (effective, undegraded)
        sig_eff, new_base_state = self.base_behavior.constitutive_update(
            eps, isv.base_state, dt
        )

        # --- pressure (positive in tension)
        p = -jnp.trace(sig_eff) / 3.0

        # --- pore dynamics ODE (dimensionless time)
        dt_tilde = dt / self.tau
        y0 = jnp.array([isv.a_tilde, isv.dot_a_tilde])
        y_new = self._solve_ode(
            ODETerm(self.pore_dynamics_rhs), dt_tilde, y0,
            (p, self.sigma_0, isv.f),
        )
        a_tilde_new, dot_a_tilde_new = y_new[0], y_new[1]
        f_new = self._clamp(self._f_from_a_tilde(a_tilde_new))

        # --- degraded stress
        g = self.degradation(f_new)
        sig = SymmetricTensor2(tensor=g * sig_eff.tensor)

        new_isv = isv.update(
            f=f_new,
            a_tilde=a_tilde_new,
            dot_a_tilde=dot_a_tilde_new,
            base_state=new_base_state,
        )
        new_state = state.update(strain=eps, stress=sig, internal=new_isv)
        return sig, new_state