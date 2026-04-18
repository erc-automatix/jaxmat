import equinox as eqx
import jax
import jax.numpy as jnp

import jaxmat.materials as jm
from jaxmat.state import AbstractState, make_batched
from jaxmat.tensors import SymmetricTensor2
from jaxmat.utils import enforce_dtype


class SLSState(AbstractState):
    """Internal state for the :class:`StandardLinearSolid` behavior."""

    epsv: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())
    r"""Viscous strain $\beps^\text{v}$."""


class StandardLinearSolid(jm.SmallStrainBehavior):
    r"""Standard Linear Solid (Zener) viscoelastic model in Maxwell representation.

    The model consists of two parallel branches:

    - A purely elastic branch with stiffness ``elasticity``.
    - A Maxwell branch containing in series:

      - An elastic component with stiffness ``maxwell_stiffness``.
      - A viscous component with viscosity ``maxwell_viscosity``.

    The total stress is given by the sum of the elastic branch and Maxwell branch stresses.
    The model captures both instantaneous and time-dependent (viscoelastic) responses.

    Notes
    -----
    The Maxwell branch defines a relaxation time $\tau = \eta / E_1$,
    where $\eta$ is the viscosity and $E_1$ is the Maxwell spring modulus.
    """

    elasticity: jm.AbstractLinearElastic
    """Elastic model for the purely elastic branch."""
    maxwell_stiffness: jm.LinearElasticIsotropic
    """Isotropic elastic model representing the spring in the Maxwell branch."""
    maxwell_viscosity: float = enforce_dtype()
    r"""Viscosity $\eta$ of the dashpot in the Maxwell branch."""
    internal_type = SLSState
    """Internal state containing the viscous strain."""

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        epsv_old = state.internal.epsv
        deps = eps - eps_old

        tau = self.maxwell_viscosity / self.maxwell_stiffness.E
        epsv_new = (
            eps + jnp.exp(-dt / tau) * (epsv_old - eps_old) - jnp.exp(-dt / 2 / tau) * deps
        ).sym

        sig = self.elasticity.C @ eps + self.maxwell_stiffness.C @ (eps - epsv_new)
        isv = state.internal.update(epsv=epsv_new)
        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GeneralizedMaxwellState(AbstractState):
    """Internal state for the :class:`GeneralizedMaxwell` viscoelastic model.

    Stores the internal viscous strains and stresses for each Maxwell branch.
    """

    epsv: SymmetricTensor2 = eqx.field(init=False)
    r"""Viscous strains $\beps^\text{v}_i$ for each Maxwell branch."""
    sigv: SymmetricTensor2 = eqx.field(init=False)
    r"""Viscoelastic stresses $\bsig^\text{v}_i$ for each Maxwell branch."""
    Nbranch: int = eqx.field(static=True, default=1)
    """Number of Maxwell branches."""

    def __post_init__(self):
        self.epsv = make_batched(SymmetricTensor2(), self.Nbranch)
        self.sigv = make_batched(SymmetricTensor2(), self.Nbranch)


class GeneralizedMaxwell(jm.SmallStrainBehavior):
    r"""Generalized Maxwell viscoelastic model (N-branch).

    Represents the stress relaxation behavior of viscoelastic materials
    using a series of Maxwell elements (spring + dashpot), each with its own
    relaxation time $\tau_i$.

    The model can be seen as a Prony series representation of stress relaxation
    function.

    Notes
    -----
    The total stress is computed as:

    $$\bsig = \bsig_\infty + \sum_{i=1}^N \bsig^\text{v}_i$$

    where in each viscous branch evolves as:

    $$\begin{align*}
    \bsig^\text{v}_i &= \CC_i:(\beps-\beps^\text{v}_i)\\
    \dot\beps^\text{v}_i &= \dfrac{1}{\tau_i}(\beps-\beps^\text{v}_i)
    \end{align*}
    $$
    """

    elasticity: jm.AbstractLinearElastic
    """Elastic model of the equilibrium (instantaneous) branch."""
    viscous_branches: jm.AbstractLinearElastic
    """Elastic stiffness models for the Maxwell branches."""
    relaxation_times: jax.Array = eqx.field(converter=jnp.asarray)
    r"""Array of relaxation times $(\tau_i)$ for the Maxwell branches."""

    def make_internal_state(self):
        return GeneralizedMaxwellState(Nbranch=len(self.relaxation_times))

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        sigv_old = state.internal.sigv
        eps_old = state.strain
        deps = eps - eps_old

        def viscous_fields(viscous_branch, relaxation_time, sigv_old):
            a = jnp.exp(-dt / relaxation_time)
            b = relaxation_time / dt * (1 - a)
            sigv = sigv_old * a + viscous_branch.C @ deps * b
            epsv = eps - viscous_branch.S @ sigv
            return epsv, sigv

        epsv_new, sigv_new = jax.vmap(viscous_fields)(
            self.viscous_branches, self.relaxation_times, sigv_old
        )
        isv = state.internal.update(epsv=epsv_new, sigv=sigv_new)
        sig = (
            self.elasticity.C @ eps + jnp.sum(sigv_new, axis=0)
        ).sym  # FIXME: make it SymmetricTensor2
        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state
