r"""Finite-strain behaviors from small-strain ones, in the logarithmic strain space.

:class:`LogarithmicStrain` turns any small-strain behavior into a finite-strain behavior
following the modular kinematic framework of Miehe, Apel and Lambrecht (2002). The update
splits into a geometric pre-processing, the unchanged small-strain update and a geometric
post-processing:

1. From the deformation gradient $\bF$, compute the right Cauchy-Green tensor
   $\bC = \bF^\text{T}\bF$ and the Lagrangian logarithmic (Hencky) strain

   $$\bE = \frac{1}{2}\log\bC$$

2. Call the small-strain behavior with $\bE$ in place of the linearized strain. It returns
   the stress $\bT$ work-conjugate to $\bE$ and updates its internal variables, so any
   small-strain model (elasticity, plasticity, viscoelasticity...) runs unchanged in the
   logarithmic strain space.

3. Pull the stress back to the second and first Piola-Kirchhoff stresses

   $$\bS = 2\,\bT:\frac{\partial\bE}{\partial\bC}, \qquad \bP = \bF\bS$$

The derivative of the tensor logarithm, including at repeated eigenvalues, is obtained by
automatic differentiation of :func:`~jaxmat.tensors.linear_algebra.logm`, so the
pull-back holds for anisotropic models as well. As the update only depends on $\bC$, the
behavior is objective: under a superposed rotation $\bF \to \bQ\bF$, the stress $\bS$ and
the internal variables are unchanged and the Cauchy stress rotates as
$\bsig \to \bQ\bsig\bQ^\text{T}$. For a hyperelastic small-strain model of energy
$\psi(\beps)$, the result is the hyperelastic model of energy $\psi(\bE)$.

.. admonition:: References
    :class: seealso

    - Miehe, C., Apel, N., & Lambrecht, M. (2002). Anisotropic additive plasticity in the
      logarithmic strain space: modular kinematic formulation and implementation based on
      incremental minimization principles for standard materials. Computer Methods in
      Applied Mechanics and Engineering, 191(47-48), 5383-5425.
"""

import equinox as eqx
import jax

from jaxmat.state import FiniteStrainState, SmallStrainState
from jaxmat.tensors import SymmetricTensor2, Tensor2
from jaxmat.tensors.linear_algebra import logm

from .behavior import FiniteStrainBehavior, SmallStrainBehavior


def _log_strain_array(C_array):
    """Hencky strain from the right Cauchy-Green tensor, in the stored (Mandel) form."""
    C = SymmetricTensor2(array=C_array)
    return SymmetricTensor2(tensor=0.5 * logm(C)).array


class LogarithmicStrainState(FiniteStrainState):
    r"""State of a :class:`LogarithmicStrain` behavior.

    On top of $\bF$ and $\bP$, it stores the strain and stress of the small-strain behavior
    in the logarithmic strain space, which the latter needs from one step to the next.
    ``internal`` holds the internal variables of the small-strain behavior, exactly as when it
    is used directly, e.g. ``state.internal.p`` for von Mises plasticity.
    """

    log_strain: SymmetricTensor2 = eqx.field(default_factory=SymmetricTensor2)
    r"""Lagrangian logarithmic (Hencky) strain $\bE = \frac{1}{2}\log\bC$."""
    log_stress: SymmetricTensor2 = eqx.field(default_factory=SymmetricTensor2)
    r"""Stress $\bT$ work-conjugate to $\bE$, computed by the small-strain behavior."""


class LogarithmicStrain(FiniteStrainBehavior):
    r"""Finite-strain behavior running a small-strain behavior in the logarithmic strain space.

    The small-strain behavior is evaluated at the Hencky strain $\bE = \frac{1}{2}\log\bC$
    and its stress $\bT$ is pulled back as $\bS = 2\,\bT:\partial\bE/\partial\bC$.

    Its state is a :class:`LogarithmicStrainState`: ``state.internal`` holds the internal
    variables of the small-strain behavior, as when the latter is used directly, while
    ``state.log_strain`` and ``state.log_stress`` are $\bE$ and $\bT$.

    Example
    -------
    .. code-block:: python

        small = jm.vonMisesIsotropicHardening(elasticity=..., yield_stress=...)
        material = jm.LogarithmicStrain(small)
        PK1, state = material.constitutive_update(F, material.init_state(), dt)
    """

    behavior: SmallStrainBehavior
    """Small-strain behavior, evaluated in the logarithmic strain space."""

    def init_state(self, Nbatch=None):
        """Initialize the state, in the logarithmic strain space."""
        return self._init_state(LogarithmicStrainState, Nbatch)

    def make_internal_state(self):
        """Initial internal variables of the small-strain behavior."""
        return self.behavior.init_state().internal

    @staticmethod
    def log_strain(F: Tensor2) -> SymmetricTensor2:
        r"""Lagrangian logarithmic (Hencky) strain $\bE = \frac{1}{2}\log(\bF^\text{T}\bF)$."""
        return SymmetricTensor2(array=_log_strain_array((F.T @ F).sym.array))

    @eqx.filter_jit
    def constitutive_update(self, F, state, dt):
        C = (F.T @ F).sym
        E_array, pull_back = jax.vjp(_log_strain_array, C.array)
        E = SymmetricTensor2(array=E_array)
        # state of the small-strain behavior at the beginning of the step
        small_state = SmallStrainState(
            strain=state.log_strain, stress=state.log_stress, internal=state.internal
        )
        T, small_state = self.behavior.constitutive_update(E, small_state, dt)
        # the stored basis is orthonormal: the pull-back of T gives T : dE/dC directly
        (T_dE_dC,) = pull_back(T.array)
        S = SymmetricTensor2(array=2.0 * T_dE_dC)
        PK1 = F @ S
        new_state = state.update(
            F=F,
            PK1=PK1,
            log_strain=small_state.strain,
            log_stress=small_state.stress,
            internal=small_state.internal,
        )
        return PK1, new_state
