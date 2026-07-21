from abc import abstractmethod
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.tensors import eigenvalues, Tensor2, SymmetricTensor4
from jaxmat.tensors.linear_algebra import det33, principal_invariants


from .behavior import FiniteStrainBehavior


class HyperelasticPotential(eqx.Module):
    r"""Hyperelastic potential $\psi(\bF)$.

    Contains helper functions to compute Piola-Kirchhoff and Cauchy stresses."""

    @abstractmethod
    def __call__(self):
        pass

    def PK1(self, F):
        r"""First-Piola Kirchhoff stress $\bP=\dfrac{\partial \psi}{\partial \bF}$."""
        return jax.jacfwd(self.__call__)(F)

    def PK2(self, F):
        r"""Second Piola-Kirchhoff stress $\bS=\bF^{-1}\bP$.

        .. note:: Symmetry is explicitly enforced.
        """
        return (F.inv @ self.PK1(F)).sym

    def Cauchy(self, F):
        r"""Cauchy stress $\bsig=J^{-1}\bP\bF\T$.

        .. note:: Symmetry is explicitly enforced.
        """
        # Divide on the right rather than on the left to preserve Tensor object
        # due to operator dispatch priority.
        return (self.PK1(F) @ F.T).sym / det33(F)


class Hyperelasticity(FiniteStrainBehavior):
    r"""A hyperelastic finite-strain behavior based on a hyperelastic potential $\psi(\bF)$."""

    potential: HyperelasticPotential

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, F, state, dt):
        PK1 = self.potential.PK1(F)
        new_state = state.update(PK1=PK1, F=F)
        return PK1, new_state


class HyperelasticityWithEOS(Hyperelasticity):
    r"""Hyperelastic behavior combining an isochoric potential with a separate
    pressure equation of state.

    The total first Piola--Kirchhoff stress is split additively into an
    isochoric part, obtained by differentiating ``potential`` (e.g.
    :class:`AnisotropicCompressedPotential`), and a volumetric part driven by a
    user-supplied pressure law ``eos`` rather than by a volumetric energy.
    """

    eos: Callable

    def _PK1_volumetric(self, F):
        J = det33(F)
        p = self.eos(J)
        Finv_T = F.inv.T
        return (-p * J) * Finv_T

    def PK1(self, F):
        """Total PK1 = isochoric (from potential) + volumetric (from EOS)."""
        return self.potential.PK1(F) + self._PK1_volumetric(F)

    def constitutive_update(self, F, state, dt):
        PK1 = self.PK1(F)
        new_state = state.update(PK1=PK1, F=F)
        return PK1, new_state


# ===========================================================================
# Volumetric energy parts (for energy-based potentials)
# ===========================================================================


class VolumetricPart(eqx.Module):
    r"""Volumetric part $\psi_\text{vol}(J)$ of the hyperelastic potential."""

    beta: float = 2.0

    def __call__(self, J):
        return 1 / self.beta**2 * (J ** (self.beta) - self.beta * jnp.log(J) - 1)


class SquaredVolumetric(eqx.Module):
    def __call__(self, J):
        return (J - 1) ** 2 / 2


# ===========================================================================
# Isotropic potentials
# ===========================================================================


class CompressibleNeoHookean(HyperelasticPotential):
    mu: float
    kappa: float
    volumetric: eqx.Module = VolumetricPart()

    def __call__(self, F):
        C = F.T @ F
        I1, _, I3 = principal_invariants(C)
        J = jnp.sqrt(I3)
        return self.mu / 2 * (J ** (-2.0 / 3) * I1 - 3) + self.kappa * self.volumetric(J)


class CompressibleMooneyRivlin(HyperelasticPotential):
    c1: float
    c2: float
    kappa: float
    volumetric: eqx.Module = VolumetricPart()

    def __call__(self, F):
        C = F.T @ F
        I1, I2, I3 = principal_invariants(C)
        J = jnp.sqrt(I3)
        return (
            0.5 * self.c1 * (I1 - 3 - 2 * jnp.log(J))
            + 0.5 * self.c2 * (I2 - 3)
            + self.kappa * self.volumetric(J)
        )


class CompressibleGhentMooneyRivlin(HyperelasticPotential):
    c1: float
    c2: float
    Jm: float
    kappa: float
    volumetric: eqx.Module = VolumetricPart()

    def __call__(self, F):
        C = F.T @ F
        I1, I2, I3 = principal_invariants(C)
        J = jnp.sqrt(I3)
        arg = 1 - (I1 - 3 - 2 * jnp.log(J)) / self.Jm
        return (
            -0.5 * self.c1 * self.Jm * jnp.log(arg)
            + 0.5 * self.c2 * (I2 - 3)
            + self.kappa * self.volumetric(J)
        )


class CompressibleOgden(HyperelasticPotential):
    mu: jax.Array
    alpha: jax.Array
    kappa: float
    volumetric: eqx.Module = VolumetricPart()  #    SquaredVolumetric()

    def __call__(self, F):
        C = F.T @ F
        J = jnp.sqrt(det33(C))
        Cb = J ** (-2 / 3) * C
        lambCb = eigenvalues(Cb)
        return self.W_lamb(lambCb) + self.kappa * self.volumetric(J)

    def W_lamb(self, lambCb):
        alp2 = self.alpha / 2
        return jnp.sum(
            self.mu / self.alpha * (lambCb[0] ** alp2 + lambCb[1] ** alp2 + lambCb[2] ** alp2 - 3)
        )


# ===========================================================================
# Anisotropic potentials
# ===========================================================================


class AnisotropicCompressedPotential(HyperelasticPotential):
    r"""Isochoric quadratic free energy on a pressure-dependent intermediate
        (compressed) configuration, based on the model of
        "Finite-pressure and small shear anisotropic hyperelasticity: A
        phenomenological approach", Int. J. Eng. Sci. 226, 104578 (2026).

    Implements only the quadratic term of the expansion

    .. math::
        \rho_0\psi_\text{iso}(J, \tilde{E})
            = \tfrac12\, \tilde{E} : \tilde{\mathcal{C}}(J) : \tilde{E}

    where the intermediate Green--Lagrange strain is built from the
    multiplicative split :math:`F = \tilde{F}\cdot F_0(J)`:

    .. math::
        \tilde{E} = \tfrac12\big(F_0^{-\top} C\, F_0^{-1} - I\big),
        \qquad C = F^\top F .

    The volumetric part is intentionally excluded: it is meant to be supplied
    separately as an equation of state (see :class:`HyperelasticityWithEOS`).
    Parameters
    ----------
    F0_coeffs : dict[tuple[int, int], Callable]
        Coefficients of the upper-triangular transformation gradient
        :math:`F_0(J)`, keyed by 1-based ``(i, j)`` with ``i <= j``. Each value
        is a callable ``J -> scalar``. Missing diagonal entries default to 1,
        missing off-diagonal entries default to 0. ``F0`` is a full (non
        symmetric) second-rank tensor, so no Kelvin convention applies here.

    Ctilde_coeffs : dict[tuple[int, int], Callable]
        Coefficients of the isochoric stiffness :math:`\tilde{\mathcal{C}}(J)`,
        given as entries of the **Kelvin** ``(6, 6)`` array of
        ``SymmetricTensor4``, keyed by 1-based ``(I, J)`` with ``I <= J`` (upper
        triangle of the symmetric matrix). Each value is a callable
        ``J -> scalar`` returning the Kelvin coefficient (shear/coupling terms
        already carrying their :math:`\sqrt{2}` / :math:`2` factors). 
    """
    # TODO(inverse-problem): the coefficients of F0(J) and C̃(J) are currently
    # supplied as Python closures stored in `static=True` dict fields. This is
    # fine for the forward problem (evaluating the potential and its PK1), but
    # makes the model unsuitable for parameter calibration / inverse problems.
    #
    # To enable inverse identification, refactor so that the structure (which
    # (i,j) / (I,J) entries are non-zero) stays static, while the VALUES become
    # dynamic jax.Array leaves -- e.g. polynomial coefficients in (J-1):

    F0_coeffs: dict[tuple[int, int], Callable] = eqx.field(static=True)
    Ctilde_coeffs: dict[tuple[int, int], Callable] = eqx.field(static=True)

    def __init__(self, F0_coeffs, Ctilde_coeffs):
        self.F0_coeffs = dict(F0_coeffs)
        self.Ctilde_coeffs = dict(Ctilde_coeffs)
        
    def _assemble_F0(self, J):
        """Build the (upper-triangular) F0(J) as a full Tensor2 (scalar J)."""
        F0 = jnp.eye(3)
        for (i, j), fn in self.F0_coeffs.items():
            F0 = F0.at[i - 1, j - 1].set(fn(J))
        return Tensor2(tensor=F0)
    
    def _assemble_Ctilde(self, J):
        """Build C̃(J) in the Kelvin (6, 6) array (scalar J)."""
        Ck = jnp.zeros((6, 6))
        for (I, Jv), fn in self.Ctilde_coeffs.items():
            val = fn(J)
            a, b = I - 1, Jv - 1
            Ck = Ck.at[a, b].set(val)
            Ck = Ck.at[b, a].set(val)
        return SymmetricTensor4(array=Ck)

    def _Etilde(self, F, F0):
        r""":math:`\tilde{E} = \tfrac12 (F_0^{-\top} C F_0^{-1} - I)`."""
        C = F.T @ F
        F0inv = F0.inv
        Ctilde = F0inv.T @ (C @ F0inv)
        Etilde = 0.5 * (Ctilde - Tensor2.identity())
        return Etilde.sym

    def __call__(self, F):
        J = jnp.sqrt(det33(F.T @ F))
        F0 = self._assemble_F0(J)
        Ctilde = self._assemble_Ctilde(J)
        E = self._Etilde(F, F0)
        CE = Ctilde @ E
        return 0.5 * E.double_contract(CE)