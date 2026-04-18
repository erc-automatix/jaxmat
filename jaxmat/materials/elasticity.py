from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsc

from jaxmat.tensors import (
    IsotropicTensor4,
    SymmetricTensor4,
    TransverseIsotropicTensor4,
)
from jaxmat.utils import enforce_dtype

from .behavior import SmallStrainBehavior


class AbstractLinearElastic(eqx.Module):
    """Small-strain elastic model."""

    @property
    @abstractmethod
    def C(self):
        pass

    @property
    def S(self):
        r"""4th-rank isotropic compliance tensor
        $$\mathbb{S}=\mathbb{C}^{-1}$$
        """
        return self.C.inv

    def strain_energy(self, eps):
        r"""Strain energy density

        $$\psi(\beps)=\dfrac{1}{2}\beps:\mathbb{C}:\beps$$

        Parameters
        ----------
        eps: SymmetricTensor2
            Strain tensor
        """
        return 0.5 * eps.double_contract(self.C @ eps)


class LinearElastic(AbstractLinearElastic):
    """A generic linear elastic model with custom stiffness tensor."""

    stiffness: SymmetricTensor4
    """4th-rank generic stiffness tensor"""

    @property
    def C(self) -> SymmetricTensor4:
        """Return the stiffness tensor."""
        return self.stiffness


class LinearElasticIsotropic(AbstractLinearElastic):
    """An isotropic linear elastic model."""

    E: float = enforce_dtype()
    r"""Young modulus $E$"""
    nu: float = enforce_dtype()
    r"""Poisson ratio $\nu$"""

    @property
    def C(self):
        r"""4th-rank isotropic stiffness tensor

        $$\mathbb{C}=3\kappa\mathbb{J}+2\mu\mathbb{K}$$

        where $\mathbb{J}$ and $\mathbb{K}$ are the hydrostatic and deviatoric projectors.
        """
        return IsotropicTensor4(kappa=self.kappa, mu=self.mu)

    @property
    def kappa(self):
        r"""
        Bulk modulus

        $$\kappa = \dfrac{E}{3(1-2\nu)} = \lambda +\frac{2}{3}\mu$$
        """
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def mu(self):
        r"""
        Shear modulus

        $$\mu = \dfrac{E}{2(1+\nu)}$$
        """
        return self.E / (2 * (1 + self.nu))

    @property
    def lmbda(self):
        r"""
        Lamé modulus

        $$\lambda = \dfrac{E\nu}{(1+\nu)(1-2\nu)}$$
        """
        return self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)


class LinearElasticTransverseIsotropic(AbstractLinearElastic):
    """A transversely isotropic linear elastic model."""

    axis: jax.Array = enforce_dtype()
    EL: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{L}$"""
    ET: float = enforce_dtype()
    r"""Transverse Young modulus $E_{T}$"""
    nuL: float = enforce_dtype()
    r"""Longitudinal Poisson ratio $\nu_{L}$"""
    nuT: float = enforce_dtype()
    r"""Transverse Poisson ratio $\nu_{T}$"""
    muL: float = enforce_dtype()
    r"""Longitudinal shear modulus $\mu_{L}$"""

    @property
    def C(self):
        r"""4th-rank transversely isotropic stiffness tensor"""
        alpha_ = 1 / self.EL
        beta_ = (1 - self.nuT) / self.ET
        muT = self.ET / 2 / (1 + self.nuT)
        gamma_ = -jnp.sqrt(2) * self.nuL / self.EL
        Delta_ = alpha_ * beta_ - gamma_**2
        coeffs = jnp.array(
            [
                beta_ / Delta_,
                alpha_ / Delta_,
                -gamma_ / Delta_,
                -gamma_ / Delta_,
                2 * muT,
                2 * self.muL,
            ]
        )
        return TransverseIsotropicTensor4(axis=self.axis, coeffs=coeffs)


class LinearElasticOrthotropic(AbstractLinearElastic):
    """An orthotropic linear elastic model."""

    EL: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{L}$"""
    ET: float = enforce_dtype()
    r"""Transverse Young modulus $E_{T}$"""
    EN: float = enforce_dtype()
    r"""Normal Young modulus $E_{N}$"""
    nuLT: float = enforce_dtype()
    r"""Longitudinal-transverse Poisson ratio $\nu_{LT}$"""
    nuLN: float = enforce_dtype()
    r"""Longitudinal-normal Poisson ratio $\nu_{LN}$"""
    nuTN: float = enforce_dtype()
    r"""Transverse-normal Poisson ratio $\nu_{TN}$"""
    muLT: float = enforce_dtype()
    r"""Longitudinal-transverse shear modulus $\mu_{LT}$"""
    muLN: float = enforce_dtype()
    r"""Longitudinal-normal shear modulus $\mu_{LN}$"""
    muTN: float = enforce_dtype()
    r"""Transverse-normal shear modulus $\mu_{TN}$"""

    @property
    def C(self):
        r"""Build stiffness matrix for orthotropic material.

        Convention: $L=x, T=y, N=z$ in Voigt notation $[xx, yy, zz, xy, xz, yz]$
        """
        # Build compliance matrix
        S_diag = jnp.array(
            [
                [1.0 / self.EL, -(self.nuLT / self.EL), -(self.nuLN / self.EL)],
                [-(self.nuLT / self.EL), 1 / self.ET, -(self.nuTN / self.ET)],
                [-(self.nuLN / self.EL), -(self.nuTN / self.ET), 1 / self.EN],
            ]
        )

        S_shear = jnp.diag(
            jnp.array(
                [
                    1.0 / (2.0 * self.muLT),
                    1.0 / (2.0 * self.muLN),
                    1.0 / (2.0 * self.muTN),
                ]
            )
        )  # [xy, xz, yz] order
        S_Mandel = jsc.linalg.block_diag(S_diag, S_shear)
        C_Mandel = jnp.linalg.inv(S_Mandel)
        return SymmetricTensor4(array=C_Mandel)


class ElasticBehavior(SmallStrainBehavior):
    """A small strain linear elastic behavior."""

    elasticity: eqx.Module
    """The corresponding linear elastic model."""

    @eqx.filter_jit
    def constitutive_update(self, eps, state, dt):
        sig = self.elasticity.C @ eps
        new_state = state.update(strain=eps, stress=sig)
        return sig, new_state
