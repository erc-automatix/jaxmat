from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.utils import enforce_dtype


class VoceHardening(eqx.Module):
    r"""
    Voce hardening model for stress-strain behavior.

    $$
    \sigma_Y(p)=\sigma_0 + (\sigma_\text{u}-\sigma_0)(1-\exp(-bp))
    $$

    .. admonition:: References
        :class: seealso

        - Voce, E. (1955). "A Practical Strain-Hardening Function." Metallurgia, 51, 219-226.
    """

    sig0: float = enforce_dtype()
    r"""Initial yield stress $\sigma_0$."""
    sigu: float = enforce_dtype()
    r"""Saturation stress at large strains $\sigma_\text{u}$."""
    b: float = enforce_dtype()
    r"""Rate of hardedning $b$."""

    def __call__(self, p):
        r"""Compute the yield stress $\sigma_Y(p)$ for a given plastic strain $p$."""
        return self.sig0 + (self.sigu - self.sig0) * (1 - jnp.exp(-1.0 * self.b * p))


class NortonFlow(eqx.Module):
    r"""A Norton viscoplastic flow with overstress.

    $$\dot{\beps}^\text{vp} = \left\langle\dfrac{f(\bsig) - \sigma_y}{K}\right\rangle_+^m$$

    where $f(\bsig)-\sigma_y$ is the overstress, $\langle \cdot\rangle_+$ is the positive part.
    """

    K: float = enforce_dtype()
    """Characteristic stress $K$ of the Norton flow."""
    m: float = enforce_dtype()
    """Norton power-law exponent"""

    def __call__(self, overstress):
        return jnp.maximum(overstress / self.K, 0) ** self.m


class AbstractKinematicHardening(eqx.Module):
    """An abstract module for Armstrong-Frederic type kinematic hardening."""

    nvars: eqx.AbstractVar[int]
    """The number of kinematic hardening variables"""

    @abstractmethod
    def __call__(self, X, *args):
        r"""Returns the expression for $\dot{\bX}$ as a function of the
        backstress $\bX$ and any other variables."""
        pass

    def sig_eff(self, sig, X):
        r"""Effective stress $\bsig-\sum_i \bX_i$ where $\bX_i$ is the $i$-th backstress."""
        return sig - jnp.sum(X, axis=0)


class LinearKinematicHardening(eqx.Module):
    r"""
    Linear kinematic hardening model.

    $$\dot{\bX} = \dfrac{2}{3}H\dot{\bepsp}$$

    .. admonition:: References
        :class: seealso

        Prager, W. (1956). A new method of analyzing stresses and strains in work-hardening
        plastic solids.
    """

    H: float = enforce_dtype()
    """Linear kinematic hardening modulus"""
    nvars = 1

    @abstractmethod
    def __call__(self, eps_dot):
        r"""Returns the expression for $\dot{\bX}$ as a function of the
        backstress $\bX$ and any other variables."""
        return 2 / 3 * self.H * eps_dot

    def sig_eff(self, sig, X):
        r"""Effective stress $\bsig-\sum_i \bX_i$ where $\bX_i$ is the $i$-th backstress."""
        return sig - X


class ArmstrongFrederickHardening(AbstractKinematicHardening):
    r"""
    Armstrong-Frederick kinematic hardening model.

    .. admonition:: References
        :class: seealso

        - Armstrong, P. J., & Frederick, C. O. (1966).
            "A Mathematical Representation of the Multiaxial Bauschinger Effect for
            Hardening Materials." CEGB Report RD/B/N731.
        - Chaboche, J. L. (1991). On some modifications of kinematic hardening to
            improve the description of ratchetting effects. International journal
            of plasticity, 7(7), 661-678.
    """

    C: jax.Array = enforce_dtype()
    """Kinematic hardening modulus"""
    gamma: jax.Array = enforce_dtype()
    """Nonlinear recall modulus"""
    nvars = 2

    def __call__(self, X, p_dot, epsp_dot):
        r"""Returns the backstress variables  $\dot{\bX}$:

        $$\dot{\bX}_i = 2/3C_i\dot{\bepsp} - \gamma_i X_i \dot{p}$$
        """

        def evolution(X, C, gamma):
            return epsp_dot * (2 / 3 * C) - X * gamma * p_dot

        return jax.vmap(evolution)(X, self.C, self.gamma)

    def sig_eff(self, sig, X):
        r"""Effective stress is here:

        $$\bsig-\frac{2}{3}C\sum_{i=1}^\text{nvars}a_i$$
        """
        return (sig - jnp.sum(X, axis=0)).sym
