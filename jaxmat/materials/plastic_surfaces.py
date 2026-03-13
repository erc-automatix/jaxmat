from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxmat.tensors import SymmetricTensor2, dev, eigenvalues
from jaxmat.tensors.utils import safe_norm, safe_sqrt
from jaxmat.utils import default_value, enforce_dtype


def safe_zero(method):
    """Decorator for yield surfaces to avoid NaNs for zero stress in both fwd and bwd AD."""

    def wrapper(self, x, *args, eps=1e-16):
        x_norm = jnp.linalg.norm(x)
        x_safe = SymmetricTensor2(tensor=jnp.where(x_norm > eps, x, 0 * x))
        return jnp.where(x_norm > eps, method(self, x_safe, *args), 0.0)

    return wrapper


class AbstractPlasticSurface(eqx.Module):
    """Abstract plastic surface class."""

    @abstractmethod
    def __call__(self, sig: SymmetricTensor2, *args):
        """Yield surface expression.

        .. Tip::
            We recommend using the :func:`safe_zero` decorator on this method to avoid
            NaNs for zero stresses.

        Parameters
        ----------
        sig:
            Stress tensor.
        args:
            Additional thermodynamic forces entering the yield surface definition.
        """
        pass

    def normal(self, sig: SymmetricTensor2, *args):
        """Normal to the yield surface. Computed automatically using forward AD on :func:`__call__`.

        Args:
            sig: Stress tensor.
            args: Additional thermodynamic forces entering the yield surface definition.
        """
        return jax.jacfwd(self.__call__, argnums=0)(sig, *args)


class vonMises(AbstractPlasticSurface):
    r"""von Mises yield surface

    $$\sqrt{\dfrac{3}{2}\bs:\bs}$$

    where $\bs = \dev(\bsig)$"""

    @safe_zero
    def __call__(self, sig):
        return jnp.sqrt(3 / 2.0) * safe_norm(dev(sig))


class DruckerPrager(AbstractPlasticSurface):
    r"""Drucker-Prager yield surface

    $$\alpha I_1 + \sqrt{J_2}$$

    where $I_1=\tr(\bsig)$  is the first stress invariant,
    $J_2=\dfrac{1}{2}\bs:\bs$ is the second deviatoric invariant
    and $\alpha$ a material constant describing the slope of the conic
    yield surface (friction effects).


    Parameters
    ----------
    alpha : float
        Pressure sensitivity parameter
    """

    alpha: float = enforce_dtype()

    @safe_zero
    def __call__(self, sig):
        I1 = jnp.trace(sig)
        s = dev(sig)
        sqrt_I2 = jnp.sqrt(1 / 2.0) * safe_sqrt(jnp.vdot(s, s))
        return self.alpha * I1 + sqrt_I2


class Hosford(AbstractPlasticSurface):
    r"""Hosford yield surface

    $$\left(\dfrac{1}{2}(\lvert\sigma_\text{I}-\sigma_\text{II}\rvert^a +
    \lvert\sigma_\text{II}-\sigma_\text{III}\rvert^a +
    \lvert\sigma_\text{I}-\sigma_\text{III}\rvert^a)\right)^{1/a}$$

    with $\sigma_\text{I}$ being the stress principal values.

    Parameters
    ----------
    a : float
        Hosford shape parameter
    """

    a: float = default_value(2.0)

    @safe_zero
    def __call__(self, sig):
        sI = eigenvalues(sig)
        return (
            1
            / 2
            * (
                jnp.abs(sI[0] - sI[1]) ** self.a
                + jnp.abs(sI[0] - sI[2]) ** self.a
                + jnp.abs(sI[2] - sI[1]) ** self.a
            )
        ) ** (1 / self.a)


class Tresca(AbstractPlasticSurface):
    r"""Tresca yield surface

    $$\max_{\text{I},\text{J}}|\sigma_\text{I}-\sigma_\text{J}|$$

    with $\sigma_\text{I}$ being the stress principal values.
    """

    @safe_zero
    def __call__(self, sig):
        sI = eigenvalues(sig)
        jax.debug.print("{}", sI)
        return jnp.maximum(
            jnp.abs(sI[0] - sI[1]),
            jnp.maximum(jnp.abs(sI[0] - sI[2]), jnp.abs(sI[2] - sI[1])),
        )
