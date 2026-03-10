from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import eigenvalues
from jaxmat.tensors.linear_algebra import principal_invariants, det33
from .behavior import FiniteStrainBehavior


class HyperelasticPotential(eqx.Module):
    @abstractmethod
    def __call__(self):
        pass

    def PK1(self, F):
        return jax.jacfwd(self.__call__)(F)

    def PK2(self, F):
        return (F.inv @ self.PK1(F)).sym

    def Cauchy(self, F):
        # Divide on the right rather than on the left to preserve Tensor object due to operator dispatch priority.
        return (self.PK1(F) @ F.T).sym / det33(F)


class Hyperelasticity(FiniteStrainBehavior):
    potential: HyperelasticPotential

    def constitutive_update(self, F, state, dt):
        PK1 = self.potential.PK1(F)
        new_state = state.update(PK1=PK1, F=F)
        return PK1, new_state


class VolumetricPart(eqx.Module):
    beta: float = 2.0

    def __call__(self, J):
        return 1 / self.beta**2 * (J ** (self.beta) - self.beta * jnp.log(J) - 1)


class SquaredVolumetric(eqx.Module):
    def __call__(self, J):
        return (J - 1) ** 2 / 2


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
