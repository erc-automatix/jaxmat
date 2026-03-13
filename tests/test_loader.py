import jax.numpy as jnp
import pytest

import jaxmat.materials as jm
from jaxmat.loader import ImposedLoading, global_solve, stack_loadings


def test_small_strain():
    material = jm.ElasticBehavior(elasticity=jm.LinearElasticIsotropic(E=1e3, nu=0.3))

    loading1 = ImposedLoading("small_strain", epsxx=0.02, sigxy=5.0)
    loading2 = ImposedLoading("small_strain", sigxx=10.0)
    loading3 = ImposedLoading("small_strain", epsyy=0.02 * jnp.ones((10,)))

    loadings = [loading1, loading2, loading3]
    loading = stack_loadings(loadings)
    dt = 0.1

    Nbatch = len(loading)
    state = material.init_state(Nbatch)
    eps0 = state.strain
    _eps_sol, state_sol, _stats = global_solve(eps0, state, loading, material, dt)
    stress = state_sol.stress
    assert jnp.allclose(stress[0], jnp.asarray([[20.0, 5.0, 0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert jnp.allclose(stress[1], jnp.asarray([[10.0, 0.0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert jnp.allclose(stress[2], jnp.asarray([[0.0, 0.0, 0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]]))

    with pytest.raises(ValueError):
        ImposedLoading("small_strain", epsXX=0.02)


def test_finite_strain():
    mu, kappa = 7.0, 1e3
    material = jm.Hyperelasticity(jm.CompressibleNeoHookean(mu=mu, kappa=kappa))

    lamb = 2.5
    lamb_ = 1 / jnp.sqrt(lamb)
    C = jnp.asarray([lamb**2, lamb_**2, lamb_**2])
    iC = 1 / C
    J = jnp.sqrt(jnp.prod(C))

    I1 = jnp.sum(C)
    S = mu * (1 - J ** (-2 / 3) * I1 / 3 * iC)
    sig = mu / J * (C - J ** (-2 / 3) * I1 / 3)

    loading1 = ImposedLoading("finite_strain", FXX=lamb, FYY=lamb_, FZZ=lamb_)
    loading2 = ImposedLoading(
        "finite_strain",
        FXX=lamb,
        FYY=lamb_,
        FZZ=lamb_,
        FXY=0,
        FYX=0,
        FXZ=0,
        FZX=0,
        FYZ=0,
        FZY=0,
    )
    lamb_list = jnp.full((10,), lamb)
    _lamb_list = 1 / jnp.sqrt(lamb_list)
    loading3 = ImposedLoading("finite_strain", FXX=lamb_list, FYY=_lamb_list, FZZ=_lamb_list)

    loadings = stack_loadings([loading1, loading2, loading3])
    Nbatch = len(loadings)

    state = material.init_state(Nbatch)
    F0 = state.F
    dt = 0.0
    _F_sol, state_sol, _stats = global_solve(F0, state, loadings, material, dt)
    assert jnp.allclose(state_sol.PK2[0], jnp.diag(S))
    assert jnp.allclose(state_sol.Cauchy[0], jnp.diag(sig))
    assert jnp.allclose(state_sol.PK2[1], jnp.diag(S))
    assert jnp.allclose(state_sol.Cauchy[1], jnp.diag(sig))
    assert jnp.allclose(state_sol.PK2[2], jnp.diag(S))
    assert jnp.allclose(state_sol.Cauchy[2], jnp.diag(sig))
