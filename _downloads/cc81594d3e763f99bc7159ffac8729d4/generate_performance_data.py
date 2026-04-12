# %%
from time import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import jaxmat.materials as jm
from jaxmat.state import make_batched
from jaxmat.tensors import Tensor2

platform = "gpu"
with_jac = True

jax.config.update("jax_platform_name", platform)


def test_FeFp_elastoplasticity(material, with_jac=False, Nbatch=1, Nsteps=20):

    state = material.init_state(Nbatch)

    # force jitting
    F = state.F
    integrator = material.batched_constitutive_update
    if with_jac:
        integrator = eqx.filter_jit(
            eqx.filter_vmap(
                eqx.filter_jacfwd(material.constitutive_update, has_aux=True),
                in_axes=(0, 0, None),
                out_axes=(0, 0),
            )
        )
    integrator(F, state, jnp.float64(0.1))

    eps_dot = 50e-3

    steps = jnp.linspace(0, 1.0, Nsteps + 1)
    time_steps = jnp.diff(steps)
    times = np.zeros_like(time_steps)
    t = 0
    for i, dt in enumerate(time_steps):
        t += dt
        lamb = 1 + eps_dot * t

        F_ = Tensor2(tensor=jnp.diag(jnp.asarray([lamb, 1 / jnp.sqrt(lamb), 1 / jnp.sqrt(lamb)])))
        F = make_batched(F_, Nbatch)

        tic = time()
        _, state = integrator(F, state, dt)
        times[i] = time() - tic
    return times


Nsteps = 20
Nbatch_list = np.logspace(1, 5.5, 10)[::-1]


data = np.zeros((len(Nbatch_list), Nsteps + 1))

E, nu = 70e3, 0.3
elastic_model = jm.LinearElasticIsotropic(E, nu)

sig0 = 500.0
sigu = 750.0
b = 1000.0

material = jm.FeFpJ2Plasticity(elastic_model, jm.VoceHardening(sig0=sig0, sigu=sigu, b=b))
data[:, 0] = Nbatch_list
for i, Nbatch in enumerate(Nbatch_list):
    data[i, 1:] = test_FeFp_elastoplasticity(
        material, with_jac=with_jac, Nbatch=int(Nbatch), Nsteps=Nsteps
    )
    print(f"Batch size {i + 1}/{len(Nbatch_list)} : avg time = {np.mean(data[i, 1:])}")
jac_suffix = "_jac" if with_jac else "_no_jac"
np.savetxt(f"performance{jac_suffix}_{platform}.csv", data, delimiter=",")
