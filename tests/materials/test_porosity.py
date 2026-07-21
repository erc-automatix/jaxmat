r"""
Accuracy test for ``StaticJohnsonPorosity`` (JAXMAT) against a closed-form
solution.

Static Johnson model:

.. math::

    \dot f = \left\langle \frac{3\,f\,\sigma_0\,F_{\mathrm{mot}}}{4\,\eta}
             \right\rangle_+,
    \qquad
    F_{\mathrm{mot}} = \tfrac{2}{3}\ln f - (1-f)\,\frac{p}{\sigma_0}.

The full driving force depends on ``f`` (through ``ln f`` *and* ``(1 - f)``),
so the ODE has no simple closed form. Freezing the driving force to a positive
constant ``F_cst`` reduces it to

.. math::

    \dot f = \underbrace{\frac{3\,\sigma_0\,F_{\mathrm{cst}}}{4\,\eta}}_{A}\,f
    \quad\Longrightarrow\quad f(t) = f_0\,e^{A t},

which has an exact solution. These tests reuse the *entire* integration
pipeline (``_solve_ode``, clamp, degradation, ``constitutive_update``, state
handling) and only override the porosity RHS.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2
from jaxmat.materials.porosity import StaticJohnsonPorosity

SIGMA_0 = 100.0
ETA = 75.0
F0 = 1.0e-3
F_MAX = 0.999
F_CST = 1.0          # (constant) driving force
T_TOTAL = 5.0

# Exponential growth rate  A = 3 sigma_0 F_cst / (4 eta)
A_RATE = 3.0 * SIGMA_0 * F_CST / (4.0 * ETA)

def f_exact(t):
    r"""Exact solution of :math:`\dot f = A f`, i.e. :math:`f(t)=f_0 e^{At}`."""
    return F0 * jnp.exp(A_RATE * jnp.asarray(t))


_elasticity = jm.LinearElasticIsotropic(E=200e9, nu=0.3)
BASE_MATERIAL = jm.ElasticBehavior(elasticity=_elasticity)

# ===========================================================================
#  TEST subclass: constant driving force.
#  Only the porosity ODE right-hand side is replaced, so the integration runs
#  on  f' = A f  (known exact solution).
# ===========================================================================
class _ConstantForceStaticJohnson(StaticJohnsonPorosity):
    """Test variant: replaces F_mot(f, p) by the constant ``F_CST``."""

    def porosity_rhs(self, t, f, args):  # noqa: D401  (override of the staticmethod)
        p, sigma_0, eta = args  # p intentionally ignored
        rate = 3.0 * f * sigma_0 * F_CST / (4.0 * eta)
        return jnp.maximum(rate, 0.0)


def make_material(solver_type):
    """Build the test material with the requested solver."""
    return _ConstantForceStaticJohnson(
        base_behavior=BASE_MATERIAL,
        sigma_0=SIGMA_0,
        eta=ETA,
        f0=F0,
        f_max=F_MAX,
        solver_type=solver_type,
    )

def integrate(material, eps, n_steps):
    """Return ``(ts, fs, final_state)`` after ``n_steps`` constitutive steps."""
    dt = T_TOTAL / n_steps
    state0 = material.init_state()

    @jax.jit
    def run(state0):
        def step(state, _):
            _, new_state = material.constitutive_update(eps, state, dt)
            return new_state, new_state.internal.f

        return jax.lax.scan(step, state0, xs=None, length=n_steps)

    final_state, f_hist = run(state0)
    fs = jnp.concatenate([jnp.array([F0]), f_hist])
    ts = jnp.linspace(0.0, T_TOTAL, n_steps + 1)
    return ts, fs, final_state

def _relerr(f_num, t):
    """Relative error of ``f_num`` w.r.t. the exact solution at time ``t``."""
    return float(jnp.abs(f_num - f_exact(t)) / jnp.abs(f_exact(t)))

if __name__ == "__main__":
    _EPS = SymmetricTensor2(tensor=1.0e-3 * jnp.eye(3))
    print(f"A = {A_RATE:.6f}   f_exact(T) = {float(f_exact(T_TOTAL)):.12e}\n")

    solvers = ["Euler", "Tsit5", "Kvaerno3"]
    N_list = [2, 10, 50, 100, 500]

    # ---- convergence table (relevance check, no plot) ----------------------
    header = "   N  | " + " | ".join(f"{s:>11s}" for s in solvers)
    print(header)
    print("-" * len(header))
    for N in N_list:
        row = []
        for s in solvers:
            _, fs, _ = integrate(make_material(s), _EPS, n_steps=N)
            row.append(f"{_relerr(fs[-1], T_TOTAL):.3e}")
        print(f"{N:5d} | " + " | ".join(f"{r:>11s}" for r in row))