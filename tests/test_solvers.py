"""Smoke tests for every solver exposed by ``jaxmat.solvers``.

These are deliberately minimal: instantiate each solver and run one solve on a
trivial well-posed problem. Their whole purpose is to catch *import/API* breakage
across dependency versions -- exactly the ``'frozenset' object is not callable``
regression that slipped through because optimistix changed its ``verbose`` field
and the custom solvers were pinned to nothing.

If a solver cannot even instantiate-and-run on the installed optimistix, that is a
release blocker, and this test makes it loud instead of surfacing later as a cryptic
crash deep inside a GSM constitutive update.
"""

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

jax.config.update("jax_enable_x64", True)

import jaxmat.solvers as solvers  # noqa: E402


def _quadratic(x, args):
    """Root at x = 3 (for root finders) / minimum at x = 3 (for minimisers/LS)."""
    return x - 3.0


def _sq_residual(x, args):
    return x - 3.0  # least-squares residual, min at 3


def _quadratic_min(x, args):
    return jnp.sum((x - 3.0) ** 2)


# Collect solver factories exposed by the package, tagged by the optimistix
# entry point they are meant for.
ROOT_FINDERS = {}
LEAST_SQUARES = {}
MINIMISERS = {}

for _name in dir(solvers):
    _obj = getattr(solvers, _name)
    if not callable(_obj) or _name.startswith("_"):
        continue
    # our custom trust-region factories take (rtol, atol)
    if _name in ("GaussNewtonTrustRegion", "NewtonTrustRegion"):
        LEAST_SQUARES[_name] = _obj
    elif _name in ("BFGSLinearTrustRegion",):
        MINIMISERS[_name] = _obj


@pytest.mark.parametrize("name", list(LEAST_SQUARES))
def test_least_squares_solver_runs(name):
    solver = LEAST_SQUARES[name](rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(_sq_residual, solver, jnp.array([0.0]), max_steps=100, throw=False)
    assert jnp.isfinite(sol.value).all()
    assert float(jnp.abs(sol.value[0] - 3.0)) < 1e-4


@pytest.mark.parametrize("name", list(MINIMISERS))
def test_minimiser_solver_runs(name):
    solver = MINIMISERS[name](rtol=1e-8, atol=1e-8)
    sol = optx.minimise(_quadratic_min, solver, jnp.array([0.0]), max_steps=200, throw=False)
    assert jnp.isfinite(sol.value).all()
    assert float(jnp.abs(sol.value[0] - 3.0)) < 1e-3


def test_default_solvers_present_and_usable():
    """The (solver, adjoint) pair used by every behavior must exist and run."""
    assert hasattr(solvers, "DEFAULT_SOLVERS")
    solver, adjoint = solvers.DEFAULT_SOLVERS
    sol = optx.root_find(
        _quadratic, solver, jnp.array(0.0), max_steps=50, throw=False, adjoint=adjoint
    )
    assert float(jnp.abs(sol.value - 3.0)) < 1e-6


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
