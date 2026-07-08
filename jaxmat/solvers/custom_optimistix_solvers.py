from collections.abc import Callable

import equinox as eqx
import lineax as lx
import optimistix as optx
from jaxtyping import PyTree, Scalar

_DEFAULT_LS = lx.AutoLinearSolver(well_posed=True)


def GaussNewtonTrustRegion(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar] = optx.max_norm,
    linear_solver: lx.AbstractLinearSolver = _DEFAULT_LS,
) -> optx.AbstractLeastSquaresSolver:
    """Gauss-Newton descent globalised with a classical trust region."""
    base = optx.GaussNewton(rtol=rtol, atol=atol, norm=norm, linear_solver=linear_solver)
    return eqx.tree_at(lambda s: s.search, base, optx.ClassicalTrustRegion())


def NewtonTrustRegion(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar] = optx.max_norm,
    linear_solver: lx.AbstractLinearSolver = _DEFAULT_LS,
) -> optx.AbstractLeastSquaresSolver:
    """Full Newton descent globalised with a classical trust region.

    Derived from `optimistix.LevenbergMarquardt` but using a full Newton descent
    instead of the damped one.
    """
    base = optx.LevenbergMarquardt(rtol=rtol, atol=atol, norm=norm)
    base = eqx.tree_at(lambda s: s.descent, base, optx.NewtonDescent(linear_solver=linear_solver))
    return eqx.tree_at(lambda s: s.search, base, optx.ClassicalTrustRegion())


def BFGSLinearTrustRegion(
    rtol: float,
    atol: float,
    norm: Callable[[PyTree], Scalar] = optx.max_norm,
    use_inverse: bool = True,
) -> optx.AbstractMinimiser:
    """BFGS quasi-Newton descent globalised with a linear trust region."""
    base = optx.BFGS(rtol=rtol, atol=atol, norm=norm, use_inverse=use_inverse)
    return eqx.tree_at(lambda s: s.search, base, optx.LinearTrustRegion())
