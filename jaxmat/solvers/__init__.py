import lineax as lx
import optimistix as optx

from .custom_optimistix_solvers import (
    BFGSLinearTrustRegion,
    GaussNewtonTrustRegion,
    NewtonTrustRegion,
)

DEFAULT_LINEAR_SOLVER = lx.AutoLinearSolver(well_posed=True)
DEFAULT_SOLVERS = (
    optx.Newton(
        rtol=1e-8,
        atol=1e-8,
        linear_solver=DEFAULT_LINEAR_SOLVER,
    ),
    optx.ImplicitAdjoint(linear_solver=DEFAULT_LINEAR_SOLVER),
)
