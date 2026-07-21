"""
Test illustrating the implementation of a jaxmat constitutive law in FEniCSx.

A hyperelastic constitutive law built with jaxmat is
plugged into a FEniCSx variational problem through dolfinx_external_operator,
and validated against a reference solution assembled with a native UFL law.

The reference UFL implementation is borrowed from the comet-fenicsx tutorial:
https://bleyerj.github.io/comet-fenicsx/intro/hyperelasticity/hyperelasticity.html

Setup: square cross-section bar, clamped at the bottom (z=0), rigid rotation of
angle theta about the z-axis imposed at the top (z=L). Both methods share the
mesh, space, BCs, load stepping and Newton solver: ONLY the construction of
P(F) differs.

Both problems are solved with the native SNES-based dolfinx.fem.petsc.
NonlinearProblem (DOLFINx >= 0.10). The external operators (P and dP/dF) are
refreshed at each SNES iteration through the assemble_residual_with_callback
helper shipped by dolfinx-external-operator.

Authors: Paul Bouteiller and Jeremy Bleyer.
"""
import locale
locale.setlocale(locale.LC_NUMERIC, "C")

from functools import partial
import time
import numpy as np

import jax
import jax.numpy as jnp
import jaxmat.materials as jm
from jaxmat.tensors import Tensor2

jax.config.update("jax_enable_x64", True)

try:
    from mpi4py import MPI
    from petsc4py import PETSc

    import basix
    import ufl
    from ufl import (Identity, grad, tr, det, inner, derivative, variable,
                     as_matrix, cos, sin, dot, SpatialCoordinate,
                     TestFunction, TrialFunction)

    import dolfinx.mesh as dmesh
    import dolfinx.fem as fem
    from dolfinx.fem.petsc import NonlinearProblem

    from dolfinx_external_operator import (
        FEMExternalOperator,
        replace_external_operators,
        evaluate_operands,
        evaluate_external_operators,
    )
    from dolfinx_external_operator.petsc import assemble_residual_with_callback
except ImportError as exc:
    raise ImportError(
        "Running this demo requires FEniCSx (dolfinx, ufl, basix, mpi4py, "
        "petsc4py) and the dolfinx-external-operator package, which are not "
        "installed. Please install FEniCSx and dolfinx-external-operator to "
        "run this script."
    ) from exc

# =============================================================================
# 1. Parameters
# =============================================================================
E, nu = 210e3, 0.3
mu_val = E / (2.0 * (1.0 + nu))
kappa_val = E / (3.0 * (1.0 - 2.0 * nu))

L = 3.0
Nxy, Nz = 4, 12
degree = 1
quad_degree = 2

angle_max = 2 * np.pi
n_steps = 15

newton_max_iters = 25
newton_atol = 1e-10
newton_rtol = 1e-8

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "basic",
    "snes_atol": newton_atol,
    "snes_rtol": newton_rtol,
    "snes_max_it": newton_max_iters,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# =============================================================================
# 2. Mesh, space, quadrature
# =============================================================================
comm = MPI.COMM_WORLD
domain = dmesh.create_box(
    comm,
    [[-0.5, -0.5, 0.0], [0.5, 0.5, L]],
    [Nxy, Nxy, Nz],
    dmesh.CellType.hexahedron,
)
gdim = domain.geometry.dim
V = fem.functionspace(domain, ("Lagrange", degree, (gdim,)))

dx = ufl.Measure("dx", domain=domain,
                 metadata={"quadrature_scheme": "default",
                           "quadrature_degree": quad_degree})

v = TestFunction(V)
du = TrialFunction(V)
Id = Identity(gdim)

u_ufl = fem.Function(V, name="u_UFL")
u_jax = fem.Function(V, name="u_jaxmat")

# =============================================================================
# 3. COMMON boundary conditions (bottom clamp + top rotation)
# =============================================================================
def bottom(x): return np.isclose(x[2], 0.0)
def top(x):    return np.isclose(x[2], L)

bottom_dofs = fem.locate_dofs_geometrical(V, bottom)
top_dofs = fem.locate_dofs_geometrical(V, top)

u_bot = fem.Function(V)
u_top = fem.Function(V)
bcs = [fem.dirichletbc(u_bot, bottom_dofs),
       fem.dirichletbc(u_top, top_dofs)]

# Imposed displacement = rigid rotation of angle theta about z
x = SpatialCoordinate(domain)
theta = fem.Constant(domain, 0.0)
Rot = as_matrix([[cos(theta),  sin(theta), 0.0],
                 [-sin(theta), cos(theta), 0.0],
                 [0.0,         0.0,        1.0]])
rot_expr = fem.Expression(dot(Rot, x) - x, V.element.interpolation_points)

# =============================================================================
# 4a. NATIVE UFL law
# =============================================================================
mu = fem.Constant(domain, mu_val)
kappa = fem.Constant(domain, kappa_val)

def psi_neo_hookean(F):
    J = det(F)
    I1 = tr(F.T * F)
    I1bar = J ** (-2.0 / 3.0) * I1
    beta = 2
    volumetric_energy = 1 / beta**2 * (J ** (beta) - beta * ufl.ln(J) - 1)
    return mu / 2.0 * (I1bar - 3.0) + kappa * volumetric_energy


F_ufl = Id + grad(u_ufl)
R_ufl = derivative(psi_neo_hookean(F_ufl) * dx, u_ufl, v)
J_ufl = derivative(R_ufl, u_ufl, du)

# =============================================================================
# 4b. jaxmat law via external operator
# =============================================================================
potential = jm.CompressibleNeoHookean(kappa=kappa_val, mu=mu_val)
material = jm.Hyperelasticity(potential=potential)
mat_state = material.init_state()

def compute_P(F_array):                     # single F (3,3) -> PK1 (3,3)
    F_t2 = Tensor2(tensor=F_array)
    PK1, _ = material.constitutive_update(F_t2, mat_state, dt=0.0)
    return PK1.tensor

def pk1_external(derivatives):
    if derivatives == (0,):                 # P(F)
        def eval_P(F_vals):
            F_flat = jnp.array(F_vals).reshape(-1, 3, 3)
            return np.asarray(jax.vmap(compute_P)(F_flat),
                              dtype=PETSc.ScalarType).reshape(-1)
        return eval_P
    elif derivatives == (1,):               # dP/dF
        def eval_dP(F_vals):
            F_flat = jnp.array(F_vals).reshape(-1, 3, 3)
            return np.asarray(jax.vmap(jax.jacfwd(compute_P))(F_flat),
                              dtype=PETSc.ScalarType).reshape(-1)
        return eval_dP
    raise NotImplementedError(f"Order {derivatives} not implemented")

Qe = basix.ufl.quadrature_element(domain.topology.cell_name(),
                                  degree=quad_degree, value_shape=(gdim, gdim))
Q_tensor = fem.functionspace(domain, Qe)

F_jax = variable(Id + grad(u_jax))
P_op = FEMExternalOperator(F_jax, function_space=Q_tensor)
P_op.external_function = pk1_external

R_jax = inner(P_op, grad(v)) * dx
J_jax = derivative(R_jax, u_jax, du)

R_jax_exp = ufl.algorithms.expand_derivatives(R_jax)
J_jax_exp = ufl.algorithms.expand_derivatives(J_jax)
R_jax_rep, R_ops = replace_external_operators(R_jax_exp)
J_jax_rep, J_ops = replace_external_operators(J_jax_exp)

# =============================================================================
# 5. SNES NonlinearProblems 
#    UFL: no callback needed. 
#    jaxmat: external operators refreshed at each
#    SNES iteration via assemble_residual_with_callback.
# =============================================================================
problem_ufl = NonlinearProblem(
    R_ufl, u_ufl, J=J_ufl, bcs=bcs,
    petsc_options_prefix="ufl_", petsc_options=petsc_options,
)

problem_jax = NonlinearProblem(
    R_jax_rep, u_jax, J=J_jax_rep, bcs=bcs,
    petsc_options_prefix="jaxmat_", petsc_options=petsc_options,
)

def constitutive_update(R_external_ops, J_external_ops):
    evaluate_external_operators(R_external_ops, evaluate_operands(R_external_ops))
    evaluate_external_operators(J_external_ops, evaluate_operands(J_external_ops))

assemble_residual_jax = partial(
    assemble_residual_with_callback,
    problem_jax.u,          # unknown
    problem_jax.F,          # residual assembly form
    problem_jax.J,          # Jacobian assembly form
    bcs,
    constitutive_update,    # callback run before each SNES assembly
    [R_ops, J_ops],         # arguments forwarded to the callback
)
problem_jax.solver.setFunction(assemble_residual_jax, problem_jax.b)

# =============================================================================
# 6. Load stepping + comparison
# =============================================================================
err_form = fem.form(inner(u_ufl - u_jax, u_ufl - u_jax) * dx)
nrm_form = fem.form(inner(u_ufl, u_ufl) * dx)

def l2_relative_error():
    d = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    n = np.sqrt(comm.allreduce(fem.assemble_scalar(nrm_form), op=MPI.SUM))
    return d / n if n > 0 else d

if comm.rank == 0:
    print(f"{'angle(deg)':>10} | {'it_UFL':>6} | {'it_jax':>6} | {'err_L2_rel':>12}")
    print("-" * 46)


t_ufl = t_jax = 0.0
for angle in np.linspace(0.0, angle_max, n_steps + 1)[1:]:
    theta.value = float(angle)
    u_top.interpolate(rot_expr)

    t0 = time.perf_counter()
    problem_ufl.solve()
    t_ufl += time.perf_counter() - t0
    it_u = problem_ufl.solver.getIterationNumber()
    reason_u = problem_ufl.solver.getConvergedReason()

    t0 = time.perf_counter()
    problem_jax.solve()
    t_jax += time.perf_counter() - t0
    it_j = problem_jax.solver.getIterationNumber()
    reason_j = problem_jax.solver.getConvergedReason()

    if reason_u <= 0 or reason_j <= 0:
        raise RuntimeError(
            f"SNES did not converge at angle {angle*180/np.pi:.1f} deg "
            f"(UFL reason={reason_u}, jaxmat reason={reason_j})."
        )

    if comm.rank == 0:
        print(f"{angle*180/np.pi:10.1f} | {it_u:6d} | {it_j:6d} | "
              f"{l2_relative_error():12.3e}")

# =============================================================================
# 7. Summary
# =============================================================================
diff = u_ufl.x.array - u_jax.x.array
if comm.rank == 0:
    print("\n=== Summary ===")
    print(f"Final relative L2 error   : {l2_relative_error():.3e}")
    print(f"Max |u_UFL - u_jax|       : {np.max(np.abs(diff)):.3e}")
    print(f"Max displacement (UFL)    : {np.max(np.abs(u_ufl.x.array)):.3e}")
    print(f"UFL solve time    : {t_ufl:.3f} s")
    print(f"jaxmat solve time : {t_jax:.3f} s")
    print(f"jaxmat/UFL ratio  : {t_jax/t_ufl:.1f}x")