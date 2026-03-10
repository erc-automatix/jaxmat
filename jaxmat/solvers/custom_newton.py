from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import lineax as lx
from jax import lax, tree_util


def newton_solve_jittable(
    f: Callable,
    x0: Any,
    args: Any,
    solver: lx.AbstractLinearSolver,
    *,
    tol: float = 1e-6,
    maxiter: int = 20,
    damping: float = 1.0,
    has_aux: bool = False,
    jac="fwd",
):
    flat_leaves, treedef = tree_util.tree_flatten(x0)
    leaf_shapes = [leaf.shape for leaf in flat_leaves]
    leaf_sizes = [leaf.size for leaf in flat_leaves]

    def vec_to_pytree(vec):
        parts = []
        idx = 0
        for shape, size in zip(leaf_shapes, leaf_sizes):
            part = vec[idx : idx + size].reshape(shape)
            parts.append(part)
            idx += size
        return tree_util.tree_unflatten(treedef, parts)

    def flat_f(vec):
        x = vec_to_pytree(vec)
        if has_aux:
            fx, aux = f(x, args)
        else:
            fx = f(x, args)
            aux = None
        flat_fx, _ = tree_util.tree_flatten(fx)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in flat_fx]), aux

    fx0, aux0 = flat_f(jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves]))

    def cond_fun(state):
        i, x, fx, _, _ = state
        return jnp.logical_and(i < maxiter, jnp.linalg.norm(fx) > tol)

    def body_fun(state):
        i, x, fx, _, _ = state

        def f_no_aux(vec, args):
            return flat_f(vec)[0]

        operator = lx.JacobianLinearOperator(f_no_aux, x, jac=jac)
        solution = lx.linear_solve(operator, -fx, throw=False, solver=solver)
        dx = solution.value
        x_new = x + damping * dx
        fx_new, aux = flat_f(x_new)
        return (i + 1, x_new, fx_new, dx, aux)

    x0_vec = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])
    init_state = (0, x0_vec, fx0, jnp.zeros_like(x0_vec), aux0)
    iters, x_final, _, _, aux_final = lax.while_loop(cond_fun, body_fun, init_state)
    # jax.debug.print("Converged in {} iterations", iters)
    result = vec_to_pytree(x_final)
    return (result, aux_final) if has_aux else result
