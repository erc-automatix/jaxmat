# Sharp bits

This section covers implementation details which are relevant to computational efficiency.

## Computational mechanics aspects

### Enforcing plastic consistency conditions

In computational plasticity, ensuring the plastic consistency condition is
crucial for accurate stress updates. Traditionally, this is done using the
*elastic predictor / plastic corrector scheme*, where an initial elastic trial
stress is corrected iteratively to satisfy the yield condition. While
effective, this approach often requires conditional flow control and makes
behavior implementation more cumbersome, especially when handling multiple
plastic surfaces.

An alternative is to enforce consistency directly using complementarity
$C$-functions such as the *Fischer-Burmeister* ($\text{FB}$) function defined
as:

$$\text{FB}(x, y) = x + y - \sqrt{x^2 + y^2}$$

This function provides a semi-smooth [^semismooth] reformulation of the
complementarity conditions:

$$x \geq 0, y \geq 0, xy = 0 \quad \Leftrightarrow \text{FB}(x, y) = 0.$$

[^semismooth]: The Fischer-Burmeister is not differentiable at the origin
    $(0,0)$, but it is semi-smooth, meaning directional derivatives exist and
    Newton-type methods can still converge robustly.

In the context of plasticity, complementarity conditions of plastic evolution
generally write:

$$
f(\bsig, p) \leq 0, \quad \dot{p} \geq 0, \quad \dot{p}f(\sigma,p) = 0
$$

where $f(\bsig, p)$ is the yield function and $p$ the cumulated plastic strain
acting as a plastic multiplier.

Using the $\text{FB} function allows to encode both elastic and plastic
evolution into a single non-linear equation:

$$\text{FB}(-f(\bsig,p),\dot{p}) = 0$$

where the negative sign ensures that $f(\bsig,p) \leq 0$ as common practice in plasticity.

The advantages are that:

1. The FB function transforms the non-smooth complementarity problem into a
   semi-smooth differentiable system, improving numerical robustness.
2. Both the yield condition and non-negativity of the plastic multiplier are
   enforced simultaneously, avoiding case disjunctions or active-set methods.
3. Newton-type solvers can be applied directly to the FB function, often
   leading to faster and more stable convergence than traditional elastic
   predictor/plastic corrector schemes.

## JAX-related sharp bits


### `optimistix` and `lineax` solver options

By default `lineax` and `optimistix` check that the solve was successful,
verifying that the return doesn't have NaNs etc. This extra check may induce
additional costs. This can be disabled by passing `solver(..., throw=False)`.

### Recompilation due to changing `dtype`

If not careful, you may induce recompilation of jitted constitutive models when
inadvertly changing `dtype`s between function calls. For instance, if calling
first:

```python
dt = 0.
material.constitutive_update(eps, state, dt)
```

and then doing:

```python
for dt in jnp.ones((10,)):
    material.constitutive_update(eps, state, dt)
```

JIT compilation will be triggered once more at the start of the loop since `dt`
changed `dtype` from `float` to `jnp.float64` for instance.
