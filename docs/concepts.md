# Nonlinear constitutive behaviors

## Solving nonlinear models at the structure scale

### Variational formulation and FEM context

Consider a body occupying a domain $\Omega \subset \mathbb{R}^d$ with
displacement field $\bu$.  For a small-strain setting, equilibrium in weak form
reads:

```{math}
:label: nonlinear-variational-model
\int_{\Omega} \bsig(\beps):\nabla^s \bv \,\mathrm{d}\Omega 
= \int_\Omega \boldsymbol{f}\cdot\bv \,\mathrm{d}\Omega 
+ \int_{\partial \Omega_\text{N}} \bT\cdot\bv \,\mathrm{d}S 
\quad \forall\bv \in V,
```

where:

- $\bsig(\beps)$ is the Cauchy stress,
- $\beps=\nabla^s\bu$ is the symmetric strain tensor,
- $\bv$ is a virtual displacement (test function),
- $\boldsymbol{f}$ is a body force,
- $\bT$ is a traction on the Neumann boundary $\partial\Omega_\text{N}$.

### Discretization

In a finite element (FE) context, the displacement field $\bu$ is approximated
by a discrete finite-dimensional space $V_h$, with the domain $\Omega$ being
discretized into elements $\Omega_e$. Integrals are then evaluated element-wise
using numerical quadrature at integration (Gauss) points. This yields a
discrete nonlinear system:

```{math}
\bR(\bU) = \sum_{e=1}^{N_\text{el}} \sum_{q=1}^{N_q} w_q \bB_q^T \, \bsig_q(\beps_q, \balpha_q) - \bF_\text{ext} = 0,
```

where $\bB_q$ maps nodal displacements to strains at quadrature point $q$,
$\bsig_q$ is the stress returned by the constitutive model at that quadrature
point for a strain $\beps_q(\bu)$,  $\balpha_q$ are the internal variables at
that quadrature point, $w_q$ are quadrature weights, $\bF_\text{ext}$ is the
external force vector.

To solve this nonlinear system, the FEM solver needs first to evaluate the
residual through the evaluation of $\bsig_q$. Besides, it also often relies on
a Newton-Raphson method to solve the system iteratively.

This approach requires the Jacobian of the nonlinear system $\partial
\bR/\partial \bU$. The continuous version of the Jacobian is given by the
following tangent bilinear form:

```{math}
:label: tangent-bilinear-form

a_\text{tangent}(\bu,\bv) = \int_{\Omega} \nabla^s \bu: \mathbb{C}_\text{tang}(\beps):\nabla^s \bv \dOm
```

which involves the so-called *tangent operator* $\CC_\text{tang} = \partial
\bsig/\partial\beps$ at each quadrature point.

The role of the **material model** is thus to provide these local quantities to
the FEM solver for all quadrature points. For complex materials, there is
generally no closed form expression of the stress for a given state. The stress
and new state must then be computed locally by solving the constitutive update
problem.

## Constitutive update problem

In solid mechanics, a nonlinear constitutive behavior may thus be seen as a
**black-box mapping** from strains to stresses, possibly also involving a known
**state** of internal variables that capture history and irreversible
processes. Contrary to most expositions in the literature, we also make
explicit the fact that this mapping depends on **material parameters** which
describe the chosen constitutive law (e.g. elastic stiffness, yield strength,
etc.).

Focusing in the following on the small strain case, we denote by $\bsig_n$ and
$\beps_n$ the stress and strain and by $\balpha_n$ the collection of internal
state variables at time $t_n$. For a given new value of strain $\beps_{n+1}$ at
time $t_{n+1}=t_n+\Delta t$, we aim to evaluate the new stress $\bsig_{n+1}$
and new state $\balpha_{n+1}$. Finally, we also denote by $\btheta$ the
collection of material parameters. For simplicity, we do not consider any time
variation of these parameters between $t_n$ and $t_{n+1}$.

The black-box constitutive mapping can therefore be seen as follows:

```{math}
:label: constitutive-black-box

\beps=\beps_n+\Delta\beps, \balpha_n, \btheta \longrightarrow \boxed{\text{CONSTITUTIVE RELATION}}\longrightarrow \bsig_{n+1}, \balpha_{n+1}
```

The explicit mathematical form of the constitutive relation depends on the
chosen material model. Quite frequently, internal state variables evolution is
described by a system of evolution equations such that:

```{math}
F(\beps,\balpha,\dot{\balpha})=0
```

Upon choosing a specific time-discretization scheme to solve the resulting
differential equations, a system of explicit or implicit discretized evolution
equations $F_n$ can then be formed to find the corresponding mechanical state
as follows:

```{math}
\bsig_{n+1}=\bsig(\beps_n+\Delta\beps,\balpha_{n+1}) \quad \text{s.t.}\quad F_n(\beps_n+\Delta\beps,\balpha_{n+1};\btheta;\Delta t)=0
```

For more details on the various discretization strategies and solving schemes,
we refer to {cite:p}`simo2006computational` and the [material models
gallery](models).

## Consistent tangent operator

Deriving the consistent tangent operator for each material behavior is a
tedious and error-prone task. In `jaxmat`, we use AD to automate the
computation of these derivatives in a robust and efficient manner.

In practice, the constitutive update is a function which takes as inputs a
material model, a new strain state, a previous mechanical state and a time step
i.e. with the following signature:

```{code} python
stress, new_state = constitutive_update(material, strain, state, dt)
```

As a result, the consistent tangent operator can naturally be obtained using AD
on the first output (`stress`) of `constitutive_update` with respect to the
second argument `strain`. In 3D, this operator can be represented as a 6x6
matrix. There is therefore no computational gain in using reverse-mode AD and
we thus compute the derivative with `jax.jacfwd`.

## Finite-strain extension

The same principles extend to finite strain, where equilibrium is written in
the reference configuration (total Lagrangian):

```{math}
:label: finite-strain-variational-model

\int_{\Omega} \bP(\bF):\nabla^s \bv \dOm = \int_\Omega \boldsymbol{f}\cdot\bv \dOm + \int_{\partial \Omega_\text{N}} \bT\cdot\bv \dS \quad \forall\bv \in V
```

where we use now the first Piola-Kirchhoff (PK1) stress $\bP(\bF)$ as an
implicit function of the total deformation gradient $\bF=\bI+\nabla\bu$.
Depending on the chosen material, some constitutive equations are more easily
written by changing the stress/strain measure, using for instance the second
Piola-Kirchhoff stress and the Green-Lagrange strain.

`jaxmat` typically provides the necessary helper functions to convert from one
stress measure to another.

## References

```{bibliography}
:filter: docname in docnames
```
