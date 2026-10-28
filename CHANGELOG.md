# Changelog

All notable changes to jaxmat are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning aims to follow [Semantic Versioning](https://semver.org/) (with
the usual 0.x caveat that minor bumps may include breaking changes).

## v0.1 - 2026-10-28

### Fixed

- `utils.print_eqx_fields` printed constrained parameters with their latent values (e.g. the logarithm of a `Positive` parameter): they are now printed with their physical values. It also printed spurious `Missing` attributes for modules rebuilt by a tree map, e.g. after `eqx.combine` in a calibration: it now prints the declared fields, in order.
- `FeFpJ2Plasticity` was bugged: not objective trial state, total deformation gradient not updated in final state
  and shear components of the plastic flow were doubled because of bad tensor conversion after scalar multiply.
- `FischerBurmeister` used a floored square root (`safe_sqrt`), which made the complementarity residual linear within `sqrt(x**2 + y**2) < 1e-8`. Near the yield onset (`|f| < 1e-8 E`) the return mapping of the `FB`-based behaviors then returned a negative `dp` on the elastic side and a wrong implicit gradient. The square root is now exact, with a zero gradient at the origin.
- `vonMisesIsotropicHardening` starts its Newton solve from the linearized return mapping instead of `dp = 0`: just past the onset `dp` is below the solver's `atol` and the first step was accepted unconverged, giving a wrong implicit gradient. The new start also needs fewer Newton iterations.
- The derivatives of `logm`, `expm`, `powm` and `isotropic_function` were wrong at repeated eigenvalues, including at a multiple of the identity (by 100% for `logm`): they were obtained by differentiating through the eigendecomposition. They now follow from custom JVP rules (Daleckii-Krein formulas with divided differences), exact to second order. This fixes the `LogarithmicStrain` tangent (off by 38% at `F = I`) and its stress for anisotropic laws at two equal principal stretches (off by 12%). The new `eigenvalue_function` does the same for symmetric scalar functions of the eigenvalues; `CompressibleOgden` (tangent off by 1% at `F = I`), `Hosford` (derivative of the normal off by 50% at a uniaxial stress) and `Tresca` now use it. `isotropic_function` takes extra arguments of the scalar function, which can be differentiated. Log-strain updates with their tangent also compile and run several times faster.
- Tensor operations on batches of tensors (leading dimensions): `.det`, `.inv`, `principal_invariants`, `main_invariants`, `pq_invariants` and `axl` were silently wrong, `dev`, `vol` and `von_mises` failed or were silently wrong for a batch of 6, as were `SymmetricTensor4 @ SymmetricTensor2` and `Tensor4 @ Tensor2` with a single stiffness and a batch of strains, and the rotation of fourth-rank tensors for a batch of 3; `eig33`, `eigenvalues`, `polar`, `stretch_tensor`, `sqrtm`, `logm` and the other spectral functions failed. They all accept batches now. Multiplying a tensor by an array of scalars of shape `(...)` scales each tensor of the batch (it used to multiply component by component).
- `SymmetricTensor2(tensor=A)` kept the upper triangle of a non-symmetric `A` instead of its symmetric part, and so did adding a dense array to a `SymmetricTensor2` or subtracting one from it. They now take the symmetric part. `ImposedLoading` sets the imposed shear components of small-strain loadings in both triangles accordingly.
- `rotation.from_euler_bunge` returned the transpose of the documented rotation `Rz(phi1) Rx(Phi) Rz(phi2)`, i.e. the Bunge orientation matrix `g` instead of the active rotation from sample to crystal axes. It now returns the documented rotation, consistent with `from_axis_angle` and `rotate`: a tensor given in the crystal axes is expressed in the sample axes by `T.rotate(R)`.
- `IsotropicTensor4`, `CubicTensor4` and `TransverseIsotropicTensor4` held their constant projector basis as a pytree leaf, so it was differentiated, trained, batched and saved with the coefficients; it is now a constant. They are now `Tensor` objects that interoperate with `SymmetricTensor4` and with each other (`SymmetricTensor4 @ IsotropicTensor4`, `IsotropicTensor4 + CubicTensor4` and `jnp.asarray(C)` used to fail).

### Changed

- The trace-count assertion on the `constitutive_update` of library behaviors is now off by default and controlled by the new `jaxmat.config` module: `jaxmat.config.update("check_max_traces", True)`, the `jaxmat.config.override(check_max_traces=True)` context manager, or the `JAXMAT_CHECK_MAX_TRACES=1` environment variable. Its counter is shared by all instances of a behavior class, so it used to reject legitimate retracing, e.g. two `vonMisesIsotropicHardening` materials with different hardening laws in the same process. Own behaviors can use the same switch with `jaxmat.utils.assert_max_traces` in place of `eqx.debug.assert_max_traces` (#55).

- `nn.ICNN` is reworked. Its output scale grew each time a hidden layer was added at initialization, whatever `scale`. Also, it was non-decreasing in every input, so that it could not represent convex functions such as `x**2`. It now has input skip connections and a `monotone` option, per input (default `True`, i.e. non-decreasing in all inputs as before), non-negative weights through a chosen reparametrization (`positive="softplus"`, `"exp"` or `"square"`), and the shifted softplus `nn.shifted_softplus` as default activation, so that `f(0) = 0` at initialization. The initialization keeps the output of order `scale` for inputs of unit norm, whatever the width and the depth. The parameters are laid out differently (`Us`, `Ws`, `bs`, `final_W`, `final_U`): models trained with the previous version must be retrained.

- The neural GSM demo did not run: `GeneralizedMaxwell` now needs `Nbranch`, and a plot label used `\textrm`, which matplotlib does not support.

### Deprecated

- `utils.enforce_dtype` is now deprecated and should be replaced with the new `parameter` field converter for parameters.
- `utils.default_value` is now deprecated and renamed to `utils.initial_value`. To be used only for state variable initialization.
- `nn.icnn.ICNNSkip` is now a deprecated alias of `nn.ICNN`, which has input skip connections.

### Added

- `utils.print_eqx_fields` takes a `format` option: a format specification such as `".3e"`, a dict of them per field (nested for sub-modules), or a list per element of a list or tuple. It applies to numbers and numeric arrays only, other values being printed as they are.

- `utils.save_model` and `utils.load_model` save a model (e.g. a calibrated material) or a material state to a file and load it back. Leaves are written with `equinox.tree_serialise_leaves` after a one-line JSON header (jaxmat version, model class, optional hyperparameters, paths of the saved leaves). Loading takes either a template module of the same structure, or a factory called with the saved hyperparameters; a file of another model class, of another structure (e.g. saved with constrained parameters unwrapped by `paramax.unwrap`) or not written by `save_model` raises a clear error. Constrained parameters are saved with their latent values.

- A generic `parameters` module has been added. It contains the following features:
  * `parameter()`: a field converter to declare parameters, accepting as default values constants, `eqx.Module` or `AbstractParameter`.
  * Constrained parameters are introduced using latent representations, relying on [`paramax`](https://github.com/danielward27/paramax). Gradients can  be taken with respect to latent representations to remain in the feasible region of the parameter. Currently supports
  `Positive`, `Bounded/Interval`, `Scaled` and `Unconstrained` parameters.
  * Temperature-dependent (or more generally input-dependent, eg. humidity, chemical content, etc.) parameters
  are introduced, replacing constants with temperature-dependent expressions (`AbstractParameter` inheriting from `eqx.Module`)
  with their associated material parameters. ``evaluate_parameters(model, T)`` returns a copy of any model in which every
  ``AbstractParameter`` field has been replaced by its scalar value ``p.evaluate(T)``.
  `TemperatureDependent` is a wrapper around any `AbstractBehavior` which make its constitutive update accept a
  temperature input alongside the mechanical input in its signature. At each update it materializes the wrapped behavior's parameter
  expressions at the current temperature (via ``evaluate_parameters``) and delegates to the wrapped behavior's own ``constitutive_update``, typically via:
  ```python
  def constitutive_update(self, inputs, state, dt):
      mech, T = self._split_inputs(inputs)
      resolved = evaluate_parameters(self.behavior, T)
      return resolved.constitutive_update(mech, state, dt)
  ```
  Note that **thermal strains are not supported** at the moment.

- Generic multiphysics behaviors. `jaxmat.state.make_physics` declares the gradients (inputs), fluxes (outputs) and external variables of a behavior and generates the matching state class, cached so that equal declarations give the same class. A `MultiphysicsBehavior` sets `physics` and implements `response(x, state, dt) -> (fluxes, internal)`: inputs and fluxes are tuples (or dicts) in declaration order, every variable is recorded in the new state, parameters are resolved at the declared external variables, and `tangent` returns the consistent tangent blocks. The library fixes no multiphysics declaration: the new poroelastic consolidation demo declares its own.

- `LogarithmicStrain` turns any small-strain behavior into a finite-strain one in the logarithmic strain space (Miehe, Apel & Lambrecht, 2002). The small-strain behavior runs unchanged on the Hencky strain `E = log(C)/2`, and its stress `T` is pulled back as `S = 2 T : dE/dC`, `P = F S`, the derivative of the tensor logarithm being obtained by automatic differentiation. The resulting behavior is objective, works with any internal variables and with anisotropic models, and its consistent tangent follows by AD. Usage: `jm.LogarithmicStrain(small_strain_behavior)`; the state is a `LogarithmicStrainState` where `state.internal` holds the internal variables of the small-strain behavior, as when it is used directly, and `state.log_strain`, `state.log_stress` are `E` and `T`.

## v0.0.4 - 2026-07-08

### Fixed
- Issue #39 Trust-region solvers (`GaussNewtonTrustRegion`, `NewtonTrustRegion`,
  `BFGSLinearTrustRegion`) crashed with `'frozenset' object is not callable` on
  optimistix >= 0.1.0 (its `verbose` field became a callable). This broke every
  `GeneralizedStandardMaterial`-based model at runtime.
- Issue #38 Fix init=False warnings: solvers are now keyword only, we use `default=None`
  instead of `init=False` to initialize fields depending on other fields

### Changed
- Pinned lower bounds on `optimistix`, `lineax`, `equinox`, `diffrax`, `paramax`
  and added a `test` extra, so environments are reproducible.

### Added
- Solver smoke test that instantiates and runs each solver (guards against this
  class of dependency-drift regression).
