# Changelog

All notable changes to jaxmat are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning aims to follow [Semantic Versioning](https://semver.org/) (with
the usual 0.x caveat that minor bumps may include breaking changes).

## Unreleased

### Deprecated

- `utils.enforce_dtype` is now deprecated and should be replaced with the new `parameter` field converter for parameters.
- `utils.default_value` is now deprecated and renamed to `utils.initial_value`. To be used only for state variable initialization.

### Added

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
