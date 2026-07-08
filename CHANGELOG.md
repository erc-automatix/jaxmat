# Changelog

All notable changes to jaxmat are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning aims to follow [Semantic Versioning](https://semver.org/) (with
the usual 0.x caveat that minor bumps may include breaking changes).

## [Unreleased]

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
