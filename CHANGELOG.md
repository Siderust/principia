# Changelog

All notable changes to `principia` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Publish workflow (`.github/workflows/publish.yml`) triggered on `v*.*.*`
  tags; verifies crate version matches the tag before uploading to crates.io.

## [0.1.0] - 2026-05-22

### Added
- Initial extraction of generic Newtonian numerical dynamics from
  `siderust::astro::dynamics`:
  - `DynamicsState<S, C, F>` and `StateDerivative<F>`
  - `AccelerationModel<Ctx, S, C, F>`, `AccelerationPartials<F>`,
    `CompositeModel`
  - Numerical `TwoBody` and generic `J2` acceleration models
  - Gravity-field provider trait and typed spherical-harmonic
    acceleration kernel
  - RK4, DOPRI5, and DOP853 integrators with a shared `Stepper` trait
  - Adaptive propagation driver, configuration, results, and event
    interface (including `RadialThresholdEvent`)
  - Variational equations and state-transition matrix propagation
  - Local trajectory frames (`LocalTrajectoryFrame`) with `RTN`,
    `VNC`, `LVLH` markers
  - Typed covariance transport with `ProcessNoise`
  - `PrincipiaError` crate-level error family

### Migration
Replaces the former generic surface of `siderust::astro::dynamics`.
Astronomy-specific perturbations (drag, SRP, third-body, Earth
geopotential adapters, Earth rotation, atmospheres, ephemeris-backed
contexts) now live in `siderust::astro::dynamics`. See the
project README for the full old-to-new mapping.

[0.1.0]: https://github.com/siderust/siderust/releases/tag/principia-v0.1.0
