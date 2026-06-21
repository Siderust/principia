# Changelog

All notable changes to `principia` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.1] - 2026-06-01

### Changed

- update affn to v0.8

## [0.2.0] - 2026-05-26

### Added
- `PrincipiaError` variants: `InvalidParameter`, `NonFiniteValue`, `NonPositiveValue`, `InvalidTolerance`, `InvalidGravityRequest`, `InvalidPropagationConfig`
- `DynamicsState::try_new`, `is_finite`, `position_norm`, `velocity_norm`
- `StateDerivative::try_new`
- `TwoBody::try_new`, `try_with_min_radius`, `with_min_radius`; configurable `min_radius` field
- `J2::try_new`, `try_with_min_radius`, `with_min_radius`; configurable `min_radius` field
- `GravityConstants::try_new`
- `PropagationConfig::validate()`
- Typed `spherical_harmonic_acceleration` wrapper; raw kernel renamed to `spherical_harmonic_acceleration_raw_km`
- `rtn_from_raw_km_km_s`, `vnc_from_raw_km_km_s`, `lvlh_from_raw_km_km_s` (renamed from `*_from_state`)
- `StateCovariance::try_diagonal_from_sigmas`, `try_from_row_major`
- `ProcessNoise::try_diagonal_from_sigmas`
- `VariationalConfig::try_new`
- `EventDirection` enum; `RadialThresholdEvent` uses `direction: EventDirection` instead of `falling: bool`
- `serde` support for stable value types including states, models, frames, covariance, errors, and integrator configs
- Five runnable examples: `two_body_rk4`, `dopri5_orbit`, `local_frames`, `gravity_field`, `stm_propagation`

### Changed
- All SPDX headers corrected from `AGPL-3.0-or-later` to `AGPL-3.0-only`
- `rk4_propagate` now returns `Err(InvalidParameter)` for zero or non-finite step size
- `spherical_harmonic_acceleration_raw_km` no longer silently truncates degree/order; the typed wrapper validates requests explicitly
- README now documents WIP status, unit policy, crate boundary, serde coverage, and example entry points
- CI now checks additional feature combinations and runs `cargo test --no-default-features`

### Removed
- Raw `rtn_from_state`, `vnc_from_state`, `lvlh_from_state` helpers replaced by `*_raw_km_km_s` variants

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

[0.2.0]: https://github.com/siderust/siderust/releases/tag/principia-v0.2.0
[0.1.0]: https://github.com/siderust/siderust/releases/tag/principia-v0.1.0
