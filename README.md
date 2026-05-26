# principia

[![Crates.io](https://img.shields.io/crates/v/principia)](https://crates.io/crates/principia)
[![docs.rs](https://img.shields.io/docsrs/principia)](https://docs.rs/principia)
[![CI](https://github.com/Siderust/principia/actions/workflows/ci.yml/badge.svg)](https://github.com/Siderust/principia/actions/workflows/ci.yml)
[![License: AGPL-3.0-only](https://img.shields.io/badge/license-AGPL--3.0--only-blue)](LICENSE)

> **Pre-1.0 — API is not yet stable.** Expect breaking changes in every minor release.

Typed Newtonian numerical dynamics for Rust: state propagation,
acceleration-model composition, RK4 / DOPRI5 / DOP853 integrators,
variational equations / STM, covariance transport, local trajectory
frames, and gravity-field kernels.

`principia` is the **numerical** companion to
[`keplerian`](https://github.com/Siderust/keplerian) (analytic
Kepler / conic mechanics). Both build on the typed primitives in
[`qtty`](https://github.com/Siderust/qtty),
[`affn`](https://github.com/Siderust/affn), and
[`tempoch`](https://github.com/Siderust/tempoch).

## Unit conventions

All internal computations use kilometres (km) for distance and km s⁻¹ for velocity.  
Gravitational parameters are in km³ s⁻².  
Time is in seconds via [`tempoch`] typed instants; no raw epoch floats appear in public APIs.  
No time-scale conversion is performed inside principia; callers must supply a consistent epoch.

## Crate boundary

`principia` is domain-agnostic: it contains no constants for specific celestial bodies (no Earth GM, no Moon radius).  
Body-specific constants belong in `siderust`.  
`principia` depends only on `qtty` (units), `tempoch` (time), and `affn` (coordinates).

## Layering

```text
qtty + affn + tempoch
        |
        +--> keplerian   # analytic Kepler / conic mechanics
        |
        +--> principia   # numerical Newtonian dynamics + propagation
                    |
                    +--> siderust   # astronomy/geodesy adapters
                                    # (ephemerides, EOP, atmospheres, ...)
```

`principia` does **not** own ephemerides, Earth orientation data,
atmosphere policy, named body constants, observatories, or mission
policies. Raw numerical kernels are explicitly marked as `*_raw_*`
helpers; typed wrappers are preferred at the public API boundary.

## Features

- `default = ["std"]`
- `std` / `alloc` — standard / allocator support
- `serde` — serialization support for stable value types listed below
- `astro` — optional convenience aliases over `affn` astronomical frame
  markers only; it does **not** add body constants, ephemerides,
  observatory policy, or any Siderust-specific adapter

## `serde` coverage

With the `serde` feature enabled, `principia` serializes these stable value types:

| Type | Notes |
|---|---|
| `DynamicsState` | Serializes epoch, center parameters, and Cartesian components |
| `StateDerivative` | Serializes raw velocity/acceleration components |
| `TwoBody` | Includes configurable `min_radius` |
| `J2` | Includes configurable `min_radius` |
| `GravityConstants` | |
| `AccelerationPartials` | Serializes Jacobian blocks as raw 3×3 arrays |
| `Rk4`, `Dopri5`, `Dop853` | Adaptive integrators serialize embedded tolerance values and step bounds |
| `Dop853Step` | Serializes cached step endpoints and derivatives |
| `EventDirection` | |
| `RadialThresholdEvent` | |
| `LocalTrajectoryFrame` | Serializes the DCM as a raw 3×3 array |
| `StateCovariance` | Serializes as a row-major 6×6 matrix |
| `ProcessNoise` | Serializes as a row-major 6×6 matrix |
| `VariationalConfig` | |
| `PrincipiaError` | |

`PropagationConfig` is **not** serializable because it holds a boxed event detector.
`qtty::IntegratorTolerances` is re-exported for convenience, but its serde support is owned by `qtty`, not by `principia`.

## Public vocabulary

| Concept | Type |
|---|---|
| Cartesian state | `DynamicsState<S, C, F>` |
| State time derivative | `StateDerivative<F>` |
| Acceleration model trait | `AccelerationModel<Ctx, S, C, F>` |
| Acceleration Jacobian blocks | `AccelerationPartials<F>` |
| Local trajectory frame | `LocalTrajectoryFrame` + `RTN` / `VNC` / `LVLH` |
| Crate-level error | `PrincipiaError` |

## Examples

Runnable examples live under [`examples/`](examples):

- `two_body_rk4`
- `dopri5_orbit`
- `local_frames`
- `gravity_field`
- `stm_propagation`

Run them with:

```bash
cargo run --example two_body_rk4 --all-features
cargo run --example dopri5_orbit --all-features
cargo run --example local_frames --all-features
cargo run --example gravity_field --all-features
cargo run --example stm_propagation --all-features
```

## License

AGPL-3.0-only. See [`LICENSE`](LICENSE).
