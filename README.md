# principia

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

`principia` depends only on `qtty`, `affn`, `tempoch`, and `thiserror`
(plus optional `serde`). It does **not** depend on `siderust`,
`keplerian`, `cheby`, ephemerides, Earth orientation data,
observatory/atmosphere policy, or any adapter / dataset crate. Astronomy
specifics live downstream in `siderust::astro::dynamics`.

`principia` and `keplerian` deliberately overlap on two-body
arithmetic: `keplerian` is the analytic closed-form path, `principia`
is the numerical integration path. There is no runtime
`principia -> keplerian` dependency.

## Public vocabulary

`principia` uses generic mechanics vocabulary rather than astronomy
specific names:

| Concept                       | Type                                |
|-------------------------------|-------------------------------------|
| Cartesian state               | `DynamicsState<S, C, F>`            |
| State time derivative         | `StateDerivative<F>`                |
| Acceleration model trait      | `AccelerationModel<Ctx, S, C, F>`   |
| Acceleration Jacobian blocks  | `AccelerationPartials<F>`           |
| Composite model               | `CompositeModel<…>`                 |
| Local trajectory frame        | `LocalTrajectoryFrame` + `RTN` / `VNC` / `LVLH` |
| Crate-level error             | `PrincipiaError`                    |

Time is carried as a `tempoch::Time<S>` parameterized by the continuous
time scale `S`. Propagation accepts typed `qtty::Second` steps. The
mechanics kernel never performs time-scale conversion — downstream
adapters provide states and contexts already on the intended scale.

## Features

- `default = ["std"]`
- `std` / `alloc` — standard / allocator support
- `serde` — derives for stable public data types
- `astro` — optional convenience aliases over `affn` astronomical frame
  markers. Does *not* pull in Earth/Sun/Moon constants, ephemerides,
  observatory/atmosphere policy, or any astronomy provider.

## License

AGPL-3.0-only. See [`LICENSE`](LICENSE).
