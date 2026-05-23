// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Numerical integrators for [`DynamicsState`](crate::state::DynamicsState).
//!
//! ## Scientific scope
//!
//! Provides explicit Runge-Kutta methods for the Cartesian equations of
//! motion `dr/dt = v`, `dv/dt = a(r, v, t)`:
//!
//! * [`rk4`] — fixed-step classical Runge-Kutta of order 4.
//! * [`dopri5`] — Dormand-Prince embedded RK 5(4) with adaptive step control.
//! * [`dop853`] — Hairer et al. high-order embedded RK 8(5,3).
//!
//! ## Technical scope
//!
//! Fixed-step integrators implement [`Stepper`]; adaptive-step integrators
//! implement [`AdaptiveStepper`] and expose the PI step-size controller
//! internals that the [`crate::propagation`] driver needs.
//!
//! Internal helpers (`state_component`, `deriv_component`, `rhs`, `state_at`)
//! are `pub(super)` and shared across all sub-modules.
//!
//! ## References
//!
//! * Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I*, §II.

pub mod dop853;
pub mod dopri5;
pub mod rk4;

pub use dop853::{dop853_propagate, dop853_step, Dop853, Dop853Step};
pub use dopri5::{dopri5_propagate, dopri5_step, Dopri5};
pub use rk4::{rk4_propagate, rk4_step, Rk4};
#[cfg(any(feature = "alloc", feature = "std"))]
pub use rk4::rk4_propagate_series;

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use tempoch::ContinuousScale;

use crate::error::PrincipiaError;
use crate::models::AccelerationModel;
use crate::state::{DynamicsState, StateDerivative};
use qtty::Second;

/// Contract for fixed-step integrators (e.g. classical RK4).
pub trait Stepper<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Advance `state` by `dt` using `model` evaluated under `ctx`.
    fn step<M: AccelerationModel<Ctx, S, C, F>>(
        &self,
        model: &M,
        state: &DynamicsState<S, C, F>,
        h: Second,
        ctx: &Ctx,
    ) -> Result<DynamicsState<S, C, F>, PrincipiaError>;
}

/// Contract for adaptive-step integrators (DOPRI5, DOP853).
///
/// Returns `(accepted_state, h_used, h_next, steps_rejected)`.
pub trait AdaptiveStepper<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Attempt a single step of size `h_try` with automatic error control.
    #[allow(clippy::type_complexity)]
    fn step<M: AccelerationModel<Ctx, S, C, F>>(
        &self,
        model: &M,
        state: &DynamicsState<S, C, F>,
        h_try: Second,
        ctx: &Ctx,
    ) -> Result<(DynamicsState<S, C, F>, Second, Second, u32), PrincipiaError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
pub(super) fn state_component<S, C, F>(s: &DynamicsState<S, C, F>, i: usize) -> f64
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    match i {
        0 => s.position.x().value(),
        1 => s.position.y().value(),
        2 => s.position.z().value(),
        3 => s.velocity.x().value(),
        4 => s.velocity.y().value(),
        5 => s.velocity.z().value(),
        _ => panic!("index {i} out of range [0,5]"),
    }
}

#[inline]
pub(super) fn deriv_component<F: ReferenceFrame>(d: &StateDerivative<F>, i: usize) -> f64 {
    match i {
        0 => d.vel.x().value(),
        1 => d.vel.y().value(),
        2 => d.vel.z().value(),
        3 => d.acc.x().value(),
        4 => d.acc.y().value(),
        5 => d.acc.z().value(),
        _ => panic!("index {i} out of range [0,5]"),
    }
}

/// Evaluate the RHS: `[v, a(s, ctx)]`.
#[inline]
pub(super) fn rhs<M, Ctx, S, C, F>(
    model: &M,
    s: &DynamicsState<S, C, F>,
    ctx: &Ctx,
) -> Result<StateDerivative<F>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    Ok(StateDerivative::new(
        s.velocity,
        model.acceleration(s, ctx)?,
    ))
}

/// Build an intermediate RK stage state:
/// `y_i = s + h · d`,  `epoch_i = s.epoch + dt`.
#[inline]
pub(super) fn state_at<S, C, F>(
    s: &DynamicsState<S, C, F>,
    d: &StateDerivative<F>,
    h: f64,
    dt: f64,
) -> DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    let mut next = s.advance(d, Second::new(h));
    next.epoch = s.epoch + Second::new(dt);
    next
}

// Re-export tolerances for integrator users.
pub use qtty::IntegratorTolerances;
