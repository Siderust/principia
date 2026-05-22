// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Classical fixed-step Runge-Kutta 4 (RK4).
//!
//! ## Scientific scope
//!
//! Implements the standard RK4 quadrature for first-order ODE systems
//! `ẋ = f(x, t)` with fixed step size:
//!
//! ```text
//! k1 = f(x_n,            t_n)
//! k2 = f(x_n + h/2 · k1, t_n + h/2)
//! k3 = f(x_n + h/2 · k2, t_n + h/2)
//! k4 = f(x_n + h   · k3, t_n + h)
//! x_{n+1} = x_n + h/6 · (k1 + 2 k2 + 2 k3 + k4)
//! ```
//!
//! ## Technical scope
//!
//! Generic over caller-owned context `Ctx`, continuous time scale `S`,
//! reference center `C`, and frame `F`. Epochs at intermediate stages are
//! advanced correctly for time-varying force models.
//!
//! ## References
//!
//! * Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I*, §II.1.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use qtty::Second;
use tempoch::ContinuousScale;

use super::{rhs, state_at};
use crate::error::PrincipiaError;
use crate::integrators::Stepper;
use crate::models::AccelerationModel;
use crate::state::DynamicsState;

/// Fixed-step classical RK4 integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rk4;

impl<Ctx, S, C, F> Stepper<Ctx, S, C, F> for Rk4
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn step<M: AccelerationModel<Ctx, S, C, F>>(
        &self,
        model: &M,
        state: &DynamicsState<S, C, F>,
        h: Second,
        ctx: &Ctx,
    ) -> Result<DynamicsState<S, C, F>, PrincipiaError> {
        let h_s = h.value();
        let half = 0.5 * h_s;

        let k1 = rhs(model, state, ctx)?;
        let k2 = rhs(model, &state_at(state, &k1, half, half), ctx)?;
        let k3 = rhs(model, &state_at(state, &k2, half, half), ctx)?;
        let k4 = rhs(model, &state_at(state, &k3, h_s, h_s), ctx)?;

        let combined = crate::state::StateDerivative::rk4_combine(&k1, &k2, &k3, &k4);
        Ok(state.advance_with_epoch(&combined, h))
    }
}

/// Single RK4 step (free function).
///
/// # Arguments
///
/// * `model` — acceleration model supplying `dv/dt`.
/// * `state` — current state at the start of the step.
/// * `h`     — step size (signed, seconds).
/// * `ctx`   — caller-owned context forwarded to the model.
///
/// # Errors
///
/// Propagates any [`PrincipiaError`] returned by the model.
pub fn rk4_step<M, Ctx, S, C, F>(
    model: &M,
    state: &DynamicsState<S, C, F>,
    h: Second,
    ctx: &Ctx,
) -> Result<DynamicsState<S, C, F>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    Rk4.step(model, state, h, ctx)
}

/// Propagate `state` for `total_dt` with a fixed step of `h`.
///
/// The final sub-step is shortened if `total_dt` is not an exact multiple of
/// `h`. Returns the accumulated state after all steps.
///
/// # Errors
///
/// Propagates any [`PrincipiaError`] returned by the model.
pub fn rk4_propagate<M, Ctx, S, C, F>(
    model: &M,
    mut state: DynamicsState<S, C, F>,
    h: Second,
    total_dt: Second,
    ctx: &Ctx,
) -> Result<DynamicsState<S, C, F>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    let total = total_dt.value();
    let sign = if total >= 0.0 { 1.0_f64 } else { -1.0_f64 };
    let h_abs = h.value().abs();
    let mut elapsed = 0.0_f64;
    while (total - elapsed).abs() > 1e-9 {
        let remaining = total - elapsed;
        let step = sign * h_abs.min(remaining.abs());
        state = rk4_step(model, &state, Second::new(step), ctx)?;
        elapsed += step;
    }
    Ok(state)
}

/// Propagate `state` for `n` fixed steps of `h` and collect all intermediate
/// states (including `state` itself as index 0).
///
/// # Errors
///
/// Propagates any [`PrincipiaError`] returned by the model.
#[cfg(any(feature = "alloc", feature = "std"))]
pub fn rk4_propagate_series<M, Ctx, S, C, F>(
    model: &M,
    state: DynamicsState<S, C, F>,
    h: Second,
    n: usize,
    ctx: &Ctx,
) -> Result<alloc::vec::Vec<DynamicsState<S, C, F>>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    DynamicsState<S, C, F>: Clone,
{
    let mut states = alloc::vec::Vec::with_capacity(n + 1);
    states.push(state.clone());
    let mut current = state;
    for _ in 0..n {
        current = rk4_step(model, &current, h, ctx)?;
        states.push(current.clone());
    }
    Ok(states)
}
