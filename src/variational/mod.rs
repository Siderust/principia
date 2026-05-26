// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

//! Variational equations and state-transition matrices.
//!
//! ## Scientific scope
//!
//! Solves the linearized system
//!
//! ```text
//! Φ̇(t, t₀) = F(t) · Φ(t, t₀),    Φ(t₀, t₀) = I
//! ```
//!
//! where `F(t) = [[0, I], [A_r, A_v]]` is built from analytic acceleration
//! partial derivatives.
//!
//! ## Technical scope
//!
//! [`propagate_stm`] and [`propagate_stm_with`] integrate the nonlinear state
//! and its state-transition matrix together with fixed-step RK4. Finite-
//! difference validation helpers are also provided.
//!
//! ## References
//!
//! * Montenbruck & Gill, *Satellite Orbits*, §7.1.
//! * Tapley, Schutz, Born, *Statistical Orbit Determination*, §4.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use affn::matrix6::FrameMatrix6;
use qtty::Second;
use tempoch::ContinuousScale;

use crate::error::PrincipiaError;
use crate::integrators::rk4_propagate;
use crate::models::{AccelerationModel, AccelerationPartials};
use crate::state::DynamicsState;

#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::vec;
#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::vec::Vec;

#[cfg(any(feature = "alloc", feature = "std"))]
use crate::integrators::rk4_propagate_series;

/// State-transition matrix `Φ(t, t₀)` tagged with frame `F`.
pub type StateTransitionMatrix<F> = FrameMatrix6<F>;

/// Fixed-step configuration for the variational propagator.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VariationalConfig {
    /// Positive RK4 sub-step magnitude.
    pub step: Second,
}

impl Default for VariationalConfig {
    fn default() -> Self {
        Self {
            step: Second::new(30.0),
        }
    }
}

impl VariationalConfig {
    /// Construct a validated variational propagator configuration.
    pub fn try_new(step: Second) -> Result<Self, PrincipiaError> {
        if !step.value().is_finite() || step.value() <= 0.0 {
            return Err(PrincipiaError::NonPositiveValue {
                context: "VariationalConfig: step must be finite and positive",
            });
        }
        Ok(Self { step })
    }
}

fn build_a_matrix<F: ReferenceFrame>(partials: &AccelerationPartials<F>) -> [[f64; 6]; 6] {
    let ar = partials.d_acc_d_pos.as_array();
    let av = partials.d_acc_d_vel.as_array();
    let mut a = [[0.0; 6]; 6];
    a[0][3] = 1.0;
    a[1][4] = 1.0;
    a[2][5] = 1.0;
    for i in 0..3 {
        for j in 0..3 {
            a[3 + i][j] = ar[i][j];
            a[3 + i][3 + j] = av[i][j];
        }
    }
    a
}

fn mat6_mul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for k in 0..6 {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..6 {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn mat6_scale(m: &[[f64; 6]; 6], s: f64) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = m[i][j] * s;
        }
    }
    out
}

fn mat6_add(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

#[allow(clippy::needless_range_loop)]
fn identity_6x6() -> [[f64; 6]; 6] {
    let mut m = [[0.0; 6]; 6];
    for i in 0..6 {
        m[i][i] = 1.0;
    }
    m
}

fn variational_derivative<M, Ctx, S, C, F>(
    model: &M,
    state: &DynamicsState<S, C, F>,
    phi: &[[f64; 6]; 6],
    ctx: &Ctx,
) -> Result<([f64; 6], [[f64; 6]; 6]), PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    let acc = model.acceleration(state, ctx)?;
    let dy = [
        state.velocity.x().value(),
        state.velocity.y().value(),
        state.velocity.z().value(),
        acc.x().value(),
        acc.y().value(),
        acc.z().value(),
    ];
    let partials = model.partials(state, ctx)?;
    Ok((dy, mat6_mul(&build_a_matrix(&partials), phi)))
}

fn advance_state<S, C, F>(
    state: &DynamicsState<S, C, F>,
    dy: &[f64; 6],
    h: f64,
) -> DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    DynamicsState::new(
        state.epoch,
        affn::cartesian::Position::<C, F, qtty::unit::Kilometer>::new_with_params(
            state.position.center_params().clone(),
            state.position.x().value() + h * dy[0],
            state.position.y().value() + h * dy[1],
            state.position.z().value() + h * dy[2],
        ),
        affn::cartesian::Velocity::<F, qtty::KmPerSecond>::new(
            state.velocity.x().value() + h * dy[3],
            state.velocity.y().value() + h * dy[4],
            state.velocity.z().value() + h * dy[5],
        ),
    )
}

fn advance_state_with_epoch<S, C, F>(
    state: &DynamicsState<S, C, F>,
    dy: &[f64; 6],
    h: f64,
) -> DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    let mut next = advance_state(state, dy, h);
    next.epoch = state.epoch + Second::new(h);
    next
}

fn rk4_combine_vec(k1: &[f64; 6], k2: &[f64; 6], k3: &[f64; 6], k4: &[f64; 6]) -> [f64; 6] {
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
    out
}

fn rk4_combine_mat(
    k1: &[[f64; 6]; 6],
    k2: &[[f64; 6]; 6],
    k3: &[[f64; 6]; 6],
    k4: &[[f64; 6]; 6],
) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]) / 6.0;
        }
    }
    out
}

#[allow(clippy::type_complexity)]
fn variational_rk4_step<M, Ctx, S, C, F>(
    model: &M,
    state: &DynamicsState<S, C, F>,
    phi: &[[f64; 6]; 6],
    h: Second,
    ctx: &Ctx,
) -> Result<(DynamicsState<S, C, F>, [[f64; 6]; 6]), PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    let h_s = h.value();
    let half = 0.5 * h_s;
    let (dy1, dphi1) = variational_derivative(model, state, phi, ctx)?;
    let state2 = advance_state(state, &dy1, half);
    let phi2 = mat6_add(phi, &mat6_scale(&dphi1, half));
    let (dy2, dphi2) = variational_derivative(model, &state2, &phi2, ctx)?;
    let state3 = advance_state(state, &dy2, half);
    let phi3 = mat6_add(phi, &mat6_scale(&dphi2, half));
    let (dy3, dphi3) = variational_derivative(model, &state3, &phi3, ctx)?;
    let state4 = advance_state(state, &dy3, h_s);
    let phi4 = mat6_add(phi, &mat6_scale(&dphi3, h_s));
    let (dy4, dphi4) = variational_derivative(model, &state4, &phi4, ctx)?;
    let dy = rk4_combine_vec(&dy1, &dy2, &dy3, &dy4);
    let dphi = rk4_combine_mat(&dphi1, &dphi2, &dphi3, &dphi4);
    Ok((
        advance_state_with_epoch(state, &dy, h_s),
        mat6_add(phi, &mat6_scale(&dphi, h_s)),
    ))
}

/// Propagate the nonlinear state and STM over `dt` using a conservative default step.
#[allow(clippy::type_complexity)]
pub fn propagate_stm<M, Ctx, S, C, F>(
    model: &M,
    state: DynamicsState<S, C, F>,
    dt: Second,
    ctx: &Ctx,
) -> Result<(DynamicsState<S, C, F>, StateTransitionMatrix<F>), PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    let dt_s = dt.value().abs();
    let step_s = (dt_s / 100.0).clamp(1e-12, 30.0);
    propagate_stm_with(
        model,
        state,
        dt,
        ctx,
        &VariationalConfig {
            step: Second::new(step_s),
        },
    )
}

/// Propagate the nonlinear state and STM with an explicit fixed-step configuration.
#[allow(clippy::type_complexity)]
pub fn propagate_stm_with<M, Ctx, S, C, F>(
    model: &M,
    state: DynamicsState<S, C, F>,
    dt: Second,
    ctx: &Ctx,
    config: &VariationalConfig,
) -> Result<(DynamicsState<S, C, F>, StateTransitionMatrix<F>), PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    if dt.value() == 0.0 {
        return Ok((state, FrameMatrix6::identity()));
    }
    if !config.step.value().is_finite() || config.step.value() <= 0.0 {
        return Err(PrincipiaError::NonPositiveValue {
            context: "VariationalConfig: step must be finite and positive",
        });
    }
    let step_abs = config.step.value().abs();
    let n_steps = (dt.value().abs() / step_abs).ceil() as usize;
    let n_steps = n_steps.max(1);
    let h = dt.value() / n_steps as f64;
    let mut current_state = state;
    let mut phi = identity_6x6();
    for _ in 0..n_steps {
        let (new_state, new_phi) =
            variational_rk4_step(model, &current_state, &phi, Second::new(h), ctx)?;
        current_state = new_state;
        phi = new_phi;
    }
    Ok((current_state, FrameMatrix6::from_array(phi)))
}

fn state_component<S, C, F>(state: &DynamicsState<S, C, F>, j: usize) -> f64
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    match j {
        0 => state.position.x().value(),
        1 => state.position.y().value(),
        2 => state.position.z().value(),
        3 => state.velocity.x().value(),
        4 => state.velocity.y().value(),
        5 => state.velocity.z().value(),
        _ => panic!("index out of range"),
    }
}

fn perturb_component<S, C, F>(
    state: &DynamicsState<S, C, F>,
    j: usize,
    delta: f64,
) -> DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    DynamicsState::new(
        state.epoch,
        affn::cartesian::Position::<C, F, qtty::unit::Kilometer>::new_with_params(
            state.position.center_params().clone(),
            state.position.x().value() + if j == 0 { delta } else { 0.0 },
            state.position.y().value() + if j == 1 { delta } else { 0.0 },
            state.position.z().value() + if j == 2 { delta } else { 0.0 },
        ),
        affn::cartesian::Velocity::<F, qtty::KmPerSecond>::new(
            state.velocity.x().value() + if j == 3 { delta } else { 0.0 },
            state.velocity.y().value() + if j == 4 { delta } else { 0.0 },
            state.velocity.z().value() + if j == 5 { delta } else { 0.0 },
        ),
    )
}

/// Finite-difference STM, preserved as a validation helper.
#[allow(clippy::needless_range_loop)]
pub fn finite_diff_stm<M, Ctx, S, C, F>(
    model: &M,
    s0: DynamicsState<S, C, F>,
    h: Second,
    n_steps: usize,
    ctx: &Ctx,
) -> Result<StateTransitionMatrix<F>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    if n_steps == 0 {
        return Ok(FrameMatrix6::identity());
    }
    let mut raw = [[0.0; 6]; 6];
    for j in 0..6 {
        let x0j = state_component(&s0, j);
        let delta = 1e-6 * x0j.abs().max(1.0);
        let total = Second::new(h.value() * n_steps as f64);
        let s_plus = rk4_propagate(model, perturb_component(&s0, j, delta), h, total, ctx)?;
        let s_minus = rk4_propagate(model, perturb_component(&s0, j, -delta), h, total, ctx)?;
        for i in 0..6 {
            raw[i][j] =
                (state_component(&s_plus, i) - state_component(&s_minus, i)) / (2.0 * delta);
        }
    }
    Ok(FrameMatrix6::from_array(raw))
}

/// Finite-difference STM series evaluated at every fixed step.
#[cfg(any(feature = "alloc", feature = "std"))]
#[allow(clippy::needless_range_loop)]
pub fn finite_diff_stm_series<M, Ctx, S, C, F>(
    model: &M,
    s0: DynamicsState<S, C, F>,
    h: Second,
    n_steps: usize,
    ctx: &Ctx,
) -> Result<Vec<StateTransitionMatrix<F>>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    if n_steps == 0 {
        return Ok(vec![FrameMatrix6::identity()]);
    }
    let mut perturbed: [[Vec<[f64; 6]>; 2]; 6] = Default::default();
    let mut hs = [0.0; 6];
    for j in 0..6 {
        let x0j = state_component(&s0, j);
        let delta = 1e-6 * x0j.abs().max(1.0);
        hs[j] = delta;
        for (sign_idx, sign) in [-1.0_f64, 1.0_f64].iter().enumerate() {
            let sp = perturb_component(&s0, j, sign * delta);
            perturbed[j][sign_idx] = rk4_propagate_series(model, sp, h, n_steps, ctx)?
                .into_iter()
                .map(|s| {
                    [
                        state_component(&s, 0),
                        state_component(&s, 1),
                        state_component(&s, 2),
                        state_component(&s, 3),
                        state_component(&s, 4),
                        state_component(&s, 5),
                    ]
                })
                .collect();
        }
    }
    let mut out = Vec::with_capacity(n_steps + 1);
    for k in 0..=n_steps {
        let mut raw = [[0.0; 6]; 6];
        for j in 0..6 {
            let plus = perturbed[j][1][k];
            let minus = perturbed[j][0][k];
            for i in 0..6 {
                raw[i][j] = (plus[i] - minus[i]) / (2.0 * hs[j]);
            }
        }
        out.push(FrameMatrix6::from_array(raw));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TwoBody;
    use affn::centers::ReferenceCenter;
    use qtty::{GravitationalParameter, Second};
    use tempoch::{Time, TT};

    #[derive(Debug, Clone, Copy)]
    struct Frame;
    impl ReferenceFrame for Frame {
        fn frame_name() -> &'static str {
            "Frame"
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Center;
    impl ReferenceCenter for Center {
        type Params = ();
        fn center_name() -> &'static str {
            "Center"
        }
    }

    fn state() -> DynamicsState<TT, Center, Frame> {
        DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Frame, qtty::unit::Kilometer>::new(
                7000.0, 0.0, 0.0,
            ),
            affn::cartesian::Velocity::<Frame, qtty::KmPerSecond>::new(0.0, 7.54605329, 0.0),
        )
    }

    #[test]
    fn zero_dt_returns_identity() {
        let (_, phi): (_, StateTransitionMatrix<Frame>) = propagate_stm(
            &TwoBody::new(GravitationalParameter::new(398_600.441_8)),
            state(),
            Second::new(0.0),
            &(),
        )
        .unwrap();
        assert_eq!(
            *phi.as_array(),
            *FrameMatrix6::<Frame>::identity().as_array()
        );
    }

    #[test]
    fn finite_diff_zero_steps_is_identity() {
        let phi = finite_diff_stm(
            &TwoBody::new(GravitationalParameter::new(398_600.441_8)),
            state(),
            Second::new(1.0),
            0,
            &(),
        )
        .unwrap();
        assert_eq!(
            *phi.as_array(),
            *FrameMatrix6::<Frame>::identity().as_array()
        );
    }

    #[test]
    fn propagate_stm_with_rejects_non_positive_step() {
        let result = propagate_stm_with(
            &TwoBody::new(GravitationalParameter::new(398_600.441_8)),
            state(),
            Second::new(60.0),
            &(),
            &VariationalConfig {
                step: Second::new(0.0),
            },
        );
        assert!(matches!(
            result,
            Err(PrincipiaError::NonPositiveValue { .. })
        ));
    }
}
