// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Adaptive RK 5(4) Dormand-Prince integrator with PI step controller.
//!
//! ## Scientific scope
//!
//! Provides [`Dopri5`] — an adaptive-step integrator implementing the 5th-order
//! Dormand-Prince method with embedded 4th-order error estimator, controlled by
//! a proportional-integral (PI) step-size controller.
//!
//! The DOPRI5 tableau (Dormand & Prince, 1980; Hairer et al., 1993) uses seven
//! stages to compute a 5th-order solution and a 4th-order error estimate:
//!
//! ```text
//! error = |y5 - y4|  (element-wise)
//! tolerance = atol + rtol * |y_n|
//! step accepted if: rms(error_i / tolerance_i) ≤ 1
//! ```
//!
//! ## Step control
//!
//! ```text
//! h_new = h · (tol / err)^(0.2) clamped to [0.2 h, 5.0 h]
//! ```
//!
//! ## Technical scope
//!
//! Generic over caller-owned context `Ctx`, continuous time scale `S`,
//! reference center `C`, and frame `F`. All typed.
//!
//! ## References
//!
//! * Hairer, Norsett & Wanner, *Solving ODEs I*, 2nd ed., Springer (1993), §II.4.
//! * Montenbruck & Gill, *Satellite Orbits* (2001), §4.4.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use qtty::{IntegratorTolerances, Second};
use tempoch::ContinuousScale;

use super::{deriv_component, rhs, state_at, state_component, AdaptiveStepper};
use crate::error::PrincipiaError;
use crate::models::AccelerationModel;
use crate::state::DynamicsState;

/// Dormand-Prince 5(4) adaptive integrator.
pub struct Dopri5 {
    /// Error control tolerances.
    pub tolerances: IntegratorTolerances,
    /// Maximum allowed step size (seconds).
    pub h_max: Second,
    /// Minimum allowed step size (seconds).
    pub h_min: Second,
}

impl Dopri5 {
    /// Construct with default bounds: `h_max = 86 400 s`, `h_min = 1 μs`.
    pub fn new(tolerances: IntegratorTolerances) -> Self {
        Self {
            tolerances,
            h_max: Second::new(86_400.0),
            h_min: Second::new(1e-6),
        }
    }

    /// Override the maximum step size.
    pub fn with_h_max(mut self, h_max: Second) -> Self {
        self.h_max = h_max;
        self
    }

    /// Override the minimum step size.
    pub fn with_h_min(mut self, h_min: Second) -> Self {
        self.h_min = h_min;
        self
    }
}

impl<Ctx, S, C, F> AdaptiveStepper<Ctx, S, C, F> for Dopri5
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    #[allow(clippy::type_complexity)]
    fn step<M: AccelerationModel<Ctx, S, C, F>>(
        &self,
        model: &M,
        state: &DynamicsState<S, C, F>,
        h_try: Second,
        ctx: &Ctx,
    ) -> Result<(DynamicsState<S, C, F>, Second, Second, u32), PrincipiaError> {
        dopri5_step(
            model,
            state,
            h_try,
            self.tolerances,
            self.h_min,
            self.h_max,
            ctx,
        )
    }
}

/// Single adaptive DOPRI5 step.
///
/// Returns `(new_state, h_used, h_next, steps_rejected)`.
/// `h_try` is clamped to `[h_min, h_max]` at entry.
///
/// # Errors
///
/// * [`PrincipiaError::StepControlFailed`] if the PI controller fails to
///   converge after 50 iterations.
/// * [`PrincipiaError::StepBelowMinimum`] if the step shrinks below `h_min`.
#[allow(clippy::too_many_lines, clippy::type_complexity)]
pub fn dopri5_step<M, Ctx, S, C, F>(
    model: &M,
    s: &DynamicsState<S, C, F>,
    h_try: Second,
    tol: IntegratorTolerances,
    h_min: Second,
    h_max: Second,
    ctx: &Ctx,
) -> Result<(DynamicsState<S, C, F>, Second, Second, u32), PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    // Butcher tableau — Dormand-Prince 5(4).
    let c2 = 1.0 / 5.0;
    let c3 = 3.0 / 10.0;
    let c4 = 4.0 / 5.0;
    let c5 = 8.0 / 9.0;

    let a21 = 1.0 / 5.0;
    let a31 = 3.0 / 40.0;
    let a32 = 9.0 / 40.0;
    let a41 = 44.0 / 45.0;
    let a42 = -56.0 / 15.0;
    let a43 = 32.0 / 9.0;
    let a51 = 19_372.0 / 6_561.0;
    let a52 = -25_360.0 / 2_187.0;
    let a53 = 64_448.0 / 6_561.0;
    let a54 = -212.0 / 729.0;
    let a61 = 9_017.0 / 3_168.0;
    let a62 = -355.0 / 33.0;
    let a63 = 46_732.0 / 5_247.0;
    let a64 = 49.0 / 176.0;
    let a65 = -5_103.0 / 18_656.0;
    let a71 = 35.0 / 384.0;
    let a73 = 500.0 / 1_113.0;
    let a74 = 125.0 / 192.0;
    let a75 = -2_187.0 / 6_784.0;
    let a76 = 11.0 / 84.0;

    let e1 = 71.0 / 57_600.0;
    let e3 = -71.0 / 16_695.0;
    let e4 = 71.0 / 1_920.0;
    let e5 = -17_253.0 / 339_200.0;
    let e6 = 22.0 / 525.0;
    let e7 = -1.0 / 40.0;

    let h_min_abs = h_min.value().abs();
    let h_max_abs = h_max.value().abs();
    let sign = if h_try.value() >= 0.0 {
        1.0_f64
    } else {
        -1.0_f64
    };
    let mut h = sign * h_try.value().abs().clamp(h_min_abs, h_max_abs);
    let mut iters = 0u32;
    let mut rejected = 0u32;

    loop {
        let k1 = rhs(model, s, ctx)?;
        let k2 = rhs(model, &state_at(s, &k1.scaled(a21), h, c2 * h), ctx)?;
        let k3 = rhs(
            model,
            &state_at(s, &k1.scaled(a31).add(&k2.scaled(a32)), h, c3 * h),
            ctx,
        )?;
        let k4 = rhs(
            model,
            &state_at(
                s,
                &k1.scaled(a41).add(&k2.scaled(a42)).add(&k3.scaled(a43)),
                h,
                c4 * h,
            ),
            ctx,
        )?;
        let k5 = rhs(
            model,
            &state_at(
                s,
                &k1.scaled(a51)
                    .add(&k2.scaled(a52))
                    .add(&k3.scaled(a53))
                    .add(&k4.scaled(a54)),
                h,
                c5 * h,
            ),
            ctx,
        )?;
        let k6 = rhs(
            model,
            &state_at(
                s,
                &k1.scaled(a61)
                    .add(&k2.scaled(a62))
                    .add(&k3.scaled(a63))
                    .add(&k4.scaled(a64))
                    .add(&k5.scaled(a65)),
                h,
                h,
            ),
            ctx,
        )?;
        let d7 = k1
            .scaled(a71)
            .add(&k3.scaled(a73))
            .add(&k4.scaled(a74))
            .add(&k5.scaled(a75))
            .add(&k6.scaled(a76));
        let s7 = state_at(s, &d7, h, h);
        let k7 = rhs(model, &s7, ctx)?;

        let err_d = k1
            .scaled(e1)
            .add(&k3.scaled(e3))
            .add(&k4.scaled(e4))
            .add(&k5.scaled(e5))
            .add(&k6.scaled(e6))
            .add(&k7.scaled(e7));

        let mut err_norm = 0.0;
        for i in 0..6 {
            let err = h * deriv_component(&err_d, i);
            let y0i = state_component(s, i);
            let y7i = state_component(&s7, i);
            let abs_tol = if i < 3 {
                tol.abs_pos[i].value()
            } else {
                tol.abs_vel[i - 3].value()
            };
            let sc = abs_tol + tol.rel.value() * y0i.abs().max(y7i.abs());
            let r = err / sc;
            err_norm += r * r;
        }
        err_norm = (err_norm / 6.0).sqrt();

        if err_norm <= 1.0 {
            let h_next_raw = if err_norm == 0.0 {
                h * 5.0
            } else {
                let factor = 0.9 * err_norm.powf(-0.2);
                h * factor.clamp(0.2, 5.0)
            };
            let h_next = sign * h_next_raw.abs().clamp(h_min_abs, h_max_abs);
            return Ok((s7, Second::new(h), Second::new(h_next), rejected));
        }

        rejected += 1;
        iters += 1;
        if iters > 50 {
            return Err(PrincipiaError::StepControlFailed {
                reason: "DOPRI5: step controller failed to converge after 50 iterations",
            });
        }
        let factor = 0.9 * err_norm.powf(-0.2);
        h *= factor.clamp(0.1, 0.9);
        if h.abs() < h_min_abs {
            return Err(PrincipiaError::StepBelowMinimum {
                reason: "DOPRI5: step size fell below h_min; tolerances may be too tight",
            });
        }
    }
}

/// Propagate `state` for `total_dt` with automatically sized DOPRI5 steps.
///
/// # Errors
///
/// Propagates any [`PrincipiaError`] returned by the model or step controller.
pub fn dopri5_propagate<M, Ctx, S, C, F>(
    model: &M,
    state: DynamicsState<S, C, F>,
    total_dt: Second,
    tol: IntegratorTolerances,
    ctx: &Ctx,
) -> Result<DynamicsState<S, C, F>, PrincipiaError>
where
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    let total_dt_s = total_dt.value();
    let mut s = state;
    let mut t = 0.0;
    let mut h = total_dt_s.signum() * 30.0_f64.min(total_dt_s.abs());
    let h_min = Second::new(1e-6);
    let h_max = Second::new(86_400.0);
    while (total_dt_s - t).abs() > 1e-9 {
        if (t + h - total_dt_s) * total_dt_s.signum() > 0.0 {
            h = total_dt_s - t;
        }
        let (s_new, h_used, h_next, _) =
            dopri5_step(model, &s, Second::new(h), tol, h_min, h_max, ctx)?;
        s = s_new;
        t += h_used.value();
        h = h_next.value();
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TwoBody;
    use affn::centers::ReferenceCenter;
    use affn::frames::ReferenceFrame;
    use qtty::unit::Kilometer;
    use qtty::{GravitationalParameter, KmPerSecond, Second};
    use tempoch::{Time, TT};

    #[derive(Debug, Clone, Copy)]
    struct Inertial;
    impl ReferenceFrame for Inertial {
        fn frame_name() -> &'static str {
            "Inertial"
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

    fn circular_state() -> crate::state::DynamicsState<TT, Center, Inertial> {
        let mu = 398_600.441_8_f64;
        let r = 7000.0_f64;
        let v = (mu / r).sqrt();
        crate::state::DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(r, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, v, 0.0),
        )
    }

    fn model() -> TwoBody {
        TwoBody::new(GravitationalParameter::new(398_600.441_8))
    }

    #[test]
    fn with_h_max_overrides() {
        let tol = IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9);
        let d = Dopri5::new(tol).with_h_max(Second::new(300.0));
        assert!((d.h_max.value() - 300.0).abs() < 1e-12);
    }

    #[test]
    fn with_h_min_overrides() {
        let tol = IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9);
        let d = Dopri5::new(tol).with_h_min(Second::new(0.1));
        assert!((d.h_min.value() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn adaptive_stepper_trait_step_succeeds() {
        let tol = IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9);
        let integrator = Dopri5::new(tol);
        let s0 = circular_state();
        let (s1, _h_used, _h_next, _rejected) = integrator
            .step(&model(), &s0, Second::new(30.0), &())
            .unwrap();
        let r = (s1.position.x().value().powi(2)
            + s1.position.y().value().powi(2)
            + s1.position.z().value().powi(2))
        .sqrt();
        assert!((r - 7000.0).abs() < 1.0);
    }

    #[test]
    fn dopri5_propagate_full_orbit_radius_conserved() {
        let tol = IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9);
        let s0 = circular_state();
        let mu = 398_600.441_8_f64;
        let period = 2.0 * core::f64::consts::PI * (7000.0_f64.powi(3) / mu).sqrt();
        let s = dopri5_propagate(&model(), s0, Second::new(period), tol, &()).unwrap();
        let r = (s.position.x().value().powi(2)
            + s.position.y().value().powi(2)
            + s.position.z().value().powi(2))
        .sqrt();
        assert!((r - 7000.0).abs() < 0.1);
    }

    #[test]
    fn step_below_minimum_triggered_by_tight_tolerance() {
        let tight = IntegratorTolerances::uniform(1e-30, 1e-30, 1e-30);
        let s0 = circular_state();
        let h_min = Second::new(90.0);
        let h_max = Second::new(100.0);
        let result = dopri5_step(&model(), &s0, Second::new(100.0), tight, h_min, h_max, &());
        assert!(matches!(
            result,
            Err(crate::error::PrincipiaError::StepBelowMinimum { .. })
        ));
    }
}
