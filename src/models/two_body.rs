// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Numerical two-body (point-mass) gravity model.
//!
//! ## Scientific scope
//!
//! Implements the Newtonian central-force acceleration
//!
//! ```text
//! a = -μ · r / |r|³
//! ```
//!
//! with analytic Jacobian
//!
//! ```text
//! ∂a/∂r = -μ/|r|³ · (I - 3 r̂ r̂ᵀ),   ∂a/∂v = 0.
//! ```
//!
//! This is the numerical companion to the analytic two-body propagation in
//! [`keplerian`]; both are deliberate and have no runtime dependency on
//! each other.
//!
//! ## Technical scope
//!
//! [`TwoBody`] is generic over the central-body gravitational parameter
//! `μ`. No astronomy-specific body identity is assumed. Earth /
//! Sun / Moon convenience constructors live in
//! `siderust::astro::perturbations::earth` (and analogous astronomy
//! adapters) — not in `principia`.
//!
//! ## References
//!
//! * Vallado, *Fundamentals of Astrodynamics and Applications*, §1.
//! * Montenbruck & Gill, *Satellite Orbits*, §3.1.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use affn::matrix3::FrameMatrix3;
use qtty::dynamics::GravitationalParameter;
use tempoch::ContinuousScale;

use crate::error::PrincipiaError;
use crate::models::{AccelerationModel, AccelerationPartials};
use crate::state::{Acceleration, DynamicsState};

/// Minimum radial magnitude below which two-body evaluation is considered
/// degenerate (`100 km`). Avoids divide-by-zero without rejecting realistic
/// orbits.
const DEGENERATE_RADIUS_KM: f64 = 100.0;

/// Newtonian point-mass gravity acceleration model.
#[derive(Debug, Clone, Copy)]
pub struct TwoBody {
    /// Standard gravitational parameter `μ = G·M` (km³/s²).
    pub mu: GravitationalParameter,
}

impl TwoBody {
    /// Construct a two-body model from a typed gravitational parameter.
    #[inline]
    pub const fn new(mu: GravitationalParameter) -> Self {
        Self { mu }
    }
}

impl<Ctx, S, C, F> AccelerationModel<Ctx, S, C, F> for TwoBody
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn name(&self) -> &'static str {
        "two_body"
    }

    fn acceleration(
        &self,
        state: &DynamicsState<S, C, F>,
        _ctx: &Ctx,
    ) -> Result<Acceleration<F>, PrincipiaError> {
        let rx = state.position.x().value();
        let ry = state.position.y().value();
        let rz = state.position.z().value();
        let r2 = rx * rx + ry * ry + rz * rz;
        let r = r2.sqrt();
        if r < DEGENERATE_RADIUS_KM {
            return Err(PrincipiaError::DegenerateGeometry {
                reason: "radial magnitude below two-body degeneracy threshold",
            });
        }
        let inv_r3 = 1.0 / (r2 * r);
        let mu = self.mu.value();
        Ok(Acceleration::<F>::new(
            -mu * rx * inv_r3,
            -mu * ry * inv_r3,
            -mu * rz * inv_r3,
        ))
    }

    fn partials(
        &self,
        state: &DynamicsState<S, C, F>,
        _ctx: &Ctx,
    ) -> Result<AccelerationPartials<F>, PrincipiaError> {
        let rx = state.position.x().value();
        let ry = state.position.y().value();
        let rz = state.position.z().value();
        let r2 = rx * rx + ry * ry + rz * rz;
        let r = r2.sqrt();
        if r < DEGENERATE_RADIUS_KM {
            return Err(PrincipiaError::DegenerateGeometry {
                reason: "radial magnitude below two-body degeneracy threshold",
            });
        }
        let inv_r3 = 1.0 / (r2 * r);
        let inv_r5 = inv_r3 / r2;
        let mu = self.mu.value();
        // A_r = -μ/r³ · (I - 3 r̂ r̂ᵀ)  = -μ/r³ I + 3μ/r⁵ · r rᵀ
        let m_xx = -mu * inv_r3 + 3.0 * mu * rx * rx * inv_r5;
        let m_yy = -mu * inv_r3 + 3.0 * mu * ry * ry * inv_r5;
        let m_zz = -mu * inv_r3 + 3.0 * mu * rz * rz * inv_r5;
        let m_xy = 3.0 * mu * rx * ry * inv_r5;
        let m_xz = 3.0 * mu * rx * rz * inv_r5;
        let m_yz = 3.0 * mu * ry * rz * inv_r5;
        let d_acc_d_pos = FrameMatrix3::<F>::from_array([
            [m_xx, m_xy, m_xz],
            [m_xy, m_yy, m_yz],
            [m_xz, m_yz, m_zz],
        ]);
        Ok(AccelerationPartials {
            d_acc_d_pos,
            d_acc_d_vel: FrameMatrix3::<F>::zero(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn make_state(r: f64) -> crate::state::DynamicsState<TT, Center, Inertial> {
        let mu = 398_600.441_8_f64;
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
    fn name_is_two_body() {
        let m = model();
        assert_eq!(
            <TwoBody as AccelerationModel<(), TT, Center, Inertial>>::name(&m),
            "two_body"
        );
    }

    #[test]
    fn acceleration_magnitude_approx_correct() {
        let r = 7000.0;
        let mu = 398_600.441_8_f64;
        let a = model().acceleration(&make_state(r), &()).unwrap();
        let a_mag = (a.x().value().powi(2) + a.y().value().powi(2) + a.z().value().powi(2)).sqrt();
        let expected = mu / (r * r);
        assert!((a_mag - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn acceleration_direction_is_minus_radial() {
        let a = model().acceleration(&make_state(7000.0), &()).unwrap();
        assert!(a.x().value() < 0.0);
        assert!(a.y().value().abs() < 1e-30);
    }

    #[test]
    fn acceleration_degenerate_returns_error() {
        let state = crate::state::DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(0.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(model().acceleration(&state, &()).is_err());
    }

    #[test]
    fn partials_symmetry() {
        let p = model().partials(&make_state(7000.0), &()).unwrap();
        let m = p.d_acc_d_pos.as_array();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (m[i][j] - m[j][i]).abs() < 1e-20,
                    "not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn partials_degenerate_returns_error() {
        let state = crate::state::DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(0.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(model().partials(&state, &()).is_err());
    }
}
