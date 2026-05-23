// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Generic zonal-`J2` oblateness perturbation.
//!
//! ## Scientific scope
//!
//! Implements the second-zonal (`J2`) gravitational perturbation
//!
//! ```text
//! a_J2 = (3/2) · J2 · μ · R² / r⁵ · ( (5 z²/r² − 1) · r − 2 z · ẑ )
//! ```
//!
//! parameterized over the central-body gravitational parameter `μ`, the
//! reference radius `R`, and the dimensionless `J2` coefficient. The
//! model assumes an inertial frame whose `+ẑ` axis is aligned with the
//! central body's spin pole.
//!
//! ## Technical scope
//!
//! [`J2`] is fully generic. Earth-specific convenience constructors live
//! in `siderust::astro::dynamics::earth`. No astronomy-specific
//! constants are baked into `principia`.
//!
//! ## References
//!
//! * Vallado, *Fundamentals of Astrodynamics and Applications*, §9.7.
//! * Montenbruck & Gill, *Satellite Orbits*, §3.2.5.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use affn::matrix3::FrameMatrix3;
use qtty::dynamics::GravitationalParameter;
use qtty::length::Kilometers;
use tempoch::ContinuousScale;

use crate::error::PrincipiaError;
use crate::models::{AccelerationModel, AccelerationPartials};
use crate::state::{Acceleration, DynamicsState};

/// Zonal-`J2` gravity acceleration model.
#[derive(Debug, Clone, Copy)]
pub struct J2 {
    /// Central-body gravitational parameter, km³/s².
    pub mu: GravitationalParameter,
    /// Central-body reference radius, km.
    pub r_ref: Kilometers,
    /// Dimensionless `J2` zonal coefficient.
    pub j2: f64,
}

impl J2 {
    /// Construct a `J2` model from typed parameters.
    #[inline]
    pub const fn new(mu: GravitationalParameter, r_ref: Kilometers, j2: f64) -> Self {
        Self { mu, r_ref, j2 }
    }
}

impl<Ctx, S, C, F> AccelerationModel<Ctx, S, C, F> for J2
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn name(&self) -> &'static str {
        "j2"
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
        if r < 100.0 {
            return Err(PrincipiaError::DegenerateGeometry {
                reason: "radial magnitude below J2 degeneracy threshold",
            });
        }
        let mu = self.mu.value();
        let r_eq = self.r_ref.value();
        let coef = 1.5 * self.j2 * mu * r_eq * r_eq / (r2 * r2 * r);
        let zr2 = (rz * rz) / r2;
        let common = 5.0 * zr2 - 1.0;
        Ok(Acceleration::<F>::new(
            coef * common * rx,
            coef * common * ry,
            coef * (common - 2.0) * rz,
        ))
    }

    fn partials(
        &self,
        state: &DynamicsState<S, C, F>,
        _ctx: &Ctx,
    ) -> Result<AccelerationPartials<F>, PrincipiaError> {
        let x = state.position.x().value();
        let y = state.position.y().value();
        let z = state.position.z().value();
        let r2 = x * x + y * y + z * z;
        let r = r2.sqrt();
        if r < 100.0 {
            return Err(PrincipiaError::DegenerateGeometry {
                reason: "radial magnitude below J2 degeneracy threshold",
            });
        }
        let req = self.r_ref.value();
        let mu = self.mu.value();
        let j2 = self.j2;
        let c = 1.5 * j2 * mu * req * req;
        let r7 = r2 * r2 * r2 * r;
        let d = c / r7;
        let q = (z * z) / r2;
        let diag_xy_coeff = (5.0 * q - 1.0) * r2;
        let off_xy = 5.0 * (7.0 * q - 1.0);
        let dxx = d * (diag_xy_coeff - off_xy * x * x);
        let dyy = d * (diag_xy_coeff - off_xy * y * y);
        let dzz = d * r2 * (30.0 * q - 3.0 - 35.0 * q * q);
        let dxy = d * 5.0 * (1.0 - 7.0 * q) * x * y;
        let dxz = d * 5.0 * (3.0 - 7.0 * q) * x * z;
        let dyz = d * 5.0 * (3.0 - 7.0 * q) * y * z;
        Ok(AccelerationPartials {
            d_acc_d_pos: FrameMatrix3::from_array([
                [dxx, dxy, dxz],
                [dxy, dyy, dyz],
                [dxz, dyz, dzz],
            ]),
            d_acc_d_vel: FrameMatrix3::zero(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use affn::centers::ReferenceCenter;
    use affn::frames::ReferenceFrame;
    use qtty::length::Kilometers;
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

    fn model() -> J2 {
        J2::new(
            GravitationalParameter::new(398_600.441_8),
            Kilometers::new(6_378.137),
            1.082_626_68e-3,
        )
    }

    fn state_equatorial(r: f64) -> crate::state::DynamicsState<TT, Center, Inertial> {
        crate::state::DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(r, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        )
    }

    fn state_degenerate() -> crate::state::DynamicsState<TT, Center, Inertial> {
        crate::state::DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(0.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        )
    }

    #[test]
    fn name_is_j2() {
        let m = model();
        assert_eq!(
            <J2 as AccelerationModel<(), TT, Center, Inertial>>::name(&m),
            "j2"
        );
    }

    #[test]
    fn acceleration_equatorial_z_component_zero() {
        let a = model()
            .acceleration(&state_equatorial(7000.0), &())
            .unwrap();
        assert!(a.z().value().abs() < 1e-20);
    }

    #[test]
    fn acceleration_degenerate_returns_error() {
        assert!(model().acceleration(&state_degenerate(), &()).is_err());
    }

    #[test]
    fn partials_d_acc_d_vel_is_zero() {
        let p = model().partials(&state_equatorial(7000.0), &()).unwrap();
        let v = p.d_acc_d_vel.as_array();
        for row in v {
            for val in row {
                assert_eq!(*val, 0.0);
            }
        }
    }

    #[test]
    fn partials_degenerate_returns_error() {
        assert!(model().partials(&state_degenerate(), &()).is_err());
    }
}
