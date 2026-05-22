// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Local trajectory frames.
//!
//! ## Scientific scope
//!
//! Provides the state-dependent local frames commonly used in orbit
//! mechanics:
//!
//! * [`RTN`] — radial / transverse / normal.
//! * [`VNC`] — velocity / normal / co-normal.
//! * [`LVLH`] — local-vertical / local-horizontal.
//!
//! ## Technical scope
//!
//! [`LocalTrajectoryFrame<Inertial, Local>`] stores the inertial-to-local
//! direction-cosine matrix for a specific Cartesian state. Constructors are
//! generic over the propagated continuous time scale, reference center, and
//! inertial frame.
//!
//! ## References
//!
//! * Vallado, *Fundamentals of Astrodynamics and Applications*, §3.3.

use core::marker::PhantomData;

use affn::cartesian::{Direction, Displacement};
use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use affn::matrix3::FrameMatrix3;
use affn::ops::Rotation3;
use qtty::unit::Kilometer;
use qtty::{KmPerSecond, Quantity};
use tempoch::ContinuousScale;

use crate::error::PrincipiaError;
use crate::state::DynamicsState;

const POS_THRESHOLD_KM: f64 = 1e-9;
const VEL_THRESHOLD_KM_S: f64 = 1e-9;

/// Radial / transverse / normal local trajectory frame marker.
#[derive(Debug, Clone, Copy)]
pub struct RTN;
impl ReferenceFrame for RTN {
    fn frame_name() -> &'static str {
        "RTN"
    }
}

/// Velocity / normal / co-normal local trajectory frame marker.
#[derive(Debug, Clone, Copy)]
pub struct VNC;
impl ReferenceFrame for VNC {
    fn frame_name() -> &'static str {
        "VNC"
    }
}

/// Local-vertical / local-horizontal local trajectory frame marker.
#[derive(Debug, Clone, Copy)]
pub struct LVLH;
impl ReferenceFrame for LVLH {
    fn frame_name() -> &'static str {
        "LVLH"
    }
}

/// Materialized inertial-to-local direction-cosine matrix.
#[derive(Debug, Clone, Copy)]
pub struct LocalTrajectoryFrame<Inertial, Local>
where
    Inertial: ReferenceFrame,
    Local: ReferenceFrame,
{
    /// Inertial-to-local direction-cosine matrix.
    pub dcm: FrameMatrix3<Inertial>,
    _marker: PhantomData<Local>,
}

impl<Inertial, Local> LocalTrajectoryFrame<Inertial, Local>
where
    Inertial: ReferenceFrame,
    Local: ReferenceFrame,
{
    /// Construct from an inertial-to-local direction-cosine matrix.
    #[inline]
    pub fn from_dcm(dcm: FrameMatrix3<Inertial>) -> Self {
        Self {
            dcm,
            _marker: PhantomData,
        }
    }

    /// Return the inertial-to-local direction-cosine matrix.
    #[inline]
    pub fn dcm(&self) -> FrameMatrix3<Inertial> {
        self.dcm
    }

    /// Return the inertial-to-local rotation.
    #[inline]
    pub fn rotation(&self) -> Rotation3 {
        Rotation3::from_matrix_unchecked(*self.dcm.as_array())
    }

    /// Return the local-to-inertial rotation.
    #[inline]
    pub fn rotation_inverse(&self) -> Rotation3 {
        self.rotation().transpose()
    }

    /// Rotate an inertial displacement into the local frame.
    #[inline]
    pub fn to_local(&self, v: Displacement<Inertial, Kilometer>) -> Displacement<Local, Kilometer> {
        (self.rotation() * v).reinterpret_frame()
    }

    /// Rotate a local displacement back into the inertial frame.
    #[inline]
    pub fn to_inertial(
        &self,
        v: Displacement<Local, Kilometer>,
    ) -> Displacement<Inertial, Kilometer> {
        (self.rotation_inverse() * v).reinterpret_frame()
    }
}

fn checked_position_direction<S, C, Inertial>(
    state: &DynamicsState<S, C, Inertial>,
) -> Result<Direction<Inertial>, PrincipiaError>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    Inertial: ReferenceFrame,
{
    let threshold = Quantity::<Kilometer>::new(POS_THRESHOLD_KM);
    if state.position.distance() <= threshold {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "zero position magnitude in local frame construction",
        });
    }
    state
        .position
        .direction()
        .ok_or(PrincipiaError::DegenerateGeometry {
            reason: "zero position magnitude in local frame construction",
        })
}

fn checked_velocity_direction<S, C, Inertial>(
    state: &DynamicsState<S, C, Inertial>,
) -> Result<Direction<Inertial>, PrincipiaError>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    Inertial: ReferenceFrame,
{
    let threshold = Quantity::<KmPerSecond>::new(VEL_THRESHOLD_KM_S);
    if state.velocity.magnitude() <= threshold {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "zero velocity magnitude in local frame construction",
        });
    }
    Direction::try_new(
        state.velocity.x().value(),
        state.velocity.y().value(),
        state.velocity.z().value(),
    )
    .ok_or(PrincipiaError::DegenerateGeometry {
        reason: "zero velocity magnitude in local frame construction",
    })
}

impl<Inertial> LocalTrajectoryFrame<Inertial, RTN>
where
    Inertial: ReferenceFrame,
{
    /// Build the RTN frame from a Cartesian dynamics state.
    pub fn try_from_state<S, C>(
        state: &DynamicsState<S, C, Inertial>,
    ) -> Result<Self, PrincipiaError>
    where
        S: ContinuousScale,
        C: ReferenceCenter,
    {
        let r_hat = checked_position_direction(state)?;
        let v_hat = checked_velocity_direction(state)?;
        let n_hat = r_hat
            .cross(&v_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in RTN frame construction",
            })?;
        let t_hat = n_hat
            .cross(&r_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in RTN frame construction",
            })?;
        Ok(Self::from_dcm(FrameMatrix3::from_array([
            r_hat.as_array(),
            t_hat.as_array(),
            n_hat.as_array(),
        ])))
    }
}

impl<Inertial> LocalTrajectoryFrame<Inertial, VNC>
where
    Inertial: ReferenceFrame,
{
    /// Build the VNC frame from a Cartesian dynamics state.
    pub fn try_from_state<S, C>(
        state: &DynamicsState<S, C, Inertial>,
    ) -> Result<Self, PrincipiaError>
    where
        S: ContinuousScale,
        C: ReferenceCenter,
    {
        let v_hat = checked_velocity_direction(state)?;
        let r_hat = checked_position_direction(state)?;
        let n_hat = r_hat
            .cross(&v_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in VNC frame construction",
            })?;
        let c_hat = v_hat
            .cross(&n_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in VNC frame construction",
            })?;
        Ok(Self::from_dcm(FrameMatrix3::from_array([
            v_hat.as_array(),
            n_hat.as_array(),
            c_hat.as_array(),
        ])))
    }
}

impl<Inertial> LocalTrajectoryFrame<Inertial, LVLH>
where
    Inertial: ReferenceFrame,
{
    /// Build the LVLH frame from a Cartesian dynamics state.
    pub fn try_from_state<S, C>(
        state: &DynamicsState<S, C, Inertial>,
    ) -> Result<Self, PrincipiaError>
    where
        S: ContinuousScale,
        C: ReferenceCenter,
    {
        let r_hat = checked_position_direction(state)?;
        let v_hat = checked_velocity_direction(state)?;
        let z_hat = r_hat.negate();
        let x_hat = r_hat
            .cross(&v_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in LVLH frame construction",
            })?
            .cross(&z_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in LVLH frame construction",
            })?;
        let y_hat = z_hat
            .cross(&x_hat)
            .ok_or(PrincipiaError::DegenerateGeometry {
                reason: "position and velocity are parallel in LVLH frame construction",
            })?;
        Ok(Self::from_dcm(FrameMatrix3::from_array([
            x_hat.as_array(),
            y_hat.as_array(),
            z_hat.as_array(),
        ])))
    }
}

/// Construct an RTN frame from raw Cartesian state components.
pub fn rtn_from_state<Inertial: ReferenceFrame>(
    r: [f64; 3],
    v: [f64; 3],
) -> Result<LocalTrajectoryFrame<Inertial, RTN>, PrincipiaError> {
    let r_norm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    if r_norm <= POS_THRESHOLD_KM {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "zero position magnitude in RTN frame construction",
        });
    }
    let r_hat = [r[0] / r_norm, r[1] / r_norm, r[2] / r_norm];
    let h = [
        r[1] * v[2] - r[2] * v[1],
        r[2] * v[0] - r[0] * v[2],
        r[0] * v[1] - r[1] * v[0],
    ];
    let h_norm = (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt();
    if h_norm == 0.0 {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "position and velocity are parallel in RTN frame construction",
        });
    }
    let n_hat = [h[0] / h_norm, h[1] / h_norm, h[2] / h_norm];
    let t_hat = [
        n_hat[1] * r_hat[2] - n_hat[2] * r_hat[1],
        n_hat[2] * r_hat[0] - n_hat[0] * r_hat[2],
        n_hat[0] * r_hat[1] - n_hat[1] * r_hat[0],
    ];
    Ok(LocalTrajectoryFrame::from_dcm(FrameMatrix3::from_array([
        r_hat, t_hat, n_hat,
    ])))
}

/// Construct a VNC frame from raw Cartesian state components.
pub fn vnc_from_state<Inertial: ReferenceFrame>(
    r: [f64; 3],
    v: [f64; 3],
) -> Result<LocalTrajectoryFrame<Inertial, VNC>, PrincipiaError> {
    let v_norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if v_norm <= VEL_THRESHOLD_KM_S {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "zero velocity magnitude in VNC frame construction",
        });
    }
    let v_hat = [v[0] / v_norm, v[1] / v_norm, v[2] / v_norm];
    let rtn = rtn_from_state::<Inertial>(r, v)?;
    let n_hat = rtn.dcm.as_array()[2];
    let c_hat = [
        v_hat[1] * n_hat[2] - v_hat[2] * n_hat[1],
        v_hat[2] * n_hat[0] - v_hat[0] * n_hat[2],
        v_hat[0] * n_hat[1] - v_hat[1] * n_hat[0],
    ];
    Ok(LocalTrajectoryFrame::from_dcm(FrameMatrix3::from_array([
        v_hat, n_hat, c_hat,
    ])))
}

/// Construct an LVLH frame from raw Cartesian state components.
pub fn lvlh_from_state<Inertial: ReferenceFrame>(
    r: [f64; 3],
    v: [f64; 3],
) -> Result<LocalTrajectoryFrame<Inertial, LVLH>, PrincipiaError> {
    let rtn = rtn_from_state::<Inertial>(r, v)?;
    let r_hat = rtn.dcm.as_array()[0];
    let n_hat = rtn.dcm.as_array()[2];
    let z_hat = [-r_hat[0], -r_hat[1], -r_hat[2]];
    let x_hat = [
        n_hat[1] * z_hat[2] - n_hat[2] * z_hat[1],
        n_hat[2] * z_hat[0] - n_hat[0] * z_hat[2],
        n_hat[0] * z_hat[1] - n_hat[1] * z_hat[0],
    ];
    let y_hat = [
        z_hat[1] * x_hat[2] - z_hat[2] * x_hat[1],
        z_hat[2] * x_hat[0] - z_hat[0] * x_hat[2],
        z_hat[0] * x_hat[1] - z_hat[1] * x_hat[0],
    ];
    Ok(LocalTrajectoryFrame::from_dcm(FrameMatrix3::from_array([
        x_hat, y_hat, z_hat,
    ])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use affn::centers::ReferenceCenter;
    use qtty::Second;
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

    fn circular_state() -> DynamicsState<TT, Center, Inertial> {
        DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(7000.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        )
    }

    #[test]
    fn rtn_axes_match_circular_orbit() {
        let frame =
            LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&circular_state()).unwrap();
        let m = frame.dcm.as_array();
        assert!((m[0][0] - 1.0).abs() < 1e-12);
        assert!((m[1][1] - 1.0).abs() < 1e-12);
        assert!((m[2][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn vnc_velocity_axis_matches_velocity() {
        let frame =
            LocalTrajectoryFrame::<Inertial, VNC>::try_from_state(&circular_state()).unwrap();
        let m = frame.dcm.as_array();
        assert!(m[0][0].abs() < 1e-12);
        assert!((m[0][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lvlh_inward_axis_matches_minus_radial() {
        let frame =
            LocalTrajectoryFrame::<Inertial, LVLH>::try_from_state(&circular_state()).unwrap();
        let m = frame.dcm.as_array();
        assert!((m[2][0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn rtn_from_state_free_function_matches_try_from_state() {
        let s = circular_state();
        let r = [
            s.position.x().value(),
            s.position.y().value(),
            s.position.z().value(),
        ];
        let v = [
            s.velocity.x().value(),
            s.velocity.y().value(),
            s.velocity.z().value(),
        ];
        let f1 = LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&s).unwrap();
        let f2 = rtn_from_state::<Inertial>(r, v).unwrap();
        let m1 = f1.dcm.as_array();
        let m2 = f2.dcm.as_array();
        for i in 0..3 {
            for j in 0..3 {
                assert!((m1[i][j] - m2[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn vnc_from_state_free_function_succeeds() {
        let s = circular_state();
        let r = [
            s.position.x().value(),
            s.position.y().value(),
            s.position.z().value(),
        ];
        let v = [
            s.velocity.x().value(),
            s.velocity.y().value(),
            s.velocity.z().value(),
        ];
        let frame = vnc_from_state::<Inertial>(r, v).unwrap();
        let m = frame.dcm.as_array();
        // First row should be velocity direction (y-axis for our state)
        assert!(m[0][0].abs() < 1e-12);
        assert!((m[0][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lvlh_from_state_free_function_succeeds() {
        let s = circular_state();
        let r = [
            s.position.x().value(),
            s.position.y().value(),
            s.position.z().value(),
        ];
        let v = [
            s.velocity.x().value(),
            s.velocity.y().value(),
            s.velocity.z().value(),
        ];
        let frame = lvlh_from_state::<Inertial>(r, v).unwrap();
        let m = frame.dcm.as_array();
        // z row should be -radial = [-1,0,0]
        assert!((m[2][0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn rtn_from_state_zero_position_error() {
        let result = rtn_from_state::<Inertial>([0.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn rtn_from_state_parallel_pos_vel_error() {
        // Parallel r and v => no angular momentum
        let result = rtn_from_state::<Inertial>([7000.0, 0.0, 0.0], [7.5, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn vnc_from_state_zero_velocity_error() {
        let result = vnc_from_state::<Inertial>([7000.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn try_from_state_rtn_degenerate_returns_error() {
        let s = DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(0.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&s).is_err());
    }

    #[test]
    fn try_from_state_vnc_degenerate_returns_error() {
        let s = DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(7000.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 0.0, 0.0),
        );
        assert!(LocalTrajectoryFrame::<Inertial, VNC>::try_from_state(&s).is_err());
    }

    #[test]
    fn try_from_state_lvlh_degenerate_returns_error() {
        let s = DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(0.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(LocalTrajectoryFrame::<Inertial, LVLH>::try_from_state(&s).is_err());
    }

    #[test]
    fn dcm_accessor_returns_dcm() {
        let frame =
            LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&circular_state()).unwrap();
        let dcm = frame.dcm();
        let m = dcm.as_array();
        assert!((m[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotation_and_inverse_are_transposes() {
        let frame =
            LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&circular_state()).unwrap();
        let r = frame.rotation();
        let ri = frame.rotation_inverse();
        // r applied then ri applied should give back the original vector
        let v = [1.0, 2.0, 3.0];
        let rv = r.apply_array(v);
        let back = ri.apply_array(rv);
        assert!((back[0] - v[0]).abs() < 1e-12);
        assert!((back[1] - v[1]).abs() < 1e-12);
        assert!((back[2] - v[2]).abs() < 1e-12);
    }

    #[test]
    fn to_local_and_to_inertial_roundtrip() {
        use affn::cartesian::Displacement;
        let frame =
            LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&circular_state()).unwrap();
        let v = Displacement::<Inertial, Kilometer>::new(1.0, 0.0, 0.0);
        let local = frame.to_local(v);
        let back = frame.to_inertial(local);
        assert!((back.x().value() - 1.0).abs() < 1e-12);
        assert!(back.y().value().abs() < 1e-12);
        assert!(back.z().value().abs() < 1e-12);
    }
}
