// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

//! Cartesian dynamics state and time-derivative aggregate.
//!
//! ## Scientific scope
//!
//! [`DynamicsState<S, C, F>`] is the typed Cartesian state `(r, v, t)` used
//! by all numerical Newtonian propagation in this crate. It is generic over
//! the continuous time scale `S`, the affine reference center `C`, and the
//! reference frame `F`. Defaults exist only via the type parameters of
//! downstream callers — `principia` does not bind a specific scale, frame,
//! or center.
//!
//! The Cartesian equations of motion that the integrators solve are:
//!
//! ```text
//! dr/dt = v
//! dv/dt = a(r, v, t)
//! ```
//!
//! where `a(r, v, t)` is supplied by an
//! [`AccelerationModel`](crate::models::AccelerationModel).
//!
//! ## Technical scope
//!
//! * Position is an [`affn::cartesian::Position<C, F, Kilometer>`] — typed
//!   length with frame and center tags enforced at compile time.
//! * Velocity is an [`affn::cartesian::Velocity<F, KmPerSecond>`].
//! * Epoch is a [`tempoch::Time<S>`] where `S: ContinuousScale`. Civil /
//!   discontinuous scales (e.g. UTC) cannot be used directly in
//!   propagation; downstream adapters must convert to a continuous scale
//!   before invoking the kernel.
//! * Time steps are typed [`qtty::Second`].
//!
//! ## References
//!
//! * Vallado, *Fundamentals of Astrodynamics and Applications*, §1.
//! * Montenbruck & Gill, *Satellite Orbits*, §3.1.

use affn::cartesian;
use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use qtty::unit::Kilometer;
use qtty::{KmPerSecond, KmPerSecondSquared, Second};
use tempoch::{ContinuousScale, Time};

use crate::error::PrincipiaError;

/// Default position type carried by [`DynamicsState`].
pub type Position<C, F> = cartesian::Position<C, F, Kilometer>;

/// Default velocity type carried by [`DynamicsState`].
pub type Velocity<F> = cartesian::Velocity<F, KmPerSecond>;

/// Default acceleration type produced by [`AccelerationModel`](crate::models::AccelerationModel).
pub type Acceleration<F> = cartesian::Acceleration<F, KmPerSecondSquared>;

/// Cartesian dynamics state `(epoch, position, velocity)` parameterized
/// over the continuous time scale `S`, the affine center `C`, and the
/// reference frame `F`.
#[derive(Debug)]
pub struct DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Epoch as a continuous [`Time<S>`] instant.
    pub epoch: Time<S>,
    /// Position in frame `F` centered on `C`, km.
    pub position: Position<C, F>,
    /// Velocity in frame `F`, km/s.
    pub velocity: Velocity<F>,
}

// Manual trait impls — `derive` cannot express the `C::Params: Copy` bound.

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "S: tempoch::Scale + tempoch::CoordinateScale, P: serde::Serialize",
    deserialize = "S: tempoch::Scale + tempoch::CoordinateScale, P: serde::Deserialize<'de>"
))]
struct DynamicsStateSerde<S: tempoch::Scale + tempoch::CoordinateScale, P> {
    epoch: Time<S>,
    center_params: P,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
}

#[cfg(feature = "serde")]
impl<S, C, F> serde::Serialize for DynamicsState<S, C, F>
where
    S: ContinuousScale + tempoch::Scale + tempoch::CoordinateScale,
    C: ReferenceCenter,
    C::Params: Clone + serde::Serialize,
    F: ReferenceFrame,
    Time<S>: serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        DynamicsStateSerde {
            epoch: self.epoch,
            center_params: self.position.center_params().clone(),
            position_km: [
                self.position.x().value(),
                self.position.y().value(),
                self.position.z().value(),
            ],
            velocity_km_s: [
                self.velocity.x().value(),
                self.velocity.y().value(),
                self.velocity.z().value(),
            ],
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, S, C, F> serde::Deserialize<'de> for DynamicsState<S, C, F>
where
    S: ContinuousScale + tempoch::Scale + tempoch::CoordinateScale,
    C: ReferenceCenter,
    C::Params: serde::Deserialize<'de>,
    F: ReferenceFrame,
    Time<S>: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = DynamicsStateSerde::<S, C::Params>::deserialize(deserializer)?;
        Ok(Self {
            epoch: helper.epoch,
            position: Position::<C, F>::new_with_params(
                helper.center_params,
                helper.position_km[0],
                helper.position_km[1],
                helper.position_km[2],
            ),
            velocity: Velocity::<F>::new(
                helper.velocity_km_s[0],
                helper.velocity_km_s[1],
                helper.velocity_km_s[2],
            ),
        })
    }
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct StateDerivativeSerde {
    vel_km_s: [f64; 3],
    acc_km_s2: [f64; 3],
}

#[cfg(feature = "serde")]
impl<F> serde::Serialize for StateDerivative<F>
where
    F: ReferenceFrame,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        StateDerivativeSerde {
            vel_km_s: [
                self.vel.x().value(),
                self.vel.y().value(),
                self.vel.z().value(),
            ],
            acc_km_s2: [
                self.acc.x().value(),
                self.acc.y().value(),
                self.acc.z().value(),
            ],
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, F> serde::Deserialize<'de> for StateDerivative<F>
where
    F: ReferenceFrame,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = StateDerivativeSerde::deserialize(deserializer)?;
        Ok(Self {
            vel: Velocity::<F>::new(helper.vel_km_s[0], helper.vel_km_s[1], helper.vel_km_s[2]),
            acc: Acceleration::<F>::new(
                helper.acc_km_s2[0],
                helper.acc_km_s2[1],
                helper.acc_km_s2[2],
            ),
        })
    }
}

impl<S, C, F> Clone for DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            position: self.position.clone(),
            velocity: self.velocity,
        }
    }
}

impl<S, C, F> Copy for DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Copy,
{
}

impl<S, C, F> PartialEq for DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn eq(&self, other: &Self) -> bool {
        self.epoch == other.epoch
            && self.position.x() == other.position.x()
            && self.position.y() == other.position.y()
            && self.position.z() == other.position.z()
            && self.velocity.x() == other.velocity.x()
            && self.velocity.y() == other.velocity.y()
            && self.velocity.z() == other.velocity.z()
    }
}

impl<S, C, F> DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Construct a `DynamicsState` from a continuous-scale epoch, a typed
    /// position, and a typed velocity.
    #[inline]
    pub fn new(epoch: Time<S>, position: Position<C, F>, velocity: Velocity<F>) -> Self {
        Self {
            epoch,
            position,
            velocity,
        }
    }

    /// Constructs a new [`DynamicsState`], returning [`PrincipiaError::NonFiniteValue`]
    /// if any position or velocity component is non-finite.
    pub fn try_new(
        epoch: Time<S>,
        position: Position<C, F>,
        velocity: Velocity<F>,
    ) -> Result<Self, PrincipiaError> {
        let state = Self {
            epoch,
            position,
            velocity,
        };
        if !state.is_finite() {
            return Err(PrincipiaError::NonFiniteValue {
                context: "DynamicsState: position or velocity component is non-finite",
            });
        }
        Ok(state)
    }

    /// Returns `true` if every component of position and velocity is finite.
    pub fn is_finite(&self) -> bool {
        self.position.x().value().is_finite()
            && self.position.y().value().is_finite()
            && self.position.z().value().is_finite()
            && self.velocity.x().value().is_finite()
            && self.velocity.y().value().is_finite()
            && self.velocity.z().value().is_finite()
    }

    /// Returns the Euclidean norm of the position vector in kilometres.
    pub fn position_norm(&self) -> f64 {
        let x = self.position.x().value();
        let y = self.position.y().value();
        let z = self.position.z().value();
        (x * x + y * y + z * z).sqrt()
    }

    /// Returns the Euclidean norm of the velocity vector in km s⁻¹.
    pub fn velocity_norm(&self) -> f64 {
        let x = self.velocity.x().value();
        let y = self.velocity.y().value();
        let z = self.velocity.z().value();
        (x * x + y * y + z * z).sqrt()
    }

    /// Advance position and velocity by `dt` along `deriv`.
    ///
    /// Performs a single forward Euler step:
    /// `x(t+h) = x(t) + h · ẋ(t)`. The epoch is **not** updated — the
    /// caller is responsible for advancing it.
    #[inline]
    pub fn advance(&self, deriv: &StateDerivative<F>, dt: Second) -> Self {
        let dt_s = dt.value();
        let new_pos = Position::<C, F>::new_with_params(
            self.position.center_params().clone(),
            self.position.x().value() + dt_s * deriv.vel.x().value(),
            self.position.y().value() + dt_s * deriv.vel.y().value(),
            self.position.z().value() + dt_s * deriv.vel.z().value(),
        );
        let new_vel = Velocity::<F>::new(
            self.velocity.x().value() + dt_s * deriv.acc.x().value(),
            self.velocity.y().value() + dt_s * deriv.acc.y().value(),
            self.velocity.z().value() + dt_s * deriv.acc.z().value(),
        );
        Self {
            epoch: self.epoch,
            position: new_pos,
            velocity: new_vel,
        }
    }

    /// Like [`advance`](Self::advance) but also advances the epoch by `dt`.
    #[inline]
    pub fn advance_with_epoch(&self, deriv: &StateDerivative<F>, dt: Second) -> Self {
        let mut next = self.advance(deriv, dt);
        next.epoch = self.epoch + dt;
        next
    }
}

/// Time derivative of a [`DynamicsState`]: `[dr/dt, dv/dt] = [v, a]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateDerivative<F>
where
    F: ReferenceFrame,
{
    /// Position rate (= velocity), km/s.
    pub vel: Velocity<F>,
    /// Velocity rate (= acceleration), km/s².
    pub acc: Acceleration<F>,
}

impl<F: ReferenceFrame> StateDerivative<F> {
    /// Construct from a typed velocity and typed acceleration in frame `F`.
    #[inline]
    pub fn new(vel: Velocity<F>, acc: Acceleration<F>) -> Self {
        Self { vel, acc }
    }

    /// Constructs a new [`StateDerivative`], returning [`PrincipiaError::NonFiniteValue`]
    /// if any component is non-finite.
    pub fn try_new(vel: Velocity<F>, acc: Acceleration<F>) -> Result<Self, PrincipiaError> {
        let deriv = Self { vel, acc };
        if !deriv.vel.x().value().is_finite()
            || !deriv.vel.y().value().is_finite()
            || !deriv.vel.z().value().is_finite()
            || !deriv.acc.x().value().is_finite()
            || !deriv.acc.y().value().is_finite()
            || !deriv.acc.z().value().is_finite()
        {
            return Err(PrincipiaError::NonFiniteValue {
                context: "StateDerivative: velocity or acceleration component is non-finite",
            });
        }
        Ok(deriv)
    }

    /// Return the velocity component (`dr/dt`).
    #[inline]
    pub fn velocity(&self) -> Velocity<F> {
        self.vel
    }

    /// Return the acceleration component (`dv/dt`).
    #[inline]
    pub fn acceleration(&self) -> Acceleration<F> {
        self.acc
    }

    /// Classical RK4 linear combination
    /// `(k1 + 2·k2 + 2·k3 + k4) / 6`.
    #[inline]
    pub fn rk4_combine(k1: &Self, k2: &Self, k3: &Self, k4: &Self) -> Self {
        let inv6 = 1.0 / 6.0;
        let vx = inv6
            * (k1.vel.x().value()
                + 2.0 * k2.vel.x().value()
                + 2.0 * k3.vel.x().value()
                + k4.vel.x().value());
        let vy = inv6
            * (k1.vel.y().value()
                + 2.0 * k2.vel.y().value()
                + 2.0 * k3.vel.y().value()
                + k4.vel.y().value());
        let vz = inv6
            * (k1.vel.z().value()
                + 2.0 * k2.vel.z().value()
                + 2.0 * k3.vel.z().value()
                + k4.vel.z().value());
        let ax = inv6
            * (k1.acc.x().value()
                + 2.0 * k2.acc.x().value()
                + 2.0 * k3.acc.x().value()
                + k4.acc.x().value());
        let ay = inv6
            * (k1.acc.y().value()
                + 2.0 * k2.acc.y().value()
                + 2.0 * k3.acc.y().value()
                + k4.acc.y().value());
        let az = inv6
            * (k1.acc.z().value()
                + 2.0 * k2.acc.z().value()
                + 2.0 * k3.acc.z().value()
                + k4.acc.z().value());
        Self {
            vel: Velocity::<F>::new(vx, vy, vz),
            acc: Acceleration::<F>::new(ax, ay, az),
        }
    }

    /// Scale this derivative by a dimensionless factor: `factor · (vel, acc)`.
    #[inline]
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            vel: self.vel.scale(factor),
            acc: self.acc.scale(factor),
        }
    }

    /// Element-wise addition of two derivatives.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            vel: self.vel + other.vel,
            acc: self.acc + other.acc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use affn::centers::ReferenceCenter;
    use affn::frames::ReferenceFrame;
    use qtty::unit::Kilometer;
    use qtty::{KmPerSecond, KmPerSecondSquared, Second};
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

    fn make_state() -> DynamicsState<TT, Center, Inertial> {
        DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(7000.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        )
    }

    fn make_deriv() -> StateDerivative<Inertial> {
        StateDerivative::new(
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
            affn::cartesian::Acceleration::<Inertial, KmPerSecondSquared>::new(-0.8, 0.0, 0.0),
        )
    }

    #[test]
    fn partial_eq_same_state() {
        let s = make_state();
        let t = make_state();
        assert_eq!(s, t);
    }

    #[test]
    fn try_new_rejects_non_finite_position() {
        let epoch = Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap();
        let result = DynamicsState::try_new(
            epoch,
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(f64::NAN, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(matches!(result, Err(PrincipiaError::NonFiniteValue { .. })));
    }

    #[test]
    fn is_finite_detects_nan_component() {
        let state = DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(f64::NAN, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(!state.is_finite());
    }

    #[test]
    fn position_norm_matches_euclidean_length() {
        let state = DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(3.0, 4.0, 12.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 0.0, 0.0),
        );
        assert!((state.position_norm() - 13.0).abs() < 1e-12);
    }

    #[test]
    fn velocity_norm_matches_euclidean_length() {
        let state = DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(0.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(2.0, 3.0, 6.0),
        );
        assert!((state.velocity_norm() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn state_derivative_try_new_rejects_non_finite_component() {
        let result = StateDerivative::try_new(
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
            affn::cartesian::Acceleration::<Inertial, KmPerSecondSquared>::new(
                f64::INFINITY,
                0.0,
                0.0,
            ),
        );
        assert!(matches!(result, Err(PrincipiaError::NonFiniteValue { .. })));
    }

    #[test]
    fn advance_with_epoch_advances_epoch() {
        let s = make_state();
        let d = make_deriv();
        let dt = Second::new(10.0);
        let s2 = s.advance_with_epoch(&d, dt);
        assert!((s2.epoch - s.epoch - dt).value().abs() < 1e-12);
    }

    #[test]
    fn state_derivative_velocity_accessor() {
        let d = make_deriv();
        assert_eq!(d.velocity().x().value(), d.vel.x().value());
    }

    #[test]
    fn state_derivative_acceleration_accessor() {
        let d = make_deriv();
        assert_eq!(d.acceleration().x().value(), d.acc.x().value());
    }

    #[test]
    fn scaled_doubles_components() {
        let d = make_deriv();
        let d2 = d.scaled(2.0);
        assert!((d2.vel.y().value() - 2.0 * d.vel.y().value()).abs() < 1e-14);
        assert!((d2.acc.x().value() - 2.0 * d.acc.x().value()).abs() < 1e-14);
    }

    #[test]
    fn add_sums_components() {
        let d = make_deriv();
        let d2 = d.add(&d);
        assert!((d2.vel.y().value() - 2.0 * d.vel.y().value()).abs() < 1e-14);
    }

    #[test]
    fn rk4_combine_uniform_gives_same_as_scaled() {
        let d = make_deriv();
        let combined = StateDerivative::rk4_combine(&d, &d, &d, &d);
        assert!((combined.vel.x().value() - d.vel.x().value()).abs() < 1e-14);
        assert!((combined.acc.x().value() - d.acc.x().value()).abs() < 1e-14);
    }

    #[test]
    fn advance_does_not_update_epoch() {
        let s = make_state();
        let d = make_deriv();
        let dt = Second::new(5.0);
        let s2 = s.advance(&d, dt);
        // advance() leaves epoch unchanged
        assert_eq!(s2.epoch, s.epoch);
        // but advance_with_epoch() does update it
        let s3 = s.advance_with_epoch(&d, dt);
        assert!((s3.epoch - s.epoch - dt).value().abs() < 1e-12);
    }

    #[test]
    fn is_finite_true_for_valid_state() {
        assert!(make_state().is_finite());
    }

    #[test]
    fn try_new_accepts_finite_state() {
        let epoch = Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap();
        let result = DynamicsState::try_new(
            epoch,
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(7000.0, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn state_derivative_try_new_accepts_finite() {
        let result = StateDerivative::try_new(
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
            affn::cartesian::Acceleration::<Inertial, KmPerSecondSquared>::new(-0.8, 0.0, 0.0),
        );
        assert!(result.is_ok());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn dynamics_state_serde_roundtrip() {
        let epoch = Time::<TT>::from_raw_j2000_seconds(Second::new(12345.0)).unwrap();
        let s = DynamicsState::new(
            epoch,
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(7000.0, 100.0, -50.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.1, 7.5, 0.2),
        );
        let json = serde_json::to_string(&s).expect("serialize");
        let s2: DynamicsState<TT, Center, Inertial> =
            serde_json::from_str(&json).expect("deserialize");
        assert!((s.position.x().value() - s2.position.x().value()).abs() < 1e-12);
        assert!((s.position.y().value() - s2.position.y().value()).abs() < 1e-12);
        assert!((s.position.z().value() - s2.position.z().value()).abs() < 1e-12);
        assert!((s.velocity.x().value() - s2.velocity.x().value()).abs() < 1e-12);
        assert!((s.velocity.y().value() - s2.velocity.y().value()).abs() < 1e-12);
        assert!((s.velocity.z().value() - s2.velocity.z().value()).abs() < 1e-12);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn state_derivative_serde_roundtrip() {
        let d = make_deriv();
        let json = serde_json::to_string(&d).expect("serialize");
        let d2: StateDerivative<Inertial> = serde_json::from_str(&json).expect("deserialize");
        assert!((d.vel.x().value() - d2.vel.x().value()).abs() < 1e-12);
        assert!((d.acc.x().value() - d2.acc.x().value()).abs() < 1e-12);
    }
}
