// SPDX-License-Identifier: AGPL-3.0-or-later
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
}
