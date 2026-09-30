// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

//! Acceleration models and composition.
//!
//! ## Scientific scope
//!
//! An [`AccelerationModel`] returns the inertial acceleration `a(r, v, t)`
//! produced by some physical effect (gravity, drag, third-body, …). Models
//! evaluated on the same state can be summed component-wise into a
//! [`CompositeModel`] sharing one caller-owned context and error surface.
//!
//! The Jacobian `∂a/∂[r, v]` is exposed through
//! [`AccelerationPartials`]; models that do not implement analytic
//! partials return [`PrincipiaError::PartialsUnavailable`].
//!
//! ## Technical scope
//!
//! The trait is generic over the caller-owned context type `Ctx`. The
//! mechanics kernel does not own `Ctx`; downstream adapters (e.g.
//! `siderust::astro::dynamics::PerturbationContext`) supply ephemeris
//! / atmosphere / EOP slots.
//!
//! ## References
//!
//! * Montenbruck & Gill, *Satellite Orbits*, §3.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use affn::matrix3::FrameMatrix3;
use tempoch::ContinuousScale;

use crate::error::PrincipiaError;
use crate::state::{Acceleration, DynamicsState};

pub mod j2;
pub mod two_body;

pub use j2::J2;
pub use two_body::TwoBody;

/// Frame-tagged Jacobian blocks of the acceleration: `∂a/∂[r, v]`.
///
/// `d_acc_d_pos = A_r = ∂a/∂r` has units of `1/s²`.
/// `d_acc_d_vel = A_v = ∂a/∂v` has units of `1/s`.
/// For conservative forces (gravity), `A_v` is zero.
#[derive(Debug, Clone, Copy)]
pub struct AccelerationPartials<F: ReferenceFrame> {
    /// `∂a/∂r` in frame `F` (units: `1/s²`).
    pub d_acc_d_pos: FrameMatrix3<F>,
    /// `∂a/∂v` in frame `F` (units: `1/s`). Zero for conservative forces.
    pub d_acc_d_vel: FrameMatrix3<F>,
}

#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct AccelerationPartialsSerde {
    d_acc_d_pos: [[f64; 3]; 3],
    d_acc_d_vel: [[f64; 3]; 3],
}

#[cfg(feature = "serde")]
impl<F: ReferenceFrame> serde::Serialize for AccelerationPartials<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        AccelerationPartialsSerde {
            d_acc_d_pos: *self.d_acc_d_pos.as_array(),
            d_acc_d_vel: *self.d_acc_d_vel.as_array(),
        }
        .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, F: ReferenceFrame> serde::Deserialize<'de> for AccelerationPartials<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let helper = AccelerationPartialsSerde::deserialize(deserializer)?;
        Ok(Self {
            d_acc_d_pos: FrameMatrix3::from_array(helper.d_acc_d_pos),
            d_acc_d_vel: FrameMatrix3::from_array(helper.d_acc_d_vel),
        })
    }
}

impl<F: ReferenceFrame> AccelerationPartials<F> {
    /// Neutral element: both Jacobian blocks are zero matrices.
    pub fn zero() -> Self {
        Self {
            d_acc_d_pos: FrameMatrix3::zero(),
            d_acc_d_vel: FrameMatrix3::zero(),
        }
    }
}

/// Acceleration model trait.
///
/// Implementors return the inertial acceleration produced by their physical
/// effect on the given state, using data supplied through the caller-owned
/// context `Ctx`. The default `partials` implementation reports that
/// analytic partials are not available; analytic models override it.
pub trait AccelerationModel<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Short model identifier used in diagnostics (e.g. `"two_body"`).
    fn name(&self) -> &'static str;

    /// Evaluate the inertial acceleration at the given state.
    fn acceleration(
        &self,
        state: &DynamicsState<S, C, F>,
        ctx: &Ctx,
    ) -> Result<Acceleration<F>, PrincipiaError>;

    /// Evaluate the Jacobian `∂a/∂[r, v]`. Defaults to
    /// [`PrincipiaError::PartialsUnavailable`].
    fn partials(
        &self,
        _state: &DynamicsState<S, C, F>,
        _ctx: &Ctx,
    ) -> Result<AccelerationPartials<F>, PrincipiaError> {
        Err(PrincipiaError::PartialsUnavailable { model: self.name() })
    }
}

/// Linear sum of [`AccelerationModel`] components sharing the same `Ctx`,
/// state, frame, and error surface.
#[cfg(any(feature = "alloc", feature = "std"))]
pub struct CompositeModel<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    models: alloc::vec::Vec<alloc::boxed::Box<dyn AccelerationModel<Ctx, S, C, F>>>,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl<Ctx, S, C, F> Default for CompositeModel<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn default() -> Self {
        Self::empty()
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl<Ctx, S, C, F> CompositeModel<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Construct an empty composite (no models).
    pub fn empty() -> Self {
        Self {
            models: alloc::vec::Vec::new(),
        }
    }

    /// Append a model to the composite.
    pub fn push(mut self, model: alloc::boxed::Box<dyn AccelerationModel<Ctx, S, C, F>>) -> Self {
        self.models.push(model);
        self
    }

    /// Number of contained models.
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns `true` if no models are registered.
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl<Ctx, S, C, F> AccelerationModel<Ctx, S, C, F> for CompositeModel<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn name(&self) -> &'static str {
        "composite"
    }

    fn acceleration(
        &self,
        state: &DynamicsState<S, C, F>,
        ctx: &Ctx,
    ) -> Result<Acceleration<F>, PrincipiaError> {
        let mut ax = 0.0_f64;
        let mut ay = 0.0_f64;
        let mut az = 0.0_f64;
        for m in &self.models {
            let a = m.acceleration(state, ctx)?;
            ax += a.x().value();
            ay += a.y().value();
            az += a.z().value();
        }
        Ok(Acceleration::<F>::new(ax, ay, az))
    }

    fn partials(
        &self,
        state: &DynamicsState<S, C, F>,
        ctx: &Ctx,
    ) -> Result<AccelerationPartials<F>, PrincipiaError> {
        let mut acc_r = FrameMatrix3::<F>::zero();
        let mut acc_v = FrameMatrix3::<F>::zero();
        for m in &self.models {
            let p = m.partials(state, ctx)?;
            acc_r.add_in_place(&p.d_acc_d_pos);
            acc_v.add_in_place(&p.d_acc_d_vel);
        }
        Ok(AccelerationPartials {
            d_acc_d_pos: acc_r,
            d_acc_d_vel: acc_v,
        })
    }
}

#[cfg(all(test, any(feature = "alloc", feature = "std")))]
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

    fn make_state() -> DynamicsState<TT, Center, Inertial> {
        let mu = 398_600.441_8_f64;
        let r = 7000.0_f64;
        let v = (mu / r).sqrt();
        DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, Kilometer>::new(r, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, v, 0.0),
        )
    }

    struct MinimalModel;
    impl<Ctx, S: tempoch::ContinuousScale, C: ReferenceCenter, F: ReferenceFrame>
        AccelerationModel<Ctx, S, C, F> for MinimalModel
    {
        fn name(&self) -> &'static str {
            "minimal"
        }
        fn acceleration(
            &self,
            _s: &DynamicsState<S, C, F>,
            _ctx: &Ctx,
        ) -> Result<Acceleration<F>, PrincipiaError> {
            Ok(Acceleration::<F>::new(0.0, 0.0, 0.0))
        }
    }

    #[test]
    fn default_partials_returns_unavailable() {
        let state = make_state();
        let result = <MinimalModel as AccelerationModel<(), TT, Center, Inertial>>::partials(
            &MinimalModel,
            &state,
            &(),
        );
        assert!(matches!(
            result,
            Err(PrincipiaError::PartialsUnavailable { .. })
        ));
    }

    #[test]
    fn acceleration_partials_zero() {
        let p = AccelerationPartials::<Inertial>::zero();
        let arr = p.d_acc_d_pos.as_array();
        assert_eq!(arr[0][0], 0.0);
    }

    #[test]
    fn composite_empty_name_and_counts() {
        let c = CompositeModel::<(), TT, Center, Inertial>::empty();
        assert_eq!(c.name(), "composite");
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn composite_default_is_empty() {
        let c = CompositeModel::<(), TT, Center, Inertial>::default();
        assert!(c.is_empty());
    }

    #[test]
    fn composite_push_increments_len() {
        let mu = GravitationalParameter::new(398_600.441_8);
        let c =
            CompositeModel::<(), TT, Center, Inertial>::empty().push(Box::new(TwoBody::new(mu)));
        assert_eq!(c.len(), 1);
        assert!(!c.is_empty());
    }

    #[test]
    fn composite_acceleration_matches_single_model() {
        let mu = GravitationalParameter::new(398_600.441_8);
        let single = TwoBody::new(mu);
        let composite =
            CompositeModel::<(), TT, Center, Inertial>::empty().push(Box::new(TwoBody::new(mu)));
        let state = make_state();
        let a_single = single.acceleration(&state, &()).unwrap();
        let a_composite = composite.acceleration(&state, &()).unwrap();
        assert!((a_single.x().value() - a_composite.x().value()).abs() < 1e-20);
    }

    #[test]
    fn composite_partials_matches_single_model() {
        let mu = GravitationalParameter::new(398_600.441_8);
        let single = TwoBody::new(mu);
        let composite =
            CompositeModel::<(), TT, Center, Inertial>::empty().push(Box::new(TwoBody::new(mu)));
        let state = make_state();
        let p_single = single.partials(&state, &()).unwrap();
        let p_composite = composite.partials(&state, &()).unwrap();
        let m1 = p_single.d_acc_d_pos.as_array();
        let m2 = p_composite.d_acc_d_pos.as_array();
        for i in 0..3 {
            for j in 0..3 {
                assert!((m1[i][j] - m2[i][j]).abs() < 1e-20);
            }
        }
    }
}
