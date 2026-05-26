// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

//! # principia — typed Newtonian numerical dynamics
//!
//! `principia` owns the reusable, domain-agnostic numerical mechanics layer:
//! typed Cartesian states, acceleration models, integrators, propagation,
//! variational equations, covariance transport, and gravity-field kernels.
//!
//! ## References
//!
//! * Vallado, *Fundamentals of Astrodynamics and Applications*.
//! * Montenbruck & Gill, *Satellite Orbits*.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(any(feature = "alloc", feature = "std"))]
extern crate alloc;

pub mod covariance;
pub mod error;
pub mod frames;
pub mod gravity;
pub mod integrators;
pub mod models;
pub mod propagation;
pub mod state;
pub mod variational;

pub use covariance::{ProcessNoise, StateCovariance};
pub use error::PrincipiaError;
pub use frames::{
    lvlh_from_raw_km_km_s, rtn_from_raw_km_km_s, vnc_from_raw_km_km_s, LocalTrajectoryFrame, LVLH,
    RTN, VNC,
};
#[cfg(any(feature = "alloc", feature = "std"))]
pub use gravity::{spherical_harmonic_acceleration, spherical_harmonic_acceleration_raw_km};
pub use gravity::{GravityConstants, GravityFieldProvider};
pub use integrators::{
    dop853_propagate, dop853_step, dopri5_propagate, dopri5_step, rk4_propagate, rk4_step,
    AdaptiveStepper, Dop853, Dop853Step, Dopri5, IntegratorTolerances, Rk4, Stepper,
};
#[cfg(any(feature = "alloc", feature = "std"))]
pub use models::CompositeModel;
pub use models::{AccelerationModel, AccelerationPartials, TwoBody, J2};
#[cfg(any(feature = "alloc", feature = "std"))]
pub use propagation::{propagate, PropagationConfig, PropagationResult};
pub use propagation::{
    EventDetector, EventDirection, EventOccurrence, PropagationError, RadialThresholdEvent,
};
pub use state::{DynamicsState, StateDerivative};
#[cfg(any(feature = "alloc", feature = "std"))]
pub use variational::finite_diff_stm_series;
pub use variational::{
    finite_diff_stm, propagate_stm, propagate_stm_with, StateTransitionMatrix, VariationalConfig,
};
