// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Crate-level error family for `principia`.
//!
//! ## Scientific scope
//!
//! Covers the failure modes the numerical mechanics kernel can encounter
//! without binding to any astronomy provider semantics:
//!
//! * degenerate geometry (zero position vector, parallel `r × v`, etc.),
//! * invalid integrator step requests,
//! * unsupported/missing partials,
//! * propagation failures (step rejection cascade, NaN detection),
//! * generic provider unavailability at the trait boundary
//!   (gravity-field coefficient, caller-owned context member).
//!
//! Provider-specific astronomy errors (ephemeris, EOP, atmosphere)
//! belong downstream in `siderust::astro::perturbations`.
//!
//! ## Technical scope
//!
//! [`PrincipiaError`] is `Send + Sync + 'static` and implements
//! [`core::error::Error`] (or `std::error::Error` under `std`). It is
//! `#[non_exhaustive]` — new variants may be added as the kernel grows.
//!
//! ## References
//!
//! * Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I*, §II.4.

use core::fmt;

/// Errors produced by `principia` mechanics kernels.
#[derive(Debug)]
#[non_exhaustive]
pub enum PrincipiaError {
    /// A geometric computation degenerated (e.g. zero cross-product,
    /// zero radial magnitude, singular rotation matrix).
    DegenerateGeometry {
        /// Short human-readable explanation of what went wrong.
        reason: &'static str,
    },

    /// An integration step size, count, or tolerance is invalid.
    InvalidStepRequest {
        /// Short human-readable explanation of what constraint was violated.
        reason: &'static str,
    },

    /// The adaptive step controller failed to converge within the iteration
    /// budget (typically 50 inner iterations).
    StepControlFailed {
        /// Short human-readable explanation.
        reason: &'static str,
    },

    /// The adaptive step controller shrunk the step below the configured
    /// `h_min`; the tolerances may be too tight for the current model.
    StepBelowMinimum {
        /// Short human-readable explanation.
        reason: &'static str,
    },

    /// A spherical-harmonic coefficient at degree/order `(n, m)` is not
    /// available from the current gravity-field provider.
    GravityCoefficientUnavailable {
        /// Spherical-harmonic degree `n`.
        degree: u16,
        /// Spherical-harmonic order `m`.
        order: u16,
    },

    /// The requested spherical-harmonic degree/order exceeds what the
    /// gravity-field provider supports.
    GeopotentialDegreeOutOfRange {
        /// Degree requested by the caller.
        requested: usize,
        /// Maximum degree the provider supports.
        max: usize,
    },

    /// The acceleration model does not implement analytic partials.
    PartialsUnavailable {
        /// Short model identifier for diagnostic context.
        model: &'static str,
    },

    /// Propagation failed (e.g. NaN/Inf produced, step rejection cascade).
    PropagationFailed {
        /// Short human-readable explanation of what happened.
        reason: &'static str,
    },

    /// A caller-owned context did not supply data required by the model
    /// at evaluation time.
    ContextDataUnavailable {
        /// Short human-readable identifier of the missing context member.
        what: &'static str,
    },
}

impl fmt::Display for PrincipiaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateGeometry { reason } => {
                write!(f, "degenerate geometry: {reason}")
            }
            Self::InvalidStepRequest { reason } => {
                write!(f, "invalid step request: {reason}")
            }
            Self::StepControlFailed { reason } => {
                write!(f, "step controller failed: {reason}")
            }
            Self::StepBelowMinimum { reason } => {
                write!(f, "step below minimum: {reason}")
            }
            Self::GravityCoefficientUnavailable { degree, order } => {
                write!(
                    f,
                    "gravity coefficient C_{degree},{order} not available in current model"
                )
            }
            Self::GeopotentialDegreeOutOfRange { requested, max } => {
                write!(
                    f,
                    "requested geopotential degree {requested} exceeds provider maximum {max}"
                )
            }
            Self::PartialsUnavailable { model } => {
                write!(f, "analytic partials not available for model '{model}'")
            }
            Self::PropagationFailed { reason } => {
                write!(f, "propagation failed: {reason}")
            }
            Self::ContextDataUnavailable { what } => {
                write!(f, "context data unavailable: {what}")
            }
        }
    }
}

impl core::error::Error for PrincipiaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_examples() {
        assert!(PrincipiaError::DegenerateGeometry { reason: "r=0" }
            .to_string()
            .contains("r=0"));
        assert!(PrincipiaError::InvalidStepRequest { reason: "h<=0" }
            .to_string()
            .contains("h<=0"));
        assert!(PrincipiaError::GravityCoefficientUnavailable {
            degree: 8,
            order: 3
        }
        .to_string()
        .contains("C_8,3"));
        assert!(PrincipiaError::GeopotentialDegreeOutOfRange {
            requested: 70,
            max: 21
        }
        .to_string()
        .contains("70"));
        assert!(PrincipiaError::PartialsUnavailable { model: "drag" }
            .to_string()
            .contains("drag"));
        assert!(PrincipiaError::PropagationFailed { reason: "NaN" }
            .to_string()
            .contains("NaN"));
        assert!(PrincipiaError::ContextDataUnavailable { what: "ephemeris" }
            .to_string()
            .contains("ephemeris"));
    }
}
