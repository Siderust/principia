// SPDX-License-Identifier: AGPL-3.0-only
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
//! belong downstream in `siderust::astro::dynamics`.
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// An integrator tolerance value is invalid (e.g., non-finite or ≤ 0).
    InvalidTolerance {
        /// Short human-readable explanation of what constraint was violated.
        context: &'static str,
    },

    /// An integrator parameter (e.g., `h_min`, `h_max`) is invalid.
    InvalidParameter {
        /// Short human-readable explanation of what constraint was violated.
        reason: &'static str,
    },

    /// A physical quantity has a non-positive value where a positive value
    /// is required (e.g., gravitational parameter, equatorial radius).
    NonPositiveValue {
        /// Short human-readable context identifying the invalid quantity.
        context: &'static str,
    },

    /// A state vector or computed quantity contains a non-finite value
    /// (NaN or infinity).
    NonFiniteValue {
        /// Short human-readable context identifying the problematic component.
        context: &'static str,
    },

    /// A gravity-model evaluation request is invalid (e.g., degree/order
    /// out of bounds, or inconsistent degree/order pair).
    InvalidGravityRequest {
        /// Short human-readable explanation.
        reason: &'static str,
    },

    /// A propagation configuration parameter is invalid (e.g., step-size
    /// bounds, tolerances, or span duration are inconsistent or non-finite).
    InvalidPropagationConfig {
        /// Short human-readable explanation.
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
            Self::InvalidTolerance { context } => {
                write!(f, "invalid tolerance: {context}")
            }
            Self::InvalidParameter { reason } => {
                write!(f, "invalid parameter: {reason}")
            }
            Self::NonPositiveValue { context } => {
                write!(f, "non-positive value: {context}")
            }
            Self::NonFiniteValue { context } => {
                write!(f, "non-finite value: {context}")
            }
            Self::InvalidGravityRequest { reason } => {
                write!(f, "invalid gravity request: {reason}")
            }
            Self::InvalidPropagationConfig { reason } => {
                write!(f, "invalid propagation config: {reason}")
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
    use alloc::string::ToString;

    #[cfg(any(feature = "alloc", feature = "std"))]
    #[test]
    fn display_examples() {
        use super::PrincipiaError;
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

    #[test]
    fn display_all_new_variants() {
        use super::PrincipiaError;

        let s = PrincipiaError::InvalidTolerance { context: "rtol" }.to_string();
        assert!(s.contains("rtol"), "got: {s}");

        let s = PrincipiaError::InvalidParameter {
            reason: "h_min>h_max",
        }
        .to_string();
        assert!(s.contains("h_min>h_max"), "got: {s}");

        let s = PrincipiaError::NonPositiveValue { context: "mu" }.to_string();
        assert!(s.contains("mu"), "got: {s}");

        let s = PrincipiaError::NonFiniteValue {
            context: "position",
        }
        .to_string();
        assert!(s.contains("position"), "got: {s}");

        let s = PrincipiaError::InvalidGravityRequest {
            reason: "degree too high",
        }
        .to_string();
        assert!(s.contains("degree"), "got: {s}");

        let s = PrincipiaError::InvalidPropagationConfig {
            reason: "h0 is zero",
        }
        .to_string();
        assert!(s.contains("h0"), "got: {s}");

        let s = PrincipiaError::StepControlFailed { reason: "diverged" }.to_string();
        assert!(s.contains("diverged"), "got: {s}");

        let s = PrincipiaError::StepBelowMinimum {
            reason: "too small",
        }
        .to_string();
        assert!(s.contains("too small"), "got: {s}");
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::PrincipiaError>();
    }
}
