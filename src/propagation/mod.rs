// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Adaptive propagation driver, events, and results.
//!
//! ## Scientific scope
//!
//! Couples an adaptive Runge-Kutta integrator to a Cartesian acceleration
//! model while handling time bounds, output sampling, and event detection.
//!
//! ## Technical scope
//!
//! [`PropagationConfig`] carries the time window, step-size bounds,
//! tolerances, event detectors, and output schedule. [`propagate`] is generic
//! over the caller-owned context type, time scale, center, and frame.
//!
//! ## References
//!
//! * Hairer, Nørsett, Wanner, *Solving Ordinary Differential Equations I*, §II.5.

use core::fmt;

#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::boxed::Box;
#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::vec::Vec;

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use qtty::length::Kilometers;
use tempoch::ContinuousScale;

#[cfg(any(feature = "alloc", feature = "std"))]
use qtty::Second;
#[cfg(any(feature = "alloc", feature = "std"))]
use qtty::IntegratorTolerances;
#[cfg(any(feature = "alloc", feature = "std"))]
use tempoch::Time;

use crate::error::PrincipiaError;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::integrators::AdaptiveStepper;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::models::AccelerationModel;
use crate::state::DynamicsState;

/// Error family produced by the propagation driver.
#[derive(Debug)]
pub enum PropagationError {
    /// Wrap an underlying stepper failure.
    StepControl(PrincipiaError),
    /// The step controller requested a step smaller than the configured minimum.
    StepBelowMinimum {
        /// Requested step magnitude in seconds.
        h_requested: f64,
        /// Configured minimum allowed step magnitude in seconds.
        h_min: f64,
    },
    /// The driver exceeded the configured accepted-step budget.
    MaxStepsExceeded {
        /// Configured maximum number of accepted steps.
        max_steps: usize,
    },
    /// An event detector failed while evaluating its switching function.
    EventEvaluation {
        /// Detector name.
        name: &'static str,
        /// Underlying detector error.
        source: PrincipiaError,
    },
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StepControl(source) => write!(f, "integrator step control error: {source}"),
            Self::StepBelowMinimum { h_requested, h_min } => write!(
                f,
                "requested step {h_requested:e} s falls below configured minimum {h_min:e} s"
            ),
            Self::MaxStepsExceeded { max_steps } => {
                write!(f, "propagation exceeded max_steps={max_steps}")
            }
            Self::EventEvaluation { name, source } => {
                write!(f, "event '{name}' evaluation failed: {source}")
            }
        }
    }
}

impl core::error::Error for PropagationError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::StepControl(source) => Some(source),
            Self::EventEvaluation { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Propagation configuration for the low-level driver.
#[cfg(any(feature = "alloc", feature = "std"))]
pub struct PropagationConfig<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Start epoch of the propagation window.
    pub t_start: Time<S>,
    /// End epoch of the propagation window.
    pub t_end: Time<S>,
    /// Initial step-size guess.
    pub h0: Second,
    /// Maximum allowed step magnitude.
    pub h_max: Second,
    /// Minimum allowed step magnitude.
    pub h_min: Second,
    /// Adaptive-step tolerances.
    pub tolerances: IntegratorTolerances,
    /// Optional regular output cadence.
    pub output_every: Option<Second>,
    /// Additional explicit output epochs.
    pub output_at: Vec<Time<S>>,
    /// Registered event detectors.
    pub events: Vec<Box<dyn EventDetector<Ctx, S, C, F>>>,
    /// Maximum number of accepted steps.
    pub max_steps: usize,
}

#[cfg(any(feature = "alloc", feature = "std"))]
impl<Ctx, S, C, F> PropagationConfig<Ctx, S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Construct a propagation configuration with conservative defaults.
    pub fn new(t_start: Time<S>, t_end: Time<S>) -> Self {
        Self {
            t_start,
            t_end,
            h0: Second::new(30.0),
            h_max: Second::new(86_400.0),
            h_min: Second::new(1.0e-6),
            tolerances: IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9),
            output_every: None,
            output_at: Vec::new(),
            events: Vec::new(),
            max_steps: 1_000_000,
        }
    }

    /// Override the initial step guess.
    pub fn with_initial_step(mut self, h0: Second) -> Self {
        self.h0 = h0;
        self
    }

    /// Override the maximum step magnitude.
    pub fn with_max_step(mut self, h_max: Second) -> Self {
        self.h_max = h_max;
        self
    }

    /// Override the minimum step magnitude.
    pub fn with_min_step(mut self, h_min: Second) -> Self {
        self.h_min = h_min;
        self
    }

    /// Override the adaptive-step tolerances.
    pub fn with_tolerances(mut self, tolerances: IntegratorTolerances) -> Self {
        self.tolerances = tolerances;
        self
    }

    /// Request outputs at a regular cadence.
    pub fn with_output_every(mut self, dt: Second) -> Self {
        self.output_every = Some(dt);
        self
    }

    /// Request outputs at explicit epochs.
    pub fn with_output_at(mut self, times: Vec<Time<S>>) -> Self {
        self.output_at = times;
        self
    }

    /// Append one event detector.
    pub fn with_event(mut self, ev: Box<dyn EventDetector<Ctx, S, C, F>>) -> Self {
        self.events.push(ev);
        self
    }

    /// Override the accepted-step budget.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Return the signed propagation duration in seconds.
    pub fn total_duration_s(&self) -> f64 {
        (self.t_end - self.t_start).value()
    }
}

/// Zero-crossing event detector evaluated on propagated states.
pub trait EventDetector<Ctx, S, C, F>: Send + Sync
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Short event identifier.
    fn name(&self) -> &'static str;

    /// Evaluate the detector switching function `g(t, x)`.
    fn evaluate(&self, state: &DynamicsState<S, C, F>, ctx: &Ctx) -> Result<f64, PrincipiaError>;

    /// Return `true` if the event terminates propagation.
    fn terminal(&self) -> bool {
        false
    }

    /// Return `true` if a sign change from `g_before` to `g_after` should be
    /// treated as an event occurrence.
    fn accepts_crossing(&self, g_before: f64, g_after: f64) -> bool {
        g_before == 0.0 || g_after == 0.0 || (g_before * g_after) < 0.0
    }
}

/// Detector for a radial threshold crossing `|r| = threshold`.
#[derive(Debug, Clone, Copy)]
pub struct RadialThresholdEvent {
    /// Trigger radius.
    pub threshold: Kilometers,
    /// If `true`, only detect descending crossings.
    pub falling: bool,
    /// If `true`, stop propagation when the event fires.
    pub terminal: bool,
}

impl RadialThresholdEvent {
    /// Construct a non-terminal radial-threshold detector.
    pub fn new(threshold: Kilometers) -> Self {
        Self {
            threshold,
            falling: false,
            terminal: false,
        }
    }

    /// Configure the detector as terminal or non-terminal.
    pub fn terminal(mut self, terminal: bool) -> Self {
        self.terminal = terminal;
        self
    }

    /// Configure the accepted crossing direction.
    pub fn falling(mut self, falling: bool) -> Self {
        self.falling = falling;
        self
    }
}

impl<Ctx, S, C, F> EventDetector<Ctx, S, C, F> for RadialThresholdEvent
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    fn name(&self) -> &'static str {
        "radial_threshold"
    }

    fn evaluate(&self, state: &DynamicsState<S, C, F>, _ctx: &Ctx) -> Result<f64, PrincipiaError> {
        let x = state.position.x().value();
        let y = state.position.y().value();
        let z = state.position.z().value();
        Ok((x * x + y * y + z * z).sqrt() - self.threshold.value())
    }

    fn terminal(&self) -> bool {
        self.terminal
    }

    fn accepts_crossing(&self, g_before: f64, g_after: f64) -> bool {
        if self.falling {
            g_before > 0.0 && g_after <= 0.0
        } else {
            g_before < 0.0 && g_after >= 0.0
        }
    }
}

/// Recorded event occurrence.
#[derive(Debug, Clone)]
pub struct EventOccurrence<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Event identifier.
    pub event_name: &'static str,
    /// Interpolated state at the event epoch.
    pub state: DynamicsState<S, C, F>,
}

/// Result returned by the propagation driver.
#[cfg(any(feature = "alloc", feature = "std"))]
#[derive(Debug, Clone)]
pub struct PropagationResult<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
{
    /// Recorded samples, including the initial state and final state.
    pub samples: Vec<DynamicsState<S, C, F>>,
    /// Recorded event occurrences.
    pub events: Vec<EventOccurrence<S, C, F>>,
    /// Number of accepted steps.
    pub steps_taken: usize,
    /// Number of internally rejected trial steps reported by the adaptive integrator.
    pub steps_rejected: u32,
}

#[cfg(any(feature = "alloc", feature = "std"))]
fn lerp_state<S, C, F>(
    a: &DynamicsState<S, C, F>,
    b: &DynamicsState<S, C, F>,
    theta: f64,
) -> DynamicsState<S, C, F>
where
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    let px = a.position.x().value() + theta * (b.position.x().value() - a.position.x().value());
    let py = a.position.y().value() + theta * (b.position.y().value() - a.position.y().value());
    let pz = a.position.z().value() + theta * (b.position.z().value() - a.position.z().value());
    let vx = a.velocity.x().value() + theta * (b.velocity.x().value() - a.velocity.x().value());
    let vy = a.velocity.y().value() + theta * (b.velocity.y().value() - a.velocity.y().value());
    let vz = a.velocity.z().value() + theta * (b.velocity.z().value() - a.velocity.z().value());
    DynamicsState::new(
        a.epoch + Second::new(theta * (b.epoch - a.epoch).value()),
        affn::cartesian::Position::<C, F, qtty::unit::Kilometer>::new_with_params(
            a.position.center_params().clone(),
            px,
            py,
            pz,
        ),
        affn::cartesian::Velocity::<F, qtty::KmPerSecond>::new(vx, vy, vz),
    )
}

/// Propagate a state with an adaptive stepper over the interval configured in `cfg`.
#[cfg(any(feature = "alloc", feature = "std"))]
#[allow(clippy::too_many_lines)]
pub fn propagate<I, M, Ctx, S, C, F>(
    integrator: &I,
    model: &M,
    initial: DynamicsState<S, C, F>,
    cfg: &PropagationConfig<Ctx, S, C, F>,
    ctx: &Ctx,
) -> Result<PropagationResult<S, C, F>, PropagationError>
where
    I: AdaptiveStepper<Ctx, S, C, F>,
    M: AccelerationModel<Ctx, S, C, F>,
    S: ContinuousScale,
    C: ReferenceCenter,
    F: ReferenceFrame,
    C::Params: Clone,
{
    let total = cfg.total_duration_s();
    let direction = if total >= 0.0 { 1.0 } else { -1.0 };
    let mut samples = Vec::new();
    let mut events = Vec::new();
    let mut steps_taken = 0usize;
    let mut steps_rejected = 0u32;
    samples.push(initial.clone());
    if total == 0.0 {
        return Ok(PropagationResult {
            samples,
            events,
            steps_taken,
            steps_rejected,
        });
    }

    let mut g_prev = Vec::with_capacity(cfg.events.len());
    for ev in &cfg.events {
        g_prev.push(ev.evaluate(&initial, ctx).map_err(|source| {
            PropagationError::EventEvaluation {
                name: ev.name(),
                source,
            }
        })?);
    }

    let mut state = initial;
    let mut h = direction * cfg.h0.value().abs().min(cfg.h_max.value().abs());
    let mut next_output_t = cfg
        .output_every
        .map(|dt| cfg.t_start + Second::new(direction * dt.value().abs()));
    let mut output_at_iter = cfg.output_at.iter().peekable();
    let mut terminated = false;

    while !terminated {
        let remaining = (cfg.t_end - state.epoch).value();
        if remaining * direction <= 1e-12 {
            break;
        }
        if h.abs() > remaining.abs() {
            h = remaining;
        }
        if h.abs() > cfg.h_max.value().abs() {
            h = direction * cfg.h_max.value().abs();
        }
        if h.abs() < cfg.h_min.value().abs() && h.abs() < remaining.abs() {
            return Err(PropagationError::StepBelowMinimum {
                h_requested: h.abs(),
                h_min: cfg.h_min.value().abs(),
            });
        }

        if let Some(t_target) = next_output_t {
            let to_target = (t_target - state.epoch).value();
            if to_target.abs() > 1e-12 && to_target * direction >= 0.0 && to_target.abs() < h.abs()
            {
                h = to_target;
            }
        }
        if let Some(&t_target) = output_at_iter.peek() {
            let to_target = (*t_target - state.epoch).value();
            if to_target.abs() > 1e-12 && to_target * direction >= 0.0 && to_target.abs() < h.abs()
            {
                h = to_target;
            }
        }

        let (new_state, _h_used, h_next, rejected) = integrator
            .step(model, &state, Second::new(h), ctx)
            .map_err(PropagationError::StepControl)?;
        steps_taken += 1;
        steps_rejected += rejected;
        if steps_taken > cfg.max_steps {
            return Err(PropagationError::MaxStepsExceeded {
                max_steps: cfg.max_steps,
            });
        }

        for (i, ev) in cfg.events.iter().enumerate() {
            let g_new = ev.evaluate(&new_state, ctx).map_err(|source| {
                PropagationError::EventEvaluation {
                    name: ev.name(),
                    source,
                }
            })?;
            let sign_changed = g_prev[i] == 0.0 || g_new == 0.0 || (g_prev[i] * g_new) < 0.0;
            if sign_changed && ev.accepts_crossing(g_prev[i], g_new) {
                let theta = if (g_new - g_prev[i]).abs() > 1e-300 {
                    g_prev[i].abs() / (g_prev[i].abs() + g_new.abs())
                } else {
                    0.5
                };
                events.push(EventOccurrence {
                    event_name: ev.name(),
                    state: lerp_state(&state, &new_state, theta),
                });
                if ev.terminal() {
                    terminated = true;
                }
            }
            g_prev[i] = g_new;
        }

        if let Some(t_target) = next_output_t {
            let crossed = (t_target - state.epoch).value() * direction >= 0.0
                && (new_state.epoch - t_target).value() * direction >= 0.0;
            if crossed {
                samples.push(new_state.clone());
                if let Some(dt) = cfg.output_every {
                    next_output_t = Some(t_target + Second::new(direction * dt.value().abs()));
                }
            }
        }
        while let Some(&t_target) = output_at_iter.peek() {
            let crossed = (*t_target - state.epoch).value() * direction >= 0.0
                && (new_state.epoch - *t_target).value() * direction >= 0.0;
            if crossed {
                samples.push(new_state.clone());
                output_at_iter.next();
            } else {
                break;
            }
        }

        state = new_state;
        h = h_next.value();
    }

    if samples
        .last()
        .map(|sample| (sample.epoch - state.epoch).value().abs() > 0.0)
        .unwrap_or(true)
    {
        samples.push(state);
    }

    Ok(PropagationResult {
        samples,
        events,
        steps_taken,
        steps_rejected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrators::Dopri5;
    use crate::models::TwoBody;
    use affn::centers::ReferenceCenter;
    use affn::frames::ReferenceFrame;
    use qtty::{GravitationalParameter, Second};
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
        let mu = 398_600.441_8_f64;
        let r = 7000.0_f64;
        let v = (mu / r).sqrt();
        DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, qtty::unit::Kilometer>::new(r, 0.0, 0.0),
            affn::cartesian::Velocity::<Inertial, qtty::KmPerSecond>::new(0.0, v, 0.0),
        )
    }

    #[test]
    fn zero_duration_returns_initial_state() {
        let s0 = circular_state();
        let cfg = PropagationConfig::<(), TT, Center, Inertial>::new(s0.epoch, s0.epoch);
        let result = propagate(
            &Dopri5::new(IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9)),
            &TwoBody::new(GravitationalParameter::new(398_600.441_8)),
            s0,
            &cfg,
            &(),
        )
        .unwrap();
        assert_eq!(result.samples.len(), 1);
        assert_eq!(result.steps_taken, 0);
    }

    #[test]
    fn radial_threshold_event_fires() {
        let s0 = circular_state();
        let period = 2.0 * core::f64::consts::PI * (7000.0_f64.powi(3) / 398_600.441_8).sqrt();
        let ecc_state = DynamicsState::new(
            s0.epoch,
            s0.position,
            affn::cartesian::Velocity::<Inertial, qtty::KmPerSecond>::new(
                0.0,
                s0.velocity.y().value() * 1.01,
                0.0,
            ),
        );
        let cfg = PropagationConfig::<(), TT, Center, Inertial>::new(
            ecc_state.epoch,
            ecc_state.epoch + Second::new(period),
        )
        .with_event(Box::new(RadialThresholdEvent::new(Kilometers::new(7121.0))));
        let result = propagate(
            &Dopri5::new(IntegratorTolerances::uniform(1e-9, 1e-6, 1e-9)),
            &TwoBody::new(GravitationalParameter::new(398_600.441_8)),
            ecc_state,
            &cfg,
            &(),
        )
        .unwrap();
        assert!(!result.events.is_empty());
    }
}
