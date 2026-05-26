// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

//! Regression test: RK4 propagation of a circular two-body orbit.
//!
//! After one Keplerian period the state must return to its initial value
//! (within the truncation tolerance of RK4 at the chosen step size), and
//! the specific orbital energy must be conserved to high precision.

use affn::cartesian::Position;
use affn::cartesian::Velocity;
use qtty::dynamics::GravitationalParameter;
use qtty::length::Kilometers;
use qtty::{KmPerSeconds, Second};
use tempoch::{Time, TT};

use principia::integrators::{Rk4, Stepper};
use principia::models::TwoBody;
use principia::state::DynamicsState;

#[derive(Debug, Clone, Copy)]
struct TestFrame;

impl affn::frames::ReferenceFrame for TestFrame {
    fn frame_name() -> &'static str {
        "TestFrame"
    }
}

#[test]
fn rk4_two_body_circular_orbit_closes_after_one_period() {
    // Earth-like GM and a 7000 km circular orbit.
    let mu_value = 398_600.441_8_f64; // km^3/s^2
    let r = 7_000.0_f64; // km
    let v = (mu_value / r).sqrt(); // km/s
    let period = 2.0 * core::f64::consts::PI * (r.powi(3) / mu_value).sqrt();

    let mu = GravitationalParameter::new(mu_value);
    let model = TwoBody::new(mu);

    let epoch = Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).expect("epoch");
    let position: Position<(), TestFrame, _> = Position::new(
        Kilometers::new(r),
        Kilometers::new(0.0),
        Kilometers::new(0.0),
    );
    let velocity: Velocity<TestFrame, _> = Velocity::new(
        KmPerSeconds::new(0.0),
        KmPerSeconds::new(v),
        KmPerSeconds::new(0.0),
    );
    let mut state = DynamicsState::new(epoch, position, velocity);

    let initial_energy = 0.5 * v * v - mu_value / r;

    let dt_value = period / 6000.0;
    let dt = Second::new(dt_value);
    let n_steps: usize = 6000;
    let stepper = Rk4;

    for _ in 0..n_steps {
        state = stepper.step(&model, &state, dt, &()).expect("rk4 step");
    }

    let rx = state.position.x().value();
    let ry = state.position.y().value();
    let rz = state.position.z().value();
    let vx = state.velocity.x().value();
    let vy = state.velocity.y().value();
    let vz = state.velocity.z().value();

    let r_final = (rx * rx + ry * ry + rz * rz).sqrt();
    let v_final = (vx * vx + vy * vy + vz * vz).sqrt();
    let final_energy = 0.5 * v_final * v_final - mu_value / r_final;

    // After one full period, position should match the initial state to
    // within RK4 truncation tolerance.
    assert!(
        (rx - r).abs() < 1.0e-2,
        "x mismatch after one period: {} vs {}",
        rx,
        r
    );
    assert!(ry.abs() < 1.0e-2, "y drift after one period: {}", ry);
    assert!(rz.abs() < 1.0e-9, "out-of-plane drift: {}", rz);

    // Specific orbital energy conservation (closed-form invariant).
    let rel_energy_err = ((final_energy - initial_energy) / initial_energy).abs();
    assert!(
        rel_energy_err < 1.0e-9,
        "energy drift: rel err = {}",
        rel_energy_err
    );
}
