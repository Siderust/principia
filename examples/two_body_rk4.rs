// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

#![allow(clippy::print_stdout)]

//! Fixed-step two-body orbit propagation with RK4.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use principia::{DynamicsState, Rk4, Stepper, TwoBody};
use qtty::unit::Kilometer;
use qtty::{GravitationalParameter, KmPerSecond, Second};
use std::f64::consts::TAU;
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

fn main() -> Result<(), principia::PrincipiaError> {
    let mu = GravitationalParameter::new(398_600.441_8);
    let radius_km = 7_000.0;
    let circular_speed = (mu.value() / radius_km).sqrt();
    let initial = DynamicsState::try_new(
        Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
        affn::cartesian::Position::<Center, Inertial, Kilometer>::new(radius_km, 0.0, 0.0),
        affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, circular_speed, 0.0),
    )?;

    let model = TwoBody::try_new(mu)?;
    let rk4 = Rk4;
    let step = Second::new(10.0);
    let period = TAU * (radius_km.powi(3) / mu.value()).sqrt();
    let steps = (period / step.value()).round() as usize;

    let mut state = initial;
    for _ in 0..steps {
        state = rk4.step(&model, &state, step, &())?;
    }

    let dx = state.position.x().value() - initial.position.x().value();
    let dy = state.position.y().value() - initial.position.y().value();
    let dz = state.position.z().value() - initial.position.z().value();
    let closure_error_km = (dx * dx + dy * dy + dz * dz).sqrt();

    println!(
        "Propagated {:.1} s with fixed-step RK4.",
        steps as f64 * step.value()
    );
    println!("Final position: {:?}", state.position.as_array());
    println!("Closure error after ~1 orbit: {:.6} km", closure_error_km);
    Ok(())
}
