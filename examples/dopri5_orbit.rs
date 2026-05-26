// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

#![allow(clippy::print_stdout)]

//! Adaptive two-body orbit propagation with DOPRI5 tolerances.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use principia::{dopri5_propagate, DynamicsState, IntegratorTolerances, TwoBody};
use qtty::tolerances::{AbsoluteTolerancePosition, AbsoluteToleranceVelocity, RelativeTolerance};
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

fn tolerances() -> IntegratorTolerances {
    IntegratorTolerances {
        rel: RelativeTolerance::new(1.0e-12),
        abs_pos: [AbsoluteTolerancePosition::new_km(1.0e-9); 3],
        abs_vel: [AbsoluteToleranceVelocity::new_km_s(1.0e-12); 3],
    }
}

fn main() -> Result<(), principia::PrincipiaError> {
    let mu = GravitationalParameter::new(398_600.441_8);
    let radius_km = 7_200.0;
    let circular_speed = (mu.value() / radius_km).sqrt();
    let initial = DynamicsState::try_new(
        Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
        affn::cartesian::Position::<Center, Inertial, Kilometer>::new(radius_km, 0.0, 0.0),
        affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, circular_speed, 0.0),
    )?;
    let model = TwoBody::try_new(mu)?;
    let period = Second::new(TAU * (radius_km.powi(3) / mu.value()).sqrt());
    let final_state = dopri5_propagate(&model, initial, period, tolerances(), &())?;

    println!("Adaptive DOPRI5 propagation over one orbital period.");
    println!(
        "Final epoch offset: {:.3} s",
        (final_state.epoch - initial.epoch).value()
    );
    println!("Final position: {:?}", final_state.position.as_array());
    println!("Final velocity: {:?}", final_state.velocity.as_array());
    Ok(())
}
