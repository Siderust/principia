// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

#![allow(clippy::print_stdout)]

//! Propagate a state-transition matrix alongside a two-body trajectory.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use principia::{propagate_stm_with, DynamicsState, TwoBody, VariationalConfig};
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
    let config = VariationalConfig::try_new(Second::new(30.0))?;
    let quarter_period = Second::new(0.25 * TAU * (radius_km.powi(3) / mu.value()).sqrt());
    let (final_state, phi) = propagate_stm_with(&model, initial, quarter_period, &(), &config)?;

    println!(
        "Final state after quarter orbit: {:?}",
        final_state.position.as_array()
    );
    println!("STM first row: {:?}", phi.as_array()[0]);
    Ok(())
}
