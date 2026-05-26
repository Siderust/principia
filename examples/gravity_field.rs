// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

#![allow(clippy::print_stdout)]

//! Evaluate typed spherical-harmonic gravity acceleration from a custom provider.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use principia::{
    spherical_harmonic_acceleration, GravityConstants, GravityFieldProvider, PrincipiaError,
};
use qtty::dynamics::GravitationalParameter;
use qtty::length::Kilometers;
use qtty::unit::Kilometer;

#[derive(Debug, Clone, Copy)]
struct BodyFixed;
impl ReferenceFrame for BodyFixed {
    fn frame_name() -> &'static str {
        "BodyFixed"
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

struct DemoGravityField;
impl GravityFieldProvider for DemoGravityField {
    fn mu(&self) -> GravitationalParameter {
        GravitationalParameter::new(398_600.441_8)
    }

    fn reference_radius(&self) -> Kilometers {
        Kilometers::new(6_378.137)
    }

    fn max_degree(&self) -> usize {
        2
    }

    fn max_order(&self) -> usize {
        0
    }

    fn c_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
        if n > self.max_degree() || m > self.max_order() || m > n {
            return Err(PrincipiaError::GravityCoefficientUnavailable {
                degree: n as u16,
                order: m as u16,
            });
        }
        Ok(match (n, m) {
            (0, 0) => 1.0,
            (2, 0) => -4.841_653_717_36e-4,
            _ => 0.0,
        })
    }

    fn s_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
        if n > self.max_degree() || m > self.max_order() || m > n {
            return Err(PrincipiaError::GravityCoefficientUnavailable {
                degree: n as u16,
                order: m as u16,
            });
        }
        Ok(0.0)
    }
}

fn main() -> Result<(), PrincipiaError> {
    let provider = DemoGravityField;
    let constants = GravityConstants::try_new(provider.mu(), provider.reference_radius(), 2)?;
    let position =
        affn::cartesian::Position::<Center, BodyFixed, Kilometer>::new(7_000.0, 0.0, 0.0);

    let central = spherical_harmonic_acceleration(
        &position,
        0,
        0,
        &constants,
        &provider,
        Kilometers::new(100.0),
    )?;
    let with_j2 = spherical_harmonic_acceleration(
        &position,
        2,
        0,
        &constants,
        &provider,
        Kilometers::new(100.0),
    )?;

    println!("Degree-0 acceleration: {:?}", central.as_array());
    println!("Degree-2 acceleration: {:?}", with_j2.as_array());
    Ok(())
}
