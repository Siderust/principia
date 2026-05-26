// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

#![allow(clippy::print_stdout)]

//! Construct RTN, VNC, and LVLH frames from a dynamics state.

use affn::centers::ReferenceCenter;
use affn::frames::ReferenceFrame;
use principia::{DynamicsState, LocalTrajectoryFrame, LVLH, RTN, VNC};
use qtty::unit::Kilometer;
use qtty::{KmPerSecond, Second};
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
    let state = DynamicsState::try_new(
        Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
        affn::cartesian::Position::<Center, Inertial, Kilometer>::new(7_000.0, 0.0, 0.0),
        affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 1.0),
    )?;

    let rtn = LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&state)?;
    let vnc = LocalTrajectoryFrame::<Inertial, VNC>::try_from_state(&state)?;
    let lvlh = LocalTrajectoryFrame::<Inertial, LVLH>::try_from_state(&state)?;

    println!("RTN DCM: {:?}", rtn.dcm().as_array());
    println!("VNC DCM: {:?}", vnc.dcm().as_array());
    println!("LVLH DCM: {:?}", lvlh.dcm().as_array());
    Ok(())
}
