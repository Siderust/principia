// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Vallés Puig, Ramon

//! Gravity-field provider trait and spherical-harmonic acceleration kernel.
//!
//! ## Scientific scope
//!
//! Evaluates the Cartesian acceleration corresponding to a fully normalized
//! spherical-harmonic gravity field.
//!
//! ## Technical scope
//!
//! Downstream crates implement [`GravityFieldProvider`] to supply normalized
//! Stokes coefficients. [`spherical_harmonic_acceleration`] is the typed
//! public entry point; the recursive Legendre/Cunningham kernel remains
//! private to this module.
//!
//! ## References
//!
//! * Montenbruck & Gill, *Satellite Orbits*, §3.2.
//! * Vallado, *Fundamentals of Astrodynamics and Applications*, §8.6.

use alloc::vec;

use qtty::length::Kilometers;
use qtty::GravitationalParameter;

use crate::error::PrincipiaError;

const SQRT3: f64 = 1.732_050_808_568_877;

/// Gravity-field constants packed into one helper value.
#[derive(Debug, Clone, Copy)]
pub struct GravityConstants {
    /// Standard gravitational parameter `μ = G·M`, km³/s².
    pub mu: GravitationalParameter,
    /// Reference radius used by the field expansion, km.
    pub radius: Kilometers,
    /// Maximum supported degree.
    pub max_degree: u16,
}

/// Provider of fully normalized spherical-harmonic coefficients.
pub trait GravityFieldProvider {
    /// Return the central-body gravitational parameter.
    fn mu(&self) -> GravitationalParameter;

    /// Return the reference radius used by the field model.
    fn reference_radius(&self) -> Kilometers;

    /// Return the maximum supported degree.
    fn max_degree(&self) -> usize;

    /// Return the maximum supported order.
    fn max_order(&self) -> usize {
        self.max_degree()
    }

    /// Return the normalized cosine coefficient `C̄ₙₘ`.
    fn c_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError>;

    /// Return the normalized sine coefficient `S̄ₙₘ`.
    fn s_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError>;

    /// Legacy convenience pack of common constants.
    fn constants(&self) -> GravityConstants {
        GravityConstants {
            mu: self.mu(),
            radius: self.reference_radius(),
            max_degree: self.max_degree() as u16,
        }
    }

    /// Compatibility alias for normalized cosine coefficients.
    fn c_nm(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
        self.c_normalized(n, m)
    }

    /// Compatibility alias for normalized sine coefficients.
    fn s_nm(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
        self.s_normalized(n, m)
    }
}

/// Evaluate the body-fixed spherical-harmonic acceleration.
pub fn spherical_harmonic_acceleration<P: GravityFieldProvider + ?Sized>(
    provider: &P,
    body_fixed_pos_km: [f64; 3],
    degree: usize,
    order: usize,
) -> Result<[f64; 3], PrincipiaError> {
    if degree > provider.max_degree() {
        return Err(PrincipiaError::GeopotentialDegreeOutOfRange {
            requested: degree,
            max: provider.max_degree(),
        });
    }
    if order > degree {
        return Err(PrincipiaError::InvalidStepRequest {
            reason: "spherical-harmonic order must satisfy m <= n",
        });
    }
    let max_m = order.min(provider.max_order()).min(degree);
    let mu = provider.mu().value();
    let re = provider.reference_radius().value();
    let [x, y, z] = body_fixed_pos_km;
    let r = (x * x + y * y + z * z).sqrt();
    if r < 100.0 {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "radial magnitude below spherical-harmonic degeneracy threshold",
        });
    }
    compute_inner(mu, re, degree, max_m, provider, body_fixed_pos_km)
}

fn compute_inner<P: GravityFieldProvider + ?Sized>(
    mu: f64,
    re: f64,
    max_n: usize,
    max_m: usize,
    provider: &P,
    pos: [f64; 3],
) -> Result<[f64; 3], PrincipiaError> {
    let (x, y, z) = (pos[0], pos[1], pos[2]);
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    let rxy2 = x * x + y * y;
    let rxy = rxy2.sqrt();
    let sinphi = z / r;
    let cosphi = rxy / r;
    let (coslam, sinlam) = if rxy2 > 0.0 {
        (x / rxy, y / rxy)
    } else {
        (1.0, 0.0)
    };

    let plen = max_n + 2;
    let mut p = vec![0.0_f64; plen * plen];
    macro_rules! pnm {
        ($n:expr, $m:expr) => {
            p[($n) * plen + ($m)]
        };
    }

    pnm!(0, 0) = 1.0;
    if max_n >= 1 {
        pnm!(1, 0) = SQRT3 * sinphi;
        pnm!(1, 1) = SQRT3 * cosphi;
    }
    for n in 2..=max_n {
        let nf = n as f64;
        pnm!(n, n) = ((2.0 * nf + 1.0) / (2.0 * nf)).sqrt() * cosphi * pnm!(n - 1, n - 1);
        pnm!(n, n - 1) = (2.0 * nf + 1.0).sqrt() * sinphi * pnm!(n - 1, n - 1);
        for m in 0..=(n - 2) {
            let mf = m as f64;
            let a = ((4.0 * nf * nf - 1.0) / (nf * nf - mf * mf)).sqrt();
            let b = ((2.0 * nf + 1.0) * (nf - mf - 1.0) * (nf + mf - 1.0)
                / ((2.0 * nf - 3.0) * (nf - mf) * (nf + mf)))
                .sqrt();
            pnm!(n, m) = a * sinphi * pnm!(n - 1, m) - b * pnm!(n - 2, m);
        }
    }

    let mut cosml = vec![0.0_f64; max_m + 1];
    let mut sinml = vec![0.0_f64; max_m + 1];
    cosml[0] = 1.0;
    if max_m >= 1 {
        cosml[1] = coslam;
        sinml[1] = sinlam;
    }
    for m in 2..=max_m {
        cosml[m] = 2.0 * coslam * cosml[m - 1] - cosml[m - 2];
        sinml[m] = 2.0 * coslam * sinml[m - 1] - sinml[m - 2];
    }

    let mut du_dr = 0.0_f64;
    let mut big_a = 0.0_f64;
    let mut du_dlam = 0.0_f64;

    for n in 0..=max_n {
        let nf = n as f64;
        let rn_factor = mu / r2 * (re / r).powi(n as i32);
        for m in 0..=max_m.min(n) {
            let mf = m as f64;
            let cnm = provider.c_normalized(n, m)?;
            let snm = provider.s_normalized(n, m)?;
            let cm = cosml[m];
            let sm = sinml[m];
            let pnm = pnm!(n, m);
            let f_nm = cnm * cm + snm * sm;
            du_dr -= rn_factor * (nf + 1.0) * pnm * f_nm;
            let cdp = if n == m {
                -nf * sinphi * pnm
            } else {
                let alpha = ((nf * nf - mf * mf) * (2.0 * nf + 1.0) / (2.0 * nf - 1.0)).sqrt();
                -nf * sinphi * pnm + alpha * pnm!(n - 1, m)
            };
            big_a += rn_factor * f_nm * cdp;
            if m > 0 {
                let g_nm = mf * (snm * cm - cnm * sm);
                du_dlam += rn_factor * r * pnm * g_nm;
            }
        }
    }

    let az = (z / r) * du_dr + big_a;
    let (ax, ay) = if rxy2 > r2 * 1.0e-24 {
        (
            (x / r) * du_dr - (x * z / rxy2) * big_a - (y / rxy2) * du_dlam,
            (y / r) * du_dr - (y * z / rxy2) * big_a + (x / rxy2) * du_dlam,
        )
    } else {
        ((x / r) * du_dr, (y / r) * du_dr)
    };
    Ok([ax, ay, az])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TwoBodyOnly;
    impl GravityFieldProvider for TwoBodyOnly {
        fn mu(&self) -> GravitationalParameter {
            GravitationalParameter::new(398_600.441_8)
        }
        fn reference_radius(&self) -> Kilometers {
            Kilometers::new(6_378.137)
        }
        fn max_degree(&self) -> usize {
            0
        }
        fn c_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
            if n == 0 && m == 0 {
                Ok(1.0)
            } else {
                Ok(0.0)
            }
        }
        fn s_normalized(&self, _n: usize, _m: usize) -> Result<f64, PrincipiaError> {
            Ok(0.0)
        }
    }

    #[test]
    fn degree_zero_matches_two_body() {
        let acc = spherical_harmonic_acceleration(&TwoBodyOnly, [7000.0, 0.0, 0.0], 0, 0).unwrap();
        let expected = -398_600.441_8 / (7000.0 * 7000.0);
        assert!((acc[0] - expected).abs() < 1e-12);
        assert!(acc[1].abs() < 1e-30);
        assert!(acc[2].abs() < 1e-30);
    }
}
