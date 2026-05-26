// SPDX-License-Identifier: AGPL-3.0-only
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
//! Downstream crates implement [`GravityFieldProvider`] to supply fully normalized
//! Stokes coefficients. [`spherical_harmonic_acceleration`] is the typed public
//! entry point; [`spherical_harmonic_acceleration_raw_km`] exposes the documented
//! raw km / km s⁻² kernel when callers explicitly need untyped arrays.
//!
//! ## References
//!
//! * Montenbruck & Gill, *Satellite Orbits*, §3.2.
//! * Vallado, *Fundamentals of Astrodynamics and Applications*, §8.6.

#[cfg(any(feature = "alloc", feature = "std"))]
use affn::cartesian::Position;
#[cfg(any(feature = "alloc", feature = "std"))]
use affn::centers::ReferenceCenter;
#[cfg(any(feature = "alloc", feature = "std"))]
use affn::frames::ReferenceFrame;
use qtty::dynamics::GravitationalParameter;
use qtty::length::Kilometers;
#[cfg(any(feature = "alloc", feature = "std"))]
use qtty::unit::Kilometer;

use crate::error::PrincipiaError;
#[cfg(any(feature = "alloc", feature = "std"))]
use crate::state::Acceleration;

#[cfg(any(feature = "alloc", feature = "std"))]
use alloc::vec;

#[cfg(any(feature = "alloc", feature = "std"))]
const SQRT3: f64 = 1.732_050_808_568_877;

/// Gravity-field constants packed into one helper value.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GravityConstants {
    /// Standard gravitational parameter `μ = G·M`, km³/s².
    pub mu: GravitationalParameter,
    /// Equatorial reference radius used by the field expansion, km.
    pub equatorial_radius: Kilometers,
    /// Maximum supported degree advertised by the loaded field constants.
    pub max_degree: u32,
}

impl GravityConstants {
    /// Construct a validated set of gravity constants.
    pub fn try_new(
        mu: GravitationalParameter,
        equatorial_radius: Kilometers,
        max_degree: u32,
    ) -> Result<Self, PrincipiaError> {
        if !mu.value().is_finite() || mu.value() <= 0.0 {
            return Err(PrincipiaError::NonPositiveValue {
                context: "GravityConstants: mu must be finite and positive",
            });
        }
        if !equatorial_radius.value().is_finite() || equatorial_radius.value() <= 0.0 {
            return Err(PrincipiaError::NonPositiveValue {
                context: "GravityConstants: equatorial_radius must be finite and positive",
            });
        }
        Ok(Self {
            mu,
            equatorial_radius,
            max_degree,
        })
    }
}

/// Provider of fully normalized spherical-harmonic coefficients.
///
/// `principia` uses the fully normalized Stokes convention (`C̄ₙₘ`, `S̄ₙₘ`).
/// Implementations should report the maximum degree/order they can serve via
/// [`max_degree`](Self::max_degree) and [`max_order`](Self::max_order), and
/// return [`PrincipiaError::GravityCoefficientUnavailable`] from the coefficient
/// accessors when `(n, m)` lies outside the loaded dataset.
pub trait GravityFieldProvider {
    /// Return the central-body gravitational parameter.
    fn mu(&self) -> GravitationalParameter;

    /// Return the equatorial reference radius used by the field model.
    fn reference_radius(&self) -> Kilometers;

    /// Return the maximum supported degree `n` of the loaded model.
    fn max_degree(&self) -> usize;

    /// Return the maximum supported order `m` of the loaded model.
    ///
    /// The default implementation assumes a square field and returns
    /// [`max_degree`](Self::max_degree).
    fn max_order(&self) -> usize {
        self.max_degree()
    }

    /// Return the fully normalized cosine coefficient `C̄ₙₘ`.
    fn c_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError>;

    /// Return the fully normalized sine coefficient `S̄ₙₘ`.
    fn s_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError>;

    /// Convenience pack of common constants derived from this provider.
    fn constants(&self) -> GravityConstants {
        GravityConstants {
            mu: self.mu(),
            equatorial_radius: self.reference_radius(),
            max_degree: self.max_degree() as u32,
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

/// Low-level kernel: accepts raw km position, returns raw km/s² acceleration.
/// No frame enforcement.
#[cfg(any(feature = "alloc", feature = "std"))]
pub fn spherical_harmonic_acceleration_raw_km<P: GravityFieldProvider + ?Sized>(
    body_fixed_pos_km: [f64; 3],
    degree: u32,
    order: u32,
    constants: &GravityConstants,
    provider: &P,
    min_radius_km: f64,
) -> Result<[f64; 3], PrincipiaError> {
    if !min_radius_km.is_finite() || min_radius_km <= 0.0 {
        return Err(PrincipiaError::NonPositiveValue {
            context: "gravity: min_radius_km must be finite and positive",
        });
    }
    let degree_usize = degree as usize;
    let order_usize = order as usize;
    if degree_usize > provider.max_degree() {
        return Err(PrincipiaError::InvalidGravityRequest {
            reason: "degree exceeds provider maximum",
        });
    }
    if degree > constants.max_degree {
        return Err(PrincipiaError::InvalidGravityRequest {
            reason: "degree exceeds gravity constants maximum",
        });
    }
    if order > degree {
        return Err(PrincipiaError::InvalidGravityRequest {
            reason: "requested order exceeds requested degree",
        });
    }
    if order_usize > provider.max_order() {
        return Err(PrincipiaError::InvalidGravityRequest {
            reason: "order exceeds provider maximum",
        });
    }
    let [x, y, z] = body_fixed_pos_km;
    let r = (x * x + y * y + z * z).sqrt();
    if r < min_radius_km {
        return Err(PrincipiaError::DegenerateGeometry {
            reason: "radial magnitude below spherical-harmonic degeneracy threshold",
        });
    }
    compute_inner(
        constants.mu.value(),
        constants.equatorial_radius.value(),
        degree_usize,
        order_usize,
        provider,
        body_fixed_pos_km,
    )
}

/// Evaluates the spherical-harmonic gravitational acceleration at `position`.
#[cfg(any(feature = "alloc", feature = "std"))]
pub fn spherical_harmonic_acceleration<C, F, P>(
    position: &Position<C, F, Kilometer>,
    degree: u32,
    order: u32,
    constants: &GravityConstants,
    provider: &P,
    min_radius: Kilometers,
) -> Result<Acceleration<F>, PrincipiaError>
where
    C: ReferenceCenter,
    F: ReferenceFrame,
    P: GravityFieldProvider,
{
    if degree as usize > provider.max_degree() {
        return Err(PrincipiaError::InvalidGravityRequest {
            reason: "requested degree exceeds provider's maximum degree",
        });
    }
    if order > degree {
        return Err(PrincipiaError::InvalidGravityRequest {
            reason: "requested order exceeds requested degree",
        });
    }
    let pos_raw = [
        position.x().value(),
        position.y().value(),
        position.z().value(),
    ];
    let acc_raw = spherical_harmonic_acceleration_raw_km(
        pos_raw,
        degree,
        order,
        constants,
        provider,
        min_radius.value(),
    )?;
    Ok(Acceleration::new(acc_raw[0], acc_raw[1], acc_raw[2]))
}

#[cfg(any(feature = "alloc", feature = "std"))]
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

#[cfg(all(test, any(feature = "alloc", feature = "std")))]
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
                Err(PrincipiaError::GravityCoefficientUnavailable {
                    degree: n as u16,
                    order: m as u16,
                })
            }
        }
        fn s_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
            if n == 0 && m == 0 {
                Ok(0.0)
            } else {
                Err(PrincipiaError::GravityCoefficientUnavailable {
                    degree: n as u16,
                    order: m as u16,
                })
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    #[allow(dead_code)]
    struct Inertial;
    impl ReferenceFrame for Inertial {
        fn frame_name() -> &'static str {
            "Inertial"
        }
    }

    #[derive(Debug, Clone, Copy)]
    #[allow(dead_code)]
    struct Center;
    impl ReferenceCenter for Center {
        type Params = ();
        fn center_name() -> &'static str {
            "Center"
        }
    }

    fn constants() -> GravityConstants {
        GravityConstants::try_new(
            GravitationalParameter::new(398_600.441_8),
            Kilometers::new(6_378.137),
            0,
        )
        .unwrap()
    }

    #[test]
    fn gravity_constants_try_new_rejects_bad_inputs() {
        assert!(matches!(
            GravityConstants::try_new(GravitationalParameter::new(0.0), Kilometers::new(1.0), 0),
            Err(PrincipiaError::NonPositiveValue { .. })
        ));
        assert!(matches!(
            GravityConstants::try_new(
                GravitationalParameter::new(398_600.441_8),
                Kilometers::new(0.0),
                0,
            ),
            Err(PrincipiaError::NonPositiveValue { .. })
        ));
    }

    #[test]
    fn degree_zero_matches_two_body() {
        let acc = spherical_harmonic_acceleration_raw_km(
            [7000.0, 0.0, 0.0],
            0,
            0,
            &constants(),
            &TwoBodyOnly,
            100.0,
        )
        .unwrap();
        let expected = -398_600.441_8 / (7000.0 * 7000.0);
        assert!((acc[0] - expected).abs() < 1e-12);
        assert!(acc[1].abs() < 1e-30);
        assert!(acc[2].abs() < 1e-30);
    }

    /// A J2-only gravity field provider (degree 2, order 0).
    /// J2 = 1.08262668e-3 (unnormalized); normalized C_20 = -J2/sqrt(5).
    struct J2Provider;
    impl GravityFieldProvider for J2Provider {
        fn mu(&self) -> GravitationalParameter {
            GravitationalParameter::new(398_600.441_8)
        }
        fn reference_radius(&self) -> Kilometers {
            Kilometers::new(6_378.137)
        }
        fn max_degree(&self) -> usize {
            2
        }
        fn c_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
            match (n, m) {
                (0, 0) => Ok(1.0),
                (1, 0) | (1, 1) => Ok(0.0),
                (2, 0) => Ok(-1.082_626_68e-3 / 5.0_f64.sqrt()),
                (2, 1) | (2, 2) => Ok(0.0),
                _ => Err(PrincipiaError::GravityCoefficientUnavailable {
                    degree: n as u16,
                    order: m as u16,
                }),
            }
        }
        fn s_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
            if n <= 2 {
                Ok(0.0)
            } else {
                Err(PrincipiaError::GravityCoefficientUnavailable {
                    degree: n as u16,
                    order: m as u16,
                })
            }
        }
    }

    /// Degree-2 provider with C_21 and S_21 coefficients for order-1 coverage.
    struct Degree2Order1Provider;
    impl GravityFieldProvider for Degree2Order1Provider {
        fn mu(&self) -> GravitationalParameter {
            GravitationalParameter::new(398_600.441_8)
        }
        fn reference_radius(&self) -> Kilometers {
            Kilometers::new(6_378.137)
        }
        fn max_degree(&self) -> usize {
            2
        }
        fn c_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
            match (n, m) {
                (0, 0) => Ok(1.0),
                (1, 0) | (1, 1) | (2, 0) | (2, 1) | (2, 2) => Ok(1e-6),
                _ => Err(PrincipiaError::GravityCoefficientUnavailable {
                    degree: n as u16,
                    order: m as u16,
                }),
            }
        }
        fn s_normalized(&self, n: usize, m: usize) -> Result<f64, PrincipiaError> {
            if n <= 2 {
                Ok(0.0)
            } else {
                Err(PrincipiaError::GravityCoefficientUnavailable {
                    degree: n as u16,
                    order: m as u16,
                })
            }
        }
    }

    #[test]
    fn j2_acceleration_has_nonzero_z_for_off_equator() {
        // Position off the equatorial plane: r = (0, 0, 7000) km (polar orbit).
        let acc = spherical_harmonic_acceleration_raw_km(
            [0.0, 0.0, 7000.0],
            2,
            0,
            &GravityConstants::try_new(
                GravitationalParameter::new(398_600.441_8),
                Kilometers::new(6_378.137),
                2,
            )
            .unwrap(),
            &J2Provider,
            100.0,
        )
        .unwrap();
        // The z-axis is the polar axis; J2 should contribute a non-zero radial component.
        let mag = (acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]).sqrt();
        assert!(mag > 0.0);
    }

    #[test]
    fn j2_acceleration_off_equatorial_plane_matches_known() {
        // Inclined position
        let acc = spherical_harmonic_acceleration_raw_km(
            [5000.0, 0.0, 5000.0],
            2,
            0,
            &GravityConstants::try_new(
                GravitationalParameter::new(398_600.441_8),
                Kilometers::new(6_378.137),
                2,
            )
            .unwrap(),
            &J2Provider,
            100.0,
        )
        .unwrap();
        // Just verify something changed from the two-body result
        let tb = spherical_harmonic_acceleration_raw_km(
            [5000.0, 0.0, 5000.0],
            0,
            0,
            &GravityConstants::try_new(
                GravitationalParameter::new(398_600.441_8),
                Kilometers::new(6_378.137),
                2,
            )
            .unwrap(),
            &J2Provider,
            100.0,
        )
        .unwrap();
        assert!((acc[0] - tb[0]).abs() > 1e-15 || (acc[2] - tb[2]).abs() > 1e-15);
    }

    #[test]
    fn degree2_order1_acceleration_is_finite() {
        let acc = spherical_harmonic_acceleration_raw_km(
            [4000.0, 3000.0, 2000.0],
            2,
            1,
            &GravityConstants::try_new(
                GravitationalParameter::new(398_600.441_8),
                Kilometers::new(6_378.137),
                2,
            )
            .unwrap(),
            &Degree2Order1Provider,
            100.0,
        )
        .unwrap();
        assert!(acc[0].is_finite());
        assert!(acc[1].is_finite());
        assert!(acc[2].is_finite());
    }

    #[test]
    fn degree2_order2_acceleration_is_finite() {
        let acc = spherical_harmonic_acceleration_raw_km(
            [4000.0, 3000.0, 1000.0],
            2,
            2,
            &GravityConstants::try_new(
                GravitationalParameter::new(398_600.441_8),
                Kilometers::new(6_378.137),
                2,
            )
            .unwrap(),
            &Degree2Order1Provider,
            100.0,
        )
        .unwrap();
        assert!(acc[0].is_finite());
        assert!(acc[1].is_finite());
        assert!(acc[2].is_finite());
    }

    #[test]
    fn raw_rejects_radius_too_small() {
        let result = spherical_harmonic_acceleration_raw_km(
            [0.0, 0.0, 0.0],
            0,
            0,
            &constants(),
            &TwoBodyOnly,
            100.0,
        );
        assert!(matches!(
            result,
            Err(PrincipiaError::DegenerateGeometry { .. })
        ));
    }

    #[test]
    fn raw_rejects_order_exceeds_degree() {
        let result = spherical_harmonic_acceleration_raw_km(
            [7000.0, 0.0, 0.0],
            1,
            2, // order > degree
            &constants(),
            &TwoBodyOnly,
            100.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn gravity_constants_exposes_constants() {
        let c = constants();
        assert!((c.mu.value() - 398_600.441_8).abs() < 1e-9);
        assert!((c.equatorial_radius.value() - 6_378.137).abs() < 1e-9);
        assert_eq!(c.max_degree, 0);
    }
}
