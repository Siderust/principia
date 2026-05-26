// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Vallés Puig, Ramon

//! Cartesian state covariance and process noise.
//!
//! ## Scientific scope
//!
//! [`StateCovariance<F>`] stores a `6 × 6` Cartesian covariance in `[r, v]`
//! block form. [`ProcessNoise<F>`] stores an additive process-noise matrix in
//! the same layout. Covariances can be transported through a state-transition
//! matrix `Φ` and rotated between inertial and local trajectory frames.
//!
//! ## Technical scope
//!
//! The covariance uses the canonical block decomposition
//! `[[Σ_rr, Σ_rv], [Σ_vr, Σ_vv]]` with `Σ_vr = Σ_rvᵀ` derived on demand.
//!
//! ## References
//!
//! * Tapley, Schutz, Born, *Statistical Orbit Determination*, §4.
//! * Montenbruck & Gill, *Satellite Orbits*, §7.

use affn::frames::ReferenceFrame;
use affn::matrix3::{FrameMatrix3, SymmetricFrameMatrix3};
use affn::matrix6::FrameMatrix6;
use affn::ops::Rotation3;
use qtty::length::Kilometers;
use qtty::{KmPerSecond, KmPerSecondSquared, Quantity, RelativeTolerance, Second};

use crate::error::PrincipiaError;
use crate::frames::LocalTrajectoryFrame;
use crate::variational::StateTransitionMatrix;

/// Frame-tagged Cartesian state covariance stored as three `3 × 3` blocks.
#[derive(Debug, Clone, Copy)]
pub struct StateCovariance<F: ReferenceFrame> {
    rr: SymmetricFrameMatrix3<F>,
    rv: FrameMatrix3<F>,
    vv: SymmetricFrameMatrix3<F>,
}

#[cfg(feature = "serde")]
impl<F: ReferenceFrame> serde::Serialize for StateCovariance<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.to_row_major().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, F: ReferenceFrame> serde::Deserialize<'de> for StateCovariance<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let matrix = <[[f64; 6]; 6]>::deserialize(deserializer)?;
        let mut flat = [0.0_f64; 36];
        for i in 0..6 {
            for j in 0..6 {
                flat[i * 6 + j] = matrix[i][j];
            }
        }
        Self::try_from_row_major(flat).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "serde")]
impl<F: ReferenceFrame> serde::Serialize for ProcessNoise<F> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, F: ReferenceFrame> serde::Deserialize<'de> for ProcessNoise<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = <[[f64; 6]; 6]>::deserialize(deserializer)?;
        for row in &data {
            for value in row {
                if !value.is_finite() {
                    return Err(serde::de::Error::custom(
                        "ProcessNoise: matrix entries must be finite",
                    ));
                }
            }
        }
        Ok(Self {
            data,
            _frame: core::marker::PhantomData,
        })
    }
}

impl<F: ReferenceFrame> StateCovariance<F> {
    /// Construct from explicit `rr`, `rv`, and `vv` blocks.
    #[inline]
    pub fn from_blocks(
        rr: SymmetricFrameMatrix3<F>,
        rv: FrameMatrix3<F>,
        vv: SymmetricFrameMatrix3<F>,
    ) -> Self {
        Self { rr, rv, vv }
    }

    /// Construct a diagonal covariance from typed one-sigma values.
    pub fn diagonal_from_sigmas(
        sigma_pos: [Kilometers; 3],
        sigma_vel: [Quantity<KmPerSecond>; 3],
    ) -> Self {
        Self {
            rr: SymmetricFrameMatrix3::from_diagonal([
                sigma_pos[0].value().powi(2),
                sigma_pos[1].value().powi(2),
                sigma_pos[2].value().powi(2),
            ]),
            rv: FrameMatrix3::zero(),
            vv: SymmetricFrameMatrix3::from_diagonal([
                sigma_vel[0].value().powi(2),
                sigma_vel[1].value().powi(2),
                sigma_vel[2].value().powi(2),
            ]),
        }
    }

    /// Construct a diagonal covariance from raw one-sigma values `[σx, σy, σz, σvx, σvy, σvz]`.
    pub fn try_diagonal_from_sigmas(sigmas: [f64; 6]) -> Result<Self, PrincipiaError> {
        for sigma in sigmas {
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(PrincipiaError::NonPositiveValue {
                    context: "StateCovariance: sigmas must be finite and positive",
                });
            }
        }
        Ok(Self::diagonal_from_sigmas(
            [
                Kilometers::new(sigmas[0]),
                Kilometers::new(sigmas[1]),
                Kilometers::new(sigmas[2]),
            ],
            [
                Quantity::<KmPerSecond>::new(sigmas[3]),
                Quantity::<KmPerSecond>::new(sigmas[4]),
                Quantity::<KmPerSecond>::new(sigmas[5]),
            ],
        ))
    }

    /// Construct from a raw row-major `6 × 6` matrix.
    pub fn from_row_major(m: [[f64; 6]; 6]) -> Self {
        Self {
            rr: SymmetricFrameMatrix3::from_upper([
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ]),
            rv: FrameMatrix3::from_array([
                [m[0][3], m[0][4], m[0][5]],
                [m[1][3], m[1][4], m[1][5]],
                [m[2][3], m[2][4], m[2][5]],
            ]),
            vv: SymmetricFrameMatrix3::from_upper([
                [m[3][3], m[3][4], m[3][5]],
                [m[4][3], m[4][4], m[4][5]],
                [m[5][3], m[5][4], m[5][5]],
            ]),
        }
    }

    /// Construct from a raw row-major `6 × 6` matrix flattened row-major.
    pub fn try_from_row_major(m: [f64; 36]) -> Result<Self, PrincipiaError> {
        let mut matrix = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                matrix[i][j] = m[i * 6 + j];
            }
        }
        validate_covariance_matrix(&matrix)?;
        let covariance = Self::from_row_major(matrix);
        if !covariance.is_positive_semidefinite(RelativeTolerance::new(1e-12)) {
            return Err(PrincipiaError::InvalidParameter {
                reason:
                    "StateCovariance: covariance matrix must be symmetric positive semidefinite",
            });
        }
        Ok(covariance)
    }

    /// Construct from already-prepared block components.
    #[inline]
    pub fn from_block_components(
        rr: SymmetricFrameMatrix3<F>,
        rv: FrameMatrix3<F>,
        vv: SymmetricFrameMatrix3<F>,
    ) -> Self {
        Self::from_blocks(rr, rv, vv)
    }

    /// Return the `Σ_rr` block.
    #[inline]
    pub fn rr(&self) -> &SymmetricFrameMatrix3<F> {
        &self.rr
    }

    /// Return the `Σ_rv` block.
    #[inline]
    pub fn rv(&self) -> &FrameMatrix3<F> {
        &self.rv
    }

    /// Return the derived `Σ_vr = Σ_rvᵀ` block.
    #[inline]
    pub fn vr(&self) -> FrameMatrix3<F> {
        self.rv.transpose()
    }

    /// Return the `Σ_vv` block.
    #[inline]
    pub fn vv(&self) -> &SymmetricFrameMatrix3<F> {
        &self.vv
    }

    /// Export the covariance as a full row-major `6 × 6` matrix.
    pub fn to_row_major(&self) -> [[f64; 6]; 6] {
        let rr = self.rr.as_array();
        let rv = self.rv.as_array();
        let vv = self.vv.as_array();
        let mut out = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = rr[i][j];
                out[i][j + 3] = rv[i][j];
                out[i + 3][j] = rv[j][i];
                out[i + 3][j + 3] = vv[i][j];
            }
        }
        out
    }

    /// Rotate the covariance by an instantaneous `3 × 3` rotation.
    pub fn rotate_by<G: ReferenceFrame>(self, r: &Rotation3) -> StateCovariance<G> {
        StateCovariance {
            rr: self.rr.rotated_by::<G>(r),
            rv: self.rv.rotated_by::<G>(r),
            vv: self.vv.rotated_by::<G>(r),
        }
    }

    /// Relabel the covariance with a different frame tag without changing data.
    pub fn relabel<G: ReferenceFrame>(self) -> StateCovariance<G> {
        StateCovariance {
            rr: self.rr.relabel::<G>(),
            rv: self.rv.relabel::<G>(),
            vv: self.vv.relabel::<G>(),
        }
    }

    /// Return `true` if the full `6 × 6` matrix is symmetric within `tol`.
    #[allow(clippy::needless_range_loop)]
    pub fn is_symmetric(&self, tol: RelativeTolerance) -> bool {
        let m = self.to_row_major();
        let t = tol.value();
        for i in 0..6 {
            for j in (i + 1)..6 {
                let a = m[i][j];
                let b = m[j][i];
                let scale = a.abs().max(b.abs());
                let diff = (a - b).abs();
                if scale == 0.0 {
                    if diff != 0.0 {
                        return false;
                    }
                } else if diff > t * scale {
                    return false;
                }
            }
        }
        true
    }

    /// Return `true` if the covariance is positive semidefinite within `tol`.
    #[allow(clippy::needless_range_loop)]
    pub fn is_positive_semidefinite(&self, tol: RelativeTolerance) -> bool {
        let m = self.to_row_major();
        let mut a = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                a[i][j] = 0.5 * (m[i][j] + m[j][i]);
            }
        }
        let trace: f64 = (0..6).map(|i| a[i][i]).sum();
        let eps = tol.value() * trace / 6.0;
        cholesky_in_place(&mut a, eps)
    }

    /// Symmetrize the stored blocks in place.
    pub fn symmetrise_in_place(&mut self) {
        let rr_arr = *self.rr.as_array();
        let vv_arr = *self.vv.as_array();
        let rv_arr = *self.rv.as_array();
        self.rr = SymmetricFrameMatrix3::from_upper([
            [
                rr_arr[0][0],
                0.5 * (rr_arr[0][1] + rr_arr[1][0]),
                0.5 * (rr_arr[0][2] + rr_arr[2][0]),
            ],
            [
                rr_arr[1][0],
                rr_arr[1][1],
                0.5 * (rr_arr[1][2] + rr_arr[2][1]),
            ],
            [rr_arr[2][0], rr_arr[2][1], rr_arr[2][2]],
        ]);
        self.vv = SymmetricFrameMatrix3::from_upper([
            [
                vv_arr[0][0],
                0.5 * (vv_arr[0][1] + vv_arr[1][0]),
                0.5 * (vv_arr[0][2] + vv_arr[2][0]),
            ],
            [
                vv_arr[1][0],
                vv_arr[1][1],
                0.5 * (vv_arr[1][2] + vv_arr[2][1]),
            ],
            [vv_arr[2][0], vv_arr[2][1], vv_arr[2][2]],
        ]);
        self.rv = FrameMatrix3::from_array([
            [
                rv_arr[0][0],
                0.5 * (rv_arr[0][1] + rv_arr[1][0]),
                0.5 * (rv_arr[0][2] + rv_arr[2][0]),
            ],
            [
                0.5 * (rv_arr[1][0] + rv_arr[0][1]),
                rv_arr[1][1],
                0.5 * (rv_arr[1][2] + rv_arr[2][1]),
            ],
            [
                0.5 * (rv_arr[2][0] + rv_arr[0][2]),
                0.5 * (rv_arr[2][1] + rv_arr[1][2]),
                rv_arr[2][2],
            ],
        ]);
    }

    /// Propagate the covariance through a state-transition matrix: `P ← Φ P Φᵀ`.
    pub fn transported_by(&self, phi: &StateTransitionMatrix<F>) -> Self {
        let p = FrameMatrix6::<F>::from_array(self.to_row_major());
        let propagated = phi.mat_mul(&p).mat_mul(&phi.transpose());
        Self::from_row_major(*propagated.as_array())
    }

    /// Rotate this covariance into a local trajectory frame.
    pub fn transform_into<Local: ReferenceFrame>(
        &self,
        frame: &LocalTrajectoryFrame<F, Local>,
    ) -> StateCovariance<Local> {
        StateCovariance {
            rr: frame.dcm.similarity::<Local>(&self.rr),
            rv: frame.dcm.similarity_general::<Local>(&self.rv),
            vv: frame.dcm.similarity::<Local>(&self.vv),
        }
    }
}

impl<Local: ReferenceFrame> StateCovariance<Local> {
    /// Rotate a local covariance back into its inertial parent frame.
    pub fn transform_into_inertial<Inertial: ReferenceFrame>(
        &self,
        frame: &LocalTrajectoryFrame<Inertial, Local>,
    ) -> StateCovariance<Inertial> {
        let r = frame.rotation_inverse();
        StateCovariance {
            rr: self.rr.rotated_by::<Inertial>(&r),
            rv: self.rv.rotated_by::<Inertial>(&r),
            vv: self.vv.rotated_by::<Inertial>(&r),
        }
    }
}

/// Frame-tagged additive process-noise matrix in `[r, v]` ordering.
#[derive(Debug, Clone, Copy)]
pub struct ProcessNoise<F: ReferenceFrame> {
    data: [[f64; 6]; 6],
    _frame: core::marker::PhantomData<F>,
}

impl<F: ReferenceFrame> ProcessNoise<F> {
    /// Return the zero process-noise matrix.
    pub fn zero() -> Self {
        Self {
            data: [[0.0; 6]; 6],
            _frame: core::marker::PhantomData,
        }
    }

    /// Build a diagonal process-noise model from position-rate and
    /// acceleration-rate sigmas over step `dt`.
    pub fn diagonal_from_sigmas(
        pos_rate: [Quantity<KmPerSecond>; 3],
        vel_rate: [Quantity<KmPerSecondSquared>; 3],
        dt: Second,
    ) -> Self {
        let dt_val = dt.value();
        let mut data = [[0.0; 6]; 6];
        for i in 0..3 {
            data[i][i] = pos_rate[i].value().powi(2) * dt_val;
            data[i + 3][i + 3] = vel_rate[i].value().powi(2) * dt_val;
        }
        Self {
            data,
            _frame: core::marker::PhantomData,
        }
    }

    /// Build a validated diagonal process-noise model from raw sigma rates.
    pub fn try_diagonal_from_sigmas(sigmas: [f64; 6], dt: Second) -> Result<Self, PrincipiaError> {
        for sigma in sigmas {
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(PrincipiaError::NonPositiveValue {
                    context: "ProcessNoise: sigmas must be finite and positive",
                });
            }
        }
        if !dt.value().is_finite() || dt.value() <= 0.0 {
            return Err(PrincipiaError::NonPositiveValue {
                context: "ProcessNoise: dt must be finite and positive",
            });
        }
        Ok(Self::diagonal_from_sigmas(
            [
                Quantity::<KmPerSecond>::new(sigmas[0]),
                Quantity::<KmPerSecond>::new(sigmas[1]),
                Quantity::<KmPerSecond>::new(sigmas[2]),
            ],
            [
                Quantity::<KmPerSecondSquared>::new(sigmas[3]),
                Quantity::<KmPerSecondSquared>::new(sigmas[4]),
                Quantity::<KmPerSecondSquared>::new(sigmas[5]),
            ],
            dt,
        ))
    }

    /// Add this process noise into a covariance in place.
    #[allow(clippy::needless_range_loop)]
    pub fn add_to(&self, cov: &mut StateCovariance<F>) {
        *cov = StateCovariance::from_row_major({
            let mut m = cov.to_row_major();
            for i in 0..6 {
                for j in 0..6 {
                    m[i][j] += self.data[i][j];
                }
            }
            m
        });
    }

    /// Export the raw row-major `6 × 6` matrix.
    #[inline]
    pub fn to_row_major(&self) -> [[f64; 6]; 6] {
        self.data
    }
}

#[allow(clippy::needless_range_loop)]
fn validate_covariance_matrix(m: &[[f64; 6]; 6]) -> Result<(), PrincipiaError> {
    for i in 0..6 {
        for j in 0..6 {
            if !m[i][j].is_finite() {
                return Err(PrincipiaError::InvalidParameter {
                    reason: "StateCovariance: matrix entries must be finite",
                });
            }
        }
        if m[i][i] <= 0.0 {
            return Err(PrincipiaError::InvalidParameter {
                reason: "StateCovariance: diagonal entries must be strictly positive",
            });
        }
    }
    for i in 0..6 {
        for j in (i + 1)..6 {
            let scale = m[i][j].abs().max(m[j][i].abs()).max(1.0);
            if (m[i][j] - m[j][i]).abs() > 1e-12 * scale {
                return Err(PrincipiaError::InvalidParameter {
                    reason: "StateCovariance: matrix must be symmetric",
                });
            }
        }
    }
    Ok(())
}

#[allow(clippy::needless_range_loop)]
fn cholesky_in_place(a: &mut [[f64; 6]; 6], eps: f64) -> bool {
    for i in 0..6 {
        let pivot = a[i][i] + eps;
        if pivot < 0.0 {
            return false;
        }
        let sqrt_pivot = pivot.sqrt();
        a[i][i] = sqrt_pivot;
        if sqrt_pivot == 0.0 {
            for j in (i + 1)..6 {
                a[j][i] = 0.0;
            }
            continue;
        }
        let inv = 1.0 / sqrt_pivot;
        for j in (i + 1)..6 {
            a[j][i] *= inv;
        }
        for j in (i + 1)..6 {
            let factor = a[j][i];
            for k in j..6 {
                a[k][j] -= factor * a[k][i];
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::{LocalTrajectoryFrame, RTN};
    use affn::centers::ReferenceCenter;
    use affn::frames::ReferenceFrame;
    use affn::matrix6::FrameMatrix6;
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

    fn state() -> crate::state::DynamicsState<TT, Center, Inertial> {
        crate::state::DynamicsState::new(
            Time::<TT>::from_raw_j2000_seconds(Second::new(0.0)).unwrap(),
            affn::cartesian::Position::<Center, Inertial, qtty::unit::Kilometer>::new(
                7000.0, 0.0, 0.0,
            ),
            affn::cartesian::Velocity::<Inertial, KmPerSecond>::new(0.0, 7.5, 0.0),
        )
    }

    #[test]
    fn diagonal_covariance_is_psd() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        assert!(p.is_positive_semidefinite(RelativeTolerance::new(1e-10)));
    }

    #[test]
    fn try_diagonal_from_sigmas_rejects_non_positive_sigma() {
        let result = StateCovariance::<Inertial>::try_diagonal_from_sigmas([
            1.0, 1.0, 0.0, 1e-3, 1e-3, 1e-3,
        ]);
        assert!(matches!(
            result,
            Err(PrincipiaError::NonPositiveValue { .. })
        ));
    }

    #[test]
    fn try_from_row_major_rejects_non_finite_entry() {
        let mut matrix = [0.0_f64; 36];
        for i in 0..6 {
            matrix[i * 6 + i] = 1.0;
        }
        matrix[1] = f64::NAN;
        let result = StateCovariance::<Inertial>::try_from_row_major(matrix);
        assert!(matches!(
            result,
            Err(PrincipiaError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn transport_with_identity_is_noop() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let phi = FrameMatrix6::<Inertial>::identity();
        assert_eq!(p.to_row_major(), p.transported_by(&phi).to_row_major());
    }

    #[test]
    fn inertial_local_round_trip() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let frame = LocalTrajectoryFrame::<Inertial, RTN>::try_from_state(&state()).unwrap();
        let back = p.transform_into(&frame).transform_into_inertial(&frame);
        let a = p.to_row_major();
        let b = back.to_row_major();
        for i in 0..6 {
            for j in 0..6 {
                assert!((a[i][j] - b[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn from_row_major_round_trip() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let m = p.to_row_major();
        let p2 = StateCovariance::<Inertial>::from_row_major(m);
        let m2 = p2.to_row_major();
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!((*value - m2[i][j]).abs() < 1e-30);
            }
        }
    }

    #[test]
    fn from_block_components_same_as_from_row_major() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let p2 = StateCovariance::from_block_components(*p.rr(), *p.rv(), *p.vv());
        let m1 = p.to_row_major();
        let m2 = p2.to_row_major();
        for (i, row) in m1.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!((*value - m2[i][j]).abs() < 1e-30);
            }
        }
    }

    #[test]
    fn block_getters_rr_rv_vr_vv() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let m = p.to_row_major();
        // rr block
        assert!((p.rr().as_array()[0][0] - m[0][0]).abs() < 1e-30);
        // vv block
        assert!((p.vv().as_array()[0][0] - m[3][3]).abs() < 1e-30);
        // rv block
        assert!((p.rv().as_array()[0][0] - m[0][3]).abs() < 1e-30);
        // vr = rvᵀ
        assert!((p.vr().as_array()[0][0] - m[3][0]).abs() < 1e-30);
    }

    #[test]
    fn is_symmetric_true_for_diagonal() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        assert!(p.is_symmetric(qtty::RelativeTolerance::new(1e-10)));
    }

    #[test]
    fn symmetrise_in_place_makes_diagonal_symmetric() {
        let mut p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        p.symmetrise_in_place();
        assert!(p.is_symmetric(qtty::RelativeTolerance::new(1e-10)));
    }

    #[test]
    fn rotate_by_identity_is_noop() {
        use affn::ops::Rotation3;
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let identity =
            Rotation3::from_matrix_unchecked([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let p2: StateCovariance<Inertial> = p.rotate_by(&identity);
        let m1 = p.to_row_major();
        let m2 = p2.to_row_major();
        for i in 0..6 {
            for j in 0..6 {
                assert!((m1[i][j] - m2[i][j]).abs() < 1e-20);
            }
        }
    }

    #[test]
    fn relabel_preserves_data() {
        let p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let m1 = p.to_row_major();
        let p2: StateCovariance<Inertial> = p.relabel();
        let m2 = p2.to_row_major();
        for i in 0..6 {
            for j in 0..6 {
                assert!((m1[i][j] - m2[i][j]).abs() < 1e-30);
            }
        }
    }

    #[test]
    fn process_noise_zero_has_all_zeros() {
        let pn = ProcessNoise::<Inertial>::zero();
        let m = pn.to_row_major();
        for row in &m {
            for val in row {
                assert_eq!(*val, 0.0);
            }
        }
    }

    #[test]
    fn process_noise_try_diagonal_from_sigmas_rejects_non_positive_dt() {
        let result = ProcessNoise::<Inertial>::try_diagonal_from_sigmas(
            [1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6],
            Second::new(0.0),
        );
        assert!(matches!(
            result,
            Err(PrincipiaError::NonPositiveValue { .. })
        ));
    }

    #[test]
    fn process_noise_add_to_increases_diagonal() {
        let pn = ProcessNoise::<Inertial>::diagonal_from_sigmas(
            [Quantity::<qtty::KmPerSecond>::new(1e-3); 3],
            [Quantity::<qtty::KmPerSecondSquared>::new(1e-6); 3],
            Second::new(1.0),
        );
        let mut p = StateCovariance::<Inertial>::diagonal_from_sigmas(
            [Kilometers::new(1.0); 3],
            [Quantity::<KmPerSecond>::new(1e-3); 3],
        );
        let before = p.to_row_major()[0][0];
        pn.add_to(&mut p);
        let after = p.to_row_major()[0][0];
        assert!(after > before);
    }

    #[test]
    fn cholesky_negative_definite_returns_not_psd() {
        // Build a matrix with a negative diagonal entry
        let mut m = [[0.0_f64; 6]; 6];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = if i == 2 { -1.0 } else { 1.0 };
        }
        let p = StateCovariance::<Inertial>::from_row_major(m);
        assert!(!p.is_positive_semidefinite(qtty::RelativeTolerance::new(1e-10)));
    }
}
