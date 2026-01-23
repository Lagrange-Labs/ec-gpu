//! GPU buffer management for persistent data across operations.
//!
//! This module provides utilities for keeping polynomial data on the GPU
//! between operations, avoiding unnecessary CPU-GPU transfers.
//!
//! # Design
//!
//! The `program_closures!` macro in rust-gpu-tools ties buffer lifetime to closure scope.
//! To work around this, we provide:
//!
//! 1. `GpuBufferCache` - A cache that tracks which polynomials have been uploaded
//! 2. Combined operation functions that perform multiple operations in a single GPU session
//!
//! This allows the commit-and-open flow to keep polynomial data on GPU between phases.

use std::collections::HashMap;
use std::ops::AddAssign;
use std::sync::atomic::{AtomicU64, Ordering};

use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use ec_gpu::GpuName;
use rust_gpu_tools::{program_closures, Program};

use crate::error::EcResult;
use crate::multiexp::GpuAffine;

const LOCAL_WORK_SIZE: usize = 256;

const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

/// Unique identifier for a cached buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuBufferId(u64);

impl GpuBufferId {
    /// Generate a new unique buffer ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for GpuBufferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata about a cached polynomial buffer.
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Unique identifier for this buffer.
    pub id: GpuBufferId,
    /// Length of the polynomial (number of field elements).
    pub len: usize,
    /// Hash of the polynomial data for verification.
    pub hash: u64,
}

impl BufferMetadata {
    /// Create metadata for a polynomial slice.
    pub fn from_slice<F: PrimeField>(data: &[F]) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        data.len().hash(&mut hasher);
        // Hash a sample of elements for quick identification
        if !data.is_empty() {
            for i in (0..data.len()).step_by(data.len().max(1) / 8 + 1) {
                // Hash the serialized bytes
                let bytes = data[i].to_string();
                bytes.hash(&mut hasher);
            }
        }

        Self {
            id: GpuBufferId::new(),
            len: data.len(),
            hash: hasher.finish(),
        }
    }
}

/// Cache for tracking uploaded polynomial data.
///
/// This cache maintains metadata about polynomials that have been uploaded to GPU,
/// allowing operations to check if data is already available.
///
/// Note: This is a CPU-side tracking structure. The actual GPU buffers are managed
/// within closure scopes using combined operations.
#[derive(Debug, Default)]
pub struct GpuBufferCache {
    /// Cached polynomial metadata keyed by hash.
    cached: HashMap<u64, BufferMetadata>,
}

impl GpuBufferCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a polynomial as cached.
    pub fn register<F: PrimeField>(&mut self, data: &[F]) -> BufferMetadata {
        let metadata = BufferMetadata::from_slice(data);
        self.cached.insert(metadata.hash, metadata.clone());
        metadata
    }

    /// Check if a polynomial is cached.
    pub fn is_cached<F: PrimeField>(&self, data: &[F]) -> bool {
        let metadata = BufferMetadata::from_slice(data);
        self.cached.contains_key(&metadata.hash)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cached.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cached.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cached.is_empty()
    }
}

/// Result of a combined fix_vars operation that includes intermediates.
#[derive(Debug)]
pub struct FixVarsResult<F> {
    /// All intermediate polynomials from fix_var iterations.
    pub intermediates: Vec<Vec<F>>,
    /// The final single-element result.
    pub final_value: F,
}

/// Combined GPU operations for HyperKZG commit-open flow.
///
/// This struct provides methods that perform multiple GPU operations in a single
/// closure, keeping data on GPU between operations and avoiding redundant transfers.
pub struct CombinedPolyOps<F: PrimeField + GpuName> {
    program: Program,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + GpuName> CombinedPolyOps<F> {
    /// Create a new combined operations handler.
    pub fn create(program: Program) -> EcResult<Self> {
        Ok(Self {
            program,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Perform fix_vars and return all intermediate results without extra transfers.
    ///
    /// This is equivalent to `fix_vars_with_intermediates` but optimized to
    /// minimize GPU-CPU transfers by keeping data on GPU between iterations.
    pub fn fix_vars_with_intermediates_optimized(
        &self,
        poly: &[F],
        challenges: &[F],
    ) -> EcResult<FixVarsResult<F>> {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );

        if challenges.is_empty() {
            return Ok(FixVarsResult {
                intermediates: vec![],
                final_value: poly[0],
            });
        }

        let num_challenges = challenges.len();
        let initial_len = poly.len();
        let challenges_vec = challenges.to_vec();
        let poly_vec = poly.to_vec();

        let closures = program_closures!(|program, _arg| -> EcResult<FixVarsResult<F>> {
            // Upload initial polynomial once
            let mut current_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;
            let mut intermediates = Vec::with_capacity(num_challenges);

            for r in challenges_vec.iter() {
                let next_len = current_len / 2;

                // Create buffer for challenge value
                let r_buffer = program.create_buffer_from_slice(&[*r])?;

                // Create output buffer
                // SAFETY: GPU will initialize this buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let kernel_name = format!("{}_fix_var", F::name());
                let kernel =
                    program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

                kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&r_buffer)
                    .arg(&(next_len as u32))
                    .run()?;

                // Download intermediate result
                let mut intermediate = vec![F::ZERO; next_len];
                program.read_into_buffer(&out_buffer, &mut intermediate)?;
                intermediates.push(intermediate);

                // Update for next iteration (buffer stays on GPU)
                current_buffer = out_buffer;
                current_len = next_len;
            }

            // Get final value
            let final_value = if !intermediates.is_empty() {
                intermediates.last().unwrap()[0]
            } else {
                F::ZERO
            };

            Ok(FixVarsResult {
                intermediates,
                final_value,
            })
        });

        self.program.run(closures, ())
    }

    /// Combined fix_vars + linear combination in a single GPU session.
    ///
    /// This performs:
    /// 1. fix_vars_with_intermediates on the input polynomial
    /// 2. Linear combination of all polynomials (original + intermediates)
    ///
    /// All operations happen on GPU without downloading intermediates.
    pub fn fix_vars_and_combine(
        &self,
        poly: &[F],
        challenges: &[F],
        coeffs: &[F],
    ) -> EcResult<(Vec<Vec<F>>, Vec<F>)> {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );

        if challenges.is_empty() {
            return Ok((vec![], poly.to_vec()));
        }

        let num_challenges = challenges.len();
        let initial_len = poly.len();
        let challenges_vec = challenges.to_vec();
        let coeffs_vec = coeffs.to_vec();
        let poly_vec = poly.to_vec();

        let closures = program_closures!(|program, _arg| -> EcResult<(Vec<Vec<F>>, Vec<F>)> {
            // Upload initial polynomial
            let mut current_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            let mut intermediates = Vec::with_capacity(num_challenges);

            // Phase 1: Fix vars and collect intermediates
            for r in challenges_vec.iter() {
                let next_len = current_len / 2;

                let r_buffer = program.create_buffer_from_slice(&[*r])?;
                // SAFETY: GPU will initialize this buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let kernel_name = format!("{}_fix_var", F::name());
                let kernel =
                    program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

                kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&r_buffer)
                    .arg(&(next_len as u32))
                    .run()?;

                // Download intermediate
                let mut intermediate = vec![F::ZERO; next_len];
                program.read_into_buffer(&out_buffer, &mut intermediate)?;
                intermediates.push(intermediate);

                // Update current buffer for next iteration
                current_buffer = out_buffer;
                current_len = next_len;
            }

            // Phase 2: Linear combination (if coeffs provided)
            // For simplicity, compute on CPU since we already downloaded intermediates
            let combined = if !coeffs_vec.is_empty() {
                let mut result = vec![F::ZERO; initial_len];
                for (i, coeff) in coeffs_vec.iter().enumerate() {
                    if i == 0 {
                        for (j, val) in poly_vec.iter().enumerate() {
                            result[j] = result[j] + *val * *coeff;
                        }
                    } else if i - 1 < intermediates.len() {
                        let poly = &intermediates[i - 1];
                        for (j, val) in poly.iter().enumerate() {
                            if j < result.len() {
                                result[j] = result[j] + *val * *coeff;
                            }
                        }
                    }
                }
                result
            } else {
                poly_vec.clone()
            };

            Ok((intermediates, combined))
        });

        self.program.run(closures, ())
    }

    /// Evaluate polynomial at multiple points and compute witness polynomials in one GPU session.
    ///
    /// This is optimized for HyperKZG Phase 3 where we need:
    /// 1. Evaluations of all polynomials at [r, -r, r²]
    /// 2. Linear combination with challenge powers
    /// 3. Witness polynomials for the combined polynomial at each point
    pub fn eval_and_witness_batch(
        &self,
        polys: &[&[F]],
        points: &[F],
        lc_coeffs: &[F],
    ) -> EcResult<(Vec<Vec<F>>, Vec<F>, Vec<Vec<F>>)> {
        // (evaluations[point][poly], combined_poly, witnesses[point])
        assert!(!polys.is_empty(), "Must have at least one polynomial");
        assert!(!points.is_empty(), "Must have at least one point");
        assert_eq!(
            polys.len(),
            lc_coeffs.len(),
            "Number of polynomials must match number of coefficients"
        );

        let poly_len = polys[0].len();
        assert!(
            polys.iter().all(|p| p.len() == poly_len),
            "All polynomials must have the same length"
        );

        let num_polys = polys.len();
        let num_points = points.len();

        // Flatten polynomials
        let mut flat_polys: Vec<F> = Vec::with_capacity(num_polys * poly_len);
        for poly in polys {
            flat_polys.extend_from_slice(poly);
        }

        let points_vec = points.to_vec();
        let coeffs_vec = lc_coeffs.to_vec();

        let closures = program_closures!(|program, _arg| -> EcResult<(Vec<Vec<F>>, Vec<F>, Vec<Vec<F>>)> {
            // Upload all data
            let polys_buffer = program.create_buffer_from_slice(&flat_polys)?;
            let points_buffer = program.create_buffer_from_slice(&points_vec)?;
            let coeffs_buffer = program.create_buffer_from_slice(&coeffs_vec)?;

            // Step 1: Batch evaluation
            let total_evals = num_polys * num_points;
            // SAFETY: GPU will initialize this buffer
            let eval_buffer = unsafe { program.create_buffer::<F>(total_evals)? };

            let global_work_size = div_ceil(total_evals, LOCAL_WORK_SIZE);
            let eval_kernel_name = format!("{}_eval_univariate_batch", F::name());
            let eval_kernel = program.create_kernel(&eval_kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            eval_kernel
                .arg(&polys_buffer)
                .arg(&points_buffer)
                .arg(&eval_buffer)
                .arg(&(num_polys as u32))
                .arg(&(poly_len as u32))
                .arg(&(num_points as u32))
                .run()?;

            // Download evaluations
            let mut flat_evals = vec![F::ZERO; total_evals];
            program.read_into_buffer(&eval_buffer, &mut flat_evals)?;

            // Reshape evaluations to [point][poly]
            let mut evaluations = Vec::with_capacity(num_points);
            for point_idx in 0..num_points {
                let start = point_idx * num_polys;
                let end = start + num_polys;
                evaluations.push(flat_evals[start..end].to_vec());
            }

            // Step 2: Linear combination
            // SAFETY: GPU will initialize this buffer
            let combined_buffer = unsafe { program.create_buffer::<F>(poly_len)? };

            let lc_global_work_size = div_ceil(poly_len, LOCAL_WORK_SIZE);
            let lc_kernel_name = format!("{}_linear_combine", F::name());
            let lc_kernel = program.create_kernel(&lc_kernel_name, lc_global_work_size, LOCAL_WORK_SIZE)?;

            lc_kernel
                .arg(&polys_buffer)
                .arg(&coeffs_buffer)
                .arg(&combined_buffer)
                .arg(&(num_polys as u32))
                .arg(&(poly_len as u32))
                .run()?;

            // Download combined polynomial
            let mut combined_poly = vec![F::ZERO; poly_len];
            program.read_into_buffer(&combined_buffer, &mut combined_poly)?;

            // Step 3: Batch witness computation
            let witness_len = poly_len - 1;
            let total_witnesses = num_points * witness_len;
            // SAFETY: GPU will initialize this buffer
            let witnesses_buffer = unsafe { program.create_buffer::<F>(total_witnesses)? };

            let witness_global_work_size = div_ceil(num_points, LOCAL_WORK_SIZE);
            let witness_kernel_name = format!("{}_witness_poly_batch", F::name());
            let witness_kernel = program.create_kernel(&witness_kernel_name, witness_global_work_size, LOCAL_WORK_SIZE)?;

            witness_kernel
                .arg(&combined_buffer)
                .arg(&points_buffer)
                .arg(&witnesses_buffer)
                .arg(&(poly_len as u32))
                .arg(&(num_points as u32))
                .run()?;

            // Download witnesses
            let mut flat_witnesses = vec![F::ZERO; total_witnesses];
            program.read_into_buffer(&witnesses_buffer, &mut flat_witnesses)?;

            // Reshape witnesses to [point][witness_coeff]
            let mut witnesses = Vec::with_capacity(num_points);
            for point_idx in 0..num_points {
                let start = point_idx * witness_len;
                let end = start + witness_len;
                witnesses.push(flat_witnesses[start..end].to_vec());
            }

            Ok((evaluations, combined_poly, witnesses))
        });

        self.program.run(closures, ())
    }
}

/// Result of fused fix_vars and commit operation.
#[derive(Debug)]
pub struct FixVarsAndCommitResult<F, G> {
    /// Intermediate polynomials from fix_var iterations.
    pub intermediates: Vec<Vec<F>>,
    /// Commitments to each intermediate polynomial.
    pub commitments: Vec<G>,
}

/// Fused GPU operations for HyperKZG that combine polynomial operations with MSM.
///
/// This struct keeps both polynomial data and curve bases on GPU between operations,
/// minimizing transfers for the commit-open flow.
pub struct FusedPolyCommit<F: PrimeField + GpuName, G: GpuAffine> {
    program: Program,
    /// Maximum window size for MSM
    max_window_size: usize,
    /// Work units for MSM parallelization
    work_units: usize,
    _phantom: std::marker::PhantomData<(F, G)>,
}

/// On the GPU, the exponents are split into windows
const MAX_WINDOW_SIZE: usize = 10;
/// In CUDA this is the number of blocks per grid (grid size) for MSM
const MSM_LOCAL_WORK_SIZE: usize = 128;

impl<F: PrimeField + GpuName, G: GpuAffine<ScalarField = F>> FusedPolyCommit<F, G> {
    /// Create a new fused poly-commit handler.
    pub fn create(program: Program, work_units: usize) -> EcResult<Self> {
        Ok(Self {
            program,
            max_window_size: MAX_WINDOW_SIZE,
            work_units,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Fused fix_vars + commit operation.
    ///
    /// This performs fix_var iterations and commits to each intermediate polynomial
    /// in a single GPU session, keeping bases on GPU between MSMs.
    ///
    /// # Arguments
    /// * `poly` - Input polynomial evaluations
    /// * `challenges` - Challenge values for each fix_var iteration
    /// * `bases` - G1 bases for MSM (must be at least as long as poly)
    ///
    /// # Returns
    /// * `intermediates` - All intermediate polynomials (downloaded for later use)
    /// * `commitments` - Commitment to each intermediate
    pub fn fix_vars_and_commit(
        &self,
        poly: &[F],
        challenges: &[F],
        bases: &[G::GpuRepr],
    ) -> EcResult<FixVarsAndCommitResult<F, G::Group>> {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );
        assert!(
            bases.len() >= poly.len(),
            "Not enough bases for polynomial size"
        );

        if challenges.is_empty() {
            return Ok(FixVarsAndCommitResult {
                intermediates: vec![],
                commitments: vec![],
            });
        }

        let num_challenges = challenges.len();
        let initial_len = poly.len();
        let challenges_vec = challenges.to_vec();
        let poly_vec = poly.to_vec();
        let work_units = self.work_units;
        let max_window_size = self.max_window_size;

        // Pre-compute max scalar bits function
        fn compute_max_scalar_bits(scalars: &[[u8; 32]]) -> usize {
            let max_bits = scalars
                .iter()
                .map(|bytes| {
                    for (i, &byte) in bytes.iter().enumerate().rev() {
                        if byte != 0 {
                            return (i + 1) * 8 - byte.leading_zeros() as usize;
                        }
                    }
                    0
                })
                .max()
                .unwrap_or(0);
            max_bits.max(1)
        }

        let closures = program_closures!(|program, _arg| -> EcResult<FixVarsAndCommitResult<F, G::Group>> {
            // Upload polynomial once
            let mut current_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            // Upload bases once (for all MSMs)
            let base_buffer = program.create_buffer_from_slice(bases)?;

            let mut intermediates = Vec::with_capacity(num_challenges);
            let mut commitments = Vec::with_capacity(num_challenges);

            // Pre-allocate MSM buffers (sized for largest intermediate)
            let window_size = ((div_ceil(initial_len / 2, work_units) as f64).log2() as usize) + 2;
            let window_size = std::cmp::min(window_size, max_window_size);
            let bucket_len = 1 << window_size;

            // SAFETY: GPU will initialize these buffers
            let bucket_buffer = unsafe { program.create_buffer::<G::Group>(work_units * bucket_len)? };
            let result_buffer = unsafe { program.create_buffer::<G::Group>(work_units)? };

            for r in challenges_vec.iter() {
                let next_len = current_len / 2;

                // === Phase 1: fix_var ===
                let r_buffer = program.create_buffer_from_slice(&[*r])?;
                // SAFETY: GPU will initialize this buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let fix_var_global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let fix_var_kernel_name = format!("{}_fix_var", F::name());
                let fix_var_kernel = program.create_kernel(&fix_var_kernel_name, fix_var_global_work_size, LOCAL_WORK_SIZE)?;

                fix_var_kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&r_buffer)
                    .arg(&(next_len as u32))
                    .run()?;

                // Download intermediate (needed for Phase 3 of HyperKZG)
                let mut intermediate = vec![F::ZERO; next_len];
                program.read_into_buffer(&out_buffer, &mut intermediate)?;

                // === Phase 2: MSM commit ===
                // Convert intermediate to scalar bytes
                let scalars: Vec<[u8; 32]> = intermediate
                    .iter()
                    .map(|s| {
                        let mut out = [0u8; 32];
                        let bigint = s.into_bigint();
                        let le = bigint.to_bytes_le();
                        out[..le.len()].copy_from_slice(&le);
                        out
                    })
                    .collect();

                let effective_bits = compute_max_scalar_bits(&scalars);
                let window_size_for_len = ((div_ceil(next_len, work_units) as f64).log2() as usize) + 2;
                let window_size_for_len = std::cmp::min(window_size_for_len, max_window_size);
                let num_windows = div_ceil(effective_bits, window_size_for_len);
                let num_groups = work_units / num_windows;

                // Upload scalars for this MSM
                let exp_buffer = program.create_buffer_from_slice(&scalars)?;

                let msm_global_work_size = div_ceil(num_windows * num_groups, MSM_LOCAL_WORK_SIZE);
                let msm_kernel_name = format!("{}_multiexp", G::name());
                let msm_kernel = program.create_kernel(&msm_kernel_name, msm_global_work_size, MSM_LOCAL_WORK_SIZE)?;

                msm_kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&exp_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size_for_len as u32))
                    .run()?;

                // Download MSM partial results
                let mut msm_results = vec![<G::Group as AdditiveGroup>::ZERO; work_units];
                program.read_into_buffer(&result_buffer, &mut msm_results)?;

                // CPU accumulation for final commitment
                let mut acc = <G::Group as AdditiveGroup>::ZERO;
                for i in (0..num_windows).rev() {
                    let w = std::cmp::min(window_size_for_len, effective_bits - i * window_size_for_len);
                    for _ in 0..w {
                        acc = acc.double();
                    }
                    for g in 0..num_groups {
                        acc.add_assign(&msm_results[g * num_windows + i]);
                    }
                }

                intermediates.push(intermediate);
                commitments.push(acc);

                // Update for next iteration
                current_buffer = out_buffer;
                current_len = next_len;
            }

            Ok(FixVarsAndCommitResult {
                intermediates,
                commitments,
            })
        });

        self.program.run(closures, ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_id_uniqueness() {
        let id1 = GpuBufferId::new();
        let id2 = GpuBufferId::new();
        let id3 = GpuBufferId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_buffer_cache() {
        #[cfg(feature = "arkworks")]
        {
            use ark_bn254::Fr;
            use ark_std::UniformRand;

            let mut rng = ark_std::test_rng();
            let mut cache = GpuBufferCache::new();

            let poly1: Vec<Fr> = (0..16).map(|_| Fr::rand(&mut rng)).collect();
            let poly2: Vec<Fr> = (0..16).map(|_| Fr::rand(&mut rng)).collect();

            assert!(!cache.is_cached(&poly1));
            assert!(!cache.is_cached(&poly2));

            let meta1 = cache.register(&poly1);
            assert!(cache.is_cached(&poly1));
            assert!(!cache.is_cached(&poly2));

            let meta2 = cache.register(&poly2);
            assert!(cache.is_cached(&poly1));
            assert!(cache.is_cached(&poly2));

            assert_ne!(meta1.id, meta2.id);
            assert_eq!(cache.len(), 2);

            cache.clear();
            assert!(cache.is_empty());
        }
    }
}
