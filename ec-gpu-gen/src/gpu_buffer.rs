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

use ark_ff::{AdditiveGroup, PrimeField};
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

            // Upload ALL challenges in a single buffer (optimization #9)
            let challenges_buffer = program.create_buffer_from_slice(&challenges_vec)?;

            for challenge_idx in 0..num_challenges {
                let next_len = current_len / 2;

                // Create output buffer
                // SAFETY: GPU will initialize this buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let kernel_name = format!("{}_fix_var_indexed", F::name());
                let kernel =
                    program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

                kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&challenges_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(challenge_idx as u32))
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

            // Upload ALL challenges in a single buffer (optimization #9)
            let challenges_buffer = program.create_buffer_from_slice(&challenges_vec)?;

            // Phase 1: Fix vars and collect intermediates
            for challenge_idx in 0..num_challenges {
                let next_len = current_len / 2;

                // SAFETY: GPU will initialize this buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let kernel_name = format!("{}_fix_var_indexed", F::name());
                let kernel =
                    program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

                kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&challenges_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(challenge_idx as u32))
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

            // Step 3: Parallel batch witness computation (3-phase)
            let witness_len = poly_len - 1;
            let total_witnesses = num_points * witness_len;
            // SAFETY: GPU will initialize this buffer
            let witnesses_buffer = unsafe { program.create_buffer::<F>(total_witnesses)? };

            let chunk_size = std::cmp::max(1, witness_len / 4096);
            let num_chunks = div_ceil(witness_len, chunk_size);
            let total_phase1_threads = num_points * num_chunks;
            let carries_len = num_points * num_chunks;

            let carries_buffer = unsafe { program.create_buffer::<F>(carries_len)? };

            // Phase 1
            let phase1_kernel = program.create_kernel(
                &format!("{}_witness_poly_batch_phase1", F::name()),
                div_ceil(total_phase1_threads, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
            phase1_kernel
                .arg(&combined_buffer).arg(&witnesses_buffer).arg(&carries_buffer)
                .arg(&points_buffer)
                .arg(&(poly_len as u32)).arg(&(num_points as u32))
                .arg(&(chunk_size as u32)).arg(&(num_chunks as u32))
                .run()?;

            // Phase 2
            let propagated_carries_buffer = unsafe { program.create_buffer::<F>(carries_len)? };
            let phase2_kernel = program.create_kernel(
                &format!("{}_witness_carry_propagate", F::name()),
                div_ceil(num_points, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
            phase2_kernel
                .arg(&carries_buffer).arg(&propagated_carries_buffer)
                .arg(&points_buffer)
                .arg(&(num_chunks as u32)).arg(&(num_points as u32))
                .arg(&(chunk_size as u32)).arg(&(poly_len as u32))
                .run()?;

            // Phase 3
            let phase3_kernel = program.create_kernel(
                &format!("{}_witness_poly_batch_phase3", F::name()),
                div_ceil(total_phase1_threads, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
            phase3_kernel
                .arg(&witnesses_buffer).arg(&propagated_carries_buffer)
                .arg(&points_buffer)
                .arg(&(poly_len as u32)).arg(&(num_points as u32))
                .arg(&(chunk_size as u32)).arg(&(num_chunks as u32))
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
    /// Optimizations over the naive approach:
    /// 1. Bases are uploaded once and reused for all MSMs
    /// 2. Scalar bytes are generated on GPU (no CPU conversion + re-upload)
    /// 3. All operations happen in a single GPU session
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

        let closures = program_closures!(|program, _arg| -> EcResult<FixVarsAndCommitResult<F, G::Group>> {
            // Upload polynomial once
            let mut current_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            // Upload bases once (for all MSMs)
            let base_buffer = program.create_buffer_from_slice(bases)?;

            // Upload ALL challenges in a single buffer (optimization #9)
            let challenges_buffer = program.create_buffer_from_slice(&challenges_vec)?;

            let mut intermediates = Vec::with_capacity(num_challenges);
            let mut commitments = Vec::with_capacity(num_challenges);

            // Pre-allocate MSM buffers (sized for largest intermediate)
            // Signed-digit: half the buckets
            let window_size = ((div_ceil(initial_len / 2, work_units) as f64).log2() as usize) + 2;
            let window_size = std::cmp::min(window_size, max_window_size);
            let signed_bucket_len = 1 << (window_size - 1);

            // SAFETY: GPU will initialize these buffers
            let bucket_buffer = unsafe { program.create_buffer::<G::Group>(work_units * signed_bucket_len)? };
            let result_buffer = unsafe { program.create_buffer::<G::Group>(work_units)? };

            for challenge_idx in 0..num_challenges {
                let next_len = current_len / 2;

                // === Phase 1: fix_var (outputs Fr elements) ===
                // SAFETY: GPU will initialize this buffer
                let fr_out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let fix_var_global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let fix_var_kernel_name = format!("{}_fix_var_indexed", F::name());
                let fix_var_kernel = program.create_kernel(&fix_var_kernel_name, fix_var_global_work_size, LOCAL_WORK_SIZE)?;

                fix_var_kernel
                    .arg(&current_buffer)
                    .arg(&fr_out_buffer)
                    .arg(&challenges_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(challenge_idx as u32))
                    .run()?;

                // Download intermediate (needed for Phase 3 of HyperKZG)
                let mut intermediate = vec![F::ZERO; next_len];
                program.read_into_buffer(&fr_out_buffer, &mut intermediate)?;

                // === Phase 2: Convert Fr from Montgomery to standard form ON GPU ===
                // SAFETY: GPU will initialize this buffer
                let scalar_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let to_scalar_kernel_name = format!("{}_to_scalar_bytes", F::name());
                let to_scalar_kernel = program.create_kernel(&to_scalar_kernel_name, fix_var_global_work_size, LOCAL_WORK_SIZE)?;

                to_scalar_kernel
                    .arg(&fr_out_buffer)
                    .arg(&scalar_buffer)
                    .arg(&(next_len as u32))
                    .run()?;

                // === Phase 3: Signed-digit MSM commit ===
                const BN254_SCALAR_BITS: usize = 254;
                let effective_bits = BN254_SCALAR_BITS;
                let window_size_for_len = ((div_ceil(next_len, work_units) as f64).log2() as usize) + 2;
                let window_size_for_len = std::cmp::min(window_size_for_len, max_window_size);
                let num_windows = div_ceil(effective_bits + 1, window_size_for_len);
                let num_groups = work_units / num_windows;

                // Preprocess to signed digits
                let digits_len = next_len * num_windows;
                let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };
                let preprocess_global = div_ceil(next_len, LOCAL_WORK_SIZE);
                let preprocess_kernel = program.create_kernel(
                    &format!("{}_preprocess_signed_digits", G::name()),
                    preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&scalar_buffer)
                    .arg(&digits_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size_for_len as u32))
                    .run()?;

                // Signed multiexp
                let msm_global_work_size = div_ceil(num_windows * num_groups, MSM_LOCAL_WORK_SIZE);
                let msm_kernel = program.create_kernel(
                    &format!("{}_multiexp_signed", G::name()),
                    msm_global_work_size, MSM_LOCAL_WORK_SIZE)?;

                msm_kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&digits_buffer)
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
                current_buffer = fr_out_buffer;
                current_len = next_len;
            }

            Ok(FixVarsAndCommitResult {
                intermediates,
                commitments,
            })
        });

        self.program.run(closures, ())
    }

    /// Fused witness polynomial computation + MSM commit.
    ///
    /// This computes witness polynomials for KZG opening at multiple points and
    /// immediately commits to each witness polynomial, all in a single GPU session.
    /// The witness polynomials are never downloaded to CPU.
    ///
    /// This eliminates a GPU→CPU→GPU round-trip compared to calling
    /// witness_poly_batch followed by batch_commit separately.
    ///
    /// # Arguments
    /// * `poly` - The polynomial to open (B(x) in HyperKZG Phase 3)
    /// * `points` - Evaluation points (e.g., [r, -r, r²] in HyperKZG)
    /// * `bases` - G1 bases for MSM (must be at least poly.len() - 1)
    ///
    /// # Returns
    /// Commitments to each witness polynomial (one per point)
    pub fn witness_poly_batch_and_commit(
        &self,
        poly: &[F],
        points: &[F],
        bases: &[G::GpuRepr],
    ) -> EcResult<Vec<G::Group>> {
        assert!(!poly.is_empty(), "Polynomial must not be empty");
        assert!(!points.is_empty(), "Must have at least one point");

        let n = poly.len();
        let witness_len = n - 1;
        let num_points = points.len();

        assert!(
            bases.len() >= witness_len,
            "Not enough bases for witness polynomial size"
        );

        let poly_vec = poly.to_vec();
        let points_vec = points.to_vec();
        let work_units = self.work_units;
        let max_window_size = self.max_window_size;

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Group>> {
            // Upload polynomial once
            let poly_buffer = program.create_buffer_from_slice(&poly_vec)?;

            // Upload points
            let points_buffer = program.create_buffer_from_slice(&points_vec)?;

            // Upload bases once (for all witness MSMs)
            let base_buffer = program.create_buffer_from_slice(&bases[..witness_len])?;

            // Parallel witness polynomial batch computation (3-phase)
            // Output: num_points witness polynomials, each of length witness_len
            let total_witness_elements = num_points * witness_len;
            // SAFETY: GPU will initialize this buffer
            let witnesses_buffer = unsafe { program.create_buffer::<F>(total_witness_elements)? };

            let chunk_size = std::cmp::max(1, witness_len / 4096);
            let num_chunks = div_ceil(witness_len, chunk_size);
            let total_phase1_threads = num_points * num_chunks;
            let carries_len = num_points * num_chunks;

            // SAFETY: GPU will initialize these buffers
            let carries_buffer = unsafe { program.create_buffer::<F>(carries_len)? };

            // Phase 1
            let phase1_global_work_size = div_ceil(total_phase1_threads, LOCAL_WORK_SIZE);
            let phase1_kernel = program.create_kernel(
                &format!("{}_witness_poly_batch_phase1", F::name()),
                phase1_global_work_size, LOCAL_WORK_SIZE)?;
            phase1_kernel
                .arg(&poly_buffer).arg(&witnesses_buffer).arg(&carries_buffer)
                .arg(&points_buffer)
                .arg(&(n as u32)).arg(&(num_points as u32))
                .arg(&(chunk_size as u32)).arg(&(num_chunks as u32))
                .run()?;

            // Phase 2
            let propagated_carries_buffer = unsafe { program.create_buffer::<F>(carries_len)? };
            let phase2_kernel = program.create_kernel(
                &format!("{}_witness_carry_propagate", F::name()),
                div_ceil(num_points, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
            phase2_kernel
                .arg(&carries_buffer).arg(&propagated_carries_buffer)
                .arg(&points_buffer)
                .arg(&(num_chunks as u32)).arg(&(num_points as u32))
                .arg(&(chunk_size as u32)).arg(&(n as u32))
                .run()?;

            // Phase 3
            let phase3_kernel = program.create_kernel(
                &format!("{}_witness_poly_batch_phase3", F::name()),
                phase1_global_work_size, LOCAL_WORK_SIZE)?;
            phase3_kernel
                .arg(&witnesses_buffer).arg(&propagated_carries_buffer)
                .arg(&points_buffer)
                .arg(&(n as u32)).arg(&(num_points as u32))
                .arg(&(chunk_size as u32)).arg(&(num_chunks as u32))
                .run()?;

            // Now commit each witness polynomial using signed-digit MSM
            // Pre-allocate MSM buffers (half the buckets with signed decomposition)
            let window_size = ((div_ceil(witness_len, work_units) as f64).log2() as usize) + 2;
            let window_size = std::cmp::min(window_size, max_window_size);
            let signed_bucket_len = 1 << (window_size - 1);

            // SAFETY: GPU will initialize these buffers
            let bucket_buffer = unsafe { program.create_buffer::<G::Group>(work_units * signed_bucket_len)? };
            let result_buffer = unsafe { program.create_buffer::<G::Group>(work_units)? };

            // Buffer for scalar conversion (reused for each witness)
            // SAFETY: GPU will initialize this buffer
            let scalar_buffer = unsafe { program.create_buffer::<F>(witness_len)? };

            let mut commitments = Vec::with_capacity(num_points);

            // Use fixed 254-bit assumption for bn254 scalars
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;
            let num_windows = div_ceil(effective_bits + 1, window_size);
            let num_groups = work_units / num_windows;
            let msm_global_work_size = div_ceil(num_windows * num_groups, MSM_LOCAL_WORK_SIZE);

            // Signed digits buffer (reused for each witness)
            let digits_len = witness_len * num_windows;
            let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };

            for point_idx in 0..num_points {
                let witness_offset = point_idx * witness_len;

                // Convert witness Fr elements to scalar bytes ON GPU
                let to_scalar_kernel_name = format!("{}_to_scalar_bytes_offset", F::name());
                let to_scalar_global_work_size = div_ceil(witness_len, LOCAL_WORK_SIZE);
                let to_scalar_kernel = program.create_kernel(&to_scalar_kernel_name, to_scalar_global_work_size, LOCAL_WORK_SIZE)?;

                to_scalar_kernel
                    .arg(&witnesses_buffer)
                    .arg(&scalar_buffer)
                    .arg(&(witness_len as u32))
                    .arg(&(witness_offset as u32))
                    .run()?;

                // Preprocess to signed digits
                let preprocess_global = div_ceil(witness_len, LOCAL_WORK_SIZE);
                let preprocess_kernel = program.create_kernel(
                    &format!("{}_preprocess_signed_digits", G::name()),
                    preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&scalar_buffer)
                    .arg(&digits_buffer)
                    .arg(&(witness_len as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                // Signed MSM commit
                let msm_kernel = program.create_kernel(
                    &format!("{}_multiexp_signed", G::name()),
                    msm_global_work_size, MSM_LOCAL_WORK_SIZE)?;

                msm_kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&digits_buffer)
                    .arg(&(witness_len as u32))
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                // Download MSM partial results and accumulate
                let mut msm_results = vec![<G::Group as AdditiveGroup>::ZERO; work_units];
                program.read_into_buffer(&result_buffer, &mut msm_results)?;

                let mut acc = <G::Group as AdditiveGroup>::ZERO;
                for i in (0..num_windows).rev() {
                    let w = std::cmp::min(window_size, effective_bits - i * window_size);
                    for _ in 0..w {
                        acc = acc.double();
                    }
                    for g in 0..num_groups {
                        acc.add_assign(&msm_results[g * num_windows + i]);
                    }
                }

                commitments.push(acc);
            }

            Ok(commitments)
        });

        self.program.run(closures, ())
    }
}

/// Input data for Phase 3 of the fused HyperKZG open operation.
///
/// Returned by the middle callback to provide the CPU-computed parameters
/// needed for Phase 3 (linear combination + witness polynomials + witness MSMs).
///
/// Note: padded_polys are no longer needed — intermediate polynomial buffers are
/// kept alive on GPU and packed using copy_and_pad kernel, eliminating a ~2.8GB
/// CPU→GPU transfer.
pub struct Phase3Input<F, G> {
    /// Linear combination coefficients (q_powers from transcript).
    pub lc_coeffs: Vec<F>,
    /// Evaluation points (e.g., [r, -r, r²] in HyperKZG).
    pub eval_points: Vec<F>,
    /// Intermediate commitments in affine form (pass-through for the proof).
    pub intermediate_commitments_affine: Vec<G>,
    /// Evaluations v[i][j] = f_j(u_i) (pass-through for the proof).
    pub evaluations: Vec<Vec<F>>,
}

/// Result of a fused HyperKZG open operation.
///
/// `AffineG` is the affine commitment type (pass-through from the callback).
/// `ProjectiveG` is the projective group type (from GPU MSM output).
pub struct FusedOpenResult<F, AffineG, ProjectiveG> {
    /// Intermediate polynomials from fix_var iterations.
    pub intermediates: Vec<Vec<F>>,
    /// Commitments to intermediate polynomials (affine, from Phase3Input pass-through).
    pub intermediate_commitments_affine: Vec<AffineG>,
    /// Witness commitments (projective, from Phase 3 MSMs).
    pub witness_commitments: Vec<ProjectiveG>,
    /// Evaluations v[i][j] = f_j(u_i) (from Phase3Input pass-through).
    pub evaluations: Vec<Vec<F>>,
}

impl<F: PrimeField + GpuName, G: GpuAffine<ScalarField = F>> FusedPolyCommit<F, G> {
    /// Fused HyperKZG open: runs the entire open operation in one GPU session.
    ///
    /// This keeps bases on GPU for both Phase 2 (intermediate commits) and Phase 3
    /// (witness commits), and keeps intermediate polynomial data on GPU where possible.
    ///
    /// The `middle_fn` callback runs CPU-side transcript work between Phase 2 and Phase 3
    /// while GPU buffers remain alive. It receives the intermediate polynomials and their
    /// commitments, and must return `Phase3Input` with the parameters needed for Phase 3.
    ///
    /// # Savings vs current approach
    /// - Bases uploaded once (not twice)
    /// - No witness polynomial round-trip (GPU→CPU→GPU)
    /// - Scalar conversion happens on GPU (no CPU `convert_scalars_to_bigint`)
    /// - Linear combination result stays on GPU
    pub fn fused_open<MiddleFn, AffineG>(
        &self,
        poly: &[F],
        challenges: &[F],
        bases: &[G::GpuRepr],
        middle_fn: MiddleFn,
    ) -> EcResult<FusedOpenResult<F, AffineG, G::Group>>
    where
        MiddleFn: FnOnce(&[Vec<F>], &[G::Group]) -> Phase3Input<F, AffineG>,
    {
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

        let num_challenges = challenges.len();
        let initial_len = poly.len();
        let challenges_vec = challenges.to_vec();
        let poly_vec = poly.to_vec();
        let work_units = self.work_units;
        let max_window_size = self.max_window_size;

        let closures = program_closures!(|program, middle_fn: MiddleFn| -> EcResult<FusedOpenResult<F, AffineG, G::Group>> {
            // ================================================================
            // Upload bases ONCE — reused for Phase 2 and Phase 3 MSMs
            // ================================================================
            let base_buffer = program.create_buffer_from_slice(bases)?;

            // Upload polynomial — keep this buffer alive for GPU-side packing later
            let poly_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            let mut intermediates = Vec::with_capacity(num_challenges);
            let mut commitments = Vec::with_capacity(num_challenges);
            // Keep all intermediate GPU buffers alive for GPU-side packing in Phase 3
            let mut intermediate_gpu_buffers = Vec::with_capacity(num_challenges);
            let mut intermediate_lens = Vec::with_capacity(num_challenges);

            // ================================================================
            // Phase 1+2: fix_vars + intermediate MSM commits
            // ================================================================
            // We need current_buffer as a reference that advances through fix_var iterations.
            // But we also need to keep all intermediate buffers alive.
            // Use a "previous buffer index" approach: the first input is poly_buffer,
            // subsequent inputs are the previous intermediate buffer.
            let mut prev_buffer_is_poly = true;
            if num_challenges > 0 {
            // Upload challenges in a single buffer
            let challenges_buffer = program.create_buffer_from_slice(&challenges_vec)?;

            // Pre-allocate MSM buffers (sized for largest intermediate)
            // With signed-digit decomposition, bucket count is 2^(w-1) instead of 2^w - 1
            let window_size = {
                let ws = ((div_ceil(initial_len / 2, work_units) as f64).log2() as usize) + 2;
                std::cmp::min(ws, max_window_size)
            };
            let signed_bucket_len = 1 << (window_size - 1);

            // SAFETY: GPU will initialize these buffers
            let bucket_buffer = unsafe { program.create_buffer::<G::Group>(work_units * signed_bucket_len)? };
            let result_buffer = unsafe { program.create_buffer::<G::Group>(work_units)? };

            for challenge_idx in 0..num_challenges {
                let next_len = current_len / 2;

                // Phase 1: fix_var
                // SAFETY: GPU will initialize this buffer
                let fr_out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let fix_var_global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let fix_var_kernel_name = format!("{}_fix_var_indexed", F::name());
                let fix_var_kernel = program.create_kernel(&fix_var_kernel_name, fix_var_global_work_size, LOCAL_WORK_SIZE)?;

                // Input is either the original poly buffer or the previous intermediate
                let input_buffer = if prev_buffer_is_poly {
                    &poly_buffer
                } else {
                    &intermediate_gpu_buffers[challenge_idx - 1]
                };

                fix_var_kernel
                    .arg(input_buffer)
                    .arg(&fr_out_buffer)
                    .arg(&challenges_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(challenge_idx as u32))
                    .run()?;

                // Download intermediate (needed for CPU eval in middle_fn)
                let mut intermediate = vec![F::ZERO; next_len];
                program.read_into_buffer(&fr_out_buffer, &mut intermediate)?;

                // Phase 2: Convert Fr from Montgomery to standard form ON GPU
                // SAFETY: GPU will initialize this buffer
                let scalar_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let to_scalar_kernel_name = format!("{}_to_scalar_bytes", F::name());
                let to_scalar_kernel = program.create_kernel(&to_scalar_kernel_name, fix_var_global_work_size, LOCAL_WORK_SIZE)?;

                to_scalar_kernel
                    .arg(&fr_out_buffer)
                    .arg(&scalar_buffer)
                    .arg(&(next_len as u32))
                    .run()?;

                // Signed-digit MSM: preprocess + multiexp_signed
                const BN254_SCALAR_BITS: usize = 254;
                let effective_bits = BN254_SCALAR_BITS;
                let window_size_for_len = {
                    let ws = ((div_ceil(next_len, work_units) as f64).log2() as usize) + 2;
                    std::cmp::min(ws, max_window_size)
                };
                let num_windows = div_ceil(effective_bits + 1, window_size_for_len);
                let num_groups = work_units / num_windows;
                assert!(num_groups > 0, "MSM num_groups must be > 0 (work_units={work_units}, num_windows={num_windows})");

                // Preprocess exponents to signed digits
                let digits_len = next_len * num_windows;
                // SAFETY: GPU will initialize this buffer
                let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };
                let preprocess_global = div_ceil(next_len, LOCAL_WORK_SIZE);
                let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());
                let preprocess_kernel = program.create_kernel(&preprocess_kernel_name, preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&scalar_buffer)
                    .arg(&digits_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size_for_len as u32))
                    .run()?;

                // Signed multiexp with half the buckets
                let msm_global_work_size = div_ceil(num_windows * num_groups, MSM_LOCAL_WORK_SIZE);
                let msm_kernel_name = format!("{}_multiexp_signed", G::name());
                let msm_kernel = program.create_kernel(&msm_kernel_name, msm_global_work_size, MSM_LOCAL_WORK_SIZE)?;

                msm_kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&digits_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size_for_len as u32))
                    .run()?;

                // Download MSM partial results and accumulate
                let mut msm_results = vec![<G::Group as AdditiveGroup>::ZERO; work_units];
                program.read_into_buffer(&result_buffer, &mut msm_results)?;

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

                // Keep the fr_out_buffer alive for GPU-side packing
                intermediate_gpu_buffers.push(fr_out_buffer);
                intermediate_lens.push(next_len);

                prev_buffer_is_poly = false;
                current_len = next_len;
            }
            } // end if num_challenges > 0

            // ================================================================
            // CPU callback: transcript work between Phase 2 and Phase 3
            // GPU buffers (base_buffer, poly_buffer, intermediate_gpu_buffers) remain alive!
            // ================================================================
            let phase3 = middle_fn(&intermediates, &commitments);

            // ================================================================
            // Phase 3: GPU-side packing + linear_combine + witness + MSMs
            // ================================================================
            // Build flat padded polynomial buffer ON GPU using copy_and_pad kernel.
            // This eliminates the ~2.8GB CPU→GPU upload of padded_polys.
            let num_polys = 1 + num_challenges; // original poly + intermediates
            let poly_len = initial_len;          // all padded to max length
            let num_points = phase3.eval_points.len();

            // Allocate flat buffer: num_polys * poly_len elements
            // SAFETY: GPU will initialize this buffer via copy_and_pad kernels
            let polys_buffer = unsafe { program.create_buffer::<F>(num_polys * poly_len)? };

            // Copy original polynomial (full length, no padding needed)
            let copy_pad_kernel_name = format!("{}_copy_and_pad", F::name());
            let pad_global_work_size = div_ceil(poly_len, LOCAL_WORK_SIZE);
            let copy_kernel_0 = program.create_kernel(&copy_pad_kernel_name, pad_global_work_size, LOCAL_WORK_SIZE)?;
            copy_kernel_0
                .arg(&poly_buffer)
                .arg(&polys_buffer)
                .arg(&(initial_len as u32))
                .arg(&(poly_len as u32))
                .arg(&(0u32))
                .run()?;

            // Copy each intermediate (shorter, padded with zeros to poly_len)
            for (idx, (buf, &src_len)) in intermediate_gpu_buffers.iter()
                .zip(intermediate_lens.iter()).enumerate()
            {
                let copy_kernel = program.create_kernel(&copy_pad_kernel_name, pad_global_work_size, LOCAL_WORK_SIZE)?;
                copy_kernel
                    .arg(buf)
                    .arg(&polys_buffer)
                    .arg(&(src_len as u32))
                    .arg(&(poly_len as u32))
                    .arg(&((idx + 1) as u32))
                    .run()?;
            }

            // Upload coefficients and eval points
            let coeffs_buffer = program.create_buffer_from_slice(&phase3.lc_coeffs)?;
            let points_buffer = program.create_buffer_from_slice(&phase3.eval_points)?;

            // Linear combination on GPU
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

            // Parallel witness polynomial batch computation on GPU (3-phase)
            let witness_len = poly_len - 1;
            let total_witness_elements = num_points * witness_len;
            // SAFETY: GPU will initialize this buffer
            let witnesses_buffer = unsafe { program.create_buffer::<F>(total_witness_elements)? };

            // Choose chunk_size so we get enough threads for GPU saturation.
            // Target ~4096 chunks per point, but at least 1 element per chunk.
            let chunk_size = std::cmp::max(1, witness_len / 4096);
            let num_chunks = div_ceil(witness_len, chunk_size);

            // Phase 1: each thread processes one (point, chunk) pair independently
            let total_phase1_threads = num_points * num_chunks;
            let carries_len = num_points * num_chunks;
            // SAFETY: GPU will initialize these buffers
            let carries_buffer = unsafe { program.create_buffer::<F>(carries_len)? };

            let phase1_global_work_size = div_ceil(total_phase1_threads, LOCAL_WORK_SIZE);
            let phase1_kernel_name = format!("{}_witness_poly_batch_phase1", F::name());
            let phase1_kernel = program.create_kernel(&phase1_kernel_name, phase1_global_work_size, LOCAL_WORK_SIZE)?;

            phase1_kernel
                .arg(&combined_buffer)
                .arg(&witnesses_buffer)
                .arg(&carries_buffer)
                .arg(&points_buffer)
                .arg(&(poly_len as u32))
                .arg(&(num_points as u32))
                .arg(&(chunk_size as u32))
                .arg(&(num_chunks as u32))
                .run()?;

            // Phase 2: propagate carries across chunks (one thread per eval point)
            // SAFETY: GPU will initialize this buffer
            let propagated_carries_buffer = unsafe { program.create_buffer::<F>(carries_len)? };

            let phase2_global_work_size = div_ceil(num_points, LOCAL_WORK_SIZE);
            let phase2_kernel_name = format!("{}_witness_carry_propagate", F::name());
            let phase2_kernel = program.create_kernel(&phase2_kernel_name, phase2_global_work_size, LOCAL_WORK_SIZE)?;

            phase2_kernel
                .arg(&carries_buffer)
                .arg(&propagated_carries_buffer)
                .arg(&points_buffer)
                .arg(&(num_chunks as u32))
                .arg(&(num_points as u32))
                .arg(&(chunk_size as u32))
                .arg(&(poly_len as u32))
                .run()?;

            // Phase 3: apply carry corrections to each chunk's witness values
            let phase3_global_work_size = div_ceil(total_phase1_threads, LOCAL_WORK_SIZE);
            let phase3_kernel_name = format!("{}_witness_poly_batch_phase3", F::name());
            let phase3_kernel = program.create_kernel(&phase3_kernel_name, phase3_global_work_size, LOCAL_WORK_SIZE)?;

            phase3_kernel
                .arg(&witnesses_buffer)
                .arg(&propagated_carries_buffer)
                .arg(&points_buffer)
                .arg(&(poly_len as u32))
                .arg(&(num_points as u32))
                .arg(&(chunk_size as u32))
                .arg(&(num_chunks as u32))
                .run()?;

            // MSM for each witness polynomial, reusing base_buffer from Phase 2
            // Uses signed-digit decomposition for half the buckets
            let witness_window_size = {
                let ws = ((div_ceil(witness_len, work_units) as f64).log2() as usize) + 2;
                std::cmp::min(ws, max_window_size)
            };

            const BN254_SCALAR_BITS_P3: usize = 254;
            let effective_bits_p3 = BN254_SCALAR_BITS_P3;
            let num_windows_p3 = div_ceil(effective_bits_p3 + 1, witness_window_size);
            let num_groups_p3 = work_units / num_windows_p3;
            assert!(num_groups_p3 > 0, "Phase 3 MSM num_groups must be > 0 (work_units={work_units}, num_windows={num_windows_p3})");
            let msm_global_work_size_p3 = div_ceil(num_windows_p3 * num_groups_p3, MSM_LOCAL_WORK_SIZE);

            // Signed-digit: half the buckets
            let witness_signed_bucket_len = 1 << (witness_window_size - 1);
            let bucket_buffer_p3 = unsafe { program.create_buffer::<G::Group>(work_units * witness_signed_bucket_len)? };
            let result_buffer_p3 = unsafe { program.create_buffer::<G::Group>(work_units)? };

            // Scalar buffer for witness conversion (FIELD layout = EXPONENT layout after unmont)
            // SAFETY: GPU will initialize this buffer
            let scalar_buffer_p3 = unsafe { program.create_buffer::<F>(witness_len)? };
            // Signed digits buffer for witness MSMs
            let digits_len_p3 = witness_len * num_windows_p3;
            let digits_buffer_p3 = unsafe { program.create_buffer::<u16>(digits_len_p3)? };

            let mut witness_commitments = Vec::with_capacity(num_points);

            for point_idx in 0..num_points {
                let witness_offset = point_idx * witness_len;

                // Convert witness Fr → scalar bytes ON GPU using offset kernel
                let to_scalar_kernel_name = format!("{}_to_scalar_bytes_offset", F::name());
                let to_scalar_global_work_size = div_ceil(witness_len, LOCAL_WORK_SIZE);
                let to_scalar_kernel = program.create_kernel(&to_scalar_kernel_name, to_scalar_global_work_size, LOCAL_WORK_SIZE)?;

                to_scalar_kernel
                    .arg(&witnesses_buffer)
                    .arg(&scalar_buffer_p3)
                    .arg(&(witness_len as u32))
                    .arg(&(witness_offset as u32))
                    .run()?;

                // Preprocess to signed digits
                let preprocess_global_p3 = div_ceil(witness_len, LOCAL_WORK_SIZE);
                let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());
                let preprocess_kernel = program.create_kernel(&preprocess_kernel_name, preprocess_global_p3, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&scalar_buffer_p3)
                    .arg(&digits_buffer_p3)
                    .arg(&(witness_len as u32))
                    .arg(&(num_windows_p3 as u32))
                    .arg(&(witness_window_size as u32))
                    .run()?;

                // Signed MSM commit reusing base_buffer
                let msm_kernel_name = format!("{}_multiexp_signed", G::name());
                let msm_kernel = program.create_kernel(&msm_kernel_name, msm_global_work_size_p3, MSM_LOCAL_WORK_SIZE)?;

                msm_kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer_p3)
                    .arg(&result_buffer_p3)
                    .arg(&digits_buffer_p3)
                    .arg(&(witness_len as u32))
                    .arg(&(num_groups_p3 as u32))
                    .arg(&(num_windows_p3 as u32))
                    .arg(&(witness_window_size as u32))
                    .run()?;

                // Download MSM partial results and accumulate
                let mut msm_results = vec![<G::Group as AdditiveGroup>::ZERO; work_units];
                program.read_into_buffer(&result_buffer_p3, &mut msm_results)?;

                let mut acc = <G::Group as AdditiveGroup>::ZERO;
                for i in (0..num_windows_p3).rev() {
                    let w = std::cmp::min(witness_window_size, effective_bits_p3 - i * witness_window_size);
                    for _ in 0..w {
                        acc = acc.double();
                    }
                    for g in 0..num_groups_p3 {
                        acc.add_assign(&msm_results[g * num_windows_p3 + i]);
                    }
                }

                witness_commitments.push(acc);
            }

            Ok(FusedOpenResult {
                intermediates,
                intermediate_commitments_affine: phase3.intermediate_commitments_affine,
                witness_commitments,
                evaluations: phase3.evaluations,
            })
        });

        self.program.run(closures, middle_fn)
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
