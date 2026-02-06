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
                            result[j] += *val * *coeff;
                        }
                    } else if i - 1 < intermediates.len() {
                        let poly = &intermediates[i - 1];
                        for (j, val) in poly.iter().enumerate() {
                            if j < result.len() {
                                result[j] += *val * *coeff;
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
    #[allow(clippy::type_complexity)]
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

        #[allow(clippy::type_complexity)]
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
    _phantom: std::marker::PhantomData<(F, G)>,
}

/// On the GPU, the exponents are split into windows
const MAX_WINDOW_SIZE: usize = 16;
/// In CUDA this is the number of blocks per grid (grid size) for MSM
const MSM_LOCAL_WORK_SIZE: usize = 128;

/// Compute optimal window size for sort-based MSM.
///
/// For sort-based MSM, each bucket is processed by 1 thread serially.
/// Bucket size = n / 2^(ws-1), so larger ws = smaller buckets = more parallelism.
/// Target: ~1000 points per bucket (ws ≈ log2(n) - 8).
///
/// This replaces the old per-thread formula `log2(n/work_units) + 2` which gave ws=9
/// for n=4M, creating ~15,600 pts/bucket — massive serial chains.
fn calc_sort_window_size(n_bases: usize) -> usize {
    let log2_val = (n_bases as f64).log2() as usize;
    let ws = std::cmp::max(3, log2_val.saturating_sub(8));
    std::cmp::min(ws, MAX_WINDOW_SIZE)
}

impl<F: PrimeField + GpuName, G: GpuAffine<ScalarField = F>> FusedPolyCommit<F, G> {
    /// Create a new fused poly-commit handler.
    pub fn create(program: Program, _work_units: usize) -> EcResult<Self> {
        Ok(Self {
            program,
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

            // === Pre-allocate sort-based MSM buffers at max sizes ===
            // n_bases halves each iteration (initial_len/2, initial_len/4, ..., 2).
            // Smaller n_bases → smaller window_size → more windows → potentially more total_pairs.
            // We compute the max over all iterations.
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;
            let mut max_total_pairs = 0usize;
            let mut max_total_buckets = 0usize;
            let mut max_digits_len = 0usize;
            let mut max_num_windows = 0usize;
            {
                let mut len = initial_len;
                for _ in 0..num_challenges {
                    let nb = len / 2;
                    let ws = calc_sort_window_size(nb);
                    let nw = div_ceil(effective_bits + 1, ws);
                    let tp = nb * nw;
                    let bpw = 1usize << (ws - 1);
                    let tb = nw * bpw;
                    max_total_pairs = std::cmp::max(max_total_pairs, tp);
                    max_total_buckets = std::cmp::max(max_total_buckets, tb);
                    max_digits_len = std::cmp::max(max_digits_len, nb * nw);
                    max_num_windows = std::cmp::max(max_num_windows, nw);
                    len = nb;
                }
            }

            // Pre-allocate GPU buffers once at max sizes
            let digits_buffer = unsafe { program.create_buffer::<u16>(max_digits_len)? };
            let keys_buffer = unsafe { program.create_buffer::<u32>(max_total_pairs)? };
            let values_buffer = unsafe { program.create_buffer::<u32>(max_total_pairs)? };
            let sorted_values_buffer = unsafe { program.create_buffer::<u32>(max_total_pairs)? };
            let mut counts_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let offsets_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let nonempty_ids_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let mut scatter_offsets_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let num_nonempty_buffer = unsafe { program.create_buffer::<u32>(1)? };
            let mut bucket_results_buffer = {
                let identity_points = vec![<G::Group as AdditiveGroup>::ZERO; max_total_buckets];
                program.create_buffer_from_slice(&identity_points)?
            };
            let mut window_results_buffer = {
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; max_num_windows];
                program.create_buffer_from_slice(&window_results)?
            };
            let mut final_result_buffer = {
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.create_buffer_from_slice(&final_result)?
            };
            // CPU-side scratch vectors
            let mut offsets_copy = vec![0u32; max_total_buckets];
            let mut counts_cpu = vec![0u32; max_total_buckets];
            let mut nonempty_ids_cpu = vec![0u32; max_total_buckets];

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

                // === Phase 3: Sort-based MSM commit ===
                let window_size_for_len = calc_sort_window_size(next_len);
                let num_windows = div_ceil(effective_bits + 1, window_size_for_len);
                let n_bases = next_len;

                // Preprocess to signed digits (reuse pre-allocated digits_buffer)
                let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                let preprocess_kernel = program.create_kernel(
                    &format!("{}_preprocess_signed_digits", G::name()),
                    preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&scalar_buffer)
                    .arg(&digits_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size_for_len as u32))
                    .run()?;

                // Sort-based MSM pipeline
                let total_pairs = n_bases * num_windows;
                let buckets_per_window = 1usize << (window_size_for_len - 1);
                let total_buckets = num_windows * buckets_per_window;

                // Step 1: Decompose to (key, value) pairs (reuse pre-allocated buffers)
                let decompose_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let decompose_kernel = program.create_kernel(
                    &format!("{}_decompose_to_pairs", G::name()),
                    decompose_global, MSM_LOCAL_WORK_SIZE)?;
                decompose_kernel
                    .arg(&digits_buffer)
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                // Step 2: Count buckets (re-init counts to zero via write_from_buffer)
                // Must write max_total_buckets elements to match buffer size
                program.write_from_buffer(&mut counts_buffer, &vec![0u32; max_total_buckets])?;
                let count_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let count_kernel = program.create_kernel(
                    &format!("{}_count_buckets", G::name()),
                    count_global, MSM_LOCAL_WORK_SIZE)?;
                count_kernel
                    .arg(&keys_buffer)
                    .arg(&counts_buffer)
                    .arg(&(total_pairs as u32))
                    .run()?;

                // Step 3: Prefix sum (single thread, reuse pre-allocated buffers)
                let prefix_kernel = program.create_kernel(
                    &format!("{}_prefix_sum", G::name()), 1, 1)?;
                prefix_kernel
                    .arg(&counts_buffer)
                    .arg(&offsets_buffer)
                    .arg(&nonempty_ids_buffer)
                    .arg(&num_nonempty_buffer)
                    .arg(&(total_buckets as u32))
                    .run()?;

                // Step 4: Download num_nonempty
                let mut num_nonempty_vec = vec![0u32; 1];
                program.read_into_buffer(&num_nonempty_buffer, &mut num_nonempty_vec)?;
                let num_nonempty = num_nonempty_vec[0] as usize;

                // Step 5: Copy offsets for scatter (scatter modifies them via atomicAdd)
                // Read/write full buffer size; kernel only accesses [0..total_buckets)
                program.read_into_buffer(&offsets_buffer, &mut offsets_copy)?;
                program.write_from_buffer(&mut scatter_offsets_buffer, &offsets_copy)?;

                // Step 6: Scatter to sorted (reuse pre-allocated sorted_values_buffer)
                let scatter_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let scatter_kernel = program.create_kernel(
                    &format!("{}_scatter_to_sorted", G::name()),
                    scatter_global, MSM_LOCAL_WORK_SIZE)?;
                scatter_kernel
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&scatter_offsets_buffer)
                    .arg(&sorted_values_buffer)
                    .arg(&(total_pairs as u32))
                    .run()?;

                // Step 7: Accumulate sorted buckets (with chunked dispatch for large buckets)
                // Re-init bucket_results to identity (full buffer size)
                program.write_from_buffer(&mut bucket_results_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; max_total_buckets])?;
                if num_nonempty > 0 {
                    // Download counts and nonempty IDs for dispatch table construction
                    // Read full buffer; only use [0..total_buckets) on CPU
                    program.read_into_buffer(&counts_buffer, &mut counts_cpu)?;
                    program.read_into_buffer(&nonempty_ids_buffer, &mut nonempty_ids_cpu)?;

                    let (dispatch_table, reduce_table, num_dispatches) =
                        crate::multiexp::build_dispatch_tables(&counts_cpu[..total_buckets], &nonempty_ids_cpu[..total_buckets], num_nonempty);

                    if num_dispatches == num_nonempty {
                        // No large buckets — use simple 1-thread-per-bucket kernel (faster for small buckets)
                        let accum_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let accum_kernel = program.create_kernel(
                            &format!("{}_accumulate_sorted_buckets", G::name()),
                            accum_global, MSM_LOCAL_WORK_SIZE)?;
                        accum_kernel
                            .arg(&base_buffer)
                            .arg(&sorted_values_buffer)
                            .arg(&offsets_buffer)
                            .arg(&counts_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    } else {
                        // Large buckets detected — use chunked accumulation
                        let dispatch_buffer = program.create_buffer_from_slice(&dispatch_table)?;
                        let partial_results_buffer = {
                            let partials = vec![<G::Group as AdditiveGroup>::ZERO; num_dispatches];
                            program.create_buffer_from_slice(&partials)?
                        };

                        // Phase 3b: Chunked accumulation
                        let chunked_global = div_ceil(num_dispatches, MSM_LOCAL_WORK_SIZE);
                        let chunked_kernel = program.create_kernel(
                            &format!("{}_accumulate_chunked", G::name()),
                            chunked_global, MSM_LOCAL_WORK_SIZE)?;
                        chunked_kernel
                            .arg(&base_buffer)
                            .arg(&sorted_values_buffer)
                            .arg(&offsets_buffer)
                            .arg(&dispatch_buffer)
                            .arg(&partial_results_buffer)
                            .arg(&(num_dispatches as u32))
                            .run()?;

                        // Phase 3c: Reduce partial results per bucket
                        let reduce_table_buffer = program.create_buffer_from_slice(&reduce_table)?;
                        let reduce_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let reduce_kernel = program.create_kernel(
                            &format!("{}_reduce_partial_buckets", G::name()),
                            reduce_global, MSM_LOCAL_WORK_SIZE)?;
                        reduce_kernel
                            .arg(&partial_results_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&reduce_table_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    }
                }

                // Step 8: Reduce buckets by window (re-init window_results to identity, full buffer size)
                program.write_from_buffer(&mut window_results_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; max_num_windows])?;
                let reduce_buckets_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                let reduce_buckets_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_by_window", G::name()),
                    reduce_buckets_global, MSM_LOCAL_WORK_SIZE)?;
                reduce_buckets_kernel
                    .arg(&bucket_results_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                // Step 9: Horner reduction on GPU (single thread, re-init final_result to identity)
                program.write_from_buffer(&mut final_result_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; 1])?;
                let reduce_windows_kernel = program.create_kernel(
                    &format!("{}_reduce_windows", G::name()), 1, 1)?;
                reduce_windows_kernel
                    .arg(&window_results_buffer)
                    .arg(&final_result_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(window_size_for_len as u32))
                    .arg(&(effective_bits as u32))
                    .run()?;

                // Step 10: Download final result (just 1 point!)
                let mut final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.read_into_buffer(&final_result_buffer, &mut final_result)?;
                let commitment = final_result[0];

                intermediates.push(intermediate);
                commitments.push(commitment);

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

            // Now commit each witness polynomial using sort-based MSM
            // Buffer for scalar conversion (reused for each witness)
            // SAFETY: GPU will initialize this buffer
            let scalar_buffer = unsafe { program.create_buffer::<F>(witness_len)? };

            let mut commitments = Vec::with_capacity(num_points);

            // Use fixed 254-bit assumption for bn254 scalars
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;
            let window_size = calc_sort_window_size(witness_len);
            let num_windows = div_ceil(effective_bits + 1, window_size);

            // Signed digits buffer (reused for each witness)
            let digits_len = witness_len * num_windows;
            let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };

            // === Pre-allocate sort-based MSM buffers (constant size across iterations) ===
            let n_bases = witness_len;
            let total_pairs = n_bases * num_windows;
            let buckets_per_window = 1usize << (window_size - 1);
            let total_buckets = num_windows * buckets_per_window;

            let keys_buffer = unsafe { program.create_buffer::<u32>(total_pairs)? };
            let values_buffer = unsafe { program.create_buffer::<u32>(total_pairs)? };
            let sorted_values_buffer = unsafe { program.create_buffer::<u32>(total_pairs)? };
            let mut counts_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let offsets_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let nonempty_ids_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let mut scatter_offsets_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let num_nonempty_buffer = unsafe { program.create_buffer::<u32>(1)? };
            let mut bucket_results_buffer = {
                let identity_points = vec![<G::Group as AdditiveGroup>::ZERO; total_buckets];
                program.create_buffer_from_slice(&identity_points)?
            };
            let mut window_results_buffer = {
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; num_windows];
                program.create_buffer_from_slice(&window_results)?
            };
            let mut final_result_buffer = {
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.create_buffer_from_slice(&final_result)?
            };
            // CPU-side scratch vectors
            let mut offsets_copy = vec![0u32; total_buckets];
            let mut counts_cpu = vec![0u32; total_buckets];
            let mut nonempty_ids_cpu = vec![0u32; total_buckets];

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

                // Sort-based MSM pipeline (reuse pre-allocated buffers)

                // Step 1: Decompose to (key, value) pairs
                let decompose_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let decompose_kernel = program.create_kernel(
                    &format!("{}_decompose_to_pairs", G::name()),
                    decompose_global, MSM_LOCAL_WORK_SIZE)?;
                decompose_kernel
                    .arg(&digits_buffer)
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                // Step 2: Count buckets (re-init counts to zero via write_from_buffer)
                program.write_from_buffer(&mut counts_buffer, &vec![0u32; total_buckets])?;
                let count_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let count_kernel = program.create_kernel(
                    &format!("{}_count_buckets", G::name()),
                    count_global, MSM_LOCAL_WORK_SIZE)?;
                count_kernel
                    .arg(&keys_buffer)
                    .arg(&counts_buffer)
                    .arg(&(total_pairs as u32))
                    .run()?;

                // Step 3: Prefix sum (single thread, reuse pre-allocated buffers)
                let prefix_kernel = program.create_kernel(
                    &format!("{}_prefix_sum", G::name()), 1, 1)?;
                prefix_kernel
                    .arg(&counts_buffer)
                    .arg(&offsets_buffer)
                    .arg(&nonempty_ids_buffer)
                    .arg(&num_nonempty_buffer)
                    .arg(&(total_buckets as u32))
                    .run()?;

                // Step 4: Download num_nonempty
                let mut num_nonempty_vec = vec![0u32; 1];
                program.read_into_buffer(&num_nonempty_buffer, &mut num_nonempty_vec)?;
                let num_nonempty = num_nonempty_vec[0] as usize;

                // Step 5: Copy offsets for scatter (scatter modifies them via atomicAdd)
                program.read_into_buffer(&offsets_buffer, &mut offsets_copy)?;
                program.write_from_buffer(&mut scatter_offsets_buffer, &offsets_copy)?;

                // Step 6: Scatter to sorted (reuse pre-allocated sorted_values_buffer)
                let scatter_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let scatter_kernel = program.create_kernel(
                    &format!("{}_scatter_to_sorted", G::name()),
                    scatter_global, MSM_LOCAL_WORK_SIZE)?;
                scatter_kernel
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&scatter_offsets_buffer)
                    .arg(&sorted_values_buffer)
                    .arg(&(total_pairs as u32))
                    .run()?;

                // Step 7: Accumulate sorted buckets (with chunked dispatch for large buckets)
                // Re-init bucket_results to identity
                program.write_from_buffer(&mut bucket_results_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; total_buckets])?;
                if num_nonempty > 0 {
                    // Download counts and nonempty IDs for dispatch table construction
                    program.read_into_buffer(&counts_buffer, &mut counts_cpu)?;
                    program.read_into_buffer(&nonempty_ids_buffer, &mut nonempty_ids_cpu)?;

                    let (dispatch_table, reduce_table, num_dispatches) =
                        crate::multiexp::build_dispatch_tables(&counts_cpu, &nonempty_ids_cpu, num_nonempty);

                    if num_dispatches == num_nonempty {
                        // No large buckets — use simple 1-thread-per-bucket kernel (faster for small buckets)
                        let accum_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let accum_kernel = program.create_kernel(
                            &format!("{}_accumulate_sorted_buckets", G::name()),
                            accum_global, MSM_LOCAL_WORK_SIZE)?;
                        accum_kernel
                            .arg(&base_buffer)
                            .arg(&sorted_values_buffer)
                            .arg(&offsets_buffer)
                            .arg(&counts_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    } else {
                        // Large buckets detected — use chunked accumulation
                        let dispatch_buffer = program.create_buffer_from_slice(&dispatch_table)?;
                        let partial_results_buffer = {
                            let partials = vec![<G::Group as AdditiveGroup>::ZERO; num_dispatches];
                            program.create_buffer_from_slice(&partials)?
                        };

                        // Phase 3b: Chunked accumulation
                        let chunked_global = div_ceil(num_dispatches, MSM_LOCAL_WORK_SIZE);
                        let chunked_kernel = program.create_kernel(
                            &format!("{}_accumulate_chunked", G::name()),
                            chunked_global, MSM_LOCAL_WORK_SIZE)?;
                        chunked_kernel
                            .arg(&base_buffer)
                            .arg(&sorted_values_buffer)
                            .arg(&offsets_buffer)
                            .arg(&dispatch_buffer)
                            .arg(&partial_results_buffer)
                            .arg(&(num_dispatches as u32))
                            .run()?;

                        // Phase 3c: Reduce partial results per bucket
                        let reduce_table_buffer = program.create_buffer_from_slice(&reduce_table)?;
                        let reduce_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let reduce_kernel = program.create_kernel(
                            &format!("{}_reduce_partial_buckets", G::name()),
                            reduce_global, MSM_LOCAL_WORK_SIZE)?;
                        reduce_kernel
                            .arg(&partial_results_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&reduce_table_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    }
                }

                // Step 8: Reduce buckets by window (re-init window_results to identity)
                program.write_from_buffer(&mut window_results_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; num_windows])?;
                let reduce_buckets_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                let reduce_buckets_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_by_window", G::name()),
                    reduce_buckets_global, MSM_LOCAL_WORK_SIZE)?;
                reduce_buckets_kernel
                    .arg(&bucket_results_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                // Step 9: Horner reduction on GPU (single thread, re-init final_result to identity)
                program.write_from_buffer(&mut final_result_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; 1])?;
                let reduce_windows_kernel = program.create_kernel(
                    &format!("{}_reduce_windows", G::name()), 1, 1)?;
                reduce_windows_kernel
                    .arg(&window_results_buffer)
                    .arg(&final_result_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .arg(&(effective_bits as u32))
                    .run()?;

                // Step 10: Download final result (just 1 point!)
                let mut final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.read_into_buffer(&final_result_buffer, &mut final_result)?;
                let commitment = final_result[0];

                commitments.push(commitment);
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
    /// Batch commit: computes MSM commitments for multiple polynomials in a single GPU session.
    ///
    /// Uses the sort-based MSM pipeline (same as fused_open) rather than the old
    /// per-thread bucket approach. All polynomials must be the same length (caller
    /// pads shorter ones).
    ///
    /// # Arguments
    /// * `polys` - Polynomial evaluations in Montgomery form (all same length)
    /// * `bases` - Pre-converted GPU bases (at least as long as each poly)
    ///
    /// # Returns
    /// One commitment (projective point) per polynomial.
    pub fn batch_commit(
        &self,
        polys: &[&[F]],
        bases: &[G::GpuRepr],
    ) -> EcResult<Vec<G::Group>> {
        if polys.is_empty() {
            return Ok(vec![]);
        }

        let max_len = polys.iter().map(|p| p.len()).max().unwrap();
        assert!(max_len > 0, "All polynomials are empty");
        assert!(
            bases.len() >= max_len,
            "Not enough bases for polynomial size"
        );

        let num_polys = polys.len();

        // CPU-side scratch buffer for zero-padded poly data (reused per poly)
        let mut padded_scratch: Vec<F> = vec![F::ZERO; max_len];

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Group>> {
            // Upload bases once (for all MSMs)
            let base_buffer = program.create_buffer_from_slice(&bases[..max_len])?;

            // Reusable GPU buffer for one poly at a time (replaces massive flat buffer)
            let mut fr_buffer = program.create_buffer_from_slice(&padded_scratch)?;

            // Compute MSM params (constant across all polys since using max_len)
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;
            let n_bases = max_len;
            let window_size = calc_sort_window_size(n_bases);
            let num_windows = div_ceil(effective_bits + 1, window_size);
            let total_pairs = n_bases * num_windows;
            let buckets_per_window = 1usize << (window_size - 1);
            let total_buckets = num_windows * buckets_per_window;

            // Pre-allocate all sort buffers once (constant size across iterations)
            let scalar_buffer = unsafe { program.create_buffer::<F>(max_len)? };
            let digits_len = max_len * num_windows;
            let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };
            let keys_buffer = unsafe { program.create_buffer::<u32>(total_pairs)? };
            let values_buffer = unsafe { program.create_buffer::<u32>(total_pairs)? };
            let sorted_values_buffer = unsafe { program.create_buffer::<u32>(total_pairs)? };
            let mut counts_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let offsets_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let nonempty_ids_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let mut scatter_offsets_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
            let num_nonempty_buffer = unsafe { program.create_buffer::<u32>(1)? };
            let mut bucket_results_buffer = {
                let identity_points = vec![<G::Group as AdditiveGroup>::ZERO; total_buckets];
                program.create_buffer_from_slice(&identity_points)?
            };
            let mut window_results_buffer = {
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; num_windows];
                program.create_buffer_from_slice(&window_results)?
            };
            let mut final_result_buffer = {
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.create_buffer_from_slice(&final_result)?
            };
            // CPU-side scratch vectors
            let mut offsets_copy = vec![0u32; total_buckets];
            let mut counts_cpu = vec![0u32; total_buckets];
            let mut nonempty_ids_cpu = vec![0u32; total_buckets];

            let mut commitments = Vec::with_capacity(num_polys);

            for poly_idx in 0..num_polys {
                // Build zero-padded poly on CPU and upload to reusable fr_buffer
                let poly = polys[poly_idx];
                padded_scratch[..poly.len()].copy_from_slice(poly);
                // Zero-fill the tail (only needed if poly is shorter than max_len)
                for v in padded_scratch[poly.len()..].iter_mut() {
                    *v = F::ZERO;
                }
                program.write_from_buffer(&mut fr_buffer, &padded_scratch)?;

                // Convert Montgomery → standard form on GPU via to_scalar_bytes
                let to_scalar_kernel_name = format!("{}_to_scalar_bytes", F::name());
                let to_scalar_global_work_size = div_ceil(max_len, LOCAL_WORK_SIZE);
                let to_scalar_kernel = program.create_kernel(&to_scalar_kernel_name, to_scalar_global_work_size, LOCAL_WORK_SIZE)?;

                to_scalar_kernel
                    .arg(&fr_buffer)
                    .arg(&scalar_buffer)
                    .arg(&(max_len as u32))
                    .run()?;

                // Preprocess to signed digits
                let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                let preprocess_kernel = program.create_kernel(
                    &format!("{}_preprocess_signed_digits", G::name()),
                    preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&scalar_buffer)
                    .arg(&digits_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                // Sort-based MSM pipeline (reuse pre-allocated buffers)

                // Step 1: Decompose to (key, value) pairs
                let decompose_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let decompose_kernel = program.create_kernel(
                    &format!("{}_decompose_to_pairs", G::name()),
                    decompose_global, MSM_LOCAL_WORK_SIZE)?;
                decompose_kernel
                    .arg(&digits_buffer)
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                // Step 2: Count buckets (re-init counts to zero)
                program.write_from_buffer(&mut counts_buffer, &vec![0u32; total_buckets])?;
                let count_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let count_kernel = program.create_kernel(
                    &format!("{}_count_buckets", G::name()),
                    count_global, MSM_LOCAL_WORK_SIZE)?;
                count_kernel
                    .arg(&keys_buffer)
                    .arg(&counts_buffer)
                    .arg(&(total_pairs as u32))
                    .run()?;

                // Step 3: Prefix sum
                let prefix_kernel = program.create_kernel(
                    &format!("{}_prefix_sum", G::name()), 1, 1)?;
                prefix_kernel
                    .arg(&counts_buffer)
                    .arg(&offsets_buffer)
                    .arg(&nonempty_ids_buffer)
                    .arg(&num_nonempty_buffer)
                    .arg(&(total_buckets as u32))
                    .run()?;

                // Step 4: Download num_nonempty
                let mut num_nonempty_vec = vec![0u32; 1];
                program.read_into_buffer(&num_nonempty_buffer, &mut num_nonempty_vec)?;
                let num_nonempty = num_nonempty_vec[0] as usize;

                // Step 5: Copy offsets for scatter (scatter modifies them via atomicAdd)
                program.read_into_buffer(&offsets_buffer, &mut offsets_copy)?;
                program.write_from_buffer(&mut scatter_offsets_buffer, &offsets_copy)?;

                // Step 6: Scatter to sorted
                let scatter_global = div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE);
                let scatter_kernel = program.create_kernel(
                    &format!("{}_scatter_to_sorted", G::name()),
                    scatter_global, MSM_LOCAL_WORK_SIZE)?;
                scatter_kernel
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&scatter_offsets_buffer)
                    .arg(&sorted_values_buffer)
                    .arg(&(total_pairs as u32))
                    .run()?;

                // Step 7: Accumulate sorted buckets (with chunked dispatch for large buckets)
                program.write_from_buffer(&mut bucket_results_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; total_buckets])?;
                if num_nonempty > 0 {
                    program.read_into_buffer(&counts_buffer, &mut counts_cpu)?;
                    program.read_into_buffer(&nonempty_ids_buffer, &mut nonempty_ids_cpu)?;

                    let (dispatch_table, reduce_table, num_dispatches) =
                        crate::multiexp::build_dispatch_tables(&counts_cpu, &nonempty_ids_cpu, num_nonempty);

                    if num_dispatches == num_nonempty {
                        // No large buckets — use simple 1-thread-per-bucket kernel
                        let accum_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let accum_kernel = program.create_kernel(
                            &format!("{}_accumulate_sorted_buckets", G::name()),
                            accum_global, MSM_LOCAL_WORK_SIZE)?;
                        accum_kernel
                            .arg(&base_buffer)
                            .arg(&sorted_values_buffer)
                            .arg(&offsets_buffer)
                            .arg(&counts_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    } else {
                        // Large buckets detected — use chunked accumulation
                        let dispatch_buffer = program.create_buffer_from_slice(&dispatch_table)?;
                        let partial_results_buffer = {
                            let partials = vec![<G::Group as AdditiveGroup>::ZERO; num_dispatches];
                            program.create_buffer_from_slice(&partials)?
                        };

                        let chunked_global = div_ceil(num_dispatches, MSM_LOCAL_WORK_SIZE);
                        let chunked_kernel = program.create_kernel(
                            &format!("{}_accumulate_chunked", G::name()),
                            chunked_global, MSM_LOCAL_WORK_SIZE)?;
                        chunked_kernel
                            .arg(&base_buffer)
                            .arg(&sorted_values_buffer)
                            .arg(&offsets_buffer)
                            .arg(&dispatch_buffer)
                            .arg(&partial_results_buffer)
                            .arg(&(num_dispatches as u32))
                            .run()?;

                        let reduce_table_buffer = program.create_buffer_from_slice(&reduce_table)?;
                        let reduce_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let reduce_kernel = program.create_kernel(
                            &format!("{}_reduce_partial_buckets", G::name()),
                            reduce_global, MSM_LOCAL_WORK_SIZE)?;
                        reduce_kernel
                            .arg(&partial_results_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&reduce_table_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    }
                }

                // Step 8: Reduce buckets by window
                program.write_from_buffer(&mut window_results_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; num_windows])?;
                let reduce_buckets_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                let reduce_buckets_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_by_window", G::name()),
                    reduce_buckets_global, MSM_LOCAL_WORK_SIZE)?;
                reduce_buckets_kernel
                    .arg(&bucket_results_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                // Step 9: Horner reduction on GPU
                program.write_from_buffer(&mut final_result_buffer, &vec![<G::Group as AdditiveGroup>::ZERO; 1])?;
                let reduce_windows_kernel = program.create_kernel(
                    &format!("{}_reduce_windows", G::name()), 1, 1)?;
                reduce_windows_kernel
                    .arg(&window_results_buffer)
                    .arg(&final_result_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .arg(&(effective_bits as u32))
                    .run()?;

                // Step 10: Download final result (just 1 point!)
                let mut final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.read_into_buffer(&final_result_buffer, &mut final_result)?;
                commitments.push(final_result[0]);
            }

            Ok(commitments)
        });

        self.program.run(closures, ())
    }

    /// Fused HyperKZG open: runs the entire open operation in one GPU session.
    ///
    /// This keeps bases on GPU for both Phase 2 (intermediate commits) and Phase 3
    /// (witness commits), and keeps intermediate polynomial data on GPU where possible.
    ///
    /// The `middle_fn` callback runs CPU-side transcript work between Phase 2 and Phase 3
    /// while GPU buffers remain alive. It receives the intermediate polynomials and their
    /// commitments, and must return `Phase3Input` with the parameters needed for Phase 3.
    ///
    /// # Memory optimizations
    /// - Shared pre-allocated MSM sort buffers reused across all iterations
    /// - Chunked dispatch for large buckets (CPU dispatch table, pre-allocated GPU buffers)
    /// - Batched intermediate + commitment downloads after Phase 2 loop
    /// - scale_poly for combined_buffer init (no 128MB zero upload)
    /// - Per-iteration Fr buffers kept alive for streaming LC (no re-upload)
    /// - Streaming linear combination via linear_combine_accumulate (no flat polys_buffer)
    /// - Streaming witness computation: one eval point at a time (no bulk witnesses_buffer)
    /// - Sort-based window size optimized for ~1000 pts/bucket
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

        let closures = program_closures!(|program, middle_fn: MiddleFn| -> EcResult<FusedOpenResult<F, AffineG, G::Group>> {
            // ================================================================
            // Upload bases ONCE — reused for Phase 2 and Phase 3 MSMs
            // ================================================================
            let base_buffer = program.create_buffer_from_slice(bases)?;

            // Upload polynomial
            let poly_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            let mut intermediates = Vec::with_capacity(num_challenges);
            let mut commitments = Vec::with_capacity(num_challenges);
            let mut intermediate_lens = Vec::with_capacity(num_challenges);

            // ================================================================
            // Pre-allocate shared MSM sort buffers for Phase 2 and Phase 3
            // ================================================================
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;

            let mut max_total_pairs: usize = 0;
            let mut max_total_buckets: usize = 0;
            let mut max_num_windows: usize = 0;

            // Scan Phase 2 iterations to find max buffer sizes
            let mut scan_len = initial_len;
            for _ in 0..num_challenges {
                scan_len /= 2;
                let ws = calc_sort_window_size(scan_len);
                let nw = div_ceil(effective_bits + 1, ws);
                let bpw = 1 << (ws - 1);
                let tp = scan_len * nw;
                let tb = nw * bpw;
                max_total_pairs = max_total_pairs.max(tp);
                max_total_buckets = max_total_buckets.max(tb);
                max_num_windows = max_num_windows.max(nw);
            }

            // Phase 3: witness MSMs use n_bases = initial_len - 1
            let witness_len = initial_len - 1;
            let witness_window_size = calc_sort_window_size(witness_len);
            let num_windows_p3 = div_ceil(effective_bits + 1, witness_window_size);
            let buckets_per_window_p3 = 1 << (witness_window_size - 1);
            let total_pairs_p3 = witness_len * num_windows_p3;
            let total_buckets_p3 = num_windows_p3 * buckets_per_window_p3;

            max_total_pairs = max_total_pairs.max(total_pairs_p3);
            max_total_buckets = max_total_buckets.max(total_buckets_p3);
            max_num_windows = max_num_windows.max(num_windows_p3);

            // Pre-allocate shared MSM buffers at max sizes
            // SAFETY: GPU will initialize these buffers before use via GPU kernels
            let shared_digits_buffer = unsafe { program.create_buffer::<u16>(max_total_pairs)? };
            let shared_keys_buffer = unsafe { program.create_buffer::<u32>(max_total_pairs)? };
            let shared_values_buffer = unsafe { program.create_buffer::<u32>(max_total_pairs)? };
            let shared_sorted_values_buffer = unsafe { program.create_buffer::<u32>(max_total_pairs)? };
            let shared_counts_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let shared_offsets_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let shared_scatter_offsets_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            // nonempty_ids and num_nonempty still allocated (prefix_sum kernel writes to them)
            let shared_nonempty_ids_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets)? };
            let shared_num_nonempty_buffer = unsafe { program.create_buffer::<u32>(1)? };
            let shared_bucket_results_buffer = unsafe { program.create_buffer::<G::Group>(max_total_buckets)? };
            let shared_window_results_buffer = unsafe { program.create_buffer::<G::Group>(max_num_windows)? };
            let shared_final_result_buffer = unsafe { program.create_buffer::<G::Group>(1)? };

            // Pre-allocate dispatch-related buffers at max sizes.
            // Upper bound for num_dispatches: each non-empty bucket produces at least 1 dispatch,
            // large buckets produce ceil(count/CHUNK_SIZE) dispatches.
            // Worst case: max_total_pairs / CHUNK_SIZE + max_total_buckets.
            let max_num_dispatches = max_total_pairs / crate::multiexp::CHUNK_SIZE + max_total_buckets;
            let mut shared_dispatch_table_buffer = unsafe { program.create_buffer::<u32>(max_num_dispatches * 3)? };
            let mut shared_reduce_table_buffer = unsafe { program.create_buffer::<u32>(max_total_buckets * 2)? };
            let shared_partial_results_buffer = unsafe { program.create_buffer::<G::Group>(max_num_dispatches)? };

            // CPU-side scratch vectors for dispatch table construction
            let mut counts_cpu = vec![0u32; max_total_buckets];
            let mut nonempty_ids_cpu = vec![0u32; max_total_buckets];
            // Padded CPU vecs for write_from_buffer (must match GPU buffer size)
            let mut dispatch_table_padded = vec![0u32; max_num_dispatches * 3];
            let mut reduce_table_padded = vec![0u32; max_total_buckets * 2];

            // GPU kernel names
            let u32_fill_zero_name = "u32_fill_zero".to_string();
            let u32_copy_buffer_name = "u32_copy_buffer".to_string();
            let fill_identity_name = format!("{}_fill_identity", G::name());
            let copy_at_offset_name = format!("{}_copy_at_offset", G::name());

            // ================================================================
            // Phase 1+2: fix_vars + intermediate MSM commits
            // ================================================================
            let max_intermediate_len = if num_challenges > 0 { initial_len / 2 } else { 1 };
            let shared_scalar_buffer = unsafe { program.create_buffer::<F>(max_intermediate_len)? };

            // Pre-allocate per-iteration Fr buffers (sizes n/2, n/4, ..., 2)
            // These stay alive for Phase 3 streaming LC.
            let mut per_iter_buffers = Vec::with_capacity(num_challenges);
            {
                let mut iter_len = initial_len;
                for _ in 0..num_challenges {
                    iter_len /= 2;
                    let buf = unsafe { program.create_buffer::<F>(iter_len)? };
                    per_iter_buffers.push(buf);
                }
            }

            // GPU buffer to accumulate Phase 2 commitments (batch download after loop)
            let commitments_gpu = unsafe { program.create_buffer::<G::Group>(num_challenges.max(1))? };

            if num_challenges > 0 {
                let challenges_buffer = program.create_buffer_from_slice(&challenges_vec)?;
                let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());

                for challenge_idx in 0..num_challenges {
                    let next_len = current_len / 2;

                    // === Phase 1: fix_var ===
                    let fix_var_global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                    let fix_var_kernel = program.create_kernel(
                        &format!("{}_fix_var_indexed", F::name()),
                        fix_var_global_work_size, LOCAL_WORK_SIZE)?;

                    let input_buf = if challenge_idx == 0 {
                        &poly_buffer
                    } else {
                        &per_iter_buffers[challenge_idx - 1]
                    };

                    fix_var_kernel
                        .arg(input_buf)
                        .arg(&per_iter_buffers[challenge_idx])
                        .arg(&challenges_buffer)
                        .arg(&(next_len as u32))
                        .arg(&(challenge_idx as u32))
                        .run()?;

                    // No intermediate download here — batched after the loop
                    intermediate_lens.push(next_len);

                    // === Phase 2: Convert Fr → scalar bytes on GPU ===
                    let to_scalar_kernel = program.create_kernel(
                        &format!("{}_to_scalar_bytes", F::name()),
                        fix_var_global_work_size, LOCAL_WORK_SIZE)?;
                    to_scalar_kernel
                        .arg(&per_iter_buffers[challenge_idx])
                        .arg(&shared_scalar_buffer)
                        .arg(&(next_len as u32))
                        .run()?;

                    // === Sort-based MSM pipeline (zero CPU-GPU sync points) ===
                    let window_size_for_len = calc_sort_window_size(next_len);
                    let num_windows = div_ceil(effective_bits + 1, window_size_for_len);
                    let n_bases = next_len;
                    let total_pairs = n_bases * num_windows;
                    let buckets_per_window = 1 << (window_size_for_len - 1);
                    let total_buckets = num_windows * buckets_per_window;

                    // Preprocess to signed digits
                    let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                    let preprocess_kernel = program.create_kernel(&preprocess_kernel_name, preprocess_global, LOCAL_WORK_SIZE)?;
                    preprocess_kernel
                        .arg(&shared_scalar_buffer)
                        .arg(&shared_digits_buffer)
                        .arg(&(n_bases as u32))
                        .arg(&(num_windows as u32))
                        .arg(&(window_size_for_len as u32))
                        .run()?;

                    // 1. Decompose to (key, value) pairs
                    let decompose_kernel = program.create_kernel(
                        &format!("{}_decompose_to_pairs", G::name()),
                        div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                    decompose_kernel
                        .arg(&shared_digits_buffer)
                        .arg(&shared_keys_buffer)
                        .arg(&shared_values_buffer)
                        .arg(&(n_bases as u32))
                        .arg(&(num_windows as u32))
                        .arg(&(buckets_per_window as u32))
                        .run()?;

                    // 2. Count buckets (GPU zero-init)
                    let counts_fill_kernel = program.create_kernel(&u32_fill_zero_name,
                        div_ceil(total_buckets, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
                    counts_fill_kernel.arg(&shared_counts_buffer).arg(&(total_buckets as u32)).run()?;

                    let count_kernel = program.create_kernel(
                        &format!("{}_count_buckets", G::name()),
                        div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                    count_kernel
                        .arg(&shared_keys_buffer)
                        .arg(&shared_counts_buffer)
                        .arg(&(total_pairs as u32))
                        .run()?;

                    // 3. Prefix sum (nonempty_ids/num_nonempty written but not downloaded)
                    let prefix_kernel = program.create_kernel(
                        &format!("{}_prefix_sum", G::name()), 1, 1)?;
                    prefix_kernel
                        .arg(&shared_counts_buffer)
                        .arg(&shared_offsets_buffer)
                        .arg(&shared_nonempty_ids_buffer)
                        .arg(&shared_num_nonempty_buffer)
                        .arg(&(total_buckets as u32))
                        .run()?;

                    // 4. Copy offsets for scatter (GPU device-to-device)
                    let copy_offsets_kernel = program.create_kernel(&u32_copy_buffer_name,
                        div_ceil(total_buckets, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
                    copy_offsets_kernel
                        .arg(&shared_offsets_buffer)
                        .arg(&shared_scatter_offsets_buffer)
                        .arg(&(total_buckets as u32))
                        .run()?;

                    // 5. Scatter to sorted
                    let scatter_kernel = program.create_kernel(
                        &format!("{}_scatter_to_sorted", G::name()),
                        div_ceil(total_pairs, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                    scatter_kernel
                        .arg(&shared_keys_buffer)
                        .arg(&shared_values_buffer)
                        .arg(&shared_scatter_offsets_buffer)
                        .arg(&shared_sorted_values_buffer)
                        .arg(&(total_pairs as u32))
                        .run()?;

                    // 6. Initialize bucket_results to identity, then accumulate
                    let fill_bucket_kernel = program.create_kernel(&fill_identity_name,
                        div_ceil(total_buckets, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
                    fill_bucket_kernel.arg(&shared_bucket_results_buffer).arg(&(total_buckets as u32)).run()?;

                    // Download counts and nonempty IDs for dispatch table construction
                    let mut num_nonempty_vec = vec![0u32; 1];
                    program.read_into_buffer(&shared_num_nonempty_buffer, &mut num_nonempty_vec)?;
                    let num_nonempty = num_nonempty_vec[0] as usize;

                    if num_nonempty > 0 {
                        program.read_into_buffer(&shared_counts_buffer, &mut counts_cpu)?;
                        program.read_into_buffer(&shared_nonempty_ids_buffer, &mut nonempty_ids_cpu)?;

                        let (dispatch_table, reduce_table, num_dispatches) =
                            crate::multiexp::build_dispatch_tables(&counts_cpu[..total_buckets], &nonempty_ids_cpu[..total_buckets], num_nonempty);

                        if num_dispatches == num_nonempty {
                            // No large buckets — use simple 1-thread-per-bucket kernel
                            let accum_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                            let accum_kernel = program.create_kernel(
                                &format!("{}_accumulate_sorted_buckets", G::name()),
                                accum_global, MSM_LOCAL_WORK_SIZE)?;
                            accum_kernel
                                .arg(&base_buffer)
                                .arg(&shared_sorted_values_buffer)
                                .arg(&shared_offsets_buffer)
                                .arg(&shared_counts_buffer)
                                .arg(&shared_nonempty_ids_buffer)
                                .arg(&shared_bucket_results_buffer)
                                .arg(&(num_nonempty as u32))
                                .run()?;
                        } else {
                            // Large buckets — use chunked accumulation with pre-allocated buffers
                            dispatch_table_padded[..dispatch_table.len()].copy_from_slice(&dispatch_table);
                            program.write_from_buffer(&mut shared_dispatch_table_buffer, &dispatch_table_padded)?;

                            reduce_table_padded[..reduce_table.len()].copy_from_slice(&reduce_table);
                            program.write_from_buffer(&mut shared_reduce_table_buffer, &reduce_table_padded)?;

                            let chunked_global = div_ceil(num_dispatches, MSM_LOCAL_WORK_SIZE);
                            let chunked_kernel = program.create_kernel(
                                &format!("{}_accumulate_chunked", G::name()),
                                chunked_global, MSM_LOCAL_WORK_SIZE)?;
                            chunked_kernel
                                .arg(&base_buffer)
                                .arg(&shared_sorted_values_buffer)
                                .arg(&shared_offsets_buffer)
                                .arg(&shared_dispatch_table_buffer)
                                .arg(&shared_partial_results_buffer)
                                .arg(&(num_dispatches as u32))
                                .run()?;

                            let reduce_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                            let reduce_kernel = program.create_kernel(
                                &format!("{}_reduce_partial_buckets", G::name()),
                                reduce_global, MSM_LOCAL_WORK_SIZE)?;
                            reduce_kernel
                                .arg(&shared_partial_results_buffer)
                                .arg(&shared_nonempty_ids_buffer)
                                .arg(&shared_reduce_table_buffer)
                                .arg(&shared_bucket_results_buffer)
                                .arg(&(num_nonempty as u32))
                                .run()?;
                        }
                    }

                    // 7. Reduce buckets by window
                    let reduce_buckets_kernel = program.create_kernel(
                        &format!("{}_reduce_buckets_by_window", G::name()),
                        div_ceil(num_windows, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                    reduce_buckets_kernel
                        .arg(&shared_bucket_results_buffer)
                        .arg(&shared_window_results_buffer)
                        .arg(&(num_windows as u32))
                        .arg(&(buckets_per_window as u32))
                        .run()?;

                    // 8. Horner reduction
                    let reduce_windows_kernel = program.create_kernel(
                        &format!("{}_reduce_windows", G::name()), 1, 1)?;
                    reduce_windows_kernel
                        .arg(&shared_window_results_buffer)
                        .arg(&shared_final_result_buffer)
                        .arg(&(num_windows as u32))
                        .arg(&(window_size_for_len as u32))
                        .arg(&(effective_bits as u32))
                        .run()?;

                    // 9. Save commitment to GPU array (no per-iteration download)
                    let copy_kernel = program.create_kernel(&copy_at_offset_name, 1, 1)?;
                    copy_kernel
                        .arg(&shared_final_result_buffer)
                        .arg(&commitments_gpu)
                        .arg(&(challenge_idx as u32))
                        .run()?;

                    current_len = next_len;
                }

                // Batch download all intermediates (per_iter_buffers still alive on GPU)
                for challenge_idx in 0..num_challenges {
                    let next_len = intermediate_lens[challenge_idx];
                    let mut intermediate = vec![F::ZERO; next_len];
                    program.read_into_buffer(&per_iter_buffers[challenge_idx], &mut intermediate)?;
                    intermediates.push(intermediate);
                }

                // Batch download all commitments (1 sync instead of 22)
                commitments.resize(num_challenges, <G::Group as AdditiveGroup>::ZERO);
                program.read_into_buffer(&commitments_gpu, &mut commitments)?;
            } // end if num_challenges > 0

            // ================================================================
            // CPU callback: transcript work between Phase 2 and Phase 3
            // GPU buffers (base_buffer, poly_buffer, per_iter_buffers) remain alive!
            // ================================================================
            let phase3_input = middle_fn(&intermediates, &commitments);

            // ================================================================
            // Phase 3: Streaming LC using on-GPU per-iteration buffers
            // ================================================================
            let num_polys = 1 + num_challenges;
            let poly_len = initial_len;
            let num_points = phase3_input.eval_points.len();

            // Allocate combined_buffer (uninitialized — scale_poly writes without reading)
            let combined_buffer = unsafe { program.create_buffer::<F>(poly_len)? };

            let coeffs_vec = phase3_input.lc_coeffs;
            assert!(
                coeffs_vec.len() >= num_polys,
                "lc_coeffs must have at least {} coefficients (got {})",
                num_polys,
                coeffs_vec.len()
            );
            let lc_accum_kernel_name = format!("{}_linear_combine_accumulate", F::name());
            let lc_global_work_size = div_ceil(poly_len, LOCAL_WORK_SIZE);

            // First polynomial: use scale_poly (writes combined = coeff * poly, no read of combined)
            let coeff0_buffer = program.create_buffer_from_slice(&[coeffs_vec[0]])?;
            let scale_poly_kernel = program.create_kernel(
                &format!("{}_scale_poly", F::name()),
                lc_global_work_size, LOCAL_WORK_SIZE)?;
            scale_poly_kernel
                .arg(&poly_buffer)
                .arg(&combined_buffer)
                .arg(&coeff0_buffer)
                .arg(&(poly_len as u32))
                .run()?;

            // Accumulate each intermediate directly from on-GPU per-iteration buffers
            for (idx, iter_buf) in per_iter_buffers.iter().enumerate() {
                let src_len = intermediate_lens[idx];
                let coeff_buffer = program.create_buffer_from_slice(&[coeffs_vec[idx + 1]])?;

                let lc_accum_kernel = program.create_kernel(&lc_accum_kernel_name, lc_global_work_size, LOCAL_WORK_SIZE)?;
                lc_accum_kernel
                    .arg(iter_buf)
                    .arg(&combined_buffer)
                    .arg(&coeff_buffer)
                    .arg(&(src_len as u32))
                    .arg(&(poly_len as u32))
                    .run()?;
            }

            // Per-iteration buffers no longer needed — free GPU memory
            drop(per_iter_buffers);

            // ================================================================
            // Phase 3: Streaming witness computation + MSMs
            // ================================================================
            // Single witness buffer (reusable across all eval points)
            let single_witness_buffer = unsafe { program.create_buffer::<F>(witness_len)? };

            // Witness computation uses 3-phase parallel approach
            let chunk_size = std::cmp::max(1, witness_len / 4096);
            let num_chunks = div_ceil(witness_len, chunk_size);
            let carries_buffer = unsafe { program.create_buffer::<F>(num_chunks)? };
            let propagated_carries_buffer = unsafe { program.create_buffer::<F>(num_chunks)? };

            // Scalar buffer for witness MSM
            let witness_scalar_buffer = unsafe { program.create_buffer::<F>(witness_len)? };

            // Signed digits buffer for witness MSMs
            let digits_len_p3 = witness_len * num_windows_p3;
            let digits_buffer_p3 = unsafe { program.create_buffer::<u16>(digits_len_p3)? };

            let phase1_kernel_name = format!("{}_witness_poly_batch_phase1", F::name());
            let phase2_carry_kernel_name = format!("{}_witness_carry_propagate", F::name());
            let phase3_apply_kernel_name = format!("{}_witness_poly_batch_phase3", F::name());
            let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());

            // GPU buffer to accumulate witness commitments (batch download after loop)
            let witness_commitments_gpu = unsafe { program.create_buffer::<G::Group>(num_points.max(1))? };

            for point_idx in 0..num_points {
                let single_point_buffer = program.create_buffer_from_slice(&[phase3_input.eval_points[point_idx]])?;

                // === Witness computation Phase 1: parallel chunk processing ===
                let phase1_global_work_size = div_ceil(num_chunks, LOCAL_WORK_SIZE);
                let phase1_kernel = program.create_kernel(&phase1_kernel_name, phase1_global_work_size, LOCAL_WORK_SIZE)?;
                phase1_kernel
                    .arg(&combined_buffer)
                    .arg(&single_witness_buffer)
                    .arg(&carries_buffer)
                    .arg(&single_point_buffer)
                    .arg(&(poly_len as u32))
                    .arg(&(1u32))
                    .arg(&(chunk_size as u32))
                    .arg(&(num_chunks as u32))
                    .run()?;

                // === Witness computation Phase 2: carry propagation ===
                let phase2_kernel = program.create_kernel(&phase2_carry_kernel_name, 1, LOCAL_WORK_SIZE)?;
                phase2_kernel
                    .arg(&carries_buffer)
                    .arg(&propagated_carries_buffer)
                    .arg(&single_point_buffer)
                    .arg(&(num_chunks as u32))
                    .arg(&(1u32))
                    .arg(&(chunk_size as u32))
                    .arg(&(poly_len as u32))
                    .run()?;

                // === Witness computation Phase 3: apply carry corrections ===
                let phase3_kernel = program.create_kernel(&phase3_apply_kernel_name, phase1_global_work_size, LOCAL_WORK_SIZE)?;
                phase3_kernel
                    .arg(&single_witness_buffer)
                    .arg(&propagated_carries_buffer)
                    .arg(&single_point_buffer)
                    .arg(&(poly_len as u32))
                    .arg(&(1u32))
                    .arg(&(chunk_size as u32))
                    .arg(&(num_chunks as u32))
                    .run()?;

                // === Convert witness Fr → scalar bytes on GPU ===
                let to_scalar_global_work_size = div_ceil(witness_len, LOCAL_WORK_SIZE);
                let to_scalar_kernel = program.create_kernel(
                    &format!("{}_to_scalar_bytes", F::name()),
                    to_scalar_global_work_size, LOCAL_WORK_SIZE)?;
                to_scalar_kernel
                    .arg(&single_witness_buffer)
                    .arg(&witness_scalar_buffer)
                    .arg(&(witness_len as u32))
                    .run()?;

                // Preprocess to signed digits
                let preprocess_global = div_ceil(witness_len, LOCAL_WORK_SIZE);
                let preprocess_kernel = program.create_kernel(&preprocess_kernel_name, preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&witness_scalar_buffer)
                    .arg(&digits_buffer_p3)
                    .arg(&(witness_len as u32))
                    .arg(&(num_windows_p3 as u32))
                    .arg(&(witness_window_size as u32))
                    .run()?;

                // Sort-based MSM pipeline (zero CPU-GPU sync points)
                let n_bases = witness_len;

                // 1. Decompose to (key, value) pairs
                let decompose_kernel = program.create_kernel(
                    &format!("{}_decompose_to_pairs", G::name()),
                    div_ceil(total_pairs_p3, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                decompose_kernel
                    .arg(&digits_buffer_p3)
                    .arg(&shared_keys_buffer)
                    .arg(&shared_values_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows_p3 as u32))
                    .arg(&(buckets_per_window_p3 as u32))
                    .run()?;

                // 2. Count buckets (GPU zero-init)
                let counts_fill_kernel = program.create_kernel(&u32_fill_zero_name,
                    div_ceil(total_buckets_p3, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
                counts_fill_kernel.arg(&shared_counts_buffer).arg(&(total_buckets_p3 as u32)).run()?;

                let count_kernel = program.create_kernel(
                    &format!("{}_count_buckets", G::name()),
                    div_ceil(total_pairs_p3, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                count_kernel
                    .arg(&shared_keys_buffer)
                    .arg(&shared_counts_buffer)
                    .arg(&(total_pairs_p3 as u32))
                    .run()?;

                // 3. Prefix sum (nonempty_ids/num_nonempty written but not downloaded)
                let prefix_kernel = program.create_kernel(
                    &format!("{}_prefix_sum", G::name()), 1, 1)?;
                prefix_kernel
                    .arg(&shared_counts_buffer)
                    .arg(&shared_offsets_buffer)
                    .arg(&shared_nonempty_ids_buffer)
                    .arg(&shared_num_nonempty_buffer)
                    .arg(&(total_buckets_p3 as u32))
                    .run()?;

                // 4. Copy offsets for scatter (GPU device-to-device)
                let copy_offsets_kernel = program.create_kernel(&u32_copy_buffer_name,
                    div_ceil(total_buckets_p3, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
                copy_offsets_kernel
                    .arg(&shared_offsets_buffer)
                    .arg(&shared_scatter_offsets_buffer)
                    .arg(&(total_buckets_p3 as u32))
                    .run()?;

                // 5. Scatter to sorted
                let scatter_kernel = program.create_kernel(
                    &format!("{}_scatter_to_sorted", G::name()),
                    div_ceil(total_pairs_p3, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                scatter_kernel
                    .arg(&shared_keys_buffer)
                    .arg(&shared_values_buffer)
                    .arg(&shared_scatter_offsets_buffer)
                    .arg(&shared_sorted_values_buffer)
                    .arg(&(total_pairs_p3 as u32))
                    .run()?;

                // 6. Initialize bucket_results to identity, then accumulate
                let fill_bucket_kernel = program.create_kernel(&fill_identity_name,
                    div_ceil(total_buckets_p3, LOCAL_WORK_SIZE), LOCAL_WORK_SIZE)?;
                fill_bucket_kernel.arg(&shared_bucket_results_buffer).arg(&(total_buckets_p3 as u32)).run()?;

                // Download counts and nonempty IDs for dispatch table construction
                let mut num_nonempty_vec = vec![0u32; 1];
                program.read_into_buffer(&shared_num_nonempty_buffer, &mut num_nonempty_vec)?;
                let num_nonempty = num_nonempty_vec[0] as usize;

                if num_nonempty > 0 {
                    program.read_into_buffer(&shared_counts_buffer, &mut counts_cpu)?;
                    program.read_into_buffer(&shared_nonempty_ids_buffer, &mut nonempty_ids_cpu)?;

                    let (dispatch_table, reduce_table, num_dispatches) =
                        crate::multiexp::build_dispatch_tables(&counts_cpu[..total_buckets_p3], &nonempty_ids_cpu[..total_buckets_p3], num_nonempty);

                    if num_dispatches == num_nonempty {
                        // No large buckets — use simple 1-thread-per-bucket kernel
                        let accum_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let accum_kernel = program.create_kernel(
                            &format!("{}_accumulate_sorted_buckets", G::name()),
                            accum_global, MSM_LOCAL_WORK_SIZE)?;
                        accum_kernel
                            .arg(&base_buffer)
                            .arg(&shared_sorted_values_buffer)
                            .arg(&shared_offsets_buffer)
                            .arg(&shared_counts_buffer)
                            .arg(&shared_nonempty_ids_buffer)
                            .arg(&shared_bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    } else {
                        // Large buckets — use chunked accumulation with pre-allocated buffers
                        dispatch_table_padded[..dispatch_table.len()].copy_from_slice(&dispatch_table);
                        program.write_from_buffer(&mut shared_dispatch_table_buffer, &dispatch_table_padded)?;

                        reduce_table_padded[..reduce_table.len()].copy_from_slice(&reduce_table);
                        program.write_from_buffer(&mut shared_reduce_table_buffer, &reduce_table_padded)?;

                        let chunked_global = div_ceil(num_dispatches, MSM_LOCAL_WORK_SIZE);
                        let chunked_kernel = program.create_kernel(
                            &format!("{}_accumulate_chunked", G::name()),
                            chunked_global, MSM_LOCAL_WORK_SIZE)?;
                        chunked_kernel
                            .arg(&base_buffer)
                            .arg(&shared_sorted_values_buffer)
                            .arg(&shared_offsets_buffer)
                            .arg(&shared_dispatch_table_buffer)
                            .arg(&shared_partial_results_buffer)
                            .arg(&(num_dispatches as u32))
                            .run()?;

                        let reduce_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let reduce_kernel = program.create_kernel(
                            &format!("{}_reduce_partial_buckets", G::name()),
                            reduce_global, MSM_LOCAL_WORK_SIZE)?;
                        reduce_kernel
                            .arg(&shared_partial_results_buffer)
                            .arg(&shared_nonempty_ids_buffer)
                            .arg(&shared_reduce_table_buffer)
                            .arg(&shared_bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    }
                }

                // 7. Reduce buckets by window
                let reduce_buckets_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_by_window", G::name()),
                    div_ceil(num_windows_p3, MSM_LOCAL_WORK_SIZE), MSM_LOCAL_WORK_SIZE)?;
                reduce_buckets_kernel
                    .arg(&shared_bucket_results_buffer)
                    .arg(&shared_window_results_buffer)
                    .arg(&(num_windows_p3 as u32))
                    .arg(&(buckets_per_window_p3 as u32))
                    .run()?;

                // 8. Horner reduction
                let reduce_windows_kernel = program.create_kernel(
                    &format!("{}_reduce_windows", G::name()), 1, 1)?;
                reduce_windows_kernel
                    .arg(&shared_window_results_buffer)
                    .arg(&shared_final_result_buffer)
                    .arg(&(num_windows_p3 as u32))
                    .arg(&(witness_window_size as u32))
                    .arg(&(effective_bits as u32))
                    .run()?;

                // 9. Save witness commitment to GPU array (no per-iteration download)
                let copy_kernel = program.create_kernel(&copy_at_offset_name, 1, 1)?;
                copy_kernel
                    .arg(&shared_final_result_buffer)
                    .arg(&witness_commitments_gpu)
                    .arg(&(point_idx as u32))
                    .run()?;
            }

            // Batch download all witness commitments (1 sync instead of 3)
            let mut witness_commitments = vec![<G::Group as AdditiveGroup>::ZERO; num_points];
            if num_points > 0 {
                program.read_into_buffer(&witness_commitments_gpu, &mut witness_commitments)?;
            }

            Ok(FusedOpenResult {
                intermediates,
                intermediate_commitments_affine: phase3_input.intermediate_commitments_affine,
                witness_commitments,
                evaluations: phase3_input.evaluations,
            })
        });

        self.program.run(closures, middle_fn)
    }

    /// Batch scalar multiplication: computes `scalars[i] * base` for each scalar.
    ///
    /// This is used for GPU-accelerated SRS (trusted setup) generation, where we need
    /// to compute [G, tau*G, tau^2*G, ..., tau^n*G] for the KZG powers.
    ///
    /// Uses a windowed lookup table for efficiency (same algorithm as arkworks batch_mul).
    ///
    /// # Arguments
    /// * `base` - The base point (affine, ec-gpu wrapper type)
    /// * `scalars` - Scalars to multiply (in Montgomery form, will be converted on GPU)
    ///
    /// # Returns
    /// One affine point per scalar (arkworks affine type).
    pub fn batch_scalar_mul(
        &self,
        base: &G,
        scalars: &[F],
    ) -> EcResult<Vec<<G::Group as ark_ec::CurveGroup>::Affine>>
    where
        G: From<<G::Group as ark_ec::CurveGroup>::Affine>
            + std::ops::Deref<Target = <G::Group as ark_ec::CurveGroup>::Affine>,
    {
        use ark_ec::CurveGroup;

        if scalars.is_empty() {
            return Ok(vec![]);
        }

        let n = scalars.len();

        // Compute window size (same formula as arkworks)
        let window = if n < 32 {
            3
        } else {
            // ln_without_floats
            let ln_n = ((n as f64).ln() * 100.0 / 69.0) as usize;
            ln_n
        };
        let window = std::cmp::min(window, 16); // Cap at 16 bits for memory

        const BN254_SCALAR_BITS: usize = 254;
        let scalar_bits = BN254_SCALAR_BITS;
        let num_windows = div_ceil(scalar_bits, window);
        let in_window = 1usize << window;
        let table_size = num_windows * in_window;

        // Build precomputation table on CPU (same algorithm as arkworks BatchMulPreprocessing)
        // table[outer][inner] = inner * (2^(outer*window) * base) = inner * g_outer
        let mut table: Vec<G::GpuRepr> = vec![G::GpuRepr::default(); table_size];

        // Convert base (ec-gpu affine wrapper) to projective for arithmetic
        // Deref gives us the inner arkworks affine, then convert to projective
        let base_affine: &<G::Group as CurveGroup>::Affine = &*base;
        let base_proj: G::Group = (*base_affine).into();

        // g_outer starts as base, then gets doubled `window` times per outer loop
        let mut g_outer = base_proj;
        for outer in 0..num_windows {
            let last_in_window = if outer == num_windows - 1 {
                1 << (scalar_bits - (num_windows - 1) * window)
            } else {
                in_window
            };

            // table[outer][inner] = inner * g_outer
            let mut g_inner = <G::Group as AdditiveGroup>::ZERO;
            for inner in 0..std::cmp::min(in_window, last_in_window) {
                // Convert projective to affine then to GpuRepr
                let affine: <G::Group as CurveGroup>::Affine = g_inner.into_affine();
                table[outer * in_window + inner] = G::from(affine).to_gpu();
                g_inner += g_outer;
            }

            // Advance g_outer: g_outer *= 2^window (i.e., window doublings)
            for _ in 0..window {
                g_outer = g_outer.double();
            }
        }

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<<G::Group as CurveGroup>::Affine>> {
            // Upload precomputation table
            let table_buffer = program.create_buffer_from_slice(&table)?;

            // Upload scalars (in Montgomery form)
            let fr_buffer = program.create_buffer_from_slice(scalars)?;

            // Convert Montgomery → standard form on GPU
            // SAFETY: GPU will initialize this buffer
            let scalars_buffer = unsafe { program.create_buffer::<F>(n)? };
            let to_scalar_kernel_name = format!("{}_to_scalar_bytes", F::name());
            let to_scalar_global = div_ceil(n, LOCAL_WORK_SIZE);
            let to_scalar_kernel = program.create_kernel(&to_scalar_kernel_name, to_scalar_global, LOCAL_WORK_SIZE)?;
            to_scalar_kernel
                .arg(&fr_buffer)
                .arg(&scalars_buffer)
                .arg(&(n as u32))
                .run()?;

            // Allocate output buffer
            // SAFETY: GPU will initialize this buffer
            let results_buffer = unsafe { program.create_buffer::<G::Group>(n)? };

            // Run batch scalar multiplication kernel
            let kernel_name = format!("{}_batch_scalar_mul", G::name());
            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&table_buffer)
                .arg(&scalars_buffer)
                .arg(&results_buffer)
                .arg(&(n as u32))
                .arg(&(window as u32))
                .arg(&(num_windows as u32))
                .arg(&(scalar_bits as u32))
                .run()?;

            // Download results
            let mut results_proj = vec![<G::Group as AdditiveGroup>::ZERO; n];
            program.read_into_buffer(&results_buffer, &mut results_proj)?;

            // Convert projective to affine
            let results: Vec<<G::Group as CurveGroup>::Affine> = results_proj
                .iter()
                .map(|p| p.into_affine())
                .collect();

            Ok(results)
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
