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

use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{AdditiveGroup, PrimeField};
use ec_gpu::GpuName;
use rust_gpu_tools::{program_closures, LocalBuffer, Program};

use crate::error::EcResult;
use crate::multiexp::GpuAffine;

/// Below this number of bases, Phase 2 MSM iterations fall back to CPU.
/// CPU is faster for small MSMs because GPU has per-kernel launch overhead.
const GPU_MSM_THRESHOLD: usize = 4096;

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
    /// Maximum window size for MSM (used by non-fused methods)
    max_window_size: usize,
    /// Work units for MSM parallelization (used by non-fused methods)
    work_units: usize,
    /// Persistent GPU buffer for SRS bases, uploaded once and reused across calls.
    base_buffer: Option<rust_gpu_tools::PersistentBuffer<G::GpuRepr>>,
    /// Number of elements in the persistent base buffer.
    base_buffer_len: usize,
    _phantom: std::marker::PhantomData<(F, G)>,
}

impl<F: PrimeField + GpuName, G: GpuAffine> Drop for FusedPolyCommit<F, G> {
    fn drop(&mut self) {
        if self.base_buffer.is_some() {
            // Push CUDA context before dropping the persistent buffer,
            // so that cuMemFree runs with the correct context active.
            if self.program.push_context().is_ok() {
                self.base_buffer = None;
                self.program.pop_context();
            }
            // If push_context fails, the buffer leaks (cuMemFree skipped).
            // This is safer than panicking in Drop.
        }
    }
}

/// On the GPU, the exponents are split into windows
const MAX_WINDOW_SIZE: usize = 16;
/// In CUDA this is the number of blocks per grid (grid size) for MSM
const MSM_LOCAL_WORK_SIZE: usize = 128;
/// Chunk size for parallel bucket reduction (shared memory suffix sum).
/// Each thread block processes this many buckets. Must be a power of 2.
/// 256 threads × 96 bytes (Jacobian point) = 24 KB shared memory per block.
const REDUCTION_CHUNK_SIZE: usize = 256;


impl<F: PrimeField + GpuName, G: GpuAffine<ScalarField = F>> FusedPolyCommit<F, G> {
    /// Create a new fused poly-commit handler.
    pub fn create(program: Program, work_units: usize) -> EcResult<Self> {
        Ok(Self {
            program,
            max_window_size: MAX_WINDOW_SIZE,
            work_units,
            base_buffer: None,
            base_buffer_len: 0,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Upload SRS bases to GPU as a persistent buffer.
    ///
    /// The buffer persists across `program.run()` calls, eliminating the need to
    /// re-upload ~256MB of bases on every `batch_commit` or `fused_open` call.
    /// If a persistent buffer already exists with a different size, it is replaced.
    pub fn upload_bases(&mut self, bases: &[G::GpuRepr]) -> EcResult<()> {
        if self.base_buffer.as_ref().is_some() && self.base_buffer_len == bases.len() {
            return Ok(());
        }
        // Drop old buffer inside a context scope (for CUDA cuMemFree)
        if self.base_buffer.is_some() {
            self.program.push_context()?;
            self.base_buffer = None;
            self.program.pop_context();
        }
        // Create new persistent buffer inside a context scope
        self.program.push_context()?;
        let buf = match self.program.create_persistent_buffer_from_slice(bases) {
            Ok(b) => b,
            Err(e) => {
                self.program.pop_context();
                return Err(e.into());
            }
        };
        self.program.pop_context();
        self.base_buffer = Some(buf);
        self.base_buffer_len = bases.len();
        Ok(())
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
        let effective_base_len = if self.base_buffer.is_some() {
            self.base_buffer_len
        } else {
            bases.len()
        };
        assert!(
            effective_base_len >= poly.len(),
            "Not enough bases for polynomial size (have {}, need {})", effective_base_len, poly.len()
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
        let persistent_base = self.base_buffer.as_ref();
        let persistent_base_len = self.base_buffer_len;

        let closures = program_closures!(|program, _arg| -> EcResult<FixVarsAndCommitResult<F, G::Group>> {
            // Upload polynomial once
            let mut current_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            // Use persistent base buffer if available and large enough, otherwise upload fresh
            let local_base_buf;
            let base_buffer = if let Some(pb) = persistent_base {
                if persistent_base_len >= initial_len {
                    pb
                } else {
                    local_base_buf = program.create_persistent_buffer_from_slice(bases)?;
                    &local_base_buf
                }
            } else {
                local_base_buf = program.create_persistent_buffer_from_slice(bases)?;
                &local_base_buf
            };

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
            let mut max_total_chunks = 0usize;
            {
                let mut len = initial_len;
                for _ in 0..num_challenges {
                    let nb = len / 2;
                    let ws = std::cmp::min(((div_ceil(nb, work_units) as f64).log2() as usize) + 2, max_window_size);
                    let nw = div_ceil(effective_bits + 1, ws);
                    let tp = nb * nw;
                    let bpw = 1usize << (ws - 1);
                    let tb = nw * bpw;
                    let num_chunks_pw = div_ceil(bpw, REDUCTION_CHUNK_SIZE);
                    let total_chunks = nw * num_chunks_pw;
                    max_total_pairs = std::cmp::max(max_total_pairs, tp);
                    max_total_buckets = std::cmp::max(max_total_buckets, tb);
                    max_digits_len = std::cmp::max(max_digits_len, nb * nw);
                    max_num_windows = std::cmp::max(max_num_windows, nw);
                    max_total_chunks = std::cmp::max(max_total_chunks, total_chunks);
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
            let window_results_buffer = {
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; max_num_windows];
                program.create_buffer_from_slice(&window_results)?
            };
            let mut final_result_buffer = {
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.create_buffer_from_slice(&final_result)?
            };
            // Chunk buffers for parallel bucket reduction
            let chunk_sbp_buffer = unsafe { program.create_buffer::<G::Group>(max_total_chunks.max(1))? };
            let chunk_sum_buffer = unsafe { program.create_buffer::<G::Group>(max_total_chunks.max(1))? };
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
                let window_size_for_len = std::cmp::min(((div_ceil(next_len, work_units) as f64).log2() as usize) + 2, max_window_size);
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
                            .arg(base_buffer)
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
                            .arg(base_buffer)
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

                // Step 8: Parallel chunked bucket reduction (Level 1)
                let num_chunks_per_window = div_ceil(buckets_per_window, REDUCTION_CHUNK_SIZE);
                let total_blocks = num_windows * num_chunks_per_window;
                let chunked_reduce_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_chunked", G::name()),
                    total_blocks, REDUCTION_CHUNK_SIZE)?;
                chunked_reduce_kernel
                    .arg(&bucket_results_buffer)
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&(buckets_per_window as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&LocalBuffer::<G::Group>::new(REDUCTION_CHUNK_SIZE))
                    .run()?;

                // Step 8b: Combine chunks to windows (Level 2)
                let combine_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                let combine_kernel = program.create_kernel(
                    &format!("{}_combine_chunks_to_windows", G::name()),
                    combine_global, MSM_LOCAL_WORK_SIZE)?;
                combine_kernel
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&(REDUCTION_CHUNK_SIZE as u32))
                    .run()?;

                // Step 9: Horner reduction on GPU (single thread, re-init final_result to identity)
                program.write_from_buffer(&mut final_result_buffer, &[<G::Group as AdditiveGroup>::ZERO; 1])?;
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

        let effective_base_len = if self.base_buffer.is_some() {
            self.base_buffer_len
        } else {
            bases.len()
        };
        assert!(
            effective_base_len >= witness_len,
            "Not enough bases for witness polynomial size (have {}, need {})", effective_base_len, witness_len
        );

        let poly_vec = poly.to_vec();
        let points_vec = points.to_vec();
        let work_units = self.work_units;
        let max_window_size = self.max_window_size;
        let persistent_base = self.base_buffer.as_ref();
        let persistent_base_len = self.base_buffer_len;

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Group>> {
            // Upload polynomial once
            let poly_buffer = program.create_buffer_from_slice(&poly_vec)?;

            // Upload points
            let points_buffer = program.create_buffer_from_slice(&points_vec)?;

            // Use persistent base buffer if available and large enough, otherwise upload fresh
            let local_base_buf;
            let base_buffer = if let Some(pb) = persistent_base {
                if persistent_base_len >= witness_len {
                    pb
                } else {
                    local_base_buf = program.create_persistent_buffer_from_slice(&bases[..witness_len])?;
                    &local_base_buf
                }
            } else {
                local_base_buf = program.create_persistent_buffer_from_slice(&bases[..witness_len])?;
                &local_base_buf
            };

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
            let window_size = std::cmp::min(((div_ceil(witness_len, work_units) as f64).log2() as usize) + 2, max_window_size);
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
            let window_results_buffer = {
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; num_windows];
                program.create_buffer_from_slice(&window_results)?
            };
            let mut final_result_buffer = {
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.create_buffer_from_slice(&final_result)?
            };
            // Chunk buffers for parallel bucket reduction
            let num_chunks_per_window = div_ceil(buckets_per_window, REDUCTION_CHUNK_SIZE);
            let total_chunks = num_windows * num_chunks_per_window;
            let chunk_sbp_buffer = unsafe { program.create_buffer::<G::Group>(total_chunks.max(1))? };
            let chunk_sum_buffer = unsafe { program.create_buffer::<G::Group>(total_chunks.max(1))? };
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
                            .arg(base_buffer)
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
                            .arg(base_buffer)
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

                // Step 8: Parallel chunked bucket reduction (Level 1)
                let total_blocks = num_windows * num_chunks_per_window;
                let chunked_reduce_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_chunked", G::name()),
                    total_blocks, REDUCTION_CHUNK_SIZE)?;
                chunked_reduce_kernel
                    .arg(&bucket_results_buffer)
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&(buckets_per_window as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&LocalBuffer::<G::Group>::new(REDUCTION_CHUNK_SIZE))
                    .run()?;

                // Step 8b: Combine chunks to windows (Level 2)
                let combine_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                let combine_kernel = program.create_kernel(
                    &format!("{}_combine_chunks_to_windows", G::name()),
                    combine_global, MSM_LOCAL_WORK_SIZE)?;
                combine_kernel
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&(REDUCTION_CHUNK_SIZE as u32))
                    .run()?;

                // Step 9: Horner reduction on GPU (single thread, re-init final_result to identity)
                program.write_from_buffer(&mut final_result_buffer, &[<G::Group as AdditiveGroup>::ZERO; 1])?;
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
        let effective_base_len = if self.base_buffer.is_some() {
            self.base_buffer_len
        } else {
            bases.len()
        };
        assert!(
            effective_base_len >= max_len,
            "Not enough bases for polynomial size (have {}, need {})", effective_base_len, max_len
        );

        let num_polys = polys.len();

        // CPU-side scratch buffer for zero-padded poly data (reused per poly)
        let mut padded_scratch: Vec<F> = vec![F::ZERO; max_len];
        let work_units = self.work_units;
        let max_window_size = self.max_window_size;
        let persistent_base = self.base_buffer.as_ref();
        let persistent_base_len = self.base_buffer_len;

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Group>> {
            // Use persistent base buffer if available and large enough, otherwise upload fresh
            let local_base_buf;
            let base_buffer = if let Some(pb) = persistent_base {
                if persistent_base_len >= max_len {
                    pb
                } else {
                    local_base_buf = program.create_persistent_buffer_from_slice(&bases[..max_len])?;
                    &local_base_buf
                }
            } else {
                local_base_buf = program.create_persistent_buffer_from_slice(&bases[..max_len])?;
                &local_base_buf
            };

            // Reusable GPU buffer for one poly at a time (replaces massive flat buffer)
            let mut fr_buffer = program.create_buffer_from_slice(&padded_scratch)?;

            // Compute MSM params (constant across all polys since using max_len)
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;
            let n_bases = max_len;
            let window_size = std::cmp::min(((div_ceil(n_bases, work_units) as f64).log2() as usize) + 2, max_window_size);
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
            let window_results_buffer = {
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; num_windows];
                program.create_buffer_from_slice(&window_results)?
            };
            let mut final_result_buffer = {
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.create_buffer_from_slice(&final_result)?
            };
            // Chunk buffers for parallel bucket reduction
            let num_chunks_per_window = div_ceil(buckets_per_window, REDUCTION_CHUNK_SIZE);
            let total_chunks = num_windows * num_chunks_per_window;
            let chunk_sbp_buffer = unsafe { program.create_buffer::<G::Group>(total_chunks.max(1))? };
            let chunk_sum_buffer = unsafe { program.create_buffer::<G::Group>(total_chunks.max(1))? };
            // CPU-side scratch vectors
            let mut offsets_copy = vec![0u32; total_buckets];
            let mut counts_cpu = vec![0u32; total_buckets];
            let mut nonempty_ids_cpu = vec![0u32; total_buckets];

            let mut commitments = Vec::with_capacity(num_polys);

            for poly in polys.iter().take(num_polys) {
                // Build zero-padded poly on CPU and upload to reusable fr_buffer
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
                            .arg(base_buffer)
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
                            .arg(base_buffer)
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

                // Step 8: Parallel chunked bucket reduction (Level 1)
                let total_blocks = num_windows * num_chunks_per_window;
                let chunked_reduce_kernel = program.create_kernel(
                    &format!("{}_reduce_buckets_chunked", G::name()),
                    total_blocks, REDUCTION_CHUNK_SIZE)?;
                chunked_reduce_kernel
                    .arg(&bucket_results_buffer)
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&(buckets_per_window as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&LocalBuffer::<G::Group>::new(REDUCTION_CHUNK_SIZE))
                    .run()?;

                // Step 8b: Combine chunks to windows (Level 2)
                let combine_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                let combine_kernel = program.create_kernel(
                    &format!("{}_combine_chunks_to_windows", G::name()),
                    combine_global, MSM_LOCAL_WORK_SIZE)?;
                combine_kernel
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&(REDUCTION_CHUNK_SIZE as u32))
                    .run()?;

                // Step 9: Horner reduction on GPU
                program.write_from_buffer(&mut final_result_buffer, &[<G::Group as AdditiveGroup>::ZERO; 1])?;
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
    /// - Hybrid MSM: Pippenger for Phase 2 (small MSMs, 0 CPU roundtrips), sort-based for Phase 3 (large MSMs, coalesced access)
    /// - Window size formula: min(log2(n/work_units) + 2, MAX_WINDOW_SIZE)
    /// - Phase 2 Pippenger buffers dropped before Phase 3 sort buffer allocation
    /// - Batched intermediate + commitment downloads after Phase 2 loop
    /// - scale_poly for combined_buffer init (no 128MB zero upload)
    /// - Per-iteration Fr buffers kept alive for streaming LC (no re-upload)
    /// - Streaming linear combination via linear_combine_accumulate (no flat polys_buffer)
    /// - Streaming witness computation: one eval point at a time (no bulk witnesses_buffer)
    pub fn fused_open<MiddleFn, AffineG>(
        &self,
        poly: &[F],
        challenges: &[F],
        bases: &[G::GpuRepr],
        cpu_bases: &[<G::Group as CurveGroup>::Affine],
        middle_fn: MiddleFn,
    ) -> EcResult<FusedOpenResult<F, AffineG, G::Group>>
    where
        MiddleFn: FnOnce(&[Vec<F>], &[G::Group]) -> Phase3Input<F, AffineG>,
        G::Group: VariableBaseMSM<MulBase = <G::Group as CurveGroup>::Affine>,
    {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );
        // Bases can come from either the persistent buffer or the passed-in slice
        let effective_base_len = if self.base_buffer.is_some() {
            self.base_buffer_len
        } else {
            bases.len()
        };
        assert!(
            effective_base_len >= poly.len(),
            "Not enough bases for polynomial size (have {}, need {})", effective_base_len, poly.len()
        );

        let num_challenges = challenges.len();
        let initial_len = poly.len();
        let challenges_vec = challenges.to_vec();
        let poly_vec = poly.to_vec();
        let cpu_bases_vec = cpu_bases.to_vec();
        let work_units = self.work_units;
        let max_window_size = self.max_window_size;
        let persistent_base = self.base_buffer.as_ref();
        let persistent_base_len = self.base_buffer_len;

        let closures = program_closures!(|program, middle_fn: MiddleFn| -> EcResult<FusedOpenResult<F, AffineG, G::Group>> {
            let fused_open_start = std::time::Instant::now();

            // ================================================================
            // Upload bases ONCE — reused for Phase 2 and Phase 3 MSMs
            // Use persistent buffer if available and large enough.
            // ================================================================
            let local_base_buf;
            let base_buffer = if let Some(pb) = persistent_base {
                if persistent_base_len >= initial_len {
                    pb
                } else {
                    local_base_buf = program.create_persistent_buffer_from_slice(bases)?;
                    &local_base_buf
                }
            } else {
                local_base_buf = program.create_persistent_buffer_from_slice(bases)?;
                &local_base_buf
            };

            // Upload polynomial
            let poly_buffer = program.create_buffer_from_slice(&poly_vec)?;
            let mut current_len = initial_len;

            let mut intermediates = Vec::with_capacity(num_challenges);
            let mut commitments = Vec::with_capacity(num_challenges);
            let mut intermediate_lens = Vec::with_capacity(num_challenges);

            // ================================================================
            // Pre-allocate shared Pippenger MSM buffers for Phase 2 and Phase 3
            // ================================================================
            const BN254_SCALAR_BITS: usize = 254;
            let effective_bits = BN254_SCALAR_BITS;

            let mut max_digits_len: usize = 0;
            let mut max_num_windows: usize = 0;
            let mut max_signed_bucket_len: usize = 0;

            // Determine how many Phase 2 iterations run on GPU vs CPU
            // Below GPU_MSM_THRESHOLD, CPU MSM is faster than GPU (avoids kernel launch overhead)
            let num_gpu_iterations = {
                let mut count = 0;
                let mut scan = initial_len;
                for _ in 0..num_challenges {
                    scan /= 2;
                    if scan > GPU_MSM_THRESHOLD {
                        count += 1;
                    } else {
                        break;
                    }
                }
                count
            };

            // Scan Phase 2 GPU iterations to find max buffer sizes
            let mut scan_len = initial_len;
            for _ in 0..num_gpu_iterations {
                scan_len /= 2;
                let ws = std::cmp::min(((div_ceil(scan_len, work_units) as f64).log2() as usize) + 2, max_window_size);
                let nw = div_ceil(effective_bits + 1, ws);
                let bpw = 1usize << (ws - 1);
                max_digits_len = max_digits_len.max(scan_len * nw);
                max_num_windows = max_num_windows.max(nw);
                max_signed_bucket_len = max_signed_bucket_len.max(bpw);
            }

            // Phase 3: witness MSMs use sort-based pipeline with n_bases = initial_len - 1
            let witness_len = initial_len - 1;
            let p3_ws = std::cmp::min(((div_ceil(witness_len, work_units) as f64).log2() as usize) + 2, max_window_size);
            let p3_num_windows = div_ceil(effective_bits + 1, p3_ws);
            let p3_buckets_per_window = 1usize << (p3_ws - 1);
            let p3_total_pairs = witness_len * p3_num_windows;
            let p3_total_buckets = p3_num_windows * p3_buckets_per_window;
            let p3_num_chunks_per_window = div_ceil(p3_buckets_per_window, REDUCTION_CHUNK_SIZE);
            let p3_total_chunks = p3_num_windows * p3_num_chunks_per_window;
            // Include Phase 3 in shared digits/windows sizes
            max_digits_len = max_digits_len.max(witness_len * p3_num_windows);
            max_num_windows = max_num_windows.max(p3_num_windows);

            // Shared buffers (used by both Phase 2 Pippenger and Phase 3 sort)
            // SAFETY: GPU kernels write to these buffers before reading
            let shared_digits_buffer = unsafe { program.create_buffer::<u16>(max_digits_len)? };
            let shared_window_results_buffer = unsafe { program.create_buffer::<G::Group>(max_num_windows)? };
            let shared_final_result_buffer = unsafe { program.create_buffer::<G::Group>(1)? };

            // Phase 2 Pippenger-specific buffers (dropped after Phase 2 to free GPU memory)
            let shared_bucket_buffer = if num_gpu_iterations > 0 {
                // multiexp_signed self-initializes buckets, so no external zero-fill needed
                Some(unsafe { program.create_buffer::<G::Group>(work_units * max_signed_bucket_len)? })
            } else {
                None
            };
            let shared_multiexp_results = if num_gpu_iterations > 0 {
                Some(unsafe { program.create_buffer::<G::Group>(work_units)? })
            } else {
                None
            };

            // GPU kernel names — shared
            let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());
            let reduce_windows_kernel_name = format!("{}_reduce_windows", G::name());
            let copy_at_offset_name = format!("{}_copy_at_offset", G::name());
            // Phase 2 Pippenger kernel names
            let multiexp_signed_name = format!("{}_multiexp_signed", G::name());
            let reduce_groups_name = format!("{}_reduce_multiexp_groups", G::name());
            // Phase 3 sort-based kernel names
            let decompose_kernel_name = format!("{}_decompose_to_pairs", G::name());
            let count_buckets_kernel_name = format!("{}_count_buckets", G::name());
            let prefix_sum_kernel_name = format!("{}_prefix_sum", G::name());
            let scatter_kernel_name = format!("{}_scatter_to_sorted", G::name());
            let accumulate_sorted_name = format!("{}_accumulate_sorted_buckets", G::name());
            let accumulate_chunked_name = format!("{}_accumulate_chunked", G::name());
            let reduce_partial_name = format!("{}_reduce_partial_buckets", G::name());
            let reduce_chunked_name = format!("{}_reduce_buckets_chunked", G::name());
            let combine_chunks_name = format!("{}_combine_chunks_to_windows", G::name());

            eprintln!("[fused_open] pre-alloc (buffers+kernel names): {:?}", fused_open_start.elapsed());

            // ================================================================
            // Phase 1+2: fix_vars + intermediate MSM commits
            // ================================================================
            let phase2_start = std::time::Instant::now();
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

                // === GPU iterations: fix_var on GPU + MSM on GPU ===
                for challenge_idx in 0..num_gpu_iterations {
                    let next_len = current_len / 2;

                    // === fix_var on GPU ===
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
                        .run_async()?;

                    intermediate_lens.push(next_len);

                    // === Convert Fr → scalar bytes on GPU ===
                    let to_scalar_kernel = program.create_kernel(
                        &format!("{}_to_scalar_bytes", F::name()),
                        fix_var_global_work_size, LOCAL_WORK_SIZE)?;
                    to_scalar_kernel
                        .arg(&per_iter_buffers[challenge_idx])
                        .arg(&shared_scalar_buffer)
                        .arg(&(next_len as u32))
                        .run_async()?;

                    // === Pippenger MSM on GPU (0 CPU roundtrips) ===
                    let ws = std::cmp::min(((div_ceil(next_len, work_units) as f64).log2() as usize) + 2, max_window_size);
                    let num_windows = div_ceil(effective_bits + 1, ws);
                    let n_bases = next_len;
                    let num_groups = work_units / num_windows;

                    // 1. Preprocess to signed digits
                    let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                    let preprocess_kernel = program.create_kernel(&preprocess_kernel_name,
                        preprocess_global, LOCAL_WORK_SIZE)?;
                    preprocess_kernel
                        .arg(&shared_scalar_buffer)
                        .arg(&shared_digits_buffer)
                        .arg(&(n_bases as u32))
                        .arg(&(num_windows as u32))
                        .arg(&(ws as u32))
                        .run_async()?;

                    // 2. Pippenger multiexp_signed (self-initializes buckets)
                    let msm_global = div_ceil(num_windows * num_groups, MSM_LOCAL_WORK_SIZE);
                    let msm_kernel = program.create_kernel(&multiexp_signed_name,
                        msm_global, MSM_LOCAL_WORK_SIZE)?;
                    msm_kernel
                        .arg(base_buffer)
                        .arg(shared_bucket_buffer.as_ref().unwrap())
                        .arg(shared_multiexp_results.as_ref().unwrap())
                        .arg(&shared_digits_buffer)
                        .arg(&(n_bases as u32))
                        .arg(&(num_groups as u32))
                        .arg(&(num_windows as u32))
                        .arg(&(ws as u32))
                        .run_async()?;

                    // 3. Reduce group results per window on GPU
                    let reduce_groups_global = div_ceil(num_windows, MSM_LOCAL_WORK_SIZE);
                    let reduce_groups_kernel = program.create_kernel(&reduce_groups_name,
                        reduce_groups_global, MSM_LOCAL_WORK_SIZE)?;
                    reduce_groups_kernel
                        .arg(shared_multiexp_results.as_ref().unwrap())
                        .arg(&shared_window_results_buffer)
                        .arg(&(num_groups as u32))
                        .arg(&(num_windows as u32))
                        .run_async()?;

                    // 4. Horner reduction across windows
                    let reduce_windows_kernel = program.create_kernel(&reduce_windows_kernel_name, 1, 1)?;
                    reduce_windows_kernel
                        .arg(&shared_window_results_buffer)
                        .arg(&shared_final_result_buffer)
                        .arg(&(num_windows as u32))
                        .arg(&(ws as u32))
                        .arg(&(effective_bits as u32))
                        .run_async()?;

                    // 5. Save commitment to GPU array (no per-iteration download)
                    let copy_kernel = program.create_kernel(&copy_at_offset_name, 1, 1)?;
                    copy_kernel
                        .arg(&shared_final_result_buffer)
                        .arg(&commitments_gpu)
                        .arg(&(challenge_idx as u32))
                        .run_async()?;

                    current_len = next_len;
                }

                eprintln!("[fused_open] phase2 GPU loop ({} iters): {:?}", num_gpu_iterations, phase2_start.elapsed());
                let cpu_fallback_start = std::time::Instant::now();

                // === CPU fallback iterations: fix_var on GPU, MSM on CPU ===
                // For small n_bases (≤ GPU_MSM_THRESHOLD), CPU MSM is faster
                // because it avoids kernel launch overhead per iteration.
                for challenge_idx in num_gpu_iterations..num_challenges {
                    let next_len = current_len / 2;

                    // fix_var still runs on GPU (1 kernel, trivially fast, keeps data on GPU for LC)
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
                        .run_async()?;

                    intermediate_lens.push(next_len);

                    // Download intermediate for CPU MSM
                    let mut intermediate = vec![F::ZERO; next_len];
                    program.read_into_buffer(&per_iter_buffers[challenge_idx], &mut intermediate)?;

                    // CPU MSM (arkworks VariableBaseMSM — faster than GPU for small n_bases)
                    let commitment = <G::Group as VariableBaseMSM>::msm(
                        &cpu_bases_vec[..next_len],
                        &intermediate,
                    ).map_err(|_| crate::error::EcError::Simple("CPU MSM fallback failed"))?;

                    // Store commitment directly (write to GPU array for batch download)
                    let commitment_vec = vec![commitment];
                    let commitment_buf = program.create_buffer_from_slice(&commitment_vec)?;
                    let copy_kernel = program.create_kernel(&copy_at_offset_name, 1, 1)?;
                    copy_kernel
                        .arg(&commitment_buf)
                        .arg(&commitments_gpu)
                        .arg(&(challenge_idx as u32))
                        .run_async()?;

                    current_len = next_len;
                }

                eprintln!("[fused_open] phase2 CPU fallback ({} iters): {:?}", num_challenges - num_gpu_iterations, cpu_fallback_start.elapsed());
                let download_start = std::time::Instant::now();

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
                eprintln!("[fused_open] batch download: {:?}", download_start.elapsed());
            } // end if num_challenges > 0

            // ================================================================
            // CPU callback: transcript work between Phase 2 and Phase 3
            // GPU buffers (base_buffer, poly_buffer, per_iter_buffers) remain alive!
            // ================================================================
            let callback_start = std::time::Instant::now();
            let phase3_input = middle_fn(&intermediates, &commitments);
            eprintln!("[fused_open] CPU callback: {:?}", callback_start.elapsed());

            // ================================================================
            // Phase 3: Streaming LC using on-GPU per-iteration buffers
            // ================================================================
            let lc_start = std::time::Instant::now();
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
            // Reusable single-element coefficient buffer (avoids per-iteration cuMalloc)
            let mut coeff_buffer = program.create_buffer_from_slice(&[coeffs_vec[0]])?;
            let scale_poly_kernel = program.create_kernel(
                &format!("{}_scale_poly", F::name()),
                lc_global_work_size, LOCAL_WORK_SIZE)?;
            scale_poly_kernel
                .arg(&poly_buffer)
                .arg(&combined_buffer)
                .arg(&coeff_buffer)
                .arg(&(poly_len as u32))
                .run_async()?;

            // Accumulate each intermediate directly from on-GPU per-iteration buffers
            for (idx, iter_buf) in per_iter_buffers.iter().enumerate() {
                let src_len = intermediate_lens[idx];
                // Reuse single coeff_buffer instead of allocating a new one each iteration
                program.write_from_buffer(&mut coeff_buffer, &[coeffs_vec[idx + 1]])?;

                let lc_accum_kernel = program.create_kernel(&lc_accum_kernel_name, lc_global_work_size, LOCAL_WORK_SIZE)?;
                lc_accum_kernel
                    .arg(iter_buf)
                    .arg(&combined_buffer)
                    .arg(&coeff_buffer)
                    .arg(&(src_len as u32))
                    .arg(&(poly_len as u32))
                    .run_async()?;
            }

            eprintln!("[fused_open] phase3 LC: {:?}", lc_start.elapsed());

            // Free Phase 2 Pippenger buffers + per-iteration buffers before Phase 3
            drop(per_iter_buffers);
            drop(shared_bucket_buffer);
            drop(shared_multiexp_results);

            // ================================================================
            // Phase 3: Streaming witness computation + sort-based MSMs
            // Sort-based MSM is ~70x faster than Pippenger for large 4M-base MSMs
            // because sort enables coalesced memory access (sequential bases per bucket).
            // ================================================================
            let witness_msm_start = std::time::Instant::now();
            // Single witness buffer (reusable across all eval points)
            let single_witness_buffer = unsafe { program.create_buffer::<F>(witness_len)? };

            // Witness computation uses 3-phase parallel approach
            let chunk_size = std::cmp::max(1, witness_len / 4096);
            let num_chunks = div_ceil(witness_len, chunk_size);
            let carries_buffer = unsafe { program.create_buffer::<F>(num_chunks)? };
            let propagated_carries_buffer = unsafe { program.create_buffer::<F>(num_chunks)? };

            // Scalar buffer for witness MSM
            let witness_scalar_buffer = unsafe { program.create_buffer::<F>(witness_len)? };

            let phase1_kernel_name = format!("{}_witness_poly_batch_phase1", F::name());
            let phase2_carry_kernel_name = format!("{}_witness_carry_propagate", F::name());
            let phase3_apply_kernel_name = format!("{}_witness_poly_batch_phase3", F::name());

            // Phase 3 sort-based MSM buffers (constant size across all 3 witness MSMs)
            let p3_keys_buffer = unsafe { program.create_buffer::<u32>(p3_total_pairs)? };
            let p3_values_buffer = unsafe { program.create_buffer::<u32>(p3_total_pairs)? };
            let p3_sorted_values_buffer = unsafe { program.create_buffer::<u32>(p3_total_pairs)? };
            let mut p3_counts_buffer = unsafe { program.create_buffer::<u32>(p3_total_buckets)? };
            let p3_offsets_buffer = unsafe { program.create_buffer::<u32>(p3_total_buckets)? };
            let p3_nonempty_ids_buffer = unsafe { program.create_buffer::<u32>(p3_total_buckets)? };
            let mut p3_scatter_offsets_buffer = unsafe { program.create_buffer::<u32>(p3_total_buckets)? };
            let p3_num_nonempty_buffer = unsafe { program.create_buffer::<u32>(1)? };
            let mut p3_bucket_results_buffer = {
                let identity_points = vec![<G::Group as AdditiveGroup>::ZERO; p3_total_buckets];
                program.create_buffer_from_slice(&identity_points)?
            };
            let p3_chunk_sbp_buffer = unsafe { program.create_buffer::<G::Group>(p3_total_chunks.max(1))? };
            let p3_chunk_sum_buffer = unsafe { program.create_buffer::<G::Group>(p3_total_chunks.max(1))? };
            // CPU scratch vectors for sort dispatch tables
            let mut p3_counts_cpu = vec![0u32; p3_total_buckets];
            let mut p3_nonempty_ids_cpu = vec![0u32; p3_total_buckets];
            let mut p3_offsets_copy = vec![0u32; p3_total_buckets];

            // GPU buffer to accumulate witness commitments (batch download after loop)
            let witness_commitments_gpu = unsafe { program.create_buffer::<G::Group>(num_points.max(1))? };

            // Reusable single-element eval point buffer (avoids per-iteration cuMalloc)
            let mut single_point_buffer = if num_points > 0 {
                program.create_buffer_from_slice(&[phase3_input.eval_points[0]])?
            } else {
                // Dummy buffer, won't be used
                unsafe { program.create_buffer::<F>(1)? }
            };

            let n_bases = witness_len;

            for point_idx in 0..num_points {
                let witness_iter_start = std::time::Instant::now();
                if point_idx > 0 {
                    program.write_from_buffer(&mut single_point_buffer, &[phase3_input.eval_points[point_idx]])?;
                }

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
                    .run_async()?;

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
                    .run_async()?;

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
                    .run_async()?;

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

                eprintln!("[fused_open]   witness[{}] compute+to_scalar: {:?}", point_idx, witness_iter_start.elapsed());
                let sort_msm_start = std::time::Instant::now();

                // === Sort-based MSM (coalesced memory access for large MSMs) ===

                // 1. Preprocess to signed digits
                let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                let preprocess_kernel = program.create_kernel(&preprocess_kernel_name,
                    preprocess_global, LOCAL_WORK_SIZE)?;
                preprocess_kernel
                    .arg(&witness_scalar_buffer)
                    .arg(&shared_digits_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(p3_num_windows as u32))
                    .arg(&(p3_ws as u32))
                    .run()?;

                // 2. Decompose to (key, value) pairs
                let decompose_global = div_ceil(p3_total_pairs, MSM_LOCAL_WORK_SIZE);
                let decompose_kernel = program.create_kernel(&decompose_kernel_name,
                    decompose_global, MSM_LOCAL_WORK_SIZE)?;
                decompose_kernel
                    .arg(&shared_digits_buffer)
                    .arg(&p3_keys_buffer)
                    .arg(&p3_values_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(p3_num_windows as u32))
                    .arg(&(p3_buckets_per_window as u32))
                    .run()?;

                // 3. Count buckets (re-init counts to zero)
                program.write_from_buffer(&mut p3_counts_buffer, &vec![0u32; p3_total_buckets])?;
                let count_global = div_ceil(p3_total_pairs, MSM_LOCAL_WORK_SIZE);
                let count_kernel = program.create_kernel(&count_buckets_kernel_name,
                    count_global, MSM_LOCAL_WORK_SIZE)?;
                count_kernel
                    .arg(&p3_keys_buffer)
                    .arg(&p3_counts_buffer)
                    .arg(&(p3_total_pairs as u32))
                    .run()?;

                // 4. Prefix sum
                let prefix_kernel = program.create_kernel(&prefix_sum_kernel_name, 1, 1)?;
                prefix_kernel
                    .arg(&p3_counts_buffer)
                    .arg(&p3_offsets_buffer)
                    .arg(&p3_nonempty_ids_buffer)
                    .arg(&p3_num_nonempty_buffer)
                    .arg(&(p3_total_buckets as u32))
                    .run()?;

                eprintln!("[fused_open]   witness[{}] sort prep (preprocess+decompose+count+prefix): {:?}", point_idx, sort_msm_start.elapsed());

                // 5. Download num_nonempty
                let mut num_nonempty_vec = vec![0u32; 1];
                program.read_into_buffer(&p3_num_nonempty_buffer, &mut num_nonempty_vec)?;
                let num_nonempty = num_nonempty_vec[0] as usize;

                // 6. Copy offsets for scatter (scatter modifies them via atomicAdd)
                program.read_into_buffer(&p3_offsets_buffer, &mut p3_offsets_copy)?;
                program.write_from_buffer(&mut p3_scatter_offsets_buffer, &p3_offsets_copy)?;

                // 7. Scatter to sorted
                let scatter_global = div_ceil(p3_total_pairs, MSM_LOCAL_WORK_SIZE);
                let scatter_kernel = program.create_kernel(&scatter_kernel_name,
                    scatter_global, MSM_LOCAL_WORK_SIZE)?;
                scatter_kernel
                    .arg(&p3_keys_buffer)
                    .arg(&p3_values_buffer)
                    .arg(&p3_scatter_offsets_buffer)
                    .arg(&p3_sorted_values_buffer)
                    .arg(&(p3_total_pairs as u32))
                    .run()?;

                let accum_start = std::time::Instant::now();
                // 8. Accumulate sorted buckets (with chunked dispatch for large buckets)
                program.write_from_buffer(&mut p3_bucket_results_buffer,
                    &vec![<G::Group as AdditiveGroup>::ZERO; p3_total_buckets])?;
                if num_nonempty > 0 {
                    program.read_into_buffer(&p3_counts_buffer, &mut p3_counts_cpu)?;
                    program.read_into_buffer(&p3_nonempty_ids_buffer, &mut p3_nonempty_ids_cpu)?;

                    let (dispatch_table, reduce_table, num_dispatches) =
                        crate::multiexp::build_dispatch_tables(&p3_counts_cpu, &p3_nonempty_ids_cpu, num_nonempty);

                    if num_dispatches == num_nonempty {
                        // No large buckets — use simple 1-thread-per-bucket kernel
                        let accum_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let accum_kernel = program.create_kernel(&accumulate_sorted_name,
                            accum_global, MSM_LOCAL_WORK_SIZE)?;
                        accum_kernel
                            .arg(base_buffer)
                            .arg(&p3_sorted_values_buffer)
                            .arg(&p3_offsets_buffer)
                            .arg(&p3_counts_buffer)
                            .arg(&p3_nonempty_ids_buffer)
                            .arg(&p3_bucket_results_buffer)
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
                        let chunked_kernel = program.create_kernel(&accumulate_chunked_name,
                            chunked_global, MSM_LOCAL_WORK_SIZE)?;
                        chunked_kernel
                            .arg(base_buffer)
                            .arg(&p3_sorted_values_buffer)
                            .arg(&p3_offsets_buffer)
                            .arg(&dispatch_buffer)
                            .arg(&partial_results_buffer)
                            .arg(&(num_dispatches as u32))
                            .run()?;

                        let reduce_table_buffer = program.create_buffer_from_slice(&reduce_table)?;
                        let reduce_global = div_ceil(num_nonempty, MSM_LOCAL_WORK_SIZE);
                        let reduce_kernel = program.create_kernel(&reduce_partial_name,
                            reduce_global, MSM_LOCAL_WORK_SIZE)?;
                        reduce_kernel
                            .arg(&partial_results_buffer)
                            .arg(&p3_nonempty_ids_buffer)
                            .arg(&reduce_table_buffer)
                            .arg(&p3_bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    }
                }

                // 9. Parallel chunked bucket reduction (Level 1)
                let total_blocks = p3_num_windows * p3_num_chunks_per_window;
                let chunked_reduce_kernel = program.create_kernel(&reduce_chunked_name,
                    total_blocks, REDUCTION_CHUNK_SIZE)?;
                chunked_reduce_kernel
                    .arg(&p3_bucket_results_buffer)
                    .arg(&p3_chunk_sbp_buffer)
                    .arg(&p3_chunk_sum_buffer)
                    .arg(&(p3_buckets_per_window as u32))
                    .arg(&(p3_num_chunks_per_window as u32))
                    .arg(&LocalBuffer::<G::Group>::new(REDUCTION_CHUNK_SIZE))
                    .run()?;

                // 9b. Combine chunks to windows (Level 2)
                let combine_global = div_ceil(p3_num_windows, MSM_LOCAL_WORK_SIZE);
                let combine_kernel = program.create_kernel(&combine_chunks_name,
                    combine_global, MSM_LOCAL_WORK_SIZE)?;
                combine_kernel
                    .arg(&p3_chunk_sbp_buffer)
                    .arg(&p3_chunk_sum_buffer)
                    .arg(&shared_window_results_buffer)
                    .arg(&(p3_num_windows as u32))
                    .arg(&(p3_num_chunks_per_window as u32))
                    .arg(&(REDUCTION_CHUNK_SIZE as u32))
                    .run()?;

                eprintln!("[fused_open]   witness[{}] accum+reduce: {:?}", point_idx, accum_start.elapsed());

                // 10. Horner reduction on GPU
                let reduce_windows_kernel = program.create_kernel(&reduce_windows_kernel_name, 1, 1)?;
                reduce_windows_kernel
                    .arg(&shared_window_results_buffer)
                    .arg(&shared_final_result_buffer)
                    .arg(&(p3_num_windows as u32))
                    .arg(&(p3_ws as u32))
                    .arg(&(effective_bits as u32))
                    .run()?;

                // 11. Save witness commitment to GPU array
                let copy_kernel = program.create_kernel(&copy_at_offset_name, 1, 1)?;
                copy_kernel
                    .arg(&shared_final_result_buffer)
                    .arg(&witness_commitments_gpu)
                    .arg(&(point_idx as u32))
                    .run()?;

                eprintln!("[fused_open]   witness[{}] TOTAL: {:?}", point_idx, witness_iter_start.elapsed());
            }

            eprintln!("[fused_open] phase3 witness+MSM ({} points): {:?}", num_points, witness_msm_start.elapsed());

            // Batch download all witness commitments (1 sync instead of 3)
            let mut witness_commitments = vec![<G::Group as AdditiveGroup>::ZERO; num_points];
            if num_points > 0 {
                program.read_into_buffer(&witness_commitments_gpu, &mut witness_commitments)?;
            }

            eprintln!("[fused_open] TOTAL: {:?}", fused_open_start.elapsed());

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
            ((n as f64).ln() * 100.0 / 69.0) as usize
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
        let base_affine: &<G::Group as CurveGroup>::Affine = base;
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
