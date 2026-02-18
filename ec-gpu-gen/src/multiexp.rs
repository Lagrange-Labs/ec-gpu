use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use ec_gpu::GpuName;
use log::{error, info};
use rust_gpu_tools::{program_closures, Device, LocalBuffer, Program};
use tracing::debug_span;
use yastl::Scope;

use crate::{
    error::{EcError, EcResult},
    threadpool::Worker,
};

/// Trait for curve affine points that have a GPU-compatible representation.
pub trait GpuAffine: GpuName + Clone + Send + Sync + Sized {
    /// The GPU-compatible representation type.
    type GpuRepr: Copy + Clone + Default + Send + Sync;
    /// The scalar field type.
    type ScalarField: PrimeField;
    /// The projective group type.
    type Group: CurveGroup<ScalarField = Self::ScalarField> + AddAssign;

    /// Convert the affine point to its GPU representation.
    fn to_gpu(&self) -> Self::GpuRepr;
}

/// On the GPU, the exponents are split into windows, this is the maximum number of such windows.
const MAX_WINDOW_SIZE: usize = 16;
/// In CUDA this is the number of blocks per grid (grid size).
const LOCAL_WORK_SIZE: usize = 128;
/// Let 20% of GPU memory be free, this is an arbitrary value.
const MEMORY_PADDING: f64 = 0.2f64;
/// The Nvidia Ampere architecture is compute capability major version 8.
const AMPERE: u32 = 8;

/// Divide and ceil to the next value.
const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

/// The number of units the work is split into. One unit will result in one CUDA thread.
///
/// Based on empirical results, it turns out that on Nvidia devices with the Ampere architecture,
/// it's faster to use two times the number of work units.
const fn work_units(compute_units: u32, compute_capabilities: Option<(u32, u32)>) -> usize {
    match compute_capabilities {
        Some((AMPERE, _)) => LOCAL_WORK_SIZE * compute_units as usize * 2,
        _ => LOCAL_WORK_SIZE * compute_units as usize,
    }
}

/// Compute the number of work units for a device.
///
/// This is needed by [`FusedPolyCommit`](crate::gpu_buffer::FusedPolyCommit) to create
/// a fused poly-commit handler without duplicating the work-unit calculation logic.
pub fn compute_work_units(device: &Device) -> usize {
    work_units(device.compute_units(), device.compute_capability())
}

/// Multiexp kernel for a single GPU.
pub struct SingleMultiexpKernel<'a, G>
where
    G: GpuAffine,
{
    program: Program,
    /// The number of exponentiations the GPU can handle in a single execution of the kernel.
    n: usize,
    /// The number of units the work is split into. It will results in this amount of threads on
    /// the GPU.
    work_units: usize,
    /// An optional function which will be called at places where it is possible to abort the
    /// multiexp calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,

    _phantom: std::marker::PhantomData<G>,
}

/// Calculates the maximum number of terms that can be put onto the GPU memory.
///
/// Returns `(chunk_size, effective_work_units)`. The work units may be reduced from the
/// requested value if the GPU doesn't have enough memory for the bucket overhead.
fn calc_chunk_size<G>(mem: u64, work_units: usize) -> (usize, usize)
where
    G: GpuAffine,
{
    let aff_size = std::mem::size_of::<G::GpuRepr>();
    let exp_size = exp_size::<G::ScalarField>();
    let proj_size = std::mem::size_of::<G::Group>();

    // Leave `MEMORY_PADDING` percent of the memory free.
    let max_memory = ((mem as f64) * (1f64 - MEMORY_PADDING)) as usize;
    // The amount of memory (in bytes) of a single term.
    let term_size = aff_size + exp_size;
    // The number of buckets needed for one work unit (signed-digit: half)
    let max_buckets_per_work_unit = 1 << (MAX_WINDOW_SIZE - 1);
    // Per-work-unit overhead: one bucket array + one result element
    let overhead_per_wu = max_buckets_per_work_unit * proj_size + proj_size;

    // Clamp work_units so that the overhead fits in available memory (leave room for ≥1 term).
    let max_wu = max_memory / (overhead_per_wu + term_size);
    let effective_wu = work_units.min(max_wu).max(1);

    let overhead = effective_wu * overhead_per_wu;
    ((max_memory - overhead) / term_size, effective_wu)
}

/// Calculates the maximum number of terms that can be put onto the GPU memory for sorted MSM.
///
/// The sorted MSM uses a different memory layout:
/// - Bases and exponents (same as regular MSM)
/// - Signed digits: n * num_windows * sizeof(u16)
/// - Pairs (keys + values): n * num_windows * sizeof(u32) * 2
/// - Sorted values: n * num_windows * sizeof(u32)
/// - Counts, offsets, nonempty_ids: total_buckets * sizeof(u32) * 3
/// - Bucket results: total_buckets * proj_size
/// - Window results: num_windows * proj_size
/// - Final result: 1 * proj_size
#[allow(dead_code)]
fn calc_chunk_size_sorted<G>(mem: u64) -> usize
where
    G: GpuAffine,
{
    let aff_size = std::mem::size_of::<G::GpuRepr>();
    let exp_size = exp_size::<G::ScalarField>();
    let proj_size = std::mem::size_of::<G::Group>();

    // Leave `MEMORY_PADDING` percent of the memory free.
    let max_memory = ((mem as f64) * (1f64 - MEMORY_PADDING)) as usize;

    // For sorted MSM, we use max window size to calculate worst-case memory
    let num_windows = div_ceil(256, MAX_WINDOW_SIZE); // 256 bits max for Fr
    let buckets_per_window = 1 << (MAX_WINDOW_SIZE - 1); // signed digit
    let total_buckets = num_windows * buckets_per_window;

    // Per-base memory
    let term_size = aff_size + exp_size;

    // Per-base intermediate buffers (digits, pairs, sorted values)
    let per_base_intermediate =
        num_windows * 2 +        // digits (u16)
        num_windows * 4 * 2 +    // pairs (u32 key, u32 value)
        num_windows * 4;         // sorted values (u32)

    // Fixed-size buffers (independent of n)
    let fixed_buffers =
        total_buckets * 4 * 3 +  // counts, offsets, nonempty_ids (u32)
        total_buckets * proj_size +  // bucket_results
        num_windows * proj_size +    // window_results
        proj_size +                  // final_result
        4;                           // num_nonempty (u32)

    // n * (term_size + per_base_intermediate) + fixed_buffers <= max_memory
    (max_memory - fixed_buffers) / (term_size + per_base_intermediate)
}

/// The size of the exponent in bytes.
///
/// It's the actual bytes size it needs in memory, not it's theoretical bit size.
fn exp_size<F: ark_ff::PrimeField>() -> usize {
    std::mem::size_of::<F::BigInt>()
}

/// Computes the maximum number of significant bits across all scalar byte arrays.
/// Returns the position of the highest set bit + 1, or 1 if all scalars are zero.
fn compute_max_scalar_bits(scalars: &[[u8; 32]]) -> usize {
    let max_bits = scalars
        .iter()
        .map(|bytes| {
            // Scan from MSB to find highest non-zero byte
            for (i, &byte) in bytes.iter().enumerate().rev() {
                if byte != 0 {
                    return (i + 1) * 8 - byte.leading_zeros() as usize;
                }
            }
            0
        })
        .max()
        .unwrap_or(0);
    // Ensure at least 1 to avoid edge cases
    max_bits.max(1)
}

/// GPU-compatible representation of an affine point.
/// Coordinates are stored as 32-byte little-endian field elements in Montgomery form.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct G1AffineM {
    /// X coordinate as 32 bytes in little-endian Montgomery form
    pub x: [u8; 32],
    /// Y coordinate as 32 bytes in little-endian Montgomery form
    pub y: [u8; 32],
}

#[cfg(feature = "arkworks")]
fn fq_to_montgomery_bytes(f: &ark_bn254::Fq) -> [u8; 32] {
    // Arkworks stores Fq as 4 u64 limbs in Montgomery form
    // We need the raw Montgomery representation, not the serialized (standard) form
    let limbs: [u64; 4] = unsafe { std::mem::transmute_copy(f) };
    let mut out = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    out
}

#[cfg(feature = "arkworks")]
impl From<ark_bn254::G1Affine> for G1AffineM {
    fn from(p: ark_bn254::G1Affine) -> Self {
        use ark_ec::AffineRepr;

        if p.is_zero() {
            return Self::default();
        }

        Self {
            x: fq_to_montgomery_bytes(&p.x),
            y: fq_to_montgomery_bytes(&p.y),
        }
    }
}

#[cfg(feature = "arkworks")]
impl From<&ark_bn254::G1Affine> for G1AffineM {
    fn from(p: &ark_bn254::G1Affine) -> Self {
        (*p).into()
    }
}

/// GPU-compatible representation of a G2 affine point.
/// Coordinates are stored as 64-byte little-endian Fq2 elements (each Fq2 = two 32-byte Fq elements).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct G2AffineM {
    /// X coordinate as Fq2 (64 bytes: c0 followed by c1)
    pub x: [u8; 64],
    /// Y coordinate as Fq2 (64 bytes: c0 followed by c1)
    pub y: [u8; 64],
}

impl Default for G2AffineM {
    fn default() -> Self {
        Self {
            x: [0u8; 64],
            y: [0u8; 64],
        }
    }
}

#[cfg(feature = "arkworks")]
fn fq2_to_montgomery_bytes(f: &ark_bn254::Fq2) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[..32].copy_from_slice(&fq_to_montgomery_bytes(&f.c0));
    out[32..].copy_from_slice(&fq_to_montgomery_bytes(&f.c1));
    out
}

#[cfg(feature = "arkworks")]
impl From<ark_bn254::G2Affine> for G2AffineM {
    fn from(p: ark_bn254::G2Affine) -> Self {
        use ark_ec::AffineRepr;

        if p.is_zero() {
            return Self::default();
        }

        Self {
            x: fq2_to_montgomery_bytes(&p.x),
            y: fq2_to_montgomery_bytes(&p.y),
        }
    }
}

#[cfg(feature = "arkworks")]
impl From<&ark_bn254::G2Affine> for G2AffineM {
    fn from(p: &ark_bn254::G2Affine) -> Self {
        (*p).into()
    }
}

#[cfg(feature = "arkworks")]
impl GpuAffine for ec_gpu::arkworks_bn254::G1Affine {
    type GpuRepr = G1AffineM;
    type ScalarField = ark_bn254::Fr;
    type Group = ark_bn254::G1Projective;

    fn to_gpu(&self) -> G1AffineM {
        self.0.into()
    }
}

#[cfg(feature = "arkworks")]
impl GpuAffine for ec_gpu::arkworks_bn254::G2Affine {
    type GpuRepr = G2AffineM;
    type ScalarField = ark_bn254::Fr;
    type Group = ark_bn254::G2Projective;

    fn to_gpu(&self) -> G2AffineM {
        self.0.into()
    }
}

impl<'a, G> SingleMultiexpKernel<'a, G>
where
    G: GpuAffine,
{
    /// Create a new Multiexp kernel instance for a device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        program: Program,
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let _span = debug_span!("single_multiexp_kernel_create").entered();
        let mem = device.memory();
        let compute_units = device.compute_units();
        let compute_capability = device.compute_capability();
        let work_units = work_units(compute_units, compute_capability);
        let (chunk_size, effective_work_units) = calc_chunk_size::<G>(mem, work_units);

        Ok(SingleMultiexpKernel {
            program,
            n: chunk_size,
            work_units: effective_work_units,
            maybe_abort,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Run the actual multiexp computation on the GPU using signed-digit decomposition.
    ///
    /// Uses Booth encoding to halve the number of buckets per thread (from 2^w-1 to 2^(w-1)),
    /// which roughly halves the summation-by-parts cost in each thread.
    ///
    /// The number of `bases` and `exponents` are determined by [`SingleMultiexpKernel`]`::n`, this
    /// means that it is guaranteed that this amount of calculations fit on the GPU this kernel is
    /// running on.
    pub fn multiexp(
        &self,
        bases: &[G::GpuRepr],
        exponents: &[<G::ScalarField as ark_ff::PrimeField>::BigInt],
    ) -> EcResult<G::Group> {
        let _span = debug_span!("single_multiexp", n = bases.len()).entered();
        assert_eq!(bases.len(), exponents.len());

        let exponents: Vec<_> = {
            let _span = debug_span!("convert_exponents").entered();
            exponents
                .iter()
                .map(|b| {
                    let mut out = [0u8; 32];
                    let le = b.to_bytes_le();
                    out[..le.len()].copy_from_slice(&le);

                    out
                })
                .collect()
        };

        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }

        // Compute actual bit length needed for small scalar optimization
        let effective_bits = compute_max_scalar_bits(&exponents);

        let window_size = self.calc_window_size(bases.len());
        // Signed-digit (Booth) encoding can produce a carry out of the last window,
        // so we need one extra bit of headroom: effective_bits + 1.
        let num_windows = div_ceil(effective_bits + 1, window_size);
        let num_groups = self.work_units / num_windows;
        // Signed-digit: half the buckets (2^(w-1) instead of 2^w - 1)
        let signed_bucket_len = 1 << (window_size - 1);
        let n_bases = bases.len();

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<G::Group>> {
            let base_buffer = {
                let _span = debug_span!("upload_bases").entered();
                program.create_buffer_from_slice(bases)?
            };
            let exp_buffer = {
                let _span = debug_span!("upload_exponents").entered();
                program.create_buffer_from_slice(&exponents)?
            };

            // Preprocessing: convert exponents to signed digits
            let digits_buffer = {
                let _span = debug_span!("preprocess_signed_digits").entered();
                let digits_len = n_bases * num_windows;
                // SAFETY: GPU will initialize this buffer
                let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };

                let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());
                let preprocess_kernel = program.create_kernel(
                    &preprocess_kernel_name, preprocess_global, LOCAL_WORK_SIZE)?;

                preprocess_kernel
                    .arg(&exp_buffer)
                    .arg(&digits_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                digits_buffer
            };

            let (bucket_buffer, result_buffer) = {
                let _span = debug_span!("allocate_gpu_buffers").entered();
                // SAFETY: GPU will initialize these buffers
                let bucket_buffer =
                    unsafe { program.create_buffer::<G::Group>(self.work_units * signed_bucket_len)? };
                let result_buffer = unsafe { program.create_buffer::<G::Group>(self.work_units)? };
                (bucket_buffer, result_buffer)
            };

            // The global work size follows CUDA's definition and is the number of
            // `LOCAL_WORK_SIZE` sized thread groups.
            let global_work_size = div_ceil(num_windows * num_groups, LOCAL_WORK_SIZE);

            let kernel = {
                let _span = debug_span!("create_kernel").entered();
                let kernel_name = format!("{}_multiexp_signed", G::name());
                program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?
            };

            {
                let _span = debug_span!("kernel_run").entered();
                kernel
                    .arg(&base_buffer)
                    .arg(&bucket_buffer)
                    .arg(&result_buffer)
                    .arg(&digits_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_groups as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;
            }

            let mut results = vec![<G::Group as AdditiveGroup>::ZERO; self.work_units];
            {
                let _span = debug_span!("download_results").entered();
                program.read_into_buffer(&result_buffer, &mut results)?;
            }

            Ok(results)
        });

        let results = self.program.run(closures, ())?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        // Since we use LSB-first bit extraction, window 0 contains the LSB and window (num_windows-1)
        // contains the MSB. We process windows in reverse order (MSB first) using Horner's method.
        let acc = {
            let _span = debug_span!("cpu_accumulation").entered();
            let mut acc = <G::Group as AdditiveGroup>::ZERO;
            for i in (0..num_windows).rev() {
                // Window i covers bits [i * window_size, min((i+1) * window_size, effective_bits))
                let w = std::cmp::min(window_size, effective_bits - i * window_size);
                for _ in 0..w {
                    acc = acc.double();
                }
                for g in 0..num_groups {
                    acc.add_assign(&results[g * num_windows + i]);
                }
            }
            acc
        };

        Ok(acc)
    }

    /// Calculates the window size, based on the given number of terms.
    ///
    /// For best performance, the window size is reduced, so that maximum parallelism is possible.
    /// If you e.g. have put only a subset of the terms into the GPU memory, then a smaller window
    /// size leads to more windows, hence more units to work on, as we split the work into
    /// `num_windows * num_groups`.
    fn calc_window_size(&self, num_terms: usize) -> usize {
        // The window size was determined by running the `gpu_multiexp_consistency` test and
        // looking at the resulting numbers.
        let window_size = ((div_ceil(num_terms, self.work_units) as f64).log2() as usize) + 2;
        std::cmp::min(window_size, MAX_WINDOW_SIZE)
    }

    /// Run MSM using sort-based bucket accumulation.
    ///
    /// This replaces per-thread private buckets with a counting-sort approach
    /// where bases are sorted by bucket index, giving sequential memory reads
    /// and better GPU utilization.
    pub fn multiexp_sorted(
        &self,
        bases: &[G::GpuRepr],
        exponents: &[<G::ScalarField as ark_ff::PrimeField>::BigInt],
    ) -> EcResult<G::Group> {
        let _span = debug_span!("single_multiexp_sorted", n = bases.len()).entered();
        assert_eq!(bases.len(), exponents.len());

        let exponents: Vec<_> = {
            let _span = debug_span!("convert_exponents").entered();
            exponents
                .iter()
                .map(|b| {
                    let mut out = [0u8; 32];
                    let le = b.to_bytes_le();
                    out[..le.len()].copy_from_slice(&le);
                    out
                })
                .collect()
        };

        if let Some(maybe_abort) = &self.maybe_abort {
            if maybe_abort() {
                return Err(EcError::Aborted);
            }
        }

        // Compute actual bit length needed for small scalar optimization
        let effective_bits = compute_max_scalar_bits(&exponents);

        let window_size = self.calc_window_size(bases.len());
        let num_windows = div_ceil(effective_bits + 1, window_size);
        let buckets_per_window = 1 << (window_size - 1); // signed digit
        let total_buckets = num_windows * buckets_per_window;
        let n_bases = bases.len();

        let closures = program_closures!(|program, _arg| -> EcResult<G::Group> {
            // Upload bases and exponents
            let base_buffer = {
                let _span = debug_span!("upload_bases").entered();
                program.create_buffer_from_slice(bases)?
            };
            let exp_buffer = {
                let _span = debug_span!("upload_exponents").entered();
                program.create_buffer_from_slice(&exponents)?
            };

            // Step 1: Preprocess signed digits (same as regular multiexp)
            let digits_buffer = {
                let _span = debug_span!("preprocess_signed_digits").entered();
                let digits_len = n_bases * num_windows;
                // SAFETY: GPU will initialize this buffer
                let digits_buffer = unsafe { program.create_buffer::<u16>(digits_len)? };

                let preprocess_global = div_ceil(n_bases, LOCAL_WORK_SIZE);
                let preprocess_kernel_name = format!("{}_preprocess_signed_digits", G::name());
                let preprocess_kernel = program.create_kernel(
                    &preprocess_kernel_name, preprocess_global, LOCAL_WORK_SIZE)?;

                preprocess_kernel
                    .arg(&exp_buffer)
                    .arg(&digits_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .run()?;

                digits_buffer
            };

            // Step 2: Decompose to (key, value) pairs
            let (keys_buffer, values_buffer) = {
                let _span = debug_span!("decompose_to_pairs").entered();
                let pairs_len = n_bases * num_windows;
                // SAFETY: GPU will initialize these buffers
                let keys_buffer = unsafe { program.create_buffer::<u32>(pairs_len)? };
                let values_buffer = unsafe { program.create_buffer::<u32>(pairs_len)? };

                let total_pairs = n_bases * num_windows;
                let decompose_global = div_ceil(total_pairs, LOCAL_WORK_SIZE);
                let decompose_kernel_name = format!("{}_decompose_to_pairs", G::name());
                let decompose_kernel = program.create_kernel(
                    &decompose_kernel_name, decompose_global, LOCAL_WORK_SIZE)?;

                decompose_kernel
                    .arg(&digits_buffer)
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&(n_bases as u32))
                    .arg(&(num_windows as u32))
                    .arg(&(buckets_per_window as u32))
                    .run()?;

                (keys_buffer, values_buffer)
            };

            // Step 3: Allocate and zero-initialize counts buffer
            let counts_buffer = {
                let _span = debug_span!("allocate_counts").entered();
                program.create_buffer_from_slice(&vec![0u32; total_buckets])?
            };

            // Step 4: Count buckets (atomic histogram)
            {
                let _span = debug_span!("count_buckets").entered();
                let count_global = div_ceil(n_bases * num_windows, LOCAL_WORK_SIZE);
                let count_kernel_name = format!("{}_count_buckets", G::name());
                let count_kernel = program.create_kernel(
                    &count_kernel_name, count_global, LOCAL_WORK_SIZE)?;

                let total_pairs = (n_bases * num_windows) as u32;
                count_kernel
                    .arg(&keys_buffer)
                    .arg(&counts_buffer)
                    .arg(&total_pairs)
                    .run()?;
            }

            // Step 5: Prefix sum (single thread)
            let (offsets_buffer, nonempty_ids_buffer, num_nonempty_buffer) = {
                let _span = debug_span!("prefix_sum").entered();
                // SAFETY: GPU will initialize these buffers
                let offsets_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
                let nonempty_ids_buffer = unsafe { program.create_buffer::<u32>(total_buckets)? };
                let num_nonempty_buffer = unsafe { program.create_buffer::<u32>(1)? };

                let prefix_kernel_name = format!("{}_prefix_sum", G::name());
                let prefix_kernel = program.create_kernel(&prefix_kernel_name, 1, 1)?;

                prefix_kernel
                    .arg(&counts_buffer)
                    .arg(&offsets_buffer)
                    .arg(&nonempty_ids_buffer)
                    .arg(&num_nonempty_buffer)
                    .arg(&(total_buckets as u32))
                    .run()?;

                (offsets_buffer, nonempty_ids_buffer, num_nonempty_buffer)
            };

            // Step 6: Download num_nonempty
            let num_nonempty = {
                let _span = debug_span!("download_num_nonempty").entered();
                let mut num_nonempty = vec![0u32; 1];
                program.read_into_buffer(&num_nonempty_buffer, &mut num_nonempty)?;
                num_nonempty[0] as usize
            };

            // Step 7: Copy offsets for scatter (scatter will modify them via atomicAdd)
            let scatter_offsets_buffer = {
                let _span = debug_span!("copy_offsets_for_scatter").entered();
                let mut offsets = vec![0u32; total_buckets];
                program.read_into_buffer(&offsets_buffer, &mut offsets)?;
                program.create_buffer_from_slice(&offsets)?
            };

            // Step 8: Allocate sorted_values buffer and scatter
            let sorted_values_buffer = {
                let _span = debug_span!("scatter_to_sorted").entered();
                let sorted_len = n_bases * num_windows;
                // SAFETY: GPU will initialize this buffer
                let sorted_values_buffer = unsafe { program.create_buffer::<u32>(sorted_len)? };

                let scatter_global = div_ceil(n_bases * num_windows, LOCAL_WORK_SIZE);
                let scatter_kernel_name = format!("{}_scatter_to_sorted", G::name());
                let scatter_kernel = program.create_kernel(
                    &scatter_kernel_name, scatter_global, LOCAL_WORK_SIZE)?;

                let total_pairs = (n_bases * num_windows) as u32;
                scatter_kernel
                    .arg(&keys_buffer)
                    .arg(&values_buffer)
                    .arg(&scatter_offsets_buffer)
                    .arg(&sorted_values_buffer)
                    .arg(&total_pairs)
                    .run()?;

                sorted_values_buffer
            };

            // Step 9: Allocate and initialize bucket_results (identity points)
            let bucket_results_buffer = {
                let _span = debug_span!("allocate_bucket_results").entered();
                let identity_points = vec![<G::Group as AdditiveGroup>::ZERO; total_buckets];
                program.create_buffer_from_slice(&identity_points)?
            };

            // Step 10: Accumulate sorted buckets (with chunked dispatch for large buckets)
            {
                let _span = debug_span!("accumulate_sorted_buckets").entered();
                if num_nonempty > 0 {
                    // Download counts and nonempty IDs for dispatch table construction
                    let mut counts_cpu = vec![0u32; total_buckets];
                    program.read_into_buffer(&counts_buffer, &mut counts_cpu)?;
                    let mut nonempty_ids_cpu = vec![0u32; total_buckets];
                    program.read_into_buffer(&nonempty_ids_buffer, &mut nonempty_ids_cpu)?;

                    let (dispatch_table, reduce_table, num_dispatches) =
                        build_dispatch_tables(&counts_cpu, &nonempty_ids_cpu, num_nonempty);

                    if num_dispatches == num_nonempty {
                        // No large buckets — use simple 1-thread-per-bucket kernel (faster for small buckets)
                        let accum_global = div_ceil(num_nonempty, LOCAL_WORK_SIZE);
                        let accum_kernel = program.create_kernel(
                            &format!("{}_accumulate_sorted_buckets", G::name()),
                            accum_global, LOCAL_WORK_SIZE)?;
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
                        let chunked_global = div_ceil(num_dispatches, LOCAL_WORK_SIZE);
                        let chunked_kernel = program.create_kernel(
                            &format!("{}_accumulate_chunked", G::name()),
                            chunked_global, LOCAL_WORK_SIZE)?;
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
                        let reduce_global = div_ceil(num_nonempty, LOCAL_WORK_SIZE);
                        let reduce_kernel = program.create_kernel(
                            &format!("{}_reduce_partial_buckets", G::name()),
                            reduce_global, LOCAL_WORK_SIZE)?;
                        reduce_kernel
                            .arg(&partial_results_buffer)
                            .arg(&nonempty_ids_buffer)
                            .arg(&reduce_table_buffer)
                            .arg(&bucket_results_buffer)
                            .arg(&(num_nonempty as u32))
                            .run()?;
                    }
                }
            }

            // Step 11: Parallel chunked bucket reduction + combine to windows
            let window_results_buffer = {
                let _span = debug_span!("reduce_buckets_chunked").entered();
                const REDUCTION_CHUNK_SIZE: usize = 256;
                let num_chunks_per_window = div_ceil(buckets_per_window, REDUCTION_CHUNK_SIZE);
                let total_blocks = num_windows * num_chunks_per_window;
                let total_chunks = total_blocks;

                let chunk_sbp_buffer = unsafe { program.create_buffer::<G::Group>(total_chunks.max(1))? };
                let chunk_sum_buffer = unsafe { program.create_buffer::<G::Group>(total_chunks.max(1))? };

                // Level 1: Parallel suffix sum + reduction per chunk
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

                // Level 2: Combine chunks to window results
                let window_results = vec![<G::Group as AdditiveGroup>::ZERO; num_windows];
                let window_results_buffer = program.create_buffer_from_slice(&window_results)?;
                let combine_global = div_ceil(num_windows, LOCAL_WORK_SIZE);
                let combine_kernel = program.create_kernel(
                    &format!("{}_combine_chunks_to_windows", G::name()),
                    combine_global, LOCAL_WORK_SIZE)?;
                combine_kernel
                    .arg(&chunk_sbp_buffer)
                    .arg(&chunk_sum_buffer)
                    .arg(&window_results_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(num_chunks_per_window as u32))
                    .arg(&(REDUCTION_CHUNK_SIZE as u32))
                    .run()?;

                window_results_buffer
            };

            // Step 12: Final Horner reduction (single thread)
            let final_result_buffer = {
                let _span = debug_span!("reduce_windows").entered();
                let final_result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                let final_result_buffer = program.create_buffer_from_slice(&final_result)?;

                let reduce_windows_kernel_name = format!("{}_reduce_windows", G::name());
                let reduce_windows_kernel = program.create_kernel(
                    &reduce_windows_kernel_name, 1, 1)?;

                reduce_windows_kernel
                    .arg(&window_results_buffer)
                    .arg(&final_result_buffer)
                    .arg(&(num_windows as u32))
                    .arg(&(window_size as u32))
                    .arg(&(effective_bits as u32))
                    .run()?;

                final_result_buffer
            };

            // Step 13: Download final result
            let result = {
                let _span = debug_span!("download_result").entered();
                let mut result = vec![<G::Group as AdditiveGroup>::ZERO; 1];
                program.read_into_buffer(&final_result_buffer, &mut result)?;
                result[0]
            };

            Ok(result)
        });

        let result = self.program.run(closures, ())?;
        Ok(result)
    }
}

/// A struct that contains several multiexp kernels for different devices.
pub struct MultiexpKernel<'a, G>
where
    G: GpuAffine,
{
    kernels: Vec<SingleMultiexpKernel<'a, G>>,
}

impl<'a, G> MultiexpKernel<'a, G>
where
    G: GpuAffine,
{
    /// Create new kernels, one for each given device.
    pub fn create(programs: Vec<Program>, devices: &[&Device]) -> EcResult<Self> {
        Self::create_optional_abort(programs, devices, None)
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        programs: Vec<Program>,
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        Self::create_optional_abort(programs, devices, Some(maybe_abort))
    }

    fn create_optional_abort(
        programs: Vec<Program>,
        devices: &[&Device],
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let _span = debug_span!("multiexp_kernel_create").entered();
        let kernels: Vec<_> = programs
            .into_iter()
            .zip(devices.iter())
            .filter_map(|(program, device)| {
                let device_name = program.device_name().to_string();
                let kernel = SingleMultiexpKernel::create(program, device, maybe_abort);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}'! Error: {}",
                        device_name, e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(EcError::Simple("No working GPUs found!"));
        }
        info!("Multiexp: {} working device(s) selected.", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!(
                "Multiexp: Device {}: {} (Chunk-size: {})",
                i,
                k.program.device_name(),
                k.n
            );
        }
        Ok(MultiexpKernel { kernels })
    }

    /// Calculate multiexp on all available GPUs.
    ///
    /// It needs to run within a [`yastl::Scope`]. This method usually isn't called directly, use
    /// [`MultiexpKernel::multiexp`] instead.
    pub fn parallel_multiexp<'s>(
        &'s mut self,
        scope: &Scope<'s>,
        bases: &'s [G::GpuRepr],
        exps: &'s [<G::ScalarField as ark_ff::PrimeField>::BigInt],
        results: &'s mut [G::Group],
        error: Arc<RwLock<EcResult<()>>>,
    ) {
        let num_devices = self.kernels.len();
        let num_exps = exps.len();
        // The maximum number of exponentiations per device.
        let chunk_size = ((num_exps as f64) / (num_devices as f64)).ceil() as usize;

        for (((bases, exps), kern), result) in bases
            .chunks(chunk_size)
            .zip(exps.chunks(chunk_size))
            // NOTE vmx 2021-11-17: This doesn't need to be a mutable iterator. But when it isn't
            // there will be errors that the OpenCL CommandQueue cannot be shared between threads
            // safely.
            .zip(self.kernels.iter_mut())
            .zip(results.iter_mut())
        {
            let error = error.clone();
            scope.execute(move || {
                let mut acc = <G::Group as AdditiveGroup>::ZERO;
                for (bases, exps) in bases.chunks(kern.n).zip(exps.chunks(kern.n)) {
                    if error.read().unwrap().is_err() {
                        break;
                    }
                    match kern.multiexp(bases, exps) {
                        Ok(result) => acc.add_assign(&result),
                        Err(e) => {
                            *error.write().unwrap() = Err(e);
                            break;
                        }
                    }
                }
                if error.read().unwrap().is_ok() {
                    *result = acc;
                }
            });
        }
    }

    /// Calculate multiexp.
    ///
    /// This is the main entry point.
    pub fn multiexp(
        &mut self,
        pool: &Worker,
        bases_arc: Arc<Vec<G::GpuRepr>>,
        exps: Arc<Vec<<G::ScalarField as ark_ff::PrimeField>::BigInt>>,
        skip: usize,
    ) -> EcResult<G::Group> {
        let _span = debug_span!("multiexp", n = exps.len()).entered();
        // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
        // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
        let bases = &bases_arc[skip..(skip + exps.len())];
        let exps = &exps[..];

        let mut results = Vec::new();
        let error = Arc::new(RwLock::new(Ok(())));

        pool.scoped(|s| {
            results = vec![<G::Group as AdditiveGroup>::ZERO; self.kernels.len()];
            self.parallel_multiexp(s, bases, exps, &mut results, error.clone());
        });

        Arc::try_unwrap(error)
            .expect("only one ref left")
            .into_inner()
            .unwrap()?;

        let mut acc = <G::Group as AdditiveGroup>::ZERO;
        for r in results {
            acc.add_assign(&r);
        }

        Ok(acc)
    }

    /// Returns the number of kernels (one per device).
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }
}

/// Parameters for running sort-based MSM within an existing GPU session.
/// This is used by gpu_buffer.rs fused paths where bases and digits are already on GPU.
pub struct SortedMsmParams {
    /// Number of bases in the MSM.
    pub n_bases: usize,
    /// Number of windows for scalar decomposition.
    pub num_windows: usize,
    /// Number of buckets per window (2^(window_size-1) for signed digits).
    pub buckets_per_window: usize,
    /// Window size in bits.
    pub window_size: usize,
    /// Actual bit length needed for the scalars.
    pub effective_bits: usize,
}

/// Maximum number of points per chunk when splitting large buckets.
/// Buckets with more points than this are split across multiple threads.
pub const CHUNK_SIZE: usize = 256;

/// Build dispatch and reduce tables for chunked bucket accumulation.
///
/// For each non-empty bucket, if it has <= CHUNK_SIZE points, create one dispatch entry.
/// If it has > CHUNK_SIZE points, split it into ceil(count/CHUNK_SIZE) chunks.
///
/// Returns (dispatch_table, reduce_table, num_dispatches) where:
/// - dispatch_table: Vec<u32> with [bucket_id, chunk_start, chunk_count] per dispatch
/// - reduce_table: Vec<u32> with [partial_start, num_partials] per non-empty bucket
/// - num_dispatches: total number of dispatch entries
pub fn build_dispatch_tables(
    bucket_sizes: &[u32],
    nonempty_bucket_ids: &[u32],
    num_nonempty: usize,
) -> (Vec<u32>, Vec<u32>, usize) {
    let mut dispatch_table = Vec::new();
    let mut reduce_table = Vec::with_capacity(num_nonempty * 2);
    let mut dispatch_idx = 0;

    for &bid in &nonempty_bucket_ids[..num_nonempty] {
        let count = bucket_sizes[bid as usize];
        let num_chunks = div_ceil(count as usize, CHUNK_SIZE);
        let partial_start = dispatch_idx;

        for chunk in 0..num_chunks {
            let chunk_start = chunk * CHUNK_SIZE;
            let chunk_count = std::cmp::min(CHUNK_SIZE, count as usize - chunk_start);
            dispatch_table.push(bid);
            dispatch_table.push(chunk_start as u32);
            dispatch_table.push(chunk_count as u32);
            dispatch_idx += 1;
        }

        reduce_table.push(partial_start as u32);
        reduce_table.push(num_chunks as u32);
    }

    (dispatch_table, reduce_table, dispatch_idx)
}

/// Computes MSM parameters for the sort-based approach.
///
/// This determines optimal window size and derived parameters based on the number
/// of bases. The window size calculation balances parallelism and memory usage.
pub fn compute_sorted_msm_params(n_bases: usize, effective_bits: usize) -> SortedMsmParams {
    // For sort-based MSM, each bucket is processed by 1 thread serially.
    // Target ~1000 pts/bucket: ws ≈ log2(n) - 8.
    let window_size = {
        let ws = std::cmp::max(3, (n_bases as f64).log2() as usize - 8);
        std::cmp::min(ws, MAX_WINDOW_SIZE)
    };

    let num_windows = div_ceil(effective_bits + 1, window_size);
    let buckets_per_window = 1 << (window_size - 1); // signed digit

    SortedMsmParams {
        n_bases,
        num_windows,
        buckets_per_window,
        window_size,
        effective_bits,
    }
}
