use std::ops::AddAssign;
use std::sync::{Arc, RwLock};

use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, BigInteger, PrimeField};
use ec_gpu::GpuName;
use log::{error, info};
use rust_gpu_tools::{program_closures, Device, Program};
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
const MAX_WINDOW_SIZE: usize = 10;
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
fn calc_chunk_size<G>(mem: u64, work_units: usize) -> usize
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
    // The amount of memory (in bytes) we need for the intermediate steps (buckets).
    let buckets_size = work_units * max_buckets_per_work_unit * proj_size;
    // The amount of memory (in bytes) we need for the results.
    let results_size = work_units * proj_size;

    (max_memory - buckets_size - results_size) / term_size
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
        let chunk_size = calc_chunk_size::<G>(mem, work_units);

        Ok(SingleMultiexpKernel {
            program,
            n: chunk_size,
            work_units,
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
