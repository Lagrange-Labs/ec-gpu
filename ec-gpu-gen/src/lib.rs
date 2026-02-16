#![warn(missing_docs)]
//! CUDA code generator for finite-field arithmetic over prime fields and elliptic curve
//! arithmetic constructed with Rust.
//!
//! There is also support for Fast Fourier Transform and Multiexponentiation.
//!
//! This crate creates GPU kernels at compile-time. CUDA generates a [fatbin] that is embedded
//! into the binary.
//!
//! In order to make things easier to use, there are helper functions available. You would put some
//! code into `build.rs`, that generates the kernels, and some code into your library which then
//! consumes those generated kernels. The kernels will be directly embedded into your program/library.
//! If something goes wrong, you will get an error at compile-time.
//!
//! In this example we will make use of the FFT functionality. Add to your `build.rs`:
//!
//! ```no_run
//! use ark_bn254::Fr;
//! use ec_gpu_gen::SourceBuilder;
//!
//! let source_builder = SourceBuilder::new().add_fft::<Fr>();
//! ec_gpu_gen::generate(&source_builder);
//! ```
//!
//! The `ec_gpu_gen::generate()` takes care of the actual code generation/compilation. It will
//! create a CUDA kernel. It defines the `_EC_GPU_CUDA_KERNEL_FATBIN` environment variable
//! that points to the compiled CUDA kernel.
//!
//! Those variables are then picked up by the `ec_gpu_gen::program!()` macro, which generates a
//! program, for a given GPU device. Using FFT within your library would then look like this:
//!
//! ```ignore
//! use ark_bn254::Fr;
//! use ec_gpu_gen::{fft::FftKernel, rust_gpu_tools::Device};
//!
//! let devices = Device::all();
//! let programs = devices
//!     .iter()
//!     .map(|device| ec_gpu_gen::program!(device))
//!     .collect::<Result<_, _>>()
//!     .expect("Cannot create programs!");
//!
//! let mut kern = FftKernel::<Fr>::create(programs).expect("Cannot initialize kernel!");
//! kern.radix_fft_many(&mut [&mut coeffs], &[omega], &[log_d]).expect("GPU FFT failed!");
//! ```
//!
//! Feature flags
//! -------------
//!
//! CUDA is supported, enabled with the `cuda` [feature flag].
//!
//! [fatbin]: https://en.wikipedia.org/wiki/Fat_binary#Heterogeneous_computing
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section
mod error;
#[cfg(feature = "cuda")]
mod program;
mod source;

/// Fast Fourier Transform on the GPU.
#[cfg(feature = "cuda")]
pub mod fft;
/// Fast Fourier Transform on the CPU.
pub mod fft_cpu;
/// Multiexponentiation on the GPU.
#[cfg(feature = "cuda")]
pub mod multiexp;
/// Polynomial operations on the GPU (fix_var, linear_combine, witness_poly).
#[cfg(feature = "cuda")]
pub mod poly_ops;
/// GPU buffer management and combined operations for persistent data.
#[cfg(feature = "cuda")]
pub mod gpu_buffer;
/// Helpers for multithreaded code.
pub mod threadpool;

/// Re-export rust-gpu-tools as things like [`rust_gpu_tools::Device`] might be needed.
#[cfg(feature = "cuda")]
pub use rust_gpu_tools;

pub use error::{EcError, EcResult};
pub use source::{generate, SourceBuilder};

#[cfg(feature = "cuda")]
pub use fft::{FftKernel, FftKernelArk, SingleFftKernel, SingleFftKernelArk};
#[cfg(feature = "cuda")]
pub use multiexp::{G1AffineM, G2AffineM, GpuAffine, MultiexpKernel, SingleMultiexpKernel, compute_work_units, SortedMsmParams, compute_sorted_msm_params, build_dispatch_tables, CHUNK_SIZE};
#[cfg(feature = "cuda")]
pub use poly_ops::{PolyOpsKernel, SinglePolyOpsKernel};
#[cfg(feature = "cuda")]
pub use gpu_buffer::{CombinedPolyOps, GpuBufferCache, GpuBufferId, BufferMetadata, FixVarsResult, FusedPolyCommit, FixVarsAndCommitResult, Phase3Input, FusedOpenResult};
