//! Abstraction layer for OpenCL and CUDA.
//!
//! Feature flags
//! -------------
//!
//! There are two [feature flags], `cuda` and `opencl`. By default `opencl` is enabled. You can
//! enable both at the same time. At least one of them needs to be enabled at any time.
//!
//! [feature flags]: https://doc.rust-lang.org/cargo/reference/manifest.html#the-features-section

#![warn(missing_docs)]

mod device;
mod error;
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod program;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;

pub use device::{Device, DeviceUuid, Framework, PciId, UniqueId, Vendor};
pub use error::GPUError;
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use program::Program;

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
compile_error!("At least one of the features `cuda` or `opencl` must be enabled.");

/// A GPU buffer that persists across `program.run()` calls.
///
/// Wraps backend-specific buffer types so that it can be used as a kernel argument
/// transparently inside `program_closures!`. This enables uploading large data (e.g. SRS bases)
/// once and reusing across multiple GPU sessions.
pub enum PersistentBuffer<T> {
    /// CUDA persistent buffer.
    #[cfg(feature = "cuda")]
    Cuda(cuda::Buffer<T>),
    /// OpenCL persistent buffer.
    #[cfg(feature = "opencl")]
    Opencl(opencl::Buffer<T>),
}

// SAFETY: PersistentBuffer wraps GPU device memory handles (pointers/IDs) that are safe to
// send between threads. The GPU memory itself is not thread-local. Actual GPU operations
// require the correct context to be active (handled by push_context/pop_context).
unsafe impl<T> Send for PersistentBuffer<T> {}

/// A GPU stream that persists across `program.run()` calls.
///
/// Wraps backend-specific stream types so that multiple CUDA/OpenCL streams can be
/// stored on a struct and reused across GPU sessions. The caller must ensure
/// `push_context()` is called before dropping on CUDA.
pub enum PersistentStream {
    /// CUDA persistent stream.
    #[cfg(feature = "cuda")]
    Cuda(cuda::Stream),
    /// OpenCL persistent stream.
    #[cfg(feature = "opencl")]
    Opencl(opencl::Stream),
}

impl PersistentStream {
    /// Synchronize the stream, waiting for all pending operations to complete.
    pub fn synchronize(&self) -> Result<(), GPUError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(s) => s.synchronize().map_err(Into::into),
            #[cfg(feature = "opencl")]
            Self::Opencl(s) => s.synchronize().map_err(Into::into),
        }
    }
}

// SAFETY: PersistentStream wraps GPU stream handles that are safe to send between threads.
unsafe impl Send for PersistentStream {}

/// Page-locked (pinned) host memory that persists across `program.run()` calls.
///
/// Wraps backend-specific pinned host buffer types. Pinned memory enables truly
/// async GPU transfers — without pinning, each `cuMemcpyHtoDAsync` implicitly
/// syncs the stream (~2-3ms block per call). The caller must ensure
/// `push_context()` is called before dropping on CUDA.
pub enum PersistentPinnedBuffer<T> {
    /// CUDA persistent pinned buffer (page-locked memory via cuMemAllocHost).
    #[cfg(feature = "cuda")]
    Cuda(cuda::PinnedHostBuffer<T>),
    /// OpenCL persistent pinned buffer (regular heap allocation fallback).
    #[cfg(feature = "opencl")]
    Opencl(opencl::PinnedHostBuffer<T>),
}

impl<T> std::ops::Deref for PersistentPinnedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b,
            #[cfg(feature = "opencl")]
            Self::Opencl(b) => b,
        }
    }
}

impl<T> std::ops::DerefMut for PersistentPinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match self {
            #[cfg(feature = "cuda")]
            Self::Cuda(b) => b,
            #[cfg(feature = "opencl")]
            Self::Opencl(b) => b,
        }
    }
}

// SAFETY: PersistentPinnedBuffer wraps page-locked host memory handles safe to send between threads.
unsafe impl<T> Send for PersistentPinnedBuffer<T> {}

/// A buffer on the GPU.
///
/// The concept of a local buffer is from OpenCL. In CUDA you don't allocate a buffer directly
/// via API call. Instead you pass in the amount of shared memory that should be used.
///
/// There can be at most a single local buffer per kernel. On CUDA a null pointer will be passed
/// in, instead of an actual value. The memory that should get allocated is then passed into the
/// kernel call automatically.
#[derive(Debug)]
pub struct LocalBuffer<T> {
    /// The number of T sized elements.
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> LocalBuffer<T> {
    /// Returns a new buffer of the specified `length`.
    pub fn new(length: usize) -> Self {
        LocalBuffer::<T> {
            length,
            _phantom: std::marker::PhantomData,
        }
    }
}
