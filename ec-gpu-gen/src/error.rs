use std::io;

#[cfg(feature = "cuda")]
use rust_gpu_tools::GPUError;

/// Errors of this library.
#[derive(thiserror::Error, Debug)]
pub enum EcError {
    /// A simple error that is described by a string.
    #[error("EcError: {0}")]
    Simple(&'static str),

    /// Error in case a GPU kernel execution was aborted.
    #[cfg(feature = "cuda")]
    #[error("GPU call was aborted!")]
    Aborted,

    /// An error that is bubbled up from the rust-gpu-tools library.
    #[cfg(feature = "cuda")]
    #[error("GPU tools error: {0}")]
    GpuTools(#[from] GPUError),

    /// IO error.
    #[error("Encountered an I/O error: {0}")]
    Io(#[from] io::Error),
}

/// Result wrapper that is always using [`EcError`] as error.
pub type EcResult<T> = std::result::Result<T, EcError>;
