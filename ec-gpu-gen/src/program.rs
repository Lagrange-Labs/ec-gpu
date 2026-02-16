#[macro_export]
/// Helper macro to create a program for a device.
///
/// It will embed the CUDA fatbin within your binary. The source needs to be
/// generated via [`crate::source::generate`] in your `build.rs`.
///
/// It returns a `[crate::rust_gpu_tools::Program`] instance.
macro_rules! program {
    ($device:ident) => {{
        use $crate::rust_gpu_tools::{Framework, GPUError, Program};
        (|device: &Device| -> Result<Program, $crate::EcError> {
            let default_framework = device.framework();
            let framework = match ::std::env::var("EC_GPU_FRAMEWORK") {
                Ok(env) => match env.as_ref() {
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        {
                            Framework::Cuda
                        }

                        #[cfg(not(feature = "cuda"))]
                        return Err($crate::EcError::Simple("CUDA framework is not supported, please compile with the `cuda` feature enabled."))
                    }
                    _ => default_framework,
                },
                Err(_) => default_framework,
            };

            match framework {
                #[cfg(feature = "cuda")]
                Framework::Cuda => {
                    let kernel = include_bytes!(env!("_EC_GPU_CUDA_KERNEL_FATBIN"));
                    let cuda_device = device.cuda_device().ok_or(GPUError::DeviceNotFound)?;
                    let program = $crate::rust_gpu_tools::cuda::Program::from_bytes(cuda_device, kernel)?;
                    Ok(Program::Cuda(program))
                }
            }
        })($device)
    }};
}
