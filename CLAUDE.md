# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build the workspace (requires CUDA toolkit or OpenCL dev libraries)
cargo build --workspace

# Build with specific GPU backend
cargo build --workspace --features cuda
cargo build --workspace --features opencl
cargo build --workspace --features cuda,opencl

# Run tests (requires GPU hardware for cuda/opencl features)
cargo test --workspace --no-default-features  # CPU-only tests
cargo test --workspace                         # With default GPU features

# Run a specific test
cargo test --package gpu-tests gpu_multiexp_consistency

# Run benchmarks
cargo bench --package gpu-tests

# Check formatting and linting
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features cuda,opencl -- -D warnings
```

## Architecture

This workspace generates GPU kernels for finite-field and elliptic curve arithmetic at compile time.

### Crates

- **ec-gpu**: Traits (`GpuField`, `GpuName`) that field/curve types must implement for GPU code generation. Contains arkworks BN254 implementations.
- **ec-gpu-gen**: The code generator. Contains GPU kernel source templates (`.cl` files) and Rust APIs for FFT and MSM operations.
- **gpu-tests**: Integration tests and benchmarks. Has a `build.rs` that generates actual kernels.

### Code Generation Flow

1. `build.rs` uses `SourceBuilder` to configure which fields/curves to include
2. `SourceBuilder::add_fft::<Fr>()` or `add_multiexp::<G1Affine, Fq, Fr>()` registers types
3. `ec_gpu_gen::generate()` compiles CUDA fatbin or generates OpenCL source
4. Runtime code uses `program!(device)` macro to load the compiled kernel

### GPU Kernel Sources (ec-gpu-gen/src/cl/)

- **common.cl**: Platform abstraction (CUDA/OpenCL), PTX intrinsics for carry chains
- **field.cl**: Montgomery arithmetic (add, sub, mul, sqr, pow). CUDA path uses optimized Niall Emmart reduction
- **field2.cl**: Extension field (Fq2) operations
- **ec.cl**: Elliptic curve point operations (Jacobian coordinates): double, add, add_mixed
- **multiexp.cl**: Pippenger bucket method MSM kernel
- **fft.cl**: Radix-2 FFT kernel

### Key Constants in multiexp.rs

- `MAX_WINDOW_SIZE = 10`: Maximum bits per window in Pippenger
- `LOCAL_WORK_SIZE = 128`: CUDA blocks per grid
- `MEMORY_PADDING = 0.2`: Reserve 20% GPU memory headroom
- Window size computed dynamically based on number of terms and work units

### Field Representation

- Uses 32-bit limbs on CUDA, 64-bit limbs on OpenCL
- BN254 scalar field Fr uses 8 x 32-bit limbs (256 bits)
- Montgomery form throughout GPU operations
- `G1AffineM` struct: x,y coordinates as `[u8; 32]` in little-endian Montgomery form

## Environment Variables

- `EC_GPU_CUDA_NVCC_ARGS`: Override nvcc compilation flags
- `EC_GPU_FRAMEWORK`: Choose `cuda` or `opencl` at runtime when both compiled
- `EC_GPU_NUM_THREADS`: Limit CPU thread pool size
