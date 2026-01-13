#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    use ark_bn254::{Fq, Fr, G1Affine};
    use ec_gpu_gen::SourceBuilder;

    let source_builder = SourceBuilder::new()
        .add_fft::<Fr>()
        .add_multiexp::<G1Affine, Fq, Fr>();
    ec_gpu_gen::generate(&source_builder);
}
