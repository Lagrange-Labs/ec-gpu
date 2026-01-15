//! Benchmark comparing MSM performance with full-size vs small scalars.
//!
//! This demonstrates the speedup from the small scalar optimization:
//! when scalars only use 64 bits, we skip processing the upper 190+ zero bits.
//!
//! Run with: cargo bench --bench small_scalars

fn main() {
    #[cfg(any(feature = "cuda", feature = "opencl"))]
    divan::main();

    #[cfg(not(any(feature = "cuda", feature = "opencl")))]
    println!("Benchmarks require cuda or opencl feature");
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
use std::sync::Arc;

#[cfg(any(feature = "cuda", feature = "opencl"))]
use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use ark_ec::CurveGroup;
#[cfg(any(feature = "cuda", feature = "opencl"))]
use ark_ff::{PrimeField, UniformRand};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use divan::{black_box, Bencher};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use ec_gpu_gen::{
    multiexp::{G1AffineM, MultiexpKernel},
    rust_gpu_tools::Device,
    threadpool::Worker,
};
#[cfg(any(feature = "cuda", feature = "opencl"))]
use rand::Rng;

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn fq_to_32_le(x: &Fq) -> [u8; 32] {
    let limbs = unsafe { std::mem::transmute::<Fq, [u64; 4]>(*x) };
    let mut out = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        let bytes = limb.to_le_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }
    out
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn g1_xy_bytes_le(p: &G1Affine) -> Option<([u8; 32], [u8; 32])> {
    use ark_ec::AffineRepr;
    p.xy().map(|(x, y)| (fq_to_32_le(&x), fq_to_32_le(&y)))
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
const NUM_POINTS: usize = 1 << 16; // 65536 points

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[divan::bench]
fn msm_full_254bit_scalars(bencher: Bencher) {
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G1Affine>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    // Generate random bases
    let bases: Vec<G1Affine> = (0..NUM_POINTS)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let bases_gpu: Vec<G1AffineM> = bases
        .iter()
        .map(|affine| {
            let (x, y) = g1_xy_bytes_le(affine).expect("point not at infinity");
            G1AffineM { x, y }
        })
        .collect();
    let bases_gpu = Arc::new(bases_gpu);

    // Generate full 254-bit random scalars
    let full_scalars: Vec<Fr> = (0..NUM_POINTS).map(|_| Fr::rand(&mut rng)).collect();
    let full_exps: Arc<Vec<_>> = Arc::new(full_scalars.iter().map(|e| e.into_bigint()).collect());

    bencher.bench_local(|| {
        black_box(
            kern.multiexp(&pool, bases_gpu.clone(), full_exps.clone(), 0)
                .unwrap(),
        )
    });
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
#[divan::bench]
fn msm_small_64bit_scalars(bencher: Bencher) {
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G1Affine>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    // Generate random bases
    let bases: Vec<G1Affine> = (0..NUM_POINTS)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    let bases_gpu: Vec<G1AffineM> = bases
        .iter()
        .map(|affine| {
            let (x, y) = g1_xy_bytes_le(affine).expect("point not at infinity");
            G1AffineM { x, y }
        })
        .collect();
    let bases_gpu = Arc::new(bases_gpu);

    // Generate small 64-bit scalars (0 to 2^64)
    let small_scalars: Vec<Fr> = (0..NUM_POINTS)
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect();
    let small_exps: Arc<Vec<_>> = Arc::new(small_scalars.iter().map(|e| e.into_bigint()).collect());

    bencher.bench_local(|| {
        black_box(
            kern.multiexp(&pool, bases_gpu.clone(), small_exps.clone(), 0)
                .unwrap(),
        )
    });
}
