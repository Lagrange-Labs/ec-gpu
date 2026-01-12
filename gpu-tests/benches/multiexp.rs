use std::sync::Arc;

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{PrimeField, UniformRand};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ec_gpu_gen::{
    multiexp::{G1AffineM, MultiexpKernel},
    rust_gpu_tools::Device,
    threadpool::Worker,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// The power that will be used to define the maximum number of elements. The number of elements
/// is `2^MAX_ELEMENTS_POWER`.
const MAX_ELEMENTS_POWER: usize = 20;
/// The maximum number of elements for this benchmark.
const MAX_ELEMENTS: usize = 1 << MAX_ELEMENTS_POWER;

pub trait QueryDensity: Sized {
    /// Returns whether the base exists.
    type Iter: Iterator<Item = bool>;

    fn iter(self) -> Self::Iter;
    fn get_query_size(self) -> Option<usize>;
    fn generate_exps<F: PrimeField>(self, exponents: Arc<Vec<F::BigInt>>) -> Arc<Vec<F::BigInt>>;
}

#[derive(Clone)]
pub struct FullDensity;

impl AsRef<FullDensity> for FullDensity {
    fn as_ref(&self) -> &FullDensity {
        self
    }
}

impl QueryDensity for &FullDensity {
    type Iter = std::iter::Repeat<bool>;

    fn iter(self) -> Self::Iter {
        std::iter::repeat(true)
    }

    fn get_query_size(self) -> Option<usize> {
        None
    }

    fn generate_exps<F: PrimeField>(self, exponents: Arc<Vec<F::BigInt>>) -> Arc<Vec<F::BigInt>> {
        exponents
    }
}

fn fq_to_32_le(x: &Fq) -> [u8; 32] {
    // Extract raw Montgomery representation directly
    // Arkworks stores Fq as 4 u64 limbs in Montgomery form
    let limbs = unsafe { std::mem::transmute::<Fq, [u64; 4]>(*x) };

    let mut out = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        let bytes = limb.to_le_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
    }
    out
}

fn g1_xy_bytes_le(p: &G1Affine) -> Option<([u8; 32], [u8; 32])> {
    p.xy().map(|(x, y)| (fq_to_32_le(&x), fq_to_32_le(&y)))
}

fn bench_multiexp(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("multiexp");
    // The difference between runs is so little, hence a low sample size is OK.
    group.sample_size(10);

    let devices = Device::all();

    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G1Affine>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let max_bases: Vec<_> = (0..MAX_ELEMENTS)
        .into_par_iter()
        .map(|_| G1Projective::rand(&mut rand::thread_rng()).into_affine())
        .collect();
    let max_exponents: Vec<_> = (0..MAX_ELEMENTS)
        .into_par_iter()
        .map(|_| Fr::rand(&mut rand::thread_rng()))
        .collect();

    let num_elements: Vec<_> = (10..MAX_ELEMENTS_POWER).map(|shift| 1 << shift).collect();
    for num in num_elements {
        group.bench_with_input(BenchmarkId::from_parameter(num), &num, |bencher, &_num| {
            // Convert arkworks affine points to GPU format (G1AffineM)
            let bases_gpu: Vec<G1AffineM> = max_bases
                .iter()
                .map(|affine| {
                    let (x, y) = g1_xy_bytes_le(affine).expect("point not at infinity");
                    G1AffineM { x, y }
                })
                .collect();
            let bases_gpu = Arc::new(bases_gpu);

            // Convert scalars to BigInt for density map processing
            let exps_bigint: Arc<Vec<_>> =
                Arc::new(max_exponents.iter().map(|e| e.into_bigint()).collect());

            let exps = FullDensity.as_ref().generate_exps::<Fr>(exps_bigint);

            bencher.iter(|| {
                let _ = black_box(
                    kern.multiexp(&pool, bases_gpu.clone(), exps.clone(), 0)
                        .unwrap(),
                );
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_multiexp);
criterion_main!(benches);
