#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::sync::Arc;
use std::time::Instant;

use ark_bn254::{Fr, G1Projective};
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::{PrimeField, UniformRand};
use ec_gpu::arkworks_bn254::{G1Affine, G2Affine};
use ec_gpu_gen::multiexp::GpuAffine;
use ec_gpu_gen::{
    multiexp::MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker, EcError,
};
use tracing::debug_span;
use tracing_profile::{PrintTreeConfig, PrintTreeLayer};
use tracing_subscriber::{filter::filter_fn, prelude::*};

pub trait QueryDensity: Sized {
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

fn multiexp_gpu<G, Q, D>(
    pool: &Worker,
    bases: Arc<Vec<G>>,
    density_map: D,
    exponents: Arc<Vec<Fr>>,
    kern: &mut MultiexpKernel<G>,
) -> Result<G::Group, EcError>
where
    G: GpuAffine<ScalarField = Fr>,
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
{
    let bases_gpu: Vec<G::GpuRepr> = bases.iter().map(|affine| affine.to_gpu()).collect();
    let exps_bigint: Arc<Vec<_>> = Arc::new(exponents.iter().map(|e| e.into_bigint()).collect());
    let exps = density_map.as_ref().generate_exps::<Fr>(exps_bigint);
    kern.multiexp(pool, Arc::new(bases_gpu), exps, 0)
}

/// Trait to bridge our newtype wrappers with arkworks types for testing
trait TestableAffine: GpuAffine<ScalarField = Fr> + From<Self::ArkAffine> {
    type ArkAffine: ark_ec::AffineRepr<ScalarField = Fr> + Copy;
    type ArkProjective: CurveGroup<Affine = Self::ArkAffine, ScalarField = Fr> + UniformRand;

    fn group_name() -> &'static str;
}

impl TestableAffine for G1Affine {
    type ArkAffine = ark_bn254::G1Affine;
    type ArkProjective = ark_bn254::G1Projective;

    fn group_name() -> &'static str {
        "G1"
    }
}

impl TestableAffine for G2Affine {
    type ArkAffine = ark_bn254::G2Affine;
    type ArkProjective = ark_bn254::G2Projective;

    fn group_name() -> &'static str {
        "G2"
    }
}

fn gpu_multiexp_consistency_test<G>(start_log_d: usize, max_log_d: usize)
where
    G: TestableAffine,
    G::Group: PartialEq<G::ArkProjective>,
{
    fil_logger::maybe_init();

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| crate::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();

    let mut bases_ark: Vec<G::ArkAffine> = (0..(1 << start_log_d))
        .map(|_| G::ArkProjective::rand(&mut rng).into_affine())
        .collect();

    for log_d in start_log_d..=max_log_d {
        let bases: Vec<G> = bases_ark.iter().map(|p| G::from(*p)).collect();
        let g = Arc::new(bases);

        let samples = 1 << log_d;
        println!(
            "Testing {} Multiexp for {} elements...",
            G::group_name(),
            samples
        );

        let v: Vec<Fr> = (0..samples).map(|_| Fr::rand(&mut rng)).collect();
        let v_arc = Arc::new(v.clone());

        let mut now = Instant::now();
        let gpu = multiexp_gpu(&pool, g.clone(), FullDensity, v_arc.clone(), &mut kern).unwrap();
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        let cpu: G::ArkProjective =
            VariableBaseMSM::msm(bases_ark.as_slice(), v.as_slice()).unwrap();
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(
            gpu,
            cpu,
            "GPU and CPU results differ for {} MSM",
            G::group_name()
        );

        println!("============================");

        bases_ark = [bases_ark.clone(), bases_ark.clone()].concat();
    }
}

#[test]
fn gpu_multiexp_g1_consistency() {
    gpu_multiexp_consistency_test::<G1Affine>(10, 16);
}

#[test]
fn gpu_multiexp_g2_consistency() {
    gpu_multiexp_consistency_test::<G2Affine>(10, 16);
}

/// Test that the small scalar optimization works correctly.
/// When scalars are small (e.g., 64-bit), the optimization should skip
/// processing upper zero bits while producing correct results.
#[test]
fn gpu_multiexp_small_scalars() {
    fil_logger::maybe_init();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| crate::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G1Affine>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();
    use ark_ff::UniformRand;
    use rand::Rng;

    // Test with small scalars (64-bit range)
    let num_points = 1 << 16;
    println!(
        "Testing small scalar MSM optimization with {} points...",
        num_points
    );

    let bases_ark: Vec<ark_bn254::G1Affine> = (0..num_points)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Generate small scalars (only 64 bits used out of 254)
    let small_scalars: Vec<Fr> = (0..num_points)
        .map(|_| Fr::from(rng.gen::<u64>()))
        .collect();

    let bases: Vec<G1Affine> = bases_ark.iter().map(|p| G1Affine::from(*p)).collect();
    let g = Arc::new(bases);
    let v_arc: Arc<Vec<_>> = Arc::new(small_scalars.clone());

    let now = Instant::now();
    let gpu: G1Projective =
        multiexp_gpu(&pool, g.clone(), FullDensity, v_arc.clone(), &mut kern).unwrap();
    let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
    println!("Small scalar GPU MSM took {}ms.", gpu_dur);

    let cpu: G1Projective =
        VariableBaseMSM::msm(bases_ark.as_slice(), small_scalars.as_slice()).unwrap();

    assert_eq!(cpu, gpu, "Small scalar MSM mismatch!");
    println!("Small scalar MSM test passed!");
}

/// Test edge case with very small scalars (32-bit)
#[test]
fn gpu_multiexp_very_small_scalars() {
    fil_logger::maybe_init();
    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| crate::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern =
        MultiexpKernel::<G1Affine>::create(programs, &devices).expect("Cannot initialize kernel!");
    let pool = Worker::new();

    let mut rng = rand::thread_rng();
    use ark_ff::UniformRand;
    use rand::Rng;

    let num_points = 1 << 14;
    println!(
        "Testing very small scalar (32-bit) MSM with {} points...",
        num_points
    );

    let bases_ark: Vec<ark_bn254::G1Affine> = (0..num_points)
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect();

    // Generate very small scalars (only 32 bits used)
    let small_scalars: Vec<Fr> = (0..num_points)
        .map(|_| Fr::from(rng.gen::<u32>() as u64))
        .collect();

    let bases: Vec<G1Affine> = bases_ark.iter().map(|p| G1Affine::from(*p)).collect();
    let g = Arc::new(bases);
    let v_arc: Arc<Vec<_>> = Arc::new(small_scalars.clone());

    let now = Instant::now();
    let gpu: G1Projective =
        multiexp_gpu(&pool, g.clone(), FullDensity, v_arc.clone(), &mut kern).unwrap();
    let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
    println!("Very small scalar GPU MSM took {}ms.", gpu_dur);

    let cpu: G1Projective =
        VariableBaseMSM::msm(bases_ark.as_slice(), small_scalars.as_slice()).unwrap();

    assert_eq!(cpu, gpu, "Very small scalar MSM mismatch!");
    println!("Very small scalar MSM test passed!");
}

#[test]
fn gpu_multiexp_profile() {
    use ec_gpu_gen::multiexp::SingleMultiexpKernel;

    let config = PrintTreeConfig {
        hide_below_percent: 0.0,
        accumulate_events: false,
        ..PrintTreeConfig::default()
    };
    let (layer, _guard) = PrintTreeLayer::new(config);
    // Filter out events, only keep spans
    let layer = layer.with_filter(filter_fn(|metadata| metadata.is_span()));
    tracing_subscriber::registry().with(layer).init();

    let root = debug_span!("profile_multiexp");
    let _root_guard = root.enter();

    let devices = Device::all();
    let device = &devices[0];
    let program = crate::program!(device).expect("Cannot create program!");

    let kern = {
        let span = debug_span!("create_kernel");
        let _guard = span.enter();
        SingleMultiexpKernel::<G1Affine>::create(program, device, None)
            .expect("Cannot initialize kernel!")
    };

    let mut rng = rand::thread_rng();
    let log_n = 16;
    let n = 1 << log_n;

    let bases_ark: Vec<ark_bn254::G1Affine> = {
        let span = debug_span!("generate_bases", n = n);
        let _guard = span.enter();
        (0..n)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect()
    };

    let bases_gpu: Vec<_> = {
        let span = debug_span!("convert_bases_to_gpu", n = n);
        let _guard = span.enter();
        bases_ark
            .iter()
            .map(|p| G1Affine::from(*p).to_gpu())
            .collect()
    };

    let exponents: Vec<_> = {
        let span = debug_span!("generate_exponents", n = n);
        let _guard = span.enter();
        (0..n).map(|_| Fr::rand(&mut rng).into_bigint()).collect()
    };

    // Run multiexp on main thread - this will show all nested spans
    let _result = {
        let span = debug_span!("run_multiexp", n = n);
        let _guard = span.enter();
        kern.multiexp(&bases_gpu, &exponents)
            .expect("multiexp failed")
    };

    drop(_root_guard);
}
