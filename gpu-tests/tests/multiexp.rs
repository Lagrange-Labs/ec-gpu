#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::sync::Arc;
use std::time::Instant;

use ark_bn254::{Fq, Fr, G1Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::PrimeField;
use ec_gpu_gen::multiexp::G1AffineM;
use ec_gpu_gen::{
    multiexp::MultiexpKernel, program, rust_gpu_tools::Device, threadpool::Worker, EcError,
};

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

fn multiexp_gpu<Q, D>(
    pool: &Worker,
    bases: Arc<Vec<G1Affine>>,
    density_map: D,
    exponents: Arc<Vec<Fr>>,
    kern: &mut MultiexpKernel<G1Affine>,
) -> Result<G1Projective, EcError>
where
    for<'a> &'a Q: QueryDensity,
    D: Send + Sync + 'static + Clone + AsRef<Q>,
{
    let bases_gpu: Vec<G1AffineM> = bases
        .iter()
        .map(|affine| {
            let (x, y) = g1_xy_bytes_le(affine).expect("point not at infinity");
            G1AffineM { x, y }
        })
        .collect();

    let exps_bigint: Arc<Vec<_>> = Arc::new(exponents.iter().map(|e| e.into_bigint()).collect());

    let exps = density_map.as_ref().generate_exps::<Fr>(exps_bigint);

    kern.multiexp(pool, Arc::new(bases_gpu), exps, 0)
}

#[test]
fn gpu_multiexp_consistency() {
    fil_logger::maybe_init();
    const MAX_LOG_D: usize = 25;
    const START_LOG_D: usize = 20;
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

    let mut bases = (0..(1 << START_LOG_D))
        .map(|_| G1Projective::rand(&mut rng).into_affine())
        .collect::<Vec<_>>();

    for log_d in START_LOG_D..=MAX_LOG_D {
        let g = Arc::new(bases.clone());

        let samples = 1 << log_d;
        println!("Testing Multiexp for {} elements...", samples);

        let v: Vec<_> = (0..samples).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let v_arc: Arc<Vec<_>> = Arc::new(v.clone());

        let mut now = Instant::now();
        let gpu: G1Projective =
            multiexp_gpu(&pool, g.clone(), FullDensity, v_arc.clone(), &mut kern).unwrap();
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;

        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        let cpu: G1Projective = VariableBaseMSM::msm(bases.as_slice(), v.as_slice()).unwrap();

        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(cpu, gpu);

        println!("============================");

        bases = [bases.clone(), bases.clone()].concat();
    }
}

fn fq_to_32_le(x: &Fq) -> [u8; 32] {
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
