#![cfg(any(feature = "cuda", feature = "opencl"))]

use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::{FftField, UniformRand};
use ec_gpu_gen::{fft::FftKernelArk, rust_gpu_tools::Device};

fn omega<F: FftField>(num_coeffs: usize) -> F {
    let exp = (num_coeffs as f32).log2().floor() as u32;
    let mut omega = F::TWO_ADIC_ROOT_OF_UNITY;
    for _ in exp..F::TWO_ADICITY {
        omega = omega.square();
    }
    omega
}

fn serial_fft<F: FftField>(a: &mut [F], omega: &F, log_n: u32) {
    let n = a.len();
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n as usize);
        if k < rk {
            a.swap(rk, k);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow([(n / (2 * m)) as u64]);
        let mut k = 0;
        while k < n {
            let mut w = F::ONE;
            for j in 0..m {
                let t = a[k + j + m] * w;
                a[k + j + m] = a[k + j] - t;
                a[k + j] += t;
                w *= w_m;
            }
            k += 2 * m;
        }
        m *= 2;
    }
}

fn bitreverse(mut n: usize, l: usize) -> usize {
    let mut r = 0;
    for _ in 0..l {
        r = (r << 1) | (n & 1);
        n >>= 1;
    }
    r
}

#[test]
pub fn gpu_fft_consistency() {
    fil_logger::maybe_init();
    let mut rng = rand::thread_rng();

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernelArk::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for log_d in 1..=20 {
        let d = 1 << log_d;

        let mut v1_coeffs = (0..d).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let v1_omega = omega::<Fr>(v1_coeffs.len());
        let mut v2_coeffs = v1_coeffs.clone();
        let v2_omega = v1_omega;

        println!("Testing FFT for {} elements...", d);

        let mut now = Instant::now();
        kern.radix_fft_many(&mut [&mut v1_coeffs], &[v1_omega], &[log_d])
            .expect("GPU FFT failed!");
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        serial_fft::<Fr>(&mut v2_coeffs, &v2_omega, log_d);
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(v1_coeffs, v2_coeffs);
        println!("============================");
    }
}

#[test]
pub fn gpu_fft_many_consistency() {
    fil_logger::maybe_init();
    let mut rng = rand::thread_rng();

    let devices = Device::all();
    let programs = devices
        .iter()
        .map(|device| ec_gpu_gen::program!(device))
        .collect::<Result<_, _>>()
        .expect("Cannot create programs!");
    let mut kern = FftKernelArk::<Fr>::create(programs).expect("Cannot initialize kernel!");

    for log_d in 1..=20 {
        let d = 1 << log_d;

        let mut v11_coeffs = (0..d).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let mut v12_coeffs = (0..d).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let mut v13_coeffs = (0..d).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();
        let fft_omega = omega::<Fr>(d);

        let mut v21_coeffs = v11_coeffs.clone();
        let mut v22_coeffs = v12_coeffs.clone();
        let mut v23_coeffs = v13_coeffs.clone();

        println!("Testing FFT3 for {} elements...", d);

        let mut now = Instant::now();
        kern.radix_fft_many(
            &mut [&mut v11_coeffs, &mut v12_coeffs, &mut v13_coeffs],
            &[fft_omega, fft_omega, fft_omega],
            &[log_d, log_d, log_d],
        )
        .expect("GPU FFT failed!");
        let gpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("GPU took {}ms.", gpu_dur);

        now = Instant::now();
        serial_fft::<Fr>(&mut v21_coeffs, &fft_omega, log_d);
        serial_fft::<Fr>(&mut v22_coeffs, &fft_omega, log_d);
        serial_fft::<Fr>(&mut v23_coeffs, &fft_omega, log_d);
        let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
        println!("CPU took {}ms.", cpu_dur);

        println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

        assert_eq!(v11_coeffs, v21_coeffs);
        assert_eq!(v12_coeffs, v22_coeffs);
        assert_eq!(v13_coeffs, v23_coeffs);

        println!("============================");
    }
}
