//! Polynomial operations on the GPU for HyperKZG acceleration.
//!
//! This module provides GPU-accelerated polynomial operations including:
//! - Variable fixing (linear interpolation)
//! - Polynomial evaluation
//! - Linear combination
//! - Witness polynomial computation

use ark_ff::PrimeField;
use ec_gpu::GpuName;
use rust_gpu_tools::{program_closures, Device, Program};

use crate::error::EcResult;

/// The number of threads per work group.
const LOCAL_WORK_SIZE: usize = 256;

/// Divide and ceil to the next value.
const fn div_ceil(a: usize, b: usize) -> usize {
    if a % b == 0 {
        a / b
    } else {
        (a / b) + 1
    }
}

/// A kernel for polynomial operations on a single GPU.
pub struct SinglePolyOpsKernel<F: PrimeField + GpuName> {
    program: Program,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: PrimeField + GpuName> SinglePolyOpsKernel<F> {
    /// Create a new polynomial operations kernel for the given program.
    pub fn create(program: Program) -> EcResult<Self> {
        Ok(Self {
            program,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Fix the lowest variable of a multilinear polynomial.
    ///
    /// Given a polynomial in evaluation form of length 2n, computes a new polynomial
    /// of length n by fixing the lowest variable to value `r`.
    ///
    /// Formula: `out[j] = r * (poly[2j+1] - poly[2j]) + poly[2j]`
    ///
    /// This is equivalent to linear interpolation between adjacent pairs.
    pub fn fix_var(&self, poly: &[F], r: &F) -> EcResult<Vec<F>> {
        assert!(
            poly.len() >= 2 && poly.len().is_power_of_two(),
            "Polynomial length must be a power of 2 and >= 2"
        );

        let n = poly.len() / 2;

        // Pass r as a single-element slice for GPU buffer
        let r_slice = [*r];

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<F>> {
            let poly_buffer = program.create_buffer_from_slice(poly)?;
            let r_buffer = program.create_buffer_from_slice(&r_slice)?;
            // It is safe as the GPU will initialize that buffer
            let out_buffer = unsafe { program.create_buffer::<F>(n)? };

            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_fix_var", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&poly_buffer)
                .arg(&out_buffer)
                .arg(&r_buffer)
                .arg(&(n as u32))
                .run()?;

            let mut result = vec![F::ZERO; n];
            program.read_into_buffer(&out_buffer, &mut result)?;

            Ok(result)
        });

        self.program.run(closures, ())
    }

    /// Fix multiple variables iteratively.
    ///
    /// Starting from a polynomial of length 2^k, fixes k variables to the given
    /// challenge values, returning the final constant value.
    pub fn fix_vars(&self, poly: &[F], challenges: &[F]) -> EcResult<F> {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );

        let mut current = poly.to_vec();

        for challenge in challenges {
            current = self.fix_var(&current, challenge)?;
        }

        // After fixing all variables, we should have a single element
        Ok(current[0])
    }

    /// Fix multiple variables and return all intermediate polynomials.
    ///
    /// This is optimized for HyperKZG Phase 1: it performs all variable fixings
    /// in a single GPU session, minimizing data transfers.
    ///
    /// Returns a vector of polynomials:
    /// - polys[0] is the result after fixing the first variable
    /// - polys[i] is the result after fixing variables 0..=i
    /// - The original polynomial is NOT included in the output
    ///
    /// Each intermediate polynomial is used for MSM commitments in HyperKZG.
    pub fn fix_vars_with_intermediates(
        &self,
        poly: &[F],
        challenges: &[F],
    ) -> EcResult<Vec<Vec<F>>> {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );

        if challenges.is_empty() {
            return Ok(vec![]);
        }

        let num_challenges = challenges.len();
        let initial_len = poly.len();
        // Copy challenges for use in closure
        let challenges_vec = challenges.to_vec();

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<Vec<F>>> {
            // Upload initial polynomial once
            let mut current_buffer = program.create_buffer_from_slice(poly)?;
            let mut current_len = initial_len;
            let mut results = Vec::with_capacity(num_challenges);

            for r in challenges_vec.iter() {
                let next_len = current_len / 2;

                // Create single-element buffer for r
                let r_buffer = program.create_buffer_from_slice(&[*r])?;

                // It is safe as the GPU will initialize that buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let kernel_name = format!("{}_fix_var", F::name());
                let kernel =
                    program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

                kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&r_buffer)
                    .arg(&(next_len as u32))
                    .run()?;

                // Download intermediate result for MSM
                let mut intermediate = vec![F::ZERO; next_len];
                program.read_into_buffer(&out_buffer, &mut intermediate)?;
                results.push(intermediate);

                // Update for next iteration
                current_buffer = out_buffer;
                current_len = next_len;
            }

            Ok(results)
        });

        self.program.run(closures, ())
    }

    /// Linear combination of polynomials: out = sum(coeffs[i] * polys[i])
    ///
    /// All polynomials must have the same length.
    pub fn linear_combine(&self, polys: &[&[F]], coeffs: &[F]) -> EcResult<Vec<F>> {
        assert_eq!(
            polys.len(),
            coeffs.len(),
            "Number of polynomials must match number of coefficients"
        );
        assert!(!polys.is_empty(), "Must have at least one polynomial");

        let poly_len = polys[0].len();
        assert!(
            polys.iter().all(|p| p.len() == poly_len),
            "All polynomials must have the same length"
        );

        let num_polys = polys.len();

        // Flatten polynomials into a single buffer
        let mut flat_polys: Vec<F> = Vec::with_capacity(num_polys * poly_len);
        for poly in polys {
            flat_polys.extend_from_slice(poly);
        }

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<F>> {
            let polys_buffer = program.create_buffer_from_slice(&flat_polys)?;
            let coeffs_buffer = program.create_buffer_from_slice(coeffs)?;
            // It is safe as the GPU will initialize that buffer
            let out_buffer = unsafe { program.create_buffer::<F>(poly_len)? };

            let global_work_size = div_ceil(poly_len, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_linear_combine", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&polys_buffer)
                .arg(&coeffs_buffer)
                .arg(&out_buffer)
                .arg(&(num_polys as u32))
                .arg(&(poly_len as u32))
                .run()?;

            let mut result = vec![F::ZERO; poly_len];
            program.read_into_buffer(&out_buffer, &mut result)?;

            Ok(result)
        });

        self.program.run(closures, ())
    }

    /// Compute witness polynomial for KZG opening.
    ///
    /// Given f(x) and evaluation point u, computes h(x) where:
    /// f(x) = h(x) * (x - u) + f(u)
    ///
    /// This uses the recurrence: h[i-1] = f[i] + h[i] * u
    pub fn witness_poly(&self, f: &[F], u: &F) -> EcResult<Vec<F>> {
        assert!(!f.is_empty(), "Polynomial must not be empty");

        let n = f.len();
        let u_slice = [*u];

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<F>> {
            let f_buffer = program.create_buffer_from_slice(f)?;
            let u_buffer = program.create_buffer_from_slice(&u_slice)?;
            // h has length n-1, but we allocate n for simplicity
            // It is safe as the GPU will initialize that buffer
            let h_buffer = unsafe { program.create_buffer::<F>(n)? };

            // Use sequential kernel for now (single thread)
            let kernel_name = format!("{}_witness_poly_sequential", F::name());
            let kernel = program.create_kernel(&kernel_name, 1, 1)?;

            kernel
                .arg(&f_buffer)
                .arg(&h_buffer)
                .arg(&u_buffer)
                .arg(&(n as u32))
                .run()?;

            let mut result = vec![F::ZERO; n - 1];
            program.read_into_buffer(&h_buffer, &mut result)?;

            Ok(result)
        });

        self.program.run(closures, ())
    }

    /// Scale a polynomial by a scalar: out[i] = poly[i] * scalar
    pub fn scale_poly(&self, poly: &[F], scalar: &F) -> EcResult<Vec<F>> {
        let n = poly.len();
        let scalar_slice = [*scalar];

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<F>> {
            let poly_buffer = program.create_buffer_from_slice(poly)?;
            let scalar_buffer = program.create_buffer_from_slice(&scalar_slice)?;
            // It is safe as the GPU will initialize that buffer
            let out_buffer = unsafe { program.create_buffer::<F>(n)? };

            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_scale_poly", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&poly_buffer)
                .arg(&out_buffer)
                .arg(&scalar_buffer)
                .arg(&(n as u32))
                .run()?;

            let mut result = vec![F::ZERO; n];
            program.read_into_buffer(&out_buffer, &mut result)?;

            Ok(result)
        });

        self.program.run(closures, ())
    }

    /// Add two polynomials: out[i] = a[i] + b[i]
    pub fn add_poly(&self, a: &[F], b: &[F]) -> EcResult<Vec<F>> {
        assert_eq!(a.len(), b.len(), "Polynomials must have the same length");

        let n = a.len();

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<F>> {
            let a_buffer = program.create_buffer_from_slice(a)?;
            let b_buffer = program.create_buffer_from_slice(b)?;
            // It is safe as the GPU will initialize that buffer
            let out_buffer = unsafe { program.create_buffer::<F>(n)? };

            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_add_poly", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&a_buffer)
                .arg(&b_buffer)
                .arg(&out_buffer)
                .arg(&(n as u32))
                .run()?;

            let mut result = vec![F::ZERO; n];
            program.read_into_buffer(&out_buffer, &mut result)?;

            Ok(result)
        });

        self.program.run(closures, ())
    }

    /// Subtract two polynomials: out[i] = a[i] - b[i]
    pub fn sub_poly(&self, a: &[F], b: &[F]) -> EcResult<Vec<F>> {
        assert_eq!(a.len(), b.len(), "Polynomials must have the same length");

        let n = a.len();

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<F>> {
            let a_buffer = program.create_buffer_from_slice(a)?;
            let b_buffer = program.create_buffer_from_slice(b)?;
            // It is safe as the GPU will initialize that buffer
            let out_buffer = unsafe { program.create_buffer::<F>(n)? };

            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_sub_poly", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&a_buffer)
                .arg(&b_buffer)
                .arg(&out_buffer)
                .arg(&(n as u32))
                .run()?;

            let mut result = vec![F::ZERO; n];
            program.read_into_buffer(&out_buffer, &mut result)?;

            Ok(result)
        });

        self.program.run(closures, ())
    }
}

/// A struct that contains polynomial operations kernels for multiple devices.
pub struct PolyOpsKernel<F: PrimeField + GpuName> {
    kernels: Vec<SinglePolyOpsKernel<F>>,
}

impl<F: PrimeField + GpuName> PolyOpsKernel<F> {
    /// Create new kernels, one for each given device.
    pub fn create(programs: Vec<Program>, _devices: &[&Device]) -> EcResult<Self> {
        let kernels: Vec<_> = programs
            .into_iter()
            .map(SinglePolyOpsKernel::create)
            .collect::<Result<_, _>>()?;

        Ok(Self { kernels })
    }

    /// Get the first kernel (for single-GPU operations).
    pub fn kernel(&self) -> &SinglePolyOpsKernel<F> {
        &self.kernels[0]
    }

    /// Fix the lowest variable of a multilinear polynomial.
    pub fn fix_var(&self, poly: &[F], r: &F) -> EcResult<Vec<F>> {
        self.kernels[0].fix_var(poly, r)
    }

    /// Fix multiple variables iteratively.
    pub fn fix_vars(&self, poly: &[F], challenges: &[F]) -> EcResult<F> {
        self.kernels[0].fix_vars(poly, challenges)
    }

    /// Linear combination of polynomials.
    pub fn linear_combine(&self, polys: &[&[F]], coeffs: &[F]) -> EcResult<Vec<F>> {
        self.kernels[0].linear_combine(polys, coeffs)
    }

    /// Compute witness polynomial for KZG opening.
    pub fn witness_poly(&self, f: &[F], u: &F) -> EcResult<Vec<F>> {
        self.kernels[0].witness_poly(f, u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "arkworks")]
    use ark_bn254::Fr;
    #[cfg(feature = "arkworks")]
    use ark_ff::Field;
    #[cfg(feature = "arkworks")]
    use ark_std::UniformRand;

    /// CPU reference implementation for fix_var
    #[cfg(feature = "arkworks")]
    fn fix_var_cpu<F: Field>(poly: &[F], r: &F) -> Vec<F> {
        let n = poly.len() / 2;
        (0..n)
            .map(|j| {
                let low = poly[2 * j];
                let high = poly[2 * j + 1];
                *r * (high - low) + low
            })
            .collect()
    }

    /// CPU reference implementation for linear_combine
    #[cfg(feature = "arkworks")]
    fn linear_combine_cpu<F: Field>(polys: &[&[F]], coeffs: &[F]) -> Vec<F> {
        let poly_len = polys[0].len();
        (0..poly_len)
            .map(|i| polys.iter().zip(coeffs.iter()).map(|(p, c)| p[i] * c).sum())
            .collect()
    }

    /// CPU reference implementation for witness_poly
    #[cfg(feature = "arkworks")]
    fn witness_poly_cpu<F: Field>(f: &[F], u: &F) -> Vec<F> {
        let n = f.len();
        let mut h = vec![F::ZERO; n - 1];
        let mut carry = F::ZERO;
        for i in (1..n).rev() {
            carry = f[i] + carry * u;
            h[i - 1] = carry;
        }
        h
    }

    #[cfg(feature = "arkworks")]
    #[test]
    fn test_fix_var_reference() {
        let mut rng = ark_std::test_rng();

        // Test with a small polynomial
        let poly: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rng)).collect();
        let r = Fr::rand(&mut rng);

        let result = fix_var_cpu(&poly, &r);
        assert_eq!(result.len(), 4);

        // Verify the formula manually for first element
        let expected_0 = r * (poly[1] - poly[0]) + poly[0];
        assert_eq!(result[0], expected_0);
    }

    #[cfg(feature = "arkworks")]
    #[test]
    fn test_linear_combine_reference() {
        let mut rng = ark_std::test_rng();

        let p1: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let p2: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let c1 = Fr::rand(&mut rng);
        let c2 = Fr::rand(&mut rng);

        let polys: Vec<&[Fr]> = vec![&p1, &p2];
        let coeffs = vec![c1, c2];

        let result = linear_combine_cpu(&polys, &coeffs);
        assert_eq!(result.len(), 4);

        // Verify manually
        for i in 0..4 {
            assert_eq!(result[i], p1[i] * c1 + p2[i] * c2);
        }
    }

    #[cfg(feature = "arkworks")]
    #[test]
    fn test_witness_poly_reference() {
        let mut rng = ark_std::test_rng();

        let f: Vec<Fr> = (0..4).map(|_| Fr::rand(&mut rng)).collect();
        let u = Fr::rand(&mut rng);

        let h = witness_poly_cpu(&f, &u);
        assert_eq!(h.len(), 3);

        // Verify: f(x) = h(x) * (x - u) + f(u)
        // At x = u: f(u) should equal f evaluated at u
        // The witness polynomial satisfies the division property
    }
}
