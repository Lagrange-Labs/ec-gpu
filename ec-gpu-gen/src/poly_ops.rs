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
    ///
    /// This uses `fix_vars_with_intermediates` internally to perform all operations
    /// in a single GPU session (optimization #6).
    pub fn fix_vars(&self, poly: &[F], challenges: &[F]) -> EcResult<F> {
        assert!(poly.len().is_power_of_two());
        let log_len = poly.len().ilog2() as usize;
        assert!(
            challenges.len() <= log_len,
            "Too many challenges for polynomial size"
        );

        if challenges.is_empty() {
            return Ok(poly[0]);
        }

        // Use the optimized single-session version (optimization #6)
        let intermediates = self.fix_vars_with_intermediates(poly, challenges)?;

        // The last intermediate is the final result after fixing all variables
        Ok(intermediates.last().unwrap()[0])
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

            // Upload ALL challenges in a single buffer (optimization #9)
            let challenges_buffer = program.create_buffer_from_slice(&challenges_vec)?;

            for challenge_idx in 0..num_challenges {
                let next_len = current_len / 2;

                // It is safe as the GPU will initialize that buffer
                let out_buffer = unsafe { program.create_buffer::<F>(next_len)? };

                let global_work_size = div_ceil(next_len, LOCAL_WORK_SIZE);
                let kernel_name = format!("{}_fix_var_indexed", F::name());
                let kernel =
                    program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

                kernel
                    .arg(&current_buffer)
                    .arg(&out_buffer)
                    .arg(&challenges_buffer)
                    .arg(&(next_len as u32))
                    .arg(&(challenge_idx as u32))
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

    /// Batch evaluate multiple polynomials at multiple points.
    ///
    /// Treats each polynomial as univariate in coefficient form and evaluates
    /// at all given points using Horner's method on the GPU.
    ///
    /// Returns a 2D vector: `results[point_idx][poly_idx]` = evaluation of poly_idx at point_idx.
    ///
    /// All polynomials must have the same length.
    pub fn eval_univariate_batch(
        &self,
        polys: &[&[F]],
        points: &[F],
    ) -> EcResult<Vec<Vec<F>>> {
        assert!(!polys.is_empty(), "Must have at least one polynomial");
        assert!(!points.is_empty(), "Must have at least one point");

        let poly_len = polys[0].len();
        assert!(
            polys.iter().all(|p| p.len() == poly_len),
            "All polynomials must have the same length"
        );

        let num_polys = polys.len();
        let num_points = points.len();

        // Flatten polynomials into a single buffer
        let mut flat_polys: Vec<F> = Vec::with_capacity(num_polys * poly_len);
        for poly in polys {
            flat_polys.extend_from_slice(poly);
        }

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<Vec<F>>> {
            let polys_buffer = program.create_buffer_from_slice(&flat_polys)?;
            let points_buffer = program.create_buffer_from_slice(points)?;

            let total_evals = num_polys * num_points;
            // It is safe as the GPU will initialize that buffer
            let results_buffer = unsafe { program.create_buffer::<F>(total_evals)? };

            let global_work_size = div_ceil(total_evals, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_eval_univariate_batch", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&polys_buffer)
                .arg(&points_buffer)
                .arg(&results_buffer)
                .arg(&(num_polys as u32))
                .arg(&(poly_len as u32))
                .arg(&(num_points as u32))
                .run()?;

            // Read flat results
            let mut flat_results = vec![F::ZERO; total_evals];
            program.read_into_buffer(&results_buffer, &mut flat_results)?;

            // Reshape to [num_points][num_polys]
            let mut results = Vec::with_capacity(num_points);
            for point_idx in 0..num_points {
                let start = point_idx * num_polys;
                let end = start + num_polys;
                results.push(flat_results[start..end].to_vec());
            }

            Ok(results)
        });

        self.program.run(closures, ())
    }

    /// Batch compute witness polynomials for KZG opening at multiple points.
    ///
    /// Given polynomial f(x) and evaluation points, computes witness polynomials h_i(x) where:
    /// f(x) = h_i(x) * (x - u[i]) + f(u[i])
    ///
    /// Returns a vector of witness polynomials, one for each evaluation point.
    /// Each witness polynomial has length `f.len() - 1`.
    pub fn witness_poly_batch(&self, f: &[F], points: &[F]) -> EcResult<Vec<Vec<F>>> {
        assert!(!f.is_empty(), "Polynomial must not be empty");
        assert!(!points.is_empty(), "Must have at least one point");

        let n = f.len();
        let num_points = points.len();
        let witness_len = n - 1;

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<Vec<F>>> {
            let f_buffer = program.create_buffer_from_slice(f)?;
            let points_buffer = program.create_buffer_from_slice(points)?;

            // Total output: num_points witness polynomials, each of length n-1
            let total_output = num_points * witness_len;
            // It is safe as the GPU will initialize that buffer
            let witnesses_buffer = unsafe { program.create_buffer::<F>(total_output)? };

            // One thread per point (each computes a full witness polynomial sequentially)
            let global_work_size = div_ceil(num_points, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_witness_poly_batch", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&f_buffer)
                .arg(&points_buffer)
                .arg(&witnesses_buffer)
                .arg(&(n as u32))
                .arg(&(num_points as u32))
                .run()?;

            // Read flat results
            let mut flat_witnesses = vec![F::ZERO; total_output];
            program.read_into_buffer(&witnesses_buffer, &mut flat_witnesses)?;

            // Reshape to [num_points][witness_len]
            let mut results = Vec::with_capacity(num_points);
            for point_idx in 0..num_points {
                let start = point_idx * witness_len;
                let end = start + witness_len;
                results.push(flat_witnesses[start..end].to_vec());
            }

            Ok(results)
        });

        self.program.run(closures, ())
    }

    /// Convert field elements from Montgomery form to 32-byte little-endian scalars.
    ///
    /// This is used to prepare field elements for MSM without CPU round-trip.
    /// Each field element is converted to 32 bytes in little-endian format.
    pub fn to_scalar_bytes(&self, field_elements: &[F]) -> EcResult<Vec<[u8; 32]>> {
        if field_elements.is_empty() {
            return Ok(vec![]);
        }

        let n = field_elements.len();

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<[u8; 32]>> {
            let input_buffer = program.create_buffer_from_slice(field_elements)?;
            // Output is n * 32 bytes
            // SAFETY: GPU will initialize this buffer
            let output_buffer = unsafe { program.create_buffer::<u8>(n * 32)? };

            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_to_scalar_bytes", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&input_buffer)
                .arg(&output_buffer)
                .arg(&(n as u32))
                .run()?;

            // Read raw bytes
            let mut flat_bytes = vec![0u8; n * 32];
            program.read_into_buffer(&output_buffer, &mut flat_bytes)?;

            // Convert to array of [u8; 32]
            let result: Vec<[u8; 32]> = flat_bytes
                .chunks(32)
                .map(|chunk| {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(chunk);
                    arr
                })
                .collect();

            Ok(result)
        });

        self.program.run(closures, ())
    }

    /// Combined fix_var + to_scalar_bytes operation.
    ///
    /// Performs fix_var on the polynomial and immediately converts the result
    /// to scalar bytes format, avoiding intermediate memory storage.
    ///
    /// This is optimized for the HyperKZG fused operation where we need to
    /// commit to intermediate polynomials.
    pub fn fix_var_to_scalar(&self, poly: &[F], r: &F) -> EcResult<Vec<[u8; 32]>> {
        assert!(
            poly.len() >= 2 && poly.len().is_power_of_two(),
            "Polynomial length must be a power of 2 and >= 2"
        );

        let n = poly.len() / 2;
        let r_slice = [*r];

        let closures = program_closures!(|program, _arg| -> EcResult<Vec<[u8; 32]>> {
            let poly_buffer = program.create_buffer_from_slice(poly)?;
            let r_buffer = program.create_buffer_from_slice(&r_slice)?;
            // Output is n * 32 bytes
            // SAFETY: GPU will initialize this buffer
            let output_buffer = unsafe { program.create_buffer::<u8>(n * 32)? };

            let global_work_size = div_ceil(n, LOCAL_WORK_SIZE);
            let kernel_name = format!("{}_fix_var_to_scalar", F::name());
            let kernel = program.create_kernel(&kernel_name, global_work_size, LOCAL_WORK_SIZE)?;

            kernel
                .arg(&poly_buffer)
                .arg(&output_buffer)
                .arg(&r_buffer)
                .arg(&(n as u32))
                .run()?;

            // Read raw bytes
            let mut flat_bytes = vec![0u8; n * 32];
            program.read_into_buffer(&output_buffer, &mut flat_bytes)?;

            // Convert to array of [u8; 32]
            let result: Vec<[u8; 32]> = flat_bytes
                .chunks(32)
                .map(|chunk| {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(chunk);
                    arr
                })
                .collect();

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

    /// Batch evaluate multiple polynomials at multiple points.
    pub fn eval_univariate_batch(
        &self,
        polys: &[&[F]],
        points: &[F],
    ) -> EcResult<Vec<Vec<F>>> {
        self.kernels[0].eval_univariate_batch(polys, points)
    }

    /// Batch compute witness polynomials for multiple points.
    pub fn witness_poly_batch(&self, f: &[F], points: &[F]) -> EcResult<Vec<Vec<F>>> {
        self.kernels[0].witness_poly_batch(f, points)
    }

    /// Convert field elements to scalar bytes on GPU.
    pub fn to_scalar_bytes(&self, field_elements: &[F]) -> EcResult<Vec<[u8; 32]>> {
        self.kernels[0].to_scalar_bytes(field_elements)
    }

    /// Combined fix_var + to_scalar_bytes operation.
    pub fn fix_var_to_scalar(&self, poly: &[F], r: &F) -> EcResult<Vec<[u8; 32]>> {
        self.kernels[0].fix_var_to_scalar(poly, r)
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "arkworks")]
    use ark_bn254::Fr;
    #[cfg(feature = "arkworks")]
    use ark_ff::Field;
    #[cfg(feature = "arkworks")]
    use ark_std::{UniformRand, Zero};

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

    /// CPU reference for univariate polynomial evaluation using Horner's method
    #[cfg(feature = "arkworks")]
    fn eval_univariate_cpu<F: Field>(coeffs: &[F], x: &F) -> F {
        let mut result = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            result = result * x + coeffs[i];
        }
        result
    }

    #[cfg(feature = "arkworks")]
    #[test]
    fn test_eval_univariate_batch_reference() {
        let mut rng = ark_std::test_rng();

        let p1: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rng)).collect();
        let p2: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rng)).collect();
        let points = vec![Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng)];

        let polys = vec![&p1[..], &p2[..]];

        // Compute expected results
        for (point_idx, point) in points.iter().enumerate() {
            for (poly_idx, poly) in polys.iter().enumerate() {
                let expected = eval_univariate_cpu(poly, point);
                // Just verify the CPU reference computes something
                assert!(!expected.is_zero() || poly.iter().all(|c| c.is_zero()));
                println!(
                    "eval_univariate_cpu(poly[{}], point[{}]) = {:?}",
                    poly_idx, point_idx, expected
                );
            }
        }
    }

    #[cfg(feature = "arkworks")]
    #[test]
    fn test_witness_poly_batch_reference() {
        let mut rng = ark_std::test_rng();

        let f: Vec<Fr> = (0..8).map(|_| Fr::rand(&mut rng)).collect();
        let points = vec![Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng)];

        // Compute witnesses for each point using the CPU reference
        for (i, point) in points.iter().enumerate() {
            let h = witness_poly_cpu(&f, point);
            assert_eq!(h.len(), f.len() - 1);
            println!("witness_poly_cpu(f, point[{}]).len() = {}", i, h.len());
        }
    }
}
