/*
 * Polynomial operations for HyperKZG acceleration.
 * These kernels enable full GPU execution of polynomial commitment schemes.
 */

/*
 * Fix the lowest variable of a multilinear polynomial.
 * Computes: out[j] = r * (poly[2j+1] - poly[2j]) + poly[2j]
 * This is equivalent to linear interpolation between adjacent pairs.
 *
 * Input:  poly of length 2n (evaluation form)
 * Output: out of length n
 * r_buf:  Single-element buffer containing the challenge value r
 * Effect: Fixes the lowest variable to value r
 */
KERNEL void FIELD_fix_var(
    GLOBAL FIELD* poly,
    GLOBAL FIELD* out,
    GLOBAL FIELD* r_buf,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    // Load r from buffer
    FIELD r = r_buf[0];

    FIELD low = poly[2 * gid];
    FIELD high = poly[2 * gid + 1];

    // diff = high - low
    FIELD diff = FIELD_sub(high, low);

    // scaled = r * diff
    FIELD scaled = FIELD_mul(r, diff);

    // result = low + scaled = low + r * (high - low)
    out[gid] = FIELD_add(low, scaled);
}

/*
 * Fix the lowest variable with challenge read from an offset in a buffer.
 * This variant allows all challenges to be uploaded once and indexed per iteration.
 *
 * Input:  poly of length 2n (evaluation form)
 * Output: out of length n
 * challenges: Buffer containing all challenge values
 * challenge_idx: Index of the challenge to use from the buffer
 * Effect: Fixes the lowest variable to value challenges[challenge_idx]
 */
KERNEL void FIELD_fix_var_indexed(
    GLOBAL FIELD* poly,
    GLOBAL FIELD* out,
    GLOBAL FIELD* challenges,
    uint n,
    uint challenge_idx)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    // Load r from the indexed position in challenges buffer
    FIELD r = challenges[challenge_idx];

    FIELD low = poly[2 * gid];
    FIELD high = poly[2 * gid + 1];

    // diff = high - low
    FIELD diff = FIELD_sub(high, low);

    // scaled = r * diff
    FIELD scaled = FIELD_mul(r, diff);

    // result = low + scaled = low + r * (high - low)
    out[gid] = FIELD_add(low, scaled);
}

/*
 * Evaluate a univariate polynomial at a single point using Horner's method.
 * The polynomial is in coefficient form: f(x) = c0 + c1*x + c2*x^2 + ...
 *
 * This kernel performs parallel partial reductions, then a final sequential combine.
 * Each thread processes a chunk and produces a partial result.
 *
 * For evaluation of multilinear polynomials treated as univariate:
 * f(x) = evals[0] + evals[1]*x + evals[2]*x^2 + ...
 */
KERNEL void FIELD_eval_poly_partial(
    GLOBAL FIELD* coeffs,
    GLOBAL FIELD* partial_results,
    GLOBAL FIELD* x_powers,  // Precomputed powers of x: [1, x, x^2, ..., x^(chunk_size-1)]
    FIELD x_chunk_power,     // x^chunk_size for combining chunks
    uint n,
    uint chunk_size)
{
    const uint gid = GET_GLOBAL_ID();
    const uint start = gid * chunk_size;

    if (start >= n) return;

    const uint end = min(start + chunk_size, n);

    // Compute sum of coeffs[i] * x^(i - start) for i in [start, end)
    FIELD result = FIELD_ZERO;
    for (uint i = start; i < end; i++) {
        FIELD term = FIELD_mul(coeffs[i], x_powers[i - start]);
        result = FIELD_add(result, term);
    }

    partial_results[gid] = result;
}

/*
 * Linear combination of polynomials: out = sum(coeffs[i] * polys[i])
 * All polynomials must have the same length.
 *
 * polys: Concatenated polynomials [poly0, poly1, ..., poly_{k-1}]
 * coeffs: Scaling coefficients [c0, c1, ..., c_{k-1}]
 * out: Result polynomial of length poly_len
 */
KERNEL void FIELD_linear_combine(
    GLOBAL FIELD* polys,
    GLOBAL FIELD* coeffs,
    GLOBAL FIELD* out,
    uint num_polys,
    uint poly_len)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= poly_len) return;

    FIELD result = FIELD_ZERO;

    for (uint i = 0; i < num_polys; i++) {
        FIELD scaled = FIELD_mul(polys[i * poly_len + gid], coeffs[i]);
        result = FIELD_add(result, scaled);
    }

    out[gid] = result;
}

/*
 * Compute witness polynomial for KZG opening.
 * Given f(x) and evaluation point u, computes h(x) where:
 *   f(x) = h(x) * (x - u) + f(u)
 *   h[i-1] = f[i] + h[i] * u  (computed in reverse)
 *
 * This is inherently sequential due to the recurrence relation.
 * For large polynomials, we can parallelize by processing chunks
 * and combining results.
 *
 * Simple sequential version (single thread):
 */
KERNEL void FIELD_witness_poly_sequential(
    GLOBAL FIELD* f,
    GLOBAL FIELD* h,
    GLOBAL FIELD* u_buf,
    uint n)
{
    // Only thread 0 executes
    if (GET_GLOBAL_ID() != 0) return;

    // Load u from buffer
    FIELD u = u_buf[0];

    // h[n-1] = 0 (implied, h has length n-1)
    // h[i-1] = f[i] + h[i] * u for i from n-1 down to 1

    FIELD carry = FIELD_ZERO;
    for (int i = n - 1; i >= 1; i--) {
        carry = FIELD_add(f[i], FIELD_mul(carry, u));
        h[i - 1] = carry;
    }
}

/*
 * Parallel witness polynomial computation using divide-and-conquer.
 * Split the polynomial into chunks, process each in parallel,
 * then combine results.
 *
 * For chunk starting at index `start` with length `chunk_len`:
 * Local result: h_local[i] = f[start+i+1] + h_local[i+1] * u
 *
 * To combine chunks, we need to propagate the carry from the next chunk.
 * Final h[i] = h_local[i] + carry_from_next * u^(position)
 */
KERNEL void FIELD_witness_poly_chunk(
    GLOBAL FIELD* f,
    GLOBAL FIELD* h_chunks,      // Output: partial results per chunk
    GLOBAL FIELD* chunk_carries, // Output: final carry value of each chunk
    GLOBAL FIELD* u_buf,
    uint n,
    uint chunk_size)
{
    const uint chunk_id = GET_GLOBAL_ID();
    const uint num_chunks = (n + chunk_size - 1) / chunk_size;

    if (chunk_id >= num_chunks) return;

    // Load u from buffer
    FIELD u = u_buf[0];

    // Process chunk in reverse order
    const uint chunk_start = chunk_id * chunk_size;
    const uint chunk_end = min(chunk_start + chunk_size, n);

    FIELD carry = FIELD_ZERO;

    // Process from end of chunk backwards
    for (int i = chunk_end - 1; i >= (int)chunk_start && i >= 1; i--) {
        carry = FIELD_add(f[i], FIELD_mul(carry, u));
        if (i > 0) {
            h_chunks[i - 1] = carry;
        }
    }

    // Store the final carry for this chunk (to be propagated to previous chunk)
    chunk_carries[chunk_id] = carry;
}

/*
 * Scale polynomial by a scalar: out[i] = poly[i] * scalar
 */
KERNEL void FIELD_scale_poly(
    GLOBAL FIELD* poly,
    GLOBAL FIELD* out,
    GLOBAL FIELD* scalar_buf,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    FIELD scalar = scalar_buf[0];
    out[gid] = FIELD_mul(poly[gid], scalar);
}

/*
 * Add two polynomials: out[i] = a[i] + b[i]
 */
KERNEL void FIELD_add_poly(
    GLOBAL FIELD* a,
    GLOBAL FIELD* b,
    GLOBAL FIELD* out,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    out[gid] = FIELD_add(a[gid], b[gid]);
}

/*
 * Subtract two polynomials: out[i] = a[i] - b[i]
 */
KERNEL void FIELD_sub_poly(
    GLOBAL FIELD* a,
    GLOBAL FIELD* b,
    GLOBAL FIELD* out,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    out[gid] = FIELD_sub(a[gid], b[gid]);
}

/*
 * Batch evaluate multiple polynomials at multiple points using Horner's method.
 *
 * Each thread handles one (poly_idx, point_idx) pair.
 * The polynomial is treated as univariate in coefficient form:
 *   f(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
 *
 * polys:       Concatenated polynomials [poly0, poly1, ..., poly_{num_polys-1}]
 * points:      Evaluation points [point0, point1, ..., point_{num_points-1}]
 * results:     Output: results[point_idx * num_polys + poly_idx] = poly_idx(point_idx)
 * num_polys:   Number of polynomials
 * poly_len:    Length of each polynomial (all same length)
 * num_points:  Number of evaluation points
 *
 * Total threads needed: num_polys * num_points
 */
KERNEL void FIELD_eval_univariate_batch(
    GLOBAL FIELD* polys,
    GLOBAL FIELD* points,
    GLOBAL FIELD* results,
    uint num_polys,
    uint poly_len,
    uint num_points)
{
    const uint gid = GET_GLOBAL_ID();
    const uint total_evals = num_polys * num_points;
    if (gid >= total_evals) return;

    // Decode which polynomial and which point this thread handles
    const uint poly_idx = gid % num_polys;
    const uint point_idx = gid / num_polys;

    // Get the evaluation point
    FIELD x = points[point_idx];

    // Get pointer to this polynomial's coefficients
    GLOBAL FIELD* coeffs = polys + poly_idx * poly_len;

    // Evaluate using Horner's method: f(x) = c[n-1], then f = f*x + c[i] for i = n-2..0
    FIELD result = coeffs[poly_len - 1];
    for (int i = (int)poly_len - 2; i >= 0; i--) {
        result = FIELD_add(FIELD_mul(result, x), coeffs[i]);
    }

    // Store result at [point_idx][poly_idx] position
    results[point_idx * num_polys + poly_idx] = result;
}

/*
 * Batch compute witness polynomials for KZG opening at multiple points.
 *
 * Given polynomial f(x) and evaluation points u[0..num_points-1],
 * computes witness polynomials h_i(x) where:
 *   f(x) = h_i(x) * (x - u[i]) + f(u[i])
 *
 * The recurrence is: h[j-1] = f[j] + h[j] * u (computed in reverse)
 *
 * This kernel processes one polynomial at one point per thread group,
 * with each thread in the group handling a chunk of the computation.
 * Since witness computation is inherently sequential, we use a single thread.
 *
 * f:           Input polynomial coefficients (length n)
 * u_points:    Evaluation points [u0, u1, ..., u_{num_points-1}]
 * witnesses:   Output: concatenated witness polys [h0, h1, ...] each of length n-1
 * n:           Length of input polynomial
 * num_points:  Number of evaluation points
 *
 * Total threads needed: num_points (one thread per point, sequential computation)
 */
KERNEL void FIELD_witness_poly_batch(
    GLOBAL FIELD* f,
    GLOBAL FIELD* u_points,
    GLOBAL FIELD* witnesses,
    uint n,
    uint num_points)
{
    const uint point_idx = GET_GLOBAL_ID();
    if (point_idx >= num_points) return;

    // Get the evaluation point for this witness
    FIELD u = u_points[point_idx];

    // Output witness polynomial starts at this offset
    GLOBAL FIELD* h = witnesses + point_idx * (n - 1);

    // Compute h(x) = f(x)/(x - u) using the recurrence h[i-1] = f[i] + h[i] * u
    FIELD carry = FIELD_ZERO;
    for (int i = n - 1; i >= 1; i--) {
        carry = FIELD_add(f[i], FIELD_mul(carry, u));
        h[i - 1] = carry;
    }
}

/*
 * Convert field elements from Montgomery form to little-endian scalar bytes.
 *
 * This is used to prepare field elements for MSM without CPU round-trip.
 * The field elements must be in Montgomery form (as output by fix_var).
 * The output is 32-byte little-endian scalars suitable for the multiexp kernel.
 *
 * input:  Field elements in Montgomery form (length n)
 * output: Scalar bytes, 32 bytes per element (total n * 32 bytes)
 * n:      Number of elements to convert
 *
 * Total threads needed: n (one thread per element)
 */
KERNEL void FIELD_to_scalar_bytes(
    GLOBAL FIELD* input,
    GLOBAL uchar* output,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    // Load field element in Montgomery form
    FIELD mont = input[gid];

    // Convert from Montgomery form to standard form
    // This computes: mont * R^(-1) mod P
    FIELD standard = FIELD_unmont(mont);

    // Write 32 bytes in little-endian order
    GLOBAL uchar* out_ptr = output + gid * 32;

    // The standard form is stored as FIELD_LIMBS limbs, each FIELD_LIMB_BITS bits
    // We need to serialize to 32 bytes in little-endian order
    #if FIELD_LIMB_BITS == 32
        // 8 x 32-bit limbs = 32 bytes
        for (uint i = 0; i < FIELD_LIMBS; i++) {
            FIELD_limb limb = standard.val[i];
            out_ptr[i * 4 + 0] = (uchar)(limb);
            out_ptr[i * 4 + 1] = (uchar)(limb >> 8);
            out_ptr[i * 4 + 2] = (uchar)(limb >> 16);
            out_ptr[i * 4 + 3] = (uchar)(limb >> 24);
        }
    #elif FIELD_LIMB_BITS == 64
        // 4 x 64-bit limbs = 32 bytes
        for (uint i = 0; i < FIELD_LIMBS; i++) {
            FIELD_limb limb = standard.val[i];
            out_ptr[i * 8 + 0] = (uchar)(limb);
            out_ptr[i * 8 + 1] = (uchar)(limb >> 8);
            out_ptr[i * 8 + 2] = (uchar)(limb >> 16);
            out_ptr[i * 8 + 3] = (uchar)(limb >> 24);
            out_ptr[i * 8 + 4] = (uchar)(limb >> 32);
            out_ptr[i * 8 + 5] = (uchar)(limb >> 40);
            out_ptr[i * 8 + 6] = (uchar)(limb >> 48);
            out_ptr[i * 8 + 7] = (uchar)(limb >> 56);
        }
    #else
        #error "Unsupported FIELD_LIMB_BITS value"
    #endif
}

/*
 * Convert field elements from Montgomery form to 32-byte scalars with offset.
 *
 * This variant reads from a specific offset in the input buffer, useful for
 * processing batched witness polynomials without copying.
 *
 * input:       Field elements in Montgomery form
 * output:      Scalar bytes, 32 bytes per element (total n * 32 bytes)
 * n:           Number of elements to convert
 * input_offset: Offset (in elements) into the input buffer
 *
 * Total threads needed: n (one thread per element)
 */
KERNEL void FIELD_to_scalar_bytes_offset(
    GLOBAL FIELD* input,
    GLOBAL uchar* output,
    uint n,
    uint input_offset)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    // Load field element in Montgomery form from offset position
    FIELD mont = input[input_offset + gid];

    // Convert from Montgomery form to standard form
    FIELD standard = FIELD_unmont(mont);

    // Write 32 bytes in little-endian order
    GLOBAL uchar* out_ptr = output + gid * 32;

    #if FIELD_LIMB_BITS == 32
        for (uint i = 0; i < FIELD_LIMBS; i++) {
            FIELD_limb limb = standard.val[i];
            out_ptr[i * 4 + 0] = (uchar)(limb);
            out_ptr[i * 4 + 1] = (uchar)(limb >> 8);
            out_ptr[i * 4 + 2] = (uchar)(limb >> 16);
            out_ptr[i * 4 + 3] = (uchar)(limb >> 24);
        }
    #elif FIELD_LIMB_BITS == 64
        for (uint i = 0; i < FIELD_LIMBS; i++) {
            FIELD_limb limb = standard.val[i];
            out_ptr[i * 8 + 0] = (uchar)(limb);
            out_ptr[i * 8 + 1] = (uchar)(limb >> 8);
            out_ptr[i * 8 + 2] = (uchar)(limb >> 16);
            out_ptr[i * 8 + 3] = (uchar)(limb >> 24);
            out_ptr[i * 8 + 4] = (uchar)(limb >> 32);
            out_ptr[i * 8 + 5] = (uchar)(limb >> 40);
            out_ptr[i * 8 + 6] = (uchar)(limb >> 48);
            out_ptr[i * 8 + 7] = (uchar)(limb >> 56);
        }
    #else
        #error "Unsupported FIELD_LIMB_BITS value"
    #endif
}

/*
 * Combined fix_var + to_scalar_bytes in one kernel to minimize memory accesses.
 *
 * This performs the fix_var operation and immediately converts the output
 * to scalar bytes format suitable for MSM, without writing intermediate field elements.
 *
 * poly:    Input polynomial in evaluation form (length 2n)
 * out:     Output scalar bytes, 32 bytes per element (total n * 32 bytes)
 * r_buf:   Single-element buffer containing the challenge value r
 * n:       Output length (half of input length)
 *
 * Total threads needed: n (one thread per output element)
 */
KERNEL void FIELD_fix_var_to_scalar(
    GLOBAL FIELD* poly,
    GLOBAL uchar* out,
    GLOBAL FIELD* r_buf,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    // Load r from buffer
    FIELD r = r_buf[0];

    FIELD low = poly[2 * gid];
    FIELD high = poly[2 * gid + 1];

    // diff = high - low
    FIELD diff = FIELD_sub(high, low);

    // scaled = r * diff
    FIELD scaled = FIELD_mul(r, diff);

    // result = low + scaled = low + r * (high - low)
    FIELD result = FIELD_add(low, scaled);

    // Convert to scalar bytes (unmont and serialize)
    FIELD standard = FIELD_unmont(result);

    // Write 32 bytes in little-endian order
    GLOBAL uchar* out_ptr = out + gid * 32;

    #if FIELD_LIMB_BITS == 32
        for (uint i = 0; i < FIELD_LIMBS; i++) {
            FIELD_limb limb = standard.val[i];
            out_ptr[i * 4 + 0] = (uchar)(limb);
            out_ptr[i * 4 + 1] = (uchar)(limb >> 8);
            out_ptr[i * 4 + 2] = (uchar)(limb >> 16);
            out_ptr[i * 4 + 3] = (uchar)(limb >> 24);
        }
    #elif FIELD_LIMB_BITS == 64
        for (uint i = 0; i < FIELD_LIMBS; i++) {
            FIELD_limb limb = standard.val[i];
            out_ptr[i * 8 + 0] = (uchar)(limb);
            out_ptr[i * 8 + 1] = (uchar)(limb >> 8);
            out_ptr[i * 8 + 2] = (uchar)(limb >> 16);
            out_ptr[i * 8 + 3] = (uchar)(limb >> 24);
            out_ptr[i * 8 + 4] = (uchar)(limb >> 32);
            out_ptr[i * 8 + 5] = (uchar)(limb >> 40);
            out_ptr[i * 8 + 6] = (uchar)(limb >> 48);
            out_ptr[i * 8 + 7] = (uchar)(limb >> 56);
        }
    #else
        #error "Unsupported FIELD_LIMB_BITS value"
    #endif
}
