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
 * Fix variable with indexed challenge from a challenges buffer.
 * Same as fix_var but reads r from challenges[challenge_idx] instead of r_buf[0].
 *
 * This avoids uploading one challenge at a time — all challenges are uploaded once.
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

    FIELD r = challenges[challenge_idx];

    FIELD low = poly[2 * gid];
    FIELD high = poly[2 * gid + 1];

    FIELD diff = FIELD_sub(high, low);
    FIELD scaled = FIELD_mul(r, diff);
    out[gid] = FIELD_add(low, scaled);
}

/*
 * Convert field elements from Montgomery form to standard (non-Montgomery) form.
 * Output has the same limb layout as the EXPONENT type used by MSM.
 *
 * This allows scalar conversion to happen on GPU instead of downloading Fr values,
 * calling into_bigint() on CPU, and re-uploading.
 */
KERNEL void FIELD_to_scalar_bytes(
    GLOBAL FIELD* input,
    GLOBAL FIELD* output,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    output[gid] = FIELD_unmont(input[gid]);
}

/*
 * Convert field elements from Montgomery to standard form, reading from an offset.
 * Reads from input[offset + gid] and writes to output[gid].
 *
 * Used for extracting individual witness polynomials from a flattened buffer
 * (where multiple witnesses are stored contiguously).
 */
KERNEL void FIELD_to_scalar_bytes_offset(
    GLOBAL FIELD* input,
    GLOBAL FIELD* output,
    uint n,
    uint offset)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    output[gid] = FIELD_unmont(input[offset + gid]);
}

/*
 * Batch witness polynomial computation for multiple evaluation points.
 * For each point u_i, computes h_i(x) where f(x) = h_i(x) * (x - u_i) + f(u_i).
 *
 * Each thread handles one evaluation point (sequential recurrence per point).
 * Output: witnesses is a flattened buffer of num_points * (n-1) elements.
 *
 * f:         input polynomial of length n
 * points:    evaluation points [u_0, u_1, ..., u_{num_points-1}]
 * witnesses: output buffer, witnesses[i*(n-1) + j] = h_i[j]
 * n:         polynomial length
 * num_points: number of evaluation points
 */
KERNEL void FIELD_witness_poly_batch(
    GLOBAL FIELD* f,
    GLOBAL FIELD* points,
    GLOBAL FIELD* witnesses,
    uint n,
    uint num_points)
{
    const uint pid = GET_GLOBAL_ID();
    if (pid >= num_points) return;

    FIELD u = points[pid];
    uint witness_len = n - 1;
    uint out_offset = pid * witness_len;

    FIELD carry = FIELD_ZERO;
    for (int i = n - 1; i >= 1; i--) {
        carry = FIELD_add(f[i], FIELD_mul(carry, u));
        witnesses[out_offset + i - 1] = carry;
    }
}
