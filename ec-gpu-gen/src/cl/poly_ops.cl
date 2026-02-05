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

/*
 * =====================================================================
 * Parallel witness polynomial computation (3-phase approach).
 *
 * The witness recurrence h[i-1] = f[i] + h[i]*u is sequential per-point.
 * We parallelize it by splitting f into chunks, processing each independently
 * (assuming carry_in=0), then propagating carries across chunks.
 *
 * For num_points evaluation points and C chunks, Phase 1 launches
 * num_points * C threads (massive parallelism vs 3 threads before).
 * =====================================================================
 */

/*
 * Phase 1: Each thread processes one (point, chunk) pair independently.
 *
 * For chunk c of eval point p, processes f[chunk_start..chunk_end] in reverse,
 * computing local witness values assuming carry_in = 0.
 * Stores the local carry-out in carries[p * num_chunks + c].
 *
 * witnesses: output, witnesses[p*(n-1) + j] = local h[j] (before carry correction)
 * carries:   output, carries[p*num_chunks + c] = carry-out of chunk c for point p
 * f:         input polynomial of length n
 * points:    evaluation points
 * n:         polynomial length
 * num_points: number of eval points
 * chunk_size: elements per chunk
 * num_chunks: total number of chunks
 */
KERNEL void FIELD_witness_poly_batch_phase1(
    GLOBAL FIELD* f,
    GLOBAL FIELD* witnesses,
    GLOBAL FIELD* carries,
    GLOBAL FIELD* points,
    uint n,
    uint num_points,
    uint chunk_size,
    uint num_chunks)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= num_points * num_chunks) return;

    const uint pid = gid / num_chunks;
    const uint cid = gid % num_chunks;

    FIELD u = points[pid];
    uint witness_len = n - 1;
    uint out_offset = pid * witness_len;

    /*
     * Chunks partition the index range [1, n-1] (the valid range for the
     * recurrence h[i-1] = f[i] + h[i]*u).
     *
     * Chunk 0 covers the HIGHEST indices (rightmost), chunk (num_chunks-1) the lowest.
     * This matches the right-to-left scan direction of the recurrence.
     *
     * chunk_start_idx / chunk_end_idx are the f[] indices this chunk processes.
     */
    uint chunk_end_idx = n - cid * chunk_size;              /* inclusive upper bound in f */
    uint chunk_start_raw = (chunk_end_idx > chunk_size) ? (chunk_end_idx - chunk_size) : 0;
    uint chunk_start_idx = (chunk_start_raw < 1) ? 1 : chunk_start_raw;  /* f index >= 1 */

    FIELD carry = FIELD_ZERO;
    for (int i = (int)chunk_end_idx - 1; i >= (int)chunk_start_idx; i--) {
        carry = FIELD_add(f[i], FIELD_mul(carry, u));
        witnesses[out_offset + i - 1] = carry;
    }

    carries[pid * num_chunks + cid] = carry;
}

/*
 * Phase 2: Carry propagation across chunks (one thread per eval point).
 *
 * Processes chunks right-to-left. The carry from chunk c is multiplied by
 * u^(elements_in_chunk_{c+1}) and added to chunk c+1's carry, producing
 * propagated_carries that represent the cumulative correction for each chunk.
 *
 * carries:           input, carries[p*num_chunks + c] from Phase 1
 * propagated_carries: output, propagated_carries[p*num_chunks + c] = correction multiplier
 * points:            evaluation points
 * num_chunks:        number of chunks
 * num_points:        number of eval points
 * chunk_size:        elements per chunk
 * n:                 polynomial length
 */
KERNEL void FIELD_witness_carry_propagate(
    GLOBAL FIELD* carries,
    GLOBAL FIELD* propagated_carries,
    GLOBAL FIELD* points,
    uint num_chunks,
    uint num_points,
    uint chunk_size,
    uint n)
{
    const uint pid = GET_GLOBAL_ID();
    if (pid >= num_points) return;

    FIELD u = points[pid];

    /* propagated_carries[chunk 0] = ZERO (rightmost chunk has no incoming carry) */
    propagated_carries[pid * num_chunks + 0] = FIELD_ZERO;

    /* carry_in tracks the actual carry entering each chunk.
     *
     * For chunk c, carry_in = carries[c-1] + carry_in_prev * u^(size_chunk_(c-1))
     * because the previous chunk's local carry-out (carries[c-1]) is what h would be
     * at its left boundary with zero carry-in, and the carry_in to that chunk propagates
     * through size_chunk_(c-1) recurrence steps, each multiplying by u.
     *
     * propagated_carries[c] = carry_in * u, because Phase 3 applies the correction
     * starting at the rightmost element of the chunk: the first correction is
     * carry_in * u (one recurrence step from the boundary).
     */
    FIELD carry_in = FIELD_ZERO;
    for (uint c = 1; c < num_chunks; c++) {
        /* Size of the previous chunk (c-1) */
        uint prev_end = n - (c - 1) * chunk_size;
        uint prev_start_raw = (prev_end > chunk_size) ? (prev_end - chunk_size) : 0;
        uint prev_start = (prev_start_raw < 1) ? 1 : prev_start_raw;
        uint prev_size = prev_end - prev_start;

        /* u^prev_size: carry_in propagates through prev_size recurrence steps */
        FIELD u_power = FIELD_ONE;
        for (uint k = 0; k < prev_size; k++) {
            u_power = FIELD_mul(u_power, u);
        }

        /* carry_in for chunk c = local_carry_out[c-1] + carry_in_prev * u^prev_size */
        carry_in = FIELD_add(carries[pid * num_chunks + c - 1], FIELD_mul(carry_in, u_power));

        /* Phase 3 applies: correction = prop_carry, then *= u each step.
         * The rightmost element needs carry_in * u, so prop_carry = carry_in * u. */
        propagated_carries[pid * num_chunks + c] = FIELD_mul(carry_in, u);
    }
}

/*
 * Phase 3: Apply carry corrections to each chunk's witness values.
 *
 * For each witness element h[i-1] in chunk c of point p:
 *   h[i-1] += propagated_carry[c] * u^(position within chunk from the right)
 *
 * Actually, the correction is simpler: the propagated carry acts as if it were
 * the carry_in to the chunk. So for element at position j within the chunk
 * (counting from the END of the chunk), the correction is:
 *   h[i-1] += propagated_carry * u^j
 *
 * We process each chunk's elements right-to-left, accumulating the correction.
 *
 * witnesses:           in/out, witnesses[p*(n-1) + j]
 * propagated_carries:  input, from Phase 2
 * points:              evaluation points
 * n:                   polynomial length
 * num_points:          number of eval points
 * chunk_size:          elements per chunk
 * num_chunks:          total number of chunks
 */
KERNEL void FIELD_witness_poly_batch_phase3(
    GLOBAL FIELD* witnesses,
    GLOBAL FIELD* propagated_carries,
    GLOBAL FIELD* points,
    uint n,
    uint num_points,
    uint chunk_size,
    uint num_chunks)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= num_points * num_chunks) return;

    const uint pid = gid / num_chunks;
    const uint cid = gid % num_chunks;

    /* Skip chunk 0 — it has no correction (propagated_carry = 0) */
    if (cid == 0) return;

    FIELD u = points[pid];
    uint witness_len = n - 1;
    uint out_offset = pid * witness_len;

    FIELD prop_carry = propagated_carries[pid * num_chunks + cid];

    /* Chunk boundaries (same logic as Phase 1) */
    uint chunk_end_idx = n - cid * chunk_size;
    uint chunk_start_raw = (chunk_end_idx > chunk_size) ? (chunk_end_idx - chunk_size) : 0;
    uint chunk_start_idx = (chunk_start_raw < 1) ? 1 : chunk_start_raw;

    /* Apply correction: scan right-to-left within chunk, accumulating prop_carry * u^j */
    FIELD correction = prop_carry;
    for (int i = (int)chunk_end_idx - 1; i >= (int)chunk_start_idx; i--) {
        witnesses[out_offset + i - 1] = FIELD_add(witnesses[out_offset + i - 1], correction);
        correction = FIELD_mul(correction, u);
    }
}

/*
 * Copy a polynomial into a padded flat buffer on GPU.
 * Copies src[0..src_len] into dst[poly_idx * dst_stride .. poly_idx * dst_stride + src_len],
 * and zero-fills the remainder up to dst_stride.
 *
 * Each thread handles one element position within dst_stride.
 */
KERNEL void FIELD_copy_and_pad(
    GLOBAL FIELD* src,
    GLOBAL FIELD* dst,
    uint src_len,
    uint dst_stride,
    uint poly_idx)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= dst_stride) return;

    uint dst_offset = poly_idx * dst_stride + gid;
    if (gid < src_len) {
        dst[dst_offset] = src[gid];
    } else {
        dst[dst_offset] = FIELD_ZERO;
    }
}

/*
 * Streaming linear combination: accumulate one polynomial at a time.
 * Performs: out[i] += coeff * poly[i] for i in [0, poly_len)
 * where poly is zero-padded beyond src_len.
 *
 * This avoids allocating a massive flat buffer for all polynomials.
 * Instead, we accumulate each polynomial one at a time into the output.
 *
 * poly:     input polynomial (length src_len, conceptually zero-padded to poly_len)
 * out:      in/out accumulator (length poly_len)
 * coeff:    scalar coefficient buffer (single element)
 * src_len:  actual length of input polynomial
 * poly_len: target length (for bounds)
 */
KERNEL void FIELD_linear_combine_accumulate(
    GLOBAL FIELD* poly,
    GLOBAL FIELD* out,
    GLOBAL FIELD* coeff_buf,
    uint src_len,
    uint poly_len)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= poly_len) return;

    FIELD coeff = coeff_buf[0];

    if (gid < src_len) {
        FIELD scaled = FIELD_mul(poly[gid], coeff);
        out[gid] = FIELD_add(out[gid], scaled);
    }
    /* Elements beyond src_len contribute 0 * coeff = 0, so no change needed */
}

/*
 * Convert field elements from Montgomery to standard form AND compute
 * the maximum number of bits used by any scalar, via atomic max.
 *
 * This allows the caller to dynamically determine num_windows for
 * the MSM pipeline, reducing unnecessary work when scalars are small.
 *
 * max_bits_out must be initialized to 0 before calling this kernel.
 * After kernel completion, max_bits_out[0] contains the bit position
 * of the highest set bit across all scalars (1-indexed), or 0 if all zero.
 */
KERNEL void FIELD_to_scalar_bytes_max_bits(
    GLOBAL FIELD* input,
    GLOBAL FIELD* output,
    GLOBAL uint* max_bits_out,
    uint n)
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= n) return;

    FIELD val = FIELD_unmont(input[gid]);
    output[gid] = val;

    // Find highest set bit in this scalar
    uint my_bits = 0;
    for (int i = FIELD_LIMBS - 1; i >= 0; i--) {
        if (val.val[i] != 0) {
            // Count leading zeros to find highest bit
            // For 32-bit limbs: bit position = i*LIMB_BITS + (LIMB_BITS - clz(val))
            // For 64-bit limbs: same formula with 64
            uint v = (uint)val.val[i];
#if FIELD_LIMB_BITS == 64
            // For 64-bit, check high 32 bits first
            uint hi = (uint)(val.val[i] >> 32);
            if (hi != 0) {
                uint pos = 0;
                uint tmp = hi;
                while (tmp > 0) { pos++; tmp >>= 1; }
                my_bits = (uint)i * FIELD_LIMB_BITS + 32 + pos;
            } else {
                uint lo = (uint)val.val[i];
                uint pos = 0;
                while (lo > 0) { pos++; lo >>= 1; }
                my_bits = (uint)i * FIELD_LIMB_BITS + pos;
            }
#else
            uint pos = 0;
            while (v > 0) { pos++; v >>= 1; }
            my_bits = (uint)i * FIELD_LIMB_BITS + pos;
#endif
            break;
        }
    }

    if (my_bits > 0) {
#ifdef CUDA
        atomicMax(max_bits_out, my_bits);
#else
        atomic_max(max_bits_out, my_bits);
#endif
    }
}
