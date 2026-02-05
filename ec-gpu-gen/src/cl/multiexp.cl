/*
 * Same multiexp algorithm used in Bellman, with some modifications.
 * https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L174
 * The CPU version of multiexp parallelism is done by dividing the exponent
 * values into smaller windows, and then applying a sequence of rounds to each
 * window. The GPU kernel not only assigns a thread to each window but also
 * divides the bases into several groups which highly increases the number of
 * threads running in parallel for calculating a multiexp instance.
 */

KERNEL void POINT_multiexp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_jacobian *buckets,
    GLOBAL POINT_jacobian *results,
    GLOBAL EXPONENT *exps,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  const POINT_jacobian local_zero = POINT_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  // Num of elements in each group. Round the number up (ceil).
  const uint len = (n + num_groups - 1) / num_groups;

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint bits = (gid % num_windows) * window_size;
  const ushort w = min((ushort)window_size, (ushort)(EXPONENT_BITS - bits));

  POINT_jacobian res = POINT_ZERO;
  for(uint i = nstart; i < nend; i++) {
    uint ind = EXPONENT_get_bits_lsb(exps[i], bits, w);

    #if defined(OPENCL_NVIDIA) || defined(CUDA)
      // O_o, weird optimization, having a single special case makes it
      // tremendously faster!
      // 511 is chosen because it's half of the maximum bucket len, but
      // any other number works... Bigger indices seems to be better...
      if(ind == 511) buckets[510] = POINT_add_mixed(buckets[510], bases[i]);
      else if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    #else
      if(ind--) buckets[ind] = POINT_add_mixed(buckets[ind], bases[i]);
    #endif
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  POINT_jacobian acc = POINT_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = POINT_add(acc, buckets[j]);
    res = POINT_add(res, acc);
  }

  results[gid] = res;
}

/*
 * Preprocessing kernel: convert raw exponents to signed-digit representation.
 * One thread per base. For each base, scans through all windows and computes
 * signed digits using Booth encoding (carry propagation across windows).
 *
 * Output format: digits[i * num_windows + w] where:
 *   bits[14:0] = |digit| (absolute value, 0 to 2^(w-1))
 *   bit[15]    = sign (1 = negative, 0 = positive)
 *
 * This halves the number of buckets needed in the multiexp kernel from
 * (2^w - 1) to 2^(w-1), roughly halving the summation-by-parts cost.
 */
KERNEL void POINT_preprocess_signed_digits(
    GLOBAL EXPONENT *exps,
    GLOBAL ushort *digits,
    uint n,
    uint num_windows,
    uint window_size) {

  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  const uint half = 1u << (window_size - 1);
  const uint full = 1u << window_size;
  const uint wmask = full - 1;

  uint carry = 0;
  for (uint w = 0; w < num_windows; w++) {
    const uint skip = w * window_size;
    const ushort remaining = (ushort)(EXPONENT_BITS - skip);
    const ushort wbits = (remaining < (ushort)window_size) ? remaining : (ushort)window_size;
    uint bits = EXPONENT_get_bits_lsb(exps[gid], skip, wbits);
    uint val = (bits + carry) & wmask;
    // Handle overflow: if bits + carry >= 2^w, the overflow goes to next window
    // Since bits < 2^w and carry <= 1, bits+carry can be at most 2^w.
    // If bits+carry == 2^w, val=0 and we need carry=1.
    uint overflow = (bits + carry) >> window_size;

    if (val == 0) {
      digits[gid * num_windows + w] = 0;
      carry = overflow;
    } else if (val < half) {
      digits[gid * num_windows + w] = (ushort)val;
      carry = overflow;
    } else {
      // val >= half: use negative digit
      uint digit = full - val;
      digits[gid * num_windows + w] = (ushort)(digit | (1u << 15));
      carry = 1;  // borrow from next window
    }
  }
}

/*
 * Signed-digit multiexp kernel.
 * Same algorithm as POINT_multiexp but uses preprocessed signed digits,
 * halving the number of buckets from (2^w - 1) to 2^(w-1).
 *
 * Point negation for short Weierstrass: negate y-coordinate.
 * In Montgomery form: -y = P - y (where P is the base field modulus).
 */
KERNEL void POINT_multiexp_signed(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_jacobian *buckets,
    GLOBAL POINT_jacobian *results,
    GLOBAL ushort *digits,
    uint n,
    uint num_groups,
    uint num_windows,
    uint window_size) {

  const uint gid = GET_GLOBAL_ID();
  if(gid >= num_windows * num_groups) return;

  // Half the buckets compared to unsigned!
  const uint bucket_len = 1 << (window_size - 1);

  buckets += bucket_len * gid;

  const POINT_jacobian local_zero = POINT_ZERO;
  for(uint i = 0; i < bucket_len; i++) buckets[i] = local_zero;

  const uint len = (n + num_groups - 1) / num_groups;
  const uint nstart = len * (gid / num_windows);
  const uint nend = min(nstart + len, n);
  const uint window = gid % num_windows;

  POINT_jacobian res = POINT_ZERO;
  for(uint i = nstart; i < nend; i++) {
    ushort raw = digits[i * num_windows + window];
    uint ind = raw & 0x7FFF;
    uint sign = (raw >> 15) & 1;

    if(ind > 0) {
      POINT_affine base = bases[i];
      if(sign) {
        // Negate y-coordinate: y = P - y (base field modular negation)
        base.y = FIELD_sub(FIELD_ZERO, base.y);
      }

      #if defined(OPENCL_NVIDIA) || defined(CUDA)
        if(ind == (bucket_len >> 1)) buckets[(bucket_len >> 1) - 1] = POINT_add_mixed(buckets[(bucket_len >> 1) - 1], base);
        else buckets[ind - 1] = POINT_add_mixed(buckets[ind - 1], base);
      #else
        buckets[ind - 1] = POINT_add_mixed(buckets[ind - 1], base);
      #endif
    }
  }

  // Summation by parts — only half the iterations!
  POINT_jacobian acc = POINT_ZERO;
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = POINT_add(acc, buckets[j]);
    res = POINT_add(res, acc);
  }

  results[gid] = res;
}

/*
 * ============================================================================
 * Sort-based MSM kernels
 * ============================================================================
 * These kernels implement a sort-based bucket accumulation strategy that
 * replaces the per-thread private bucket approach with:
 *   1. Decompose signed digits into (bucket_key, base_index) pairs
 *   2. Counting sort by bucket_key
 *   3. One thread per bucket reads sorted bases sequentially
 *   4. Summation-by-parts on bucket results per window
 *   5. Horner reduction across windows on GPU
 *
 * Benefits: sequential memory reads (sorted), no write conflicts,
 * empty buckets skipped, natural load balancing for large buckets.
 */

/*
 * Phase 1: Decompose signed digits into (key, value) pairs for sorting.
 *
 * One thread per (base, window) pair.
 * Output key = window * num_buckets_per_window + (|digit| - 1)
 * Output value = base_index | (sign << 31)
 * Zero digits get key = 0xFFFFFFFF (sentinel, sorted to end and skipped).
 */
KERNEL void POINT_decompose_to_pairs(
    GLOBAL ushort *digits,
    GLOBAL uint *keys,
    GLOBAL uint *values,
    uint n,
    uint num_windows,
    uint buckets_per_window) {
  const uint gid = GET_GLOBAL_ID();
  const uint total = n * num_windows;
  if (gid >= total) return;

  const uint base_idx = gid / num_windows;
  const uint window = gid % num_windows;

  ushort raw = digits[base_idx * num_windows + window];
  uint ind = raw & 0x7FFF;

  if (ind == 0) {
    keys[gid] = 0xFFFFFFFF;
    values[gid] = 0;
  } else {
    uint sign = (raw >> 15) & 1;
    keys[gid] = window * buckets_per_window + (ind - 1);
    values[gid] = base_idx | (sign << 31);
  }
}

/*
 * Phase 2a: Count how many pairs fall into each bucket.
 *
 * One thread per pair. Atomically increments the count for its bucket.
 * Total buckets = num_windows * buckets_per_window.
 * Pairs with key = 0xFFFFFFFF are skipped (zero digits).
 */
KERNEL void POINT_count_buckets(
    GLOBAL uint *keys,
    GLOBAL uint *counts,
    uint total_pairs) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= total_pairs) return;

  uint key = keys[gid];
  if (key != 0xFFFFFFFF) {
    #ifdef CUDA
    atomicAdd(&counts[key], 1u);
    #else
    atomic_add(&counts[key], 1u);
    #endif
  }
}

/*
 * Phase 2b: Exclusive prefix sum on bucket counts.
 *
 * Single-thread kernel (launched with 1 thread) that computes offsets.
 * Also counts total non-empty buckets for the accumulation phase.
 *
 * Output:
 *   offsets[i] = sum of counts[0..i) (exclusive prefix sum)
 *   num_nonempty[0] = number of non-empty buckets
 *   nonempty_bucket_ids[j] = bucket index of j-th non-empty bucket
 */
KERNEL void POINT_prefix_sum(
    GLOBAL uint *counts,
    GLOBAL uint *offsets,
    GLOBAL uint *nonempty_bucket_ids,
    GLOBAL uint *num_nonempty,
    uint total_buckets) {
  const uint gid = GET_GLOBAL_ID();
  if (gid != 0) return;

  uint running = 0;
  uint ne_count = 0;
  for (uint i = 0; i < total_buckets; i++) {
    offsets[i] = running;
    if (counts[i] > 0) {
      nonempty_bucket_ids[ne_count] = i;
      ne_count++;
    }
    running += counts[i];
  }
  num_nonempty[0] = ne_count;
}

/*
 * Phase 2c: Scatter pairs into sorted order using the computed offsets.
 *
 * One thread per pair. Atomically increments offset to get its position
 * in the sorted array.
 */
KERNEL void POINT_scatter_to_sorted(
    GLOBAL uint *keys,
    GLOBAL uint *values,
    GLOBAL uint *offsets,
    GLOBAL uint *sorted_values,
    uint total_pairs) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= total_pairs) return;

  uint key = keys[gid];
  if (key != 0xFFFFFFFF) {
    uint pos;
    #ifdef CUDA
    pos = atomicAdd(&offsets[key], 1u);
    #else
    pos = atomic_add(&offsets[key], 1u);
    #endif
    sorted_values[pos] = values[gid];
  }
}

/*
 * Phase 3: Accumulate sorted bases into bucket results.
 *
 * One thread per non-empty bucket. Reads consecutive sorted base indices
 * and accumulates using XYZZ coordinates with POINT_xyzz_add_mixed.
 * XYZZ saves 2 squarings per mixed addition vs Jacobian (7M+2S vs 7M+4S).
 *
 * Output is converted from XYZZ → Jacobian using the identity:
 *   (X_jac, Y_jac, Z_jac) = (X * ZZ, Y * ZZZ, ZZ)
 * which costs only 2 field multiplications per bucket.
 *
 * Uses original offsets (before scatter modified them) stored in counts_copy
 * and bucket sizes from counts.
 */
KERNEL void POINT_accumulate_sorted_buckets(
    GLOBAL POINT_affine *bases,
    GLOBAL uint *sorted_values,
    GLOBAL uint *bucket_offsets,
    GLOBAL uint *bucket_sizes,
    GLOBAL uint *nonempty_bucket_ids,
    GLOBAL POINT_jacobian *bucket_results,
    uint num_nonempty) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= num_nonempty) return;

  uint bid = nonempty_bucket_ids[gid];
  uint start = bucket_offsets[bid];
  uint count = bucket_sizes[bid];

  POINT_xyzz acc = POINT_XYZZ_ZERO;
  for (uint j = 0; j < count; j++) {
    uint val = sorted_values[start + j];
    uint base_idx = val & 0x7FFFFFFF;
    uint sign = (val >> 31) & 1;

    POINT_affine base = bases[base_idx];
    if (sign) {
      base.y = FIELD_sub(FIELD_ZERO, base.y);
    }
    acc = POINT_xyzz_add_mixed(acc, base);
  }

  // Convert XYZZ → Jacobian: (X*ZZ, Y*ZZZ, ZZ), costs 2M
  const FIELD local_zero = FIELD_ZERO;
  if (FIELD_eq(acc.zz, local_zero)) {
    bucket_results[bid] = POINT_ZERO;
  } else {
    POINT_jacobian jac;
    jac.x = FIELD_mul(acc.x, acc.zz);
    jac.y = FIELD_mul(acc.y, acc.zzz);
    jac.z = acc.zz;
    bucket_results[bid] = jac;
  }
}

/*
 * Phase 4: Summation-by-parts per window.
 *
 * One thread per window. For window w, iterates over buckets
 * [w * buckets_per_window .. (w+1) * buckets_per_window) in reverse
 * and computes the weighted sum using the running-sum technique.
 *
 * Output: one result per window in window_results.
 */
KERNEL void POINT_reduce_buckets_by_window(
    GLOBAL POINT_jacobian *bucket_results,
    GLOBAL POINT_jacobian *window_results,
    uint num_windows,
    uint buckets_per_window) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= num_windows) return;

  const uint window = gid;
  const uint base_bucket = window * buckets_per_window;

  POINT_jacobian acc = POINT_ZERO;
  POINT_jacobian res = POINT_ZERO;

  for (int j = (int)buckets_per_window - 1; j >= 0; j--) {
    acc = POINT_add(acc, bucket_results[base_bucket + (uint)j]);
    res = POINT_add(res, acc);
  }

  window_results[window] = res;
}

/*
 * Phase 5: Horner reduction across windows (GPU-side).
 *
 * Single-thread kernel that combines window results using Horner's method:
 *   result = sum over windows (MSB-first):
 *     result = result * 2^w + window_result[i]
 *
 * This eliminates downloading partial results and CPU accumulation.
 */
KERNEL void POINT_reduce_windows(
    GLOBAL POINT_jacobian *window_results,
    GLOBAL POINT_jacobian *final_result,
    uint num_windows,
    uint window_size,
    uint effective_bits) {
  const uint gid = GET_GLOBAL_ID();
  if (gid != 0) return;

  POINT_jacobian acc = POINT_ZERO;
  for (int i = (int)num_windows - 1; i >= 0; i--) {
    uint w = window_size;
    uint remaining = effective_bits - (uint)i * window_size;
    if (w > remaining) w = remaining;
    for (uint d = 0; d < w; d++) {
      acc = POINT_double(acc);
    }
    acc = POINT_add(acc, window_results[i]);
  }

  final_result[0] = acc;
}

/*
 * Phase 3b: Chunked bucket accumulation for large bucket splitting.
 *
 * Instead of 1 thread per bucket, launches 1 thread per (bucket, chunk) pair.
 * Each thread accumulates chunk_size consecutive points from its bucket portion.
 * Results are stored in partial_results[chunk_idx_within_dispatch].
 *
 * The dispatch table maps each thread to a (bucket_id, chunk_start, chunk_count) triplet.
 * dispatch_table layout: [bucket_id, start_within_bucket, count] * num_dispatches
 *
 * After this kernel, POINT_reduce_partial_buckets combines partial results per bucket.
 */
KERNEL void POINT_accumulate_chunked(
    GLOBAL POINT_affine *bases,
    GLOBAL uint *sorted_values,
    GLOBAL uint *bucket_offsets,
    GLOBAL uint *dispatch_table,
    GLOBAL POINT_jacobian *partial_results,
    uint num_dispatches) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= num_dispatches) return;

  uint bid = dispatch_table[gid * 3 + 0];
  uint chunk_start = dispatch_table[gid * 3 + 1];
  uint chunk_count = dispatch_table[gid * 3 + 2];

  uint bucket_start = bucket_offsets[bid];

  POINT_xyzz acc = POINT_XYZZ_ZERO;
  for (uint j = 0; j < chunk_count; j++) {
    uint val = sorted_values[bucket_start + chunk_start + j];
    uint base_idx = val & 0x7FFFFFFF;
    uint sign = (val >> 31) & 1;

    POINT_affine base = bases[base_idx];
    if (sign) {
      base.y = FIELD_sub(FIELD_ZERO, base.y);
    }
    acc = POINT_xyzz_add_mixed(acc, base);
  }

  // Convert XYZZ → Jacobian: (X*ZZ, Y*ZZZ, ZZ), costs 2M
  const FIELD local_zero = FIELD_ZERO;
  if (FIELD_eq(acc.zz, local_zero)) {
    partial_results[gid] = POINT_ZERO;
  } else {
    POINT_jacobian jac;
    jac.x = FIELD_mul(acc.x, acc.zz);
    jac.y = FIELD_mul(acc.y, acc.zzz);
    jac.z = acc.zz;
    partial_results[gid] = jac;
  }
}

/*
 * Phase 3c: Reduce partial results for each bucket.
 *
 * One thread per non-empty bucket. Reads consecutive partial results
 * for that bucket and reduces them with POINT_add.
 *
 * reduce_table maps each bucket to its range in partial_results:
 *   reduce_table[gid * 2 + 0] = start index in partial_results
 *   reduce_table[gid * 2 + 1] = number of partial results for this bucket
 *
 * Output: bucket_results[nonempty_bucket_ids[gid]] = reduced result
 */
KERNEL void POINT_reduce_partial_buckets(
    GLOBAL POINT_jacobian *partial_results,
    GLOBAL uint *nonempty_bucket_ids,
    GLOBAL uint *reduce_table,
    GLOBAL POINT_jacobian *bucket_results,
    uint num_nonempty) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= num_nonempty) return;

  uint bid = nonempty_bucket_ids[gid];
  uint start = reduce_table[gid * 2 + 0];
  uint count = reduce_table[gid * 2 + 1];

  POINT_jacobian acc = POINT_ZERO;
  for (uint j = 0; j < count; j++) {
    acc = POINT_add(acc, partial_results[start + j]);
  }

  bucket_results[bid] = acc;
}

/*
 * Precompute negated affine bases.
 *
 * For each affine point (x, y), outputs (x, -y) where -y = FIELD_sub(FIELD_ZERO, y).
 * This avoids runtime negation in the accumulate kernel for signed digits.
 *
 * In HyperKZG, the same bases are used for ~25 MSMs, so the precomputation
 * cost is amortized over all MSMs in a single GPU session.
 */
KERNEL void POINT_negate_bases(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_affine *neg_bases,
    uint n) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  POINT_affine p = bases[gid];
  p.y = FIELD_sub(FIELD_ZERO, p.y);
  neg_bases[gid] = p;
}

/*
 * Phase 3 (optimized): Accumulate sorted bases using precomputed negations.
 *
 * Same as POINT_accumulate_sorted_buckets but reads from neg_bases for
 * negative digits instead of computing -y at runtime. Saves one FIELD_sub
 * per negated point.
 */
KERNEL void POINT_accumulate_sorted_buckets_precomp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_affine *neg_bases,
    GLOBAL uint *sorted_values,
    GLOBAL uint *bucket_offsets,
    GLOBAL uint *bucket_sizes,
    GLOBAL uint *nonempty_bucket_ids,
    GLOBAL POINT_jacobian *bucket_results,
    uint num_nonempty) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= num_nonempty) return;

  uint bid = nonempty_bucket_ids[gid];
  uint start = bucket_offsets[bid];
  uint count = bucket_sizes[bid];

  POINT_xyzz acc = POINT_XYZZ_ZERO;
  for (uint j = 0; j < count; j++) {
    uint val = sorted_values[start + j];
    uint base_idx = val & 0x7FFFFFFF;
    uint sign = (val >> 31) & 1;

    POINT_affine base = sign ? neg_bases[base_idx] : bases[base_idx];
    acc = POINT_xyzz_add_mixed(acc, base);
  }

  // Convert XYZZ → Jacobian: (X*ZZ, Y*ZZZ, ZZ), costs 2M
  const FIELD local_zero = FIELD_ZERO;
  if (FIELD_eq(acc.zz, local_zero)) {
    bucket_results[bid] = POINT_ZERO;
  } else {
    POINT_jacobian jac;
    jac.x = FIELD_mul(acc.x, acc.zz);
    jac.y = FIELD_mul(acc.y, acc.zzz);
    jac.z = acc.zz;
    bucket_results[bid] = jac;
  }
}

/*
 * Phase 3b (optimized): Chunked accumulation with precomputed negations.
 */
KERNEL void POINT_accumulate_chunked_precomp(
    GLOBAL POINT_affine *bases,
    GLOBAL POINT_affine *neg_bases,
    GLOBAL uint *sorted_values,
    GLOBAL uint *bucket_offsets,
    GLOBAL uint *dispatch_table,
    GLOBAL POINT_jacobian *partial_results,
    uint num_dispatches) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= num_dispatches) return;

  uint bid = dispatch_table[gid * 3 + 0];
  uint chunk_start = dispatch_table[gid * 3 + 1];
  uint chunk_count = dispatch_table[gid * 3 + 2];

  uint bucket_start = bucket_offsets[bid];

  POINT_xyzz acc = POINT_XYZZ_ZERO;
  for (uint j = 0; j < chunk_count; j++) {
    uint val = sorted_values[bucket_start + chunk_start + j];
    uint base_idx = val & 0x7FFFFFFF;
    uint sign = (val >> 31) & 1;

    POINT_affine base = sign ? neg_bases[base_idx] : bases[base_idx];
    acc = POINT_xyzz_add_mixed(acc, base);
  }

  const FIELD local_zero = FIELD_ZERO;
  if (FIELD_eq(acc.zz, local_zero)) {
    partial_results[gid] = POINT_ZERO;
  } else {
    POINT_jacobian jac;
    jac.x = FIELD_mul(acc.x, acc.zz);
    jac.y = FIELD_mul(acc.y, acc.zzz);
    jac.z = acc.zz;
    partial_results[gid] = jac;
  }
}

/*
 * Batch scalar multiplication: compute s[i] * G for each scalar s[i].
 * Uses windowed lookup table for efficiency.
 *
 * The table is organized as: table[outer][inner] = (2^(outer*window)) * inner * G
 * where outer in [0, num_windows) and inner in [0, 2^window).
 *
 * Parameters:
 * - table: Precomputed lookup table, flattened as [num_windows * (1 << window)] affine points
 * - scalars: Input scalars in standard (non-Montgomery) form
 * - results: Output points in Jacobian form
 * - n: Number of scalars
 * - window: Window size in bits
 * - num_windows: Number of windows (ceil(scalar_bits / window))
 * - scalar_bits: Number of bits in the scalar field (unused, for documentation)
 */
KERNEL void POINT_batch_scalar_mul(
    GLOBAL POINT_affine *table,
    GLOBAL EXPONENT *scalars,
    GLOBAL POINT_jacobian *results,
    uint n,
    uint window,
    uint num_windows,
    uint scalar_bits) {
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;

  const uint in_window = 1u << window;
  EXPONENT scalar = scalars[gid];

  POINT_jacobian acc = POINT_ZERO;

  for (uint outer = 0; outer < num_windows; outer++) {
    // Extract window bits from scalar using the existing helper
    uint inner = EXPONENT_get_bits_lsb(scalar, outer * window, window);

    if (inner > 0) {
      // Lookup table[outer * in_window + inner]
      POINT_affine entry = table[outer * in_window + inner];
      acc = POINT_add_mixed(acc, entry);
    }
  }

  results[gid] = acc;
}
