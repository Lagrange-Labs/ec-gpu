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
