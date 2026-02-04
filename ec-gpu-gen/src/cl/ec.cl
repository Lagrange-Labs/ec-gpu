// Elliptic curve operations (Short Weierstrass Jacobian form)

#define POINT_ZERO ((POINT_jacobian){FIELD_ZERO, FIELD_ONE, FIELD_ZERO})

typedef struct {
  FIELD x;
  FIELD y;
} POINT_affine;

typedef struct {
  FIELD x;
  FIELD y;
  FIELD z;
} POINT_jacobian;

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#doubling-dbl-2009-l
DEVICE POINT_jacobian POINT_double(POINT_jacobian inp) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(inp.z, local_zero)) {
      return inp;
  }

  const FIELD a = FIELD_sqr(inp.x); // A = X1^2
  const FIELD b = FIELD_sqr(inp.y); // B = Y1^2
  FIELD c = FIELD_sqr(b); // C = B^2

  // D = 2*((X1+B)2-A-C)
  FIELD d = FIELD_add(inp.x, b);
  d = FIELD_sqr(d); d = FIELD_sub(FIELD_sub(d, a), c); d = FIELD_double(d);

  const FIELD e = FIELD_add(FIELD_double(a), a); // E = 3*A
  const FIELD f = FIELD_sqr(e);

  inp.z = FIELD_mul(inp.y, inp.z); inp.z = FIELD_double(inp.z); // Z3 = 2*Y1*Z1
  inp.x = FIELD_sub(FIELD_sub(f, d), d); // X3 = F-2*D

  // Y3 = E*(D-X3)-8*C
  c = FIELD_double(c); c = FIELD_double(c); c = FIELD_double(c);
  inp.y = FIELD_sub(FIELD_mul(FIELD_sub(d, inp.x), e), c);

  return inp;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-madd-2007-bl
DEVICE POINT_jacobian POINT_add_mixed(POINT_jacobian a, POINT_affine b) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) {
    const FIELD local_one = FIELD_ONE;
    a.x = b.x;
    a.y = b.y;
    a.z = local_one;
    return a;
  }

  const FIELD z1z1 = FIELD_sqr(a.z);
  const FIELD u2 = FIELD_mul(b.x, z1z1);
  const FIELD s2 = FIELD_mul(FIELD_mul(b.y, a.z), z1z1);

  if(FIELD_eq(a.x, u2) && FIELD_eq(a.y, s2)) {
      return POINT_double(a);
  }

  const FIELD h = FIELD_sub(u2, a.x); // H = U2-X1
  const FIELD hh = FIELD_sqr(h); // HH = H^2
  FIELD i = FIELD_double(hh); i = FIELD_double(i); // I = 4*HH
  FIELD j = FIELD_mul(h, i); // J = H*I
  FIELD r = FIELD_sub(s2, a.y); r = FIELD_double(r); // r = 2*(S2-Y1)
  const FIELD v = FIELD_mul(a.x, i);

  POINT_jacobian ret;

  // X3 = r^2 - J - 2*V
  ret.x = FIELD_sub(FIELD_sub(FIELD_sqr(r), j), FIELD_double(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = FIELD_mul(a.y, j); j = FIELD_double(j);
  ret.y = FIELD_sub(FIELD_mul(FIELD_sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = FIELD_add(a.z, h); ret.z = FIELD_sub(FIELD_sub(FIELD_sqr(ret.z), z1z1), hh);
  return ret;
}

// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-jacobian-0.html#addition-add-2007-bl
DEVICE POINT_jacobian POINT_add(POINT_jacobian a, POINT_jacobian b) {

  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.z, local_zero)) return b;
  if(FIELD_eq(b.z, local_zero)) return a;

  const FIELD z1z1 = FIELD_sqr(a.z); // Z1Z1 = Z1^2
  const FIELD z2z2 = FIELD_sqr(b.z); // Z2Z2 = Z2^2
  const FIELD u1 = FIELD_mul(a.x, z2z2); // U1 = X1*Z2Z2
  const FIELD u2 = FIELD_mul(b.x, z1z1); // U2 = X2*Z1Z1
  FIELD s1 = FIELD_mul(FIELD_mul(a.y, b.z), z2z2); // S1 = Y1*Z2*Z2Z2
  const FIELD s2 = FIELD_mul(FIELD_mul(b.y, a.z), z1z1); // S2 = Y2*Z1*Z1Z1

  if(FIELD_eq(u1, u2) && FIELD_eq(s1, s2))
    return POINT_double(a);
  else {
    const FIELD h = FIELD_sub(u2, u1); // H = U2-U1
    FIELD i = FIELD_double(h); i = FIELD_sqr(i); // I = (2*H)^2
    const FIELD j = FIELD_mul(h, i); // J = H*I
    FIELD r = FIELD_sub(s2, s1); r = FIELD_double(r); // r = 2*(S2-S1)
    const FIELD v = FIELD_mul(u1, i); // V = U1*I
    a.x = FIELD_sub(FIELD_sub(FIELD_sub(FIELD_sqr(r), j), v), v); // X3 = r^2 - J - 2*V

    // Y3 = r*(V - X3) - 2*S1*J
    a.y = FIELD_mul(FIELD_sub(v, a.x), r);
    s1 = FIELD_mul(s1, j); s1 = FIELD_double(s1); // S1 = S1 * J * 2
    a.y = FIELD_sub(a.y, s1);

    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2)*H
    a.z = FIELD_add(a.z, b.z); a.z = FIELD_sqr(a.z);
    a.z = FIELD_sub(FIELD_sub(a.z, z1z1), z2z2);
    a.z = FIELD_mul(a.z, h);

    return a;
  }
}

// ============================================================================
// Extended Jacobian (XYZZ) coordinates
// ============================================================================
// Representation: (X, Y, ZZ, ZZZ) where ZZ = Z^2, ZZZ = Z^3
// Affine coordinates: (X/ZZ, Y/ZZZ)
// Identity: ZZ = 0 (and ZZZ = 0)
//
// Advantages over standard Jacobian for chained mixed additions:
// - madd cost: 7M + 2S (vs 7M + 4S for Jacobian madd-2007-bl)
// - Saves 2 squarings per addition (= ~1.6M savings per addition)

typedef struct {
  FIELD x;
  FIELD y;
  FIELD zz;
  FIELD zzz;
} POINT_xyzz;

#define POINT_XYZZ_ZERO ((POINT_xyzz){FIELD_ZERO, FIELD_ONE, FIELD_ZERO, FIELD_ZERO})

// XYZZ mixed addition: XYZZ + Affine -> XYZZ
// http://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html#addition-madd-2008-s
// Cost: 7M + 2S
DEVICE POINT_xyzz POINT_xyzz_add_mixed(POINT_xyzz a, POINT_affine b) {
  const FIELD local_zero = FIELD_ZERO;

  if(FIELD_eq(a.zz, local_zero)) {
    const FIELD local_one = FIELD_ONE;
    a.x = b.x;
    a.y = b.y;
    a.zz = local_one;
    a.zzz = local_one;
    return a;
  }

  const FIELD p = FIELD_sub(FIELD_mul(b.y, a.zzz), a.y);
  const FIELD r = FIELD_sub(FIELD_mul(b.x, a.zz), a.x);

  // Doubling case (extremely rare in MSM: P(equal points) ≈ 2^-254)
  if(FIELD_eq(r, local_zero) && FIELD_eq(p, local_zero)) {
    // XYZZ doubling for a=0 (BN254: y^2 = x^3 + 3)
    FIELD t0 = FIELD_sqr(a.y);
    FIELD m = FIELD_sqr(a.x);
    m = FIELD_add(FIELD_double(m), m); // 3*X^2
    FIELD s = FIELD_mul(a.x, t0);
    s = FIELD_double(FIELD_double(s)); // 4*X*Y^2
    POINT_xyzz ret;
    ret.x = FIELD_sub(FIELD_sqr(m), FIELD_double(s));
    FIELD u = FIELD_sqr(t0);
    u = FIELD_double(FIELD_double(FIELD_double(u))); // 8*Y^4
    ret.y = FIELD_sub(FIELD_mul(m, FIELD_sub(s, ret.x)), u);
    ret.zz = FIELD_mul(FIELD_double(FIELD_double(t0)), a.zz); // 4*Y^2 * ZZ
    FIELD y_cubed = FIELD_mul(t0, a.y);
    ret.zzz = FIELD_mul(FIELD_double(FIELD_double(FIELD_double(y_cubed))), a.zzz); // 8*Y^3 * ZZZ
    return ret;
  }

  // EFD madd-2008-s variable mapping:
  //   U = p = Y2*ZZZ1 - Y1  (y-difference)
  //   S = r = X2*ZZ1  - X1  (x-difference)
  //   P = S² = rr,  R = S*P = S³ = rrr,  Q = X1*P = X1*rr
  const FIELD uu = FIELD_sqr(p);       // U² (needed for X3)
  const FIELD ss = FIELD_sqr(r);       // S² = P
  const FIELD sss = FIELD_mul(r, ss);  // S³ = R = S*P
  const FIELD q = FIELD_mul(a.x, ss);  // Q = X1*P

  POINT_xyzz ret;
  ret.x = FIELD_sub(FIELD_sub(uu, sss), FIELD_double(q)); // X3 = U² - R - 2Q
  ret.y = FIELD_sub(FIELD_mul(p, FIELD_sub(q, ret.x)), FIELD_mul(a.y, sss)); // Y3 = U*(Q-X3) - Y1*R
  ret.zz = FIELD_mul(a.zz, ss);   // ZZ3 = ZZ1*P
  ret.zzz = FIELD_mul(a.zzz, sss); // ZZZ3 = ZZZ1*R
  return ret;
}

// XYZZ + XYZZ addition
// Cost: 11M + 2S
DEVICE POINT_xyzz POINT_xyzz_add(POINT_xyzz a, POINT_xyzz b) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.zz, local_zero)) return b;
  if(FIELD_eq(b.zz, local_zero)) return a;

  const FIELD u1 = FIELD_mul(a.x, b.zz);
  const FIELD u2 = FIELD_mul(b.x, a.zz);
  const FIELD s1 = FIELD_mul(a.y, b.zzz);
  const FIELD s2 = FIELD_mul(b.y, a.zzz);

  if(FIELD_eq(u1, u2) && FIELD_eq(s1, s2)) {
    // Doubling
    FIELD t0 = FIELD_sqr(a.y);
    FIELD m = FIELD_sqr(a.x);
    m = FIELD_add(FIELD_double(m), m);
    FIELD s = FIELD_mul(a.x, t0);
    s = FIELD_double(FIELD_double(s));
    POINT_xyzz ret;
    ret.x = FIELD_sub(FIELD_sqr(m), FIELD_double(s));
    FIELD u = FIELD_sqr(t0);
    u = FIELD_double(FIELD_double(FIELD_double(u)));
    ret.y = FIELD_sub(FIELD_mul(m, FIELD_sub(s, ret.x)), u);
    ret.zz = FIELD_mul(FIELD_double(FIELD_double(t0)), a.zz);
    FIELD y_cubed = FIELD_mul(t0, a.y);
    ret.zzz = FIELD_mul(FIELD_double(FIELD_double(FIELD_double(y_cubed))), a.zzz);
    return ret;
  }

  const FIELD p = FIELD_sub(s2, s1);
  const FIELD r = FIELD_sub(u2, u1);
  const FIELD pp = FIELD_sqr(p);
  const FIELD rr = FIELD_sqr(r);
  const FIELD ppp = FIELD_mul(p, pp);
  const FIELD q = FIELD_mul(u1, rr);

  POINT_xyzz ret;
  ret.x = FIELD_sub(FIELD_sub(pp, ppp), FIELD_double(q));
  ret.y = FIELD_sub(FIELD_mul(p, FIELD_sub(q, ret.x)), FIELD_mul(s1, ppp));
  ret.zz = FIELD_mul(FIELD_mul(a.zz, b.zz), rr);
  ret.zzz = FIELD_mul(FIELD_mul(a.zzz, b.zzz), ppp);
  return ret;
}

// XYZZ doubling for a=0 curves (BN254)
DEVICE POINT_xyzz POINT_xyzz_double(POINT_xyzz a) {
  const FIELD local_zero = FIELD_ZERO;
  if(FIELD_eq(a.zz, local_zero)) return a;

  FIELD t0 = FIELD_sqr(a.y);
  FIELD m = FIELD_sqr(a.x);
  m = FIELD_add(FIELD_double(m), m); // 3*X^2
  FIELD s = FIELD_mul(a.x, t0);
  s = FIELD_double(FIELD_double(s)); // 4*X*Y^2

  POINT_xyzz ret;
  ret.x = FIELD_sub(FIELD_sqr(m), FIELD_double(s));
  FIELD u = FIELD_sqr(t0);
  u = FIELD_double(FIELD_double(FIELD_double(u))); // 8*Y^4
  ret.y = FIELD_sub(FIELD_mul(m, FIELD_sub(s, ret.x)), u);
  ret.zz = FIELD_mul(FIELD_double(FIELD_double(t0)), a.zz);
  FIELD y_cubed = FIELD_mul(t0, a.y);
  ret.zzz = FIELD_mul(FIELD_double(FIELD_double(FIELD_double(y_cubed))), a.zzz);
  return ret;
}