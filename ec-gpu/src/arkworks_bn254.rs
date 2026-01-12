use ark_bn254::{Fq, Fq2, FqConfig, Fr, FrConfig, G1Affine};
use ark_ff::{BigInteger, MontConfig};

use crate::{GpuField, GpuName};

fn bytes_le_to_u32_limbs(mut bytes: Vec<u8>) -> Vec<u32> {
    // Pad to multiple of 4 bytes
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bigint_to_u32_limbs_le<B: BigInteger>(b: B) -> Vec<u32> {
    bytes_le_to_u32_limbs(b.to_bytes_le())
}

impl GpuName for Fq {
    fn name() -> String {
        crate::name!()
    }
}

impl GpuField for Fq {
    fn one() -> Vec<u32> {
        bigint_to_u32_limbs_le(FqConfig::R)
    }

    fn r2() -> Vec<u32> {
        bigint_to_u32_limbs_le(FqConfig::R2)
    }

    fn modulus() -> Vec<u32> {
        bigint_to_u32_limbs_le(FqConfig::MODULUS)
    }
}

impl GpuName for Fq2 {
    fn name() -> String {
        crate::name!()
    }
}

impl GpuField for Fq2 {
    fn one() -> Vec<u32> {
        let n = bigint_to_u32_limbs_le(FqConfig::MODULUS).len();
        let mut out = vec![0u32; 2 * n];
        out[..n].copy_from_slice(&bigint_to_u32_limbs_le(FqConfig::R));

        out
    }

    fn r2() -> Vec<u32> {
        let n = bigint_to_u32_limbs_le(FqConfig::MODULUS).len();
        let mut out = vec![0u32; 2 * n];
        out[..n].copy_from_slice(&bigint_to_u32_limbs_le(FqConfig::R2));
        out
    }

    fn modulus() -> Vec<u32> {
        bigint_to_u32_limbs_le(FqConfig::MODULUS)
    }

    fn sub_field_name() -> Option<String> {
        Some(Fq::name())
    }
}

// Implementations for Fr (scalar field)
impl GpuName for Fr {
    fn name() -> String {
        crate::name!()
    }
}

impl GpuField for Fr {
    fn one() -> Vec<u32> {
        bigint_to_u32_limbs_le(FrConfig::R)
    }

    fn r2() -> Vec<u32> {
        bigint_to_u32_limbs_le(FrConfig::R2)
    }

    fn modulus() -> Vec<u32> {
        bigint_to_u32_limbs_le(FrConfig::MODULUS)
    }
}

impl GpuName for G1Affine {
    fn name() -> String {
        crate::name!()
    }
}
