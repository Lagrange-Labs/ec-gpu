/// Helper function to convert arkworks BigInteger to u32 limbs in little-endian order.
pub fn bigint_to_u32_limbs<B: ark_ff::BigInteger>(b: B) -> Vec<u32> {
    let bytes = b.to_bytes_le();
    bytes_to_u32_limbs(bytes)
}

/// Helper function to convert bytes to u32 limbs in little-endian order.
pub fn bytes_to_u32_limbs(mut bytes: Vec<u8>) -> Vec<u32> {
    // Pad to multiple of 4 bytes
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Implement `GpuName` and `GpuField` for an arkworks prime field.
///
/// # Example
///
/// ```ignore
/// use ec_gpu::impl_gpu_field_arkworks;
/// use ark_bls12_381::{Fr, FrConfig};
///
/// impl_gpu_field_arkworks!(Fr, FrConfig);
/// ```
#[macro_export]
macro_rules! impl_gpu_field_arkworks {
    ($field:ty, $config:ty) => {
        impl $crate::GpuName for $field {
            fn name() -> String {
                $crate::name!()
            }
        }

        impl $crate::GpuField for $field {
            fn one() -> Vec<u32> {
                use ark_ff::MontConfig;
                $crate::arkworks_macros::bigint_to_u32_limbs(<$config>::R)
            }

            fn r2() -> Vec<u32> {
                use ark_ff::MontConfig;
                $crate::arkworks_macros::bigint_to_u32_limbs(<$config>::R2)
            }

            fn modulus() -> Vec<u32> {
                use ark_ff::MontConfig;
                $crate::arkworks_macros::bigint_to_u32_limbs(<$config>::MODULUS)
            }
        }
    };
}

/// Implement `GpuName` for an arkworks curve affine point type.
///
/// # Example
///
/// ```ignore
/// use ec_gpu::impl_gpu_name_arkworks_curve;
/// use ark_bls12_381::G1Affine;
///
/// impl_gpu_name_arkworks_curve!(G1Affine);
/// ```
#[macro_export]
macro_rules! impl_gpu_name_arkworks_curve {
    ($curve:ty) => {
        impl $crate::GpuName for $curve {
            fn name() -> String {
                $crate::name!()
            }
        }
    };
}

/// Implement `GpuName` and `GpuField` for an arkworks quadratic extension field (like Fq2).
///
/// # Example
///
/// ```ignore
/// use ec_gpu::impl_gpu_field_arkworks_ext2;
/// use ark_bls12_381::{Fq, Fq2, FqConfig};
///
/// impl_gpu_field_arkworks_ext2!(Fq2, FqConfig, Fq);
/// ```
#[macro_export]
macro_rules! impl_gpu_field_arkworks_ext2 {
    ($field2:ty, $base_config:ty, $base_field:ty) => {
        impl $crate::GpuName for $field2 {
            fn name() -> String {
                $crate::name!()
            }
        }

        impl $crate::GpuField for $field2 {
            fn one() -> Vec<u32> {
                use ark_ff::MontConfig;
                let n = $crate::arkworks_macros::bigint_to_u32_limbs(<$base_config>::MODULUS).len();
                let mut out = vec![0u32; 2 * n];
                out[..n].copy_from_slice(&$crate::arkworks_macros::bigint_to_u32_limbs(
                    <$base_config>::R,
                ));
                out
            }

            fn r2() -> Vec<u32> {
                use ark_ff::MontConfig;
                let n = $crate::arkworks_macros::bigint_to_u32_limbs(<$base_config>::MODULUS).len();
                let mut out = vec![0u32; 2 * n];
                out[..n].copy_from_slice(&$crate::arkworks_macros::bigint_to_u32_limbs(
                    <$base_config>::R2,
                ));
                out
            }

            fn modulus() -> Vec<u32> {
                use ark_ff::MontConfig;
                $crate::arkworks_macros::bigint_to_u32_limbs(<$base_config>::MODULUS)
            }

            fn sub_field_name() -> Option<String> {
                Some(<$base_field as $crate::GpuName>::name())
            }
        }
    };
}
