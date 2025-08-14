use crate::distances::dot_product_simd;

use half::f16;
use itertools::izip;

use std::iter::zip;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vsubq_f32};

#[cfg(target_arch = "aarch64")]
use crate::utils::conv_f16_to_f32;

use crate::Float;

#[inline]
fn dense_euclidean_distance_general<T>(query: &[T], values: &[T]) -> f32
where
    T: Float,
{
    query.iter().zip(values).fold(0.0, |acc, (a, b)| {
        let diff = a.to_f32().unwrap() - b.to_f32().unwrap();
        acc + diff * diff
    })
}

#[inline]
pub fn dense_euclidean_distance_unrolled<T>(query: &[T], values: &[T]) -> f32
where
    T: Float,
{
    const N_LANES: usize = 16;
    let mut r = [0.0; N_LANES];

    let chunks = query.len() / N_LANES;
    for (q_chunk, v_chunk) in zip(query.chunks_exact(N_LANES), values.chunks_exact(N_LANES)) {
        for i in 0..N_LANES {
            let d = q_chunk[i].to_f32().unwrap() - v_chunk[i].to_f32().unwrap();
            r[i] += d * d;
        }
    }

    r.iter().fold(0.0, |acc, &val| acc + val)
        + dense_euclidean_distance_general(&query[N_LANES * chunks..], &&values[N_LANES * chunks..])
}

#[inline]
pub fn dense_euclidean_distance_batch_4_unrolled<T>(query: &[T], vectors: [&[T]; 4]) -> [f32; 4]
where
    T: Float,
{
    const N_LANES: usize = 16;
    let len = query.len();
    let chunks = len / N_LANES;
    // Partial sums per lane for each of the 4 vectors
    let mut r0 = [0.0f32; N_LANES];
    let mut r1 = [0.0f32; N_LANES];
    let mut r2 = [0.0f32; N_LANES];
    let mut r3 = [0.0f32; N_LANES];

    // Process full chunks of size N_LANES
    for (q_chunk, v0_chunk, v1_chunk, v2_chunk, v3_chunk) in izip!(
        query.chunks_exact(N_LANES),
        vectors[0].chunks_exact(N_LANES),
        vectors[1].chunks_exact(N_LANES),
        vectors[2].chunks_exact(N_LANES),
        vectors[3].chunks_exact(N_LANES),
    ) {
        for i in 0..N_LANES {
            let qf = q_chunk[i].to_f32().unwrap();
            let v0f = v0_chunk[i].to_f32().unwrap();
            let d0 = qf - v0f;
            r0[i] += d0 * d0;

            let v1f = v1_chunk[i].to_f32().unwrap();
            let d1 = qf - v1f;
            r1[i] += d1 * d1;

            let v2f = v2_chunk[i].to_f32().unwrap();
            let d2 = qf - v2f;
            r2[i] += d2 * d2;

            let v3f = v3_chunk[i].to_f32().unwrap();
            let d3 = qf - v3f;
            r3[i] += d3 * d3;
        }
    }

    // Sum partials from r0..r3
    let mut sum0: f32 = r0.iter().sum();
    let mut sum1: f32 = r1.iter().sum();
    let mut sum2: f32 = r2.iter().sum();
    let mut sum3: f32 = r3.iter().sum();

    // Remainder elements
    let rem_start = chunks * N_LANES;
    for i in rem_start..len {
        let qf = query[i].to_f32().unwrap();

        let v0f = vectors[0][i].to_f32().unwrap();
        let d0 = qf - v0f;
        sum0 += d0 * d0;

        let v1f = vectors[1][i].to_f32().unwrap();
        let d1 = qf - v1f;
        sum1 += d1 * d1;

        let v2f = vectors[2][i].to_f32().unwrap();
        let d2 = qf - v2f;
        sum2 += d2 * d2;

        let v3f = vectors[3][i].to_f32().unwrap();
        let d3 = qf - v3f;
        sum3 += d3 * d3;
    }

    [sum0, sum1, sum2, sum3]
}

#[inline]
pub fn vectors_norm(vectors: &[f32], d: usize) -> Vec<f32> {
    vectors
        .chunks_exact(d)
        .map(|v| dot_product_simd(v, v))
        .collect()
}

/* simd */

#[inline]
pub fn euclidean_distance_batch_4_simd<T>(query: &[T], values: [&[T]; 4]) -> [f32; 4]
where
    T: Float + EuclideanDistance<T>,
{
    unsafe { T::euclidean_distance_batch_4_simd(query, values) }
}

#[inline]
pub fn euclidean_distance_simd<T>(query: &[T], values: &[T]) -> f32
where
    T: Float + EuclideanDistance<T>,
{
    unsafe { T::euclidean_distance_simd(query, values) }
}

pub trait EuclideanDistance<U> {
    unsafe fn euclidean_distance_simd(query: &[U], values: &[U]) -> f32;
    unsafe fn euclidean_distance_batch_4_simd(query: &[U], vectors: [&[U]; 4]) -> [f32; 4];
}

impl EuclideanDistance<f32> for f32 {
    unsafe fn euclidean_distance_simd(query: &[f32], values: &[f32]) -> f32 {
        // x86_64 + AVX2 runtime-check
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2_inner(query: &[f32], values: &[f32]) -> f32 {
                    use core::arch::x86_64::*;
                    const N_LANES: usize = 8;
                    let len = query.len();
                    let chunks = len / N_LANES;
                    let mut sum = _mm256_setzero_ps();
                    for i in 0..chunks {
                        let base = i * N_LANES;
                        let qv = _mm256_loadu_ps(query.as_ptr().add(base));
                        let vv = _mm256_loadu_ps(values.as_ptr().add(base));
                        // diff = qv - vv
                        let diff = _mm256_sub_ps(qv, vv);
                        let sq = _mm256_mul_ps(diff, diff);
                        sum = _mm256_add_ps(sum, sq);
                    }
                    // horizontal sum
                    let mut tmp = [0.0f32; N_LANES];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
                    let mut acc: f32 = tmp.iter().sum();
                    // remainder
                    for i in (chunks * N_LANES)..len {
                        let qf = query[i];
                        let vf = values[i];
                        let d = qf - vf;
                        acc += d * d;
                    }
                    acc
                }
                return avx2_inner(query, values);
            }
            // Fallback to scalar if AVX2 is not available
            return dense_euclidean_distance_unrolled(query, values);
        }
        // aarch64 NEON path
        #[cfg(target_arch = "aarch64")]
        {
            #[target_feature(enable = "neon")]
            unsafe fn neon_inner(query: &[f32], values: &[f32]) -> f32 {
                use core::arch::aarch64::*;
                const N_LANES: usize = 4;
                let len = query.len();
                let chunks = len / N_LANES;
                let mut sum_v = vdupq_n_f32(0.0);
                for i in 0..chunks {
                    let base = i * N_LANES;
                    let qv = vld1q_f32(query.as_ptr().add(base));
                    let vv = vld1q_f32(values.as_ptr().add(base));
                    let diff = vsubq_f32(qv, vv);
                    sum_v = vaddq_f32(sum_v, vmulq_f32(diff, diff));
                }
                let mut acc = vaddvq_f32(sum_v);
                // remainder
                for i in (chunks * N_LANES)..len {
                    let d = query[i] - values[i];
                    acc += d * d;
                }
                acc
            }
            return neon_inner(query, values);
        }
        // scalar fallback
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            target_arch = "aarch64"
        )))]
        dense_euclidean_distance_unrolled(query, values)
    }

    unsafe fn euclidean_distance_batch_4_simd(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
        // x86_64 + AVX2
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                #[target_feature(enable = "avx2", enable = "fma")]
                unsafe fn avx2_inner(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
                    use core::arch::x86_64::*;
                    const N_LANES: usize = 8;
                    let len = query.len();
                    let chunks = len / N_LANES;
                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut sum2 = _mm256_setzero_ps();
                    let mut sum3 = _mm256_setzero_ps();
                    use itertools::izip;
                    for (q_chunk, v0, v1, v2, v3) in izip!(
                        query.chunks_exact(N_LANES),
                        vectors[0].chunks_exact(N_LANES),
                        vectors[1].chunks_exact(N_LANES),
                        vectors[2].chunks_exact(N_LANES),
                        vectors[3].chunks_exact(N_LANES),
                    ) {
                        // load
                        let qv = _mm256_loadu_ps(q_chunk.as_ptr());
                        let v0v = _mm256_loadu_ps(v0.as_ptr());
                        let v1v = _mm256_loadu_ps(v1.as_ptr());
                        let v2v = _mm256_loadu_ps(v2.as_ptr());
                        let v3v = _mm256_loadu_ps(v3.as_ptr());
                        // diff and square-add
                        let d0 = _mm256_sub_ps(qv, v0v);
                        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(d0, d0));
                        let d1 = _mm256_sub_ps(qv, v1v);
                        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(d1, d1));
                        let d2 = _mm256_sub_ps(qv, v2v);
                        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(d2, d2));
                        let d3 = _mm256_sub_ps(qv, v3v);
                        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(d3, d3));
                    }
                    // horizontal sums
                    unsafe fn hsum(v: __m256) -> f32 {
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
                        tmp.iter().sum()
                    }
                    let mut out = [hsum(sum0), hsum(sum1), hsum(sum2), hsum(sum3)];
                    // remainder
                    let rem_start = chunks * N_LANES;
                    for i in rem_start..len {
                        let qf = query[i];
                        let v0f = vectors[0][i];
                        let d0 = qf - v0f;
                        out[0] += d0 * d0;
                        let v1f = vectors[1][i];
                        let d1 = qf - v1f;
                        out[1] += d1 * d1;
                        let v2f = vectors[2][i];
                        let d2 = qf - v2f;
                        out[2] += d2 * d2;
                        let v3f = vectors[3][i];
                        let d3 = qf - v3f;
                        out[3] += d3 * d3;
                    }
                    out
                }
                return avx2_inner(query, vectors);
            }
            // Fallback to scalar if AVX2 is not available
            return dense_euclidean_distance_batch_4_unrolled(query, vectors);
        }
        // aarch64 NEON
        #[cfg(target_arch = "aarch64")]
        {
            #[target_feature(enable = "neon")]
            unsafe fn neon_inner(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
                use core::arch::aarch64::*;
                const N_LANES: usize = 4;
                let len = query.len();
                let chunks = len / N_LANES;
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut sum2 = vdupq_n_f32(0.0);
                let mut sum3 = vdupq_n_f32(0.0);
                for i in 0..chunks {
                    let base = i * N_LANES;
                    let qv = vld1q_f32(query.as_ptr().add(base));
                    let v0v = vld1q_f32(vectors[0].as_ptr().add(base));
                    let v1v = vld1q_f32(vectors[1].as_ptr().add(base));
                    let v2v = vld1q_f32(vectors[2].as_ptr().add(base));
                    let v3v = vld1q_f32(vectors[3].as_ptr().add(base));
                    let d0 = vsubq_f32(qv, v0v);
                    sum0 = vaddq_f32(sum0, vmulq_f32(d0, d0));
                    let d1 = vsubq_f32(qv, v1v);
                    sum1 = vaddq_f32(sum1, vmulq_f32(d1, d1));
                    let d2 = vsubq_f32(qv, v2v);
                    sum2 = vaddq_f32(sum2, vmulq_f32(d2, d2));
                    let d3 = vsubq_f32(qv, v3v);
                    sum3 = vaddq_f32(sum3, vmulq_f32(d3, d3));
                }
                // horizontal sums
                let mut out = [0.0f32; 4];
                out[0] = vaddvq_f32(sum0);
                out[1] = vaddvq_f32(sum1);
                out[2] = vaddvq_f32(sum2);
                out[3] = vaddvq_f32(sum3);
                // remainder
                let rem_start = chunks * N_LANES;
                for i in rem_start..len {
                    let qf = query[i];
                    let v0f = vectors[0][i];
                    let d0 = qf - v0f;
                    out[0] += d0 * d0;
                    let v1f = vectors[1][i];
                    let d1 = qf - v1f;
                    out[1] += d1 * d1;
                    let v2f = vectors[2][i];
                    let d2 = qf - v2f;
                    out[2] += d2 * d2;
                    let v3f = vectors[3][i];
                    let d3 = qf - v3f;
                    out[3] += d3 * d3;
                }
                out
            }
            return neon_inner(query, vectors);
        }
        // scalar fallback
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            target_arch = "aarch64"
        )))]
        dense_euclidean_distance_batch_4_unrolled(query, vectors)
    }
}

impl EuclideanDistance<f16> for f16 {
    unsafe fn euclidean_distance_simd(query: &[f16], values: &[f16]) -> f32 {
        // x86_64 AVX2+F16C
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                #[target_feature(enable = "avx2,f16c")]
                unsafe fn euclidean_distance_avx2(query: &[f16], values: &[f16]) -> f32 {
                    use core::arch::x86_64::*;
                    const N_LANES: usize = 8;
                    let len = query.len();
                    let chunks = len / N_LANES;
                    let mut sum = _mm256_setzero_ps();
                    for i in 0..chunks {
                        let base = i * N_LANES;
                        let qh = _mm_loadu_si128(query.as_ptr().add(base) as *const __m128i);
                        let vh = _mm_loadu_si128(values.as_ptr().add(base) as *const __m128i);
                        let qv = _mm256_cvtph_ps(qh);
                        let vv = _mm256_cvtph_ps(vh);
                        let diff = _mm256_sub_ps(qv, vv);
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
                    }
                    let mut tmp = [0.0f32; N_LANES];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
                    let mut acc: f32 = tmp.iter().sum();
                    for i in (chunks * N_LANES)..len {
                        let qf = query[i].to_f32();
                        let vf = values[i].to_f32();
                        let d = qf - vf;
                        acc += d * d;
                    }
                    acc
                }
                return euclidean_distance_avx2(query, values);
            }
            // Fallback to scalar if AVX2+F16C is not available
            return dense_euclidean_distance_unrolled(query, values);
        }
        // aarch64 NEON: use convert+f32 NEON
        #[cfg(target_arch = "aarch64")]
        {
            // via f32 NEON
            pub unsafe fn euclidean_distance_neon(query: &[f16], values: &[f16]) -> f32 {
                assert_eq!(query.len(), values.len());
                let len = query.len();
                let chunks = len / 4;

                // 1) Allocate f32 buffers
                let mut qf: Vec<f32> = Vec::with_capacity(len);
                let mut vf: Vec<f32> = Vec::with_capacity(len);

                unsafe {
                    qf.set_len(len);
                    vf.set_len(len);
                }

                // 2) Convert inside the function
                conv_f16_to_f32(query, &mut qf);
                conv_f16_to_f32(values, &mut vf);

                // 3) NEON‐accelerated distance on the converted data
                let mut sum_v = vdupq_n_f32(0.0);
                for ci in 0..chunks {
                    let base = ci * 4;
                    let qv = vld1q_f32(qf.as_ptr().add(base));
                    let vv = vld1q_f32(vf.as_ptr().add(base));
                    let d = vsubq_f32(qv, vv);
                    sum_v = vaddq_f32(sum_v, vmulq_f32(d, d));
                }
                // horizontal sum of SIMD lanes
                let mut acc = vaddvq_f32(sum_v);

                // tail loop
                for i in (chunks * 4)..len {
                    let d = qf[i] - vf[i];
                    acc += d * d;
                }

                acc
            }
            return euclidean_distance_neon(query, values);
        }
        // fallback scalar
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            target_arch = "aarch64"
        )))]
        dense_euclidean_distance_unrolled(query, values)
    }

    unsafe fn euclidean_distance_batch_4_simd(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
        // x86_64 AVX2+F16C
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                #[target_feature(enable = "avx2,f16c,fma")]
                unsafe fn euclidean_distance_batch_4_avx2(
                    query: &[f16],
                    vectors: [&[f16]; 4],
                ) -> [f32; 4] {
                    use core::arch::x86_64::*;
                    const N_LANES: usize = 8;
                    let len = query.len();
                    let chunks = len / N_LANES;

                    // AVX accumulators for each of the 4 distances
                    let mut sum0 = _mm256_setzero_ps();
                    let mut sum1 = _mm256_setzero_ps();
                    let mut sum2 = _mm256_setzero_ps();
                    let mut sum3 = _mm256_setzero_ps();

                    // Process full chunks of 8 half-values
                    use itertools::izip;
                    for (q_chunk, v0_chunk, v1_chunk, v2_chunk, v3_chunk) in izip!(
                        query.chunks_exact(N_LANES),
                        vectors[0].chunks_exact(N_LANES),
                        vectors[1].chunks_exact(N_LANES),
                        vectors[2].chunks_exact(N_LANES),
                        vectors[3].chunks_exact(N_LANES),
                    ) {
                        // Load 8 halfs and convert to 8 f32 lanes
                        let qh = _mm_loadu_si128(q_chunk.as_ptr() as *const __m128i);
                        let qv = _mm256_cvtph_ps(qh);

                        // Vector 0
                        let v0h = _mm_loadu_si128(v0_chunk.as_ptr() as *const __m128i);
                        let v0v = _mm256_cvtph_ps(v0h);
                        let d0 = _mm256_sub_ps(qv, v0v);
                        sum0 = _mm256_fmadd_ps(d0, d0, sum0);

                        // Vector 1
                        let v1h = _mm_loadu_si128(v1_chunk.as_ptr() as *const __m128i);
                        let v1v = _mm256_cvtph_ps(v1h);
                        let d1 = _mm256_sub_ps(qv, v1v);
                        sum1 = _mm256_fmadd_ps(d1, d1, sum1);

                        // Vector 2
                        let v2h = _mm_loadu_si128(v2_chunk.as_ptr() as *const __m128i);
                        let v2v = _mm256_cvtph_ps(v2h);
                        let d2 = _mm256_sub_ps(qv, v2v);
                        sum2 = _mm256_fmadd_ps(d2, d2, sum2);

                        // Vector 3
                        let v3h = _mm_loadu_si128(v3_chunk.as_ptr() as *const __m128i);
                        let v3v = _mm256_cvtph_ps(v3h);
                        let d3 = _mm256_sub_ps(qv, v3v);
                        sum3 = _mm256_fmadd_ps(d3, d3, sum3);
                    }

                    // Horizontal sums of the accumulators
                    fn hsum256(v: __m256) -> f32 {
                        let mut tmp = [0.0f32; N_LANES];
                        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), v) };
                        tmp.iter().sum()
                    }
                    let mut out0 = hsum256(sum0);
                    let mut out1 = hsum256(sum1);
                    let mut out2 = hsum256(sum2);
                    let mut out3 = hsum256(sum3);

                    // Handle remainder
                    let rem_start = chunks * N_LANES;
                    for i in rem_start..len {
                        let qf = query[i].to_f32();
                        // Vector 0
                        let v0f = vectors[0][i].to_f32();
                        let d0 = qf - v0f;
                        out0 += d0 * d0;
                        // Vector 1
                        let v1f = vectors[1][i].to_f32();
                        let d1 = qf - v1f;
                        out1 += d1 * d1;
                        // Vector 2
                        let v2f = vectors[2][i].to_f32();
                        let d2 = qf - v2f;
                        out2 += d2 * d2;
                        // Vector 3
                        let v3f = vectors[3][i].to_f32();
                        let d3 = qf - v3f;
                        out3 += d3 * d3;
                    }

                    [out0, out1, out2, out3]
                }
                return euclidean_distance_batch_4_avx2(query, vectors);
            }
            // Fallback to scalar if AVX2+F16C is not available
            return dense_euclidean_distance_batch_4_unrolled(query, vectors);
        }
        // aarch64 via f32 NEON
        #[cfg(target_arch = "aarch64")]
        {
            #[target_feature(enable = "neon")]
            unsafe fn euclidean_distance_batch_4_neon(
                query: &[f16],
                vectors: [&[f16]; 4],
            ) -> [f32; 4] {
                use core::arch::aarch64::*;

                let len = query.len();
                // prepare full-length f32 buffers
                let mut qf: Vec<f32> = Vec::with_capacity(len);
                let mut v0f: Vec<f32> = Vec::with_capacity(len);
                let mut v1f: Vec<f32> = Vec::with_capacity(len);
                let mut v2f: Vec<f32> = Vec::with_capacity(len);
                let mut v3f: Vec<f32> = Vec::with_capacity(len);
                // safety: we set_len immediately after reserve
                unsafe {
                    qf.set_len(len);
                    v0f.set_len(len);
                    v1f.set_len(len);
                    v2f.set_len(len);
                    v3f.set_len(len);
                }

                // 1) do all the f16→f32 conversions up-front in big loops
                conv_f16_to_f32(query, &mut qf);
                conv_f16_to_f32(vectors[0], &mut v0f);
                conv_f16_to_f32(vectors[1], &mut v1f);
                conv_f16_to_f32(vectors[2], &mut v2f);
                conv_f16_to_f32(vectors[3], &mut v3f);

                // 2) now do your NEON-accelerated distance in 4×4 chunks
                const CHUNK: usize = 4;
                let chunks = len / CHUNK;

                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = sum0;
                let mut sum2 = sum0;
                let mut sum3 = sum0;

                for ci in 0..chunks {
                    let base = ci * CHUNK;
                    let qv = vld1q_f32(qf.as_ptr().add(base));
                    let v0v = vld1q_f32(v0f.as_ptr().add(base));
                    let v1v = vld1q_f32(v1f.as_ptr().add(base));
                    let v2v = vld1q_f32(v2f.as_ptr().add(base));
                    let v3v = vld1q_f32(v3f.as_ptr().add(base));

                    let d0 = vsubq_f32(qv, v0v);
                    sum0 = vmlaq_f32(sum0, d0, d0);
                    let d1 = vsubq_f32(qv, v1v);
                    sum1 = vmlaq_f32(sum1, d1, d1);
                    let d2 = vsubq_f32(qv, v2v);
                    sum2 = vmlaq_f32(sum2, d2, d2);
                    let d3 = vsubq_f32(qv, v3v);
                    sum3 = vmlaq_f32(sum3, d3, d3);
                }

                // horizontal sums
                let mut out0 = vaddvq_f32(sum0);
                let mut out1 = vaddvq_f32(sum1);
                let mut out2 = vaddvq_f32(sum2);
                let mut out3 = vaddvq_f32(sum3);

                // tail
                for i in (chunks * CHUNK)..len {
                    let qfv = qf[i];
                    let d0 = qfv - v0f[i];
                    out0 += d0 * d0;
                    let d1 = qfv - v1f[i];
                    out1 += d1 * d1;
                    let d2 = qfv - v2f[i];
                    out2 += d2 * d2;
                    let d3 = qfv - v3f[i];
                    out3 += d3 * d3;
                }

                [out0, out1, out2, out3]
            }
            return euclidean_distance_batch_4_neon(query, vectors);
        }
        // fallback scalar
        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "avx2"),
            target_arch = "aarch64"
        )))]
        dense_euclidean_distance_batch_4_unrolled(query, vectors)
    }
}
