use crate::simd_utils::horizontal_sum_256;
use crate::{AsRefItem, DenseVector1D, Float, SparseVector1D, Vector1D};
use half::f16;
use itertools::izip;
use std::iter::zip;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

 #[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;


#[inline]
pub fn dense_dot_product<T>(query: &[T], document: &[T]) -> f32
where
    T: Float,
{
    dense_dot_product_unrolled(query, document)
}

#[inline]
pub fn dense_dot_product_unrolled<T>(query: &[T], document: &[T]) -> f32
where
    T: Float,
{
    const N_LANES: usize = 8;
    let mut r = [0.0; N_LANES];

    let chunks = query.len() / N_LANES;
    for (q_chunk, v_chunk) in zip(query.chunks_exact(N_LANES), document.chunks_exact(N_LANES)) {
        for i in 0..N_LANES {
            let d = q_chunk[i].to_f32().unwrap() * v_chunk[i].to_f32().unwrap();
            r[i] += d;
        }
    }

    r.iter().sum::<f32>()
        + dense_dot_product_general(&query[N_LANES * chunks..], &document[N_LANES * chunks..])
}

#[inline]
pub fn dense_dot_product_general<T>(query: &[T], document: &[T]) -> f32
where
    T: Float,
{
    query.iter().zip(document).fold(0.0, |acc, (a, b)| {
        acc + (a.to_f32().unwrap() * b.to_f32().unwrap())
    })
}

#[inline]
pub fn dense_dot_product_batch_4_unrolled<T>(query: &[T], vectors: [&[T]; 4]) -> [f32; 4]
where
    T: Float,
{
    const N_LANES: usize = 8;
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
            r0[i] += qf * v0_chunk[i].to_f32().unwrap();
            r1[i] += qf * v1_chunk[i].to_f32().unwrap();
            r2[i] += qf * v2_chunk[i].to_f32().unwrap();
            r3[i] += qf * v3_chunk[i].to_f32().unwrap();
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
        sum0 += qf * vectors[0][i].to_f32().unwrap();
        sum1 += qf * vectors[1][i].to_f32().unwrap();
        sum2 += qf * vectors[2][i].to_f32().unwrap();
        sum3 += qf * vectors[3][i].to_f32().unwrap();
    }

    [sum0, sum1, sum2, sum3]
}

#[inline]
pub fn dense_dot_product_batch_4_general<T>(query: &[T], vectors: [&[T]; 4]) -> [f32; 4]
where
    T: Float,
{
    let mut sums = [0.0f32; 4];
    for (i, q) in query.iter().enumerate() {
        let qf = q.to_f32().unwrap();
        sums[0] += qf * vectors[0][i].to_f32().unwrap();
        sums[1] += qf * vectors[1][i].to_f32().unwrap();
        sums[2] += qf * vectors[2][i].to_f32().unwrap();
        sums[3] += qf * vectors[3][i].to_f32().unwrap();
    }
    sums
}

// Sparse
#[inline]
pub fn dot_product_dense_sparse<T1, T, U, F>(
    query: &DenseVector1D<T1>,
    array: &SparseVector1D<U, T>,
) -> f32
where
    T1: AsRefItem<Item = F>,
    T: AsRefItem<Item = F>,
    U: AsRefItem<Item = u16>,
    F: Float,
{
    const N_LANES: usize = 4;

    let mut result = [0.0; N_LANES];
    let query_slice = query.values_as_slice();

    let chunk_iter = array
        .components_as_slice()
        .iter()
        .zip(array.values_as_slice())
        .array_chunks::<N_LANES>();

    for chunk in chunk_iter {
        //for i in 0..N_LANES { // Slightly faster withour this for.
        result[0] += unsafe {
            (*query_slice.get_unchecked(*chunk[0].0 as usize))
                .to_f32()
                .unwrap()
                * (*chunk[0].1).to_f32().unwrap()
        };
        result[1] += unsafe {
            (*query_slice.get_unchecked(*chunk[1].0 as usize))
                .to_f32()
                .unwrap()
                * (*chunk[1].1).to_f32().unwrap()
        };
        result[2] += unsafe {
            (*query_slice.get_unchecked(*chunk[2].0 as usize))
                .to_f32()
                .unwrap()
                * (*chunk[2].1).to_f32().unwrap()
        };
        result[3] += unsafe {
            (*query_slice.get_unchecked(*chunk[3].0 as usize))
                .to_f32()
                .unwrap()
                * (*chunk[3].1).to_f32().unwrap()
        };
    }

    let l = array.components_as_slice().len();
    let rem = l % N_LANES;

    for (&i, &v) in array.components_as_slice()[l - rem..]
        .iter()
        .zip(&array.values_as_slice()[l - rem..])
    {
        result[0] += unsafe {
            (*query_slice.get_unchecked(i as usize)).to_f32().unwrap() * v.to_f32().unwrap()
        };
    }

    result.iter().sum()
}

#[inline]
#[must_use]
pub fn sparse_dot_product_with_merge<F, U, T>(
    query: &SparseVector1D<U, T>,
    vector: &SparseVector1D<U, T>,
) -> f32
where
    U: AsRefItem<Item = u16>,
    T: AsRefItem<Item = F>,
    F: Float,
{
    let mut result = 0.0;
    let mut i = 0;
    for (&q_id, &q_v) in query
        .components_as_slice()
        .iter()
        .zip(query.values_as_slice())
    {
        unsafe {
            while i < vector.components_as_slice().len()
                && *vector.components_as_slice().get_unchecked(i) < q_id
            {
                i += 1;
            }

            if i == vector.components_as_slice().len() {
                break;
            }

            if *vector.components_as_slice().get_unchecked(i) == q_id {
                result += (*vector.values_as_slice().get_unchecked(i))
                    .to_f32()
                    .unwrap()
                    * q_v.to_f32().unwrap();
            }
        }
    }
    result
}

/* simd */
#[inline]
pub fn dot_product_batch_4<T>(query: &[T], values: [&[T]; 4]) -> [f32; 4]
where
    T: Float + DotProduct<T>,
{
    unsafe { T::dot_product_batch_4(query, values) }
}

#[inline]
pub fn dot_product_unrolled<T>(query: &[T], values: &[T]) -> f32
where
    T: Float + DotProduct<T>,
{
    unsafe { T::dot_product_unrolled(query, values) }
}

pub trait DotProduct<U> {
    unsafe fn dot_product_unrolled(query: &[U], values: &[U]) -> f32;
    unsafe fn dot_product_batch_4(query: &[U], values: [&[U]; 4]) -> [f32; 4];
}

impl DotProduct<f32> for f32 {
    unsafe fn dot_product_unrolled(query: &[f32], values: &[f32]) -> f32 {
        // x86_64 + runtime check for AVX2
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                #[target_feature(enable = "avx2")]
                unsafe fn dot_product_unrolled_avx(query: &[f32], values: &[f32]) -> f32 {
                    const N_LANES: usize = 8;
                    let mut sum = _mm256_setzero_ps();
                    let chunks = query.len() / N_LANES;
                    for (q_chunk, v_chunk) in query.chunks_exact(N_LANES).zip(values.chunks_exact(N_LANES)) {
                        let qv = _mm256_loadu_ps(q_chunk.as_ptr());
                        let vv = _mm256_loadu_ps(v_chunk.as_ptr());
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(qv, vv));
                    }
                    let mut tmp = [0.0f32; N_LANES];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
                    let simd_sum: f32 = tmp.iter().sum();
                    let rem = query[chunks * N_LANES..]
                        .iter()
                        .zip(&values[chunks * N_LANES..])
                        .fold(0.0, |acc, (&a, &b)| acc + a * b);
                    simd_sum + rem
                }
                return dot_product_unrolled_avx(query, values);
            }
        }
        // aarch64 NEON path
        #[cfg(target_arch = "aarch64")]
        {
            unsafe fn dot_product_unrolled_neon(query: &[f32], values: &[f32]) -> f32 {
                const N_LANES: usize = 4;
                let mut sum_v = vdupq_n_f32(0.0);
                let chunks = query.len() / N_LANES;
                for i in 0..chunks {
                    let base = i * N_LANES;
                    let qa = vld1q_f32(query.as_ptr().add(base));
                    let va = vld1q_f32(values.as_ptr().add(base));
                    let prod = vmulq_f32(qa, va);
                    sum_v = vaddq_f32(sum_v, prod);
                }
                // horizontal sum of sum_v
                let mut acc = vaddvq_f32(sum_v);
                // remainder
                for i in (chunks * N_LANES)..query.len() {
                    acc += query[i] * values[i];
                }
                acc
            }
            return dot_product_unrolled_neon(query, values);
        }
        // Scalar fallback
        dense_dot_product_unrolled(query, values)
    }

    unsafe fn dot_product_batch_4(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                #[target_feature(enable = "avx2", enable = "fma")]
                unsafe fn dot_product_batch_4_avx2(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
                    const N_LANES: usize = 8;
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
                        vectors[3].chunks_exact(N_LANES)
                    ) {
                        let qv = _mm256_loadu_ps(q_chunk.as_ptr());
                        let v0v = _mm256_loadu_ps(v0.as_ptr());
                        let v1v = _mm256_loadu_ps(v1.as_ptr());
                        let v2v = _mm256_loadu_ps(v2.as_ptr());
                        let v3v = _mm256_loadu_ps(v3.as_ptr());
                        sum0 = _mm256_fmadd_ps(qv, v0v, sum0);
                        sum1 = _mm256_fmadd_ps(qv, v1v, sum1);
                        sum2 = _mm256_fmadd_ps(qv, v2v, sum2);
                        sum3 = _mm256_fmadd_ps(qv, v3v, sum3);
                    }
                    unsafe fn hsum(v: __m256) -> f32 {
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
                        tmp.iter().sum()
                    }
                    [hsum(sum0), hsum(sum1), hsum(sum2), hsum(sum3)]
                }
                return dot_product_batch_4_avx2(query, vectors);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe fn dot_product_batch_4_neon(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
                const N_LANES: usize = 4;
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut sum2 = vdupq_n_f32(0.0);
                let mut sum3 = vdupq_n_f32(0.0);

                let len = query.len();
                let chunks = len / N_LANES;
                for i in 0..chunks {
                    let base = i * N_LANES;
                    let qv = vld1q_f32(query.as_ptr().add(base));
                    let v0 = vld1q_f32(vectors[0].as_ptr().add(base));
                    let v1 = vld1q_f32(vectors[1].as_ptr().add(base));
                    let v2 = vld1q_f32(vectors[2].as_ptr().add(base));
                    let v3 = vld1q_f32(vectors[3].as_ptr().add(base));
                    sum0 = vaddq_f32(sum0, vmulq_f32(qv, v0));
                    sum1 = vaddq_f32(sum1, vmulq_f32(qv, v1));
                    sum2 = vaddq_f32(sum2, vmulq_f32(qv, v2));
                    sum3 = vaddq_f32(sum3, vmulq_f32(qv, v3));
                }
                // horizontal sums
                let mut out = [0.0f32; 4];
                out[0] = vaddvq_f32(sum0);
                out[1] = vaddvq_f32(sum1);
                out[2] = vaddvq_f32(sum2);
                out[3] = vaddvq_f32(sum3);
                // remainder
                for i in (chunks * N_LANES)..len {
                    let q = query[i];
                    out[0] += q * vectors[0][i];
                    out[1] += q * vectors[1][i];
                    out[2] += q * vectors[2][i];
                    out[3] += q * vectors[3][i];
                }
                out
            }
            return dot_product_batch_4_neon(query, vectors);
        }
        // fallback scalar
        dense_dot_product_batch_4_unrolled(query, vectors)
    }
}


impl DotProduct<f16> for f16 {
    unsafe fn dot_product_unrolled(query: &[f16], values: &[f16]) -> f32 {
        // x86_64 + runtime check for AVX2+F16C
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
                #[target_feature(enable = "avx2,f16c")]
                unsafe fn dot_product_unrolled_avx2(query: &[f16], document: &[f16]) -> f32 {
                    const N_LANES: usize = 8;
                    let mut sum = _mm256_setzero_ps();

                    let chunks = query.len() / N_LANES;

                    // Process chunks of 8 half values at a time.
                    for (q_chunk, v_chunk) in query
                        .chunks_exact(N_LANES)
                        .zip(document.chunks_exact(N_LANES))
                    {
                        // Load 8 half values as a 128-bit integer.
                        let q_half = _mm_loadu_si128(q_chunk.as_ptr() as *const __m128i);
                        let v_half = _mm_loadu_si128(v_chunk.as_ptr() as *const __m128i);

                        // Convert 8 half values to 8 single-precision floats.
                        let q_vec = _mm256_cvtph_ps(q_half);
                        let v_vec = _mm256_cvtph_ps(v_half);

                        // Multiply and accumulate.
                        let prod = _mm256_mul_ps(q_vec, v_vec);
                        sum = _mm256_add_ps(sum, prod);
                    }

                    // Store the SIMD sum into an array and sum its lanes.
                    let mut result = [0.0; N_LANES];
                    _mm256_storeu_ps(result.as_mut_ptr(), sum);
                    let simd_sum: f32 = result.iter().sum();

                    // Handle any remainder elements.
                    let remainder_start = chunks * N_LANES;
                    let remainder_sum: f32 = query[remainder_start..]
                        .iter()
                        .zip(&document[remainder_start..])
                        .fold(0.0, |acc, (&q, &v)| acc + q.to_f32() * v.to_f32());

                    simd_sum + remainder_sum
                }
                return dot_product_unrolled_avx2(query, values);
            }
        }

        // --- aarch64 native FP16 NEON path (nightly + hardware must support) ---
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("fp16") {
                // Use FP16 NEON intrinsics
                #[target_feature(enable = "neon,fp16")]
                unsafe fn dot_product_unrolled_neon(query: &[f16], values: &[f16]) -> f32 {
                    use core::arch::aarch64::*;
                    const N: usize = 4; // float16x4 -> float32x4
                    let len = query.len();
                    let chunks = len / N;
                    let mut sum_v = vdupq_n_f32(0.0);
                    for i in 0..chunks {
                        let base = i * N;
                        // Load 4 f16 values into a float16x4_t
                        let qh = vld1_f16(query.as_ptr().add(base) as *const f16);
                        let vh = vld1_f16(values.as_ptr().add(base) as *const f16);
                        // Convert to float32x4_t
                        let qv = vcvt_f32_f16(qh);
                        let vv = vcvt_f32_f16(vh);
                        sum_v = vaddq_f32(sum_v, vmulq_f32(qv, vv));
                    }
                    let mut acc = vaddvq_f32(sum_v);
                    for i in (chunks * N)..len {
                        acc += query[i].to_f32() * values[i].to_f32();
                    }
                    acc
                }
                return dot_product_unrolled_neon(query, values);
            }
            // Else: no FP16 support; fall through to “convert-then-NEON-f32” or scalar
            // Option A: convert small chunks to f32 and use NEON f32:
            #[target_feature(enable = "neon")]
            unsafe fn dot_product_unrolled_neon_via_f32(query: &[f16], values: &[f16]) -> f32 {
                use core::arch::aarch64::*;
                const CHUNK: usize = 4;
                let len = query.len();
                let chunks = len / CHUNK;
                let mut sum_v = vdupq_n_f32(0.0);
                // temporary stack buffers
                let mut qf = [0f32; CHUNK];
                let mut vf = [0f32; CHUNK];
                for i in 0..chunks {
                    let base = i * CHUNK;
                    for j in 0..CHUNK {
                        qf[j] = query[base + j].to_f32();
                        vf[j] = values[base + j].to_f32();
                    }
                    let qv = vld1q_f32(qf.as_ptr());
                    let vv = vld1q_f32(vf.as_ptr());
                    sum_v = vaddq_f32(sum_v, vmulq_f32(qv, vv));
                }
                let mut acc = vaddvq_f32(sum_v);
                for i in (chunks * CHUNK)..len {
                    acc += query[i].to_f32() * values[i].to_f32();
                }
                acc
            }
            // You may benchmark neon_via_f32 vs scalar to decide; here we choose neon_via_f32:
            return dot_product_unrolled_neon_via_f32(query, values);
        }

        // Scalar fallback
        dense_dot_product_unrolled(query, values)
    }

    unsafe fn dot_product_batch_4(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
        // x86_64 + runtime check for AVX2+F16C
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("f16c") {
                #[target_feature(enable = "avx2,f16c,fma")]
                unsafe fn dot_product_batch_4_avx2(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
                    // We process 8 half-precision values (each 16 bits) at a time.
                    const N_LANES: usize = 8;

                    let mut sum_0 = _mm256_setzero_ps();
                    let mut sum_1 = _mm256_setzero_ps();
                    let mut sum_2 = _mm256_setzero_ps();
                    let mut sum_3 = _mm256_setzero_ps();

                    // Iterate over chunks of 8 half values from the query and each document.
                    for (q_chunk, v0_chunk, v1_chunk, v2_chunk, v3_chunk) in izip!(
                        query.chunks_exact(N_LANES),
                        vectors[0].chunks_exact(N_LANES),
                        vectors[1].chunks_exact(N_LANES),
                        vectors[2].chunks_exact(N_LANES),
                        vectors[3].chunks_exact(N_LANES)
                    ) {
                        // Load 8 half values as a 128-bit integer.
                        let q_half = _mm_loadu_si128(q_chunk.as_ptr() as *const __m128i);
                        let v0_half = _mm_loadu_si128(v0_chunk.as_ptr() as *const __m128i);
                        let v1_half = _mm_loadu_si128(v1_chunk.as_ptr() as *const __m128i);
                        let v2_half = _mm_loadu_si128(v2_chunk.as_ptr() as *const __m128i);
                        let v3_half = _mm_loadu_si128(v3_chunk.as_ptr() as *const __m128i);

                        // Convert the 8 f16 values to 8 f32 values.
                        let q_values = _mm256_cvtph_ps(q_half);
                        let v0_values = _mm256_cvtph_ps(v0_half);
                        let v1_values = _mm256_cvtph_ps(v1_half);
                        let v2_values = _mm256_cvtph_ps(v2_half);
                        let v3_values = _mm256_cvtph_ps(v3_half);

                        // Fused multiply-add: sum_i += q * v_i for each vector.
                        sum_0 = _mm256_fmadd_ps(q_values, v0_values, sum_0);
                        sum_1 = _mm256_fmadd_ps(q_values, v1_values, sum_1);
                        sum_2 = _mm256_fmadd_ps(q_values, v2_values, sum_2);
                        sum_3 = _mm256_fmadd_ps(q_values, v3_values, sum_3);
                    }

                    [
                        horizontal_sum_256(sum_0),
                        horizontal_sum_256(sum_1),
                        horizontal_sum_256(sum_2),
                        horizontal_sum_256(sum_3),
                    ]
                }
                return dot_product_batch_4_avx2(query, vectors);
            }
        }

        // aarch64 FP16 NEON or via-f32 or scalar
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("fp16") {
                #[target_feature(enable = "neon,fp16")]
                unsafe fn dot_product_batch_4_neon(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
                    use core::arch::aarch64::*;
                    const N: usize = 4;
                    let len = query.len();
                    let chunks = len / N;
                    let mut sum0 = vdupq_n_f32(0.0);
                    let mut sum1 = vdupq_n_f32(0.0);
                    let mut sum2 = vdupq_n_f32(0.0);
                    let mut sum3 = vdupq_n_f32(0.0);
                    for i in 0..chunks {
                        let base = i * N;
                        let qh = vld1_f16(query.as_ptr().add(base) as *const f16);
                        let qv = vcvt_f32_f16(qh);
                        let v0v = vcvt_f32_f16(vld1_f16(vectors[0].as_ptr().add(base) as *const f16));
                        let v1v = vcvt_f32_f16(vld1_f16(vectors[1].as_ptr().add(base) as *const f16));
                        let v2v = vcvt_f32_f16(vld1_f16(vectors[2].as_ptr().add(base) as *const f16));
                        let v3v = vcvt_f32_f16(vld1_f16(vectors[3].as_ptr().add(base) as *const f16));
                        sum0 = vaddq_f32(sum0, vmulq_f32(qv, v0v));
                        sum1 = vaddq_f32(sum1, vmulq_f32(qv, v1v));
                        sum2 = vaddq_f32(sum2, vmulq_f32(qv, v2v));
                        sum3 = vaddq_f32(sum3, vmulq_f32(qv, v3v));
                    }
                    let mut out = [0.0f32; 4];
                    out[0] = vaddvq_f32(sum0);
                    out[1] = vaddvq_f32(sum1);
                    out[2] = vaddvq_f32(sum2);
                    out[3] = vaddvq_f32(sum3);
                    for i in (chunks * N)..len {
                        let qf = query[i].to_f32();
                        out[0] += qf * vectors[0][i].to_f32();
                        out[1] += qf * vectors[1][i].to_f32();
                        out[2] += qf * vectors[2][i].to_f32();
                        out[3] += qf * vectors[3][i].to_f32();
                    }
                    out
                }
                return dot_product_batch_4_neon(query, vectors);
            }
            // else via-f32 NEON:
            #[target_feature(enable = "neon")]
            unsafe fn dot_product_batch_4_neon_via_f32(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
                use core::arch::aarch64::*;
                const CHUNK: usize = 4;
                let len = query.len();
                let chunks = len / CHUNK;
                // accumulators as NEON registers
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut sum2 = vdupq_n_f32(0.0);
                let mut sum3 = vdupq_n_f32(0.0);

                // temp buffers for conversion
                let mut qf = [0f32; CHUNK];
                let mut v0f = [0f32; CHUNK];
                let mut v1f = [0f32; CHUNK];
                let mut v2f = [0f32; CHUNK];
                let mut v3f = [0f32; CHUNK];

                for chunk_idx in 0..chunks {
                    let base = chunk_idx * CHUNK;
                    // convert f16 → f32 for query and each vector
                    for j in 0..CHUNK {
                        qf[j]  = query[base + j].to_f32();
                        v0f[j] = vectors[0][base + j].to_f32();
                        v1f[j] = vectors[1][base + j].to_f32();
                        v2f[j] = vectors[2][base + j].to_f32();
                        v3f[j] = vectors[3][base + j].to_f32();
                    }
                    // load into NEON registers
                    let qv  = vld1q_f32(qf.as_ptr());
                    let v0v = vld1q_f32(v0f.as_ptr());
                    let v1v = vld1q_f32(v1f.as_ptr());
                    let v2v = vld1q_f32(v2f.as_ptr());
                    let v3v = vld1q_f32(v3f.as_ptr());
                    // accumulate
                    sum0 = vaddq_f32(sum0, vmulq_f32(qv, v0v));
                    sum1 = vaddq_f32(sum1, vmulq_f32(qv, v1v));
                    sum2 = vaddq_f32(sum2, vmulq_f32(qv, v2v));
                    sum3 = vaddq_f32(sum3, vmulq_f32(qv, v3v));
                }

                // horizontal sums of vector accumulators
                let mut out0 = vaddvq_f32(sum0);
                let mut out1 = vaddvq_f32(sum1);
                let mut out2 = vaddvq_f32(sum2);
                let mut out3 = vaddvq_f32(sum3);

                // remainder elements
                for i in (chunks * CHUNK)..len {
                    let qfv = query[i].to_f32();
                    out0 += qfv * vectors[0][i].to_f32();
                    out1 += qfv * vectors[1][i].to_f32();
                    out2 += qfv * vectors[2][i].to_f32();
                    out3 += qfv * vectors[3][i].to_f32();
                }
                [out0, out1, out2, out3]
            }
            return dot_product_batch_4_neon_via_f32(query, vectors);
        }

        // fallback scalar
        dense_dot_product_batch_4_unrolled(query, vectors)
    }
}




