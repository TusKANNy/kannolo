use crate::simd_utils::horizontal_sum_256;
use crate::{AsRefItem, DArray1, DenseDArray1, Float, SparseDArray1};
use itertools::izip;
use std::arch::x86_64::*;
use std::iter::zip;

use half::f16;

#[inline]
pub fn dense_dot_product<T, U>(query: &DenseDArray1<T>, document: &DenseDArray1<T>) -> f32
where
    T: AsRef<[U]> + AsRefItem<Item = U>,
    U: Float,
{
    dense_dot_product_unrolled(query, document)
}

#[inline]
pub fn dense_dot_product_unrolled<T, U>(query: &DenseDArray1<T>, document: &DenseDArray1<T>) -> f32
where
    T: AsRef<[U]> + AsRefItem<Item = U>,
    U: Float,
{
    const N_LANES: usize = 8;
    let mut r = [0.0; N_LANES];

    let chunks = query.len() / N_LANES;
    for (q_chunk, v_chunk) in zip(
        query.values_as_slice().chunks_exact(N_LANES),
        document.values_as_slice().chunks_exact(N_LANES),
    ) {
        for i in 0..N_LANES {
            let d = q_chunk[i].to_f32().unwrap() * v_chunk[i].to_f32().unwrap();
            r[i] += d;
        }
    }

    r.iter().sum::<f32>()
        + dense_dot_product_general(
            &DenseDArray1::new(&query.values_as_slice()[N_LANES * chunks..]),
            &DenseDArray1::new(&document.values_as_slice()[N_LANES * chunks..]),
        )
}

#[inline]
pub fn dense_dot_product_general<T, U>(query: &DenseDArray1<T>, document: &DenseDArray1<T>) -> f32
where
    T: AsRef<[U]> + AsRefItem<Item = U>,
    U: Float,
{
    query
        .values_as_slice()
        .iter()
        .zip(document.values_as_slice())
        .fold(0.0, |acc, (a, b)| {
            acc + (a.to_f32().unwrap() * b.to_f32().unwrap())
        })
}

// Sparse
#[inline]
pub fn dot_product_dense_sparse<T1, T, U, F>(
    query: &DenseDArray1<T1>,
    array: &SparseDArray1<U, T>,
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
    query: &SparseDArray1<U, T>,
    vector: &SparseDArray1<U, T>,
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
pub fn dot_product_unrolled_avx<T>(query: &[T], values: &[T]) -> f32
where
    T: Float + DotProduct<T>,
{
    unsafe { T::dot_product_unrolled_avx2(query, values) }
}

pub trait DotProduct<U> {
    unsafe fn dot_product_unrolled_avx2(query: &[U], values: &[U]) -> f32;
    unsafe fn dot_product_batch_4(query: &[U], values: [&[U]; 4]) -> [f32; 4];
}

impl DotProduct<f32> for f32 {
    unsafe fn dot_product_unrolled_avx2(query: &[f32], document: &[f32]) -> f32 {
        const N_LANES: usize = 8;

        let mut sum = _mm256_setzero_ps();

        let chunks = query.len() / N_LANES;

        for (q_chunk, v_chunk) in query
            .chunks_exact(N_LANES)
            .zip(document.chunks_exact(N_LANES))
        {
            let q_vec = _mm256_loadu_ps(q_chunk.as_ptr());
            let v_vec = _mm256_loadu_ps(v_chunk.as_ptr());
            let prod = _mm256_mul_ps(q_vec, v_vec);
            sum = _mm256_add_ps(sum, prod);
        }

        let mut result = [0.0; N_LANES];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);

        let simd_sum: f32 = result.iter().sum();

        simd_sum
            + query[chunks * N_LANES..]
                .iter()
                .zip(&document[chunks * N_LANES..])
                .fold(0.0, |acc, (&a, &b)| acc + a * b)
    }

    unsafe fn dot_product_batch_4(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
        const N_LANES: usize = 8;

        let mut sum_0 = _mm256_setzero_ps();
        let mut sum_1 = _mm256_setzero_ps();
        let mut sum_2 = _mm256_setzero_ps();
        let mut sum_3 = _mm256_setzero_ps();

        for (q_chunk, v0_chunk, v1_chunk, v2_chunk, v3_chunk) in izip!(
            query.chunks_exact(N_LANES),
            vectors[0].chunks_exact(N_LANES),
            vectors[1].chunks_exact(N_LANES),
            vectors[2].chunks_exact(N_LANES),
            vectors[3].chunks_exact(N_LANES)
        ) {
            let q_values = _mm256_loadu_ps(q_chunk.as_ptr());

            let v0_values = _mm256_loadu_ps(v0_chunk.as_ptr());
            let v1_values = _mm256_loadu_ps(v1_chunk.as_ptr());
            let v2_values = _mm256_loadu_ps(v2_chunk.as_ptr());
            let v3_values = _mm256_loadu_ps(v3_chunk.as_ptr());

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
}

impl DotProduct<f16> for f16 {
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

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn dot_product_batch_4(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
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
}
