use crate::distances::dot_product_unrolled_avx;
use crate::simd_utils::horizontal_sum_256;
use crate::{AsRefItem, DArray1, DenseDArray1};
use half::f16;
use itertools::izip;
use std::arch::x86_64::*;
use std::iter::zip;

use crate::Float;

#[inline]
pub fn vectors_norm(vectors: &[f32], d: usize) -> Vec<f32> {
    vectors
        .chunks_exact(d)
        .map(|v| dot_product_unrolled_avx(v, v))
        .collect()
}

#[inline]
pub fn euclidean_distance_batch_4<T>(query: &[T], values: [&[T]; 4]) -> [f32; 4]
where
    T: Float + EuclideanDistance<T>,
{
    unsafe { T::euclidean_distance_batch_4(query, values) }
}

#[inline]
pub fn euclidean_distance_unrolled<T>(query: &[T], values: &[T]) -> f32
where
    T: Float + EuclideanDistance<T>,
{
    unsafe { T::euclidean_distance_unrolled(query, values) }
}

pub trait EuclideanDistance<U> {
    unsafe fn euclidean_distance_unrolled(query: &[U], values: &[U]) -> f32;
    fn euclidean_distance_general(query: &[U], values: &[U]) -> f32;
    unsafe fn euclidean_distance_batch_4(query: &[U], vectors: [&[U]; 4]) -> [f32; 4];
}

impl EuclideanDistance<f16> for f16 {
    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn euclidean_distance_batch_4(query: &[f16], vectors: [&[f16]; 4]) -> [f32; 4] {
        const N_LANES: usize = 8; // 8 half values per 128-bit register (8*16 = 128)

        // Zero out accumulators (each is a __m256, holding 8 f32s).
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
            // Load 8 f16 values (16 bytes) into a __m128i and convert to f32.
            let q_values = _mm256_cvtph_ps(_mm_loadu_si128(q_chunk.as_ptr() as *const __m128i));
            let v0_values = _mm256_cvtph_ps(_mm_loadu_si128(v0_chunk.as_ptr() as *const __m128i));
            let v1_values = _mm256_cvtph_ps(_mm_loadu_si128(v1_chunk.as_ptr() as *const __m128i));
            let v2_values = _mm256_cvtph_ps(_mm_loadu_si128(v2_chunk.as_ptr() as *const __m128i));
            let v3_values = _mm256_cvtph_ps(_mm_loadu_si128(v3_chunk.as_ptr() as *const __m128i));

            // Compute differences.
            let diff_0 = _mm256_sub_ps(q_values, v0_values);
            let diff_1 = _mm256_sub_ps(q_values, v1_values);
            let diff_2 = _mm256_sub_ps(q_values, v2_values);
            let diff_3 = _mm256_sub_ps(q_values, v3_values);

            // Accumulate squared differences.
            sum_0 = _mm256_fmadd_ps(diff_0, diff_0, sum_0);
            sum_1 = _mm256_fmadd_ps(diff_1, diff_1, sum_1);
            sum_2 = _mm256_fmadd_ps(diff_2, diff_2, sum_2);
            sum_3 = _mm256_fmadd_ps(diff_3, diff_3, sum_3);
        }

        // Horizontally sum the 8 f32 lanes in each __m256 accumulator.
        let distance_0 = horizontal_sum_256(sum_0);
        let distance_1 = horizontal_sum_256(sum_1);
        let distance_2 = horizontal_sum_256(sum_2);
        let distance_3 = horizontal_sum_256(sum_3);

        [distance_0, distance_1, distance_2, distance_3]
    }

    #[inline]
    unsafe fn euclidean_distance_unrolled(query: &[f16], values: &[f16]) -> f32 {
        const N_LANES: usize = 16;
        let mut r = [f16::ZERO; N_LANES];

        let chunks = query.len() / N_LANES;
        for (q_chunk, v_chunk) in zip(query.chunks_exact(N_LANES), values.chunks_exact(N_LANES)) {
            for i in 0..N_LANES {
                let diff = q_chunk[i] - v_chunk[i];
                r[i] = r[i] + diff * diff;
            }
        }

        let result: f16 = r.iter().cloned().fold(f16::default(), |acc, val| acc + val);
        result.to_f32()
            + Self::euclidean_distance_general(
                &query[N_LANES * chunks..],
                &values[N_LANES * chunks..],
            )
    }

    #[inline]
    fn euclidean_distance_general(query: &[f16], values: &[f16]) -> f32 {
        query
            .iter()
            .zip(values)
            .fold(f16::default(), |acc: f16, (a, b)| {
                let diff = *a - *b;
                acc + diff * diff
            })
            .to_f32()
    }
}

impl EuclideanDistance<f32> for f32 {
    #[inline]
    unsafe fn euclidean_distance_unrolled(query: &[f32], values: &[f32]) -> f32 {
        const N_LANES: usize = 8;

        let mut sum = _mm256_setzero_ps();
        for (q_chunk, v_chunk) in zip(query.chunks_exact(N_LANES), values.chunks_exact(N_LANES)) {
            let q_values = _mm256_loadu_ps(q_chunk.as_ptr());
            let v_values = _mm256_loadu_ps(v_chunk.as_ptr());

            let diff = _mm256_sub_ps(q_values, v_values);

            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        horizontal_sum_256(sum)
    }

    #[inline]
    fn euclidean_distance_general(query: &[f32], values: &[f32]) -> f32 {
        query.iter().zip(values).fold(0.0, |acc, (a, b)| {
            let diff = a - b;
            acc + diff * diff
        })
    }

    #[inline]
    unsafe fn euclidean_distance_batch_4(query: &[f32], vectors: [&[f32]; 4]) -> [f32; 4] {
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

            let diff_0 = _mm256_sub_ps(q_values, v0_values);
            let diff_1 = _mm256_sub_ps(q_values, v1_values);
            let diff_2 = _mm256_sub_ps(q_values, v2_values);
            let diff_3 = _mm256_sub_ps(q_values, v3_values);

            sum_0 = _mm256_fmadd_ps(diff_0, diff_0, sum_0);
            sum_1 = _mm256_fmadd_ps(diff_1, diff_1, sum_1);
            sum_2 = _mm256_fmadd_ps(diff_2, diff_2, sum_2);
            sum_3 = _mm256_fmadd_ps(diff_3, diff_3, sum_3);
        }

        let distance_0 = horizontal_sum_256(sum_0);
        let distance_1 = horizontal_sum_256(sum_1);
        let distance_2 = horizontal_sum_256(sum_2);
        let distance_3 = horizontal_sum_256(sum_3);

        [distance_0, distance_1, distance_2, distance_3]
    }
}

#[inline]
fn dense_euclidean_distance_general<T, U>(query: &DenseDArray1<T>, values: &DenseDArray1<T>) -> f32
where
    T: AsRefItem<Item = U>,
    U: Float,
{
    query
        .values_as_slice()
        .iter()
        .zip(values.values_as_slice())
        .fold(0.0, |acc, (a, b)| {
            let diff = a.to_f32().unwrap() - b.to_f32().unwrap();
            acc + diff * diff
        })
}

#[inline]
pub fn dense_euclidean_distance_unrolled<T, U>(
    query: &DenseDArray1<T>,
    values: &DenseDArray1<T>,
) -> f32
where
    T: AsRefItem<Item = U>,
    U: Float,
{
    const N_LANES: usize = 16;
    let mut r = [0.0; N_LANES];

    let chunks = query.len() / N_LANES;
    for (q_chunk, v_chunk) in zip(
        query.values_as_slice().chunks_exact(N_LANES),
        values.values_as_slice().chunks_exact(N_LANES),
    ) {
        for i in 0..N_LANES {
            let d = q_chunk[i].to_f32().unwrap() - v_chunk[i].to_f32().unwrap();
            r[i] += d * d;
        }
    }

    r.iter().fold(0.0, |acc, &val| acc + val)
        + dense_euclidean_distance_general(
            &DenseDArray1::new(&query.values_as_slice()[N_LANES * chunks..]),
            &DenseDArray1::new(&values.values_as_slice()[N_LANES * chunks..]),
        )
}
