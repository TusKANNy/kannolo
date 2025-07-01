#[cfg(target_arch = "x86_64")]
use super::transpose::{transpose_8x2, transpose_8x4, transpose_8x8};
#[cfg(target_arch = "x86_64")]
use super::utils::{
    horizontal_sum_128, horizontal_sum_256, squared_l2_dist_128, squared_l2_dist_256,
};
use crate::utils::compute_squared_l2_distance;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/* ********** SIMD OPTIMIZED FUNCTIONS ********** */

#[cfg(target_arch = "x86_64")]
pub unsafe fn compute_distance_table_ip_d4(
    distance_table: &mut [f32],
    query: &[f32],
    centroids: &[f32],
    ksub: usize,
) {
    let mut i = 0;
    let mut centroids_ptr = centroids.as_ptr();
    let centroid_groups = ksub / 8;

    if centroid_groups > 0 {
        let m0 = _mm256_set1_ps(query[0]);
        let m1 = _mm256_set1_ps(query[1]);
        let m2 = _mm256_set1_ps(query[2]);
        let m3 = _mm256_set1_ps(query[3]);

        for j in (0..centroid_groups * 8).step_by(8) {
            let [v0, v1, v2, v3] = transpose_8x4(
                _mm256_loadu_ps(centroids_ptr.add(0 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(1 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(2 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(3 * 8)),
            );

            let mut distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);
            distances = _mm256_fmadd_ps(m2, v2, distances);
            distances = _mm256_fmadd_ps(m3, v3, distances);

            _mm256_storeu_ps(distance_table.as_mut_ptr().add(j), distances);

            centroids_ptr = centroids_ptr.add(32);
        }

        i = centroid_groups * 8;
    }

    if i < ksub {
        let x0 = _mm_loadu_ps(query.as_ptr());

        for j in i..ksub {
            let accu = _mm_mul_ps(x0, _mm_loadu_ps(centroids_ptr));
            centroids_ptr = centroids_ptr.add(4);
            distance_table[j] = horizontal_sum_128(accu);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn compute_distance_table_ip_d8(
    distance_table: &mut [f32],
    query: &[f32],
    centroids: &[f32],
    ksub: usize,
) {
    let mut i = 0;
    let mut centroids_ptr = centroids.as_ptr();
    let centroid_groups = ksub / 8;

    if centroid_groups > 0 {
        // Broadcast each query component for dsub == 8.
        let m0 = _mm256_set1_ps(query[0]);
        let m1 = _mm256_set1_ps(query[1]);
        let m2 = _mm256_set1_ps(query[2]);
        let m3 = _mm256_set1_ps(query[3]);
        let m4 = _mm256_set1_ps(query[4]);
        let m5 = _mm256_set1_ps(query[5]);
        let m6 = _mm256_set1_ps(query[6]);
        let m7 = _mm256_set1_ps(query[7]);

        for j in (0..centroid_groups * 8).step_by(8) {
            // Load 8 registers (each with 8 floats) from the interleaved centroid data.
            // Each centroid consists of 8 contiguous floats.
            let [v0, v1, v2, v3, v4, v5, v6, v7] = transpose_8x8(
                _mm256_loadu_ps(centroids_ptr.add(0 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(1 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(2 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(3 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(4 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(5 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(6 * 8)),
                _mm256_loadu_ps(centroids_ptr.add(7 * 8)),
            );

            // Compute the dot product for 8 centroids:
            // distances[i] = query[0]*v0[i] + query[1]*v1[i] + ... + query[7]*v7[i]
            let mut distances = _mm256_mul_ps(m0, v0);
            distances = _mm256_fmadd_ps(m1, v1, distances);
            distances = _mm256_fmadd_ps(m2, v2, distances);
            distances = _mm256_fmadd_ps(m3, v3, distances);
            distances = _mm256_fmadd_ps(m4, v4, distances);
            distances = _mm256_fmadd_ps(m5, v5, distances);
            distances = _mm256_fmadd_ps(m6, v6, distances);
            distances = _mm256_fmadd_ps(m7, v7, distances);

            _mm256_storeu_ps(distance_table.as_mut_ptr().add(j), distances);

            // Advance the centroids pointer by 8 floats per centroid * 8 centroids.
            centroids_ptr = centroids_ptr.add(8 * 8);
        }
        i = centroid_groups * 8;
    }

    // Process any remaining centroids (if ksub is not a multiple of 8).
    if i < ksub {
        let x0 = _mm_loadu_ps(query.as_ptr());

        for j in i..ksub {
            let accu = _mm_mul_ps(x0, _mm_loadu_ps(centroids_ptr));
            centroids_ptr = centroids_ptr.add(4);
            distance_table[j] = horizontal_sum_128(accu);
        }
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn compute_l2_sqr_avx2_d4(query: &[f32], centroids_ptr: *const f32) -> [f32; 8] {
    let mut distances = [0.0; 8];

    // Prepare AVX2 registers for the query vector
    let query_avx = [
        _mm256_set1_ps(query[0]),
        _mm256_set1_ps(query[1]),
        _mm256_set1_ps(query[2]),
        _mm256_set1_ps(query[3]),
    ];

    // Load centroids data into AVX2 registers
    let centroids_avx = [
        _mm256_loadu_ps(centroids_ptr.add(0 * 8)),
        _mm256_loadu_ps(centroids_ptr.add(1 * 8)),
        _mm256_loadu_ps(centroids_ptr.add(2 * 8)),
        _mm256_loadu_ps(centroids_ptr.add(3 * 8)),
    ];

    // Transpose the centroids data
    let transposed = transpose_8x4(
        centroids_avx[0],
        centroids_avx[1],
        centroids_avx[2],
        centroids_avx[3],
    );

    // Compute the squared Euclidean distance
    let mut dists_avx = _mm256_mul_ps(
        _mm256_sub_ps(query_avx[0], transposed[0]),
        _mm256_sub_ps(query_avx[0], transposed[0]),
    );

    for k in 1..4 {
        dists_avx = _mm256_fmadd_ps(
            _mm256_sub_ps(query_avx[k], transposed[k]),
            _mm256_sub_ps(query_avx[k], transposed[k]),
            dists_avx,
        );
    }

    _mm256_storeu_ps(distances.as_mut_ptr(), dists_avx);

    distances
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn find_nearest_centroid_avx2_d4(query: &[f32], centroids: &[f32], ksub: usize) -> usize {
    let mut curr_idx = 0;
    let mut min_dist = f32::MAX;
    let mut min_idx = 0;
    let centroid_groups = ksub / 8;

    let centroids_ptr = centroids.as_ptr();

    if centroid_groups > 0 {
        // Initialize AVX2 registers for tracking minimum distances and indices
        let mut avx_min_dist = _mm256_set1_ps(f32::MAX);
        let mut avx_min_idx = _mm256_set1_epi32(0);

        // Set up AVX2 registers for indexing centroids
        let mut avx_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let idx_increment = _mm256_set1_epi32(8);

        while curr_idx < centroid_groups * 8 {
            let distances = compute_l2_sqr_avx2_d4(query, centroids_ptr.add(curr_idx * 4));

            // Compare new distances with the minimum distances and update accordingly
            let cmp = _mm256_cmp_ps(
                avx_min_dist,
                _mm256_loadu_ps(distances.as_ptr()),
                _CMP_LT_OS,
            );
            avx_min_dist = _mm256_min_ps(_mm256_loadu_ps(distances.as_ptr()), avx_min_dist);
            avx_min_idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(avx_idx),
                _mm256_castsi256_ps(avx_min_idx),
                cmp,
            ));

            // Increment the indices for the next group of centroids
            avx_idx = _mm256_add_epi32(avx_idx, idx_increment);
            curr_idx += 8;
        }

        // Convert AVX2 results to scalar
        let mut scalar_dists = [0.0_f32; 8];
        let mut scalar_idxs = [0_u32; 8];
        _mm256_storeu_ps(scalar_dists.as_mut_ptr(), avx_min_dist);
        _mm256_storeu_si256(scalar_idxs.as_mut_ptr() as *mut __m256i, avx_min_idx);

        for j in 0..8 {
            if min_dist > scalar_dists[j] {
                min_dist = scalar_dists[j];
                min_idx = scalar_idxs[j] as usize;
            }
        }
    }

    // Process the leftovers
    if curr_idx < ksub {
        while curr_idx < ksub {
            let distance = horizontal_sum_128(squared_l2_dist_128(
                _mm_loadu_ps(query.as_ptr()),
                _mm_loadu_ps(centroids_ptr.add(curr_idx * 4)),
            ));

            if min_dist > distance {
                min_dist = distance;
                min_idx = curr_idx;
            }
            curr_idx += 1;
        }
    }

    min_idx
}

#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn compute_distance_table_avx2_d2(
    distance_table: &mut [f32],
    query: &[f32],
    centroids: &[f32],
    ksub: usize,
) {
    let mut i = 0;
    let mut centroids_ptr = centroids.as_ptr();
    let centroid_groups = ksub / 8;

    if centroid_groups > 0 {
        _mm_prefetch(centroids.as_ptr() as *const i8, _MM_HINT_T0);
        _mm_prefetch(centroids.as_ptr().add(16) as *const i8, _MM_HINT_T0);

        let m0 = _mm256_set1_ps(query[0]);
        let m1 = _mm256_set1_ps(query[1]);

        for j in (0..centroid_groups * 8).step_by(8) {
            _mm_prefetch(centroids_ptr.add(32) as *const i8, _MM_HINT_T0);

            let mut v0 = _mm256_setzero_ps();
            let mut v1 = _mm256_setzero_ps();

            transpose_8x2(
                _mm256_loadu_ps(centroids_ptr.add(0)),
                _mm256_loadu_ps(centroids_ptr.add(8)),
                &mut v0,
                &mut v1,
            );

            let d0 = _mm256_sub_ps(m0, v0);
            let d1 = _mm256_sub_ps(m1, v1);

            let mut distances = _mm256_mul_ps(d0, d0);
            distances = _mm256_fmadd_ps(d1, d1, distances);

            _mm256_storeu_ps(distance_table.as_mut_ptr().add(j), distances);

            centroids_ptr = centroids_ptr.add(16);
        }

        i = centroid_groups * 8;
    }

    if i < ksub {
        let x0 = query[0];
        let x1 = query[1];

        for j in i..ksub {
            let sub0 = x0 - centroids[0];
            let sub1 = x1 - centroids[1];
            let distance = sub0 * sub0 + sub1 * sub1;

            centroids_ptr = centroids_ptr.add(2);
            distance_table[j] = distance;
        }
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn compute_distance_table_avx2_d4(
    distance_table: &mut [f32],
    query: &[f32],
    centroids: &[f32],
    ksub: usize,
) {
    let mut i = 0;
    let mut centroids_ptr = centroids.as_ptr();
    let centroid_groups = ksub / 8;

    if centroid_groups > 0 {
        while i < centroid_groups * 8 {
            let distances = compute_l2_sqr_avx2_d4(query, centroids_ptr);
            for j in 0..8 {
                distance_table[i + j] = distances[j];
            }
            centroids_ptr = centroids_ptr.add(32);
            i += 8;
        }
    }

    // Scalar fallback for remaining centroids
    if i < ksub {
        let query_avx = _mm_loadu_ps(query.as_ptr());
        for _ in i..centroid_groups {
            let accu = squared_l2_dist_128(query_avx, _mm_loadu_ps(centroids_ptr));
            distance_table[i] = horizontal_sum_128(accu);
            centroids_ptr = centroids_ptr.add(4);
        }
    }
}

/// Finds the nearest centroid to a given query vector `query_vec` from a set of centroids `centroids`
/// using SIMD (Single Instruction, Multiple Data) operations, optimized for AVX2 instruction set.
/// This function is designed for high-performance computation in scenarios where both the query vector
/// and the centroids can benefit from SIMD parallelism.
///
/// The function processes centroids in groups of 8, leveraging AVX2 capabilities, which work with
/// 256-bit wide registers, allowing for 8 floating-point operations simultaneously. This enhances
/// efficiency in high-dimensional space calculations.
///
/// # Arguments
///
/// * `query_vec` - A slice representing the input vector for which the nearest centroid is to be found.
///                 The length of `query_vec` is assumed to match the sub-dimension used in the calculation.
/// * `centroids` - A slice representing the set of centroids. Each centroid should have the same
///                 dimensionality as `query_vec`. The centroids are expected to be laid out contiguously in memory.
/// * `ksub`      - The number of centroids.
///
/// # Safety
///
/// This function is unsafe as it uses low-level SIMD intrinsics that require careful handling of
/// pointers and memory alignment.
///
/// # Returns
///
/// The index of the nearest centroid to the input vector `query_vec` within the set `centroids`.
///
/// # Detailed Workflow
///
/// 1. **Initialization**:
///    - Sets up initial variables for tracking the minimum distance and index.
///    - Calculates `centroid_groups`, the number of centroid groups to be processed in SIMD.
///
/// 2. **SIMD Processing**:
///    - Enters a SIMD-optimized loop if there are enough centroids to process in groups of 8.
///    - In each iteration, 8 centroids are loaded into SIMD registers.
///    - Performs element-wise subtraction and squaring using AVX2 operations to compute the squared
///      Euclidean distance.
///    - Tracks the minimum distances and their indices using AVX2 comparison and blend operations.
///    - Continues this process for all groups of centroids.
///
/// 3. **Extracting Minimum Distances**:
///    - Transfers the results from SIMD registers into scalar arrays for final comparison.
///    - Determines the minimum distance and its corresponding centroid index.
///
/// 4. **Processing Remaining Centroids**:
///    - If any centroids are left (not fitting into the SIMD-optimized processing),
///      they are processed individually.
///    - Uses scalar operations to compute the distance and update the minimum distance and index.
///
#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn find_nearest_centroid_avx2_d8(
    query_vec: &[f32],
    centroids: &[f32],
    ksub: usize,
) -> usize {
    let centroid_groups = ksub / 8;

    let mut min_dist = f32::MAX;
    let mut min_idx = 0;

    // Index for traversing through centroids
    let mut curr_idx = 0;

    let mut centroids_ptr = centroids.as_ptr();

    if centroid_groups > 0 {
        let mut avx_min_dist = _mm256_set1_ps(f32::MAX);
        let mut avx_min_idx = _mm256_set1_epi32(0);

        let mut avx_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        let idx_increment = _mm256_set1_epi32(8);

        let qvec_avx = [
            _mm256_set1_ps(query_vec[0]),
            _mm256_set1_ps(query_vec[1]),
            _mm256_set1_ps(query_vec[2]),
            _mm256_set1_ps(query_vec[3]),
            _mm256_set1_ps(query_vec[4]),
            _mm256_set1_ps(query_vec[5]),
            _mm256_set1_ps(query_vec[6]),
            _mm256_set1_ps(query_vec[7]),
        ];

        while curr_idx < centroid_groups * 8 {
            let c_avx = [
                _mm256_loadu_ps(centroids_ptr),
                _mm256_loadu_ps(centroids_ptr.add(8)),
                _mm256_loadu_ps(centroids_ptr.add(16)),
                _mm256_loadu_ps(centroids_ptr.add(24)),
                _mm256_loadu_ps(centroids_ptr.add(32)),
                _mm256_loadu_ps(centroids_ptr.add(40)),
                _mm256_loadu_ps(centroids_ptr.add(48)),
                _mm256_loadu_ps(centroids_ptr.add(56)),
            ];

            // Transpose the centroids for efficient SIMD computation
            let transposed = transpose_8x8(
                c_avx[0], c_avx[1], c_avx[2], c_avx[3], c_avx[4], c_avx[5], c_avx[6], c_avx[7],
            );

            // Compute squared Euclidean distances
            let mut dists_avx = _mm256_mul_ps(
                _mm256_sub_ps(qvec_avx[0], transposed[0]),
                _mm256_sub_ps(qvec_avx[0], transposed[0]),
            );
            for k in 1..8 {
                dists_avx = _mm256_fmadd_ps(
                    _mm256_sub_ps(qvec_avx[k], transposed[k]),
                    _mm256_sub_ps(qvec_avx[k], transposed[k]),
                    dists_avx,
                );
            }

            // Update the minimum distances and their indices
            let cmp = _mm256_cmp_ps(avx_min_dist, dists_avx, _CMP_LT_OS);
            avx_min_dist = _mm256_min_ps(dists_avx, avx_min_dist);
            avx_min_idx = _mm256_castps_si256(_mm256_blendv_ps(
                _mm256_castsi256_ps(avx_idx),
                _mm256_castsi256_ps(avx_min_idx),
                cmp,
            ));

            avx_idx = _mm256_add_epi32(avx_idx, idx_increment);
            centroids_ptr = centroids_ptr.add(64);
            curr_idx += 8;
        }

        // Extract minimum distances and their indices into scalar arrays
        let mut scalar_dists = [0.0_f32; 8];
        let mut scalar_idxs = [0_i32; 8];
        _mm256_storeu_ps(scalar_dists.as_mut_ptr(), avx_min_dist);
        _mm256_storeu_si256(scalar_idxs.as_mut_ptr() as *mut __m256i, avx_min_idx);

        // Find the global minimum distance and its index
        for j in 0..8 {
            if min_dist > scalar_dists[j] {
                min_dist = scalar_dists[j];
                min_idx = scalar_idxs[j] as usize;
            }
        }
    }

    // Process any remaining centroids not handled in the SIMD loop
    if curr_idx < ksub {
        let qvec_avx = _mm256_loadu_ps(query_vec.as_ptr());

        while curr_idx < ksub {
            let centroid_avx = _mm256_loadu_ps(centroids_ptr.add(curr_idx * 8));
            let dists_avx = squared_l2_dist_256(qvec_avx, centroid_avx);
            let dist = horizontal_sum_256(dists_avx);

            if min_dist > dist {
                min_dist = dist;
                min_idx = curr_idx;
            }

            curr_idx += 1;
            centroids_ptr = centroids_ptr.add(8);
        }
    }

    min_idx
}

/* ********** GENERAL METHOD ********** */

/// Calculates the squared L2 distances between a single-dimensional segment of a query vector
/// (`query_vec`) and each corresponding centroid in a set of centroids (`centroids`). This
/// function is specifically optimized for the case where `dsub = 1` in the context of
/// product quantization encoding.
///
/// SIMD (Single Instruction, Multiple Data) operations are used to optimize the computation,
/// making it suitable for high-performance scenarios where processing efficiency is crucial.
///
/// # Arguments
///
/// * `distances` - A mutable slice where the computed distances will be stored.
/// * `query_vec` - A slice representing the single-dimensional segment of the query vector.
/// * `centroids` - A slice representing the set of centroids. Each centroid should be
///   single-dimensional, corresponding to the `dsub = 1` scenario.
/// * `num_centroids` - The number of centroids in the `centroids` slice.
///
/// # Safety
///
/// This function is unsafe as it utilizes low-level SIMD intrinsics, requiring careful
/// handling of pointers and memory alignment.
///
/// # Returns
///
/// The function does not return a value but populates the `distances` slice with the
/// computed squared distances.
///
/// # Detailed Workflow
///
/// 1. **Vectorization of Query Vector**: The first element of `query_vec` is replicated
///    across a SIMD vector, enabling parallel computation of distances.
///
/// 2. **SIMD Processing of Centroids**:
///    - Centroids are processed using SIMD instructions.
///    - The function computes the squared Euclidean distance by element-wise subtraction,
///      followed by squaring, and stores the results in `distances`.
///
/// 3. **Scalar Processing for Remaining Centroids**:
///    - For any remaining centroids (if the total number isn't a multiple of 4), the
///      distances are computed individually using scalar operations.
///
#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn compute_distances_d1(
    distances: &mut [f32],
    query_vec: &[f32],
    centroids: &[f32],
    num_centroids: usize,
) {
    // Use the first element of the query vector for distance calculation
    let query_first = query_vec[0];
    // Replicate the element across SIMD vector
    let query_vectorized = _mm_set_ps(query_first, query_first, query_first, query_first);

    let mut centroid_index = 0;
    while centroid_index + 3 < num_centroids {
        // SIMD operations for batch processing of centroids
        let centroid_chunk = _mm_loadu_ps(centroids.as_ptr().add(centroid_index));
        let distance = squared_l2_dist_128(query_vectorized, centroid_chunk);

        // Unpack and store distances
        distances[centroid_index] = _mm_cvtss_f32(distance);
        distances[centroid_index + 1] = _mm_cvtss_f32(_mm_shuffle_ps(distance, distance, 0x55));
        distances[centroid_index + 2] = _mm_cvtss_f32(_mm_shuffle_ps(distance, distance, 0xAA));
        distances[centroid_index + 3] = _mm_cvtss_f32(_mm_shuffle_ps(distance, distance, 0xFF));

        centroid_index += 4;
    }

    // Handle remaining centroids if num_centroids is not a multiple of 4
    while centroid_index < num_centroids {
        let centroid_element = *centroids.get_unchecked(centroid_index);
        let diff = query_first - centroid_element;
        distances[centroid_index] = diff * diff;
        centroid_index += 1;
    }
}

/// Calculates the squared L2 distances between a 12-dimensional segment of a query vector
/// (`query_vec`) and each corresponding centroid in a set of centroids (`centroids`). This
/// function is specifically optimized for the case where `dsub = 12` in the context of
/// product quantization encoding, using SIMD operations.
///
/// The function splits the 12-dimensional segment into three 4-dimensional parts and
/// computes the distances using SIMD for parallel processing efficiency.
///
/// # Arguments
///
/// * `distances` - A mutable slice where computed distances will be stored.
/// * `query_vec` - A slice representing the 12-dimensional query vector segment.
/// * `centroids` - A slice representing the set of centroids. Each centroid
///   should be 12-dimensional.
/// * `num_centroids` - The number of centroids in the `centroids` slice.
///
/// # Safety
///
/// This function is unsafe due to the use of low-level SIMD intrinsics, requiring
/// careful handling of pointers and memory alignment.
///
/// # Returns
///
/// The function fills the `distances` slice with computed squared distances.
///
/// # Detailed Workflow
///
/// 1. **Vectorization of Query Vector**: Splits the query vector into three segments
///    and loads each into SIMD vectors.
///
/// 2. **SIMD Processing of Centroids**:
///    - Processes each centroid in segments corresponding to the query vector segments.
///    - Computes the squared Euclidean distance for each segment and sums them for
///      each centroid.
///    - Stores the total distances in `distances`.
///
#[inline]
#[cfg(target_arch = "x86_64")]
unsafe fn compute_distances_d12(
    distances: &mut [f32],
    query_vec: &[f32],
    centroids: &[f32],
    num_centroids: usize,
) {
    // Load segments of the 12-dimensional query vector into SIMD vectors
    let seg0 = _mm_loadu_ps(query_vec.as_ptr());
    let seg1 = _mm_loadu_ps(query_vec.as_ptr().add(4));
    let seg2 = _mm_loadu_ps(query_vec.as_ptr().add(8));

    let mut centroid_offset = 0;
    for i in 0..num_centroids {
        // SIMD operations for each segment of the centroids
        let centroid_seg0 = _mm_loadu_ps(centroids.as_ptr().add(centroid_offset));
        let mut distance_accumulator = squared_l2_dist_128(seg0, centroid_seg0);
        centroid_offset += 4;

        // Repeated for remaining segments
        let centroid_seg1 = _mm_loadu_ps(centroids.as_ptr().add(centroid_offset));
        let centroid_seg2 = _mm_loadu_ps(centroids.as_ptr().add(centroid_offset + 4));

        // Accumulate distances from each segment and store the result
        distance_accumulator = _mm_add_ps(
            distance_accumulator,
            squared_l2_dist_128(seg1, centroid_seg1),
        );
        distance_accumulator = _mm_add_ps(
            distance_accumulator,
            squared_l2_dist_128(seg2, centroid_seg2),
        );
        distances[i] = horizontal_sum_128(distance_accumulator); // Sum of distances from all segments
        centroid_offset += 8; // Move to the next set of centroid segments
    }
}

/// Identifies the index of the nearest centroid by finding the minimum distance in a given
/// array of distances. This function is used as the final step in determining the nearest
/// centroid in product quantization encoding.
///
/// # Arguments
///
/// * `distances` - A slice containing the distances of the query vector to each centroid.
///
/// # Returns
///
/// Returns the index of the centroid with the minimum distance in the `distances` array.
///
/// # Detailed Workflow
///
/// 1. **Minimum Distance Identification**: Iterates over the `distances` array, comparing
///    each distance to find the minimum.
///
/// 2. **Index Retrieval**: Returns the index of the minimum distance, which corresponds
///    to the nearest centroid.
///
#[inline]
fn find_nearest_centroid_index(distances: &[f32]) -> usize {
    distances
        .iter()
        .enumerate()
        .min_by(|(_, &dist_a), (_, &dist_b)| dist_a.partial_cmp(&dist_b).unwrap())
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Calculates the squared L2 distances between a query vector (`query_vec`) and each centroid
/// in a set of centroids (`centroids`). This general-purpose function is used when specific
/// SIMD optimizations (like d1 or d12) are not applicable.
///
/// It leverages `compute_squared_l2_distance` to compute distances for segments of the query vector.
///
/// # Arguments
///
/// * `distances` - A mutable slice where computed distances will be stored.
/// * `query_vec` - A slice representing the query vector.
/// * `centroids` - A slice representing the set of centroids.
/// * `ksub` - The length of each segment of the query vector (sub-vector length).
/// * `n_centroids` - The number of centroids in the `centroids` slice.
///
/// # Returns
///
/// The function fills the `distances` slice with computed squared distances.
///
/// # Detailed Workflow
///
/// 1. **Segment-wise Distance Calculation**: Iterates over segments of `query_vec` and
///    computes the squared L2 distance to corresponding segments in `centroids` using
///    `compute_squared_l2_distance`.
///
fn compute_distances_general(
    distances: &mut [f32],
    query_vec: &[f32],
    centroids: &[f32],
    ksub: usize,
    n_centroids: usize,
) {
    let mut offset = 0;
    for i in 0..n_centroids {
        // Calculate distances for each segment of the query vector
        distances[i] =
            compute_squared_l2_distance(query_vec, &centroids[offset..offset + ksub], ksub);
        offset += ksub;
    }
}

/// Finds the index of the nearest centroid to a given query vector `query_vec` from a set
/// of centroids `centroids`. This function selects the appropriate distance calculation
/// method based on `dsub` and whether the AVX2 instruction set is available.
///
/// # Arguments
///
/// * `query_vec` - A slice representing the query vector.
/// * `centroids` - A slice representing the set of centroids.
/// * `dsub` - The dimensionality of each segment in product quantization.
/// * `ksub` - The number of centroids.
///
/// # Returns
///
/// Returns the index of the nearest centroid to the `query_vec` within the set `centroids`.
///
#[cfg(target_feature = "avx2")]
fn find_nearest_centroid_general(
    query_vec: &[f32],
    centroids: &[f32],
    dsub: usize,
    ksub: usize,
) -> usize {
    let mut distances = vec![0.0; ksub];

    match dsub {
        1 => unsafe { compute_distances_d1(&mut distances, query_vec, centroids, ksub) },
        12 => unsafe { compute_distances_d12(&mut distances, query_vec, centroids, ksub) },
        _ => compute_distances_general(&mut distances, query_vec, centroids, dsub, ksub),
    }

    find_nearest_centroid_index(&distances)
}

/// Finds the index of the nearest centroid to a given query vector `query_vec` from a set
/// of centroids `centroids`.
///
/// # Arguments
///
/// * `query_vec` - A slice representing the query vector.
/// * `centroids` - A slice representing the set of centroids.
/// * `dsub` - The dimensionality of each segment in product quantization.
/// * `ksub` - The number of centroids.
///
/// # Returns
///
/// Returns the index of the nearest centroid to the `query_vec` within the set `centroids`.
///
#[cfg(not(target_feature = "avx2"))]
fn find_nearest_centroid_general(
    query_vec: &[f32],
    centroids: &[f32],
    dsub: usize,
    ksub: usize,
) -> usize {
    let mut distances = vec![0.0; ksub];

    compute_distances_general(&mut distances, query_vec, centroids, dsub, ksub);

    find_nearest_centroid_index(&distances)
}

/// Finds the nearest centroid to a given query subvector `query_sub` using AVX2-optimized SIMD operations.
/// Optimized for high-performance environments supporting AVX2, this function efficiently processes
/// centroids in groups, leveraging the parallelism capabilities of AVX2 instruction sets.
///
/// # Arguments
///
/// * `query_sub` - A slice representing a segment of the query vector.
/// * `centroids_sub` - A slice representing the corresponding segment of the centroids.
/// * `dsub` - The dimension of each segment, indicating the size of `query_sub` and `centroids_sub`.
/// * `ksub` - The number of centroid subvectors to compare against.
///
/// # Returns
///
/// The index of the nearest centroid to the input query subvector within the set of centroid subvectors.
///
#[cfg(target_feature = "avx2")]
pub fn find_nearest_centroid_idx(
    query_sub: &[f32],
    centroids_sub: &[f32],
    dsub: usize,
    ksub: usize,
) -> usize {
    match dsub {
        // 2 => unsafe { find_nearest_centroid_avx2_d2(query_sub, centroids_sub, ksub) },
        4 => unsafe { find_nearest_centroid_avx2_d4(query_sub, centroids_sub, ksub) },
        8 => unsafe { find_nearest_centroid_avx2_d8(query_sub, centroids_sub, ksub) },
        _ => find_nearest_centroid_general(query_sub, centroids_sub, dsub, ksub),
    }
}

/// Identifies the nearest centroid to a given query subvector `query_sub` in environments lacking AVX2 support.
/// This function employs standard processing techniques to determine the nearest centroid, ensuring compatibility
/// across various hardware configurations.
///
/// # Arguments
///
/// * `query_sub` - A slice representing a segment of the query vector.
/// * `centroids_sub` - A slice representing the corresponding segment of the centroids.
/// * `dsub` - The dimension of each segment.
/// * `ksub` - The number of centroid subvectors in the comparison.
///
/// # Returns
///
/// The index of the nearest centroid to the input query subvector within the set of centroid subvectors.
///
#[cfg(not(target_feature = "avx2"))]
pub fn find_nearest_centroid_idx(
    query_sub: &[f32],
    centroids_sub: &[f32],
    dsub: usize,
    ksub: usize,
) -> usize {
    find_nearest_centroid_general(query_sub, centroids_sub, dsub, ksub)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOAT_TOLERANCE: f32 = 0.0001;

    /// Helper function to create a sample query vector
    fn sample_query_vec(dsub: usize) -> Vec<f32> {
        (0..dsub).map(|i| i as f32).collect()
    }

    /// Helper function to create a set of sample centroids
    fn sample_centroids(ksub: usize, dsub: usize) -> Vec<f32> {
        (0..ksub * dsub).map(|i| i as f32).collect()
    }

    /// Tests the `find_nearest_centroid_avx2_d4` function for correctness.
    ///
    /// This test verifies that the function correctly identifies the nearest centroid
    /// to a given query vector from a set of centroids. It uses a sample query vector
    /// of dimension 4 and 10 sample centroids, each of dimension 4.
    ///
    /// The function should return the index of the nearest centroid. In this test,
    /// the first centroid is expected to be the nearest.
    ///
    /// Assertions:
    /// - The returned index of the nearest centroid matches the expected index.
    #[test]
    fn test_find_nearest_centroid_avx2_d4() {
        let query_vec = sample_query_vec(4);
        let centroids = sample_centroids(10, 4);

        let expected_index = 0;

        unsafe {
            let nearest_index =
                find_nearest_centroid_avx2_d4(&query_vec, &centroids, centroids.len() / 4);
            assert_eq!(
                nearest_index, expected_index,
                "Nearest centroid index mismatch in avx2_d4"
            );
        }
    }

    /// Tests the `find_nearest_centroid_avx2_d8` function for accuracy.
    ///
    /// This test checks if the function accurately finds the nearest centroid
    /// to a query vector using AVX2 SIMD operations. The query vector and centroids
    /// used in the test are of dimension 8, with 10 centroids provided.
    ///
    /// The expected outcome is that the function identifies the first centroid as the nearest.
    ///
    /// Assertions:
    /// - Ensures that the index of the nearest centroid returned by the function
    ///   is equal to the expected index.
    #[test]
    fn test_find_nearest_centroid_avx2_d8() {
        let query_vec = sample_query_vec(8);
        let centroids = sample_centroids(10, 8);

        let expected_index = 0;

        unsafe {
            let nearest_index =
                find_nearest_centroid_avx2_d8(&query_vec, &centroids, centroids.len() / 8);
            assert_eq!(
                nearest_index, expected_index,
                "Nearest centroid index mismatch in avx2_d8"
            );
        }
    }

    /// Tests the `compute_distances_d1` function for single-dimensional distance calculations.
    ///
    /// This test evaluates the function's ability to compute the squared L2 distances
    /// between a single-dimensional query vector and each of a set of single-dimensional centroids.
    ///
    /// The test uses a query vector with a single element and five centroids. The expected
    /// distances are pre-calculated and used to verify the correctness of the function.
    ///
    /// Assertions:
    /// - Each calculated distance is compared against the expected distance, within a small
    ///   tolerance level, to account for floating-point precision issues.
    #[test]
    fn test_compute_distances_d1() {
        let query_vec = vec![3.0];
        let centroids = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut distances = vec![0.0; centroids.len()];

        unsafe {
            compute_distances_d1(&mut distances, &query_vec, &centroids, centroids.len());
        }

        let expected_distances = vec![4.0, 1.0, 0.0, 1.0, 4.0];

        for (i, &dist) in distances.iter().enumerate() {
            assert!(
                (dist - expected_distances[i]).abs() < FLOAT_TOLERANCE,
                "Distance mismatch at index {}",
                i
            );
        }
    }

    /// Tests the `compute_distances_general` function for general-purpose distance calculations.
    ///
    /// This test assesses the function's capability to compute distances in a general scenario
    /// without specific SIMD optimizations. It uses a 4-dimensional query vector and two
    /// 4-dimensional centroids for the test.
    ///
    /// The expected distances are determined manually and used to validate the function's output.
    ///
    /// Assertions:
    /// - Compares the calculated distances with the expected values within a defined tolerance,
    ///   ensuring accuracy of the distance computation.
    #[test]
    fn test_compute_distances_d12() {
        let query_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let centroids = vec![0.0; 12 * 3];
        let mut distances = vec![0.0; 3];

        unsafe {
            compute_distances_d12(&mut distances, &query_vec, &centroids, 3);
        }

        let expected_distances = vec![506.0, 506.0, 506.0];

        for (i, &dist) in distances.iter().enumerate() {
            assert!(
                (dist - expected_distances[i]).abs() < FLOAT_TOLERANCE,
                "Distance mismatch at index {}",
                i
            );
        }
    }

    /// Tests the `compute_distances_general` function for general-purpose distance calculations.
    ///
    /// This test assesses the function's capability to compute distances in a general scenario
    /// without specific SIMD optimizations. It uses a 4-dimensional query vector and two
    /// 4-dimensional centroids for the test.
    ///
    /// The expected distances are determined manually and used to validate the function's output.
    ///
    /// Assertions:
    /// - Compares the calculated distances with the expected values within a defined tolerance,
    ///   ensuring accuracy of the distance computation.
    #[test]
    fn test_compute_distances_general() {
        let query_vec = vec![0.0, 1.0, 2.0, 3.0];
        let centroids = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let ksub = query_vec.len();
        let n_centroids = centroids.len() / ksub;
        let mut distances = vec![0.0; n_centroids];

        compute_distances_general(&mut distances, &query_vec, &centroids, ksub, n_centroids);

        let expected_distances = vec![0.0, 64.0];

        for (i, &dist) in distances.iter().enumerate() {
            assert!(
                (dist - expected_distances[i]).abs() < FLOAT_TOLERANCE,
                "Distance mismatch at index {}",
                i
            );
        }
    }
}
