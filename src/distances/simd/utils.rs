#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Computes the squared L2 distance (element-wise) between two 128-bit SIMD vectors.
///
/// This function performs an element-wise subtraction between two vectors `x` and `y`,
/// followed by squaring each element of the result. It's primarily used in SIMD-based
/// distance calculations for small vectors.
///
/// # Arguments
///
/// * `x` - A 128-bit SIMD vector.
/// * `y` - Another 128-bit SIMD vector.
///
/// # Safety
///
/// This function is unsafe as it uses SIMD intrinsics.
///
/// # Returns
///
/// A 128-bit SIMD vector where each element is the squared difference of corresponding
/// elements in `x` and `y`.
///
/// # Example
///
/// ```
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
/// use core::simd::utils::squared_l2_dist_128;
///
/// #[cfg(target_arch = "x86_64")]
/// unsafe {
///     let v1 = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
///     let v2 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
///     let result = squared_l2_dist_128(v1, v2);
///     let expected = [9.0, 1.0, 1.0, 9.0];
///     let result_array: [f32; 4] = std::mem::transmute(result);
///     assert_eq!(result_array, expected);
/// }
/// ```
#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn squared_l2_dist_128(x: __m128, y: __m128) -> __m128 {
    let diff = _mm_sub_ps(x, y);
    _mm_mul_ps(diff, diff)
}

/// Computes the squared L2 distance (element-wise) between two 256-bit SIMD vectors.
///
/// Similar to `squared_l2_dist_128`, but operates on 256-bit vectors. This function is
/// useful for SIMD-based distance calculations in higher-dimensional spaces.
///
/// # Arguments
///
/// * `x` - A 256-bit SIMD vector.
/// * `y` - Another 256-bit SIMD vector.
///
/// # Safety
///
/// This function is unsafe as it relies on AVX2 SIMD intrinsics.
///
/// # Returns
///
/// A 256-bit SIMD vector containing the squared differences of corresponding elements in `x` and `y`.
///
/// # Example
///
/// ```
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
/// use crate::simd::utils::squared_l2_dist_256;
///
/// #[cfg(target_arch = "x86_64")]
/// unsafe {
///     let v1 = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
///     let v2 = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
///     let result = squared_l2_dist_256(v1, v2);
///     let expected = [49.0, 25.0, 9.0, 1.0, 1.0, 9.0, 25.0, 49.0];
///     let result_array: [f32; 8] = std::mem::transmute(result);
///     assert_eq!(result_array, expected);
/// }
/// ```
#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn squared_l2_dist_256(x: __m256, y: __m256) -> __m256 {
    let diff = _mm256_sub_ps(x, y);
    _mm256_mul_ps(diff, diff)
}

/// Computes the horizontal sum of a 128-bit SIMD vector (`__m128`). This function
/// adds up all the elements in a SIMD vector and returns the sum.
///
/// # Safety
///
/// This function is unsafe as it uses low-level SIMD intrinsics that require
/// careful handling to avoid undefined behavior.
///
/// # Arguments
///
/// * `v` - A `__m128` SIMD vector containing four 32-bit floating-point elements.
///
/// # Returns
///
/// The sum of all elements in the `__m128` SIMD vector.
///
/// # Detailed Workflow
///
/// The function performs the following steps to compute the horizontal sum:
///
/// 1. **Shuffling and Adding**:
///    - The `shuffled` variable is created by shuffling `v` to swap its high and low parts and
///      then adding it to `v`. This operation partially sums adjacent elements.
///
/// 2. **Final Accumulation**:
///    - A second shuffle and add operation is performed to accumulate the
///      partial sums into a single sum.
///    - `shuffled2` is created by shuffling `sum1` to bring the last element to the front.
///    - `sum2` is the final addition that sums up all elements.
///
/// 3. **Result Extraction**:
///    - The sum is extracted from the lowest element of `sum2` using `_mm_cvtss_f32`.
///
/// # Example
///
/// ```
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
/// use crate::simd::utils::horizontal_sum_128;
///
/// #[cfg(target_arch = "x86_64")]
/// unsafe {
///     let v = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
///     let sum = horizontal_sum_128(v);
///     assert_eq!(sum, 10.0);
/// }
/// ```
#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn horizontal_sum_128(v: __m128) -> f32 {
    let shuffled = _mm_shuffle_ps(v, v, 0b00_00_11_10);
    let sum1 = _mm_add_ps(v, shuffled);
    let shuffled2 = _mm_shuffle_ps(sum1, sum1, 0b00_00_00_01);
    let sum2 = _mm_add_ps(sum1, shuffled2);
    _mm_cvtss_f32(sum2)
}

/// Computes the horizontal sum of a 256-bit SIMD vector (`__m256`). This function
/// adds up all the elements in a SIMD vector and returns the sum. It internally
/// leverages the `horizontal_sum_128` function for a 128-bit SIMD vector.
///
/// # Safety
///
/// This function is unsafe as it uses low-level SIMD intrinsics that require
/// careful handling to avoid undefined behavior.
///
/// # Arguments
///
/// * `v` - A `__m256` SIMD vector containing eight 32-bit floating-point elements.
///
/// # Returns
///
/// The sum of all elements in the `__m256` SIMD vector.
///
/// # Detailed Workflow
///
/// The function performs the following steps to compute the horizontal sum:
///
/// 1. **Split and Add**:
///    - The 256-bit vector is split into two 128-bit halves.
///    - These halves are added together to form a single 128-bit vector (`v0`).
///
/// 2. **Utilize `horizontal_sum_128`**:
///    - The function then calls `horizontal_sum_128` on `v0` to compute the
///      sum of its elements.
///
/// # Example
///
/// ```
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
/// use crate::simd::utils::horizontal_sum_256;
///
/// #[cfg(target_arch = "x86_64")]
/// unsafe {
///     let v = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
///     let sum = horizontal_sum_256(v);
///     assert_eq!(sum, 36.0);
/// }
/// ```
#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn horizontal_sum_256(v: __m256) -> f32 {
    let low_high_sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    horizontal_sum_128(low_high_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Tests the `squared_l2_dist_128` function for computing squared L2 distance between 128-bit SIMD vectors.
    ///
    /// This test checks if the function correctly computes the element-wise squared difference between two vectors.
    ///
    /// Safety:
    /// - This test uses unsafe blocks due to the use of SIMD intrinsics.
    ///
    /// Expected behavior:
    /// - The result should match the manually calculated squared differences of each element.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_squared_l2_dist_128() {
        unsafe {
            let v1 = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
            let v2 = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
            let result = squared_l2_dist_128(v1, v2);
            let expected = [9.0, 1.0, 1.0, 9.0];
            let result_array: [f32; 4] = std::mem::transmute(result);
            assert_eq!(result_array, expected);
        }
    }

    /// Tests the `squared_l2_dist_256` function for computing squared L2 distance between 256-bit SIMD vectors.
    ///
    /// This test checks if the function correctly computes the element-wise squared difference between two vectors.
    ///
    /// Safety:
    /// - This test uses unsafe blocks due to the use of SIMD intrinsics.
    ///
    /// Expected behavior:
    /// - The result should match the manually calculated squared differences of each element.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_squared_l2_dist_256() {
        unsafe {
            let v1 = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let v2 = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
            let result = squared_l2_dist_256(v1, v2);
            let expected = [49.0, 25.0, 9.0, 1.0, 1.0, 9.0, 25.0, 49.0];
            let result_array: [f32; 8] = std::mem::transmute(result);
            assert_eq!(result_array, expected);
        }
    }

    /// Tests the `horizontal_sum_128` function for computing the sum of elements in a 128-bit SIMD vector.
    ///
    /// This test verifies that the function correctly computes the sum of four 32-bit floating-point elements.
    ///
    /// Safety:
    /// - This test uses unsafe blocks due to the use of SIMD intrinsics.
    ///
    /// Expected behavior:
    /// - The computed sum should match the sum of the elements in the input vector.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_horizontal_sum_128() {
        unsafe {
            let v = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
            let sum = horizontal_sum_128(v);
            assert_eq!(sum, 10.0);
        }
    }

    /// Tests the `horizontal_sum_256` function for computing the sum of elements in a 256-bit SIMD vector.
    ///
    /// This test verifies that the function correctly computes the sum of eight 32-bit floating-point elements.
    ///
    /// Safety:
    /// - This test uses unsafe blocks due to the use of SIMD intrinsics.
    ///
    /// Expected behavior:
    /// - The computed sum should match the sum of the elements in the input vector.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_horizontal_sum_256() {
        unsafe {
            let v = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let sum = horizontal_sum_256(v);
            assert_eq!(sum, 36.0);
        }
    }
}
