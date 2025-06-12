#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn transpose_8x2(i0: __m256, i1: __m256, o0: &mut __m256, o1: &mut __m256) {
    let r0 = _mm256_permute2f128_ps(i0, i1, 0b0010_0000);
    let r1 = _mm256_permute2f128_ps(i0, i1, 0b0011_0001);

    *o0 = _mm256_shuffle_ps(r0, r1, 0b10_00_10_00);
    *o1 = _mm256_shuffle_ps(r0, r1, 0b11_01_11_01);
}

/// Transposes a 8x4 matrix represented across four `__m256` SIMD registers.
///
/// Each input register (`i0`, `i1`, `i2`, `i3`) contains eight 32-bit floats,
/// representing two rows of a 4-column matrix. This function transposes the
/// matrix, converting rows into columns and vice versa, and returns the result
/// as an array of four `__m256` registers, each containing a row of the
/// transposed matrix.
///
/// # Safety
/// This function is `unsafe` as it performs low-level SIMD operations which
/// require the caller to ensure correct data alignment and CPU support for AVX
/// instructions.
///
/// # Arguments
/// * `i0` - First and second row of the input matrix.
/// * `i1` - Third and fourth row of the input matrix.
/// * `i2` - Fifth and sixth row of the input matrix.
/// * `i3` - Seventh and eighth row of the input matrix.
/// # Returns
/// An array of four `__m256` registers, each representing a row of the transposed 8x4 matrix.
///
/// # Example
///
/// ```
///
/// #[cfg(target_arch = "x86_64")]
/// use std::arch::x86_64::*;
/// use crate::distances::simd::transpose::transpose_8x4;
///
/// unsafe {
///     // remember that the data in the actual registers is stored in the
///     // reversed order
///     let i0 = _mm256_set_ps( 0.0,  1.0,  2.0,  3.0, 10.0, 11.0, 12.0, 13.0);
///     let i1 = _mm256_set_ps(20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0);
///     let i2 = _mm256_set_ps(40.0, 41.0, 42.0, 43.0, 50.0, 51.0, 52.0, 53.0);
///     let i3 = _mm256_set_ps(60.0, 61.0, 62.0, 63.0, 70.0, 71.0, 72.0, 73.0);
///
///     let transposed = transpose_8x4(i0, i1, i2, i3);
///     // The `transposed` array now contains the transposed registers.
/// }
/// ```
#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn transpose_8x4(i0: __m256, i1: __m256, i2: __m256, i3: __m256) -> [__m256; 4] {
    // let's say we have the following data stored in the registers:
    // i0:  00 01 02 03 10 11 12 13
    // i1:  20 21 22 23 30 31 32 33
    // i2:  40 41 42 43 50 51 52 53
    // i3:  60 61 62 63 70 71 72 73

    // 00 01 02 03 40 41 42 43
    let r0 = _mm256_permute2f128_ps(i0, i2, _MM_SHUFFLE(0, 2, 0, 0));
    // 20 21 22 23 60 61 62 63
    let r1 = _mm256_permute2f128_ps(i1, i3, _MM_SHUFFLE(0, 2, 0, 0));
    // 10 11 12 13 50 51 52 53
    let r2 = _mm256_permute2f128_ps(i0, i2, _MM_SHUFFLE(0, 3, 0, 1));
    // 30 31 32 33 70 71 72 73
    let r3 = _mm256_permute2f128_ps(i1, i3, _MM_SHUFFLE(0, 3, 0, 1));

    // 00 02 10 12 40 42 50 52
    let t0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(2, 0, 2, 0));
    // 01 03 11 13 41 43 51 53
    let t1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 1, 3, 1));
    // 20 22 30 32 60 62 70 72
    let t2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(2, 0, 2, 0));
    // 21 23 31 33 61 63 71 73
    let t3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 1, 3, 1));

    // 00 10 20 30 40 50 60 70
    let o0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
    // 01 11 21 31 41 51 61 71
    let o1 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(2, 0, 2, 0));
    // 02 12 22 32 42 52 62 72
    let o2 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 1, 3, 1));
    // 03 13 23 33 43 53 63 73
    let o3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 1, 3, 1));

    [o0, o1, o2, o3]
}

/// Transposes an 8x8 matrix stored in eight __m256 registers.
///
/// This function utilizes AVX2 instructions to transpose an 8x8 matrix efficiently.
/// The matrix is assumed to be stored across eight __m256 registers, each
/// holding one row. The output is a transposed matrix with rows and columns
/// interchanged, stored in eight __m256 registers.
///
/// # Arguments
///
/// * `i0` to `i7` - Rows of the input matrix, each represented by a __m256 register.
///
/// # Safety
///
/// This function is unsafe because it uses AVX2 instructions.
///
/// # Returns
///
/// Returns an array of eight __m256 registers containing the transposed 8x8 matrix.
///
/// # Detailed Workflow
///
/// 1. **Unpacking Rows**: The function starts by unpacking each pair of rows.
///    Using `_mm256_unpacklo_ps` and `_mm256_unpackhi_ps`, it interleaves
///    adjacent elements from two rows (`i0` with `i1`, `i2` with `i3`, etc.).
///    This results in eight new registers (`r0` to `r7`), each holding
///    interleaved elements from two rows.
///
/// 2. **Shuffling Pairs**: Next, the function shuffles elements within these
///    pairs using `_mm256_shuffle_ps`. This is done twice for each pair,
///    resulting in eight shuffled registers (`rr0` to `rr7`).
///
/// 3. **Final Permutation**: Finally, `_mm256_permute2f128_ps` is used to
///    permute 128-bit halves from the shuffled registers, completing the
///    transpose. This results in the final transposed registers (`o0` to `o7`).
///
/// # Example
///
/// ```
/// use crate::distances::simd::transpose::transpose_8x8;
///
/// #[cfg(target_arch = "x86_64")]
/// #[cfg(target_feature = "avx2")]
/// {
///     use std::arch::x86_64::*;
///     unsafe {
///         // remember that the data in the actual registers is stored in the
///         // reversed order
///         let i0 = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
///         let i1 = _mm256_set_ps(17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0);
///         let i2 = _mm256_set_ps(27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0);
///         let i3 = _mm256_set_ps(37.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0);
///         let i4 = _mm256_set_ps(47.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0, 40.0);
///         let i5 = _mm256_set_ps(57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0);
///         let i6 = _mm256_set_ps(67.0, 66.0, 65.0, 64.0, 63.0, 62.0, 61.0, 60.0);
///         let i7 = _mm256_set_ps(77.0, 76.0, 75.0, 74.0, 73.0, 72.0, 71.0, 70.0);
///
///         let transposed = transpose_8x8(i0, i1, i2, i3, i4, i5, i6, i7);
///     }
/// }
/// ```
#[inline]
#[cfg(target_arch = "x86_64")]
pub unsafe fn transpose_8x8(
    i0: __m256,
    i1: __m256,
    i2: __m256,
    i3: __m256,
    i4: __m256,
    i5: __m256,
    i6: __m256,
    i7: __m256,
) -> [__m256; 8] {
    // let's say we have the following data stored in the registers:
    // i0:  00 01 02 03 04 05 06 07
    // i1:  10 11 12 13 14 15 16 17
    // i2:  20 21 22 23 24 25 26 27
    // i3:  30 31 32 33 34 35 36 37
    // i4:  40 41 42 43 44 45 46 47
    // i5:  50 51 52 53 54 55 56 57
    // i6:  60 61 62 63 64 65 66 67
    // i7:  70 71 72 73 74 75 76 77

    // 00 10 01 11 04 14 05 15
    let r0 = _mm256_unpacklo_ps(i0, i1);
    // 02 12 03 13 06 16 07 17
    let r1 = _mm256_unpackhi_ps(i0, i1);
    // 20 30 21 31 24 34 25 35
    let r2 = _mm256_unpacklo_ps(i2, i3);
    // 22 32 23 33 26 36 27 37
    let r3 = _mm256_unpackhi_ps(i2, i3);
    // 40 50 41 51 44 54 45 55
    let r4 = _mm256_unpacklo_ps(i4, i5);
    // 42 52 43 53 46 56 47 57
    let r5 = _mm256_unpackhi_ps(i4, i5);
    // 60 70 61 71 64 74 65 75
    let r6 = _mm256_unpacklo_ps(i6, i7);
    // 62 72 63 73 66 76 67 77
    let r7 = _mm256_unpackhi_ps(i6, i7);

    // 00 10 20 30 04 14 24 34
    let rr0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(1, 0, 1, 0));
    // 01 11 21 31 05 15 25 35
    let rr1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 2, 3, 2));
    // 02 12 22 32 06 16 26 36
    let rr2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(1, 0, 1, 0));
    // 03 13 23 33 07 17 27 37
    let rr3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 2, 3, 2));
    // 40 50 60 70 44 54 64 74
    let rr4 = _mm256_shuffle_ps(r4, r6, _MM_SHUFFLE(1, 0, 1, 0));
    // 41 51 61 71 45 55 65 75
    let rr5 = _mm256_shuffle_ps(r4, r6, _MM_SHUFFLE(3, 2, 3, 2));
    // 42 52 62 72 46 56 66 76
    let rr6 = _mm256_shuffle_ps(r5, r7, _MM_SHUFFLE(1, 0, 1, 0));
    // 43 53 63 73 47 57 67 77
    let rr7 = _mm256_shuffle_ps(r5, r7, _MM_SHUFFLE(3, 2, 3, 2));

    // 00 10 20 30 40 50 60 70
    let o0 = _mm256_permute2f128_ps(rr0, rr4, 0x20);
    // 01 11 21 31 41 51 61 71
    let o1 = _mm256_permute2f128_ps(rr1, rr5, 0x20);
    // 02 12 22 32 42 52 62 72
    let o2 = _mm256_permute2f128_ps(rr2, rr6, 0x20);
    // 03 13 23 33 43 53 63 73
    let o3 = _mm256_permute2f128_ps(rr3, rr7, 0x20);
    // 04 14 24 34 44 54 64 74
    let o4 = _mm256_permute2f128_ps(rr0, rr4, 0x31);
    // 05 15 25 35 45 55 65 75
    let o5 = _mm256_permute2f128_ps(rr1, rr5, 0x31);
    // 06 16 26 36 46 56 66 76
    let o6 = _mm256_permute2f128_ps(rr2, rr6, 0x31);
    // 07 17 27 37 47 57 67 77
    let o7 = _mm256_permute2f128_ps(rr3, rr7, 0x31);

    [o0, o1, o2, o3, o4, o5, o6, o7]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::transmute;

    /// Tests the `transpose_8x4` function.
    ///
    /// This test verifies the correctness of the `transpose_8x4` function by
    /// comparing its output against a precomputed expected result. The test
    /// initializes four `__m256` SIMD registers with specific values, representing
    /// two rows of a 4-column matrix each, and then applies the `transpose_8x4`
    /// function to these registers.
    ///
    /// The expected result is an array of four `__m256` registers, where each register
    /// contains a row of the transposed matrix. The test asserts that each element
    /// in the transposed output matches the corresponding element in the expected
    /// result.
    ///
    /// # Steps
    /// 1. Initialize four `__m256` registers (`i0`, `i1`, `i2`, `i3`) with test data.
    /// 2. Apply the `transpose_8x4` function to these registers.
    /// 3. Define the expected outcome in an array of four `__m256` registers.
    /// 4. Compare each register in the transposed output with the corresponding
    ///    register in the expected array using `assert_eq!`, ensuring that every
    ///    element matches.
    ///
    /// # Expected Behavior
    /// The `transpose_8x4` function should accurately transpose the matrix represented
    /// by the input registers. Each row in the output should match the corresponding
    /// row in the expected result, ensuring that the function correctly transposes
    /// rows and columns.
    ///
    /// # Panics
    /// The test will panic if any row in the transposed output does not match the
    /// corresponding row in the expected result, indicating a failure in the
    /// transposition logic of the `transpose_8x4` function.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_transpose_8x4() {
        unsafe {
            let i0 = _mm256_set_ps(13.0, 12.0, 11.0, 10.0, 3.0, 2.0, 1.0, 0.0);
            let i1 = _mm256_set_ps(33.0, 32.0, 31.0, 30.0, 23.0, 22.0, 21.0, 20.0);
            let i2 = _mm256_set_ps(53.0, 52.0, 51.0, 50.0, 43.0, 42.0, 41.0, 40.0);
            let i3 = _mm256_set_ps(73.0, 72.0, 71.0, 70.0, 63.0, 62.0, 61.0, 60.0);

            let transposed = transpose_8x4(i0, i1, i2, i3);

            let expected = [
                _mm256_set_ps(70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0),
                _mm256_set_ps(71.0, 61.0, 51.0, 41.0, 31.0, 21.0, 11.0, 1.0),
                _mm256_set_ps(72.0, 62.0, 52.0, 42.0, 32.0, 22.0, 12.0, 2.0),
                _mm256_set_ps(73.0, 63.0, 53.0, 43.0, 33.0, 23.0, 13.0, 3.0),
            ];

            for (i, &val) in transposed.iter().enumerate() {
                let transposed_row: [f32; 8] = transmute(val);
                let expected_row: [f32; 8] = transmute(expected[i]);
                assert_eq!(transposed_row, expected_row, "Row {} mismatch", i);
            }
        }
    }

    /// Tests the `transpose_8x8` function.
    ///
    /// This test verifies the correctness of the `transpose_8x8` function by
    /// comparing its output against a precomputed expected result. The test
    /// initializes eight `__m256` SIMD registers with specific values, each
    /// representing a row of an 8-column matrix, and then applies the
    /// `transpose_8x8` function to these registers.
    ///
    /// The expected result is an array of eight `__m256` registers, where each register
    /// contains a row of the transposed matrix. The test asserts that each element
    /// in the transposed output matches the corresponding element in the expected
    /// result.
    ///
    /// # Steps
    /// 1. Initialize eight `__m256` registers (`i0` through `i7`) with test data.
    /// 2. Apply the `transpose_8x8` function to these registers.
    /// 3. Define the expected outcome in an array of eight `__m256` registers.
    /// 4. Compare each register in the transposed output with the corresponding
    ///    register in the expected array using `assert_eq!`, ensuring that every
    ///    element matches.
    ///
    /// # Expected Behavior
    /// The `transpose_8x8` function should accurately transpose the matrix represented
    /// by the input registers. Each row in the output should match the corresponding
    /// row in the expected result, ensuring that the function correctly transposes
    /// rows and columns.
    ///
    /// # Panics
    /// The test will panic if any row in the transposed output does not match the
    /// corresponding row in the expected result, indicating a failure in the
    /// transposition logic of the `transpose_8x8` function.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_transpose_8x8() {
        unsafe {
            let i0 = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
            let i1 = _mm256_set_ps(17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0);
            let i2 = _mm256_set_ps(27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0);
            let i3 = _mm256_set_ps(37.0, 36.0, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0);
            let i4 = _mm256_set_ps(47.0, 46.0, 45.0, 44.0, 43.0, 42.0, 41.0, 40.0);
            let i5 = _mm256_set_ps(57.0, 56.0, 55.0, 54.0, 53.0, 52.0, 51.0, 50.0);
            let i6 = _mm256_set_ps(67.0, 66.0, 65.0, 64.0, 63.0, 62.0, 61.0, 60.0);
            let i7 = _mm256_set_ps(77.0, 76.0, 75.0, 74.0, 73.0, 72.0, 71.0, 70.0);

            let transposed = transpose_8x8(i0, i1, i2, i3, i4, i5, i6, i7);

            let expected = [
                _mm256_set_ps(70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0),
                _mm256_set_ps(71.0, 61.0, 51.0, 41.0, 31.0, 21.0, 11.0, 1.0),
                _mm256_set_ps(72.0, 62.0, 52.0, 42.0, 32.0, 22.0, 12.0, 2.0),
                _mm256_set_ps(73.0, 63.0, 53.0, 43.0, 33.0, 23.0, 13.0, 3.0),
                _mm256_set_ps(74.0, 64.0, 54.0, 44.0, 34.0, 24.0, 14.0, 4.0),
                _mm256_set_ps(75.0, 65.0, 55.0, 45.0, 35.0, 25.0, 15.0, 5.0),
                _mm256_set_ps(76.0, 66.0, 56.0, 46.0, 36.0, 26.0, 16.0, 6.0),
                _mm256_set_ps(77.0, 67.0, 57.0, 47.0, 37.0, 27.0, 17.0, 7.0),
            ];

            for (i, &val) in transposed.iter().enumerate() {
                let transposed_row: [f32; 8] = transmute(val);
                let expected_row: [f32; 8] = transmute(expected[i]);
                assert_eq!(transposed_row, expected_row, "Row {} mismatch", i);
            }
        }
    }
}
