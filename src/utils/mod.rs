use crate::dot_product_simd;
use crate::indexes::hnsw_utils::Candidate;
use core::hash::Hash;
use csv::WriterBuilder;
use rand::thread_rng;
use rand::Rng;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;

#[cfg(target_arch = "aarch64")]
use half::f16;

#[cfg(use_cblas)]
use cblas_sys::{self, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

#[cfg(use_cblas)]
use intel_mkl_sys;

#[cfg(not(use_cblas))]
extern crate matrixmultiply;

#[derive(PartialEq)]
pub enum MatrixLayout {
    RowMajor,
    ColMajor,
}

/// Performs single-precision general matrix multiplication.
///
/// This function computes the product of matrices `a` and `b`, then scales the
/// result by `alpha` and adds the result to the matrix `c` scaled by `beta`.
/// The computation is performed using either the CBLAS library or a
/// pure-Rust implementation, based on the availability of the CBLAS library at compile time.
///
/// The operation performed is: `c = alpha * a * b + beta * c`.
///
/// # Arguments
/// - `layout`: Specifies the layout of matrices (RowMajor or ColMajor).
/// - `_transpose_a`: Boolean flag to indicate if matrix `a` should be transposed.
/// - `_transpose_b`: Boolean flag to indicate if matrix `b` should be transposed.
/// - `alpha`: Scalar multiplier for the product of matrices `a` and `b`.
/// - `beta`: Scalar multiplier for the matrix `c`.
/// - `m`: The number of rows in matrices `a` and `c`.
/// - `k`: The number of columns in matrix `a` and rows in matrix `b`.
/// - `n`: The number of columns in matrices `b` and `c`.
/// - `a`: Pointer to the first element of matrix `a`.
/// - `lda`: Leading dimension of matrix `a`. It cannot be less than `k`.
/// - `b`: Pointer to the first element of matrix `b`.
/// - `ldb`: Leading dimension of matrix `b`. It cannot be less than `n`.
/// - `c`: Pointer to the first element of matrix `c`.
/// - `ldc`: Leading dimension of matrix `c`. It cannot be less than `n`.
///
/// # Safety
/// This function is unsafe as it involves dereferencing raw pointers. The caller
/// must ensure that the pointers `a`, `b`, and `c` are valid and that the
/// matrices they point to are properly allocated. The dimensions and leading
/// dimensions must be correctly specified to avoid out-of-bounds access.
///
/// # Panics
/// This function can panic if called with incorrect dimensions that lead to
/// out-of-bounds memory access. It can also panic if the CBLAS implementation
/// is not found when the `use_cblas` cfg flag is set.
///
/// # Examples
/// ```
/// use kannolo::utils::{sgemm, MatrixLayout};
///
/// let m = 100; // Number of rows in a and c
/// let k = 100; // Number of columns in a and rows in b
/// let n = 100; // Number of columns in b and c
///
/// let a: Vec<f32> = vec![1.0; m * k]; // Matrix a with m rows and k columns
/// let b: Vec<f32> = vec![1.0; k * n]; // Matrix b with k rows and n columns
/// let mut c: Vec<f32> = vec![0.0; m * n]; // Matrix c with m rows and n columns
///
/// // Set alpha and beta
/// let alpha = -2.0;
/// let beta = 0.0;
///
/// // Choose which matrix has to be transposed
/// let transpose_a = true;
/// let transpose_b = false;
///
/// // Set the layout
/// let layout = MatrixLayout::RowMajor;
///
/// sgemm(
///     layout,
///     transpose_a,
///     transpose_b,
///     alpha,
///     beta,
///     m,
///     k,
///     n,
///     a.as_ptr(),
///     k as isize,
///     b.as_ptr(),
///     n as isize,
///     c.as_mut_ptr(),
///     n as isize,
/// );
///
/// // c now contains the result of -2.0 * a * b + 0.0 * c
/// ```
#[inline]
#[allow(unused_variables)] // The variables `transpose_a` and `transpose_b` are not used if `use_cblas` is not set.
pub fn sgemm(
    layout: MatrixLayout,
    transpose_a: bool,
    transpose_b: bool,
    alpha: f32,
    beta: f32,
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    lda: isize,
    b: *const f32,
    ldb: isize,
    c: *mut f32,
    ldc: isize,
) {
    #[cfg(use_cblas)]
    {
        unsafe {
            let cblas_layout = match layout {
                MatrixLayout::RowMajor => CBLAS_LAYOUT::CblasRowMajor,
                MatrixLayout::ColMajor => CBLAS_LAYOUT::CblasColMajor,
            };
            let cblas_transa = if transpose_a {
                CBLAS_TRANSPOSE::CblasTrans
            } else {
                CBLAS_TRANSPOSE::CblasNoTrans
            };
            let cblas_transb = if transpose_b {
                CBLAS_TRANSPOSE::CblasTrans
            } else {
                CBLAS_TRANSPOSE::CblasNoTrans
            };

            cblas_sys::cblas_sgemm(
                cblas_layout,
                cblas_transa,
                cblas_transb,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                lda as i32,
                b,
                ldb as i32,
                beta,
                c,
                ldc as i32,
            );
        }
    }

    #[cfg(not(use_cblas))]
    {
        unsafe {
            let (rsc, csc) = match layout {
                MatrixLayout::RowMajor => (ldc, 1),
                MatrixLayout::ColMajor => (1, ldc),
            };

            matrixmultiply::sgemm(m, k, n, alpha, a, lda, 1, b, 1, ldb, beta, c, rsc, csc);
        }
    }
}

/// Compute the size of the intersection of two unsorted lists of integers.
pub fn intersection<T: Eq + Hash + Clone>(s: &[T], groundtruth: &[T]) -> usize {
    let s_set: HashSet<_> = s.iter().cloned().collect();
    let mut size = 0;
    for v in groundtruth {
        if s_set.contains(v) {
            size += 1;
        }
    }
    size
}

pub fn warm_up() {
    let m = 100;
    let k = 100;
    let n = 100;
    let mut rng = thread_rng();

    let a: Vec<f32> = (0..(m * k)).map(|_| rng.gen()).collect();
    let b: Vec<f32> = (0..(k * n)).map(|_| rng.gen()).collect();
    let mut result = vec![0.0_f32; m * n];

    let alpha = -2.0;
    let beta = 0.0;

    let transpose_a = true;
    let transpose_b = false;

    let layout = MatrixLayout::RowMajor;

    // Warm up intended for cblas
    sgemm(
        layout,
        transpose_a,
        transpose_b,
        alpha,
        beta,
        m,
        k,
        n,
        a.as_ptr(),
        k as isize,
        b.as_ptr(),
        k as isize,
        result.as_mut_ptr(),
        n as isize,
    );
}

#[inline]
pub fn vectors_norm(vectors: &[f32], d: usize) -> Vec<f32> {
    vectors
        .chunks_exact(d)
        .map(|v| dot_product_simd(v, v))
        .collect()
}

#[inline(always)]
pub fn compute_vector_norm_squared(vec: &[f32], length: usize) -> f32 {
    vec.iter().take(length).map(|&xi| xi * xi).sum()
}

#[inline]
pub fn compute_squared_l2_distance(query_vec: &[f32], centroids: &[f32], length: usize) -> f32 {
    query_vec
        .iter()
        .zip(centroids.iter())
        .take(length)
        .map(|(&qvec_element, &centroid_element)| {
            let diff = qvec_element - centroid_element; // Element-wise difference
            diff * diff // Squared difference
        })
        .sum() // Sum of all squared differences
}

#[cfg(target_arch = "aarch64")]
fn conv_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    let len = src.len();
    let chunks = len / 8;

    // process 8 at a time
    for i in 0..chunks {
        let base = i * 8;
        dst[base + 0] = src[base + 0].to_f32();
        dst[base + 1] = src[base + 1].to_f32();
        dst[base + 2] = src[base + 2].to_f32();
        dst[base + 3] = src[base + 3].to_f32();
        dst[base + 4] = src[base + 4].to_f32();
        dst[base + 5] = src[base + 5].to_f32();
        dst[base + 6] = src[base + 6].to_f32();
        dst[base + 7] = src[base + 7].to_f32();
    }

    // tail
    for i in (chunks * 8)..len {
        dst[i] = src[i].to_f32();
    }
}

pub fn save_to_tsv(
    params_ef_search: Vec<usize>,
    recalls: Vec<f64>,
    search_times: Vec<u128>,
    filename: &str,
) {
    // Open a file to write
    let file = File::create(filename).unwrap();

    // Create a TSV writer
    let mut wtr = WriterBuilder::new().delimiter(b'\t').from_writer(file);

    // Write header (optional)
    wtr.write_record(["ef_search", "Accuracy@10", "Avg_query_time"])
        .unwrap();

    // Write rows by iterating through the vectors
    for ((&param, &rec), &t) in params_ef_search.iter().zip(&recalls).zip(&search_times) {
        wtr.write_record(&[
            param.to_string(),
            ((rec * 100.0).round() / 100.0).to_string(),
            t.to_string(),
        ])
        .unwrap();
    }

    // Flush and close the writer
    wtr.flush().unwrap();
}

pub fn compute_accuracy(
    ids: Vec<usize>,
    ground_truth_values: &[u32],
    k: usize,
    gt_size: usize,
) -> f64 {
    let mut sum_recall: f64 = 0.0;
    let mut i = 0;
    let mut j = 0;

    // Compute recall
    while i < ids.len() && j < ground_truth_values.len() {
        let ids_chunk = &ids[i..i + k];
        let gt_chunk = &ground_truth_values[j..j + k];

        let intersection = ids_chunk
            .iter()
            .filter(|&&x| gt_chunk.contains(&(x as u32)))
            .count() as f64;

        let recall = intersection / k as f64;
        sum_recall += recall;

        i += k;
        j += gt_size;
    }

    100.0 * sum_recall / (ground_truth_values.len() / gt_size) as f64
}

pub fn from_max_heap_to_min_heap(
    max_heap: &mut BinaryHeap<Candidate>,
) -> BinaryHeap<Reverse<Candidate>> {
    let vec: Vec<_> = max_heap.drain().collect();
    BinaryHeap::from(vec.into_iter().map(Reverse).collect::<Vec<_>>())
}
