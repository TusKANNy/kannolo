use rustc_hash::FxHashSet;

/// Slope on `ef_search` in the capacity formula (both regimes).
/// Empirically determined.
const CAPACITY_EF_SLOPE: f64 = 1.2;

/// Slope on `lambda` in the adaptive early-termination capacity formula.
/// Empirically determined.
const CAPACITY_LAMBDA_SLOPE: f64 = 800.0;

/// Estimates the number of distinct nodes a search is likely to visit, used to
/// pre-size the `FxHashSet` visited set and limit mid-search rehashes.
fn estimate_visited_capacity(ef: usize, lambda: f32, dataset_size: usize) -> usize {
    let log2n = (dataset_size.max(2) as f64).log2();
    if lambda == 0.0 {
        (CAPACITY_EF_SLOPE * log2n * ef as f64) as usize + 200
    } else {
        (log2n * (CAPACITY_EF_SLOPE * ef as f64 + CAPACITY_LAMBDA_SLOPE * lambda as f64)) as usize
    }
}

/// Creates a visited set sized for the given search parameters.
///
/// `lambda` is the relaxation parameter of the distance-adaptive early
/// termination strategy (`0.0` if not used). `dataset_size` is the number of
/// vectors in the index, used to scale capacity with `log₂(dataset_size)`.
pub fn create_visited_set(ef: usize, lambda: f32, dataset_size: usize) -> FxHashSet<usize> {
    FxHashSet::with_capacity_and_hasher(
        estimate_visited_capacity(ef, lambda, dataset_size),
        Default::default(),
    )
}
