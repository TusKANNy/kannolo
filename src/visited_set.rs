use rustc_hash::FxHashSet;

/// Estimates the number of distinct nodes a search is likely to visit, used to
/// size the visited set's initial capacity.
///
/// With `lambda == 0.0` (standard HNSW, no early-termination relaxation), the
/// number of visited nodes tracks `ef` closely, so a small multiple suffices.
///
/// With `lambda > 0.0` (distance-adaptive early termination), the relaxation
/// allows the search to keep admitting candidates well past `ef` - in practice
/// the number of visited nodes can reach several times `ef` depending on
/// `lambda`. The `lambda * 10_000` term accounts for this, keeping the visited
/// set's load factor close to 1 (and thus avoiding resizes) for typical lambda
/// values used by the adaptive-best configs.
fn estimate_visited_capacity(ef: usize, lambda: f32) -> usize {
    if lambda == 0.0 {
        16 * ef + 200
    } else {
        200 * ef + (lambda * 10_000.0) as usize
    }
}

/// Creates a visited set sized for the given search parameters.
///
/// `lambda` is the relaxation parameter of the distance-adaptive early
/// termination strategy (`0.0` if not used), and affects the initial capacity
/// estimate via [`estimate_visited_capacity`].
pub fn create_visited_set(ef: usize, lambda: f32) -> FxHashSet<usize> {
    FxHashSet::with_capacity_and_hasher(estimate_visited_capacity(ef, lambda), Default::default())
}
