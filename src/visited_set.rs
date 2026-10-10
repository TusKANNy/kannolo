use rustc_hash::FxHashSet;

/// Growth of the explored set with the relaxation `lambda`. Relaxing the threshold admits every
/// candidate within a factor `1 + lambda` of the `ef`-th result, and how many there are depends on
/// how fast distances grow past it, which is data-dependent. `(1 + 0.05 + lambda)^8` is a round
/// heuristic for that growth assuming exponential score decay.
const CAPACITY_LAMBDA_EXPONENT: i32 = 8;

/// Added to `lambda` when early termination is on, so that small relaxations, where the growth
/// is underestimated most, get more room while large ones are barely affected.
const CAPACITY_LAMBDA_OFFSET: f64 = 0.05;

/// Nodes added to every estimate. A search expands a few candidates beyond `ef` before its
/// termination test fires, each bringing about `avg_degree` nodes; a constant covers them where
/// they matter, at small `ef`.
const CAPACITY_EXTRA_NODES: usize = 200;

/// Estimates the number of distinct nodes a search is likely to visit, used to
/// pre-size the `FxHashSet` visited set and limit mid-search rehashes.
///
/// A search expands about `ef` candidates and scores each one's unvisited neighbours, so the
/// count is about `avg_degree * expansions`, less the overlap between neighbour lists.
///
/// The estimate is capped at `n_nodes`, the most nodes a search can visit.
fn estimate_visited_capacity(ef: usize, lambda: f32, avg_degree: usize, n_nodes: usize) -> usize {
    let growth = if lambda > 0.0 {
        (1.0 + CAPACITY_LAMBDA_OFFSET + lambda as f64).powi(CAPACITY_LAMBDA_EXPONENT)
    } else {
        1.0
    };
    let expansions = ef as f64 * growth;
    let estimate = avg_degree.max(1) as f64 * expansions + CAPACITY_EXTRA_NODES as f64;
    estimate.min(n_nodes as f64) as usize
}

/// Creates a visited set sized for the given search parameters.
///
/// `lambda` is the relaxation parameter of the distance-adaptive early
/// termination strategy (`0.0` if not used). `avg_degree` is the graph's average
/// out-degree and `n_nodes` its number of nodes.
pub fn create_visited_set(
    ef: usize,
    lambda: f32,
    avg_degree: usize,
    n_nodes: usize,
) -> FxHashSet<usize> {
    FxHashSet::with_capacity_and_hasher(
        estimate_visited_capacity(ef, lambda, avg_degree, n_nodes),
        Default::default(),
    )
}

#[cfg(test)]
mod tests {
    use super::estimate_visited_capacity;

    #[test]
    fn capacity_without_relaxation_is_degree_times_ef_plus_extra() {
        assert_eq!(
            estimate_visited_capacity(100, 0.0, 32, 1_000_000),
            32 * 100 + 200
        );
    }

    #[test]
    fn capacity_is_capped_at_the_number_of_nodes() {
        assert_eq!(estimate_visited_capacity(100, 0.0, 32, 1_000), 1_000);
        // A large relaxation would otherwise ask for more slots than fit in memory.
        assert_eq!(
            estimate_visited_capacity(1_000, 5.0, 64, 2_000_000),
            2_000_000
        );
        assert_eq!(
            estimate_visited_capacity(1_000, 1e6, 64, 2_000_000),
            2_000_000
        );
    }
}
