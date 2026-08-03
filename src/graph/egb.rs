use std::time::Instant;

use rgb::forward::Doc;
use rgb::recursive_graph_bisection;

use crate::graph::graph::GraphTrait;
use rayon::prelude::*;

// EGB parameters, passed straight to `recursive_graph_bisection`.
const ITERATIONS: usize = 10; // Refinement iterations per bipartition
const MIN_PARTITION_SIZE: usize = 64; // Stop recursing once a partition is this small
const MAX_DEPTH: usize = 100; // Stop recursing past this depth
const PARALLEL_SWITCH: usize = 100; // Recurse in parallel up to this depth, sequentially beyond
const INITIAL_DEPTH: usize = 1; // Starting recursion depth
const SORT_LEAF: bool = true; // Sort each leaf partition by original id
const ID: usize = 1; // Seed of the recursion subtree identifier (rgb uses 2*id, 2*id+1)

/// Computes a node reordering permutation (`old_id -> new_id`) for a single-level graph
/// using Enhanced Graph Bisection (EGB).
///
/// Each node's adjacency list is treated as a "document" whose "terms" are its neighbor
/// IDs. Recursive bipartitioning groups nodes with overlapping neighborhoods close
/// together in the new ID space, which shrinks the deltas between consecutive neighbor
/// IDs after reordering — improving delta/StreamVByte compression of the graph.
pub fn compute_permutation<G>(graph: &G) -> Vec<usize>
where
    G: GraphTrait + Sync,
{
    let n = graph.n_nodes();
    let total_start = Instant::now();

    // 1) Collect used neighbor IDs (as "terms"). One scratch buffer per rayon task.
    let mut all_terms: Vec<u32> = (0..n)
        .into_par_iter()
        .map_init(Vec::new, |scratch, u| graph.neighbors(u, scratch).to_vec())
        .flatten()
        .collect();

    // 2) Compact term ID space while avoiding a hash lookup per edge.
    all_terms.par_sort_unstable();
    all_terms.dedup();
    let num_terms = all_terms.len();

    let mut term_map = vec![u32::MAX; n];
    for (new_id, &old_id) in all_terms.iter().enumerate() {
        term_map[old_id as usize] = new_id as u32;
    }

    // 3) Build Docs for RGB (neighbors -> compact term IDs).
    let mut docs: Vec<Doc> = (0..n)
        .into_par_iter()
        .map_init(Vec::new, |scratch, u| {
            let mut terms: Vec<u32> = graph
                .neighbors(u, scratch)
                .iter()
                .map(|&v| term_map[v as usize])
                .collect();
            terms.sort_unstable();
            terms.dedup();

            Doc {
                terms,
                org_id: u as u32,
                gain: 0.0,
                leaf_id: -1,
            }
        })
        .collect();

    // 4) Run recursive bipartite partitioning.
    recursive_graph_bisection(
        &mut docs,
        num_terms,
        ITERATIONS,
        MIN_PARTITION_SIZE,
        MAX_DEPTH,
        PARALLEL_SWITCH,
        INITIAL_DEPTH,
        SORT_LEAF,
        ID,
    );

    // 5) Build permutation (old_id -> new_id).
    let mut perm = vec![0usize; n];
    for (new_id, doc) in docs.iter().enumerate() {
        perm[doc.org_id as usize] = new_id;
    }

    println!("[EGB] total: {:.2}s", total_start.elapsed().as_secs_f32());
    perm
}
