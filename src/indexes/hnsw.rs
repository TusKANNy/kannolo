use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::graph::{GraphTrait, GrowableGraph};
use vectorium::IndexSerializer;
use vectorium::core::dataset::{ConvertFrom, ConvertInto, ScoredItemGeneric};
use vectorium::core::index::Index;
use vectorium::distances::Distance;
use vectorium::vector_encoder::VectorEncoder;
use vectorium::{Dataset, QueryEvaluator, SpaceUsage, VectorId};

// ---------------------------------------------------------------------------
// ACORN-γ pre-expanded neighbor structure
// ---------------------------------------------------------------------------

/// Pre-expanded ground-level neighbor lists for ACORN-γ filtered ANN search.
///
/// Built from a completed HNSW index by expanding each node's neighborhood to
/// include two-hop candidates (neighbors of neighbors), scoring them by distance
/// to that node, and retaining the top γ·M closest.
///
/// At search time, standard beam search is run over these larger lists; predicate-
/// failing nodes are skipped without further expansion, because the two-hop
/// connectivity is already embedded in the pre-built lists.
///
/// Use [`HNSW::build_acorn_gamma_neighbors`] to construct this.
pub struct AcornGammaNeighbors {
    /// `neighbors[v]` holds local node IDs of v's expanded neighborhood,
    /// sorted by ascending distance to v.
    neighbors: Box<[Box<[usize]>]>,
    gamma_m: usize,
}

impl GraphTrait for AcornGammaNeighbors {
    #[inline]
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a {
        self.neighbors[u].iter().copied()
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.neighbors.len()
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.neighbors.iter().map(|n| n.len()).sum()
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.gamma_m
    }

    /// Ground-level nodes use identity mapping (local ID == global ID).
    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        id
    }

    fn get_space_usage_bytes(&self) -> usize {
        self.neighbors
            .iter()
            .map(|n| n.len() * std::mem::size_of::<usize>())
            .sum()
    }
}

/// A `HNSW` struct represents a Hierarchical Navigable Small World (HNSW) graph structure that is used
/// for approximate nearest neighbor (ANN) search.
///
/// This index is constructed from a dataset and configuration settings. It efficiently finds the k-closest
/// vectors in the graph for a given query vector.
///
/// # Type Parameters
/// * `D`: The type of the dataset (vectorium dataset).
/// * `G`: The type of the graph implementation (e.g., `Graph`, `GraphFixedDegree`).
#[derive(Serialize, Deserialize)]
pub struct HNSW<D, G> {
    /// A boxed slice containing the hierarchical levels of the HNSW graph.
    /// Each level is a graph structure. Level 0 is the highest level (most sparse),
    /// and the last level is the ground level (contains all nodes).
    levels: Box<[G]>,

    /// Maps local IDs in the first non-ground level (level 1) to the corresponding
    /// global IDs in the ground level (level 0). This is used to find an efficient
    /// entry point for the search on the ground level.
    level1_to_level0_mapping: Box<[usize]>,

    /// The dataset (dense or sparse) that the graph index is built upon.
    /// This holds the original vectors for distance calculations.
    dataset: D,
    /// The number of neighbors per vector at each level in the HNSW graph.
    /// This is the `M` parameter in the HNSW algorithm.
    num_neighbors_per_vec: usize,
    /// The global ID of the vector from which every search begins.
    /// This node is located on the highest level of the hierarchy.
    entry_point: usize,
}

/// Configuration for building the HNSW index.
/// Use the builder pattern: `HNSWBuildConfiguration::default().with_num_neighbors(32).with_ef_construction(200)`
pub struct HNSWBuildConfiguration {
    /// The number of neighbors for each node on each layer of the graph.
    /// Also known as `M` in the HNSW paper.
    pub num_neighbors_per_vec: usize,
    /// The size of the dynamic candidate list for constructing the graph.
    /// Also known as `efConstruction` in the HNSW paper.
    pub ef_construction: usize,
    /// The initial number of nodes to process in parallel during the build.
    pub initial_build_batch_size: usize,
    /// The maximum number of nodes to process in parallel during the build.
    pub max_build_batch_size: usize,
}

impl HNSWBuildConfiguration {
    /// Sets the number of neighbors per vector (M parameter). Returns self for chaining.
    #[must_use]
    pub fn with_num_neighbors(mut self, num_neighbors_per_vec: usize) -> Self {
        self.num_neighbors_per_vec = num_neighbors_per_vec;
        self
    }

    /// Sets the ef_construction parameter. Returns self for chaining.
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Sets the initial build batch size (internal tuning). Returns self for chaining.
    #[must_use]
    pub fn with_initial_batch_size(mut self, initial_build_batch_size: usize) -> Self {
        self.initial_build_batch_size = initial_build_batch_size;
        self
    }

    /// Sets the maximum build batch size (internal tuning). Returns self for chaining.
    #[must_use]
    pub fn with_max_batch_size(mut self, max_build_batch_size: usize) -> Self {
        self.max_build_batch_size = max_build_batch_size;
        self
    }
}

impl Default for HNSWBuildConfiguration {
    /// Provides a default set of build parameters.
    /// These are generally reasonable starting points, but they should be
    /// tuned for specific datasets and use cases.
    fn default() -> Self {
        Self {
            num_neighbors_per_vec: 16,   // Common default value for M
            ef_construction: 150,        // Common default value
            initial_build_batch_size: 4, // Start small for parallel batches
            max_build_batch_size: 320,   // Cap parallel batches
        }
    }
}

/// Strategy for early termination during HNSW search.
#[derive(Debug, Clone, Copy, Default)]
pub enum EarlyTerminationStrategy {
    /// Standard HNSW: stop when the best frontier candidate is worse
    /// than the worst candidate in the top-k result set.
    #[default]
    None,
    /// Distance-adaptive: allow exploration within a relaxed threshold
    /// controlled by `lambda` on the worst top candidate.
    ///
    /// Reference: "Distance Adaptive Beam Search for Provably Accurate
    /// Graph-Based Nearest Neighbor Search" (Al-Jazzazi et al.)
    DistanceAdaptive {
        /// Relaxation parameter. `lambda = 0` is equivalent to `None`.
        lambda: f32,
    },
}

impl EarlyTerminationStrategy {
    /// Returns the relaxation parameter (`0.0` for `None`).
    #[inline]
    pub fn lambda(&self) -> f32 {
        match self {
            EarlyTerminationStrategy::None => 0.0,
            EarlyTerminationStrategy::DistanceAdaptive { lambda } => *lambda,
        }
    }
}

/// Configuration for searching the HNSW index.
/// Use the builder pattern: `HNSWSearchConfiguration::default().with_ef_search(200)`
pub struct HNSWSearchConfiguration {
    /// The size of the dynamic candidate list for searching the graph.
    /// Also known as `ef` or `efSearch` in the HNSW paper. A larger
    /// value leads to more accurate results at the cost of speed.
    pub ef_search: usize,
    /// Early termination strategy for search.
    pub early_termination: EarlyTerminationStrategy,
}

impl HNSWSearchConfiguration {
    /// Sets the ef_search parameter. Returns self for chaining.
    #[must_use]
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Sets the early termination strategy. Returns self for chaining.
    #[must_use]
    pub fn with_early_termination(mut self, strategy: EarlyTerminationStrategy) -> Self {
        self.early_termination = strategy;
        self
    }
}

impl Default for HNSWSearchConfiguration {
    /// Provides a default `ef_search` value.
    fn default() -> Self {
        Self {
            ef_search: 100,
            early_termination: EarlyTerminationStrategy::None,
        }
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset,
    G: GraphTrait + From<GrowableGraph>,
{
    /// Return the maximum level of the HNSW graph (0-based).
    #[must_use]
    #[inline]
    pub fn max_level(&self) -> usize {
        if self.levels.is_empty() {
            0
        } else {
            self.levels.len() - 1
        }
    }

    /// Returns a vec with the number of nodes at each level, from highest to lowest (ground).
    #[must_use]
    pub fn nodes_per_level(&self) -> Vec<usize> {
        self.levels.iter().map(|g| g.n_nodes()).collect()
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset,
    G: GraphTrait + From<GrowableGraph>,
{
    /// Converts an `HNSW` index from a different dataset type, preserving the graph structure.
    ///
    /// Only the dataset is replaced; levels, entry point, level mappings, and neighbor counts
    /// are moved unchanged. The caller must ensure that the new dataset `D` has the same
    /// number of vectors and the same logical vector order as `T`.
    pub fn convert_dataset_from<T: Dataset>(hnsw: HNSW<T, G>) -> Self
    where
        D: Dataset + ConvertFrom<T>,
    {
        let HNSW {
            levels,
            level1_to_level0_mapping,
            dataset,
            num_neighbors_per_vec,
            entry_point,
        } = hnsw;

        Self {
            levels,
            level1_to_level0_mapping,
            dataset: ConvertInto::<D>::convert_into(dataset),
            num_neighbors_per_vec,
            entry_point,
        }
    }

    /// Converts this `HNSW` into one backed by a different dataset type (consuming self).
    ///
    /// This is the mirror of [`convert_dataset_from`]. Prefer this when you own the index
    /// and want to chain from a plain build:
    ///
    /// ```rust,ignore
    /// let plain: HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph> =
    ///     HNSW::build_index(dataset, &config);
    /// let compressed: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
    ///     plain.convert_dataset_into();
    /// ```
    pub fn convert_dataset_into<T>(self) -> HNSW<T, G>
    where
        T: Dataset + ConvertFrom<D>,
    {
        HNSW::<T, G>::convert_dataset_from(self)
    }

    /// Converts this `HNSW` into one backed by a different dataset type using a borrowed source dataset.
    ///
    /// Use this when the target dataset implements `ConvertFrom<&D>` instead of `ConvertFrom<D>`.
    pub fn convert_dataset_into_ref<T>(self) -> HNSW<T, G>
    where
        T: Dataset,
        for<'a> T: ConvertFrom<&'a D>,
    {
        let HNSW {
            levels,
            level1_to_level0_mapping,
            dataset,
            num_neighbors_per_vec,
            entry_point,
        } = self;

        HNSW {
            levels,
            level1_to_level0_mapping,
            dataset: T::convert_from(&dataset),
            num_neighbors_per_vec,
            entry_point,
        }
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync,
    <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    G: GraphTrait + From<GrowableGraph>,
{
    /// Performs ACORN-1 filtered approximate nearest-neighbor search.
    ///
    /// Returns the `k` approximate nearest neighbors of `query` that satisfy
    /// `predicate(vector_id) == true`. Unlike a simple post-filter, the predicate
    /// is applied *during* graph traversal: non-matching nodes are skipped and their
    /// neighbors are inspected via a two-hop expansion to maintain connectivity in
    /// the filtered sub-graph.
    ///
    /// The HNSW index does **not** need to be rebuilt; this method works on any
    /// standard HNSW index (ACORN-1 variant).
    ///
    /// # Arguments
    /// * `query` – The query vector.
    /// * `k` – Number of nearest neighbors to return.
    /// * `search_params` – Search configuration (`ef_search`, early termination).
    /// * `predicate` – `Fn(vector_id: usize) -> bool`. Called with the global
    ///   (dataset-level) vector ID; only vectors for which this returns `true`
    ///   will appear in results.
    pub fn search_filtered<'q, F>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &HNSWSearchConfiguration,
        predicate: F,
    ) -> Vec<vectorium::dataset::ScoredVector<<D::Encoder as VectorEncoder>::Distance>>
    where
        F: Fn(usize) -> bool,
    {
        let query_eval = self.dataset.encoder().query_evaluator(query);
        let num_levels = self.levels.len();

        // --- Stage 1: upper levels (unfiltered greedy search, same as standard HNSW) ---
        let entry_graph = if num_levels > 1 {
            &self.levels[0]
        } else {
            &self.levels[num_levels - 1]
        };
        let entry_external_id = entry_graph.get_external_id(self.entry_point) as VectorId;
        let entry_distance = query_eval.compute_distance(self.dataset.get(entry_external_id));
        let mut entry_node = ScoredItemGeneric {
            distance: entry_distance,
            vector: self.entry_point,
        };
        if num_levels > 1 {
            for level_graph in &self.levels[..num_levels - 1] {
                entry_node =
                    level_graph.greedy_search_nearest(&self.dataset, &query_eval, entry_node);
            }
        }

        // --- Stage 2: ground level (ACORN-1 filtered search) ---
        let ground_graph = &self.levels[num_levels - 1];
        let entry_global_id = if num_levels > 1 {
            self.level1_to_level0_mapping[entry_node.vector]
        } else {
            self.entry_point
        };
        let ground_entry_node = ScoredItemGeneric {
            distance: entry_node.distance,
            vector: entry_global_id,
        };

        let ef = search_params.ef_search.max(k);
        let lambda = search_params.early_termination.lambda();
        let top_heap = ground_graph.acorn_search_candidates_filtered(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            ef,
            k,
            lambda,
            &predicate,
        );

        let mut topk = top_heap.into_sorted_vec();
        topk.truncate(k);
        topk.drain(..)
            .map(|candidate| vectorium::dataset::ScoredVector {
                distance: candidate.distance,
                vector: candidate.vector as VectorId,
            })
            .collect()
    }
}

impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync,
    <D::Encoder as VectorEncoder>::Distance: Distance,
    G: GraphTrait + From<GrowableGraph>,
{
    /// Build pre-expanded neighbor lists for ACORN-γ filtered search.
    ///
    /// For each ground-level node `v`, the two-hop neighborhood (direct neighbors
    /// and their neighbors) is scored by distance to `v` and pruned to `gamma * M`
    /// entries, sorted closest-first.
    ///
    /// Call this **once** after the standard HNSW build, then pass the result to
    /// [`search_filtered_gamma`] for fast predicate-aware search.
    ///
    /// # Arguments
    /// * `gamma` – Expansion factor (≥ 1). Each node stores up to `gamma * M`
    ///   neighbors. Larger values improve recall at the cost of memory and build time.
    pub fn build_acorn_gamma_neighbors(&self, gamma: usize) -> AcornGammaNeighbors {
        let n = self.dataset.len();
        let m = self.num_neighbors_per_vec;
        let gamma_m = (gamma * m).max(1);
        let ground_graph = &self.levels[self.levels.len() - 1];

        let mut expanded: Vec<Box<[usize]>> = Vec::with_capacity(n);

        for v in 0..n {
            // Collect the two-hop neighborhood, excluding v itself.
            let mut seen: HashSet<usize> = HashSet::new();
            seen.insert(v);
            let mut candidates: Vec<usize> = Vec::new();

            for u in ground_graph.neighbors(v) {
                if seen.insert(u) {
                    candidates.push(u);
                }
                for w in ground_graph.neighbors(u) {
                    if seen.insert(w) {
                        candidates.push(w);
                    }
                }
            }

            // Score each candidate by distance to v.
            let v_vec = self.dataset.get(v as VectorId);
            let eval = self.dataset.encoder().vector_evaluator(v_vec);

            let mut scored: Vec<(<D::Encoder as VectorEncoder>::Distance, usize)> = candidates
                .into_iter()
                .map(|u| {
                    let d = eval.compute_distance(self.dataset.get(u as VectorId));
                    (d, u)
                })
                .collect();

            // Sort ascending (closest first), truncate to gamma * M.
            scored.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            scored.truncate(gamma_m);

            expanded.push(
                scored
                    .into_iter()
                    .map(|(_, u)| u)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
        }

        AcornGammaNeighbors {
            neighbors: expanded.into_boxed_slice(),
            gamma_m,
        }
    }

    /// ACORN-γ filtered approximate nearest-neighbor search.
    ///
    /// Unlike ACORN-1 ([`search_filtered`]), the two-hop expansion is
    /// **pre-computed** at index construction time (via
    /// [`build_acorn_gamma_neighbors`]). At search time, standard beam search
    /// runs over the pre-expanded neighbor lists; predicate-failing nodes are
    /// simply skipped — no on-the-fly two-hop expansion is needed.
    ///
    /// # Arguments
    /// * `query` – The query vector.
    /// * `k` – Number of nearest neighbors to return.
    /// * `search_params` – Search configuration (`ef_search`, early termination).
    /// * `acorn_gamma` – Pre-built expanded neighbor lists (from `build_acorn_gamma_neighbors`).
    /// * `predicate` – `Fn(vector_id: usize) -> bool`. Only vectors satisfying
    ///   this will appear in the results.
    pub fn search_filtered_gamma<'q, F>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &HNSWSearchConfiguration,
        acorn_gamma: &AcornGammaNeighbors,
        predicate: F,
    ) -> Vec<vectorium::dataset::ScoredVector<<D::Encoder as VectorEncoder>::Distance>>
    where
        F: Fn(usize) -> bool,
    {
        let query_eval = self.dataset.encoder().query_evaluator(query);
        let num_levels = self.levels.len();

        // --- Stage 1: upper levels (unfiltered greedy search, same as ACORN-1) ---
        let entry_graph = if num_levels > 1 {
            &self.levels[0]
        } else {
            &self.levels[num_levels - 1]
        };
        let entry_external_id = entry_graph.get_external_id(self.entry_point) as VectorId;
        let entry_distance = query_eval.compute_distance(self.dataset.get(entry_external_id));
        let mut entry_node = ScoredItemGeneric {
            distance: entry_distance,
            vector: self.entry_point,
        };
        if num_levels > 1 {
            for level_graph in &self.levels[..num_levels - 1] {
                entry_node =
                    level_graph.greedy_search_nearest(&self.dataset, &query_eval, entry_node);
            }
        }

        // --- Stage 2: ground level (ACORN-γ search on pre-expanded neighbor lists) ---
        let entry_global_id = if num_levels > 1 {
            self.level1_to_level0_mapping[entry_node.vector]
        } else {
            self.entry_point
        };
        let ground_entry_node = ScoredItemGeneric {
            distance: entry_node.distance,
            vector: entry_global_id,
        };

        let ef = search_params.ef_search.max(k);
        let lambda = search_params.early_termination.lambda();
        let top_heap = acorn_gamma.acorn_gamma_search_filtered(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            ef,
            k,
            lambda,
            &predicate,
        );

        let mut topk = top_heap.into_sorted_vec();
        topk.truncate(k);
        topk.drain(..)
            .map(|candidate| vectorium::dataset::ScoredVector {
                distance: candidate.distance,
                vector: candidate.vector as VectorId,
            })
            .collect()
    }
}

impl<D, G> Index<D> for HNSW<D, G>
where
    D: Dataset + Sync + SpaceUsage,
    <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    G: GraphTrait + From<GrowableGraph>,
{
    type BuildParams = HNSWBuildConfiguration;
    type SearchParams = HNSWSearchConfiguration;

    #[inline]
    fn n_vectors(&self) -> usize {
        self.dataset.len()
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dataset.input_dim()
    }

    fn print_space_usage_bytes(&self) {
        let dataset_size = self.dataset.space_usage_bytes();
        let index_size = self
            .levels
            .iter()
            .map(|g| g.get_space_usage_bytes())
            .sum::<usize>();

        let total_size = dataset_size + index_size;
        println!(
            "[######] Space usage: Dataset: {dataset_size} bytes, Index: {index_size} bytes, Total: {total_size} bytes"
        );
    }

    fn search<'q>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<vectorium::dataset::ScoredVector<<D::Encoder as VectorEncoder>::Distance>> {
        let query_eval = self.dataset.encoder().query_evaluator(query);
        let num_levels = self.levels.len();

        // --- Stage 1: Search upper levels ---
        // Start at the single entry point on the highest level.
        let entry_graph = if num_levels > 1 {
            &self.levels[0]
        } else {
            &self.levels[num_levels - 1]
        };
        let entry_external_id = entry_graph.get_external_id(self.entry_point) as VectorId;
        let entry_distance = query_eval.compute_distance(self.dataset.get(entry_external_id));
        let mut entry_node = ScoredItemGeneric {
            distance: entry_distance,
            vector: self.entry_point,
        };
        if num_levels > 1 {
            // Greedily search from the top level down to level 1.
            for level_graph in &self.levels[..num_levels - 1] {
                entry_node =
                    level_graph.greedy_search_nearest(&self.dataset, &query_eval, entry_node);
            }
        }

        // --- Stage 2: Search ground level ---
        // The ground level contains all the vectors.
        let ground_graph = &self.levels[num_levels - 1];
        let entry_global_id = if num_levels > 1 {
            // The entry_node now holds the local ID from the last searched upper level (level 1).
            // We need to map this to a global ID for the ground level to start the final search.
            self.level1_to_level0_mapping[entry_node.vector]
        } else {
            // No upper levels, the entry point is a ground-level ID.
            self.entry_point
        };

        // The distance from the previous level's search is a good starting point.
        let ground_entry_node = ScoredItemGeneric {
            distance: entry_node.distance,
            vector: entry_global_id,
        };

        // Perform the final, most extensive search on the ground level.
        // Ensure that `ef_search` is at least `k` to guarantee we can return `k` results.
        let ef = search_params.ef_search.max(k);
        let lambda = search_params.early_termination.lambda();
        let mut topk = ground_graph.greedy_search_topk(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            k,
            ef,
            lambda,
        );

        // Map local IDs to global vector IDs and return scored vectors
        topk.drain(..)
            .map(|candidate| vectorium::dataset::ScoredVector {
                distance: candidate.distance,
                vector: candidate.vector as VectorId,
            })
            .collect()
    }

    /// Builds the HNSW index from a source dataset.
    ///
    /// This function orchestrates the entire build process:
    /// 1. It computes the random level assignments for each vector.
    /// 2. It initializes the graph structures for each level.
    /// 3. It inserts the single entry point node.
    /// 4. It iterates through all HNSW levels, from highest to lowest, inserting nodes.
    ///    - A hybrid sequential/parallel strategy is used based on the number of nodes at each level.
    /// 5. It finalizes the graph structures and creates the final `HNSW` index struct.
    fn build_index(dataset: D, build_params: &Self::BuildParams) -> Self {
        let num_vectors = dataset.len();
        let m = build_params.num_neighbors_per_vec;
        let default_probabs =
            compute_levels_probabilities(1.0 / (m as f32).ln(), num_vectors as f32);

        // // 1. Get level assignments and sorted IDs.
        let (levels_mapping, ids_sorted_by_level, cumulative_ids_per_level, max_level) =
            compute_levels(&default_probabs, num_vectors);

        // 2. Setup graphs and mappings.
        let mut growable_levels: Vec<GrowableGraph> = Vec::with_capacity(max_level as usize + 1);

        // Initialize upper levels (from highest to lowest)
        for i in (1..=max_level).rev() {
            let mut graph = GrowableGraph::with_max_degree(m);
            let num_nodes_in_level = levels_mapping[i as usize - 1].len();
            graph.reserve(num_nodes_in_level);
            graph
                .set_mapping(levels_mapping[i as usize - 1].clone())
                .expect("Graph mapping size validation should have passed");
            growable_levels.push(graph);
        }

        // Initialize ground level
        let mut ground_graph = GrowableGraph::with_max_degree(2 * m);
        ground_graph.reserve(num_vectors);
        growable_levels.push(ground_graph);

        let level1_to_level0_mapping = if max_level > 0 {
            levels_mapping[0].clone()
        } else {
            Vec::new()
        };
        let entry_point_local_id = 0;

        // 3. Build all levels by iterating through nodes level by level.
        let entry_point_global_id = ids_sorted_by_level[0];

        // --- START: Progress Bar Setup ---
        let pb = ProgressBar::new(num_vectors as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) - Building HNSW")
                .unwrap()
                .progress_chars("#>-"),
        );
        // --- END: Progress Bar Setup ---

        // Insert the entry point (the first node in the sorted list)
        Self::insert_entry_point(&mut growable_levels, entry_point_global_id, max_level, &pb);

        // Main build loop: iterate through HNSW levels from highest to lowest
        for level in (0..=max_level).rev() {
            let start_index = cumulative_ids_per_level[max_level as usize - level as usize];
            let start_index = if start_index == 0 { 1 } else { start_index };
            let end_index = cumulative_ids_per_level[max_level as usize - level as usize + 1];
            if start_index >= end_index {
                continue;
            }

            let nodes_to_insert_slice = &ids_sorted_by_level[start_index..end_index];

            // HYBRID STRATEGY: Use parallel processing only for levels with enough nodes.
            if nodes_to_insert_slice.len() > 2 * build_params.max_build_batch_size {
                Self::process_level_parallelly(
                    nodes_to_insert_slice,
                    level,
                    max_level,
                    m,
                    &mut growable_levels,
                    &dataset,
                    build_params,
                    entry_point_local_id,
                    &level1_to_level0_mapping,
                    &ids_sorted_by_level,
                    &pb,
                );
            } else {
                Self::process_level_sequentially(
                    nodes_to_insert_slice,
                    level,
                    max_level,
                    m,
                    &mut growable_levels,
                    &dataset,
                    build_params,
                    entry_point_local_id,
                    &level1_to_level0_mapping,
                    &ids_sorted_by_level,
                    &pb,
                );
            }
        }

        pb.finish_with_message("HNSW build complete.");

        // 4. Finalize and create the HNSW struct.
        let final_levels: Vec<G> = growable_levels.into_iter().map(Into::into).collect();

        Self {
            levels: final_levels.into_boxed_slice(),
            level1_to_level0_mapping: level1_to_level0_mapping.into_boxed_slice(),
            dataset,
            num_neighbors_per_vec: m,
            entry_point: entry_point_local_id,
        }
    }
}

impl<D, G> IndexSerializer for HNSW<D, G> {}

/// Computes the probabilities for a node to be assigned to each level in the HNSW graph.
///
/// # Parameters
///
/// - `level_mult`: A multiplier that affects the exponential decay of probabilities for each level.
///
/// # Returns
///
/// - A vector of probabilities for each level, where each probability is computed based on the formula:
///   `probability = exp(-level / level_mult) * (1 - exp(- 1 / level_mult))`.
///
///   The probabilities decrease exponentially with increasing level, controlled by `level_mult`.
///
/// The function continues to compute these values for increasing levels until the calculated
/// probability for a level falls below a small threshold.
///
/// # Example
///
/// After calling this function with a `level_mult` of `1.0`, the probabilities decrease exponentially,
/// e.g., starting around [0.6321, 0.3679, 0.1353, ...].
///
/// ```text
/// // Example (illustrative values):
/// // probabs_levels ≈ [0.6321, 0.3679, 0.1353, ...]
/// ```
#[must_use]
fn compute_levels_probabilities(level_mult: f32, dataset_len: f32) -> Vec<f32> {
    let mut probabs_levels = Vec::new();

    for level in 0.. {
        let proba = (-level as f32 / level_mult).exp() * (1.0 - (-1.0 / level_mult).exp());

        // Prune levels with expected number of assigned nodes below 1
        if proba < 1.0 / dataset_len {
            break;
        }
        probabs_levels.push(proba);
    }

    probabs_levels
}

/// This function generates a random level for a node in the HNSW graph.
///
/// # Description
///
/// The function begins by generating a random floating-point number `f` between 0.0 and 1.0.
/// The function then iterates over the `probabs_levels` vector, comparing `f` with the probability thresholds for
/// each level. If `f` is less than the current level's probability, that level is selected and returned as a `u8`.
/// If `f` is larger, the function reduces `f` by the threshold value and continues to the next level. If no level
/// is selected, the maximum level, which corresponds to the last index of `probabs_levels`, is returned.
///
/// # Parameters
///
/// - `probabs_levels`: A vector whose i-th entry represents the probability of selecting level `i` of the HNSW graph.
/// - `rng`: A mutable reference to a random number generator of type `StdRng`.
///
/// # Returns
///
/// - `u8`: The level selected for the node, ranging from 0 to the maximum level.
///
/// /// # Example
///
/// Assume `probabs_levels` contains `[0.6, 0.3, 0.1]` and the random value `f` is `0.65`.
/// After checking level 0 (0.6),`f` is decreased by 0.6 to become `0.05`. The function would then
/// return level 1, as `0.05` is less than the probability for level 1 (0.3).
#[must_use]
#[inline]
fn random_level(probabs_levels: &[f32], rng: &mut StdRng) -> u8 {
    let mut f: f32 = rng.gen_range(0.0..1.0);
    for (level, &prob) in probabs_levels.iter().enumerate() {
        if f < prob {
            return level as u8;
        }
        f -= prob;
    }
    // it returns the maximum level which is the size of the vector probabs_levels
    (probabs_levels.len() - 1) as u8
}

/// Assigns levels to each vector in the graph and updates the internal `offsets` and `neighbors` vectors.
///
/// # Arguments
///
/// - `default_probabs`: A vector of probabilities for each level, which is used to determine the level assignment for each vector.
/// - `num_vectors`: The number of vectors to which levels will be assigned.
///
/// # Description
///
/// This function assigns a level to each vector in the graph and computes the levels matrix which contains the IDs of vectors at each level.
/// It uses a random number generator to select a level based on the provided probabilities. Each vector is assigned to all levels up to and including its assigned level.
/// The function also keeps track of the maximum level assigned to any vector, that could be lower than the length of `default_probabs` in case no vector was assigned to a level.
/// Finally, it ensures that the levels vector does not contain any empty vectors, removing them if necessary.
///
/// # Returns
/// /// - A tuple containing:
///  - A vector of vectors, where each inner vector contains the IDs of vectors assigned to that level.
///  - The maximum level assigned to any vector.
///
#[must_use]
#[inline]
fn compute_levels(
    default_probabs: &[f32],
    num_vectors: usize,
) -> (Vec<Vec<usize>>, Vec<usize>, Vec<usize>, u8) {
    let mut rng = StdRng::seed_from_u64(523);

    // 1. Create a shuffled list of all node IDs. This is the single source of randomness.
    let mut all_ids: Vec<usize> = (0..num_vectors).collect();
    all_ids.shuffle(&mut rng);

    // 2. Assign a highest level to each node.
    // `ids_per_level[i]` will store nodes whose highest assigned level is `i`.
    let mut ids_per_level: Vec<Vec<usize>> = vec![Vec::new(); default_probabs.len() + 1];
    for &id in &all_ids {
        let level = random_level(default_probabs, &mut rng);
        ids_per_level[level as usize].push(id);
    }

    // 3. Find the actual maximum level that has any nodes assigned to it.
    let max_level = ids_per_level
        .iter()
        .rposition(|level_nodes| !level_nodes.is_empty())
        .unwrap_or(0) as u8;

    // 4. Create the final, sorted build order.
    // Candidates are ordered by level (highest to lowest). Because we populated `ids_per_level`
    // from a shuffled list, the nodes within each level block are already randomized.
    let mut ids_sorted_by_level: Vec<usize> = Vec::with_capacity(num_vectors);
    for i in (0..=max_level).rev() {
        ids_sorted_by_level.extend(&ids_per_level[i as usize]);
    }

    // 5. `cumulative_ids_per_level` tracks the number of nodes *at or above* a given HNSW level.
    // It's used to slice `ids_sorted_by_level` during the build loop.
    let mut cumulative_ids_per_level = Vec::with_capacity(max_level as usize + 2);
    cumulative_ids_per_level.push(0);
    let mut count = 0;
    for i in (0..=max_level).rev() {
        count += ids_per_level[i as usize].len();
        cumulative_ids_per_level.push(count);
    }

    // 6. `levels_mapping[i]` contains all global IDs present at HNSW level `i+1`.
    // A node at level L is also present at all levels < L. The mapping for each level
    // is now a consistent prefix of the final `ids_sorted_by_level` list.
    let mut levels_mapping: Vec<Vec<usize>> = Vec::with_capacity(max_level as usize);
    for i in 0..max_level as usize {
        // HNSW level `i+1` corresponds to `levels_mapping[i]`.
        // The nodes for this level are all nodes from the highest level down to level `i+1`.
        let num_nodes_at_this_level_or_above = cumulative_ids_per_level[max_level as usize - i];
        let mapping_for_this_level: Vec<usize> =
            ids_sorted_by_level[0..num_nodes_at_this_level_or_above].to_vec();
        levels_mapping.push(mapping_for_this_level);
    }

    (
        levels_mapping,
        ids_sorted_by_level,
        cumulative_ids_per_level,
        max_level,
    )
}

// --- Private Helper Methods for HNSW build process ---
impl<D, G> HNSW<D, G>
where
    D: Dataset + Sync,
    <D::Encoder as VectorEncoder>::Distance: Ord + Copy,
    G: GraphTrait,
{
    fn insert_entry_point(
        growable_levels: &mut [GrowableGraph],
        entry_point_global_id: usize,
        max_level: u8,
        pb: &ProgressBar,
    ) {
        for (i, graph) in growable_levels.iter_mut().enumerate() {
            if i < max_level as usize {
                // Is an upper level
                graph.push_with_precomputed_reverse_links(Some(entry_point_global_id), &[], 0, &[]);
            } else {
                // Is the ground level
                graph.push_with_precomputed_reverse_links(None, &[], entry_point_global_id, &[]);
            }
        }
        pb.inc(1); // Increment for the entry point

        // After inserting the entry point, we must advance the counter on all upper levels.
        for graph in growable_levels.iter_mut().take(max_level as usize) {
            graph.advance_inserted_nodes(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_level_sequentially(
        nodes_to_insert_slice: &[usize],
        level: u8,
        max_level: u8,
        m: usize,
        growable_levels: &mut [GrowableGraph],
        source_dataset: &D,
        build_params: &HNSWBuildConfiguration,
        entry_point_local_id: usize,
        level1_to_level0_mapping: &[usize],
        ids_sorted_by_level: &[usize],
        pb: &ProgressBar,
    ) where
        <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    {
        let entry_point_global_id = ids_sorted_by_level[0];
        for &global_id in nodes_to_insert_slice {
            let query_eval = source_dataset
                .encoder()
                .vector_evaluator(source_dataset.get(global_id as VectorId));
            let entry_distance =
                query_eval.compute_distance(source_dataset.get(entry_point_global_id as VectorId));
            let mut entry_node = ScoredItemGeneric {
                distance: entry_distance,
                vector: entry_point_local_id,
            };

            if level > 0 {
                for current_level in ((level + 1)..=max_level).rev() {
                    let graph_idx = max_level as usize - current_level as usize;
                    entry_node = growable_levels[graph_idx].greedy_search_nearest(
                        source_dataset,
                        &query_eval,
                        entry_node,
                    );
                }
                for current_level in (1..=level).rev() {
                    let graph_idx = max_level as usize - current_level as usize;
                    let graph = &mut growable_levels[graph_idx];
                    let local_id = graph.inserted_nodes();

                    let (forward, reverse, new_entry) = graph.find_and_prune_neighbors(
                        source_dataset,
                        &query_eval,
                        entry_node,
                        build_params.ef_construction,
                        m,
                        local_id,
                    );

                    graph.push_with_precomputed_reverse_links(
                        Some(global_id),
                        &forward,
                        local_id,
                        &reverse,
                    );
                    graph.advance_inserted_nodes(1);
                    entry_node = new_entry;
                }
            }

            let ground_graph = &mut growable_levels[max_level as usize];
            let ground_entry_global_id = if max_level > 0 {
                level1_to_level0_mapping[entry_node.vector]
            } else {
                ids_sorted_by_level[0]
            };
            let dist =
                query_eval.compute_distance(source_dataset.get(ground_entry_global_id as VectorId));
            let ground_entry_node = ScoredItemGeneric {
                distance: dist,
                vector: ground_entry_global_id,
            };

            let (ground_neighbors, ground_reverse_links, _) = ground_graph
                .find_and_prune_neighbors(
                    source_dataset,
                    &query_eval,
                    ground_entry_node,
                    build_params.ef_construction,
                    2 * m,
                    global_id,
                );

            ground_graph.push_with_precomputed_reverse_links(
                None,
                &ground_neighbors,
                global_id,
                &ground_reverse_links,
            );
            pb.inc(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn process_level_parallelly(
        nodes_to_insert_slice: &[usize],
        level: u8,
        max_level: u8,
        m: usize,
        growable_levels: &mut [GrowableGraph],
        source_dataset: &D,
        build_params: &HNSWBuildConfiguration,
        entry_point_local_id: usize,
        level1_to_level0_mapping: &[usize],
        ids_sorted_by_level: &[usize],
        pb: &ProgressBar,
    ) where
        <D::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    {
        let mut current_batch_size = build_params.initial_build_batch_size;
        let max_batch_size = build_params.max_build_batch_size;
        let level_start_local_ids: Vec<usize> =
            growable_levels.iter().map(|g| g.inserted_nodes()).collect();
        let mut processed_nodes = 0;
        let entry_point_global_id = ids_sorted_by_level[0];

        while processed_nodes < nodes_to_insert_slice.len() {
            let remaining_nodes = nodes_to_insert_slice.len() - processed_nodes;
            let actual_batch_size = current_batch_size.min(remaining_nodes);
            let batch =
                &nodes_to_insert_slice[processed_nodes..processed_nodes + actual_batch_size];

            let insertion_data: Vec<_> = batch
                .par_iter()
                .enumerate()
                .map(|(i, &global_id)| {
                    let query_eval = source_dataset
                        .encoder()
                        .vector_evaluator(source_dataset.get(global_id as VectorId));
                    let entry_distance = query_eval
                        .compute_distance(source_dataset.get(entry_point_global_id as VectorId));
                    let mut entry_node = ScoredItemGeneric {
                        distance: entry_distance,
                        vector: entry_point_local_id,
                    };
                    let mut upper_level_data = Vec::new();

                    if level > 0 {
                        for current_level in ((level + 1)..=max_level).rev() {
                            let graph_idx = max_level as usize - current_level as usize;
                            entry_node = growable_levels[graph_idx].greedy_search_nearest(
                                source_dataset,
                                &query_eval,
                                entry_node,
                            );
                        }
                        for current_level in (1..=level).rev() {
                            let graph_idx = max_level as usize - current_level as usize;
                            let graph = &growable_levels[graph_idx];
                            let local_id = level_start_local_ids[graph_idx] + processed_nodes + i;

                            let (forward, reverse, new_entry) = graph.find_and_prune_neighbors(
                                source_dataset,
                                &query_eval,
                                entry_node,
                                build_params.ef_construction,
                                m,
                                local_id,
                            );
                            upper_level_data.push((forward, reverse));
                            entry_node = new_entry;
                        }
                    }

                    let ground_graph = &growable_levels[max_level as usize];
                    let ground_entry_global_id = if max_level > 0 {
                        level1_to_level0_mapping[entry_node.vector]
                    } else {
                        ids_sorted_by_level[0]
                    };
                    let dist = query_eval
                        .compute_distance(source_dataset.get(ground_entry_global_id as VectorId));
                    let ground_entry_node = ScoredItemGeneric {
                        distance: dist,
                        vector: ground_entry_global_id,
                    };

                    let (ground_neighbors, ground_reverse_links, _) = ground_graph
                        .find_and_prune_neighbors(
                            source_dataset,
                            &query_eval,
                            ground_entry_node,
                            build_params.ef_construction,
                            2 * m,
                            global_id,
                        );

                    (
                        global_id,
                        upper_level_data,
                        (ground_neighbors, ground_reverse_links),
                    )
                })
                .collect();

            // Insert the computed data into the graphs
            for (i, (global_id, upper_level_data, ground_data)) in
                insertion_data.into_iter().enumerate()
            {
                for (level_idx, (forward, reverse)) in
                    upper_level_data.into_iter().rev().enumerate()
                {
                    let hnsw_level = level_idx + 1;
                    let graph_idx = max_level as usize - hnsw_level;
                    let graph = &mut growable_levels[graph_idx];
                    let local_id = level_start_local_ids[graph_idx] + processed_nodes + i;
                    graph.push_with_precomputed_reverse_links(
                        Some(global_id),
                        &forward,
                        local_id,
                        &reverse,
                    );
                }
                let (forward, reverse) = ground_data;
                let ground_graph = &mut growable_levels[max_level as usize];
                ground_graph
                    .push_with_precomputed_reverse_links(None, &forward, global_id, &reverse);
            }

            // Advance the counters for upper levels
            for current_level in (1..=level).rev() {
                let graph_idx = max_level as usize - current_level as usize;
                growable_levels[graph_idx].advance_inserted_nodes(actual_batch_size);
            }

            processed_nodes += actual_batch_size;
            pb.inc(actual_batch_size as u64);

            if current_batch_size < max_batch_size {
                current_batch_size = (current_batch_size * 2).min(max_batch_size);
            }
        }
    }
}

#[cfg(test)]
mod convert_dataset_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
    use vectorium::{
        DatasetGrowable, DotProduct, FixedU8Q, FixedU16Q, PackedSparseDataset, PlainSparseDataset,
        PlainSparseDatasetGrowable, PlainSparseQuantizer, ScalarSparseDataset, SparseVectorView,
    };

    fn build_test_hnsw() -> HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph> {
        let encoder = PlainSparseQuantizer::<u16, f32, DotProduct>::new(20, 20);
        let mut growable: PlainSparseDatasetGrowable<u16, f32, DotProduct> =
            PlainSparseDatasetGrowable::new(encoder);

        for i in 0u16..30 {
            let components: Vec<u16> = (0..5).map(|j: u16| (i * 3 + j) % 20).collect();
            let mut components = components;
            components.sort();
            components.dedup();
            let values: Vec<f32> = components.iter().map(|&c| (c as f32 + 1.0) * 0.1).collect();
            growable.push(SparseVectorView::new(&components, &values));
        }

        let dataset: PlainSparseDataset<u16, f32, DotProduct> = growable.into();

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(4)
            .with_ef_construction(20);

        HNSW::build_index(dataset, &config)
    }

    #[test]
    fn test_convert_dataset_into_dotvbyte() {
        let plain_hnsw = build_test_hnsw();
        let n = plain_hnsw.n_vectors();

        let hnsw: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
            plain_hnsw.convert_dataset_into();

        assert_eq!(hnsw.n_vectors(), n);
    }

    #[test]
    fn test_convert_dataset_into_fixedu8() {
        let plain_hnsw = build_test_hnsw();
        let n = plain_hnsw.n_vectors();

        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
            plain_hnsw.convert_dataset_into();

        assert_eq!(hnsw.n_vectors(), n);
    }

    #[test]
    fn test_convert_dataset_into_fixedu16() {
        let plain_hnsw = build_test_hnsw();
        let n = plain_hnsw.n_vectors();

        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
            plain_hnsw.convert_dataset_into();

        assert_eq!(hnsw.n_vectors(), n);
    }

    #[test]
    fn test_dotvbyte_search_returns_results() {
        let hnsw: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
            build_test_hnsw().convert_dataset_into();

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_fixedu8_search_returns_results() {
        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
            build_test_hnsw().convert_dataset_into();

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_fixedu16_search_returns_results() {
        let hnsw: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
            build_test_hnsw().convert_dataset_into();

        let query_components: Vec<u16> = vec![0, 1, 2];
        let query_values: Vec<f32> = vec![0.5, 0.3, 0.2];
        let query = SparseVectorView::new(&query_components, &query_values);

        let search_config = HNSWSearchConfiguration::default().with_ef_search(20);
        let results = hnsw.search(query, 5, &search_config);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }
}

#[cfg(test)]
mod acorn_search_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
    use vectorium::vector::DenseVectorView;
    use vectorium::{DenseDataset, PlainDenseDataset};

    /// Build a small 1-D HNSW for testing.
    /// Vectors are [0.0], [1.0], ..., [(n-1).0].
    fn build_1d_hnsw(n: usize) -> HNSW<PlainDenseDataset<f32, SquaredEuclideanDistance>, Graph> {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(50);
        HNSW::build_index(dataset, &config)
    }

    /// Every result of `search_filtered` must pass the predicate.
    #[test]
    fn search_filtered_all_results_pass_predicate() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);

        let query_val = [50.0f32];
        let query = DenseVectorView::new(&query_val);

        // Only even IDs allowed.
        let results = hnsw.search_filtered(query, 10, &search_config, |id| id % 2 == 0);

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(
                r.vector % 2,
                0,
                "result {} does not pass even predicate",
                r.vector
            );
        }
    }

    /// The nearest filtered result should be the closest vector satisfying the predicate.
    ///
    /// Query = 50.5.  Predicate: id divisible by 3.
    /// Nearest divisible-by-3 IDs to 50.5 are 51 (d=0.25), 48 (d=6.25), 54 (d=12.25) …
    #[test]
    fn search_filtered_finds_nearest_predicate_passing_neighbors() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(100);

        let query_val = [50.5f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered(query, 5, &search_config, |id| id % 3 == 0);

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.vector % 3, 0, "result {} is not divisible by 3", r.vector);
        }
        // The closest divisible-by-3 vector to 50.5 is 51.
        assert_eq!(
            results[0].vector, 51,
            "expected nearest filtered result 51, got {}",
            results[0].vector
        );
    }

    /// With a predicate that accepts every vector, filtered search must return at
    /// most `k` results and the nearest neighbor must match the unfiltered search.
    #[test]
    fn search_filtered_full_predicate_matches_unfiltered_nearest() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let k = 5;

        let query_val = [30.0f32];
        let query_filtered = DenseVectorView::new(&query_val);
        let query_plain = DenseVectorView::new(&query_val);

        let filtered = hnsw.search_filtered(query_filtered, k, &search_config, |_| true);
        let plain = hnsw.search(query_plain, k, &search_config);

        assert_eq!(filtered.len(), plain.len());
        // Both searches must agree on the nearest neighbor.
        assert_eq!(
            filtered[0].vector, plain[0].vector,
            "nearest neighbor mismatch: filtered={}, plain={}",
            filtered[0].vector, plain[0].vector
        );
    }

    /// When the predicate is very selective (only 1 vector passes), filtered search
    /// must still return exactly that vector — provided the query is placed near it
    /// so the HNSW entry point lands in the same neighbourhood.
    #[test]
    fn search_filtered_single_eligible_vector() {
        let hnsw = build_1d_hnsw(50);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);

        // Query near 42 so the HNSW navigates to that neighbourhood.
        // The two-hop expansion from nearby nodes will reach node 42.
        let query_val = [42.0f32];
        let query = DenseVectorView::new(&query_val);

        // Only vector 42 passes the predicate.
        let results = hnsw.search_filtered(query, 5, &search_config, |id| id == 42);

        assert_eq!(
            results.len(),
            1,
            "expected exactly 1 result, got {}",
            results.len()
        );
        assert_eq!(results[0].vector, 42);
    }

    /// When no vector satisfies the predicate, the result must be empty.
    #[test]
    fn search_filtered_no_eligible_vectors_returns_empty() {
        let hnsw = build_1d_hnsw(50);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);

        let query_val = [25.0f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered(query, 5, &search_config, |_| false);
        assert!(
            results.is_empty(),
            "expected empty results when predicate always returns false"
        );
    }
}

#[cfg(test)]
mod acorn_gamma_search_tests {
    use super::*;
    use crate::graph::Graph;
    use vectorium::distances::SquaredEuclideanDistance;
    use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
    use vectorium::vector::DenseVectorView;
    use vectorium::{DenseDataset, PlainDenseDataset};

    fn build_1d_hnsw(n: usize) -> HNSW<PlainDenseDataset<f32, SquaredEuclideanDistance>, Graph> {
        let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(1);
        let flat: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let dataset = DenseDataset::from_raw(flat.into_boxed_slice(), n, encoder);
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(8)
            .with_ef_construction(50);
        HNSW::build_index(dataset, &config)
    }

    /// `build_acorn_gamma_neighbors` with gamma=2 must produce at least as many
    /// neighbors per node as the original ground graph (two-hop union is a superset).
    #[test]
    fn build_acorn_gamma_neighbors_expands_neighbor_lists() {
        let hnsw = build_1d_hnsw(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(2);

        let n = 50usize;
        assert_eq!(acorn_gamma.n_nodes(), n);

        let ground = &hnsw.levels[hnsw.levels.len() - 1];
        for v in 0..n {
            let orig_deg = ground.neighbors(v).count();
            let expanded_deg = acorn_gamma.neighbors(v).count();
            // Two-hop union is at least as large as one-hop.
            assert!(
                expanded_deg >= orig_deg,
                "node {v}: expanded {expanded_deg} < original {orig_deg}"
            );
        }
    }

    /// Every result of `search_filtered_gamma` must pass the predicate.
    #[test]
    fn search_filtered_gamma_all_results_pass_predicate() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(4);

        let query_val = [50.0f32];
        let query = DenseVectorView::new(&query_val);

        let results =
            hnsw.search_filtered_gamma(query, 10, &search_config, &acorn_gamma, |id| id % 2 == 0);

        assert!(!results.is_empty());
        for r in &results {
            assert_eq!(r.vector % 2, 0, "node {} fails even predicate", r.vector);
        }
    }

    /// With an all-pass predicate, `search_filtered_gamma` must find the true nearest.
    #[test]
    fn search_filtered_gamma_full_predicate_finds_nearest() {
        let hnsw = build_1d_hnsw(100);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(4);

        let query_val = [37.0f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered_gamma(query, 1, &search_config, &acorn_gamma, |_| true);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].vector, 37, "nearest to 37.0 should be node 37");
    }

    /// When no vector satisfies the predicate, the result must be empty.
    #[test]
    fn search_filtered_gamma_no_eligible_vectors_returns_empty() {
        let hnsw = build_1d_hnsw(50);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(50);
        let acorn_gamma = hnsw.build_acorn_gamma_neighbors(2);

        let query_val = [25.0f32];
        let query = DenseVectorView::new(&query_val);

        let results = hnsw.search_filtered_gamma(query, 5, &search_config, &acorn_gamma, |_| false);
        assert!(results.is_empty());
    }
}
