use std::marker::PhantomData;

use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::graph::{GraphTrait, GrowableGraph};
use crate::graph_index::GraphIndex;
use crate::quantizer::{IdentityQuantizer, Quantizer, QueryEvaluator};
use crate::DotProduct;
use crate::EuclideanDistance;
use crate::{hnsw_utils::*, Dataset, DistanceType, Float, GrowableDataset};

/// A `HNSW` struct represents a Hierarchical Navigable Small World (HNSW) graph structure that is used
/// for approximate nearest neighbor (ANN) search.
///
/// This index is constructed from a dataset and configuration settings. It efficiently finds the k-closest
/// vectors in the graph for a given query vector.
///
/// # Type Parameters
/// * `D`: The type of the dataset (e.g., `DenseDataset`, `SparseDataset`).
/// * `Q`: The type of the quantizer used.
/// * `G`: The type of the graph implementation (e.g., `Graph`, `GraphFixedDegree`).
#[derive(Serialize, Deserialize)]
pub struct HNSW<D, Q, G> {
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
    /// A `PhantomData` marker that indicates the type `Q` is used in the context of the struct,
    /// ensuring proper type safety without actually storing a value of type `Q`.
    _phantom: PhantomData<Q>,
}

/// Parameters for building the HNSW index.
pub struct HNSWBuildParams {
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

impl HNSWBuildParams {
    /// Creates a new set of build parameters.
    #[must_use]
    pub fn new(
        num_neighbors_per_vec: usize,
        ef_construction: usize,
        initial_build_batch_size: usize,
        max_build_batch_size: usize,
    ) -> Self {
        Self {
            num_neighbors_per_vec,
            ef_construction,
            initial_build_batch_size,
            max_build_batch_size,
        }
    }
}

impl Default for HNSWBuildParams {
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

/// Parameters for searching the HNSW index.
pub struct HNSWSearchParams {
    /// The size of the dynamic candidate list for searching the graph.
    /// Also known as `ef` or `efSearch` in the HNSW paper. A larger
    /// value leads to more accurate results at the cost of speed.
    pub ef_search: usize,
}

impl HNSWSearchParams {
    /// Creates a new set of search parameters.
    #[must_use]
    pub fn new(ef_search: usize) -> Self {
        Self { ef_search }
    }
}

impl Default for HNSWSearchParams {
    /// Provides a default `ef_search` value.
    fn default() -> Self {
        Self { ef_search: 100 }
    }
}

impl<D, Q, G> HNSW<D, Q, G>
where
    D: Dataset<Q> + GrowableDataset<Q>,
    Q: Quantizer<DatasetType = D>,
    G: GraphTrait,
{
}

impl<D, Q, G> HNSW<D, Q, G>
where
    D: Dataset<Q> + Sync,
    Q: Quantizer<InputItem: Float, DatasetType = D> + Sync,
    G: GraphTrait,
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

impl<D, Q, G> GraphIndex<D, Q, G> for HNSW<D, Q, G>
where
    D: Dataset<Q> + GrowableDataset<Q> + Sync,
    Q: Quantizer<DatasetType = D>,
    Q: Quantizer<InputItem: Float, DatasetType = D> + Sync,
    G: GraphTrait,
{
    type BuildParams = HNSWBuildParams;
    type SearchParams = HNSWSearchParams;

    #[inline]
    fn n_vectors(&self) -> usize {
        self.dataset.len()
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dataset.dim()
    }

    fn print_space_usage_bytes(&self) {
        let dataset_size = self.dataset.get_space_usage_bytes();
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

    fn search<'a, QD, QQ>(
        &'a self,
        query: QD::DataType<'a>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<(f32, usize)>
    where
        QD: Dataset<QQ> + Sync + 'a,
        QQ: Quantizer<DatasetType = QD> + Sync + 'a,
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <QD as Dataset<QQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>,
        <Q as Quantizer>::InputItem: 'a,
    {
        let query_eval = self.dataset.query_evaluator(query);
        let num_levels = self.levels.len();

        // --- Stage 1: Search upper levels ---
        // Start at the single entry point on the highest level.
        let mut entry_node = Candidate(f32::MAX, self.entry_point);
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
            self.level1_to_level0_mapping[entry_node.id_vec()]
        } else {
            // No upper levels, the entry point is a ground-level ID.
            self.entry_point
        };

        // The distance from the previous level's search is a good starting point.
        let ground_entry_node = Candidate(entry_node.distance(), entry_global_id);

        // Perform the final, most extensive search on the ground level.
        let mut topk = ground_graph.greedy_search_topk(
            &self.dataset,
            ground_entry_node,
            &query_eval,
            k,
            search_params.ef_search,
        );

        // Adjust distance if using DotProduct distance type
        if self.dataset.quantizer().distance() == DistanceType::DotProduct {
            // TODO: Trait distanze per gestire il -
            topk.iter_mut().for_each(|(dis, _)| *dis = -(*dis));
        }
        topk
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
    fn build_from_dataset<'a, BD, IQ>(
        source_dataset: &'a BD,
        quantizer: Q,
        build_params: &Self::BuildParams,
    ) -> Self
    where
        BD: Dataset<IQ> + Sync + 'a,
        IQ: IdentityQuantizer<DatasetType = BD, T: Float> + Sync + 'a,
        <IQ as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <BD as Dataset<IQ>>::DataType<'a>>,
        D: GrowableDataset<Q, InputDataType<'a> = <BD as Dataset<IQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: 'a + Float,
    {
        let num_vectors = source_dataset.len();
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
            graph.set_mapping(levels_mapping[i as usize - 1].clone());
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
                    source_dataset,
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
                    source_dataset,
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
        let final_levels: Vec<G> = growable_levels
            .into_iter()
            .map(|g| G::from_growable_graph(&g))
            .collect();

        let mut dataset = D::new(quantizer, source_dataset.dim());
        for id in 0..source_dataset.len() {
            dataset.push(&source_dataset.get(id));
        }

        Self {
            levels: final_levels.into_boxed_slice(),
            level1_to_level0_mapping: level1_to_level0_mapping.into_boxed_slice(),
            dataset,
            num_neighbors_per_vec: m,
            entry_point: entry_point_local_id,
            _phantom: PhantomData,
        }
    }
}

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
    default_probabs: &Vec<f32>,
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
impl<D, Q, G> HNSW<D, Q, G>
where
    D: Dataset<Q> + GrowableDataset<Q> + Sync,
    Q: Quantizer<InputItem: Float, DatasetType = D> + Sync,
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
    fn process_level_sequentially<'a, BD, IQ>(
        nodes_to_insert_slice: &[usize],
        level: u8,
        max_level: u8,
        m: usize,
        growable_levels: &mut [GrowableGraph],
        source_dataset: &'a BD,
        build_params: &HNSWBuildParams,
        entry_point_local_id: usize,
        level1_to_level0_mapping: &[usize],
        ids_sorted_by_level: &[usize],
        pb: &ProgressBar,
    ) where
        BD: Dataset<IQ> + Sync + 'a,
        IQ: IdentityQuantizer<DatasetType = BD, T: Float> + Sync + 'a,
        <IQ as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <BD as Dataset<IQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: 'a + Float,
    {
        for &global_id in nodes_to_insert_slice {
            let query_eval = source_dataset.query_evaluator(source_dataset.get(global_id));
            let mut entry_node = Candidate(f32::MAX, entry_point_local_id);

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
                level1_to_level0_mapping[entry_node.id_vec()]
            } else {
                ids_sorted_by_level[0]
            };
            let dist = query_eval.compute_distance(source_dataset, ground_entry_global_id);
            let ground_entry_node = Candidate(dist, ground_entry_global_id);

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
    fn process_level_parallelly<'a, BD, IQ>(
        nodes_to_insert_slice: &[usize],
        level: u8,
        max_level: u8,
        m: usize,
        growable_levels: &mut [GrowableGraph],
        source_dataset: &'a BD,
        build_params: &HNSWBuildParams,
        entry_point_local_id: usize,
        level1_to_level0_mapping: &[usize],
        ids_sorted_by_level: &[usize],
        pb: &ProgressBar,
    ) where
        BD: Dataset<IQ> + Sync + 'a,
        IQ: IdentityQuantizer<DatasetType = BD, T: Float> + Sync + 'a,
        <IQ as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <BD as Dataset<IQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: 'a + Float,
    {
        let mut current_batch_size = build_params.initial_build_batch_size;
        let max_batch_size = build_params.max_build_batch_size;
        let level_start_local_ids: Vec<usize> =
            growable_levels.iter().map(|g| g.inserted_nodes()).collect();
        let mut processed_nodes = 0;

        while processed_nodes < nodes_to_insert_slice.len() {
            let remaining_nodes = nodes_to_insert_slice.len() - processed_nodes;
            let actual_batch_size = current_batch_size.min(remaining_nodes);
            let batch =
                &nodes_to_insert_slice[processed_nodes..processed_nodes + actual_batch_size];

            let insertion_data: Vec<_> = batch
                .par_iter()
                .enumerate()
                .map(|(i, &global_id)| {
                    let query_eval = source_dataset.query_evaluator(source_dataset.get(global_id));
                    let mut entry_node = Candidate(f32::MAX, entry_point_local_id);
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
                        level1_to_level0_mapping[entry_node.id_vec()]
                    } else {
                        ids_sorted_by_level[0]
                    };
                    let dist = query_eval.compute_distance(source_dataset, ground_entry_global_id);
                    let ground_entry_node = Candidate(dist, ground_entry_global_id);

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
