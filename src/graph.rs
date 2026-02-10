use std::cmp::Reverse;
use std::collections::BinaryHeap;

use optional::Optioned;
use serde::{Deserialize, Serialize};
use vectorium::core::dataset::ScoredItemGeneric;
use vectorium::vector_encoder::{QueryEvaluator, VectorEncoder};
use vectorium::{Dataset, VectorId};

use crate::hnsw_utils::{add_neighbor_to_heaps, from_max_heap_to_min_heap};
use crate::visited_set::{VisitedSet, create_visited_set};

/// A trait that defines the common interface for different graph implementations.
///
/// This allows graph indexes to be generic over the specific graph storage strategy.
/// Graph construction is handled through concrete type constructors and `Default`.
pub trait GraphTrait {
    /// Returns an iterator over the local IDs of the neighbors of node `u`.
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a;

    /// Returns the number of nodes in the graph.
    #[must_use]
    fn n_nodes(&self) -> usize;

    /// Returns true if the graph is empty, false otherwise.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.n_nodes() == 0
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    fn n_edges(&self) -> usize;

    /// Returns the maximum degree of any node in the graph.
    #[must_use]
    fn max_degree(&self) -> usize;

    /// Returns the external (original dataset) ID of a node given its local graph ID.
    /// If the graph has no external ID mapping, this function returns the local ID itself.
    #[must_use]
    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        id
    }

    /// Returns the memory space used by the graph structure in bytes.
    #[must_use]
    fn get_space_usage_bytes(&self) -> usize;

    /// Greedily searches for the single nearest neighbor to a query, starting from an `entry_point`.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `query_evaluator`: An evaluator that can compute the distance from the query to any vector in the dataset.
    /// * `entry_point`: The candidate (`distance`, `id`) from which the search begins.
    ///
    /// # Returns
    /// The best `ScoredItemGeneric` found during the search.
    #[must_use]
    fn greedy_search_nearest<'e, D>(
        &self,
        dataset: &D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_point: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    ) -> ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>
    where
        D: Dataset,
    {
        let mut nearest_id = entry_point.vector;
        let mut nearest_distance = entry_point.distance;
        let mut updated = true;

        while updated {
            updated = false;

            for neighbor in self.neighbors(nearest_id) {
                let external_id = self.get_external_id(neighbor);
                let distance_neighbor =
                    query_evaluator.compute_distance(dataset.get(external_id as VectorId));

                if distance_neighbor < nearest_distance {
                    nearest_distance = distance_neighbor;
                    nearest_id = neighbor;
                    updated = true;
                }
            }
        }

        ScoredItemGeneric {
            distance: nearest_distance,
            vector: nearest_id,
        }
    }

    /// Performs a greedy search on the graph to find the top `k` nearest neighbors.
    /// It uses a beam search-like approach, maintaining a list of candidates to visit (`ef`)
    /// and returning the `k` best results found.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `starting_node`: The candidate from which the search begins.
    /// * `query_evaluator`: An evaluator that can compute distances to the query.
    /// * `k`: The number of nearest neighbors to return.
    /// * `ef`: The size of the dynamic candidate list during the search.
    ///
    /// # Returns
    /// A `Vec` containing tuples of `(distance, id)` for the `k` nearest neighbors.
    #[must_use]
    fn greedy_search_topk<'e, D>(
        &self,
        dataset: &'e D,
        starting_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        k: usize,
        ef: usize,
    ) -> Vec<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
    {
        let top_candidates =
            self.search_candidates(dataset, starting_node, query_evaluator, ef, Some(k));

        let mut top_k = top_candidates.into_sorted_vec();
        top_k.truncate(k);
        top_k
    }

    #[must_use]
    fn search_candidates<'e, D>(
        &self,
        dataset: &'e D,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        ef: usize,
        k: Option<usize>,
    ) -> BinaryHeap<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>
    where
        D: Dataset + Sync,
    {
        let k = k.unwrap_or(0); // Default to 0 if k is not provided. Used by insertions when we don't need to keep track of the top k candidates, but just want to explore the neighborhood.

        // max-heap: We want to substitute worst result with a better one
        let mut top_candidates: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        // min-heap: We want to extract best candidate first to visit it
        let mut candidates: BinaryHeap<
            Reverse<ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>>,
        > = BinaryHeap::with_capacity(ef);

        let mut visited_table = create_visited_set(dataset.len(), ef);

        top_candidates.push(entry_node);
        candidates.push(Reverse(entry_node));

        visited_table.insert(entry_node.vector);

        while let Some(Reverse(node)) = candidates.pop() {
            let id_candidate = node.vector;
            let distance_candidate = node.distance;

            if top_candidates.len() >= k // Ensure we have enough candidates
                && distance_candidate > top_candidates.peek().unwrap().distance
            // Is the best candidate is worse than the worst in top_candidates?
            {
                break;
            }

            self.process_neighbors(
                dataset,
                self.neighbors(id_candidate),
                &mut visited_table,
                query_evaluator,
                |dis_neigh, neighbor| {
                    add_neighbor_to_heaps(
                        &mut candidates,
                        &mut top_candidates,
                        ScoredItemGeneric {
                            distance: dis_neigh,
                            vector: neighbor,
                        },
                        ef,
                    );
                },
            )
        }
        top_candidates
    }

    /// Processes the neighbors of a node.
    ///
    /// This function iterates through the neighbors of a given node, computes their distances
    /// to the query, and uses a callback function to add them to the candidate heaps.
    /// It uses a `visited_table` to avoid processing the same node multiple times.
    ///
    /// # Arguments
    /// * `dataset`: The dataset containing the vectors.
    /// * `neighbors`: An iterator over the local IDs of the neighbors to process.
    /// * `visited_table`: A `HashSet` to keep track of visited node IDs.
    /// * `query_evaluator`: An evaluator that can compute distances to the query.
    /// * `add_distances_fn`: A callback function that takes `(distance, id)` and adds the neighbor to the candidate heaps.
    fn process_neighbors<'e, D, F>(
        &self,
        dataset: &D,
        neighbors: impl Iterator<Item = usize>,
        visited_table: &mut dyn VisitedSet,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        mut add_distances_fn: F,
    ) where
        D: Dataset,
        F: FnMut(<D::Encoder as VectorEncoder>::Distance, usize),
    {
        for neighbor_local_id in neighbors {
            if !visited_table.contains(neighbor_local_id) {
                visited_table.insert(neighbor_local_id);

                let external_id = self.get_external_id(neighbor_local_id) as VectorId;
                let distance_neighbor = query_evaluator.compute_distance(dataset.get(external_id));
                add_distances_fn(distance_neighbor, neighbor_local_id);
            }
        }
    }
}

/// A representation of a graph where the adjacency lists of the nodes are stored spanning a variable length
/// portion of a vector.
/// A vector of offsets is used to indicate the start of each node's neighbors in the neighbors node.
/// Node ids are represented as `u32` but they are returned as usize ones.
///
/// # Fields
/// - `neighbors`: A list of all neighbors for nodes in the graph. The neighbors for each node
///   are stored in a contiguous block.
/// - `offsets`: An index mapping each node ID to its starting position in the `neighbors` list.
///   The `offsets[node_id]` provides the starting index in `neighbors` where the neighbors of
///   the vector with `node_id` begin.
///
#[derive(Serialize, Deserialize)]
pub struct Graph {
    neighbors: Box<[u32]>, // Compact array of neighbor node IDs
    offsets: Box<[usize]>,
    ids_mapping: Option<Box<[usize]>>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_nodes: usize,
}

impl Default for Graph {
    fn default() -> Self {
        Graph {
            neighbors: Box::new([]),
            offsets: Box::new([]),
            ids_mapping: None,
            max_degree: 0,
            n_nodes: 0,
        }
    }
}

impl GraphTrait for Graph {
    #[inline]
    fn neighbors<'a>(&'a self, id: usize) -> impl Iterator<Item = usize> + 'a {
        let start = self.offsets[id];
        let end = self.offsets[id + 1];
        self.neighbors[start..end].iter().map(|&u| u as usize)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.neighbors.len()
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping[id]
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<u32>();
        let offsets_size = self.offsets.len() * std::mem::size_of::<usize>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        neighbors_size + offsets_size + ids_mapping_size
    }
}

impl From<GrowableGraph> for Graph {
    /// Converts a `GrowableGraph` into a compact `Graph` by removing padding.
    fn from(growable_graph: GrowableGraph) -> Self {
        let n_nodes = growable_graph.n_nodes();
        let max_degree = growable_graph.max_degree();

        let mut neighbors = Vec::with_capacity(growable_graph.neighbors.len());
        let mut offsets = Vec::with_capacity(n_nodes + 1);

        offsets.push(0);
        for v in 0..n_nodes {
            let start = v * max_degree;
            let end = start + max_degree;
            neighbors.extend(
                growable_graph.neighbors[start..end]
                    .iter()
                    .filter_map(|&opt| opt.into_option()),
            );
            offsets.push(neighbors.len());
        }

        let final_mapping = growable_graph
            .ids_mapping
            .map(|mapping| mapping.into_boxed_slice());

        Graph {
            neighbors: neighbors.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            ids_mapping: final_mapping,
            max_degree,
            n_nodes,
        }
    }
}

/// A representation of a graph where the adjacency lists of the nodes are stored in a fixed degree format.
/// If a node's degree is less than the maximum degree, it is padded with `None` values.
/// None values are represented as `usize::MAX`. The nodes ids are in the range `[0, len)`
/// Node ids are represented as `u32` but they are returned as usize ones.
/// Moreover, the largest value is reserved. This means that we allow a
/// maximum of `u32::MAX - 1` nodes.
///
/// # Fields
/// - `neighbors`: A list of all neighbors for vectors in the graph. The neighbors for each vector
///   are stored in a contiguous block.
/// - `max_degree`: The maximum degree of any node in the graph.
/// - `n_edges`: The number of edges in the graph.
/// - `n_nodes`: The number of nodes in the graph.
///
#[derive(Serialize, Deserialize)]
pub struct GraphFixedDegree {
    neighbors: Box<[Optioned<u32>]>, // Using Optioned<u32> to represent neighbors, where None is represented by u32::MAX
    ids_mapping: Option<Box<[usize]>>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_edges: usize,
    n_nodes: usize,
}

impl Default for GraphFixedDegree {
    fn default() -> Self {
        GraphFixedDegree {
            neighbors: Box::new([]),
            ids_mapping: None, // No mapping by default
            max_degree: 0,
            n_edges: 0,
            n_nodes: 0,
        }
    }
}

impl GraphTrait for GraphFixedDegree {
    #[inline]
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a {
        let start = u * self.max_degree;
        let end = start + self.max_degree;
        self.neighbors[start..end]
            .iter()
            .take_while(|&opt| opt.is_some())
            .map(|opt| opt.unwrap() as usize)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping[id]
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<Optioned<u32>>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        neighbors_size + ids_mapping_size
    }
}

impl From<GrowableGraph> for GraphFixedDegree {
    /// Converts a `GrowableGraph` into a fixed-degree `GraphFixedDegree` (preserves padding).
    fn from(growable_graph: GrowableGraph) -> Self {
        let ids_mapping = growable_graph
            .ids_mapping
            .map(|mapping| mapping.into_boxed_slice());

        GraphFixedDegree {
            neighbors: growable_graph.neighbors.into_boxed_slice(),
            ids_mapping,
            max_degree: growable_graph.max_degree,
            n_edges: growable_graph.n_edges,
            n_nodes: growable_graph.n_nodes,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct GrowableGraph {
    neighbors: Vec<Optioned<u32>>, // Using Optioned<u32> to represent neighbors, where None is represented by u32::MAX
    ids_mapping: Option<Vec<usize>>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_edges: usize,
    n_nodes: usize,
    inserted_nodes: usize, // Number of nodes that have been actually inserted
}

impl Default for GrowableGraph {
    fn default() -> Self {
        GrowableGraph {
            neighbors: Vec::new(),
            max_degree: 0,
            ids_mapping: None, // No mapping by default
            n_edges: 0,
            n_nodes: 0,
            inserted_nodes: 0, // No nodes inserted yet
        }
    }
}

impl GraphTrait for GrowableGraph {
    #[inline]
    fn neighbors<'a>(&'a self, u: usize) -> impl Iterator<Item = usize> + 'a {
        let start = u * self.max_degree;
        let end = start + self.max_degree;
        self.neighbors[start..end]
            .iter()
            .take_while(|&opt| opt.is_some())
            .map(|opt| opt.unwrap() as usize)
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn max_degree(&self) -> usize {
        self.max_degree
    }

    #[inline]
    fn n_edges(&self) -> usize {
        self.n_edges
    }

    #[inline]
    fn get_external_id(&self, id: usize) -> usize {
        if let Some(mapping) = &self.ids_mapping {
            if id >= mapping.len() {
                panic!("ID out of bounds: {}", id);
            }
            mapping[id]
        } else {
            id
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<Optioned<u32>>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        neighbors_size + ids_mapping_size
    }
}

impl From<Graph> for GrowableGraph {
    fn from(graph: Graph) -> Self {
        let max_degree = graph.max_degree;
        let n_nodes = graph.n_nodes;
        let mut neighbors = Vec::with_capacity(n_nodes * max_degree);

        for v in 0..n_nodes {
            let start = graph.offsets[v];
            let end = graph.offsets[v + 1];
            let slice = &graph.neighbors[start..end];
            for &nbr in slice {
                neighbors.push(Optioned::some(nbr));
            }
            let pad = max_degree.saturating_sub(slice.len());
            neighbors.extend((0..pad).map(|_| Optioned::none()));
        }

        let ids_mapping = graph.ids_mapping.map(|mapping| mapping.into_vec());

        GrowableGraph {
            neighbors,
            ids_mapping,
            max_degree,
            n_edges: graph.neighbors.len(),
            n_nodes,
            inserted_nodes: n_nodes,
        }
    }
}

impl From<GraphFixedDegree> for GrowableGraph {
    fn from(graph: GraphFixedDegree) -> Self {
        let ids_mapping = graph.ids_mapping.map(|mapping| mapping.into_vec());

        GrowableGraph {
            neighbors: graph.neighbors.into_vec(),
            ids_mapping,
            max_degree: graph.max_degree,
            n_edges: graph.n_edges,
            n_nodes: graph.n_nodes,
            inserted_nodes: graph.n_nodes,
        }
    }
}

impl GrowableGraph {
    /// Creates a new `GrowableGraph` with the specified maximum degree.
    #[must_use]
    pub fn with_max_degree(max_degree: usize) -> Self {
        GrowableGraph {
            neighbors: Vec::new(),
            ids_mapping: None, // No mapping by default
            max_degree,
            n_edges: 0,
            n_nodes: 0,
            inserted_nodes: 0, // No nodes inserted yet
        }
    }

    /// Returns the number of nodes that have been inserted into the graph.
    #[must_use]
    #[inline]
    pub fn inserted_nodes(&self) -> usize {
        self.inserted_nodes
    }

    /// Advances the count of inserted nodes by a given amount.
    /// This is used by the parallel builder to update the state after a batch is processed.
    pub fn advance_inserted_nodes(&mut self, count: usize) {
        self.inserted_nodes += count;
    }

    /// Pre-allocates space for a fixed number of nodes.
    pub fn reserve(&mut self, n_expected_nodes: usize) {
        self.neighbors = vec![Optioned::none(); n_expected_nodes * self.max_degree];
        self.n_nodes = n_expected_nodes; // The graph now has a fixed capacity
        self.ids_mapping = None; // No mapping by default
    }

    /// Sets the ID mapping for the graph, converting local IDs to external/original IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the mapping length does not match the number of nodes in the graph.
    pub fn set_mapping(&mut self, mapping: Vec<usize>) -> Result<(), String> {
        if mapping.len() != self.n_nodes {
            return Err(format!(
                "Mapping length mismatch: got {}, expected {}",
                mapping.len(),
                self.n_nodes
            ));
        }
        self.ids_mapping = Some(mapping);
        Ok(())
    }

    /// A version of push for the parallel builder that accepts pre-computed reverse links.
    pub fn push_with_precomputed_reverse_links(
        &mut self,
        external_id: Option<usize>,
        neighbors: &[usize],
        local_id: usize,
        reverse_links: &[(usize, Vec<usize>)], // (neighbor_id, new_neighbor_list_for_it)
    ) {
        let new_node_local_id = local_id;

        // Add forward links
        let start = new_node_local_id * self.max_degree;
        for (i, &neighbor) in neighbors.iter().enumerate() {
            self.neighbors[start + i] = Optioned::some(neighbor as u32);
        }
        self.n_edges += neighbors.len();

        if let Some(vec_id) = external_id {
            if let Some(mapping) = self.ids_mapping.as_mut() {
                if new_node_local_id >= mapping.len() {
                    panic!(
                        "Attempted to write to local_id {} but ids_mapping len is {}",
                        new_node_local_id,
                        mapping.len()
                    );
                }
                mapping[new_node_local_id] = vec_id;
            } else {
                panic!("Attempted to set external ID for a graph without an ID mapping.");
            }
        } else {
            // If no external ID is provided, we assume the local ID is the external ID
            if let Some(mapping) = self.ids_mapping.as_mut() {
                if new_node_local_id >= mapping.len() {
                    panic!(
                        "Attempted to write to local_id {} but ids_mapping len is {}",
                        new_node_local_id,
                        mapping.len()
                    );
                }
                mapping[new_node_local_id] = new_node_local_id;
            }
        }

        // Add pre-computed reverse links
        for (neighbor_id, new_neighbor_list) in reverse_links {
            let start = *neighbor_id * self.max_degree;
            for (i, &n) in new_neighbor_list.iter().enumerate() {
                self.neighbors[start + i] = Optioned::some(n as u32);
            }
            // Pad with None
            for i in new_neighbor_list.len()..self.max_degree {
                self.neighbors[start + i] = Optioned::none();
            }
        }
    }

    pub fn precompute_reverse_links<'e, D>(
        &self,
        dataset: &'e D,
        node_to_insert_local_id: usize,
        forward_neighbors: &[usize],
    ) -> Vec<(usize, Vec<usize>)>
    // (neighbor_local_id, new_neighbor_list_for_it)
    where
        D: Dataset + Sync,
    {
        let mut reverse_links_data = Vec::with_capacity(forward_neighbors.len());

        for &neighbor_local_id in forward_neighbors {
            // The "query" for the heuristic is the neighbor itself, whose neighbor list we are updating.
            let neighbor_external_id = self.get_external_id(neighbor_local_id) as VectorId;
            let neighbor_query_eval = dataset
                .encoder()
                .vector_evaluator(dataset.get(neighbor_external_id));

            // 1. Build a max-heap containing the neighbor's current neighbors and the new node.
            //    The distances are all relative to the neighbor.
            let mut closest_vectors = BinaryHeap::<
                ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
            >::new();

            // Add its current neighbors
            for local_id in self.neighbors(neighbor_local_id) {
                let external_id = self.get_external_id(local_id) as VectorId;
                let dist = neighbor_query_eval.compute_distance(dataset.get(external_id));
                closest_vectors.push(ScoredItemGeneric {
                    distance: dist,
                    vector: local_id,
                });
            }

            // Add the new reverse link (the node we are inserting)
            let node_to_insert_external_id =
                self.get_external_id(node_to_insert_local_id) as VectorId;
            let dist_to_inserted_node =
                neighbor_query_eval.compute_distance(dataset.get(node_to_insert_external_id));
            closest_vectors.push(ScoredItemGeneric {
                distance: dist_to_inserted_node,
                vector: node_to_insert_local_id,
            });

            // 2. Use the robust `shrink_neighbor_list` heuristic to prune the list.
            let new_neighbor_list =
                self.shrink_neighbor_list(dataset, &mut closest_vectors, self.max_degree);

            reverse_links_data.push((neighbor_local_id, new_neighbor_list));
        }
        reverse_links_data
    }

    pub fn shrink_neighbor_list<'e, D>(
        &self,
        dataset: &'e D,
        closest_vectors: &mut BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        >,
        max_size: usize,
    ) -> Vec<usize>
    where
        D: Dataset + Sync,
    {
        if closest_vectors.len() <= max_size {
            return closest_vectors
                .iter()
                .map(|candidate| candidate.vector)
                .collect();
        }

        let mut min_heap = from_max_heap_to_min_heap(closest_vectors);
        let mut new_closest_vectors: BinaryHeap<
            ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        > = BinaryHeap::new();

        while let Some(node) = min_heap.pop() {
            let node1 = node.0;
            let mut keep_node_1 = true;

            // The robust pruning heuristic from the paper.
            // For each candidate, check if it is closer to the query than it is to any
            // other candidate already in the result set.
            for node2 in new_closest_vectors.iter() {
                let node1_external = self.get_external_id(node1.vector) as VectorId;
                let node2_external = self.get_external_id(node2.vector) as VectorId;
                let node1_eval = dataset
                    .encoder()
                    .vector_evaluator(dataset.get(node1_external));
                let dist_node_1_node2 = node1_eval.compute_distance(dataset.get(node2_external));
                if dist_node_1_node2 < node1.distance {
                    keep_node_1 = false;
                    break;
                }
            }

            if keep_node_1 {
                new_closest_vectors.push(node1);
                if new_closest_vectors.len() >= max_size {
                    return new_closest_vectors.iter().map(|c| c.vector).collect();
                }
            }
        }

        // Return the IDs of the closest vectors
        new_closest_vectors
            .iter()
            .map(|candidate| candidate.vector)
            .collect()
    }

    /// Finds and prunes neighbors for a new node and computes the necessary reverse links.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `Vec<usize>`: The pruned forward neighbors for the new node.
    /// - `Vec<(usize, Vec<usize>)>`: The pre-computed reverse links for existing neighbors.
    /// - `ScoredItemGeneric`: The best candidate found, to be used as the entry point for the next lower level.
    #[must_use]
    pub fn find_and_prune_neighbors<'e, D>(
        &self,
        dataset: &'e D,
        query_evaluator: &<D::Encoder as VectorEncoder>::Evaluator<'e>,
        entry_node: ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
        ef_construction: usize,
        m: usize,
        future_local_id: usize,
    ) -> (
        Vec<usize>,
        Vec<(usize, Vec<usize>)>,
        ScoredItemGeneric<<D::Encoder as VectorEncoder>::Distance, usize>,
    )
    where
        D: Dataset + Sync,
    {
        // 1. Get candidate neighbors
        let mut neighbors_nodes =
            self.search_candidates(dataset, entry_node, query_evaluator, ef_construction, None);

        // The new entry point for the next level is the best candidate we found.
        let new_entry_node = *neighbors_nodes.peek().unwrap();

        // 2. Prune with heuristic
        let forward_neighbors = self.shrink_neighbor_list(dataset, &mut neighbors_nodes, m);

        // 3. Compute reverse links with the PRUNED list
        let reverse_links =
            self.precompute_reverse_links(dataset, future_local_id, &forward_neighbors);

        (forward_neighbors, reverse_links, new_entry_node)
    }
}
