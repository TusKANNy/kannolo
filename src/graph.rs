use std::cmp::Reverse;
use std::collections::BinaryHeap;

use optional::Optioned;
use serde::{Deserialize, Serialize};

use crate::hnsw_utils::{add_neighbor_to_heaps, from_max_heap_to_min_heap, Candidate};
use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::visited_set::set::{create_visited_set, VisitedSet};
use crate::{Dataset, DotProduct, EuclideanDistance, Float};

/// A trait that defines the common interface for different graph implementations.
///
/// This allows graph indexes to be generic over the specific graph storage strategy.
pub trait GraphTrait {
    /// Creates a new, empty graph.
    fn new() -> Self;

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

    /// Creates a new graph from a `GrowableGraph`.
    /// This is typically used to convert a temporary, mutable graph used during the build process
    /// into a final, more compact and immutable graph for searching.
    #[must_use]
    fn from_growable_graph(growable_graph: &GrowableGraph) -> Self;

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
    /// The best `Candidate` found during the search.
    #[must_use]
    fn greedy_search_nearest<'a, Q, D, E>(
        &self,
        dataset: &D,
        query_evaluator: &E,
        entry_point: Candidate,
    ) -> Candidate
    where
        Q: Quantizer<DatasetType = D>,
        D: Dataset<Q>,
        E: QueryEvaluator<'a, Q = Q>,
    {
        let mut nearest_id = entry_point.id_vec();
        let mut nearest_distance = entry_point.distance();
        let mut updated = true;

        while updated {
            updated = false;

            for neighbor in self.neighbors(nearest_id) {
                let external_id = self.get_external_id(neighbor);
                let distance_neighbor = query_evaluator.compute_distance(dataset, external_id);

                if distance_neighbor < nearest_distance {
                    nearest_distance = distance_neighbor;
                    nearest_id = neighbor;
                    updated = true;
                }
            }
        }

        Candidate(nearest_distance, nearest_id)
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
    fn greedy_search_topk<'a, D, Q, E>(
        &self,
        dataset: &'a D,
        starting_node: Candidate,
        query_evaluator: &E,
        k: usize,
        ef: usize,
    ) -> Vec<(f32, usize)>
    where
        D: Dataset<Q> + Sync,
        Q: Quantizer<DatasetType = D> + 'a,
        E: QueryEvaluator<'a, Q = Q>, // tie evaluator’s Q = our Q
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>
            + Float,
    {
        let top_candidates =
            self.search_candidates(dataset, starting_node, query_evaluator, ef, Some(k));
        // Collect the top candidates from the max-heap
        let mut top_k = top_candidates
            .iter()
            .map(|candidate| (candidate.distance(), candidate.id_vec()))
            .collect::<Vec<_>>();

        // Sort the top k candidates by distance
        top_k.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Truncate the top_k to keep only the k best candidates
        top_k.truncate(k);

        top_k
    }

    #[must_use]
    fn search_candidates<'a, D, Q, E>(
        &self,
        dataset: &'a D,
        entry_node: Candidate,
        query_evaluator: &E,
        ef: usize,
        k: Option<usize>,
    ) -> BinaryHeap<Candidate>
    where
        Q: Quantizer<DatasetType = D> + 'a,
        D: Dataset<Q> + Sync,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>
            + Float
            + 'a,
        E: QueryEvaluator<'a, Q = Q>, // tie evaluator’s Q = our Q
    {
        let k = k.unwrap_or(0); // Default to 0 if k is not provided

        // max-heap: We want to substitute worst result with a better one
        let mut top_candidates: BinaryHeap<Candidate> = BinaryHeap::new();

        // min-heap: We want to extract best candidate first to visit it
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();

        let mut visited_table = create_visited_set(dataset.len(), ef);

        top_candidates.push(entry_node);
        candidates.push(Reverse(entry_node));

        visited_table.insert(entry_node.id_vec());

        while let Some(Reverse(node)) = candidates.pop() {
            let id_candidate = node.id_vec();
            let distance_candidate = node.distance();

            if top_candidates.len() >= k // Ensure we have enough candidates
                && distance_candidate > top_candidates.peek().unwrap().distance()
            // Is the best candidate is worse than the worst in top_candidates?
            {
                break;
            }

            self.process_neighbors(
                dataset,
                self.neighbors(id_candidate),
                &mut *visited_table,
                query_evaluator,
                |dis_neigh, neighbor| {
                    add_neighbor_to_heaps(
                        &mut candidates,
                        &mut top_candidates,
                        Candidate(dis_neigh, neighbor),
                        ef,
                    );
                },
            )
        }
        top_candidates
    }

    /// Sets the ID mapping for the graph.
    /// This mapping is used to convert local IDs to external/original IDs.
    fn set_mapping(&mut self, mapping: Vec<usize>);

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
    fn process_neighbors<'a, D, Q, E, F>(
        &self,
        dataset: &D,
        neighbors: impl Iterator<Item = usize>,
        visited_table: &mut dyn VisitedSet,
        query_evaluator: &E,
        mut add_distances_fn: F,
    ) where
        D: Dataset<Q>,
        Q: Quantizer<DatasetType = D>,
        E: QueryEvaluator<'a, Q = Q>,
        F: FnMut(f32, usize),
    {
        // Stores the IDs of the neighbors whose distances will be computed
        let mut local_ids: [usize; 4] = [0; 4];

        let mut counter = 0;
        for neighbor_local_id in neighbors {
            if !visited_table.contains(neighbor_local_id) {
                visited_table.insert(neighbor_local_id);

                local_ids[counter] = neighbor_local_id; // Store the LOCAL ID
                counter += 1;

                if counter == 4 {
                    // Get external IDs just for the distance computation
                    let external_ids_for_dist: [usize; 4] = [
                        self.get_external_id(local_ids[0]),
                        self.get_external_id(local_ids[1]),
                        self.get_external_id(local_ids[2]),
                        self.get_external_id(local_ids[3]),
                    ];
                    let distances = query_evaluator
                        .compute_four_distances(&dataset, external_ids_for_dist.iter().copied());
                    for (dis_neigh, &neighbor) in distances.zip(local_ids.iter()) {
                        add_distances_fn(dis_neigh, neighbor);
                    }
                    counter = 0;
                }
            }
        }

        // Add the remaining neighbors, if there are any left
        for &local_id in local_ids.iter().take(counter) {
            let distance_neighbor: f32 =
                query_evaluator.compute_distance(dataset, self.get_external_id(local_id));
            add_distances_fn(distance_neighbor, local_id);
        }
    }
}

/// A representation of a graph where the adjacency lists of the nodes are stored spanning a variable length
/// portion of a vector.
/// A vector of offsets is used to indicate the start of each node's neighbors in the neighbors node.
/// Nodes ids are represented as `u32` but thei are returned as usize ones.
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
    neighbors: Box<[u32]>, // Using usize to represent neighbors, where `None` is represented by usize::MAX
    offsets: Box<[usize]>,
    ids_mapping: Option<Box<[usize]>>, // This is used to map the internal IDs to external IDs
    max_degree: usize,
    n_nodes: usize,
}

impl GraphTrait for Graph {
    fn new() -> Self {
        Graph {
            neighbors: Box::new([]),
            offsets: Box::new([]),
            ids_mapping: None,
            max_degree: 0,
            n_nodes: 0,
        }
    }

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
    fn set_mapping(&mut self, mapping: Vec<usize>) {
        self.ids_mapping = Some(mapping.into_boxed_slice());
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

    /// Creates a new `Graph` from a `GrowableGraph`.
    /// This function converts a `GrowableGraph` into a graph by removing the padding
    fn from_growable_graph(growable_graph: &GrowableGraph) -> Self {
        let n_nodes = growable_graph.n_nodes();
        let max_degree = growable_graph.max_degree();

        let mut neighbors = Vec::with_capacity(growable_graph.neighbors.len());
        let mut offsets = Vec::with_capacity(n_nodes + 1);

        offsets.push(0); // Start with the first offset at 0
        for v in 0..n_nodes {
            let start = v * max_degree;
            let end = start + max_degree;
            // Collect only the non-None neighbors
            let cur_neighbors: Vec<u32> = growable_graph.neighbors[start..end]
                .iter()
                .filter_map(|&opt| {
                    if opt.is_some() {
                        Some(opt.unwrap())
                    } else {
                        None
                    }
                })
                .collect();
            neighbors.extend(cur_neighbors);
            offsets.push(neighbors.len());
        }

        let final_mapping = if let Some(mapping) = &growable_graph.ids_mapping {
            Some(mapping.clone().into_boxed_slice())
        } else {
            None
        };

        Graph {
            neighbors: neighbors.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
            ids_mapping: final_mapping,
            max_degree,
            n_nodes,
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<u32>();
        let offsets_size = self.offsets.len() * std::mem::size_of::<usize>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        let total_size = neighbors_size + offsets_size + ids_mapping_size;
        total_size
    }
}

/// A representation of a graph where the adjacency lists of the nodes are stored in a fixed degree format.
/// If a node's degree is less than the maximum degree, it is padded with `None` values.
/// None values are represented as `usize::MAX`. The nodes ids are in the range `[0, len)`
/// Nodes ids are represented as `u32` but thei are returned as usize ones.
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

impl GraphTrait for GraphFixedDegree {
    fn new() -> Self {
        GraphFixedDegree {
            neighbors: Box::new([]),
            ids_mapping: None, // No mapping by default
            max_degree: 0,
            n_edges: 0,
            n_nodes: 0,
        }
    }

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
    fn set_mapping(&mut self, mapping: Vec<usize>) {
        self.ids_mapping = Some(mapping.into_boxed_slice());
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

    /// Creates a new `Graph` from a `GrowableGraph`.
    /// This function converts a `GrowableGraph` into a graph by removing the padding
    fn from_growable_graph(growable_graph: &GrowableGraph) -> Self {
        let ids_mapping = if let Some(mapping) = &growable_graph.ids_mapping {
            Some(mapping.clone().into_boxed_slice())
        } else {
            None
        };

        GraphFixedDegree {
            neighbors: growable_graph.neighbors.clone().into_boxed_slice(),
            ids_mapping,
            max_degree: growable_graph.max_degree,
            n_edges: growable_graph.n_edges,
            n_nodes: growable_graph.n_nodes,
        }
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<Optioned<u32>>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        let total_size = neighbors_size + ids_mapping_size;
        total_size
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

impl GraphTrait for GrowableGraph {
    fn new() -> Self {
        GrowableGraph {
            neighbors: Vec::new(),
            max_degree: 0,
            ids_mapping: None, // No mapping by default
            n_edges: 0,
            n_nodes: 0,
            inserted_nodes: 0, // No nodes inserted yet
        }
    }

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

    fn from_growable_graph(growable_graph: &GrowableGraph) -> Self {
        GrowableGraph {
            neighbors: growable_graph.neighbors.clone(),
            ids_mapping: growable_graph.ids_mapping.clone(),
            max_degree: growable_graph.max_degree,
            n_edges: growable_graph.n_edges,
            n_nodes: growable_graph.n_nodes,
            inserted_nodes: growable_graph.inserted_nodes,
        }
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

    /// Sets the ID mapping for this graph.
    /// The mapping converts local IDs (indices) to external/original IDs.
    fn set_mapping(&mut self, mapping: Vec<usize>) {
        self.ids_mapping = Some(mapping);
    }

    fn get_space_usage_bytes(&self) -> usize {
        let neighbors_size = self.neighbors.len() * std::mem::size_of::<Optioned<u32>>();
        let ids_mapping_size = self
            .ids_mapping
            .as_ref()
            .map_or(0, |mapping| mapping.len() * std::mem::size_of::<usize>());

        let total_size = neighbors_size + ids_mapping_size;
        total_size
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

    pub fn set_mapping(&mut self, mapping: Vec<usize>) {
        // Check that the mapping length matches the number of nodes
        if mapping.len() != self.n_nodes {
            panic!("Mapping length does not match the number of nodes in the graph.");
        }
        self.ids_mapping = Some(mapping);
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

    pub fn precompute_reverse_links<'a, D, Q>(
        &self,
        dataset: &'a D,
        node_to_insert_local_id: usize,
        forward_neighbors: &[usize],
    ) -> Vec<(usize, Vec<usize>)>
    // (neighbor_local_id, new_neighbor_list_for_it)
    where
        Q: Quantizer<DatasetType = D> + 'a,
        D: Dataset<Q> + Sync,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>
            + Float
            + 'a,
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <D as Dataset<Q>>::DataType<'a>>,
        <Q as Quantizer>::OutputItem: Float,
    {
        let mut reverse_links_data = Vec::with_capacity(forward_neighbors.len());

        for &neighbor_local_id in forward_neighbors {
            // The "query" for the heuristic is the neighbor itself, whose neighbor list we are updating.
            let neighbor_external_id = self.get_external_id(neighbor_local_id);
            let neighbor_query_eval = dataset.query_evaluator(dataset.get(neighbor_external_id));

            // 1. Build a max-heap containing the neighbor's current neighbors and the new node.
            //    The distances are all relative to the neighbor.
            let mut closest_vectors = BinaryHeap::new();

            // Add its current neighbors
            let neighbors_of_neighbor: Vec<usize> = self.neighbors(neighbor_local_id).collect();

            // Process neighbors in chunks of 4 for batched distance calculation
            for chunk in neighbors_of_neighbor.chunks(4) {
                if chunk.len() == 4 {
                    let external_ids_for_dist: [usize; 4] = [
                        self.get_external_id(chunk[0]),
                        self.get_external_id(chunk[1]),
                        self.get_external_id(chunk[2]),
                        self.get_external_id(chunk[3]),
                    ];
                    let distances = neighbor_query_eval
                        .compute_four_distances(dataset, external_ids_for_dist.iter().copied());

                    for (dist, &local_id) in distances.zip(chunk.iter()) {
                        closest_vectors.push(Candidate(dist, local_id));
                    }
                } else {
                    // Process the remainder chunk (less than 4 elements) individually
                    for &local_id in chunk {
                        let external_id = self.get_external_id(local_id);
                        let dist = neighbor_query_eval.compute_distance(dataset, external_id);
                        closest_vectors.push(Candidate(dist, local_id));
                    }
                }
            }

            // Add the new reverse link (the node we are inserting)
            let node_to_insert_external_id = self.get_external_id(node_to_insert_local_id);
            let dist_to_inserted_node =
                neighbor_query_eval.compute_distance(dataset, node_to_insert_external_id);
            closest_vectors.push(Candidate(dist_to_inserted_node, node_to_insert_local_id));

            // 2. Use the robust `shrink_neighbor_list` heuristic to prune the list.
            let new_neighbor_list =
                self.shrink_neighbor_list(dataset, &mut closest_vectors, self.max_degree);

            reverse_links_data.push((neighbor_local_id, new_neighbor_list));
        }
        reverse_links_data
    }

    pub fn shrink_neighbor_list<'a, D, Q>(
        &self,
        dataset: &'a D,
        closest_vectors: &mut BinaryHeap<Candidate>,
        max_size: usize,
    ) -> Vec<usize>
    where
        Q: Quantizer<DatasetType = D> + 'a,
        D: Dataset<Q> + Sync,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>
            + Float
            + 'a,
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <D as Dataset<Q>>::DataType<'a>>,
        <Q as Quantizer>::OutputItem: Float,
    {
        if closest_vectors.len() <= max_size {
            return closest_vectors
                .iter()
                .map(|candidate| candidate.id_vec())
                .collect();
        }

        let mut min_heap = from_max_heap_to_min_heap(closest_vectors);
        let mut new_closest_vectors: BinaryHeap<Candidate> = BinaryHeap::new();

        while let Some(node) = min_heap.pop() {
            let node1 = node.0;
            let mut keep_node_1 = true;

            // The robust pruning heuristic from the paper.
            // For each candidate, check if it is closer to the query than it is to any
            // other candidate already in the result set.
            for node2 in new_closest_vectors.iter() {
                let dist_node_1_node2 =
                    dataset.compute_distance_by_id(node1.id_vec(), node2.id_vec());
                if dist_node_1_node2 < node1.distance() {
                    keep_node_1 = false;
                    break;
                }
            }

            if keep_node_1 {
                new_closest_vectors.push(node1);
                if new_closest_vectors.len() >= max_size {
                    return new_closest_vectors.iter().map(|c| c.id_vec()).collect();
                }
            }
        }

        // Return the IDs of the closest vectors
        new_closest_vectors
            .iter()
            .map(|candidate| candidate.id_vec())
            .collect()
    }

    /// Finds and prunes neighbors for a new node and computes the necessary reverse links.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `Vec<usize>`: The pruned forward neighbors for the new node.
    /// - `Vec<(usize, Vec<usize>)>`: The pre-computed reverse links for existing neighbors.
    /// - `Candidate`: The best candidate found, to be used as the entry point for the next lower level.
    #[must_use]
    pub fn find_and_prune_neighbors<'a, D, Q>(
        &self,
        dataset: &'a D,
        query_evaluator: &<Q as Quantizer>::Evaluator<'a>,
        entry_node: Candidate,
        ef_construction: usize,
        m: usize,
        future_local_id: usize,
    ) -> (Vec<usize>, Vec<(usize, Vec<usize>)>, Candidate)
    where
        Q: Quantizer<DatasetType = D> + 'a,
        D: Dataset<Q> + Sync,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>
            + Float
            + 'a,
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <D as Dataset<Q>>::DataType<'a>>,
        <Q as Quantizer>::OutputItem: Float,
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
