use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    marker::PhantomData,
    sync::{Mutex, MutexGuard},
};

use crate::Dataset;
use crate::{
    quantizer::{IdentityQuantizer, QueryEvaluator},
    Float,
};
use rand::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{
    add_neighbor_to_heaps, compute_closest_from_neighbors, config_hnsw::ConfigHnsw,
    from_max_heap_to_min_heap, level::Level, Node,
};

/// A builder for constructing an HNSW (Hierarchical Navigable Small World) graph.
///
/// The `HnswBuilder` struct is responsible for creating and managing the HNSW graph. It handles
/// the assignment of levels to nodes, computation of neighbors, and construction of `Level` instances,
/// which store the neighbors of nodes at each level.
///
/// # Fields
///
/// - `probabs_levels`: A vector of probabilities associated with each level of the graph. These probabilities
///    determine the likelihood that a node is assigned to a particular level during graph construction.
///
/// - `neighbors`: A vector of `Mutex<Option<usize>>` representing the neighbor connections between nodes in the graph.
///   Each `Mutex` guards an optional ID, allowing for thread-safe updates during graph construction.
///
/// - `offsets`: A vector of offsets that helps locate the neighbors for each node. The offsets indicate
///   where the neighbors of a node start in the `neighbors` vector, enabling efficient access to neighbor lists.
///
/// - `levels_assigned`: A vector that tracks the level assigned to each node in the graph. The level of a node
///   determines its position in the hierarchical structure and influences the connections it forms with other nodes.
///
/// - `max_level`: A field representing the highest level present in the graph. This value is used
///   during the construction process to determine the maximum level assigned to any node.
///
/// - `entry_vector`: An optional index indicating the entry point node for the graph. This node is used as
///   the starting point for searching potential neighbors when a new vector is inserted into the graph.
///
/// - `cum_num_neigh_per_level`: A vector that holds the cumulative number of neighbors for each level.
///   This helps in determining the total number of neighbors assigned to nodes based on their respective levels.
///
/// - `dataset`: A reference to the dataset containing the actual data vectors for the HNSW graph.
///
/// - `_phantom`: A `PhantomData` marker that indicates the type `Q` is used in the context of the struct,
///    ensuring proper type safety without actually storing a value of type `Q`.
///    This helps the Rust compiler manage safety and type constraints related to `Q`.
pub struct HnswBuilder<'a, D, Q> {
    probabs_levels: Vec<f32>,
    neighbors: Vec<Mutex<Option<usize>>>,
    offsets: Vec<usize>,
    levels_assigned: Vec<u8>,
    max_level: u8,
    entry_vector: Option<usize>,
    cum_num_neigh_per_level: Vec<usize>,
    dataset: &'a D,
    _phantom: PhantomData<&'a Q>,
}

impl<'a, D, Q> HnswBuilder<'a, D, Q>
where
    D: Dataset<Q> + Sync,
    // The IdentityQuantizer enforces that the InputItem and OutputItem of the quantizer are of the same type.
    // This is essential because:
    // - The `query_evaluator` function in the `Dataset` trait requires a vector of type `QueryType`,
    //   where the `InputItem` must implement the `Float` trait.
    // - The `get` function of the dataset returns a vector of type `DataType`, where the elements are
    //   of type `OutputItem` from the quantizer.
    // By implementing the IdentityQuantizer trait, we ensure that `InputItem` and `OutputItem` are the same type,
    // allowing the dataset's `get` function to directly provide data that can be used by `query_evaluator`
    Q: IdentityQuantizer<DatasetType = D, T: Float> + Sync,

    // This constraint is necessary because the vector returned by the dataset's "get" function is of type Datatype.
    // The query evaluator, however, requires a vector of type Querytype.
    Q::Evaluator<'a>: QueryEvaluator<'a, QueryType = <D as Dataset<Q>>::DataType<'a>>,
    <Q as IdentityQuantizer>::T: 'a,
{
    /// Constructs a new `HnswBuilder` instance.
    ///
    /// This function initializes a new `HnswBuilder` that is ready to build an HNSW (Hierarchical Navigable Small World)
    /// graph for the given dataset. The builder is set up with default values for its internal parameters, and the probability
    /// for level assignment is configured based on the number of neighbors per vector.
    ///
    /// # Parameters
    ///
    /// - `num_neighbors_per_vec`: The number of neighbors each vector should have. This value is used to determine the default
    ///   probabilities for level assignment in the graph.
    /// - `dataset`: A reference to the dataset containing the data vectors that will be used to construct the HNSW graph.
    ///
    /// # Returns
    ///
    /// Returns a new `HnswBuilder` instance with initialized internal structures and default settings.
    pub fn new(num_neighbors_per_vec: usize, dataset: &'a D) -> Self {
        let mut hnsw_builder = Self {
            probabs_levels: vec![],
            neighbors: vec![],
            offsets: vec![0],
            levels_assigned: vec![],
            max_level: 0,
            entry_vector: None,
            cum_num_neigh_per_level: vec![],
            dataset,
            _phantom: PhantomData,
        };
        let m = num_neighbors_per_vec as f32;
        hnsw_builder.set_default_probas(1.0 / m.ln(), m as usize);
        hnsw_builder
    }

    /// Computes the neighbors for each vector in the dataset and organizes them into levels.
    ///
    /// This function calculates the neighbors for each vector in the dataset according to the HNSW graph
    /// structure. It assigns levels to vectors, shuffles and processes them in parallel to compute their
    /// neighbors, and then organizes the resulting data into `Level` instances.
    ///
    /// The process involves:
    /// 1. Assigning levels to each vector.
    /// 2. Counting the number of vectors at each level.
    /// 3. Ordering vector IDs by their assigned levels.
    /// 4. Utilizing multiple threads to find the neighbors for each vector in parallel,
    ///    ensuring thread safety with mutexes and locks.
    /// 5. Returning the computed levels and a mapping of new IDs to original IDs.
    ///
    /// # Parameters
    ///
    /// - `config`: A reference to a `ConfigHnsw` structure that contains configuration settings for
    ///   computing the HNSW graph.
    ///
    /// # Returns
    ///
    /// This function returns a triple:
    ///
    /// - `Vec<Level>`: A vector of `Level` instances representing the neighbors for nodes at each level
    ///    of the HNSW graph.
    /// - `Vec<usize>`: A vector where each element maps a new ID assigned to a vector to its original ID.
    /// - `usize`: The entry vector, which is the ID of a node assigned to the maximum level, from which the
    ///   search will start.
    ///
    /// # Example
    ///
    /// Consider a dataset with 100 vectors and an HNSW graph with 3 levels.
    ///
    /// - `num_ids_per_level` might be `[86, 20, 4]`, indicating the number of vectors at each level (0, 1, 2).
    ///
    /// - `ids_per_level` might be organized as follows:
    /// ```rust
    ///    vec![
    ///       // IDs for level 0 (indices 0 to 85)
    ///       0, 1, 3, 6, 7, ..., 98,  
    ///       // IDs for level 1 (indices 86 to 95)
    ///       2, 4, 11, ..., 99,      
    ///       // IDs for level 2 (indices 96 to 99)
    ///       5, 10, 23, 60]         
    ///   ```
    /// During processing:
    /// - For each level, the `begin` and `end` indices are calculated to determine the segment of `ids_per_level`
    ///   relevant for that level.
    ///
    ///   For example:
    /// - At level 2, `end` might be `100`, and `begin` would be `96` (computed as `end - num_ids_per_level[level as usize]`),
    ///   so the IDs in the range from indices `96` to `99` are processed. This results in processing the IDs `[5, 10, 23, 60]`.
    /// - At level 1, `end` is updated to `96`, and `begin` would be `86`, so the IDs from indices `86` to `95` are processed.
    /// - At level 0, `end` is updated to `86`, and `begin` would be `0`, so the IDs from indices `0` to `85` are processed.
    pub fn compute_graph(&mut self, config: &ConfigHnsw) -> (Vec<Level>, Vec<usize>, usize) {
        let num_vectors = self.dataset.len();
        self.assign_level(num_vectors);

        let num_ids_per_level = self.count_num_ids_per_level(num_vectors);
        let ids_per_level = self.order_ids_by_level(&num_ids_per_level, num_vectors);

        let locks: Vec<Mutex<()>> = (0..num_vectors).map(|_| Mutex::new(())).collect();

        let mut rng = StdRng::seed_from_u64(537);
        let mut end = num_vectors;

        for level in (0..=self.max_level).rev() {
            let begin = end - num_ids_per_level[level as usize];
            let mut ids_curr_level: Vec<&usize> =
                ids_per_level.iter().take(end).skip(begin).collect();
            ids_curr_level.shuffle(&mut rng);

            if self.entry_vector.is_none() && level as u8 == self.max_level {
                // it assign as entry_vector the first id that got assigned the highest level
                self.entry_vector = Some(*ids_curr_level[0]);
            }

            ids_curr_level.par_iter().for_each(|&&id| {
                self.compute_neighbors_for_vector(id, level, &locks, config);
            });
            end = begin;
        }

        self.compute_levels()
    }

    /// Computes and returns the levels of the HNSW graph along with mappings from new IDs to original IDs
    /// and the entry vector.
    ///
    /// This function constructs the levels of the HNSW graph by iterating over the levels from the
    /// highest (max level) to the lowest.
    /// For each level, it creates a `Level` instance that stores the neighbors of the nodes present
    /// at that level. During this process, it also maintains two mappings:
    ///
    /// 1. **`new_to_original`:** Maps the newly assigned IDs back to the original IDs,
    ///   enabling access to the original dataset vectors.
    /// 2. **`original_to_new`:** Maps original IDs to their new IDs, allowing quick lookup
    ///   of new IDs based on the original IDs.
    ///
    /// The levels are processed in reverse order (from the highest to the lowest) to ensure proper
    /// level construction. Once all levels are computed, they are reversed to return them in the
    /// correct order, from the lowest (level 0) to the highest level.
    ///
    /// # Returns
    ///
    /// This function returns a triple:
    ///
    /// - `Vec<Level>`: A vector of `Level` instances representing the neighbors at each level in the HNSW graph.
    /// - `Vec<usize>`: A vector mapping new IDs to original IDs, enabling access to the original dataset using new IDs.
    /// - `usize`: The entry vector, which is the ID of a node assigned to the maximum level, from which the search will start.
    #[inline]
    fn compute_levels(&self) -> (Vec<Level>, Vec<usize>, usize) {
        let mut processed_ids: Vec<usize> = Vec::with_capacity(self.dataset.len());

        let mut new_to_original = Vec::with_capacity(self.dataset.len());

        let mut original_to_new: Vec<usize> = vec![0; self.dataset.len()];

        let mut levels: Vec<Level> = Vec::with_capacity((self.max_level + 1) as usize);

        for curr_level in (0..=self.max_level).rev() {
            levels.push(Level::new(
                self,
                curr_level,
                &mut processed_ids,
                &mut new_to_original,
                &mut original_to_new,
            ));
        }

        levels.reverse();

        (
            levels,
            new_to_original,
            original_to_new[self.entry_vector.unwrap()],
        )
    }

    /// Counts the number of vectors assigned to each level in the HNSW graph and returns a vector
    /// representing this distribution.
    ///
    /// This function iterates over all vectors in the dataset and counts how many vectors are assigned
    /// to each level. It creates a vector `num_ids_per_level` where each index corresponds to a level,
    /// and the value at that index represents the number of vectors assigned to that level.
    ///
    /// # Parameters
    ///
    /// - `num_vectors`: The total number of vectors in the dataset.
    ///
    /// # Returns
    ///
    /// - `Vec<usize>`: A vector where each element represents the number of vectors assigned to each
    ///    level in the HNSW graph. The index of the vector corresponds to the level number, and the
    ///    value at that index is the count of vectors at that level.
    ///
    /// # Panics
    ///
    /// This function will panic if it encounters a vector that does not have a level assigned,
    /// which should not happen under normal circumstances as each vector should be assigned a level
    /// before this function is called.
    ///
    /// /// # Example
    /// ```rust
    /// let num_ids_per_level = hnsw_builder.count_num_ids_per_level(100);
    /// // `num_ids_per_level[0]` might be 86, indicating 86 vectors are at level 0.
    /// // `num_ids_per_level[1]` might be 10, indicating 10 vectors are at level 1.
    /// // `num_ids_per_level[2]` might be 4, indicating 4 vectors are at level 2.
    /// ```
    fn count_num_ids_per_level(&self, num_vectors: usize) -> Vec<usize> {
        // The num_ids_per_level vector stores the number of vectors to be added at each level
        let mut num_ids_per_level: Vec<usize> = Vec::with_capacity(self.max_level as usize);

        for id_vec_to_add in 0..num_vectors {
            let level_vec = *self
                .levels_assigned
                .get(id_vec_to_add)
                .expect("Level not assigned");

            while level_vec >= num_ids_per_level.len() as u8 {
                num_ids_per_level.push(0);
            }
            num_ids_per_level[level_vec as usize] += 1;
        }
        num_ids_per_level
    }

    /// Orders the IDs of vectors based on their assigned levels and returns a vector where IDs
    /// are arranged in the order they should be inserted into the HNSW graph.
    ///
    /// This function organizes the vector IDs based on the levels they are assigned to.
    /// It creates a vector `ids_per_level` where the IDs of vectors are ordered by their respective levels.
    /// Vectors from lower levels appear first, followed by those from higher levels. To compute this vector
    /// it uses an offset vector which track the position where the ids for each level should be inserted
    /// in the ids_per_level vector. For each node, using the id it retreives the level assigned to that node and uses
    /// the offsets vector to get the position to be inserted in the ids_per_level
    ///
    /// # Parameters
    ///
    /// - `num_ids_per_level`: A reference to a vector containing the number of vectors assigned to each level. The index
    ///   corresponds to the level, and the value at that index represents the count of vectors at that level.
    /// - `num_vectors`: The total number of vectors in the dataset.
    ///
    /// # Returns
    ///
    /// - `Vec<usize>`: A vector containing the IDs of vectors arranged according to their assigned levels. Vectors at lower
    ///   levels are placed at the beginning of the vector, followed by vectors at higher levels.
    ///
    /// # Example
    ///
    /// If there are 10 vectors distributed across 3 levels with counts `[5, 3, 2]`:
    /// - The `offsets` vector would be `[0, 5, 8]`, indicating the starting positions for each level in the
    ///  `ids_per_level` vector.
    /// - For vectors at level 1, IDs are inserted starting at position 5. After inserting each ID, the offset
    ///   is updated to [0,6,8], so the next ID for level 1 will be placed at position 6.
    ///
    fn order_ids_by_level(&self, num_ids_per_level: &Vec<usize>, num_vectors: usize) -> Vec<usize> {
        // The ids_per_level vector holds the IDs of elements in the order they should be inserted
        // according to their respective levels.
        let mut ids_per_level: Vec<usize> = vec![0; num_vectors];

        // the offsets vectors contains the position in which the next vector for a specific level should be inserted
        // in the ids_per_level vector
        let mut offsets: Vec<usize> = vec![0; num_ids_per_level.len() - 1];

        offsets.push(0);

        for i in 0..num_ids_per_level.len() - 1 {
            offsets[i + 1] = offsets[i] + num_ids_per_level[i];
        }

        for id_vec_to_add in 0..num_vectors {
            let level_vec = *self
                .levels_assigned
                .get(id_vec_to_add)
                .expect("Level not assigned");
            ids_per_level[offsets[level_vec as usize]] = id_vec_to_add;
            offsets[level_vec as usize] += 1;
        }

        ids_per_level
    }

    /// Computes the neighbors for a given vector based on its ID and level in the HNSW graph.
    ///
    /// This function calculates the nearest neighbors of a vector specified by `id_vec` at a particular level `level_vec`.
    /// It performs the following tasks:
    ///
    /// - **Phase 1: Greedy Search in higher levels**
    ///
    ///   The function starts by performing a greedy search from the highest level (`max_level`) down to `curr_level`.
    ///   It iterates through these higher levels to find the closest existing vector to the new vector being added.
    ///   This closest vector is then used as the starting point for the subsequent phase.
    ///
    /// - **Phase 2: Neighbor Computation at insertion levels**
    ///   After identifying the closest vector from the higher levels, the function performs neighbor computation
    ///   starting from `curr_level` and working down to level 0. It does this by calling the `add_links_starting_from`
    ///   function for each of these levels. This function identifies the neighbors for the vector being added at each level,
    ///   and updates the corresponding segment in the `neighbors` field to store the IDs of these neighbors.
    fn compute_neighbors_for_vector(
        &self,
        id_vec: usize,
        level_vec: u8,
        locks: &Vec<Mutex<()>>,
        config: &ConfigHnsw,
    ) {
        // It returns since it's the first node inserted, so there are no neighbors to search
        if id_vec == self.entry_vector.unwrap() {
            return;
        }

        let mut nearest_vec = self.entry_vector.unwrap();

        let vector = self.dataset.get(id_vec);

        let query_evaluator = self.dataset.query_evaluator(vector);

        let mut curr_level = self.max_level;
        let mut dis_nearest_vec = query_evaluator.compute_distance(&self.dataset, nearest_vec);

        {
            let _lock = &locks[id_vec].lock().unwrap();

            while curr_level > level_vec {
                self.greedy_update_nearest(
                    curr_level,
                    &query_evaluator,
                    &mut nearest_vec,
                    &mut dis_nearest_vec,
                );
                curr_level -= 1;
            }
        }

        loop {
            let guard = locks[id_vec].lock().unwrap();

            self.add_links_starting_from(
                &query_evaluator,
                id_vec,
                nearest_vec,
                dis_nearest_vec,
                curr_level,
                locks,
                guard,
                config,
            );

            if curr_level == 0 {
                break;
            }
            curr_level -= 1;
        }
    }

    /// Updates the nearest vector to the vector being added using a greedy approach.
    ///
    /// # Arguments
    /// - `curr_level`: The level in the HNSW graph where the current search for the nearest neighbor is being conducted.
    /// - `query_evaluator`: A reference to an object implementing the `QueryEvaluator` trait.
    ///    This object provides the method to compute the distance between the query vector and each neighbor.
    /// - `nearest_vec`: A mutable reference to a `usize` variable that will be updated to the index of
    ///    the closest neighbor found.
    /// - `dis_nearest_vec`: A mutable reference to a `f32` variable that will be updated to the distance
    ///    of the closest neighbor found.
    ///    
    /// # Description
    ///
    /// The function begins by retrieving the neighbors of the current nearest vector at the specified level. It then uses
    /// the `compute_closest_from_neighbors` function to evaluate these neighbors and determine if any are closer to the
    /// vector being added. The nearest vector and its distance are updated if a closer neighbor is found. This process
    /// continues iteratively until no closer neighbors are found, at which point the function exits.
    fn greedy_update_nearest<E>(
        &self,
        curr_level: u8,
        query_evaluator: &E,
        nearest_vec: &mut usize,
        dis_nearest_vec: &mut f32,
    ) where
        E: QueryEvaluator<'a, Q = Q>, // <= tie evaluator’s Q to builder’s Q
    {
        loop {
            let prec_nearest = *nearest_vec;
            let neighbors = self.get_unlocked_neighbors(*nearest_vec, curr_level);

            compute_closest_from_neighbors(
                self.dataset,
                query_evaluator,
                neighbors.as_slice(),
                nearest_vec,
                dis_nearest_vec,
            );

            if prec_nearest == *nearest_vec {
                return;
            }
        }
    }

    /// Retrieves the neighbors for a vector at a specified level, unlocking the neighbors for access.
    ///
    /// This method collects and returns the neighbors of a vector identified by `id` at the given `level`. The function
    /// first obtains the list of neighbor `Mutex` objects from the specified level, then locks each `Mutex` to access
    /// the neighbor IDs. It filters out any `None` values and collects the remaining neighbor IDs into a vector.
    ///
    /// # Arguments
    /// - `id`: The index of the vector whose neighbors are to be retrieved.
    /// - `level`: The level in the HNSW graph from which to retrieve the neighbors.
    ///
    /// # Returns
    /// A `Vec<usize>` containing the IDs of the neighbors for the specified vector at the given level.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Consider we have an HNSW graph and the neighbors of the vector with id 5 at level 2 are stored as follows:
    /// // vec![Mutex::new(Some(423)), Mutex::new(Some(12)), Mutex::new(Some(42)), Mutex::new(None), Mutex::new(None)]
    ///
    /// let neighbors = hnsw_builder.get_unlocked_neighbors(5, 2);
    ///
    /// // The function will lock each `Mutex` and retrieve the neighbor IDs, filtering out `None` values.
    /// // The returned neighbors vector will be:
    /// // vec![423, 12, 42] // Only valid neighbor IDs are returned.
    /// ```
    #[inline]
    pub fn get_unlocked_neighbors(&self, id: usize, level: u8) -> Vec<usize> {
        self.get_neighbors_from_level(id, level)
            .iter()
            .map(|mutex| mutex.lock().unwrap())
            .filter_map(|neighbor| *neighbor)
            .collect()
    }

    /// This function handles the process of finding the neighbors for a vector being inserted
    /// at a specified level in the HNSW graph and establishing connections with them.
    ///
    /// # Arguments
    ///
    /// - `query_evaluator`: An object that implements the `QueryEvaluator` trait. This object is used to
    ///   evaluate distances between the vector being added and the vectors already present in the graph.
    /// - `id_vec`: The index of the vector being inserted into the HNSW graph.
    /// - `nearest_vec`: The index of the nearest vetor found in the highest levels, used as the starting point
    ///    for neighbor search.
    /// - `dis_nearest_vec`: The distance to the nearest vector found in the highest levels.
    /// - `curr_level`: The current level in the HNSW graph where links are being added.
    /// - `visited_table`: A table tracking the vectors that have already been visited during the search.
    /// - `locks`: A slice of `Mutex` objects used to lock the neighbors of each vector to ensure
    ///    thread safety during updates.
    /// - `guard`: A `MutexGuard` object used to manage the current lock.
    /// - `config`: The configuration settings for the HNSW algorithm.
    ///
    /// # Description
    ///
    /// The process includes the following steps:
    ///
    /// 1. **Find Candidate Neighbors:** The function first identifies potential neighbors for the vector
    ///    being inserted by calling the `search_neighbors_to_add` method. This method populates a max-heap
    ///    with the closest vectors, where the farthest candidate is at the top.
    /// 2. **Shrink Neighbor List:** After identifying the candidate neighbors, the function checks if the
    ///    number of candidates exceeds the allowed limit for that level. If necessary, it shrinks the list
    ///    to retain only the most relevant neighbors using the `shrink_neighbor_list` method.
    /// 3. **Attempt to Set Neighbors:** The function then calls `add_link` to attempt to establish
    ///    connections between the vector being added and its candidate neighbors. If the vector's neighbor
    ///    list is already full, the function uses heuristics to select which of the candidates to keep and
    ///    which to discard. Reciprocal links are also managed, aiming to set the vector being added as a
    ///    neighbor for the selected candidates. To ensure thread safety during this process, the function
    ///    first acquires the lock for each vector whose neighbor list is being updated. This prevents concurrent
    ///    modifications by other threads, maintaining the integrity of the neighbor lists.
    ///
    fn add_links_starting_from<E>(
        &self,
        query_evaluator: &E,
        id_vec: usize,
        nearest_vec: usize,
        dis_nearest_vec: f32,
        curr_level: u8,
        locks: &[Mutex<()>],
        guard: MutexGuard<()>,
        config: &ConfigHnsw,
    ) where
        E: QueryEvaluator<'a, Q = Q>, // <= tie evaluator’s Q to builder’s Q
    {
        //max-heap, on top is the farthest vector
        let mut closest_vectors: BinaryHeap<Node> = BinaryHeap::new();

        self.search_neighbors_to_add(
            &mut closest_vectors,
            query_evaluator,
            nearest_vec,
            dis_nearest_vec,
            curr_level,
            config,
        );

        let m = self.num_neighbors_per_level(curr_level);

        self.shrink_neighbor_list(&mut closest_vectors, m);

        closest_vectors.iter().for_each(|&neighbor| {
            self.add_link(id_vec, neighbor.id_vec(), curr_level);
        });

        std::mem::drop(guard);

        closest_vectors.iter().for_each(|&neighbor| {
            let _lock = locks[neighbor.id_vec()].lock().unwrap();
            self.add_link(neighbor.id_vec(), id_vec, curr_level);
        });
    }

    /// This function updates the neighbor list for the source vector (`id_src`) by including the
    /// destination vector (`id_dest`) at the specified level (`curr_level`). If the neighbor list
    /// is full, the function uses a heuristic to select the most relevant neighbors to retain.
    ///
    /// # Arguments
    ///
    /// - `id_src`: The index of the source vector in the HNSW graph, which is being updated to include
    ///    the new neighbor.
    /// - `id_dest`: The index of the destination vector in the HNSW graph, which is being considered
    ///    for inclusion in the neighbor list of `id_src`.
    /// - `curr_level`: The level in the HNSW graph where the neighbor list of `id_src` is being updated.
    ///
    /// # Description
    ///
    /// 1. **Get Neighbor Offsets:** Retrieves the start (`begin`) and end (`end`) offsets for the neighbor
    ///    list of `id_src` at `curr_level`. The range `[begin, end)` defines where the neighbors are stored
    ///    for `id_src` at this level, with `end` being exclusive.
    ///
    /// 2. **Allocate in Empty Position:** Attempts to insert `id_dest` into an available slot within the
    ///    neighbor list of `id_src`. If there is space available, the function returns immediately,
    ///    having successfully added the new neighbor.
    ///
    /// 3. **Handle Full Neighbor List:** If the list is full, the function creates a max-heap containing
    ///    `id_dest` along with the existing neighbors of `id_src`.
    ///
    /// 4. **Shrink Neighbor List:** After computing the candidate neighbors, the list is reduced to the
    ///    maximum allowed size for the current level (`max_size`) using `shrink_neighbor_list`.
    ///    This step ensures that only the most pertinent neighbors are kept in the list.
    ///
    /// 5. **Update Neighbor List:** Updates the neighbor list of `id_src` for the specified level
    ///    with the selected neighbors from the `shrink_neighbor_list` function.
    ///
    /// # Note
    ///
    /// The function does not guarantee that `id_dest` will be included as a neighbor of `id_src`. If the neighbor
    /// list of `id_src` is full, the heuristic applied in the `shrink_neighbor_list` function may not choose `id_dest`
    /// as a neighbor.
    fn add_link(&self, id_src: usize, id_dest: usize, curr_level: u8) {
        let (begin, end) = self.get_offsets_neighbors_from_level(id_src, curr_level);

        if self.allocate_in_empty_position(begin, end, id_dest) {
            return;
        }

        let mut candidate_neighbors = self.compute_candidate_neighbors(id_src, id_dest, begin, end);

        let max_size = end - begin;
        self.shrink_neighbor_list(&mut candidate_neighbors, max_size);

        self.fill_neighbors_list(&mut candidate_neighbors, begin, end);
    }

    /// Attempts to insert a new neighbor into an empty slot in the neighbor list of a vector.
    ///
    /// # Arguments
    ///
    /// - `begin`: The start index (inclusive) of the range in the neighbor list to check for available positions.
    /// - `end`: The end index (exclusive) of the range in the neighbor list to check for available positions.
    /// - `id_dest`: The index of the vector to be added as a neighbor.
    ///
    /// # Description
    ///
    /// This function attempts to insert `id_dest` into an empty slot in the neighbor list of a vector.
    /// It operates within the range defined by `begin` and `end` indices. If an empty slot is found,
    /// it places `id_dest` in that slot and returns `true`.
    /// If no empty slots are available, it returns `false`.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Initial state of the neighbor list, where `None` represents an empty slot
    /// let neighbors: Vec<Mutex<Option<usize>>> = vec![
    ///     Mutex::new(Some(3)), // suppose this is index 10
    ///     Mutex::new(Some(5)),
    ///     Mutex::new(None),
    ///     Mutex::new(None),
    ///     Mutex::new(None),   // suppose this is index 14
    /// ];
    ///
    /// // Call to `allocate_in_empty_position` with `begin` = 10 and `end` = 15
    /// let result = allocate_in_empty_position(10, 15, 7);
    ///
    /// // After the call, the neighbor list is updated
    /// // The state of the neighbor list will now be:
    /// // [Some(3), Some(5), Some(7), None, None]
    /// ```
    #[inline]
    fn allocate_in_empty_position(&self, begin: usize, end: usize, id_dest: usize) -> bool {
        if self.neighbors[end - 1].lock().unwrap().is_none() {
            let first_empty_spot = self.find_first_empty_spot(begin, end);
            *self.neighbors[first_empty_spot].lock().unwrap() = Some(id_dest);
            return true;
        }
        false
    }

    /// Finds the index of the first empty slot (i.e., `None`) within a specified range in the neighbor list.
    ///
    ///  # Arguments
    ///
    /// - `begin`: The start index (inclusive) of the range within the `neighbors` list to search for empty slots.
    /// - `end`: The end index (exclusive) of the range within the `neighbors` list to search for empty slots.
    ///
    /// # Returns
    ///
    /// - `usize`: The index of the first empty slot within the specified range. If no empty slots are found,
    ///    returns the `begin` index.
    ///
    /// **Note**: This function assumes there is at least one empty slot in the range.
    ///
    /// # Example
    ///
    /// ```rust
    /// // Example neighbor list with some slots filled and others empty
    /// let neighbors: Vec<Mutex<Option<usize>>> = vec![
    ///     Mutex::new(Some(1)),  \\ Suppose index is 10
    ///     Mutex::new(Some(2)),
    ///     Mutex::new(None),
    ///     Mutex::new(None),
    ///     Mutex::new(None),     \\ Suppose index is 14
    /// ];
    ///
    /// // Finding the first empty slot in the range from index 10 to 15
    /// let index = find_first_empty_spot(10, 15);
    ///
    /// // The result will be the index of the first empty slot, which is 12
    /// assert_eq!(index, 12);
    /// ```
    #[inline]
    fn find_first_empty_spot(&self, begin: usize, end: usize) -> usize {
        let mut i = end - 1;
        while i > begin {
            if self.neighbors[i - 1].lock().unwrap().is_some() {
                break;
            }
            i -= 1;
        }
        i
    }

    /// Computes a max-heap of candidate neighbors for the vector with index `id_src`, including both
    /// existing neighbors and a new vector.
    ///
    /// # Arguments
    ///
    /// - `id_src`: The index of the source vector for which potential neighbors are being computed.
    /// - `id_dest`: The index of the new vector to be considered as a potential neighbor for `id_src`.
    /// - `begin`: The start index (inclusive) of the segment in the neighbor list where neighbors of `id_src` are stored.
    /// - `end`: The end index (exclusive) of the segment in the neighbor list where neighbors of `id_src` are stored.
    ///
    /// # Description
    ///
    /// This function constructs a max-heap of candidate neighbors for the vector with index `id_src`. The heap includes:
    /// 1. The new vector with index `id_dest`, along with its distance to `id_src`.
    /// 2. The existing neighbors within the range `[begin, end)` from the neighbor list of `id_src`, each
    ///    with their distance to `id_src`.
    ///
    #[inline]
    fn compute_candidate_neighbors(
        &self,
        id_src: usize,
        id_dest: usize,
        begin: usize,
        end: usize,
    ) -> BinaryHeap<Node> {
        let distance_src_dest = self.dataset.compute_distance_by_id(id_src, id_dest);
        let mut candidate_neighbors: BinaryHeap<Node> = BinaryHeap::new();
        candidate_neighbors.push(Node(distance_src_dest, id_dest));

        for i in begin..end {
            let neigh = self.neighbors[i].lock().unwrap().unwrap();
            let distance_src_neigh = self.dataset.compute_distance_by_id(id_src, neigh);
            candidate_neighbors.push(Node(distance_src_neigh, neigh));
        }

        candidate_neighbors
    }

    /// Updates the neighbor list of a vector with a selected set of candidate neighbors.
    ///
    /// # Arguments
    ///
    /// - `candidate_neighbors`: A mutable reference to a max-heap (`BinaryHeap`) of `Node` objects representing
    ///    the selected neighbors.
    /// - `begin`: The start index (inclusive) of the range in the neighbor list where neighbors will be updated.
    /// - `end`: The end index (exclusive) of the range in the neighbor list where neighbors will be updated.
    ///
    /// # Description
    ///
    /// This function fills the neighbor list of a vector with the selected candidate neighbors from the
    /// provided `candidate_neighbors` heap.
    /// The neighbors are stored in the range defined by the `begin` and `end` indices. The function processes
    /// the heap until it is empty, populating the neighbor list starting from `begin`. If any slots remain after
    /// all neighbors are added, they are filled with `None` to indicate empty slots.
    #[inline]
    fn fill_neighbors_list(
        &self,
        candidate_neighbors: &mut BinaryHeap<Node>,
        begin: usize,
        end: usize,
    ) {
        let mut i = begin;
        while !candidate_neighbors.is_empty() {
            *self.neighbors[i].lock().unwrap() = Some(candidate_neighbors.pop().unwrap().id_vec());
            i += 1;
        }
        while i < end {
            *self.neighbors[i].lock().unwrap() = None;
            i += 1;
        }
    }

    /// Performs a search to find the nearest neighbors for a given vector being added to the HNSW graph.
    ///
    /// /// # Arguments
    ///
    /// - `closest_vectors`: A mutable reference to a max-heap (`BinaryHeap`) of `Node` objects.
    ///    This heap is initially empty and will be populated with the closest vectors found during the search.
    /// - `query_evaluator`: An implementation of the `QueryEvaluator` trait that provides the method to compute
    ///    the distance between the query vector and the vectors already added to the graph.
    /// - `nearest_vec`: The index of the nearest vector found in the highest levels that acts as the starting
    ///    node from which the search begins.
    /// - `dis_nearest_vec`: The distance of the nearest vector found so far to the query vector.
    /// - `curr_level`: The current level in the HNSW graph where the search is being conducted.
    /// - `visited_table`: A mutable reference to the `VisitedTable`, which keeps track of the vectors that have
    ///    already been visited during the search.
    /// - `config`: A reference to the `ConfigHnsw` structure, which contains configuration parameters for the search
    ///    process, such as the `ef_construction` value.
    ///
    /// # Description
    ///
    /// The search process begins by initializing a min-heap (`candidates`) with the nearest vector found so far.
    /// The function then iteratively evaluates each candidate's neighbors. For each neighbor that hasn't been
    /// visited yet, the function calculates the distance between the neighbor and the query vector using the
    /// `query_evaluator`. If the neighbor is closer than the farthest vector in the `closest_vectors` heap,
    ///  it is added to both heaps and marked as visited.
    ///
    /// The search continues until all candidates have been explored or until a candidate's distance exceeds
    /// the maximum distance in the `closest_vectors` heap. If a candidate's distance is greater than this
    /// maximum distance, the search stops as further candidates will have even larger distances and are thus
    /// less likely to be relevant.
    ///
    /// Once the search completes, `closest_vectors` will contain the closest neighbors to the query vector,
    /// and the visited table is advanced to prepare for subsequent searches.
    fn search_neighbors_to_add<E>(
        &self,
        closest_vectors: &mut BinaryHeap<Node>,
        query_evaluator: &E,
        nearest_vec: usize,
        dis_nearest_vec: f32,
        curr_level: u8,
        config: &ConfigHnsw,
    ) where
        E: QueryEvaluator<'a, Q = Q>, // <= tie evaluator’s Q to builder’s Q
    {
        //min-heap based on distance
        let mut candidates: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut visited_table: HashSet<usize> = HashSet::default();

        let node = Node(dis_nearest_vec, nearest_vec);
        candidates.push(Reverse(node));
        closest_vectors.push(node);
        visited_table.insert(nearest_vec);

        while let Some(node) = candidates.pop() {
            let curr_node = node.0;
            let farthest_in_heap = *closest_vectors.peek().unwrap();

            if curr_node.distance() > farthest_in_heap.distance() {
                break;
            }

            let curr_node = curr_node.id_vec();
            let neighbors = self.get_unlocked_neighbors(curr_node, curr_level);

            for &neighbor in neighbors.iter() {
                if !visited_table.contains(&neighbor) {
                    visited_table.insert(neighbor);

                    let distance_to_neighbor =
                        query_evaluator.compute_distance(&self.dataset, neighbor);
                    let neighbor_node = Node(distance_to_neighbor, neighbor);

                    add_neighbor_to_heaps(
                        &mut candidates,
                        closest_vectors,
                        neighbor_node,
                        config.get_ef_construction(),
                    );
                }
            }
        }
    }

    /// Shrinks the neighbor list to ensure it contains only the most relevant neighbors, up to a
    /// specified maximum size. This function applies a heuristic from the HNSW paper (https://arxiv.org/abs/1603.09320)
    /// by Malkov and Yashunin to determine which neighbors to keep.
    ///
    /// # Arguments
    ///
    /// - `closest_vectors`: A mutable reference to a `BinaryHeap<Node>` containing the current neighbors.
    /// - `max_size`: The maximum number of neighbors to include in the list.
    ///
    /// # Description
    ///
    /// This function manages the size of the `closest_vectors` heap to ensure it does not exceed `max_size`.
    /// If the heap already contains more neighbors than allowed, the function performs the following steps:
    ///
    /// 1. **Convert Heap:** Converts the max-heap of `closest_vectors` into a min-heap for easier management
    ///    of the smallest elements.
    ///
    /// 2. **Evaluate Nodes:** Iterates over the nodes in the min-heap and determines which nodes should remain.
    ///    A node will stay in the heap if it is closer to the query vector than any other node in `closest_vectors`.
    ///
    /// 3. **Rebuild Heap:** Pushes the nodes that should stay back into the `closest_vectors` heap.
    ///    The process stops when the heap reaches the `max_size` or when all relevant nodes have been processed.
    fn shrink_neighbor_list(&self, closest_vectors: &mut BinaryHeap<Node>, max_size: usize) {
        if closest_vectors.len() < max_size {
            return;
        }

        let mut min_heap = from_max_heap_to_min_heap(closest_vectors);

        while let Some(node) = min_heap.pop() {
            let node1 = node.0;

            let dist_node1_vec = node1.distance();
            let mut keep_node_1 = true;

            for node2 in closest_vectors.iter() {
                let dist_node_1_node2 = self
                    .dataset
                    .compute_distance_by_id(node1.id_vec(), node2.id_vec());
                if dist_node_1_node2 < dist_node1_vec {
                    keep_node_1 = false;
                    break;
                }
            }

            if keep_node_1 {
                closest_vectors.push(node1);
                if closest_vectors.len() >= max_size {
                    return;
                }
            }
        }
    }

    /// Assigns levels to each vector in the graph and updates the internal `offsets` and `neighbors` vectors.
    ///
    /// # Arguments
    ///
    /// - `num_vectors`: The number of vectors to which levels will be assigned.
    ///
    /// # Description
    ///
    /// This function assigns a level to each vector in the graph. It updates the `offsets` vector based on the assigned
    /// levels and initializes the `neighbors` vector with `None` to represent empty neighbor slots.
    #[inline]
    fn assign_level(&mut self, num_vectors: usize) {
        let mut rng = StdRng::seed_from_u64(523);

        for _ in 0..num_vectors {
            let level = self.random_level(&mut rng);
            self.levels_assigned.push(level);

            if level > self.max_level {
                self.max_level = level;
            }

            self.update_offset_vector(level);
        }

        // padd neighbors vector with None, which means that there are not neighbors yet
        for _ in 0..*self.offsets.last().unwrap() {
            self.neighbors.push(Mutex::new(None));
        }
    }

    /// Updates the `offset` vector by pushing the position where the current node's neighbors end and the
    /// next node's neighbors begin.
    ///
    /// # Parameters
    ///
    /// - `level`: The level assigned to the node for which the offset is being updated.
    ///
    /// # Description
    ///
    /// This function retrieves the last position in the `offset` vector, which indicates where the
    /// previous node's neighbors ended and current node's neighbors begin. Then, based on the
    /// level assigned to the current node, it calculates the number of neighbors the node has using
    /// the `cum_num_neigh_per_level` vector.
    /// Finally, it adds this number to the last position and pushes the resulting value to the `offset` vector.
    /// This new value represents the position where the current node's neighbors end and the next node's
    /// neighbors start in the neighbors vector.
    ///
    /// # Example
    ///
    /// Assume that the number of neighbors per vector is set to 16, and the `offset` vector initially
    /// looks like `[0, 48, 80, 112]`. The `cum_num_neigh_per_level` vector is `[0, 32, 48, 64, 80]`,
    /// where the neighbors for level 0 are `2 * num_neighbors_per_vec`.
    ///
    /// If the node being updated is assigned to level 1, it will have a total of 48 neighbors:
    /// 32 for level 0 and 16 for level 1.
    /// After updating, the `offset` vector becomes `[0, 48, 80, 112, 160]`, meaning that the neighbors
    /// of the current node are stored in the range `[112, 160)`.
    fn update_offset_vector(&mut self, level: u8) {
        let last_pos_offset = self.offsets.last().unwrap_or(&0);
        let cum_num_neighbors = self
            .cum_num_neigh_per_level
            .get((level + 1) as usize)
            .expect("Level not present");
        self.offsets.push(last_pos_offset + cum_num_neighbors);
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
    fn random_level(&self, rng: &mut StdRng) -> u8 {
        let mut f: f32 = rng.gen_range(0.0..1.0);
        for (level, &prob) in self.probabs_levels.iter().enumerate() {
            if f < prob {
                return level as u8;
            }
            f -= prob;
        }
        // it returns the maximum level which is the size of the vector probabs_levels
        (self.probabs_levels.len() - 1) as u8
    }

    /// This function retrieves a slice of the `neighbors` vector that corresponds to the neighbors
    /// of a vector at a specific level.
    /// # Parameters
    ///
    /// - `id_vec`: The ID of the vector for which neighbors are being accessed.
    /// - `level`: The level in the HNSW graph for which neighbors are to be retrieved.
    ///
    /// # Returns
    ///
    /// - `&[Mutex<Option<usize>>]`: A slice of the `neighbors` vector, containing the neighbors of
    ///   the specified vector at the specified level.
    #[inline]
    pub fn get_neighbors_from_level(&self, id_vec: usize, level: u8) -> &[Mutex<Option<usize>>] {
        let (begin, end) = self.get_offsets_neighbors_from_level(id_vec, level);
        &self.neighbors[begin..end]
    }

    /// Computes the range of indices for accessing neighbors of a vector at a specific level.
    ///
    /// # Parameters
    ///
    /// - `id_vec`: The ID of the vector for which the neighbor offsets are being computed.
    /// - `level`: The level in the HNSW graph for which the neighbor offsets are to be determined.
    ///
    /// # Returns
    ///
    /// - `(usize, usize)`: A tuple containing:
    ///   - The starting index (`begin`) in the `neighbors` vector where the neighbors of the vector at
    ///     the specified level begin.
    ///   - The ending index (`end`) in the `neighbors` vector where the neighbors of the vector at the
    ///     specified level end (exclusive).
    ///
    /// # Description
    ///
    /// To compute these ranges, the function first retrieves the offset for the vector's ID,
    /// which indicates the base position where the neighbors of that vector start.
    /// To determine the exact range for the neighbors at the requested level, the function:
    ///
    /// 1. Adds the cumulative number of neighbors up to the current level to the offset to
    ///    compute the `begin` index. This gives the starting position for the neighbors at the specified level.
    /// 2. Adds the cumulative number of neighbors up to the next level to the offset to compute the `end` index.
    ///    This provides the ending position for the neighbors of the current level.
    ///
    /// # Example
    ///
    /// For an `offset` vector of `[0, 48, 96, 112]` and a `cum_num_neigh_per_level` vector of `[0, 32, 48, 64, 80]`,
    /// to determine the range of neighbors for the vector with ID `1` at level `1`, follow these steps:
    ///
    /// 1. Retrieve the offset for the vector with ID `1`, which is `48`.
    /// 2. Compute the `begin` index by adding the cumulative number of neighbors for level `1`,
    ///    which is `32`, to the offset: `48 + 32 = 80`.
    /// 3. Compute the `end` index by adding the cumulative number of neighbors for level `2`,
    ///    which is `48`, to the offset: `48 + 48 = 96`.
    ///
    /// Therefore, the range of indices for the neighbors of the vector with ID `1` at level `1` is `[80, 96)`.
    #[inline]
    fn get_offsets_neighbors_from_level(&self, id_vec: usize, level: u8) -> (usize, usize) {
        assert!(
            level <= self.levels_assigned[id_vec],
            "The required level for the vector with ID {} is higher than the one assigned.",
            id_vec
        );

        let o = self.offsets.get(id_vec).expect("Id vector not present");
        let begin = o + self
            .cum_num_neigh_per_level
            .get(level as usize)
            .expect("Level not present");
        let end = o + self
            .cum_num_neigh_per_level
            .get((level + 1) as usize)
            .expect("Level not present");

        (begin, end)
    }

    /// Initializes the default probability levels and cumulative neighbor counts for each level in the HNSW graph.
    ///
    /// # Description
    ///
    /// This function sets up two key components for the HNSW graph structure:
    ///
    /// 1. **Probability Levels (`probabs_levels`)**: The probability of a vector being assigned
    ///    to each level of the graph. The probability decreases exponentially with increasing levels,
    ///    ensuring fewer vectors are assigned to higher levels.
    ///    The probabilities are computed based on the level multiplier (`level_mult`).
    ///
    /// 2. **Cumulative Neighbors per Level (`cum_num_neigh_per_level`)**: The cumulative count
    ///    of neighbors up to each level. The function calculates the number of neighbors for each level,
    ///    with level 0 having twice the number of neighbors as higher levels.
    ///    This count is accumulated and stored in the `cum_num_neigh_per_level` vector,
    ///    where each entry represents the total number of neighbors up to that level.
    ///
    /// The function continues to compute these values for increasing levels until the calculated
    /// probability for a level falls below a small threshold (`1e-9`).
    ///
    /// # Parameters
    ///
    /// - `level_mult`: A factor that controls how quickly the probability decreases as the level increases.
    ///   A larger `level_mult` causes probabilities to drop more slowly, spreading vectors across more levels.
    ///   A smaller `level_mult` leads to a faster drop-off, concentrating most vectors in the lower levels.
    /// - `num_neighbors_per_vec`: The base number of neighbors assigned to each vector at levels above 0.
    ///    Level 0 is assigned twice this number of neighbors.
    /// # Example
    ///
    /// After calling this function with a `level_mult` of `1.0` and `num_neighbors_per_vec` of `16`,
    /// the `probabs_levels` and `cum_num_neigh_per_level` vectors might look like this:
    ///
    /// ```rust
    /// probabs_levels = [0.6321, 0.3679, 0.1353, ...];
    /// cum_num_neigh_per_level = [0, 32, 48, 64, ...];
    /// ```
    ///
    fn set_default_probas(&mut self, level_mult: f32, num_neighbors_per_vec: usize) {
        let mut nn = 0;
        self.cum_num_neigh_per_level.push(0);

        for level in 0.. {
            let proba = (-level as f32 / level_mult).exp() * (1.0 - (-1.0 / level_mult).exp());
            if proba < 1e-9 {
                break;
            }
            self.probabs_levels.push(proba);

            nn += if level == 0 {
                num_neighbors_per_vec * 2
            } else {
                num_neighbors_per_vec
            };
            self.cum_num_neigh_per_level.push(nn);
        }
    }

    /// Returns the number of neighbors for a given level.
    ///
    /// This function calculates the number of neighbors assigned to a specific level by subtracting the cumulative
    /// number of neighbors for the current level from the cumulative number of neighbors for the next level.
    ///
    /// # Parameters
    /// - `level`: The level for which the number of neighbors is to be retrieved.
    ///
    /// # Returns
    /// - `usize`: The number of neighbors assigned to the specified level.
    ///
    /// # Example
    ///
    ///
    /// Assume the `cum_num_neigh_per_level` vector is `[0, 32, 48, 64]`.
    /// If we want to know the number of neighbors for level 2, we calculate it as `64 - 48`, which equals `16`.
    ///
    #[inline]
    fn num_neighbors_per_level(&self, level: u8) -> usize {
        self.cum_num_neigh_per_level[(level + 1) as usize]
            - self.cum_num_neigh_per_level[level as usize]
    }

    /// Returns a reference to the vector that stores the level assigned to each vector.
    pub fn get_level_assigned(&self) -> &Vec<u8> {
        &self.levels_assigned
    }
}
