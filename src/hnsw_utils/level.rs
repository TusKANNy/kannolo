use serde::{Deserialize, Serialize};

use crate::{
    quantizer::{IdentityQuantizer, Quantizer, QueryEvaluator},
    Dataset, Float,
};

use super::{compute_closest_from_neighbors, hnsw_builder::HnswBuilder};
/// Represents a level in the HNSW graph.
///
/// # Fields
/// - `neighbors`: A list of all neighbors for vectors at this level. The neighbors for each vector
///    are stored in a contiguous block.
/// - `offsets`: An index mapping each vector ID to its starting position in the `neighbors` list.
///    The `offsets[id_vec]` provides the starting index in `neighbors` where the neighbors of
///    the vector with `id_vec` begin.
///
#[derive(Serialize, Deserialize)]
pub struct Level {
    neighbors: Box<[usize]>,
    offsets: Box<[usize]>,
}

impl Level {
    /// Constructs a new `Level` instance representing a specific level in the HNSW graph.
    ///
    /// # Parameters
    ///
    /// - `hnsw_builder`: A reference to an `HnswBuilder` which contains the information about the HNSW graph,
    ///    including the levels assigned to each node and the neighbor assigned to each vector.
    /// - `curr_level`: The current level in the HNSW graph for which this `Level` instance is being created.
    /// - `processed_ids`: A mutable reference to a vector which store the IDs of the nodes processed until the
    ///    current level.
    /// - `new_to_original`: A mutable reference to a vector that will be populated with the mapping
    ///    from the new IDs to their original IDs.
    /// - `original_to_new`: A mutable reference to an array that will be populated with the mapping
    ///    from original IDs to new IDs.
    ///
    /// # Description
    ///
    /// This function creates a new instance of `Level` representing a specific level in the HNS) graph.
    /// It does so by processing the node IDs assigned to the specified level, mapping the original
    /// IDs to new, sequential IDs, and storing this mapping in two structures:
    ///
    /// 1. A vector that maps new IDs to their corresponding original IDs, enabling efficient access to the dataset.
    ///    Since retrieving vectors from the dataset requires the original IDs, this mapping is crucial for ensuring that
    ///    the correct vectors can be located based on their new IDs.
    ///  
    /// 2. An array mapping original IDs to new IDs. This mapping is necessary during the construction of the levels.
    ///    It is used when retrieving the neighbors of a vector to convert their original IDs to the new IDs assigned to them,
    ///    ensuring that the neighbors are stored with the correct new IDs in the level structure.
    ///
    /// In the HNSW graph, nodes assigned to higher levels receive smaller IDs.
    /// For instance, if there are 15 nodes at the highest level, they will be assigned IDs ranging from 0 to 14.
    /// Nodes at the next lower level will receive IDs starting from 15. If this level has 100 nodes,
    /// they will be assigned IDs from 15 to 114. This pattern continues, with each subsequent level of nodes
    /// getting sequentially larger IDs, reflecting the hierarchical structure of the graph.
    #[inline]
    pub fn new<'a, D, Q>(
        hnsw_builder: &HnswBuilder<'a, D, Q>,
        curr_level: u8,
        processed_ids: &mut Vec<usize>,
        new_to_original: &mut Vec<usize>,
        original_to_new: &mut [usize],
    ) -> Self
    where
        D: Dataset<Q> + Sync,
        Q: Quantizer<DatasetType = D> + Sync,
        Q: IdentityQuantizer<DatasetType = D, T: Float>,

        // This constraint is necessary because the vector returned by the dataset's get function is of type Datatype.
        // The query evaluator, however, requires a vector of type Querytype.
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <D as Dataset<Q>>::DataType<'a>>,
    {
        // Retrieve IDs of vectors assigned to the current level.
        let ids_assigned_curr_level: Vec<usize> = hnsw_builder
            .get_level_assigned()
            .iter()
            .enumerate()
            .filter(|(_, &level)| level == curr_level)
            .map(|(index, _)| index)
            .collect();

        // Update mappings from new IDs to original IDs and vice versa.
        for &id in ids_assigned_curr_level.iter() {
            new_to_original.push(id);
            original_to_new[id] = new_to_original.len() - 1;
        }

        processed_ids.extend(ids_assigned_curr_level);

        let mut neighbors: Vec<usize> = Vec::new();
        let mut offsets: Vec<usize> = Vec::with_capacity(processed_ids.len());
        offsets.push(0);

        for &original_id in processed_ids.iter() {
            let neighbors_id = hnsw_builder.get_neighbors_from_level(original_id, curr_level);

            // Filter out `None` values, map original IDs to new IDs, and collect them.
            let real_neighbors_curr_vec: Vec<usize> = neighbors_id
                .iter()
                .map(|mutex| mutex.lock().unwrap())
                .filter_map(|neighbor| *neighbor)
                .map(|id| original_to_new[id])
                .collect();

            let num_neighbors = real_neighbors_curr_vec.len();

            neighbors.extend(real_neighbors_curr_vec);

            // Update the offsets to indicate where the neighbors for the next vector begin.
            let last_pos_offset = offsets.last().unwrap();
            offsets.push(last_pos_offset + num_neighbors);
        }

        Self {
            neighbors: neighbors.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        }
    }

    /// Updates the nearest vector to a query using a greedy approach.
    ///
    /// This method iteratively explores the neighbors of the current nearest vector to find a vector closer to the query.
    /// The nearest vector and its distance are updated based on the computed distances from the query to each neighbor.
    ///
    /// # Arguments
    /// - `query_evaluator`: A reference to an object implementing the `QueryEvaluator` trait.
    ///    This object provides the method to compute the distance between the query vector and each neighbor.
    /// - `nearest_vec`: A mutable reference to a `usize` variable that will be updated to the index of the closest neighbor found.
    /// - `dis_nearest_vec`: A mutable reference to a `f32` variable that will be updated to the distance of the closest neighbor found.
    /// - `id_permutation`: A boxed slice of IDs that maps each ID to its position in the dataset. This permutation allows
    ///   access to the original vectors by translating permuted indices into their corresponding positions in the dataset.
    ///     
    /// # Description
    /// The function starts by retrieving the neighbors of the current closest vector to the query. For each neighbor,
    /// it computes the distance to the query vector using the `query_evaluator`. If the computed distance is smaller
    /// than the current shortest distance (`dis_nearest_vec`), it updates `nearest_vec` to the current neighbor
    /// and `dis_nearest_vec` to the new shortest distance.
    ///
    /// This process is performed in a loop: after updating the nearest vector with the closest neighbor found,
    /// the function retrieves the neighbors of the new closest vector and repeats the process. The loop repeats
    /// until no closer neighbors can be found, meaning the closest vector to the query has been identified.
    pub fn greedy_update_nearest<'a, Q, D, E>(
        &self,
        dataset: &D,
        query_evaluator: &E,
        nearest_vec: &mut usize,
        dis_nearest_vec: &mut f32,
    ) where
        Q: Quantizer<DatasetType = D>, // 1) your quantizer’s associated type must be exactly D
        D: Dataset<Q>,                 // 2) dataset must implement Dataset<Q>
        E: QueryEvaluator<'a, Q = Q>,  // 3) evaluator’s Q must be your Q
    {
        loop {
            let prec_nearest = *nearest_vec;

            let neighbors = self.get_neighbors_from_id(*nearest_vec);

            compute_closest_from_neighbors(
                dataset,
                query_evaluator,
                neighbors,
                nearest_vec,
                dis_nearest_vec,
            );

            if prec_nearest == *nearest_vec {
                return;
            }
        }
    }

    /// Retrieves the neighbors of a vector given its ID.
    ///
    /// This function returns a slice of neighbor IDs for a specific vector identified by `id_vec`. It uses the `offsets` and `neighbors`
    /// arrays to efficiently locate and extract the neighbors associated with the provided vector ID.
    ///
    /// # Arguments
    /// - `id_vec`: The ID of the vector whose neighbors are to be retrieved.
    ///
    /// # Returns
    /// A slice of `usize` representing the IDs of the neighbors for the vector specified by `id_vec`.
    ///
    /// # Description
    ///
    /// 1. The function determines the starting index of the neighbors for the given vector ID using the `offsets`
    ///    array.
    /// 2. It calculates the number of neighbors by finding the difference between the offset of the current vector ID
    ///    and the offset of the next vector ID.
    /// 3. The function then returns a slice from the `neighbors` array, spanning from the start index to the calculated
    ///    range of neighbors.
    ///
    #[inline]
    pub fn get_neighbors_from_id(&self, id_vec: usize) -> &[usize] {
        let start = self.offsets[id_vec];
        let num_neighbors = self.offsets[id_vec + 1] - start;
        &self.neighbors[start..start + num_neighbors]
    }

    pub fn get_space_usage_bytes(&self) -> usize {
        self.neighbors.len() * std::mem::size_of::<usize>()
            + self.offsets.len() * std::mem::size_of::<usize>()
    }
}
