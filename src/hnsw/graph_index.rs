use crate::quantizer::{IdentityQuantizer, Quantizer, QueryEvaluator};
use crate::topk_selectors::topk_heap::TopkHeap;
use crate::topk_selectors::OnlineTopKSelector;
use crate::visited_set::set::VisitedSet;
use crate::{hnsw_utils::*, DistanceType};
use crate::{Dataset, Float, GrowableDataset};
use crate::{DotProduct, EuclideanDistance};
use bitvec::prelude::BitVec;
use config_hnsw::ConfigHnsw;
use hnsw_builder::HnswBuilder;
use level::Level;
use nohash_hasher::BuildNoHashHasher;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::marker::PhantomData;

/// A `GraphIndex` represents a Hierarchical Navigable Small World (HNSW) graph structure that is used
/// for approximate nearest neighbor (ANN) search. Constructed from either a dense or sparse dataset and
/// configuration settings, it efficiently finds the k closest vectors in the graph for each query within
/// the provided query dataset.
///
/// # Fields
///
/// - `levels`: A boxed slice containing the hierarchical levels of the HNSW graph. Each level stores
///   the neighbors of the vectors at that level, allowing for multi-level search to balance speed and accuracy.
/// - `dataset`: The dataset, either dense or sparse, that the graph index is built upon. This dataset holds
///   the vectors and provides access to their representations for searching.
/// - `num_neighbors_per_vec`: The number of neighbors per vector at each level in the HNSW graph. This parameter
///   determines the connectivity of the graph and affects the search performance and accuracy.
///    This helps the Rust compiler manage safety and type constraints related to `Q`.
/// - `id_permutation`: A boxed slice containing the permutation of vector IDs. This permutation allows
///   the retrieval of the position of a node in the dataset, enabling access to the corresponding vector values.
/// - `entry_vec`: This is the ID of the vector from which the search begins. It is a vector assigned to the
///   highest level in the hierarchy.
/// - `_phantom`: A `PhantomData` marker that indicates the type `Q` is used in the context of the struct,
///    ensuring proper type safety without actually storing a value of type `Q`.
#[derive(Serialize, Deserialize)]
pub struct GraphIndex<D, Q>
where
    D: Dataset<Q>,
    Q: Quantizer<DatasetType = D>,
{
    levels: Box<[Level]>,
    dataset: D,
    num_neighbors_per_vec: usize,
    id_permutation: Box<[usize]>,
    entry_vec: usize,
    _phantom: PhantomData<Q>,
}

impl<D, Q> GraphIndex<D, Q>
where
    D: Dataset<Q> + GrowableDataset<Q>,
    Q: Quantizer<DatasetType = D>,
{
    /// Constructs a new `GraphIndex` by building an HNSW graph from a given dataset, configuration, and quantizer.
    ///
    /// This function creates a `GraphIndex` using a source dataset, which can be either dense or sparse, along
    /// with the provided HNSW configuration settings and a quantizer. The source dataset provided does not need
    /// to encode its vectors, as it is required to implement the `IdentityQuantizer` trait. This indicates
    /// that the dataset handles raw vector data and relies on the supplied quantizer to encode the vectors
    /// during the graph construction.
    ///
    /// The resulting `GraphIndex` stores a permutated dataset in which vectors are reordered according to the new
    /// IDs assigned during graph construction. This permutated dataset is used during the search.
    ///
    /// # Arguments
    ///
    /// - `source_dataset`: A reference to the dataset containing the vectors to be indexed. This dataset implements
    ///   the `Dataset` trait, and the quantizer associated with it must implement the `IdentityQuantizer` trait,
    ///   meaning the dataset handles raw vector data and does not encode the vectors itself.
    /// - `config`: A reference to `ConfigHnsw`, which holds the configuration parameters for building the HNSW graph,
    ///   such as the number of neighbors per vector.
    /// - `quantizer`: A quantizer that implements the `Quantizer` trait and is responsible for encoding the vectors
    ///   of the dataset during graph construction.
    ///
    /// # Returns
    ///
    /// A new `GraphIndex` constructed from the provided dataset, configuration settings, and quantizer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use struttura_kANNolo::{
    /// hnsw::graph_index::GraphIndex,
    /// hnsw_utils::config_hnsw::ConfigHnsw};
    /// use rand::prelude::*;
    /// use std::iter;
    /// use struttura_kANNolo::plain_quantizer::PlainQuantizer;
    /// use struttura_kANNolo::{DenseDataset,GrowableDataset,DistanceType};
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let n_vecs=1000;
    /// let dim_vecs = 10;
    ///
    /// // Set the number of threads to use to build the GraphIndex
    /// let num_threads=32;
    ///
    /// // Generate a vector of random floating-point numbers for the dataset.
    /// let vectors :Vec<f32> = iter::repeat_with(|| rng.gen::<f32>()).take(n_vecs*dim_vecs).collect();
    ///
    /// // Create a DenseDataset from the generated vectors.    
    /// let dataset = DenseDataset::from_vec(vectors, dim_vecs, PlainQuantizer::<f32>::new(dim_vecs, DistanceType::Euclidean));
    ///
    /// // Create a quantizer for encoding vectors during graph construction.
    /// let quantizer = PlainQuantizer::<f32>::new(dim_vecs, DistanceType::Euclidean);
    ///
    /// // Build the HNSW graph configuration with default settings.
    /// let config = ConfigHnsw::new().build();
    ///
    /// // Create the GraphIndex using the dataset, configuration, and quantizer.
    /// let hnsw_index = GraphIndex::from_dataset(&dataset, &config, quantizer, num_threads);
    ///
    /// // At this point, the `hnsw_index` is ready for performing nearest neighbor searches.
    /// ```
    pub fn from_dataset<'a, SD, IQ>(
        source_dataset: &'a SD,
        config: &ConfigHnsw,
        quantizer: Q,
    ) -> Self
    where
        SD: Dataset<IQ> + Sync,
        IQ: IdentityQuantizer<DatasetType = SD, T: Float> + Sync + 'a,
        // This constraint is necessary because the vector returned by the dataset's get function is of type Datatype.
        // The query evaluator, however, requires a vector of type Querytype.
        <IQ as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <SD as Dataset<IQ>>::DataType<'a>>,
        // This constraint is necessary because the `push` function of the new_dataset
        // expects input types of InputDataType, while we iterate over types of DataType from the source_dataset.
        D: GrowableDataset<Q, InputDataType<'a> = <SD as Dataset<IQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: 'a,
    {
        let mut hnsw_builder = HnswBuilder::new(config.get_num_neighbors_per_vec(), source_dataset);

        let (levels, id_permutation, entry_vector) = hnsw_builder.compute_graph(config);

        let mut encoded_dataset = D::new(quantizer, source_dataset.dim());

        for id in 0..source_dataset.len() {
            let vec = source_dataset.get(id_permutation[id]);
            encoded_dataset.push(&vec);
        }

        GraphIndex::new(
            levels,
            encoded_dataset,
            config.get_num_neighbors_per_vec(),
            id_permutation,
            entry_vector,
        )
    }

    /// This function initializes a `GraphIndex` by taking the constructed levels of the HNSW graph, the dataset
    /// that contains the vectors, the number of neighbors per vector, and an ID permutation that maps vector IDs
    /// to their positions in the dataset used for constructing the graph.
    ///
    /// # Arguments
    ///
    /// - `levels`: A `Vec<Level>` representing the different levels of the HNSW graph. Each level contains neighbor
    ///   information for vectors at that level.
    /// - `dataset`: The dataset of vectors that were used to construct the graph. This dataset can be dense or sparse
    ///   and implements the `Dataset` trait.
    /// - `num_neighbors_per_vec`: The number of neighbors per vector, used to control the connectivity in the HNSW graph.
    /// - `id_permutation`: A boxed slice containing the permutation of vector IDs. This permutation is used to permute
    ///   the dataset based on the new IDs assigned during graph construction, ensuring the dataset used for the search
    ///   reflects the updated IDs. It is also used during the search to map the IDs of the closest vectors found back
    ///   to their original positions in the dataset used to construct the graph.
    ///
    /// # Returns
    ///
    /// A new instance of `GraphIndex` containing the provided levels, dataset, number of neighbors per vector,
    /// and ID permutation.

    fn new(
        levels: Vec<Level>,
        dataset: D,
        num_neighbors_per_vec: usize,
        id_permutation: Vec<usize>,
        entry_vec: usize,
    ) -> Self {
        Self {
            levels: levels.into_boxed_slice(),
            dataset,
            num_neighbors_per_vec,
            _phantom: PhantomData,
            id_permutation: id_permutation.into_boxed_slice(),
            entry_vec,
        }
    }
}

impl<D, Q> GraphIndex<D, Q>
where
    D: Dataset<Q> + Sync,
    Q: Quantizer<InputItem: Float, DatasetType = D> + Sync,
{
    pub fn dim(&self) -> usize {
        self.dataset.dim()
    }
    /// Performs a nearest neighbor search for a given set of query vectors on the HNSW graph.
    ///
    /// This function searches for the `k` nearest neighbors for each vector in the provided query dataset.
    /// It utilizes the HNSW (Hierarchical Navigable Small World) graph structure to efficiently find the
    /// closest vectors in the index.
    ///
    /// # Arguments
    ///
    /// - `queries`: A reference to a query dataset containing the vectors for which nearest neighbors need to be found.
    ///   This dataset implements the `Dataset` trait.
    /// - `k`: The number of nearest neighbors to return for each query vector.
    /// - `config`: A reference to `ConfigHnsw`, which holds configuration parameters for the search process.
    ///
    /// # Returns
    ///
    /// A `Vec<(f32, usize)>` containing tuples of the distance and the ID of the nearest neighbors for each query vector.
    /// The distances are in ascending order, with the closest vectors listed first. The IDs are adjusted according
    /// to their positions in the dataset used to build the `GraphIndex`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use struttura_kANNolo::{
    ///     hnsw::graph_index::GraphIndex,
    ///     hnsw_utils::config_hnsw::ConfigHnsw,
    ///     DenseDataset,
    ///     plain_quantizer::PlainQuantizer,
    ///     DistanceType,
    /// };
    /// use rand::prelude::*;
    /// use std::iter;
    ///
    ///
    /// let mut rng = rand::thread_rng();
    ///
    /// let n_vecs = 1000;
    /// let dim_vecs = 10;
    ///
    /// // Initialize the dataset.
    /// let vectors: Vec<f32> = iter::repeat_with(|| rng.gen::<f32>()).take(n_vecs * dim_vecs).collect();
    /// let dataset = DenseDataset::from_vec(vectors, dim_vecs, PlainQuantizer::new(dim_vecs, DistanceType::Euclidean));
    ///
    /// let config = ConfigHnsw::new().build();
    /// let k = 10; // Number of nearest neighbors to retrieve.
    ///
    /// // Create the GraphIndex.
    /// let hnsw_index = GraphIndex::from_dataset(&dataset, &config, PlainQuantizer::new(dim_vecs, DistanceType::Euclidean), 1);
    ///
    /// // Prepare the query dataset with random vectors.
    /// let query_vectors: Vec<f32> = iter::repeat_with(|| rng.gen::<f32>()).take(20 * dim_vecs).collect(); // 20 queries, each with `dim_vecs` dimensions
    /// let query_dataset = DenseDataset::from_vec(query_vectors, dim_vecs, PlainQuantizer::new(dim_vecs, DistanceType::Euclidean));
    ///
    /// // Perform the search.
    /// let results = hnsw_index.search(&query_dataset, k, &config);
    ///
    /// // `results` now contains the nearest neighbors for each query vector.
    /// ```
    pub fn search<'a, QD, QQ>(
        &self,
        query: QD::DataType<'a>,
        k: usize,
        config: &ConfigHnsw,
    ) -> Vec<(f32, usize)>
    where
        // The query dataset type (QD) could be directly of type D, but this would not work if D is a Dataset
        // with a ProductQuantizer, this because queries is a dataset with a PlainQuantizer.
        QD: Dataset<QQ> + Sync,
        QQ: Quantizer<DatasetType = QD> + Sync,
        // This constraint is necessary because the find_k_nearest_neighbors function takes an input parameter
        // of type QueryType, which is an associated type of the QueryEvaluator associated with the quantizer Q.
        // However, the queries are of type DataType, which is an associated type of the dataset QD.
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <QD as Dataset<QQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>,
        <Q as Quantizer>::InputItem: 'a,
    {
        let query_topk = self.find_k_nearest_neighbors(query, k, config);

        // remap ids based on their position in the dataset
        let mut topk: Vec<(f32, usize)> = query_topk
            .iter()
            .map(|x| (x.0, self.id_permutation[x.1]))
            .collect();

        // Adjust distance if using DotProduct distance type
        if self.dataset.quantizer().distance() == DistanceType::DotProduct {
            topk.iter_mut().for_each(|(dis, _)| *dis = -(*dis));
        }
        topk
    }

    /// Searches for the `k`-nearest neighbors of a given query vector within the HNSW graph.
    /// It starts by the entry point and performs a greedy search through the upper levels of the HNSW graph,
    /// updating the nearest neighbor (`nearest_vec`) and its distance (`dis_nearest_vec`) at each level.
    /// Once the search reaches level 0, it performs a more exhaustive search. The search continues until the
    /// `candidates` heap is empty or the distance to the farthest element in `top_candidates`
    /// (the best results found) is less than the distance to the closest element in `candidates`.
    ///
    /// # Description
    /// This function performs a search to find the `k` closest vectors to a given query vector using the
    /// HNSW graph structure.
    ///
    /// # Parameters
    ///
    /// - `query_vec`: The query vector for which the nearest neighbors are being searched.
    /// - `k`: The number of nearest neighbors to retrieve.
    /// - `config`: A reference to `ConfigHnsw`, which holds configuration parameters for the search.
    ///
    /// # Returns
    ///
    /// A `Vec<(f32, usize)>` containing tuples where each tuple represents a nearest neighbor. The first element
    /// is the distance to the neighbor, and the second element is the neighbor's ID.
    /// The results are sorted in ascending order of distance, with the closest vectors appearing first.

    pub fn find_k_nearest_neighbors<'a>(
        &self,
        query_vec: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
        k: usize,
        config: &ConfigHnsw,
    ) -> Vec<(f32, usize)>
    where
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>,
    {
        let mut topk_heap = TopkHeap::new(k);
        let query_evaluator = self.dataset.query_evaluator(query_vec);

        // Start from the entry point
        let mut nearest_vec = self.entry_vec;
        let mut dis_nearest_vec = query_evaluator.compute_distance(&self.dataset, nearest_vec);

        // Greedy search through the upper levels
        for level in self.levels.iter().skip(1).rev() {
            level.greedy_update_nearest(
                &self.dataset,
                &query_evaluator,
                &mut nearest_vec,
                &mut dis_nearest_vec,
            );
        }

        let ef = std::cmp::max(config.get_ef_search(), k);

        // Search on ground level
        let mut top_candidates = self.search_from_candidates_unbounded(
            Node(dis_nearest_vec, nearest_vec),
            &query_evaluator,
            ef,
            &self.levels[0],
        );
        while top_candidates.len() > k {
            top_candidates.pop();
        }
        while let Some(node) = top_candidates.pop() {
            topk_heap.push_with_id(node.distance(), node.id_vec());
        }

        topk_heap.topk()
    }

    /// Performs an unbounded search at the ground level of the HNSW graph to find the nearest neighbors for the given query.
    ///
    /// # Parameters
    ///
    /// - `starting_node`: The initial candidate node from which the search starts.
    /// - `query_evaluator`: Evaluates the distance between the query vector and nodes in the graph.
    /// - `ef`: The number of neighbors to consider during the search, affecting the size of heaps.
    /// - `level`: The current graph level where the search is conducted.
    ///
    /// # Description
    ///
    /// This function performs an unbounded search starting from a single candidate node. It maintains two heaps:
    /// - **`top_candidates`**: A max-heap that stores the top candidates found so far, ordered by their distance
    ///     from the query vector.
    /// - **`candidates`**: A min-heap that holds nodes to be evaluated, ordered by their distance from the query vector.
    ///
    /// The function proceeds as follows:
    /// 1. Initializes both heaps with the starting node and marks it as visited.
    /// 2. Iteratively pops nodes from the `candidates` heap.
    /// 3. If the distance of the current node is greater than the maximum distance in `top_candidates`, the search stops.
    /// 4. Otherwise, retrieves the neighbors of the current node and updates the heaps with these neighbors
    ///    if they haven’t been visited.
    ///
    /// The search continues until:
    /// - The `candidates` heap is empty.
    /// - The distance of the current node exceeds the maximum distance in the `top_candidates` heap.
    ///
    fn search_from_candidates_unbounded<'a, E>(
        &self,
        starting_node: Node,
        query_evaluator: &E,
        ef: usize,
        level: &Level,
    ) -> BinaryHeap<Node>
    where
        E: QueryEvaluator<'a, Q = Q>,  // 1) tie evaluator’s Q = our Q
        Q: Quantizer<DatasetType = D>, // 2) ensure our Q’s DatasetType = D
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>,
    {
        // max-heap
        let mut top_candidates: BinaryHeap<Node> = BinaryHeap::new();
        // min-heap
        let mut candidates: BinaryHeap<Reverse<Node>> = BinaryHeap::new();

        let mut visited_table = create_visited_set(self.dataset.len(), ef);

        top_candidates.push(starting_node);
        candidates.push(Reverse(starting_node));

        visited_table.insert(starting_node.id_vec());

        while let Some(Reverse(node)) = candidates.peek() {
            let id_candidate = node.id_vec();
            let distance_candidate = node.distance();

            if distance_candidate > top_candidates.peek().unwrap().distance() {
                break;
            }
            candidates.pop();

            let neighbors = level.get_neighbors_from_id(id_candidate);

            self.process_neighbors(
                neighbors,
                &mut *visited_table,
                query_evaluator,
                |dis_neigh, neighbor| {
                    add_neighbor_to_heaps(
                        &mut candidates,
                        &mut top_candidates,
                        Node(dis_neigh, neighbor),
                        ef,
                    );
                },
            )
        }
        top_candidates
    }

    /// Processes a list of neighboring nodes, computes their distances from the query vector, and updates
    /// various sets based on a callback function.
    ///
    /// # Parameters
    ///
    /// - `neighbors`: A slice of node IDs representing the neighboring nodes to be processed.
    /// - `visited_table`: Keeps track of which nodes have been visited to prevent redundant evaluations.
    /// - `query_evaluator`: Computes distances between the query vector and nodes in the graph.
    /// - `add_distances_fn`: A callback function that processes the distance and node ID for each
    ///   unvisited neighbor. Depending on the context in which `process_neighbors` is called, this
    ///   function may add the distances and node IDs to various data structures
    ///
    /// /// # Description
    ///
    /// This function handles the processing of neighboring nodes in the following way:
    /// 1. **Visit Tracking**: It marks each neighbor as visited to avoid reprocessing.
    /// 2. **Batch Processing**: Neighbors are processed in batches (up to 4 at a time) for efficiency.
    /// 3. **Distance Computation**: For each batch, the function computes the distances from the query vector
    ///      using `query_evaluator`.
    /// 4. **Update Sets**: The `add_distances_fn` callback is invoked with the computed distance and node ID.
    ///    This function handles updating the appropriate data structures, such as heaps, based on how
    ///   `process_neighbors` is used.
    /// 5. **Final Handling**: Any remaining neighbors that did not form a complete batch are processed
    ///    and their distances are computed and added.
    fn process_neighbors<'a, E, F>(
        &self,
        neighbors: &[usize],
        visited_table: &mut dyn VisitedSet,
        query_evaluator: &E,
        mut add_distances_fn: F,
    ) where
        E: QueryEvaluator<'a, Q = Q>,
        F: FnMut(f32, usize),
    {
        let mut counter = 0;
        // Stores the IDs of the neighbors whose distances will be computed
        let mut ids: Vec<usize> = vec![0; 4];

        for &neighbor in neighbors.iter() {
            let visited = visited_table.contains(neighbor);
            visited_table.insert(neighbor);

            ids[counter] = neighbor;

            if !visited {
                counter += 1;
            }

            if counter == 4 {
                let distances =
                    query_evaluator.compute_four_distances(&self.dataset, ids.iter().copied());
                for (dis_neigh, &neighbor) in distances.zip(ids.iter()) {
                    add_distances_fn(dis_neigh, neighbor);
                }
                counter = 0;
            }
        }

        // Add the remaining neighbors, if there are any left
        for neighbor in ids.iter().take(counter) {
            let distance_neighbor: f32 = query_evaluator.compute_distance(&self.dataset, *neighbor);
            add_distances_fn(distance_neighbor, *neighbor);
        }
    }

    /// Help function to print the space usage of the index.
    pub fn print_space_usage_byte(&self) -> usize {
        println!("Space Usage:");
        let forward: usize = self.dataset.get_space_usage_bytes();
        println!("\tForward Index: {:} Bytes", forward);
        let levels: usize = self
            .levels
            .iter()
            .map(|level| level.get_space_usage_bytes())
            .sum();

        let permutation: usize = self.id_permutation.len() * std::mem::size_of::<usize>();

        let additional: usize = 2 * std::mem::size_of::<usize>();

        println!(
            "\tLinks structure: {:} Bytes",
            levels + permutation + additional
        );

        println!(
            "\tTotal: {:} Bytes",
            forward + permutation + additional + levels
        );

        forward + permutation + additional + levels
    }
}

pub fn create_visited_set(dataset_size: usize, ef: usize) -> Box<dyn VisitedSet> {
    if dataset_size <= 2_000_000 || (dataset_size <= 10_000_000 && ef >= 400) {
        Box::new(BitVec::repeat(false, dataset_size))
    } else {
        Box::new(HashSet::with_capacity_and_hasher(
            200 + 32 * ef,
            BuildNoHashHasher::default(),
        ))
    }
}
