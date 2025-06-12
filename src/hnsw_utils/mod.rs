use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    sync::Mutex,
};

use crate::{
    quantizer::{Quantizer, QueryEvaluator},
    Dataset,
};

pub mod config_hnsw;
pub mod hnsw_builder;
pub mod level;

#[derive(Debug, Clone, Copy, PartialOrd)]

pub struct Node(pub f32, pub usize);

impl Node {
    pub fn distance(&self) -> f32 {
        self.0
    }
    pub fn id_vec(&self) -> usize {
        self.1
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance()
            .partial_cmp(&other.distance())
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.distance() == other.distance()
    }
}

impl Eq for Node {}

/// Adds a node to the min and max heaps used during the construction and search processes in the HNSW algorithm.
///
/// This function manages the insertion of a given node into both the minimum and maximum heaps,
/// which are utilized during both the construction and search phases of the HNSW algorithm.
/// The function ensures that the heaps do not exceed the size defined by the `ef_parameter`,
/// which controls the number of elements maintained during these processes.
///
/// # Description
///
/// The function follows these steps:
/// 1. If the number of elements in the `max_heap` is less than `ef_parameter`, the node is added to both heaps.
/// 2. If the `max_heap` is full but the new node is closer (i.e., has a smaller distance) than the
///    farthest node in the `max_heap`, the node is added to both the heaps.
/// 3. If the `max_heap` exceeds the `ef_parameter` size after adding the new node, the farthest node
///    is removed from the `max_heap`.
///
/// # Parameters
/// - `min_heap`: A mutable reference to a `BinaryHeap` of `Reverse<Node>` objects.
/// - `max_heap`: A mutable reference to a `BinaryHeap` of `Node` objects.
/// - `node`: The `Node` to be potentially added to the heaps.
/// - `ef_parameter`: The maximum number of elements that should be maintained in the heaps during the
///    construction and search processes.
///
/// # Example
/// ```rust
/// use std::collections::BinaryHeap;
/// use std::cmp::Reverse;
/// use struttura_kANNolo::hnsw_utils::{add_neighbor_to_heaps,Node};
///
/// let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
/// let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
/// let node = Node(0.5, 1); // Example node
/// let ef_parameter = 10;
///
/// add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);
/// ```
pub fn add_neighbor_to_heaps(
    min_heap: &mut BinaryHeap<Reverse<Node>>,
    max_heap: &mut BinaryHeap<Node>,
    node: Node,
    ef_parameter: usize,
) {
    let should_add_node = if max_heap.len() < ef_parameter {
        true
    } else if let Some(top_node) = max_heap.peek() {
        top_node.distance() > node.distance()
    } else {
        false
    };

    if should_add_node {
        min_heap.push(Reverse(node));
        max_heap.push(node);
    }

    if max_heap.len() > ef_parameter {
        max_heap.pop();
    }
}

pub fn add_neighbors_to_heaps(
    min_heap: &mut BinaryHeap<Reverse<Node>>,
    max_heap: &mut BinaryHeap<Node>,
    distances: &[f32],
    ids: &[usize],
    ef_parameter: usize,
) {
    for (distance, id) in distances.iter().zip(ids.iter()) {
        let should_add_node = if max_heap.len() < ef_parameter {
            true
        } else if let Some(top_node) = max_heap.peek() {
            top_node.distance() > *distance
        } else {
            false
        };

        if should_add_node {
            min_heap.push(Reverse(Node(*distance, *id)));
            max_heap.push(Node(*distance, *id));
        }

        if max_heap.len() > ef_parameter {
            max_heap.pop();
        }
    }
}
/// Determines the closest neighbor from a list by comparing distances to a query vector.
///
/// This function iterates through a list of neighbor indices and calculates their distances to the query vector using the
/// provided `QueryEvaluator`. It updates the closest neighbor and its distance if a closer neighbor is found.
///
/// # Parameters
/// - `query_evaluator`: A reference to an object implementing the `QueryEvaluator` trait.
///    This object provides the method to compute the distance between the query vector and each neighbor.
/// - `neighbors`: A slice of `usize` representing the indices of the neighbors to be evaluated.
/// - `nearest_vec`: A mutable reference to a `usize` variable that will be updated to the index of the closest neighbor found.
/// - `dis_nearest_vec`: A mutable reference to a `f32` variable that will be updated to the distance of
///    the closest neighbor found.
///
/// # Description
/// The function iterates over each neighbor in the `neighbors` slice. For each neighbor, it computes the
/// distance to the query vector using the
/// `query_evaluator`. If the computed distance is less than the current closest distance (`dis_nearest_vec`),
///  the function updates
/// `nearest_vec` to the current neighbor and `dis_nearest_vec` to the new shortest distance.
///
/// # Example
/// ```
/// use struttura_kANNolo::hnsw_utils::compute_closest_from_neighbors;
/// use struttura_kANNolo::DenseDataset;
/// use struttura_kANNolo::DistanceType;
/// use struttura_kANNolo::plain_quantizer::PlainQuantizer;
/// use struttura_kANNolo::DenseVector1D;
/// use crate::struttura_kANNolo::Dataset;
///
/// // Create a query vector
/// let query_vector: &[f32; 2] = &[1.0, 1.0];
/// let query = DenseVector1D::new(query_vector.as_slice());
///
/// let neighbor_vectors = &[2.0, 2.0, 1.0, 1.5, 0.0, 0.0];
/// let quantizer = PlainQuantizer::new(2, DistanceType::Euclidean);
/// let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);
///
/// // Create a query evaluator
/// let evaluator = dataset.query_evaluator(query);
///
/// // Define neighbors and find the closest one to the query
/// let neighbors = vec![0, 1, 2];
/// let mut nearest_vec = 0;
/// let mut dis_nearest_vec = f32::MAX;
///
/// compute_closest_from_neighbors(
///     &evaluator,
///     &neighbors,
///     &mut nearest_vec,
///     &mut dis_nearest_vec,
/// );
/// // Verify the result
/// assert_eq!(nearest_vec, 1);
/// ```
#[inline]
pub fn compute_closest_from_neighbors<'a, Q, D, E>(
    dataset: &D,
    query_evaluator: &E,
    neighbors: &[usize],
    nearest_vec: &mut usize,
    dis_nearest_vec: &mut f32,
) where
    Q: Quantizer<DatasetType = D>, // 1) your quantizer’s associated type must be exactly D
    D: Dataset<Q>,                 // 2) dataset must implement Dataset<Q>
    E: QueryEvaluator<'a, Q = Q>,  // 3) evaluator’s Q must be your Q
{
    for &neighbor in neighbors {
        let distance_neighbor = query_evaluator.compute_distance(dataset, neighbor);

        if distance_neighbor < *dis_nearest_vec {
            *nearest_vec = neighbor;
            *dis_nearest_vec = distance_neighbor;
        }
    }
}

/// Inserts the k closest results for a single query into a shared top-k vector containing the
/// closest results for multiple queries.
///
/// This function updates a shared vector of top-k results, which holds the closest vectors
/// for multiple queries. It inserts the top-k results for a single query into the appropriate
/// segment of this shared vector, ensuring that the `topk_query` vector has the correct length.
///
/// # Parameters
/// - `topk`: A `Mutex<Vec<(f32, usize)>>` representing a shared vector of top-k results
///    for multiple queries. Each element is a tuple `(distance, index)`.
/// - `query_topk`: A vector of tuples `(f32, usize)` containing the top-k results for a
///    single query. Each tuple represents a closest vector.
/// - `index`: The index of the query whose results are being inserted into the shared vector.
///    This `index` determines the segment in the `topk` vector where the `query_topk` results will be placed.
/// - `k`: The number of top results to manage per query. This defines both the size of `query_topk`
///    and the length of the segment in `topk` that will be replaced.
///
/// # Description
/// - If `query_top_k` does not have a length of `k`, it is resized to `k`, with any additional elements
///   filled with `(f32::MAX, usize::MAX)`.
/// - The function ensures that `topk_query` is resized correctly to the specified length `k`. It will panic
///   if the length is not as expected.
/// - The function locks the `topk` vector and replaces the segment from `index * k` to `(index + 1) * k` with
///   the `query_topk` results.
///
/// # Example
/// ```rust
/// use struttura_kANNolo::hnsw_utils::insert_into_topk;
/// use std::sync::Mutex;
///
/// // The vector holds results for two queries, each with 3 closest vectors (k=3).
/// let topk = Mutex::new(vec![
///     (1.0, 1),
///     (2.0, 2),
///     (3.0, 3),
///     (f32::MAX, usize::MAX),
///     (f32::MAX, usize::MAX),
///     (f32::MAX, usize::MAX),
///     ]);
///
/// let index = 1;
/// let k = 3;
///
/// //top-k closest vectors for query with index 1.
/// let query_topk = vec![(0.5, 5), (1.5, 6), (2.5, 7)];
///
/// insert_into_topk(&topk, query_topk, index, k);
///
/// let topk_locked = topk.lock().unwrap();
/// let expected = vec![(1.0, 1), (2.0, 2), (3.0, 3), (0.5, 5), (1.5, 6), (2.5, 7)];
/// assert_eq!(&*topk_locked, &expected);
/// ```
#[inline]
pub fn insert_into_topk(
    topk: &Mutex<Vec<(f32, usize)>>,
    mut query_topk: Vec<(f32, usize)>,
    index: usize,
    k: usize,
) {
    if query_topk.len() != k {
        query_topk.resize_with(k, || (f32::MAX, usize::MAX));
    }
    assert_eq!(
        query_topk.len(),
        k,
        "The length of the result vec has to be equal to k"
    );

    let start_index = index * k;
    topk.lock()
        .unwrap()
        .splice(start_index..start_index + k, query_topk);
}

#[inline]
pub fn prefetch_dense_vec_with_offset<T>(vector: &[T], offset: usize, len: usize) {
    let end = offset + len;

    for i in (offset..end).step_by(64 / std::mem::size_of::<T>()) {
        prefetch_read_T2(vector, i);
    }
}

#[allow(non_snake_case)]
#[inline]
pub fn prefetch_read_T2<T>(data: &[T], offset: usize) {
    let _p = data.as_ptr().wrapping_add(offset) as *const i8;

    //#[cfg(all(feature = "prefetch", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::{_mm_prefetch, _MM_HINT_T2};
            unsafe {
                _mm_prefetch(_p, _MM_HINT_T2);
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T2};
            unsafe {
                _mm_prefetch(_p, _MM_HINT_T2);
            }
        }
    }
}

/// Converts a `BinaryHeap<Node>` from a max-heap to a min-heap.
///
/// # Description
/// This function transforms a binary heap that is organized as a max-heap into a binary heap organized as a min-heap.
/// The conversion is done by first draining all elements from the max-heap, then wrapping each element in a `Reverse`
/// wrapper to reverse the heap order, and finally creating a new min-heap from these reversed elements.
///
/// # Parameters
/// - `max_heap`: A mutable reference to a `BinaryHeap<Node>` that is currently organized as a max-heap.
///
/// # Returns
/// - Returns a new `BinaryHeap<Reverse<Node>>` that is organized as a min-heap.
/// # Example
/// ```rust
/// use std::collections::BinaryHeap;
/// use std::cmp::Reverse;
/// use struttura_kANNolo::hnsw_utils::from_max_heap_to_min_heap;
/// use struttura_kANNolo::hnsw_utils::Node;
///
/// // Create a max-heap with some nodes.
/// let mut max_heap = BinaryHeap::new();
/// max_heap.push(Node(10.0,1));
/// max_heap.push(Node(20.0,2));
/// max_heap.push(Node(15.0,3));
///
/// // Convert the max-heap to a min-heap.
/// let min_heap = from_max_heap_to_min_heap(&mut max_heap);
///
/// // The new min-heap will contain the same nodes but in min-heap order.
/// assert_eq!(min_heap.peek().unwrap().0.1, 1); // The smallest element should be on top.
/// ```
pub fn from_max_heap_to_min_heap(max_heap: &mut BinaryHeap<Node>) -> BinaryHeap<Reverse<Node>> {
    let vec: Vec<_> = max_heap.drain().collect();
    BinaryHeap::from(vec.into_iter().map(Reverse).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests_from_max_heap_to_min_heap {
    use super::*;

    /// Tests the conversion of a max-heap to a min-heap with a standard set of elements.
    ///
    /// This test initializes a max-heap with a set of `Node` elements and then converts it to a min-heap.
    /// The result is checked to ensure that the elements are ordered correctly in the min-heap.
    #[test]
    fn test_from_max_heap_to_min_heap() {
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        max_heap.push(Node(3.2, 10));
        max_heap.push(Node(2.2, 8));
        max_heap.push(Node(6.2, 12));
        max_heap.push(Node(7.2, 2));
        max_heap.push(Node(32.2, 4));
        max_heap.push(Node(4.2, 14));
        max_heap.push(Node(1.2, 6));
        max_heap.push(Node(7.2, 6));

        let mut min_heap = from_max_heap_to_min_heap(&mut max_heap);
        let mut min_heap_vec: Vec<Node> = Vec::new();
        while let Some(node) = min_heap.pop() {
            min_heap_vec.push(node.0);
        }

        let expected_vec: Vec<Node> = vec![
            Node(1.2, 6),
            Node(2.2, 8),
            Node(3.2, 10),
            Node(4.2, 14),
            Node(6.2, 12),
            Node(7.2, 2),
            Node(7.2, 6),
            Node(32.2, 4),
        ];
        assert_eq!(expected_vec, min_heap_vec);
    }

    /// Tests the conversion of a max-heap to a min-heap with a large dataset.
    ///
    /// This test evaluates the performance and correctness of the conversion function when handling a large number of elements.
    /// It ensures that all elements are correctly converted from a max-heap to a min-heap and that the ordering is preserved.
    #[test]
    fn test_from_max_heap_to_min_heap_with_large_data() {
        let n = 100000;
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();

        for i in 0..n {
            max_heap.push(Node(i as f32, i));
        }

        let mut min_heap = from_max_heap_to_min_heap(&mut max_heap);
        let mut min_heap_vec: Vec<Node> = Vec::new();
        while let Some(node) = min_heap.pop() {
            min_heap_vec.push(node.0);
        }

        let mut expected_vec: Vec<Node> = Vec::new();
        for i in 0..n {
            expected_vec.push(Node(i as f32, i));
        }

        assert_eq!(expected_vec, min_heap_vec);
    }

    /// Tests the conversion of an empty max-heap to a min-heap.
    ///
    /// This test ensures that the function correctly handles the case where the max-heap is empty.
    /// The resulting min-heap should also be empty.
    #[test]
    fn test_from_max_heap_to_min_heap_empty() {
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let min_heap = from_max_heap_to_min_heap(&mut max_heap);
        assert!(min_heap.is_empty());
    }

    /// Tests the conversion of a max-heap to a min-heap with a single element.
    ///
    /// This test verifies that the function correctly converts a max-heap containing only one element to a min-heap,
    /// ensuring that the single element is correctly handled.
    #[test]
    fn test_from_max_heap_to_min_heap_single_element() {
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        max_heap.push(Node(42.0, 1));

        let mut min_heap = from_max_heap_to_min_heap(&mut max_heap);
        let node = min_heap.pop().unwrap().0;

        assert_eq!(node, Node(42.0, 1));
        assert!(min_heap.is_empty());
    }

    /// Tests the conversion of a max-heap to a min-heap where all elements are identical.
    ///
    /// This test ensures that the function correctly handles the case where all elements in the heap have the same value,
    /// and verifies that the min-heap retains the correct number of elements.
    #[test]
    fn test_from_max_heap_to_min_heap_all_elements_same() {
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        for _ in 0..10 {
            max_heap.push(Node(5.0, 100));
        }

        let mut min_heap = from_max_heap_to_min_heap(&mut max_heap);
        let mut min_heap_vec: Vec<Node> = Vec::new();
        while let Some(Reverse(node)) = min_heap.pop() {
            min_heap_vec.push(node);
        }

        let expected_vec: Vec<Node> = vec![Node(5.0, 100); 10];
        assert_eq!(expected_vec, min_heap_vec);
    }

    /// Tests the conversion of a max-heap to a min-heap with negative values.
    ///
    /// This test ensures that the function correctly handles and orders negative values
    /// when converting a max-heap to a min-heap.
    #[test]
    fn test_from_max_heap_to_min_heap_with_negative_values() {
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        max_heap.push(Node(-1.0, 1));
        max_heap.push(Node(-2.0, 2));
        max_heap.push(Node(-3.0, 3));

        let mut min_heap = from_max_heap_to_min_heap(&mut max_heap);
        let mut min_heap_vec: Vec<Node> = Vec::new();
        while let Some(Reverse(node)) = min_heap.pop() {
            min_heap_vec.push(node);
        }

        let expected_vec: Vec<Node> = vec![Node(-3.0, 3), Node(-2.0, 2), Node(-1.0, 1)];
        assert_eq!(expected_vec, min_heap_vec);
    }

    /// Tests the conversion of a max-heap to a min-heap with mixed values.
    ///
    /// This test verifies the function's correctness when handling a max-heap containing a mix of positive, negative,
    /// and zero values. The resulting min-heap should correctly reflect the min-heap order.
    #[test]
    fn test_from_max_heap_to_min_heap_with_mixed_values() {
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        max_heap.push(Node(0.0, 0));
        max_heap.push(Node(-1.0, 1));
        max_heap.push(Node(2.0, 2));
        max_heap.push(Node(-2.0, 3));
        max_heap.push(Node(1.0, 4));

        let mut min_heap = from_max_heap_to_min_heap(&mut max_heap);
        let mut min_heap_vec: Vec<Node> = Vec::new();
        while let Some(Reverse(node)) = min_heap.pop() {
            min_heap_vec.push(node);
        }

        let expected_vec: Vec<Node> = vec![
            Node(-2.0, 3),
            Node(-1.0, 1),
            Node(0.0, 0),
            Node(1.0, 4),
            Node(2.0, 2),
        ];
        assert_eq!(expected_vec, min_heap_vec);
    }
}

#[cfg(test)]
mod tests_insert_into_topk {
    use std::{
        panic,
        sync::{Arc, Mutex},
    };

    use crate::hnsw_utils::insert_into_topk;

    /// Tests the `insert_into_topk` function when the `topk_query` has exactly `k` elements and
    /// starts inserting at index 0.
    ///
    /// This test initializes the `topk` vector with 3 elements and a `topk_query` with exactly
    /// 3 elements. It inserts the `topk_query` results into the `topk` vector starting at index 0.
    /// The expected result should be the `topk_query` elements replacing the initial values in the
    /// `topk` vector.
    #[test]
    fn test_exact_size_query_start_index() {
        let topk = Arc::new(Mutex::new(vec![(1.0, 1), (2.0, 2), (3.0, 3)]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7)];
        let index = 0;
        let k = 3;

        insert_into_topk(&topk, topk_query, index, k);

        let topk_locked = topk.lock().unwrap();
        let expected = vec![(0.5, 5), (1.5, 6), (2.5, 7)];
        assert_eq!(&*topk_locked, &expected);
    }

    /// Tests the `insert_into_topk` function when the `topk_query` is smaller than `k`, starting insertion at index 0.
    ///
    /// This test initializes the `topk` vector with 3 elements and a `topk_query` with only 1 element. It inserts the
    /// `topk_query` results into the `topk` vector starting at index 0. The remaining slots in the segment are filled
    /// with default values.
    #[test]
    fn test_smaller_query_start_index() {
        let topk = Arc::new(Mutex::new(vec![(1.0, 1), (2.0, 2), (3.0, 3)]));
        let topk_query = vec![(0.5, 5)];
        let index = 0;
        let k = 3;

        insert_into_topk(&topk, topk_query, index, k);

        let topk_locked = topk.lock().unwrap();
        let expected = vec![(0.5, 5), (f32::MAX, usize::MAX), (f32::MAX, usize::MAX)];
        assert_eq!(&*topk_locked, &expected);
    }

    /// Tests the `insert_into_topk` function with a `topk_query` that is larger than `k`, starting insertion at index 0.
    ///
    /// Initializes a `topk` vector with some values and a `topk_query` that contains more elements than `k`. The function
    /// inserts the first `k` elements of `topk_query` starting at index 0 of `topk`. Since `topk_query` has more
    /// elements than `k`, only the first `k` elements of `topk_query` are inserted, replacing the default values in `topk`.
    #[test]
    fn test_larger_query_start_index() {
        let topk = Arc::new(Mutex::new(vec![
            (1.0, 1),
            (2.0, 2),
            (3.0, 3),
            (6.0, 4),
            (8.0, 5),
            (9.0, 6),
        ]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7), (3.5, 8)];
        let index = 0;
        let k = 3;

        insert_into_topk(&topk, topk_query, index, k);

        let topk_locked = topk.lock().unwrap();
        let expected = vec![(0.5, 5), (1.5, 6), (2.5, 7), (6.0, 4), (8.0, 5), (9.0, 6)];
        assert_eq!(&*topk_locked, &expected);
    }

    /// Tests the `insert_into_topk` function when the `topk_query` is empty, starting insertion at index 0.
    ///
    /// This test initializes the `topk` vector with 3 elements and an empty `topk_query`. It inserts
    /// the `topk_query` results into the `topk` vector starting at index 0. Since the `topk_query`
    /// is empty, all elements in the `topk` vector should be replaced with default values.
    #[test]
    fn test_empty_query_start_index() {
        let topk = Arc::new(Mutex::new(vec![(1.0, 1), (2.0, 2), (3.0, 3)]));
        let topk_query = vec![];
        let index = 0;
        let k = 3;

        insert_into_topk(&topk, topk_query, index, k);

        let topk_locked = topk.lock().unwrap();
        let expected = vec![
            (f32::MAX, usize::MAX),
            (f32::MAX, usize::MAX),
            (f32::MAX, usize::MAX),
        ];
        assert_eq!(&*topk_locked, &expected);
    }

    /// Tests the `insert_into_topk` function when the `topk_query` has exactly `k` elements and starts
    /// inserting at a middle index.
    ///
    /// This test initializes the `topk` vector with 100 elements, all set to default values. It then
    /// creates a `topk_query` with 5 elements and inserts it into the `topk` vector starting at index 10.
    /// The expected result is that the elements in the `topk` vector at positions  corresponding to the middle
    /// index will be replaced with the `topk_query` elements, while the other elements remain unchanged.
    #[test]
    fn test_exact_size_query_middle_index() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 100]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7), (2.7, 1), (3.1, 3)];
        let index = 10;
        let k = 5;

        insert_into_topk(&topk, topk_query.clone(), index, k);

        let start_index = index * k;

        for i in 0..k {
            assert_eq!(topk.lock().unwrap()[i + start_index], topk_query[i]);
        }
    }
    /// Tests the `insert_into_topk` function when the `topk_query` is smaller than `k` and starts
    /// inserting at a middle index.
    ///
    /// This test initializes the `topk` vector with 1000 elements, all set to default values.
    /// It then creates a `topk_query` with only 3 elements and inserts it into the `topk` vector
    /// starting at index 20. The expected result is that the `topk_query` elements will replace
    /// the initial values in the `topk` vector at the corresponding positions,
    /// while the remaining slots in the segment are filled with default values.
    #[test]
    fn test_smaller_query_middle_index() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 1000]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7)];
        let index = 20;
        let k = 5;

        insert_into_topk(&topk, topk_query.clone(), index, k);

        let expected = vec![
            (0.5, 5),
            (1.5, 6),
            (2.5, 7),
            (f32::MAX, usize::MAX),
            (f32::MAX, usize::MAX),
        ];

        let start_index = index * k;

        for i in 0..k {
            assert_eq!(topk.lock().unwrap()[i + start_index], expected[i]);
        }
    }

    /// Tests the `insert_into_topk` function when the `topk_query` is larger than `k` and
    /// starts inserting at a middle index.
    ///
    /// This test initializes the `topk` vector with 200 elements, all set to default values.
    /// It then creates a `topk_query` with 6 elements and inserts the first `k` elements into
    /// the `topk` vector starting at index 20. The expected result is that only the first `k` elements
    /// from the `topk_query` will replace the initial values in the `topk` vector at the corresponding
    /// positions, leaving the other elements unchanged.
    #[test]
    fn test_bigger_query_middle_index() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 200]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7), (4.5, 3), (5.9, 1), (12.5, 72)];
        let index = 20;
        let k = 5;

        insert_into_topk(&topk, topk_query.clone(), index, k);

        let expected = vec![(0.5, 5), (1.5, 6), (2.5, 7), (4.5, 3), (5.9, 1)];

        let start_index = index * k;

        for i in 0..k {
            assert_eq!(topk.lock().unwrap()[i + start_index], expected[i]);
        }
    }

    /// Tests the `insert_into_topk` function when the `topk_query` has exactly `k` elements
    /// and starts inserting at the last possible index.
    ///
    /// This test initializes the `topk` vector with 200 elements, all set to default values.
    /// It then creates a `topk_query` with 5 elements and inserts it into the `topk` vector starting
    /// at index 39 (the last valid index for inserting `k` elements). The expected result
    /// is that the `topk_query` elements will replace the values in the `topk` vector at the
    /// corresponding positions, while the rest of the vector remains unchanged.
    #[test]
    fn test_exact_size_query_last_index() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 200]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7), (4.5, 3), (5.9, 1)];
        let index = 39;
        let k = 5;

        insert_into_topk(&topk, topk_query.clone(), index, k);

        let start_index = index * k;

        for i in 0..k {
            assert_eq!(topk.lock().unwrap()[i + start_index], topk_query[i]);
        }
    }

    /// Tests the `insert_into_topk` function when the `topk_query` is smaller than `k`
    /// and starts inserting at the last possible index.
    ///
    /// This test initializes the `topk` vector with 200 elements, all set to default values.
    /// It then creates a `topk_query` with 2 elements and inserts it into the `topk` vector starting
    /// at index 39 (the last valid index for inserting `k` elements). The expected result
    /// is that the `topk_query` elements will replace the values in the `topk` vector at the corresponding
    /// positions, while the remaining slots in the segment are filled with default values.
    #[test]
    fn test_smaller_query_last_index() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 200]));
        let topk_query = vec![(0.5, 5), (1.5, 6)];
        let index = 39;
        let k = 5;

        insert_into_topk(&topk, topk_query.clone(), index, k);

        let expected = vec![
            (0.5, 5),
            (1.5, 6),
            (f32::MAX, usize::MAX),
            (f32::MAX, usize::MAX),
            (f32::MAX, usize::MAX),
        ];

        let start_index = index * k;

        for i in 0..k {
            assert_eq!(topk.lock().unwrap()[i + start_index], expected[i]);
        }
    }

    /// Tests the `insert_into_topk` function when the `topk_query` is larger than `k` and
    /// starts inserting at the last possible index.
    ///
    /// This test initializes the `topk` vector with 200 elements, all set to default values.
    /// It then creates a `topk_query` with 6 elements and inserts the first `k` elements into the `topk`
    /// vector starting at index 39 (the last valid index for inserting `k` elements).
    /// The expected result is that only the first `k` elements from the `topk_query` will replace
    /// the values in the `topk` vector at the corresponding positions, while the rest of the vector remains unchanged.
    #[test]
    fn test_bigger_query_last_index() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 200]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7), (4.5, 3), (5.9, 1), (12.5, 72)];
        let index = 39;
        let k = 5;

        insert_into_topk(&topk, topk_query.clone(), index, k);

        let start_index = index * k;

        for i in 0..k {
            assert_eq!(topk.lock().unwrap()[i + start_index], topk_query[i]);
        }
    }

    /// Tests the `insert_into_topk` function when the `index` provided for insertion is out of range.
    ///
    /// This test initializes the `topk` vector with 200 elements, all set to default values. It then
    /// attempts to insert a `topk_query` with 6 elements into the `topk` vector starting at index 40,
    /// which is out of range. The expected behavior is that the function should panic. After the panic,
    /// the contents of the `topk` vector should remain unchanged, ensuring that no partial modifications
    /// are made in the event of an out-of-bounds index.
    #[test]
    fn test_index_out_of_range() {
        let topk = Arc::new(Mutex::new(vec![(f32::MAX, usize::MAX); 200]));
        let topk_query = vec![(0.5, 5), (1.5, 6), (2.5, 7), (4.5, 3), (5.9, 1), (12.5, 72)];
        let index = 40; // This index is out of range
        let k = 5;

        let topk_initial = topk.lock().unwrap().clone();

        let result = panic::catch_unwind(|| {
            insert_into_topk(&topk, topk_query.clone(), index, k);
        });

        // Ensure the function panicked
        assert!(result.is_err(), "Expected panic, but code did not panic");

        let topk_after = topk.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        assert_eq!(
            *topk_after, topk_initial,
            "Topk should remain unchanged after panic"
        );
    }
}

#[cfg(test)]
mod tests_add_neighbors_to_heaps {
    use super::*;
    use std::collections::BinaryHeap;

    /// Tests the `add_neighbor_to_heaps` function when both heaps are empty.
    ///
    /// This test initializes empty `min_heap` and `max_heap`, and adds a single `Node` with a
    /// distance of 10.0 and an ID of 1.
    /// The `ef_parameter` is set to 3, which allows for up to 3 nodes in the heaps.
    /// The expected result is that both heaps will contain the new node, and the sizes of
    /// both heaps should be 1.

    #[test]
    fn test_add_to_empty_heaps() {
        let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let node = Node(10.0, 1);
        let ef_parameter = 3;

        add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);

        assert_eq!(min_heap.len(), 1);
        assert_eq!(max_heap.len(), 1);
        assert_eq!(min_heap.peek(), Some(&Reverse(node)));
        assert_eq!(max_heap.peek(), Some(&node));
    }

    /// Tests the `add_neighbor_to_heaps` function when the number of nodes is within the `ef_parameter` limit.
    ///
    /// This test initializes empty `min_heap` and `max_heap`, and adds three nodes.
    /// The `ef_parameter` is set to 3, allowing all nodes to be added to both heaps. The expected result
    /// is that both heaps will have 3 elements, with the top of the `max_heap` containing the largest
    /// distance, and the top of the `min_heap` containing the smallest distance.
    #[test]
    fn test_add_within_ef_parameter_limit() {
        let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let ef_parameter = 3;

        let nodes = vec![Node(10.0, 1), Node(5.0, 2), Node(7.0, 3)];

        for node in nodes.iter().cloned() {
            add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);
        }

        assert_eq!(min_heap.len(), 3);
        assert_eq!(max_heap.len(), 3);
        assert_eq!(max_heap.peek().unwrap().distance(), 10.0);
        assert_eq!(min_heap.peek().unwrap().0.distance(), 5.0);
    }

    /// Tests the `add_neighbor_to_heaps` function when the number of nodes exceeds the `ef_parameter` limit.
    ///
    /// This test initializes empty `min_heap` and `max_heap`, and adds four nodes with distances 10.0, 5.0, 7.0, and 3.0.
    /// The `ef_parameter` is set to 3, so the `max_heap` should only hold the 3 closest nodes, with the largest of those
    /// distances at the top. The `min_heap` should contain all 4 nodes, with the smallest distance at the top.
    /// The expected result is that the top of the `max_heap` will contain the distance 7.0, and the top of the `min_heap`
    /// will contain the distance 3.0.
    #[test]
    fn test_add_beyond_ef_parameter_limit() {
        let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let ef_parameter = 3;

        let nodes = vec![Node(10.0, 1), Node(5.0, 2), Node(7.0, 3), Node(3.0, 4)];

        for node in nodes.iter().cloned() {
            add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);
        }

        assert_eq!(min_heap.len(), 4);
        assert_eq!(max_heap.len(), 3);
        assert_eq!(max_heap.peek().unwrap().distance(), 7.0);
        assert_eq!(min_heap.peek().unwrap().0.distance(), 3.0);
    }

    /// Tests the `add_neighbor_to_heaps` function when adding a node with a higher distance than the current maximum.
    ///
    /// This test initializes empty `min_heap` and `max_heap`, and adds two nodes with distances 5.0 and 3.0.
    /// The `ef_parameter` is set to 2, allowing only 2 nodes to be included in the heaps. Then,
    /// a new node with a distance of 8.0 is added. The expected result is that the heaps remain unchanged,
    /// as the new node's distance is greater than the current maximum at the top of the `max_heap`, which is 5.0.
    #[test]
    fn test_add_node_with_higher_distance_than_current_max() {
        let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let ef_parameter = 2;

        let nodes = vec![Node(5.0, 1), Node(3.0, 2)];

        for node in nodes.iter().cloned() {
            add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);
        }

        let new_node = Node(8.0, 3);
        add_neighbor_to_heaps(&mut min_heap, &mut max_heap, new_node, ef_parameter);

        // No change should be made, as the new node's distance is greater than current max
        assert_eq!(min_heap.len(), 2);
        assert_eq!(max_heap.len(), 2);
        assert_eq!(max_heap.peek().unwrap().distance(), 5.0);
        assert_eq!(min_heap.peek().unwrap().0.distance(), 3.0);
    }

    /// Tests the `add_neighbor_to_heaps` function with an edge case where `ef_parameter` is set to 1.
    ///
    /// This test initializes empty `min_heap` and `max_heap`, and adds three nodes with distances 5.0, 3.0, and 2.0.
    /// The `ef_parameter` is set to 1, allowing only 1 node to be included in the `max_heap`. The expected result is that
    /// the top of the `max_heap` contains only the node with the smallest distance (2.0), while the `min_heap` contains
    /// all 3 nodes, with the smallest distance node at the top.
    #[test]
    fn test_ef_parameter_of_one() {
        let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let ef_parameter = 1;

        let nodes = vec![Node(5.0, 1), Node(3.0, 2), Node(2.0, 3)];

        for node in nodes.iter().cloned() {
            add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);
        }

        assert_eq!(min_heap.len(), 3);
        assert_eq!(max_heap.len(), 1);
        assert_eq!(max_heap.peek().unwrap().distance(), 2.0);
        assert_eq!(min_heap.peek().unwrap().0.distance(), 2.0);
    }

    /// Tests the `add_neighbor_to_heaps` function when adding a node with a distance equal to the current maximum.
    ///
    /// This test initializes empty `min_heap` and `max_heap`, and adds two nodes with distances 5.0 and 3.0.
    /// The `ef_parameter` is set to 2, allowing both nodes to be included in the heaps. Then, a new node with the same
    /// distance as the current maximum (5.0) is added. The expected result is that the heaps remain unchanged, with
    /// the top of the `max_heap` containing the distance 5.0, and the top of the `min_heap` containing the distance 3.0.
    #[test]
    fn test_add_node_equal_to_current_max() {
        let mut min_heap: BinaryHeap<Reverse<Node>> = BinaryHeap::new();
        let mut max_heap: BinaryHeap<Node> = BinaryHeap::new();
        let ef_parameter = 2;

        let nodes = vec![Node(5.0, 1), Node(3.0, 2)];

        for node in nodes.iter().cloned() {
            add_neighbor_to_heaps(&mut min_heap, &mut max_heap, node, ef_parameter);
        }

        let new_node = Node(5.0, 3);
        add_neighbor_to_heaps(&mut min_heap, &mut max_heap, new_node, ef_parameter);

        assert_eq!(min_heap.len(), 2);
        assert_eq!(max_heap.len(), 2);
        assert_eq!(max_heap.peek().unwrap().distance(), 5.0);
        assert_eq!(min_heap.peek().unwrap().0.distance(), 3.0);
    }
}

#[cfg(test)]
mod tests_compute_closest_from_neighbors_euclidean_distance {

    use core::f32;

    use crate::{
        hnsw_utils::compute_closest_from_neighbors, plain_quantizer::PlainQuantizer, Dataset,
        DenseDataset, DenseVector1D, DistanceType,
    };

    /// Tests that `compute_closest_from_neighbors` correctly updates the nearest neighbor.
    ///
    /// This test verifies that the function correctly identifies and updates the nearest neighbor
    /// from a set of multiple neighbor vectors.
    #[test]
    fn test_compute_closest_from_neighbors_updates_nearest() {
        let query_vector: &[f32] = &[1.0, 1.0];
        let query = DenseVector1D::new(query_vector);

        let neighbor_vectors = &[2.0, 2.0, 1.0, 1.5, 0.0, 0.0];

        let quantizer = PlainQuantizer::new(2, DistanceType::Euclidean);
        let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);

        let evaluator = dataset.query_evaluator(query);

        let neighbors = vec![0, 1, 2];

        let mut nearest_vec = 0;
        let mut dis_nearest_vec = f32::MAX;

        compute_closest_from_neighbors(
            &dataset,
            &evaluator,
            &neighbors,
            &mut nearest_vec,
            &mut dis_nearest_vec,
        );

        assert_eq!(nearest_vec, 1);
    }

    /// Tests `compute_closest_from_neighbors` when there is only a single neighbor.
    ///
    /// This test checks that the function can handle the case where there is only one neighbor
    /// and correctly identifies it as the nearest.
    #[test]
    fn test_compute_closest_from_neighbors_single_neighbor() {
        let query_vector: &[f32] = &[1.0, 1.0];
        let query = DenseVector1D::new(query_vector);

        let neighbor_vectors = &[2.0, 2.0];

        let quantizer = PlainQuantizer::new(2, crate::DistanceType::Euclidean);
        let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);

        let evaluator = dataset.query_evaluator(query);

        let neighbors = vec![0];

        let mut nearest_vec = 0;
        let mut dis_nearest_vec = f32::MAX;

        compute_closest_from_neighbors(
            &dataset,
            &evaluator,
            &neighbors,
            &mut nearest_vec,
            &mut dis_nearest_vec,
        );

        assert_eq!(nearest_vec, 0);
    }

    /// Tests `compute_closest_from_neighbors` when multiple neighbors are equidistant from the query.
    ///
    /// This test examines the behavior of the function when several neighbors are equidistant,
    /// ensuring that the first equidistant vector is selected as the closest, since the function does
    /// not replace it with other equidistant neighbors.
    #[test]
    fn test_compute_closest_from_neighbors_equidistant_neighbors() {
        let query_vector: &[f32] = &[1.0, 1.0];
        let query = DenseVector1D::new(query_vector);

        // Neighbor vectors that are equidistant from the query except for the first one
        let neighbor_vectors = &[2.0, 5.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0];

        let quantizer = PlainQuantizer::new(2, crate::DistanceType::Euclidean);
        let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);

        let evaluator = dataset.query_evaluator(query);

        let neighbors = vec![0, 1, 2, 3];

        let mut nearest_vec = 0;
        let mut dis_nearest_vec = f32::MAX;

        compute_closest_from_neighbors(
            &dataset,
            &evaluator,
            &neighbors,
            &mut nearest_vec,
            &mut dis_nearest_vec,
        );

        assert_eq!(nearest_vec, 1);
    }

    /// Tests `compute_closest_from_neighbors` when there are no neighbors to compare.
    ///
    /// This test verifies that the function handles the case of having no neighbors by
    /// ensuring that the nearest neighbor remains unchanged.
    #[test]
    fn test_compute_closest_from_neighbors_no_neighbors() {
        let query_vector: &[f32] = &[1.0, 1.0];
        let query = DenseVector1D::new(query_vector);

        let neighbor_vectors = &[];

        let quantizer = PlainQuantizer::new(2, crate::DistanceType::Euclidean);
        let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);

        let evaluator = dataset.query_evaluator(query);
        let neighbors: Vec<usize> = vec![];

        let mut nearest_vec = 0;
        let mut dis_nearest_vec = f32::MAX;

        compute_closest_from_neighbors(
            &dataset,
            &evaluator,
            &neighbors,
            &mut nearest_vec,
            &mut dis_nearest_vec,
        );

        // Nothing should change because there are no neighbors
        assert_eq!(nearest_vec, 0);
        assert_eq!(dis_nearest_vec, f32::MAX);
    }

    /// Tests `compute_closest_from_neighbors` when the neighbor distances are set to extreme values.
    ///
    /// This test checks how the function behaves when neighbor vectors are at extreme distances,
    /// ensuring that no changes are made to the nearest neighbor if none are closer.
    #[test]
    fn test_compute_closest_from_neighbors_max_distance() {
        let query_vector: &[f32] = &[1.0, 1.0];
        let query = DenseVector1D::new(query_vector);

        let neighbor_vectors = &[f32::INFINITY, f32::INFINITY, -f32::INFINITY, -f32::INFINITY];

        let quantizer = PlainQuantizer::new(2, crate::DistanceType::Euclidean);
        let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);

        let evaluator = dataset.query_evaluator(query);

        let neighbors = vec![0, 1];

        let mut nearest_vec = 0;
        let mut dis_nearest_vec = f32::MAX;

        compute_closest_from_neighbors(
            &dataset,
            &evaluator,
            &neighbors,
            &mut nearest_vec,
            &mut dis_nearest_vec,
        );

        // The nearest neighbor should remain unchanged, as none are closer
        assert_eq!(nearest_vec, 0);
        assert_eq!(dis_nearest_vec, f32::MAX);
    }

    /// Tests `compute_closest_from_neighbors` when one of the neighbors is an exact match with the query vector.
    ///
    /// This test confirms that the function can correctly identify a neighbor that exactly matches
    /// the query vector and updates the nearest neighbor accordingly.

    #[test]
    fn test_compute_closest_from_neighbors_exact_match() {
        let query_vector: &[f32] = &[1.0, 1.0];
        let query = DenseVector1D::new(query_vector);

        // One neighbor is exactly as the query vecor
        let neighbor_vectors = &[2.0, 2.0, 1.0, 1.0];

        let quantizer = PlainQuantizer::new(2, crate::DistanceType::Euclidean);
        let dataset = DenseDataset::from_vec(neighbor_vectors.to_vec(), 2, quantizer);

        let evaluator = dataset.query_evaluator(query);

        let neighbors = vec![0, 1];

        let mut nearest_vec = 0;
        let mut dis_nearest_vec = f32::MAX;

        compute_closest_from_neighbors(
            &dataset,
            &evaluator,
            &neighbors,
            &mut nearest_vec,
            &mut dis_nearest_vec,
        );

        assert_eq!(nearest_vec, 1);
        assert_eq!(dis_nearest_vec, 0.0);
    }
}
