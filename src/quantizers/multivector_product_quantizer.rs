use crate::quantizers::pq::ProductQuantizer;
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{Dataset, DistanceType, Float, MultiVector, PlainDenseDataset};
use crate::{DotProduct, EuclideanDistance};

use crate::datasets::multivector_dataset::MultiVectorDataset;
use crate::utils::{sgemm, MatrixLayout};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A Product Quantizer specifically designed for MultiVector data.
///
/// This quantizer treats all dense vectors from all multivectors as a single training set
/// for Product Quantization. Each multivector is then encoded as a collection of quantized codes.
///
/// # Type Parameters
/// - `M`: Number of subspaces for product quantization (must be divisible by 4)
/// - `T`: The input data type (typically f32)
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiVectorProductQuantizer<const M: usize, T> {
    /// The core ProductQuantizer trained on all dense vectors
    product_quantizer: ProductQuantizer<M>,
    /// Dimension of individual vectors
    vector_dim: usize,
    /// Distance type used for comparisons
    distance: DistanceType,
    /// Phantom data for type parameter
    _phantom: PhantomData<T>,
}

impl<const M: usize, T> MultiVectorProductQuantizer<M, T> {
    #[inline]
    pub fn new(
        vector_dim: usize,
        nbits: usize,
        distance: DistanceType,
        training_data: &MultiVectorDataset<
            crate::quantizers::multivector_plain_quantizer::MultiVectorPlainQuantizer<T>,
        >,
    ) -> Self
    where
        T: Float + Copy + Default + PartialOrd + Sync + Send,
    {
        // Extract all dense vectors from the multivector dataset for training
        let dense_training_data = Self::extract_dense_vectors_for_training(training_data);

        // Train the product quantizer on all dense vectors
        let product_quantizer = ProductQuantizer::<M>::train(&dense_training_data, nbits, distance);

        MultiVectorProductQuantizer {
            product_quantizer,
            vector_dim,
            distance,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn from_pretrained(
        product_quantizer: ProductQuantizer<M>,
        vector_dim: usize,
        distance: DistanceType,
    ) -> Self {
        MultiVectorProductQuantizer {
            product_quantizer,
            vector_dim,
            distance,
            _phantom: PhantomData,
        }
    }

    /// Extract all dense vectors from multivector dataset to create a plain dense dataset for PQ training
    fn extract_dense_vectors_for_training(
        multivector_dataset: &MultiVectorDataset<
            crate::quantizers::multivector_plain_quantizer::MultiVectorPlainQuantizer<T>,
        >,
    ) -> PlainDenseDataset<f32>
    where
        T: Float + Copy,
    {
        let all_data = multivector_dataset.data();
        let vector_dim = multivector_dataset.dim();
        let total_vectors = all_data.num_vectors();

        // Convert all data to f32 for PQ training
        let mut dense_data = Vec::<f32>::with_capacity(total_vectors * vector_dim);

        for vector in all_data.iter_vectors() {
            for &value in vector {
                dense_data.push(value.to_f32().unwrap());
            }
        }

        PlainDenseDataset::<f32>::from_vec_plain(dense_data, vector_dim)
    }

    #[inline]
    pub fn product_quantizer(&self) -> &ProductQuantizer<M> {
        &self.product_quantizer
    }
}

impl<const M: usize, T: Copy + Default + PartialOrd + Sync + Send + Float> Quantizer
    for MultiVectorProductQuantizer<M, T>
{
    type InputItem = T;
    type OutputItem = u8;
    type DatasetType = MultiVectorDataset<Self>;

    type Evaluator<'a>
        = MultiVectorQueryEvaluatorPQ<'a, M, Self::InputItem>
    where
        Self::InputItem: Float + EuclideanDistance<T> + DotProduct<T> + 'a;

    #[inline]
    fn encode(&self, input_vectors: &[Self::InputItem], output_vectors: &mut [Self::OutputItem]) {
        // Convert input vectors to f32 for PQ encoding
        let mut f32_vectors = Vec::<f32>::with_capacity(input_vectors.len());
        for &value in input_vectors {
            f32_vectors.push(value.to_f32().unwrap());
        }

        // Each input vector gets quantized to M codes
        // input_vectors.len() = num_vectors * vector_dim
        // output_vectors.len() should be = num_vectors * M
        let vector_dim = M * self.product_quantizer.dsub(); // d = M * dsub
        let num_vectors = input_vectors.len() / vector_dim;
        let expected_output_size = num_vectors * M;

        assert_eq!(
            output_vectors.len(),
            expected_output_size,
            "Output vector size mismatch: expected {}, got {}",
            expected_output_size,
            output_vectors.len()
        );

        // Use the underlying ProductQuantizer to encode
        self.product_quantizer.encode(&f32_vectors, output_vectors);
    }

    #[inline]
    fn m(&self) -> usize {
        M
    }

    #[inline]
    fn distance(&self) -> DistanceType {
        self.distance
    }

    fn get_space_usage_bytes(&self) -> usize {
        self.product_quantizer.get_space_usage_bytes()
            + 2 * std::mem::size_of::<usize>()
            + std::mem::size_of::<DistanceType>()
    }
}

/// Query Evaluator for MultiVector Product Quantization
///
/// This evaluator computes distance tables efficiently by processing one subspace at a time
/// across all query vectors, using matrix multiplication for optimal performance.
pub struct MultiVectorQueryEvaluatorPQ<'a, const M: usize, T: Float> {
    _phantom: std::marker::PhantomData<&'a T>,
    /// Distance tables for each query vector, stored as [query_vector_idx][subspace_idx * ksub + centroid_idx]
    distance_tables: Vec<f32>, // flattened: num_query_vectors x (M * ksub)
    ksub: usize,
}

impl<'a, const M: usize, T: Float + DotProduct<T> + Copy> QueryEvaluator<'a>
    for MultiVectorQueryEvaluatorPQ<'a, M, T>
{
    type Q = MultiVectorProductQuantizer<M, T>;
    type QueryType = MultiVector<&'a [T]>;

    #[inline]
    fn new(query: Self::QueryType, dataset: &<Self::Q as Quantizer>::DatasetType) -> Self {
        let quantizer = dataset.quantizer();
        let product_quantizer = quantizer.product_quantizer();
        let ksub = product_quantizer.ksub();

        // Compute distance tables efficiently: one subspace at a time
        let distance_tables = Self::compute_distance_tables_efficient(&query, product_quantizer);

        Self {
            _phantom: std::marker::PhantomData,
            distance_tables,
            ksub,
        }
    }

    #[inline]
    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32 {
        let document = dataset.get(index);

        match dataset.quantizer().distance() {
            DistanceType::Euclidean => {
                panic!("Euclidean distance not supported for multivectors")
            }
            DistanceType::DotProduct => {
                // Compute MaxSim using the distance tables
                self.compute_maxsim_with_tables(&document)
            }
        }
    }

    #[inline]
    fn topk_retrieval<I, H>(&self, distances: I, heap: &mut H) -> Vec<(f32, usize)>
    where
        I: Iterator<Item = f32>,
        H: OnlineTopKSelector,
    {
        for distance in distances {
            heap.push(distance);
        }

        heap.topk()
    }
}

impl<'a, const M: usize, T: Float + Copy> MultiVectorQueryEvaluatorPQ<'a, M, T> {
    /// Efficiently compute distance tables for all query vectors
    ///
    /// This function computes distance tables one subspace at a time using matrix multiplication:
    /// - For each subspace m: multiply all query vectors' m-th subspace against all centroids of subspace m
    /// - This gives us the m-th row/column of each query vector's distance table
    pub fn compute_distance_tables_efficient(
        query: &MultiVector<&'a [T]>,
        product_quantizer: &ProductQuantizer<M>,
    ) -> Vec<f32> {
        let num_query_vectors = query.num_vectors();
        let dsub = product_quantizer.dsub();
        let ksub = product_quantizer.ksub();

        let vector_dim = M * dsub;

        // Pre-convert all query vectors to f32 in a single flat buffer: [Q x vector_dim]
        let mut q_flat = vec![0f32; num_query_vectors * vector_dim];
        for q_idx in 0..num_query_vectors {
            let qv = query.get_vector(q_idx);
            let base = q_idx * vector_dim;
            for i in 0..vector_dim {
                q_flat[base + i] = qv[i].to_f32().unwrap();
            }
        }

        // Flattened distance tables: [Q x (M * ksub)]
        let stride = M * ksub;
        let mut distance_tables = vec![0f32; num_query_vectors * stride];

        // Reused buffers for packing the current subspace from all queries
        let mut query_subspaces = vec![0f32; num_query_vectors * dsub];

        for m in 0..M {
            // Pack m-th subspace into contiguous [Q x dsub]
            let sub_src_offset = m * dsub;
            for q_idx in 0..num_query_vectors {
                let src_base = q_idx * vector_dim + sub_src_offset;
                let dst_base = q_idx * dsub;
                query_subspaces[dst_base..dst_base + dsub]
                    .copy_from_slice(&q_flat[src_base..src_base + dsub]);
            }

            // Get centroids for this subspace
            let centroids_start = m * ksub * dsub;
            let centroids_end = centroids_start + ksub * dsub;
            let centroids_subspace = &product_quantizer.centroids()[centroids_start..centroids_end];

            match product_quantizer.distance() {
                DistanceType::DotProduct => {
                    // For dot product, we use SGEMM: C = A * B^T
                    // A = query_subspaces [num_query_vectors x dsub]
                    // B^T = centroids [dsub x ksub]
                    // We want GEMM to write directly into distance_tables at column block m*ksub
                    let alpha = 1.0f32;
                    let beta = 0.0f32;

                    // C pointer points to the start of the m-th block inside each query row
                    let c_ptr = unsafe { distance_tables.as_mut_ptr().add(m * ksub) };
                    let ldc = (stride) as isize; // row stride for distance_tables: M * ksub

                    sgemm(
                        MatrixLayout::RowMajor,
                        false,
                        true,
                        alpha,
                        beta,
                        num_query_vectors,
                        dsub,
                        ksub,
                        query_subspaces.as_ptr(),
                        dsub as isize,
                        centroids_subspace.as_ptr(),
                        dsub as isize,
                        c_ptr,
                        ldc,
                    );
                }
                DistanceType::Euclidean => {
                    panic!("Euclidean distance not supported for multivectors");
                }
            }
        }

        distance_tables
    }

    /// Compute MaxSim between query and document using precomputed distance tables
    fn compute_maxsim_with_tables(&self, document: &MultiVector<&[u8]>) -> f32 {
        let mut total_similarity = 0.0;
        // For each query vector (each has its own distance table stored flattened)
        let stride = M * self.ksub;
        let num_query_vectors = if stride > 0 {
            self.distance_tables.len() / stride
        } else {
            0
        };

        for q_idx in 0..num_query_vectors {
            let table_base = q_idx * stride;
            let query_table = &self.distance_tables[table_base..table_base + stride];

            let mut max_similarity_for_this_query = f32::NEG_INFINITY;

            // Find maximum similarity with any document vector for this query vector
            for doc_vec_idx in 0..document.num_vectors() {
                let doc_vector = document.get_vector(doc_vec_idx);

                // Compute similarity using lookup table
                let mut similarity = 0.0f32;
                for (m, &code) in doc_vector.iter().enumerate() {
                    similarity += query_table[m * self.ksub + code as usize];
                }

                // Track maximum for this query vector
                max_similarity_for_this_query = max_similarity_for_this_query.max(similarity);
            }

            // Sum the maximum similarities (MaxSim = sum of per-query-vector maxima)
            total_similarity += max_similarity_for_this_query;
        }

        // Return negative for min-heap behavior (as in the plain quantizer)
        -total_similarity
    }
}
