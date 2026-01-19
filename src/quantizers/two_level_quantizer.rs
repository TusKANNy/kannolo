use crate::quantizers::pq::ProductQuantizer;
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{Dataset, DistanceType, Float, MultiVector};
use crate::{DotProduct, EuclideanDistance};

use crate::datasets::multivector_dataset::MultiVectorDataset;

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Two-level quantizer: coarse (IVF) + global PQ on residuals.
///
/// This implementation assumes the first-level components (coarse centroids,
/// per-vector coarse assignments, and PQ codes) are precomputed and provided
/// when constructing the quantizer. Training and encoding are intentionally
/// out of scope — this struct focuses on search (ADC table computation and
/// scoring) using existing components.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TwoLevelQuantizer<const M: usize, T> {
    /// Coarse centroids, stored flattened as [ncoarse x dim]
    pub coarse_centroids: Vec<f32>,
    /// Number of coarse centroids
    pub ncoarse: usize,
    /// Dimension of vectors
    pub vector_dim: usize,
    /// Product quantizer trained on residuals (global)
    pub product_quantizer: ProductQuantizer<M>,
    /// Distance type
    pub distance: DistanceType,
    _phantom: PhantomData<T>,
}

impl<const M: usize, T> TwoLevelQuantizer<M, T> {
    /// Construct from pretrained components.
    pub fn from_pretrained(
        coarse_centroids: Vec<f32>,
        ncoarse: usize,
        vector_dim: usize,
        product_quantizer: ProductQuantizer<M>,
        distance: DistanceType,
    ) -> Self {
        TwoLevelQuantizer {
            coarse_centroids,
            ncoarse,
            vector_dim,
            product_quantizer,
            distance,
            _phantom: PhantomData,
        }
    }

    /// Helper: get reference to PQ
    pub fn product_quantizer(&self) -> &ProductQuantizer<M> {
        &self.product_quantizer
    }
}

impl<const M: usize, T: Copy + Default + PartialOrd + Sync + Send + Float> Quantizer
    for TwoLevelQuantizer<M, T>
{
    type InputItem = T;
    type OutputItem = u8;
    type DatasetType = MultiVectorDataset<Self>;
    // Each encoded vector stores 4 bytes (u32 little-endian) for the coarse id
    // followed by M bytes for PQ codes. Therefore the quantized output size per
    // vector is M + 4.
    type Evaluator<'a>
        = TwoLevelQueryEvaluator<'a, M, Self::InputItem>
    where
        Self::InputItem: Float + EuclideanDistance<T> + DotProduct<T> + 'a;

    fn m(&self) -> usize {
        // encoded size per vector: 4 bytes for coarse id + M codes
        M + 4
    }

    fn distance(&self) -> DistanceType {
        self.distance
    }

    fn get_space_usage_bytes(&self) -> usize {
        self.product_quantizer.get_space_usage_bytes()
            + self.coarse_centroids.len() * std::mem::size_of::<f32>()
            + 3 * std::mem::size_of::<usize>()
    }

    #[inline]
    fn encode(&self, _input_vectors: &[Self::InputItem], _output_vectors: &mut [Self::OutputItem]) {
        unimplemented!(
            "TwoLevelQuantizer.encode is out of scope; provide precomputed codes instead"
        );
    }
}

/// Query evaluator: computes per-query ADC tables for residuals (q - coarse)
pub struct TwoLevelQueryEvaluator<'a, const M: usize, T: Float> {
    _phantom: std::marker::PhantomData<&'a T>,
    /// Distance tables for each query and each coarse centroid probed.
    /// Layout: for each probed centroid we store tables for all queries: [P x Q x (M*ksub)]
    /// For simplicity when constructing from `new` we compute tables for all centroids.
    distance_tables: Vec<f32>,
    ksub: usize,
    num_query_vectors: usize,
    stride_per_centroid: usize,
    /// Flattened query as f32 (num_query_vectors * vector_dim) used for centroid scoring
    query_flat: Vec<f32>,
    /// Full vector dimensionality (M * dsub)
    vector_dim: usize,
}

/// Query wrapper that holds both first-level and second-level views of the query.
/// TwoLevelQuery now wraps a single MultiVector query. The evaluator will use
/// the same query view for both the first (coarse) and second (residual/PQ)
/// computations. This simplifies the API: callers supply one query and the
/// quantizer uses it for both levels.
// The evaluator now accepts a plain MultiVector<&[T]> as the query type. We
// removed the TwoLevelQuery wrapper and operate directly on the provided
// multivector for both coarse and residual computations.

impl<'a, const M: usize, T: Float + DotProduct<T> + Copy> QueryEvaluator<'a>
    for TwoLevelQueryEvaluator<'a, M, T>
{
    type Q = TwoLevelQuantizer<M, T>;
    // QueryType is a TwoLevelQuery wrapper that exposes first and second level views
    // Now the evaluator directly accepts a multivector query (no wrapper)
    type QueryType = MultiVector<&'a [T]>;

    fn new(query: Self::QueryType, dataset: &<Self::Q as Quantizer>::DatasetType) -> Self {
        // Use the provided multivector query for both first and second level
        // computations (coarse centroid scoring and residual/PQ ADC tables).
        let first_level_query = &query;
        let second_level_query = &query;
        let quantizer = dataset.quantizer();
        let pq = quantizer.product_quantizer();
        let ksub = pq.ksub();

        // Ensure both levels have the same number of vectors
        assert_eq!(
            first_level_query.num_vectors(),
            second_level_query.num_vectors(),
            "first and second level queries must have same num_vectors"
        );

        let num_query_vectors = first_level_query.num_vectors();
        let dsub = pq.dsub();
        let vector_dim = M * dsub;
        let stride = M * ksub; // per-query table length

        // Pre-convert first-level queries into flat f32 buffer (for centroid scoring)
        let mut query_flat = vec![0f32; num_query_vectors * vector_dim];
        for q_idx in 0..num_query_vectors {
            let qv = first_level_query.get_vector(q_idx);
            let base = q_idx * vector_dim;
            for i in 0..vector_dim {
                query_flat[base + i] = qv[i].to_f32().unwrap();
            }
        }

        // Compute distance tables for the query using the global PQ (from second-level/residual queries)
        let distance_tables = crate::quantizers::multivector_product_quantizer::MultiVectorQueryEvaluatorPQ::
            compute_distance_tables_efficient(&second_level_query, pq);

        TwoLevelQueryEvaluator {
            _phantom: std::marker::PhantomData,
            distance_tables,
            ksub,
            num_query_vectors,
            stride_per_centroid: stride,
            query_flat,
            vector_dim,
        }
    }

    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32 {
        // dataset documents are expected to store (coarse_id + PQ codes) in each multivector
        let document = dataset.get(index);

        // For each query vector, compute MaxSim across document vectors using precomputed tables
        // We need to examine each query and the coarse id of doc vectors to index into proper tables.

        let stride = self.stride_per_centroid;

        let mut total_similarity = 0f32;

        // Simpler path: DO NOT deduplicate centroids. Build a centroid matrix that has
        // one column per document vector (allowing duplicates). This simplifies debugging
        // by making the centroid contribution explicit per document vector.
        let quantizer = dataset.quantizer();

        let doc_n = document.num_vectors();
        if doc_n == 0 {
            return -0f32;
        }

        // For each document vector we'll extract its coarse id and PQ codes and build
        // a centroid column. We'll also keep the encoded vectors for PQ lookup.
        let mut doc_vec_entries: Vec<(&[u8], usize)> = Vec::with_capacity(doc_n);
        let mut centroid_cols: Vec<f32> = vec![0f32; self.vector_dim * doc_n];

        for doc_vec_idx in 0..doc_n {
            let encoded_vec = document.get_vector(doc_vec_idx);
            if encoded_vec.len() < 4 + M {
                // leave zero centroid and push an empty entry
                doc_vec_entries.push((encoded_vec, usize::MAX));
                continue;
            }

            let coarse_id = (encoded_vec[0] as usize)
                | ((encoded_vec[1] as usize) << 8)
                | ((encoded_vec[2] as usize) << 16)
                | ((encoded_vec[3] as usize) << 24);

            let centroid_base = coarse_id * self.vector_dim;
            if centroid_base + self.vector_dim <= quantizer.coarse_centroids.len() {
                for i in 0..self.vector_dim {
                    // column-major storage: column = doc_vec_idx
                    centroid_cols[doc_vec_idx * self.vector_dim + i] =
                        quantizer.coarse_centroids[centroid_base + i];
                }
                doc_vec_entries.push((encoded_vec, doc_vec_idx));
            } else {
                // invalid centroid id: push placeholder
                doc_vec_entries.push((encoded_vec, usize::MAX));
            }
        }

        // Now compute centroid contributions. We compute per-query dot products with each
        // centroid column directly below (avoids sgemm and dedup).

        // Instead of using sgemm we compute per-query vs all centroid columns maxsims using
        // the batch-4 optimized function. This returns a single scalar MaxSim (sum over queries
        // of max over centroids). However we need per-query per-centroid contributions to combine
        // with PQ tables. To keep things simple and avoid extra API changes, we'll compute for
        // each document vector separately the similarity between the query multivector and the
        // single-column centroid multivector (d_len = 1). We'll use `multivector_maxsim_query_batch_4`.

        // Precompute centroid-only contributions per (q_idx, doc_vec_idx) by computing
        // multivector_maxsim_query_batch_4 between `q_mv` and each single-column multivector.
        // Note: this computes the sum over queries of max over 1 centroid (i.e., dot product per query),
        // so to get per-query centroid contributions we will compute per-query dot products individually
        // using the original dot product for correctness. To minimize changes, compute per-query dot
        // directly here for centroid columns.

        // We'll compute an array `centroid_by_q_doc` of length num_query_vectors * doc_n with
        // centroid dot products (query_vec dot centroid_col).
        let mut centroid_by_q_doc = vec![0f32; self.num_query_vectors * doc_n];

        for q_idx in 0..self.num_query_vectors {
            let q_start = q_idx * self.vector_dim;
            let q_slice = &self.query_flat[q_start..q_start + self.vector_dim];

            // process centroids in chunks of 4
            let mut d = 0usize;
            while d + 4 <= doc_n {
                let mut vs: [&[f32]; 4] = [&[]; 4];
                for i in 0..4 {
                    let col_start = (d + i) * self.vector_dim;
                    vs[i] = &centroid_cols[col_start..col_start + self.vector_dim];
                }
                let out =
                    crate::distances::dot_product::dot_product_batch_4_simd::<f32>(q_slice, vs);
                centroid_by_q_doc[q_idx * doc_n + d + 0] = out[0];
                centroid_by_q_doc[q_idx * doc_n + d + 1] = out[1];
                centroid_by_q_doc[q_idx * doc_n + d + 2] = out[2];
                centroid_by_q_doc[q_idx * doc_n + d + 3] = out[3];
                d += 4;
            }

            // remainder
            while d < doc_n {
                let col_start = d * self.vector_dim;
                let centroid_slice = &centroid_cols[col_start..col_start + self.vector_dim];
                let s =
                    crate::distances::dot_product::dot_product_simd::<f32>(q_slice, centroid_slice);
                centroid_by_q_doc[q_idx * doc_n + d] = s;
                d += 1;
            }
        }

        // Now for each query vector compute max over document vectors combining centroid and PQ
        for q_idx in 0..self.num_query_vectors {
            let mut max_sim = f32::NEG_INFINITY;
            let table_base = q_idx * stride;
            let query_table = &self.distance_tables[table_base..table_base + stride];

            for (encoded_vec, doc_idx) in doc_vec_entries.iter() {
                // pq contribution using ADC tables
                let mut pq_contrib = 0.0f32;
                for (m_idx, &code) in encoded_vec[4..4 + M].iter().enumerate() {
                    pq_contrib += query_table[m_idx * self.ksub + code as usize];
                }

                let centroid_val = if *doc_idx == usize::MAX {
                    0.0f32
                } else {
                    centroid_by_q_doc[q_idx * doc_n + *doc_idx]
                };

                let sim = centroid_val + pq_contrib;
                if sim > max_sim {
                    max_sim = sim;
                }
            }

            total_similarity += max_sim;
        }

        -total_similarity
    }

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
