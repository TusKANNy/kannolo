use crate::graph::GraphTrait;
use crate::index::Index;
use crate::hnsw_utils::Candidate;
use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::{Dataset, DotProduct, EuclideanDistance, Float};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::fmt;
use std::marker::PhantomData;

/// A reranking index that combines a graph index for candidate retrieval
/// with a higher-quality dataset for reranking.
///
/// # Type Parameters
/// - `FirstStageIndex`: The first stage index that implements the Index trait
/// - `D`: The dataset type used by the graph index
/// - `Q`: The quantizer type used by the graph index
/// - `G`: The graph trait used by the graph index
/// - `RerankDataset`: The dataset type used for reranking (e.g., MultiVectorDataset, DenseDataset)
/// - `RerankQuantizer`: The quantizer associated with the rerank dataset
///
/// # Constraints
/// Both the graph index and rerank dataset must contain the same number of items
/// with the same document IDs (same semantic content, different representations).
#[derive(Serialize, Deserialize)]
pub struct RerankIndex<FirstStageIndex, D, Q, G, RerankDataset, RerankQuantizer>
where
    FirstStageIndex: Index<D, Q>,
    D: Dataset<Q>,
    Q: Quantizer<DatasetType = D, InputItem: Float> + Sync,
    G: GraphTrait,
    RerankDataset: Dataset<RerankQuantizer>,
    RerankQuantizer: Quantizer<DatasetType = RerankDataset>,
{
    first_stage_index: FirstStageIndex,
    rerank_dataset: RerankDataset,
    _phantom_d: PhantomData<D>,
    _phantom_q: PhantomData<Q>,
    _phantom_g: PhantomData<G>,
    _phantom_rq: PhantomData<RerankQuantizer>,
}

impl<FirstStageIndex, D, Q, G, RerankDataset, RerankQuantizer>
    RerankIndex<FirstStageIndex, D, Q, G, RerankDataset, RerankQuantizer>
where
    FirstStageIndex: Index<D, Q>,
    D: Dataset<Q>,
    Q: Quantizer<DatasetType = D, InputItem: Float> + Sync,
    G: GraphTrait,
    RerankDataset: Dataset<RerankQuantizer>,
    RerankQuantizer: Quantizer<DatasetType = RerankDataset, InputItem: Float> + Sync,
{
    /// Creates a new rerank index from a graph index and a rerank dataset.
    ///
    /// # Arguments
    /// - `graph_index`: The graph index for candidate retrieval
    /// - `rerank_dataset`: The dataset for reranking candidates
    ///
    /// # Note
    /// The caller should ensure that both the graph index and rerank dataset
    /// have the same number of items with matching document IDs.
    pub fn new(first_stage_index: FirstStageIndex, rerank_dataset: RerankDataset) -> Self {
        Self {
            first_stage_index,
            rerank_dataset,
            _phantom_d: PhantomData,
            _phantom_q: PhantomData,
            _phantom_g: PhantomData,
            _phantom_rq: PhantomData,
        }
    }

    /// Performs a complete two-stage search: candidate retrieval followed by reranking.
    ///
    /// This is the main search method that:
    /// 1. Retrieves k_candidates from the graph index using graph_query
    /// 2. Reranks all candidates using rerank_query with the rerank dataset  
    /// 3. Returns the top k_final results sorted by score along with timing information
    ///
    /// # Arguments
    /// - `graph_query`: Query for the graph index
    /// - `rerank_query`: Query for reranking
    /// - `k_candidates`: Number of candidates to retrieve from graph index
    /// - `k_final`: Number of final results to return after reranking
    /// - `search_params`: Search parameters for the graph index
    ///
    /// # Returns
    /// Tuple of (results, graph_time_us, rerank_time_us) where:
    /// - results: Vector of (score, document_id) pairs, sorted by score in descending order
    /// - graph_time_us: Time spent on graph search in microseconds
    /// - rerank_time_us: Time spent on reranking in microseconds
    pub fn search<'a, QD, QQ, RerankQuery>(
        &'a self,
        graph_query: QD::DataType<'a>,
        rerank_query: RerankQuery,
        k_candidates: usize,
        k_final: usize,
        search_params: &FirstStageIndex::SearchParams,
        alpha: Option<f32>,
        beta: Option<usize>,
    ) -> (Vec<(f32, usize)>, u64, u64)
    where
        QD: Dataset<QQ> + Sync + 'a,
        QQ: Quantizer<DatasetType = QD> + Sync + 'a,
        Q::Evaluator<'a>: QueryEvaluator<'a, QueryType = QD::DataType<'a>>,
        Q::InputItem: EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem> + 'a,
        RerankQuantizer::Evaluator<'a>: QueryEvaluator<'a, QueryType = RerankQuery>,
        RerankQuantizer::InputItem: Float
            + EuclideanDistance<RerankQuantizer::InputItem>
            + DotProduct<RerankQuantizer::InputItem>,
    {
        // Stage 1: Get candidates from graph index using its own search method
        let search_time = std::time::Instant::now();
        let mut candidates =
            self.first_stage_index
                .search::<QD, QQ>(graph_query, k_candidates, search_params);
        let graph_time_us = search_time.elapsed().as_micros() as u64;

        if alpha.is_some() {
            let threshold = candidates[k_final - 1].0 * (1.0 - alpha.unwrap());
            candidates.retain(|&(score, _)| score >= threshold);
        }

        // Stage 2: Rerank candidates using the rerank dataset
        let rerank_time = std::time::Instant::now();
        let results = if beta.is_some() {
            self.rerank_candidates_with_early_exit(
                rerank_query,
                &candidates,
                k_final,
                beta.unwrap(),
            )
        } else {
            self.rerank_candidates(rerank_query, &candidates, k_final)
        };
        let rerank_time_us = rerank_time.elapsed().as_micros() as u64;

        (results, graph_time_us, rerank_time_us)
    }

    pub fn search_from_external_candidates<'a, QD, QQ, RerankQuery>(
        &'a self,
        rerank_query: RerankQuery,
        external_candidates: &[(f32, usize)],
        k_final: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
    ) -> (Vec<(f32, usize)>, u64)
    where
        QD: Dataset<QQ> + Sync + 'a,
        QQ: Quantizer<DatasetType = QD> + Sync + 'a,
        RerankQuantizer::Evaluator<'a>: QueryEvaluator<'a, QueryType = RerankQuery>,
        RerankQuantizer::InputItem: Float
            + EuclideanDistance<RerankQuantizer::InputItem>
            + DotProduct<RerankQuantizer::InputItem>,
    {
        let rerank_time = std::time::Instant::now();

        let candidates = if let Some(alpha_val) = alpha {
            let threshold = external_candidates[k_final - 1].0 * (1.0 - alpha_val);
            external_candidates
                .iter()
                .cloned()
                .filter(|&(score, _)| score >= threshold)
                .collect::<Vec<_>>()
        } else {
            external_candidates.to_vec()
        };

        let results = if beta.is_some() {
            self.rerank_candidates_with_early_exit(
                rerank_query,
                &candidates,
                k_final,
                beta.unwrap(),
            )
        } else {
            self.rerank_candidates(rerank_query, &candidates, k_final)
        };
        let rerank_time_us = rerank_time.elapsed().as_micros() as u64;

        (results, rerank_time_us)
    }

    /// Reranks a given list of candidates using the rerank dataset.
    ///
    /// This method takes candidates (typically from a graph index search) and
    /// reranks them using the high-quality rerank dataset.
    ///
    /// # Arguments
    /// - `rerank_query`: Query for reranking
    /// - `candidates`: List of (score, document_id) pairs from initial search
    /// - `k_final`: Number of final results to return
    ///
    /// # Returns
    /// Vector of (score, document_id) pairs, sorted by score in descending order
    fn rerank_candidates<'a, RerankQuery>(
        &'a self,
        rerank_query: RerankQuery,
        candidates: &[(f32, usize)],
        k_final: usize,
    ) -> Vec<(f32, usize)>
    where
        RerankQuantizer::Evaluator<'a>: QueryEvaluator<'a, QueryType = RerankQuery>,
        RerankQuantizer::InputItem: Float
            + EuclideanDistance<RerankQuantizer::InputItem>
            + DotProduct<RerankQuantizer::InputItem>,
    {
        // Create query evaluator for reranking
        let query_evaluator = self.rerank_dataset.query_evaluator(rerank_query);

        // Rerank candidates by computing exact distances
        let mut reranked: Vec<(f32, usize)> = candidates
            .iter()
            .map(|(_, doc_id)| {
                let distance = query_evaluator.compute_distance(&self.rerank_dataset, *doc_id);
                let score = -distance; // Negate distance to get score (higher is better)
                (score, *doc_id)
            })
            .collect();

        // Sort by score in descending order and take top k_final
        reranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k_final);

        reranked
    }

    fn rerank_candidates_with_early_exit<'a, RerankQuery>(
        &'a self,
        rerank_query: RerankQuery,
        candidates: &[(f32, usize)],
        k_final: usize,
        beta: usize,
    ) -> Vec<(f32, usize)>
    where
        RerankQuantizer::Evaluator<'a>: QueryEvaluator<'a, QueryType = RerankQuery>,
        RerankQuantizer::InputItem: Float
            + EuclideanDistance<RerankQuantizer::InputItem>
            + DotProduct<RerankQuantizer::InputItem>,
    {
        // Ensure there are enough candidates
        if candidates.len() < k_final {
            return Vec::new();
        }

        // Create query evaluator for reranking
        let query_evaluator = self.rerank_dataset.query_evaluator(rerank_query);

        // Rerank candidates by computing exact distances
        let first_reranked: Vec<Candidate> = candidates[..k_final]
            .iter()
            .map(|(_, doc_id)| {
                let distance = query_evaluator.compute_distance(&self.rerank_dataset, *doc_id);
                Candidate(distance, *doc_id)
            })
            .collect();

        // Create a max heap and put initial candidates
        let mut heap = BinaryHeap::from(first_reranked);

        let mut n_stalls = 0;
        for cand in &candidates[k_final..] {
            let distance = query_evaluator.compute_distance(&self.rerank_dataset, cand.1);
            let candidate = Candidate(distance, cand.1);
            if let Some(mut worst) = heap.peek_mut() {
                if candidate < *worst {
                    *worst = candidate;
                    n_stalls = 0;
                } else {
                    n_stalls += 1;
                    if n_stalls >= beta {
                        break;
                    }
                }
            }
        }

        // Extract and sort the final reranked candidates from the heap
        let reranked: Vec<(f32, usize)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| (-candidate.distance(), candidate.id_vec()))
            .collect();

        reranked
    }

    /// Returns a reference to the underlying graph index.
    pub fn first_stage_index(&self) -> &FirstStageIndex {
        &self.first_stage_index
    }

    /// Returns a reference to the rerank dataset.
    pub fn rerank_dataset(&self) -> &RerankDataset {
        &self.rerank_dataset
    }

    /// Returns the number of items in the index.
    pub fn len(&self) -> usize {
        self.rerank_dataset.len()
    }

    /// Returns true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.rerank_dataset.is_empty()
    }
}

impl<GraphIdx, D, Q, G, RerankDataset, RerankQuantizer> fmt::Debug
     for RerankIndex<GraphIdx, D, Q, G, RerankDataset, RerankQuantizer>
 where
     GraphIdx: Index<D, Q> + fmt::Debug,
    D: Dataset<Q>,
    Q: Quantizer<DatasetType = D, InputItem: Float> + Sync,
    G: GraphTrait,
    RerankDataset: Dataset<RerankQuantizer> + fmt::Debug,
    RerankQuantizer: Quantizer<DatasetType = RerankDataset, InputItem: Float> + Sync + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankIndex")
            .field("first_stage_index", &self.first_stage_index)
            .field("rerank_dataset", &self.rerank_dataset)
            .finish()
    }
}
