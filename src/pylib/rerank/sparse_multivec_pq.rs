//! `SparseMultivecTwoLevelsPQRerankIndex`: sparse HNSW base index reranked with a
//! two-levels PQ multivector dataset.

use std::f32;

use crate::graph::Graph;
use crate::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use half::f16;
use vectorium::IndexSerializer;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::PlainSparseDataset;
use vectorium::core::rerank_index::RerankIndex;
use vectorium::distances::DotProduct;
use vectorium::vector::DenseMultiVectorView;
use vectorium::vector::SparseVectorView;
use vectorium::{MultiVectorDataset, PlainMultiVecQuantizer};

use crate::pylib::common::{convert_components_to_u16, push_results};

// Helper to load two-level PQ multivector dataset
fn load_multivec_dataset_pq_8(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<8>(data_folder)
}

fn load_multivec_dataset_pq_16(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<16>(data_folder)
}

fn load_multivec_dataset_pq_32(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<32>(data_folder)
}

fn load_multivec_dataset_pq_64(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<64>(data_folder)
}

fn load_multivec_dataset_pq_generic<const M: usize>(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    use ndarray::Array1;
    use ndarray_npy::ReadNpyExt;
    use std::path::Path;

    let coarse_path = Path::new(data_folder).join("centroids.npy");
    let pq_centroids_path = Path::new(data_folder).join("pq_centroids.npy");
    let residuals_path = Path::new(data_folder).join("residuals.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");
    let assignment_path = Path::new(data_folder).join("index_assignment.npy");

    // Load coarse centroids (n_centroids, dim) to determine token_dim
    let coarse_file = std::fs::File::open(&coarse_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening centroids.npy at {:?}: {}",
            coarse_path, e
        ))
    })?;
    let coarse_reader = std::io::BufReader::new(coarse_file);
    let coarse_array: ndarray::Array2<f32> =
        ndarray::Array2::read_npy(coarse_reader).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading centroids.npy: {}",
                e
            ))
        })?;
    let (n_coarse, token_dim) = coarse_array.dim();
    let coarse_flat: Vec<f32> = coarse_array.into_iter().collect();

    // Load PQ centroids
    let pq_file = std::fs::File::open(&pq_centroids_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening pq_centroids.npy at {:?}: {}",
            pq_centroids_path, e
        ))
    })?;
    let pq_reader = std::io::BufReader::new(pq_file);
    let pq_array: Array1<f32> = Array1::read_npy(pq_reader).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error reading pq_centroids.npy: {}",
            e
        ))
    })?;
    let pq_flat = pq_array.to_vec();

    let dsub = token_dim / M;
    const KSUB: usize = 256;

    let mut pq_reconstruction_centroids = Vec::new();
    for m in 0..M {
        let offset = m * KSUB * dsub;
        pq_reconstruction_centroids.extend_from_slice(&pq_flat[offset..offset + KSUB * dsub]);
    }

    // Load doclens
    let doclens_file = std::fs::File::open(&doclens_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening doclens.npy at {:?}: {}",
            doclens_path, e
        ))
    })?;
    let doclens_reader = std::io::BufReader::new(doclens_file);
    let doclens_array: Array1<i32> = Array1::read_npy(doclens_reader).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading doclens.npy: {}", e))
    })?;
    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Load residuals
    let residuals_file = std::fs::File::open(&residuals_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening residuals.npy at {:?}: {}",
            residuals_path, e
        ))
    })?;
    let residuals_reader = std::io::BufReader::new(residuals_file);
    let residuals_array: ndarray::Array2<u8> = ndarray::Array2::read_npy(residuals_reader)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading residuals.npy: {}",
                e
            ))
        })?;
    let (n_tokens, m_check) = residuals_array.dim();
    if m_check != M {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "residuals.npy has {} subspaces, expected {}",
            m_check, M
        )));
    }

    // Load index assignments
    let assignment_file = std::fs::File::open(&assignment_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening index_assignment.npy at {:?}: {}",
            assignment_path, e
        ))
    })?;
    let assignment_reader = std::io::BufReader::new(assignment_file);
    let assignment_array: Array1<u64> = Array1::read_npy(assignment_reader).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error reading index_assignment.npy: {}",
            e
        ))
    })?;
    if assignment_array.len() != n_tokens {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "assignment_array length {} != n_tokens {}",
            assignment_array.len(),
            n_tokens
        )));
    }

    // Reconstruct documents from two-level PQ
    let mut reconstructed_tokens = Vec::with_capacity(n_tokens * token_dim);
    for token_idx in 0..n_tokens {
        let coarse_idx = assignment_array[token_idx] as usize;
        if coarse_idx >= n_coarse {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "coarse_idx {} >= n_coarse {}",
                coarse_idx, n_coarse
            )));
        }
        let coarse_offset = coarse_idx * token_dim;

        for subspace_idx in 0..M {
            let code = residuals_array[[token_idx, subspace_idx]];
            let pq_offset = subspace_idx * KSUB * dsub + (code as usize) * dsub;

            for d in 0..dsub {
                let coarse_val = coarse_flat[coarse_offset + subspace_idx * dsub + d];
                let residual_val = pq_reconstruction_centroids[pq_offset + d];
                reconstructed_tokens.push(coarse_val + residual_val);
            }
        }
    }

    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * token_dim);
    }

    let encoder = PlainMultiVecQuantizer::new(token_dim);
    Ok(MultiVectorDataset::from_raw(
        reconstructed_tokens.into_boxed_slice(),
        offsets.into(),
        encoder,
    ))
}

// Enum to handle different PQ subspace counts
enum SparseMultivecTwoLevelsPQRerankIndexEnum {
    M8(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
    M16(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
    M32(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
    M64(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
}

#[pyclass]
pub struct SparseMultivecTwoLevelsPQRerankIndex {
    inner: SparseMultivecTwoLevelsPQRerankIndexEnum,
}

#[pymethods]
impl SparseMultivecTwoLevelsPQRerankIndex {
    /// Build a rerank index from a pre-built sparse HNSW index and multivector data folder with two-level PQ encoding.
    ///
    /// # Arguments
    /// * `sparse_index_path` – Path to the pre-built sparse HNSW index file.
    /// * `multivec_data_folder` – Path to folder containing multivector data files (two-level PQ quantizer).
    /// * `pq_subspaces` – Number of PQ subspaces (M). Supported values: 8, 16, 32, 64.
    ///
    /// # Multivector Data Folder Structure (Two-Level PQ Quantizer)
    /// The folder must contain the following files:
    /// * `doclens.npy` – Document lengths (shape: [n_documents], dtype: int32 or int64)
    /// * `centroids.npy` – Coarse centroids from first-level quantization (shape: [n_centroids, token_dim], dtype: float32)
    /// * `index_assignment.npy` – Index assignments for documents to centroids (shape: [n_documents, n_tokens], dtype: int32 or int64)
    /// * `residuals.npy` – PQ-encoded residuals (shape: [n_documents, n_tokens, token_dim], dtype: float32)
    /// * `pq_centroids.npy` – PQ centroids (shape: [n_centroids, M, subspace_dim], dtype: float32)
    ///
    #[staticmethod]
    #[pyo3(signature = (sparse_index_path, multivec_data_folder, pq_subspaces))]
    pub fn build_from_file(
        sparse_index_path: &str,
        multivec_data_folder: &str,
        pq_subspaces: usize,
    ) -> PyResult<Self> {
        let sparse_index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> =
            <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(
                sparse_index_path,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error loading sparse index: {:?}",
                    e
                ))
            })?;

        let inner = match pq_subspaces {
            8 => {
                let multivec_dataset = load_multivec_dataset_pq_8(multivec_data_folder)?;
                SparseMultivecTwoLevelsPQRerankIndexEnum::M8(RerankIndex::new(
                    sparse_index,
                    multivec_dataset,
                ))
            }
            16 => {
                let multivec_dataset = load_multivec_dataset_pq_16(multivec_data_folder)?;
                SparseMultivecTwoLevelsPQRerankIndexEnum::M16(RerankIndex::new(
                    sparse_index,
                    multivec_dataset,
                ))
            }
            32 => {
                let multivec_dataset = load_multivec_dataset_pq_32(multivec_data_folder)?;
                SparseMultivecTwoLevelsPQRerankIndexEnum::M32(RerankIndex::new(
                    sparse_index,
                    multivec_dataset,
                ))
            }
            64 => {
                let multivec_dataset = load_multivec_dataset_pq_64(multivec_data_folder)?;
                SparseMultivecTwoLevelsPQRerankIndexEnum::M64(RerankIndex::new(
                    sparse_index,
                    multivec_dataset,
                ))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported pq_subspaces value: {}. Supported: 8, 16, 32, 64",
                    pq_subspaces
                )));
            }
        };

        Ok(SparseMultivecTwoLevelsPQRerankIndex { inner })
    }

    /// Search with reranking using two-level PQ multivector encoding (single query).
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of sparse query component indices.
    /// * `query_values` – 1-D float32 array of sparse query values.
    /// * `multivec_query` – 1-D float32 array of the multivector query (n_tokens × token_dim).
    /// * `n_tokens` – Number of tokens in the multivector query.
    /// * `token_dim` – Dimension of each token.
    /// * `k_candidates` – Number of candidates to retrieve in first stage. Default: 100.
    /// * `k` – Number of final results to return. Default: 10.
    /// * `ef_search` – Candidate list size for HNSW search. Default: 100.
    /// * `alpha` – Alpha parameter for candidate pruning (0-1). Default: None.
    /// * `beta` – Beta parameter for early exit. Default: None.
    /// * `early_exit_threshold` – Lambda for early termination. Default: None.
    /// * `residuals` – Score candidates by the sum of first-stage and rerank scores instead of the
    ///   rerank score alone. Only meaningful when the two scores are summable (e.g. the rerank
    ///   dataset holds the residual part of a decomposed representation). Default: False.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, multivec_query, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, residuals=false))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        multivec_query: PyReadonlyArray1<f32>,
        n_tokens: usize,
        token_dim: usize,
        k_candidates: usize,
        k: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        residuals: bool,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let query_values_slice = query_values.as_slice()?;
        let multivec_query_slice = multivec_query.as_slice()?;

        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let sparse_query = SparseVectorView::new(&comp_vec, query_values_slice);
        let multivec_query_view = DenseMultiVectorView::new(multivec_query_slice, token_dim);
        let _ = n_tokens;

        let results = match &self.inner {
            SparseMultivecTwoLevelsPQRerankIndexEnum::M8(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M16(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M32(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
            SparseMultivecTwoLevelsPQRerankIndexEnum::M64(rerank_index) => rerank_index.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
                residuals,
            ),
        };

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        push_results(results, k, &mut distances, &mut ids);

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search with reranking using two-level PQ multivector encoding, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    #[pyo3(signature = (query_components, query_values, sparse_offsets, multivec_queries, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, num_threads=0, residuals=false))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        sparse_offsets: PyReadonlyArray1<i64>,
        multivec_queries: PyReadonlyArray1<f32>,
        n_tokens: usize,
        token_dim: usize,
        k_candidates: usize,
        k: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
        residuals: bool,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let query_values_slice = query_values.as_slice()?;
        let sparse_offsets_slice = sparse_offsets.as_slice()?;
        let multivec_queries_slice = multivec_queries.as_slice()?;

        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let num_queries = sparse_offsets_slice.len() - 1;
        let multivec_query_size = n_tokens * token_dim;

        let search_one = |q_idx: usize| -> (Vec<f32>, Vec<i64>) {
            let sparse_start = sparse_offsets_slice[q_idx] as usize;
            let sparse_end = sparse_offsets_slice[q_idx + 1] as usize;
            let sparse_query = SparseVectorView::new(
                &comp_vec[sparse_start..sparse_end],
                &query_values_slice[sparse_start..sparse_end],
            );
            let multivec_start = q_idx * multivec_query_size;
            let multivec_query_view = DenseMultiVectorView::new(
                &multivec_queries_slice[multivec_start..multivec_start + multivec_query_size],
                token_dim,
            );
            let results = match &self.inner {
                SparseMultivecTwoLevelsPQRerankIndexEnum::M8(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M16(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M32(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M64(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    &(),
                    alpha,
                    beta,
                    residuals,
                ),
            };
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            push_results(results, k, &mut distances, &mut ids);
            (distances, ids)
        };

        let results: Vec<(Vec<f32>, Vec<i64>)> = py.detach(|| match num_threads {
            1 => (0..num_queries).map(search_one).collect(),
            0 => (0..num_queries).into_par_iter().map(search_one).collect(),
            n => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build rayon thread pool")
                .install(|| (0..num_queries).into_par_iter().map(search_one).collect()),
        });

        let mut all_distances = Vec::with_capacity(num_queries * k);
        let mut all_ids = Vec::with_capacity(num_queries * k);
        for (d, i) in results {
            all_distances.extend(d);
            all_ids.extend(i);
        }

        let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
        let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }
}
