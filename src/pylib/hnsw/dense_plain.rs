//! `DensePlainHNSW`: HNSW over dense f32 input, stored internally as f16.

use std::f32;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};
use half::f16;
use vectorium::IndexSerializer;
use vectorium::core::index::{Index, IndexStats};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::DenseDataset;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
use vectorium::vector::DenseVectorView;

use super::{GraphTypeKind, load_index_err, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{MetricKind, parse_metric, push_results, read_npy_dataset_f16};

enum DensePlainHNSWEnum {
    Euclidean(HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph>),
    DotProduct(HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph>),
    EuclideanStreamVByte(
        HNSW<
            DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
    DotProductStreamVByte(
        HNSW<
            DenseDataset<PlainDenseQuantizer<f16, DotProduct>>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
}

#[pyclass]
pub struct DensePlainHNSW {
    inner: DensePlainHNSWEnum,
    acorn_gamma: Option<Graph>,
}

#[pymethods]
impl DensePlainHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_path, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_path: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = read_npy_dataset_f16::<SquaredEuclideanDistance>(data_path)?;
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => DensePlainHNSWEnum::Euclidean(plain),
                    GraphTypeKind::Permuted => {
                        DensePlainHNSWEnum::Euclidean(plain.permute_and_encode::<PlainNeighbors>())
                    }
                    GraphTypeKind::Compressed => DensePlainHNSWEnum::EuclideanStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
            MetricKind::DotProduct => {
                let dataset = read_npy_dataset_f16::<DotProduct>(data_path)?;
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => DensePlainHNSWEnum::DotProduct(plain),
                    GraphTypeKind::Permuted => {
                        DensePlainHNSWEnum::DotProduct(plain.permute_and_encode::<PlainNeighbors>())
                    }
                    GraphTypeKind::Compressed => DensePlainHNSWEnum::DotProductStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
        };

        Ok(DensePlainHNSW {
            inner,
            acorn_gamma: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
    ) -> PyResult<Self> {
        let data_f16: Vec<f16> = data_vec
            .as_slice()?
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let n_vecs = data_f16.len() / dim;
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f16, SquaredEuclideanDistance>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => DensePlainHNSWEnum::Euclidean(plain),
                    GraphTypeKind::Permuted => {
                        DensePlainHNSWEnum::Euclidean(plain.permute_and_encode::<PlainNeighbors>())
                    }
                    GraphTypeKind::Compressed => DensePlainHNSWEnum::EuclideanStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f16, DotProduct>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => DensePlainHNSWEnum::DotProduct(plain),
                    GraphTypeKind::Permuted => {
                        DensePlainHNSWEnum::DotProduct(plain.permute_and_encode::<PlainNeighbors>())
                    }
                    GraphTypeKind::Compressed => DensePlainHNSWEnum::DotProductStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
        };

        Ok(DensePlainHNSW {
            inner,
            acorn_gamma: None,
        })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            DensePlainHNSWEnum::DotProduct(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => {
                index.save_index(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error saving index: {:?}",
                        e
                    ))
                })
            }
            DensePlainHNSWEnum::DotProductStreamVByte(index) => {
                index.save_index(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error saving index: {:?}",
                        e
                    ))
                })
            }
        }
    }

    /// Loads a previously saved index. `graph_type` must match the value used at build
    /// time (`standard`/`permuted` both load as the same on-disk representation).
    #[staticmethod]
    #[pyo3(signature = (path, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn load(path: &str, metric: String, graph_type: String) -> PyResult<Self> {
        let gt = parse_graph_type(&graph_type)?;
        let inner = match (parse_metric(&metric)?, gt) {
            (MetricKind::Euclidean, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                DensePlainHNSWEnum::Euclidean(index)
            }
            (MetricKind::Euclidean, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePlainHNSWEnum::EuclideanStreamVByte(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                DensePlainHNSWEnum::DotProduct(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<PlainDenseQuantizer<f16, DotProduct>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<PlainDenseQuantizer<f16, DotProduct>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePlainHNSWEnum::DotProductStreamVByte(index)
            }
        };
        Ok(DensePlainHNSW {
            inner,
            acorn_gamma: None,
        })
    }

    /// Search for approximate nearest neighbors for a single query.
    ///
    /// # Arguments
    /// * `query` – 1-D float32 numpy array of length `dimension`.
    /// * `k` – Number of nearest neighbors to return.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }
        let dim = match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => index.dim(),
            DensePlainHNSWEnum::DotProduct(index) => index.dim(),
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => index.dim(),
            DensePlainHNSWEnum::DotProductStreamVByte(index) => index.dim(),
        };
        let query_slice = query.as_slice()?;
        if query_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "query dimension {} does not match index dimension {}",
                query_slice.len(),
                dim
            )));
        }
        let mut ids = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);

        let query_view = DenseVectorView::new(query_slice);

        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            DensePlainHNSWEnum::DotProductStreamVByte(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
        }

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Search a batch of queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    ///
    /// # Arguments
    /// * `queries` – 1-D float32 numpy array of length `num_queries × dimension`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length `num_queries × k`.
    #[pyo3(signature = (queries, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let queries_slice = queries.as_slice()?;
        let dim = match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => index.dim(),
            DensePlainHNSWEnum::DotProduct(index) => index.dim(),
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => index.dim(),
            DensePlainHNSWEnum::DotProductStreamVByte(index) => index.dim(),
        };
        if queries_slice.len() % dim != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "queries array length {} is not a multiple of index dimension {}",
                queries_slice.len(),
                dim
            )));
        }
        let num_queries = queries_slice.len() / dim;

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let query_view = DenseVectorView::new(&queries_slice[i * dim..(i + 1) * dim]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                DensePlainHNSWEnum::Euclidean(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                DensePlainHNSWEnum::DotProduct(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                DensePlainHNSWEnum::EuclideanStreamVByte(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                DensePlainHNSWEnum::DotProductStreamVByte(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
            }
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

    /// ACORN-1 filtered search: returns the `k` approximate nearest neighbors
    /// of `query` for which `predicate(vector_id)` returns `True`.
    ///
    /// The standard HNSW index is used as-is; no rebuilding is required.
    ///
    /// # Arguments
    /// * `query` – 1-D float32 numpy array of dimension `dim`.
    /// * `k` – Number of nearest neighbors to return.
    /// * `ef_search` – Candidate list size (higher = better recall, slower).
    /// * `predicate` – Python callable `(int) -> bool`. Receives a global vector
    ///   ID (0-based) and must return `True` for vectors eligible as results.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query, k, predicate, ef_search=100, early_exit_threshold=None))]
    pub fn search_filtered(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        predicate: Py<PyAny>,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        let query_view = DenseVectorView::new(query_slice);
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let pred_fn = |id: usize| -> bool {
            predicate
                .call1(py, (id as i64,))
                .and_then(|r| r.extract::<bool>(py))
                .unwrap_or(false)
        };

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);

        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => {
                let results = index.search_filtered(query_view, k, &search_config, pred_fn);
                push_results(results, k, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let results = index.search_filtered(query_view, k, &search_config, pred_fn);
                push_results(results, k, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => {
                let results = index.search_filtered(query_view, k, &search_config, pred_fn);
                push_results(results, k, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProductStreamVByte(index) => {
                let results = index.search_filtered(query_view, k, &search_config, pred_fn);
                push_results(results, k, &mut distances, &mut ids);
            }
        }

        let distances_array = PyArray1::from_vec(py, distances).to_owned();
        let ids_array = PyArray1::from_vec(py, ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }

    /// Pre-compute expanded neighbor lists for ACORN-γ filtered search.
    ///
    /// Call this once after building the index. The expanded lists are stored on
    /// the index object and used by [`search_filtered_gamma`].
    ///
    /// # Arguments
    /// * `gamma` – Expansion factor (≥ 1). Each node stores up to `gamma × M`
    ///   neighbors (two-hop union, pruned by distance). Larger values improve recall
    ///   at the cost of memory and build time. A value of 2–4 is a good starting point.
    pub fn build_acorn_gamma(&mut self, gamma: usize) {
        let neighbors = match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => index.build_acorn_gamma_neighbors(gamma),
            DensePlainHNSWEnum::DotProduct(index) => index.build_acorn_gamma_neighbors(gamma),
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => {
                index.build_acorn_gamma_neighbors(gamma)
            }
            DensePlainHNSWEnum::DotProductStreamVByte(index) => {
                index.build_acorn_gamma_neighbors(gamma)
            }
        };
        self.acorn_gamma = Some(neighbors);
    }

    /// ACORN-γ filtered search: returns the `k` approximate nearest neighbors of
    /// `query` for which `predicate(vector_id)` returns `True`.
    ///
    /// Requires [`build_acorn_gamma`] to have been called first.
    ///
    /// Unlike ACORN-1 ([`search_filtered`]), the two-hop connectivity is pre-baked
    /// into the index at build time, so predicate-failing nodes are simply skipped
    /// during traversal — no on-the-fly two-hop expansion is performed.
    ///
    /// # Arguments
    /// * `query` – 1-D float32 numpy array of dimension `dim`.
    /// * `k` – Number of nearest neighbors to return.
    /// * `ef_search` – Candidate list size (higher = better recall, slower).
    /// * `predicate` – Python callable `(int) -> bool`. Receives a global vector
    ///   ID (0-based) and must return `True` for eligible vectors.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query, k, predicate, ef_search=100, early_exit_threshold=None))]
    pub fn search_filtered_gamma(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        predicate: Py<PyAny>,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let acorn_gamma = self.acorn_gamma.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ACORN-γ neighbor lists not built. Call `build_acorn_gamma(gamma)` first.",
            )
        })?;

        let query_slice = query.as_slice()?;
        let query_view = DenseVectorView::new(query_slice);
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let pred_fn = |id: usize| -> bool {
            predicate
                .call1(py, (id as i64,))
                .and_then(|r| r.extract::<bool>(py))
                .unwrap_or(false)
        };

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);

        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => {
                let results = index.search_filtered_gamma(
                    query_view,
                    k,
                    &search_config,
                    acorn_gamma,
                    pred_fn,
                );
                push_results(results, k, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let results = index.search_filtered_gamma(
                    query_view,
                    k,
                    &search_config,
                    acorn_gamma,
                    pred_fn,
                );
                push_results(results, k, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::EuclideanStreamVByte(index) => {
                let results = index.search_filtered_gamma(
                    query_view,
                    k,
                    &search_config,
                    acorn_gamma,
                    pred_fn,
                );
                push_results(results, k, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProductStreamVByte(index) => {
                let results = index.search_filtered_gamma(
                    query_view,
                    k,
                    &search_config,
                    acorn_gamma,
                    pred_fn,
                );
                push_results(results, k, &mut distances, &mut ids);
            }
        }

        let distances_array = PyArray1::from_vec(py, distances).to_owned();
        let ids_array = PyArray1::from_vec(py, ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }
}
