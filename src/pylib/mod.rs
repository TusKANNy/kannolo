use std::f32;

use crate::graph::Graph;
use crate::hnsw::{
    AcornGammaNeighbors, EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration,
    HNSWSearchConfiguration,
};
use half::f16;
use vectorium::IndexSerializer;
use vectorium::core::index::Index;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::core::rerank_index::RerankIndex;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector::{DenseMultiVectorView, DenseVectorView, SparseVectorView};
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, FixedU8Q, FixedU16Q, Float, FromF32,
    MultiVectorDataset, PackedSparseDataset, PlainDenseDataset, PlainMultiVecQuantizer,
    PlainSparseDataset, PlainSparseDatasetGrowable, ScalarSparseDataset, ValueType,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MetricKind {
    Euclidean,
    DotProduct,
}

fn parse_metric(metric: &str) -> PyResult<MetricKind> {
    let metric = metric.to_lowercase();
    match metric.as_str() {
        "euclidean" | "l2" => Ok(MetricKind::Euclidean),
        "dotproduct" | "ip" => Ok(MetricKind::DotProduct),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Invalid metric; choose 'euclidean' or 'dotproduct'",
        )),
    }
}

fn read_npy_dataset<D>(path: &str) -> PyResult<PlainDenseDataset<f32, D>>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading .npy file: {e:?}"))
    })
}

fn read_npy_dataset_f16<D>(path: &str) -> PyResult<PlainDenseDataset<f16, D>>
where
    D: ScalarDenseSupportedDistance + std::fmt::Debug,
{
    let dataset_f32 = read_npy_f32::<D>(path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading .npy file: {e:?}"))
    })?;

    let dim = dataset_f32.input_dim();
    let n_vecs = dataset_f32.len();
    let data_f32: Vec<f32> = dataset_f32
        .iter()
        .flat_map(|v| v.values().iter().copied())
        .collect();
    let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let encoder = PlainDenseQuantizer::<f16, D>::new(dim);
    Ok(DenseDataset::from_raw(
        data_f16.into_boxed_slice(),
        n_vecs,
        encoder,
    ))
}

fn convert_components_to_u16(components: &[i32]) -> PyResult<Vec<u16>> {
    let mut out = Vec::with_capacity(components.len());
    for &c in components {
        if c < 0 || c > u16::MAX as i32 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Component out of range for u16",
            ));
        }
        out.push(c as u16);
    }
    Ok(out)
}

fn validate_offsets(offsets: &[usize], values_len: usize) -> PyResult<()> {
    if offsets.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Offsets must be non-empty",
        ));
    }
    if offsets[0] != 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Offsets must start at 0",
        ));
    }
    if let Some(&last) = offsets.last()
        && last != values_len
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Offsets last element must equal number of values",
        ));
    }
    for w in offsets.windows(2) {
        if w[0] > w[1] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Offsets must be non-decreasing",
            ));
        }
    }
    Ok(())
}

fn build_sparse_dataset_from_parts<V, D>(
    components: Vec<u16>,
    values: Vec<V>,
    offsets: Vec<usize>,
    dim: usize,
) -> PyResult<PlainSparseDataset<u16, V, D>>
where
    V: ValueType + Float + FromF32,
    D: ScalarSparseSupportedDistance,
{
    validate_offsets(&offsets, values.len())?;

    let encoder = PlainSparseQuantizer::<u16, V, D>::new(dim, dim);
    let mut dataset: PlainSparseDatasetGrowable<u16, V, D> = DatasetGrowable::new(encoder);

    for i in 0..offsets.len() - 1 {
        let start = offsets[i];
        let end = offsets[i + 1];
        let view = SparseVectorView::new(&components[start..end], &values[start..end]);
        dataset.push(view);
    }

    Ok(dataset.into())
}

fn push_results<D: Distance>(
    results: Vec<vectorium::dataset::ScoredVector<D>>,
    distances: &mut Vec<f32>,
    ids: &mut Vec<i64>,
) {
    for scored in results {
        distances.push(scored.distance.distance());
        ids.push(scored.vector as i64);
    }
}

// Dense plain f32 (internally stored as f16)

enum DensePlainHNSWEnum {
    Euclidean(HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph>),
    DotProduct(HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph>),
}

#[pyclass]
pub struct DensePlainHNSW {
    inner: DensePlainHNSWEnum,
    acorn_gamma: Option<AcornGammaNeighbors>,
}

#[pymethods]
impl DensePlainHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_path, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_path: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = read_npy_dataset_f16::<SquaredEuclideanDistance>(data_path)?;
                DensePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset = read_npy_dataset_f16::<DotProduct>(data_path)?;
                DensePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(DensePlainHNSW {
            inner,
            acorn_gamma: None,
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let data_f32 = data_vec.as_slice()?.to_vec();
        let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let n_vecs = data_f16.len() / dim;
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f16, SquaredEuclideanDistance>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f16, DotProduct>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
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
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, metric="dotproduct".to_string()))]
    pub fn load(path: &str, metric: String) -> PyResult<Self> {
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                DensePlainHNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                DensePlainHNSWEnum::DotProduct(index)
            }
        };
        Ok(DensePlainHNSW {
            inner,
            acorn_gamma: None,
        })
    }

    /// Search for approximate nearest neighbors.
    ///
    /// # Arguments
    /// * `queries` – 1-D float32 numpy array. For a single query, pass an array of length `dimension`.
    ///   For multiple queries, pass a concatenated array of length `num_queries × dimension`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (queries, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        queries: PyReadonlyArray1<f32>,
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
        let mut ids = Vec::new();
        let mut distances = Vec::new();

        let queries_slice = queries.as_slice()?;
        let dim = match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => index.dim(),
            DensePlainHNSWEnum::DotProduct(index) => index.dim(),
        };
        let num_queries = queries_slice.len() / dim;
        ids.reserve(num_queries * k);
        distances.reserve(num_queries * k);

        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => {
                for i in 0..num_queries {
                    let query_start = i * dim;
                    let query_end = (i + 1) * dim;
                    let query_slice = &queries_slice[query_start..query_end];
                    let query_view = DenseVectorView::new(query_slice);
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                for i in 0..num_queries {
                    let query_start = i * dim;
                    let query_end = (i + 1) * dim;
                    let query_slice = &queries_slice[query_start..query_end];
                    let query_view = DenseVectorView::new(query_slice);
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
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
        predicate: PyObject,
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
                push_results(results, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let results = index.search_filtered(query_view, k, &search_config, pred_fn);
                push_results(results, &mut distances, &mut ids);
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
        predicate: PyObject,
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
                push_results(results, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let results = index.search_filtered_gamma(
                    query_view,
                    k,
                    &search_config,
                    acorn_gamma,
                    pred_fn,
                );
                push_results(results, &mut distances, &mut ids);
            }
        }

        let distances_array = PyArray1::from_vec(py, distances).to_owned();
        let ids_array = PyArray1::from_vec(py, ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }
}

// Dense plain f16

// Sparse plain f32 (internally stored as f16)

enum SparsePlainHNSWEnum {
    Euclidean(HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>),
}

#[pyclass]
pub struct SparsePlainHNSW {
    inner: SparsePlainHNSWEnum,
}

#[pymethods]
impl SparsePlainHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset: PlainSparseDataset<u16, f16, SquaredEuclideanDistance> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                SparsePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset: PlainSparseDataset<u16, f16, DotProduct> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                SparsePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(SparsePlainHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_f32 = values.as_slice()?.to_vec();
        let values_f16: Vec<f16> = values_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let offsets_vec = offsets
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();

        // Compute dimensionality from max component index
        let d = components_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f16, SquaredEuclideanDistance>(
                    components_vec,
                    values_f16,
                    offsets_vec,
                    d,
                )?;
                SparsePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f16, DotProduct>(
                    components_vec,
                    values_f16,
                    offsets_vec,
                    d,
                )?;
                SparsePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(SparsePlainHNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparsePlainHNSWEnum::Euclidean(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            SparsePlainHNSWEnum::DotProduct(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, metric="dotproduct".to_string()))]
    pub fn load(path: &str, metric: String) -> PyResult<Self> {
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let index: HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph> = <HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparsePlainHNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> = <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparsePlainHNSWEnum::DotProduct(index)
            }
        };
        Ok(SparsePlainHNSW { inner })
    }

    /// Search for approximate nearest neighbors in sparse data.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of component values (concatenated for batch).
    /// * `offsets` – 1-D int64 array defining query boundaries. For a single query, pass `[0, num_components]`.
    ///   For multiple queries, pass boundaries like `[0, n1, n1+n2, ...]`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut ids = Vec::new();
        let mut distances = Vec::new();

        let num_queries = offsets_slice.len() - 1;
        ids.reserve(num_queries * k);
        distances.reserve(num_queries * k);

        for i in 0..num_queries {
            let query_start = offsets_slice[i] as usize;
            let query_end = offsets_slice[i + 1] as usize;
            let query_comps = &comp_vec[query_start..query_end];
            let query_vals = &values_slice[query_start..query_end];
            let query_view = SparseVectorView::new(query_comps, query_vals);

            match &self.inner {
                SparsePlainHNSWEnum::Euclidean(index) => {
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
                SparsePlainHNSWEnum::DotProduct(index) => {
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

// Sparse DotVByte (dotproduct only)

#[pyclass]
pub struct SparseDotVByteHNSW {
    inner: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph>,
}

#[pymethods]
impl SparseDotVByteHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200))]
    pub fn build_from_file(data_file: &str, m: usize, ef_construction: usize) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let dataset: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(data_file)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error reading dataset: {:?}",
                    e
                ))
            })?;

        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
            plain_hnsw.convert_dataset_into();

        Ok(SparseDotVByteHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_vec = values.as_slice()?.to_vec();
        let offsets_vec = offsets
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();

        // Compute dimensionality from max component index
        let d = components_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
            components_vec,
            values_vec,
            offsets_vec,
            d,
        )?;
        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
            plain_hnsw.convert_dataset_into();

        Ok(SparseDotVByteHNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save_index(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> = <HNSW<
            PackedSparseDataset<DotVByteFixedU8Encoder>,
            Graph,
        > as IndexSerializer>::load_index(
            path
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e))
        })?;

        Ok(SparseDotVByteHNSW { inner })
    }

    /// Search for approximate nearest neighbors in compressed sparse data.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of component values (concatenated for batch).
    /// * `offsets` – 1-D int64 array defining query boundaries. For a single query, pass `[0, num_components]`.
    ///   For multiple queries, pass boundaries like `[0, n1, n1+n2, ...]`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let num_queries = offsets_slice.len() - 1;
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);

        for i in 0..num_queries {
            let query_start = offsets_slice[i] as usize;
            let query_end = offsets_slice[i + 1] as usize;
            let query_comps = &comp_vec[query_start..query_end];
            let query_vals = &values_slice[query_start..query_end];
            let query_view = SparseVectorView::new(query_comps, query_vals);
            let results = self.inner.search(query_view, k, &search_config);
            push_results(results, &mut distances, &mut ids);
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

// Sparse scalar fixedu8/fixedu16

enum SparseFixedU8HNSWEnum {
    Euclidean(HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph>),
}

#[pyclass]
pub struct SparseFixedU8HNSW {
    inner: SparseFixedU8HNSWEnum,
}

#[pymethods]
impl SparseFixedU8HNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                    Graph,
                > = plain_hnsw.convert_dataset_into();
                SparseFixedU8HNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let dataset: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
                    plain_hnsw.convert_dataset_into();
                SparseFixedU8HNSWEnum::DotProduct(index)
            }
        };

        Ok(SparseFixedU8HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_vec = values.as_slice()?.to_vec();
        let offsets_vec = offsets
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();

        // Compute dimensionality from max component index
        let d = components_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f32, SquaredEuclideanDistance>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                    Graph,
                > = plain_hnsw.convert_dataset_into();
                SparseFixedU8HNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
                    plain_hnsw.convert_dataset_into();
                SparseFixedU8HNSWEnum::DotProduct(index)
            }
        };

        Ok(SparseFixedU8HNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparseFixedU8HNSWEnum::Euclidean(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            SparseFixedU8HNSWEnum::DotProduct(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, metric="dotproduct".to_string()))]
    pub fn load(path: &str, metric: String) -> PyResult<Self> {
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparseFixedU8HNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparseFixedU8HNSWEnum::DotProduct(index)
            }
        };
        Ok(SparseFixedU8HNSW { inner })
    }

    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let num_queries = offsets_slice.len() - 1;
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);

        for i in 0..num_queries {
            let query_start = offsets_slice[i] as usize;
            let query_end = offsets_slice[i + 1] as usize;
            let query_comps = &comp_vec[query_start..query_end];
            let query_vals = &values_slice[query_start..query_end];
            let query_view = SparseVectorView::new(query_comps, query_vals);

            match &self.inner {
                SparseFixedU8HNSWEnum::Euclidean(index) => {
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
                SparseFixedU8HNSWEnum::DotProduct(index) => {
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

enum SparseFixedU16HNSWEnum {
    Euclidean(HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph>),
}

#[pyclass]
pub struct SparseFixedU16HNSW {
    inner: SparseFixedU16HNSWEnum,
}

#[pymethods]
impl SparseFixedU16HNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                    Graph,
                > = plain_hnsw.convert_dataset_into();
                SparseFixedU16HNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let dataset: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
                    plain_hnsw.convert_dataset_into();
                SparseFixedU16HNSWEnum::DotProduct(index)
            }
        };

        Ok(SparseFixedU16HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_vec = values.as_slice()?.to_vec();
        let offsets_vec = offsets
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();

        // Compute dimensionality from max component index
        let d = components_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f32, SquaredEuclideanDistance>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                    Graph,
                > = plain_hnsw.convert_dataset_into();
                SparseFixedU16HNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
                    plain_hnsw.convert_dataset_into();
                SparseFixedU16HNSWEnum::DotProduct(index)
            }
        };

        Ok(SparseFixedU16HNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparseFixedU16HNSWEnum::Euclidean(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            SparseFixedU16HNSWEnum::DotProduct(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (path, metric="dotproduct".to_string()))]
    pub fn load(path: &str, metric: String) -> PyResult<Self> {
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparseFixedU16HNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparseFixedU16HNSWEnum::DotProduct(index)
            }
        };
        Ok(SparseFixedU16HNSW { inner })
    }

    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let num_queries = offsets_slice.len() - 1;
        let mut ids = Vec::with_capacity(num_queries * k);
        let mut distances = Vec::with_capacity(num_queries * k);

        for i in 0..num_queries {
            let query_start = offsets_slice[i] as usize;
            let query_end = offsets_slice[i + 1] as usize;
            let query_comps = &comp_vec[query_start..query_end];
            let query_vals = &values_slice[query_start..query_end];
            let query_view = SparseVectorView::new(query_comps, query_vals);

            match &self.inner {
                SparseFixedU16HNSWEnum::Euclidean(index) => {
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
                SparseFixedU16HNSWEnum::DotProduct(index) => {
                    let results = index.search(query_view, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

// PQ (dense only)

enum DensePQHNSWGeneric<D>
where
    D: ProductQuantizerDistance,
{
    PQ8(HNSW<DenseDataset<ProductQuantizer<8, D>>, Graph>),
    PQ16(HNSW<DenseDataset<ProductQuantizer<16, D>>, Graph>),
    PQ32(HNSW<DenseDataset<ProductQuantizer<32, D>>, Graph>),
    PQ48(HNSW<DenseDataset<ProductQuantizer<48, D>>, Graph>),
    PQ64(HNSW<DenseDataset<ProductQuantizer<64, D>>, Graph>),
    PQ96(HNSW<DenseDataset<ProductQuantizer<96, D>>, Graph>),
    PQ128(HNSW<DenseDataset<ProductQuantizer<128, D>>, Graph>),
    PQ192(HNSW<DenseDataset<ProductQuantizer<192, D>>, Graph>),
    PQ256(HNSW<DenseDataset<ProductQuantizer<256, D>>, Graph>),
    PQ384(HNSW<DenseDataset<ProductQuantizer<384, D>>, Graph>),
}

impl DensePQHNSWGeneric<DotProduct> {
    fn build_from_dataset(
        dataset: PlainDenseDataset<f32, DotProduct>,
        config: &HNSWBuildConfiguration,
        m_pq: usize,
    ) -> PyResult<Self> {
        match m_pq {
            8 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<8, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ8(index))
            }
            16 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<16, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ16(index))
            }
            32 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<32, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ32(index))
            }
            48 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<48, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ48(index))
            }
            64 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<64, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ64(index))
            }
            96 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<96, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ96(index))
            }
            128 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<128, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ128(index))
            }
            192 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<192, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ192(index))
            }
            256 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<256, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ256(index))
            }
            384 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<DenseDataset<ProductQuantizer<384, DotProduct>>, Graph> =
                    plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ384(index))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
            )),
        }
    }
}

impl DensePQHNSWGeneric<SquaredEuclideanDistance> {
    fn build_from_dataset(
        dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
        config: &HNSWBuildConfiguration,
        m_pq: usize,
    ) -> PyResult<Self> {
        match m_pq {
            8 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<8, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ8(index))
            }
            16 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<16, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ16(index))
            }
            32 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<32, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ32(index))
            }
            48 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<48, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ48(index))
            }
            64 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<64, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ64(index))
            }
            96 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<96, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ96(index))
            }
            128 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<128, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ128(index))
            }
            192 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<192, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ192(index))
            }
            256 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<256, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ256(index))
            }
            384 => {
                let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
                let index: HNSW<
                    DenseDataset<ProductQuantizer<384, SquaredEuclideanDistance>>,
                    Graph,
                > = plain_index.convert_dataset_into();
                Ok(DensePQHNSWGeneric::PQ384(index))
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
            )),
        }
    }
}

impl<D> DensePQHNSWGeneric<D>
where
    D: ProductQuantizerDistance + Distance + ScalarDenseSupportedDistance,
{
    fn load(path: &str, m_pq: usize) -> PyResult<Self> {
        let inner = match m_pq {
            8 => {
                let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<8, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ8(index)
            }
            16 => {
                let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<16, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ16(index)
            }
            32 => {
                let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<32, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ32(index)
            }
            48 => {
                let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<48, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ48(index)
            }
            64 => {
                let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<64, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ64(index)
            }
            96 => {
                let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<96, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ96(index)
            }
            128 => {
                let index: HNSW<DenseDataset<ProductQuantizer<128, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ128(index)
            }
            192 => {
                let index: HNSW<DenseDataset<ProductQuantizer<192, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ192(index)
            }
            256 => {
                let index: HNSW<DenseDataset<ProductQuantizer<256, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ256(index)
            }
            384 => {
                let index: HNSW<DenseDataset<ProductQuantizer<384, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error loading index: {:?}",
                        e
                    ))
                })?;
                DensePQHNSWGeneric::PQ384(index)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value for load. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ));
            }
        };
        Ok(inner)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let result = match self {
            DensePQHNSWGeneric::PQ8(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ16(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ32(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ48(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ64(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ96(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ128(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ192(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ256(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ384(index) => index.save_index(path),
        };

        result.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    fn search(
        &self,
        query: DenseVectorView<'_, f32>,
        k: usize,
        search_config: &HNSWSearchConfiguration,
    ) -> Vec<vectorium::dataset::ScoredVector<D>> {
        match self {
            DensePQHNSWGeneric::PQ8(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ16(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ32(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ48(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ64(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ96(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ128(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ192(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ256(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ384(index) => index.search(query, k, search_config),
        }
    }
}

enum DensePQHNSWEnum {
    Euclidean(DensePQHNSWGeneric<SquaredEuclideanDistance>),
    DotProduct(DensePQHNSWGeneric<DotProduct>),
}

#[pyclass]
pub struct DensePQHNSW {
    inner: DensePQHNSWEnum,
}

#[pymethods]
impl DensePQHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_path, m_pq, nbits=8, m=32, ef_construction=200, metric="dotproduct".to_string(), sample_size=100_000))]
    pub fn build_from_file(
        data_path: &str,
        m_pq: usize,
        nbits: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        sample_size: usize,
    ) -> PyResult<Self> {
        if nbits != 8 {
            eprintln!("Warning: vectorium PQ ignores nbits (fixed codebook size).");
        }
        if sample_size != 100_000 {
            eprintln!("Warning: vectorium PQ ignores sample_size and uses automatic sampling.");
        }

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
                    read_npy_dataset::<SquaredEuclideanDistance>(data_path)?;
                DensePQHNSWEnum::Euclidean(
                    DensePQHNSWGeneric::<SquaredEuclideanDistance>::build_from_dataset(
                        dataset, &config, m_pq,
                    )?,
                )
            }
            MetricKind::DotProduct => {
                let dataset: PlainDenseDataset<f32, DotProduct> =
                    read_npy_dataset::<DotProduct>(data_path)?;
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::<DotProduct>::build_from_dataset(
                    dataset, &config, m_pq,
                )?)
            }
        };

        Ok(DensePQHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m_pq, nbits=8, m=32, ef_construction=200, metric="dotproduct".to_string(), sample_size=100_000))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m_pq: usize,
        nbits: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        sample_size: usize,
    ) -> PyResult<Self> {
        if nbits != 8 {
            eprintln!("Warning: vectorium PQ ignores nbits (fixed codebook size).");
        }
        if sample_size != 100_000 {
            eprintln!("Warning: vectorium PQ ignores sample_size and uses automatic sampling.");
        }

        let data_vec = data_vec.as_slice()?.to_vec();
        let n_vecs = data_vec.len() / dim;
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
                let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePQHNSWEnum::Euclidean(
                    DensePQHNSWGeneric::<SquaredEuclideanDistance>::build_from_dataset(
                        dataset, &config, m_pq,
                    )?,
                )
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(dim);
                let dataset: PlainDenseDataset<f32, DotProduct> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::<DotProduct>::build_from_dataset(
                    dataset, &config, m_pq,
                )?)
            }
        };

        Ok(DensePQHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (path, m_pq, metric="dotproduct".to_string()))]
    pub fn load(path: &str, m_pq: usize, metric: String) -> PyResult<Self> {
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                DensePQHNSWEnum::Euclidean(DensePQHNSWGeneric::load(path, m_pq)?)
            }
            MetricKind::DotProduct => {
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::load(path, m_pq)?)
            }
        };
        Ok(DensePQHNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => inner.save(path),
            DensePQHNSWEnum::DotProduct(inner) => inner.save(path),
        }
    }

    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
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

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => {
                let results = inner.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
            DensePQHNSWEnum::DotProduct(inner) => {
                let results = inner.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

// Multivector Reranking

/// Helper to load plain multivector dataset from folder
fn load_multivec_dataset_plain(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    let documents_path = Path::new(data_folder).join("documents.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");

    let documents_file = File::open(&documents_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening documents file at {:?}: {}",
            documents_path, e
        ))
    })?;
    let documents_u16: Array2<u16> =
        Array2::read_npy(BufReader::new(documents_file)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading documents array: {}",
                e
            ))
        })?;

    let (_n_tokens, token_dim) = documents_u16.dim();
    let documents_raw = documents_u16.into_raw_vec_and_offset().0;
    let mut documents_flat: Vec<f32> = Vec::with_capacity(documents_raw.len());
    for u16_val in documents_raw {
        let f16_val = f16::from_bits(u16_val);
        documents_flat.push(f32::from(f16_val));
    }

    let doclens_file = File::open(&doclens_path).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
            "Error opening doclens file at {:?}: {}",
            doclens_path, e
        ))
    })?;
    let doclens_array: ndarray::Array1<i32> =
        ndarray::Array1::read_npy(BufReader::new(doclens_file)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading doclens array: {}",
                e
            ))
        })?;

    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Build offsets array from doclens
    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * token_dim);
    }

    let encoder = PlainMultiVecQuantizer::<f32>::new(token_dim);
    Ok(MultiVectorDataset::from_raw(
        documents_flat.into(),
        offsets.into(),
        encoder,
    ))
}

// Flat indexes for ground truth computation

enum DenseFlatIndexEnum {
    Euclidean(DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>),
    DotProduct(DenseDataset<PlainDenseQuantizer<f16, DotProduct>>),
}

#[pyclass]
pub struct DenseFlatIndex {
    inner: DenseFlatIndexEnum,
}

#[pymethods]
impl DenseFlatIndex {
    #[staticmethod]
    #[pyo3(signature = (data_path, metric="dotproduct".to_string()))]
    pub fn build_from_file(data_path: &str, metric: String) -> PyResult<Self> {
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = read_npy_dataset_f16::<SquaredEuclideanDistance>(data_path)?;
                DenseFlatIndexEnum::Euclidean(dataset)
            }
            MetricKind::DotProduct => {
                let dataset = read_npy_dataset_f16::<DotProduct>(data_path)?;
                DenseFlatIndexEnum::DotProduct(dataset)
            }
        };

        Ok(DenseFlatIndex { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, metric="dotproduct".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        metric: String,
    ) -> PyResult<Self> {
        let data_f32 = data_vec.as_slice()?.to_vec();
        let data_f16: Vec<f16> = data_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let n_vecs = data_f16.len() / dim;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f16, SquaredEuclideanDistance>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DenseFlatIndexEnum::Euclidean(dataset)
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f16, DotProduct>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DenseFlatIndexEnum::DotProduct(dataset)
            }
        };

        Ok(DenseFlatIndex { inner })
    }

    /// Exhaustive search over all vectors for exact nearest neighbors.
    ///
    /// # Arguments
    /// * `queries` – 2-D float32 numpy array of shape (num_queries, dim).
    /// * `k` – Number of nearest neighbors to return per query.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (queries, k))]
    #[pyo3(text_signature = "(queries, k)")]
    pub fn search(
        &self,
        queries: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let queries_slice = queries.as_slice()?;

        // Infer dimension from dataset
        let dim = match &self.inner {
            DenseFlatIndexEnum::Euclidean(dataset) => dataset.input_dim(),
            DenseFlatIndexEnum::DotProduct(dataset) => dataset.input_dim(),
        };

        let num_queries = queries_slice.len() / dim;

        // Collect query indices
        let query_indices: Vec<usize> = (0..num_queries).collect();

        let query_results: Vec<(Vec<f32>, Vec<i64>)> = match &self.inner {
            DenseFlatIndexEnum::Euclidean(dataset) => query_indices
                .into_par_iter()
                .map(|i| {
                    let query_start = i * dim;
                    let query_end = (i + 1) * dim;
                    let query_slice = &queries_slice[query_start..query_end];
                    let query_view = DenseVectorView::new(query_slice);
                    let results = dataset.search(query_view, k);

                    let mut distances = Vec::new();
                    let mut ids = Vec::new();
                    push_results(results, &mut distances, &mut ids);
                    (distances, ids)
                })
                .collect(),
            DenseFlatIndexEnum::DotProduct(dataset) => query_indices
                .into_par_iter()
                .map(|i| {
                    let query_start = i * dim;
                    let query_end = (i + 1) * dim;
                    let query_slice = &queries_slice[query_start..query_end];
                    let query_view = DenseVectorView::new(query_slice);
                    let results = dataset.search(query_view, k);

                    let mut distances = Vec::new();
                    let mut ids = Vec::new();
                    push_results(results, &mut distances, &mut ids);
                    (distances, ids)
                })
                .collect(),
        };

        let mut all_distances = Vec::new();
        let mut all_ids = Vec::new();
        for (distances, ids) in query_results {
            all_distances.extend(distances);
            all_ids.extend(ids);
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
            let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

enum SparseFlatIndexEnum {
    DotProduct(PlainSparseDataset<u16, f16, DotProduct>),
}

#[pyclass]
pub struct SparseFlatIndex {
    inner: SparseFlatIndexEnum,
}

#[pymethods]
impl SparseFlatIndex {
    #[staticmethod]
    #[pyo3(signature = (components, values, offsets))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
    ) -> PyResult<Self> {
        let comp_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_f32 = values.as_slice()?.to_vec();
        let values_vec: Vec<f16> = values_f32.iter().map(|&x| f16::from_f32(x)).collect();
        let offsets_slice = offsets.as_slice()?;
        let offsets_usize: Vec<usize> = offsets_slice.iter().map(|&x| x as usize).collect();

        // Compute dimensionality from max component index
        let dim = comp_vec
            .iter()
            .max()
            .map(|&x| (x as usize) + 1)
            .unwrap_or(0);

        let dataset = build_sparse_dataset_from_parts::<f16, DotProduct>(
            comp_vec,
            values_vec,
            offsets_usize,
            dim,
        )?;

        Ok(SparseFlatIndex {
            inner: SparseFlatIndexEnum::DotProduct(dataset),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data_file))]
    pub fn build_from_file(data_file: &str) -> PyResult<Self> {
        let data = read_seismic_format::<u16, f16, DotProduct>(data_file).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error reading seismic format file: {e:?}"
            ))
        })?;

        Ok(SparseFlatIndex {
            inner: SparseFlatIndexEnum::DotProduct(data),
        })
    }

    /// Exhaustive search over all vectors for exact nearest neighbors.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of component values (concatenated for batch).
    /// * `offsets` – 1-D int64 array defining query boundaries. For a single query, pass `[0, num_components]`.
    ///   For multiple queries, pass boundaries like `[0, n1, n1+n2, ...]`.
    /// * `k` – Number of nearest neighbors to return per query.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;

        let num_queries = offsets_slice.len() - 1;

        let query_results: Vec<(Vec<f32>, Vec<i64>)> = match &self.inner {
            SparseFlatIndexEnum::DotProduct(dataset) => (0..num_queries)
                .into_par_iter()
                .map(|i| {
                    let query_start = offsets_slice[i] as usize;
                    let query_end = offsets_slice[i + 1] as usize;
                    let query_comps = &comp_vec[query_start..query_end];
                    let query_vals = &values_slice[query_start..query_end];
                    let query_view = SparseVectorView::new(query_comps, query_vals);
                    let results = dataset.search(query_view, k);

                    let mut distances = Vec::new();
                    let mut ids = Vec::new();
                    push_results(results, &mut distances, &mut ids);
                    (distances, ids)
                })
                .collect(),
        };

        let mut all_distances = Vec::new();
        let mut all_ids = Vec::new();
        for (distances, ids) in query_results {
            all_distances.extend(distances);
            all_ids.extend(ids);
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
            let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

#[pyclass]
pub struct SparseMultivecRerankIndex {
    inner: RerankIndex<
        HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
        PlainSparseDataset<u16, f16, DotProduct>,
        MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
    >,
}

#[pymethods]
impl SparseMultivecRerankIndex {
    /// Build a rerank index from a pre-built sparse HNSW index and multivector data folder.
    ///
    /// # Arguments
    /// * `sparse_index_path` – Path to the pre-built sparse HNSW index file.
    /// * `multivec_data_folder` – Path to folder containing multivector data files (plain quantizer).
    ///
    /// # Multivector Data Folder Structure (Plain Quantizer)
    /// The folder must contain the following files:
    /// * `documents.npy` – Dense document embeddings (shape: [n_documents, n_tokens, token_dim], dtype: float32)
    /// * `queries.npy` – Dense query embeddings (shape: [n_queries, n_tokens, token_dim], dtype: float32)
    /// * `doclens.npy` – Document lengths (shape: [n_documents], dtype: int32 or int64)
    ///
    #[staticmethod]
    #[pyo3(signature = (sparse_index_path, multivec_data_folder))]
    pub fn build_from_file(sparse_index_path: &str, multivec_data_folder: &str) -> PyResult<Self> {
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

        let multivec_dataset = load_multivec_dataset_plain(multivec_data_folder)?;

        Ok(SparseMultivecRerankIndex {
            inner: RerankIndex::new(sparse_index, multivec_dataset),
        })
    }

    /// Batch search with reranking using plain multivector encoding.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of sparse query component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of sparse query values (concatenated for batch).
    /// * `sparse_offsets` – 1-D int64 array defining sparse query boundaries. For N queries, pass [0, n1, n1+n2, ..., total].
    /// * `multivec_queries` – 1-D float32 array of all multivector queries concatenated (total_queries × n_tokens × token_dim).
    /// * `n_tokens` – Number of tokens per multivector query (fixed).
    /// * `token_dim` – Dimension of each token in the multivector queries.
    /// * `k_candidates` – Number of candidates to retrieve in first stage. Default: 100.
    /// * `k` – Number of final results to return per query. Default: 10.
    /// * `ef_search` – Candidate list size for HNSW search. Default: 100.
    /// * `alpha` – Alpha parameter for candidate pruning (0-1). Default: None.
    /// * `beta` – Beta parameter for early exit. Default: None.
    /// * `early_exit_threshold` – Lambda for early termination. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, sparse_offsets, multivec_queries, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None))]
    pub fn search(
        &self,
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

        let mut all_distances = Vec::new();
        let mut all_ids = Vec::new();
        let num_queries = sparse_offsets_slice.len() - 1;
        let multivec_query_size = n_tokens * token_dim;

        for q_idx in 0..num_queries {
            // Extract sparse query for this index
            let sparse_start = sparse_offsets_slice[q_idx] as usize;
            let sparse_end = sparse_offsets_slice[q_idx + 1] as usize;
            let query_comps = &comp_vec[sparse_start..sparse_end];
            let query_vals = &query_values_slice[sparse_start..sparse_end];
            let sparse_query = SparseVectorView::new(query_comps, query_vals);

            // Extract multivector query for this index (chunked by n_tokens * token_dim)
            let multivec_start = q_idx * multivec_query_size;
            let multivec_end = (q_idx + 1) * multivec_query_size;
            let multivec_query_flat = &multivec_queries_slice[multivec_start..multivec_end];
            let multivec_query_view = DenseMultiVectorView::new(multivec_query_flat, token_dim);

            let results = self.inner.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                alpha,
                beta,
            );

            for result in results {
                all_distances.push(result.distance.distance());
                all_ids.push(result.vector as i64);
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
            let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

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
            PlainSparseDataset<u16, f16, DotProduct>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
    M16(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            PlainSparseDataset<u16, f16, DotProduct>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
    M32(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            PlainSparseDataset<u16, f16, DotProduct>,
            MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
        >,
    ),
    M64(
        RerankIndex<
            HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
            PlainSparseDataset<u16, f16, DotProduct>,
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
    /// * `queries.npy` – Dense query embeddings (shape: [n_queries, n_tokens, token_dim], dtype: float32)
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

    /// Batch search with reranking using two-level PQ multivector encoding.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of sparse query component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of sparse query values (concatenated for batch).
    /// * `sparse_offsets` – 1-D int64 array defining sparse query boundaries. For N queries, pass [0, n1, n1+n2, ..., total].
    /// * `multivec_queries` – 1-D float32 array of all multivector queries concatenated (total_queries × n_tokens × token_dim).
    /// * `n_tokens` – Number of tokens per multivector query (fixed).
    /// * `token_dim` – Dimension of each token in the multivector queries.
    /// * `k_candidates` – Number of candidates to retrieve in first stage. Default: 100.
    /// * `k` – Number of final results to return per query. Default: 10.
    /// * `ef_search` – Candidate list size for HNSW search. Default: 100.
    /// * `alpha` – Alpha parameter for candidate pruning (0-1). Default: None.
    /// * `beta` – Beta parameter for early exit. Default: None.
    /// * `early_exit_threshold` – Lambda for early termination. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, sparse_offsets, multivec_queries, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None))]
    pub fn search(
        &self,
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

        let mut all_distances = Vec::new();
        let mut all_ids = Vec::new();
        let num_queries = sparse_offsets_slice.len() - 1;
        let multivec_query_size = n_tokens * token_dim;

        for q_idx in 0..num_queries {
            // Extract sparse query for this index
            let sparse_start = sparse_offsets_slice[q_idx] as usize;
            let sparse_end = sparse_offsets_slice[q_idx + 1] as usize;
            let query_comps = &comp_vec[sparse_start..sparse_end];
            let query_vals = &query_values_slice[sparse_start..sparse_end];
            let sparse_query = SparseVectorView::new(query_comps, query_vals);

            // Extract multivector query for this index (chunked by n_tokens * token_dim)
            let multivec_start = q_idx * multivec_query_size;
            let multivec_end = (q_idx + 1) * multivec_query_size;
            let multivec_query_flat = &multivec_queries_slice[multivec_start..multivec_end];
            let multivec_query_view = DenseMultiVectorView::new(multivec_query_flat, token_dim);

            let results = match &self.inner {
                SparseMultivecTwoLevelsPQRerankIndexEnum::M8(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    alpha,
                    beta,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M16(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    alpha,
                    beta,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M32(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    alpha,
                    beta,
                ),
                SparseMultivecTwoLevelsPQRerankIndexEnum::M64(rerank_index) => rerank_index.search(
                    sparse_query,
                    multivec_query_view,
                    k_candidates,
                    k,
                    &search_config,
                    alpha,
                    beta,
                ),
            };

            for result in results {
                all_distances.push(result.distance.distance());
                all_ids.push(result.vector as i64);
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
            let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}
