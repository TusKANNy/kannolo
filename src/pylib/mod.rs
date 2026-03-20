use std::f32;

use crate::graph::Graph;
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};
use crate::index::Index;
use vectorium::IndexSerializer;

use half::f16;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector::{DenseVectorView, SparseVectorView};
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, FixedU8Q, FixedU16Q, Float, FromF32,
    PackedSparseDataset, PlainDenseDataset, PlainSparseDataset, PlainSparseDatasetGrowable,
    ScalarSparseDataset, ValueType,
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

// Dense plain f32

enum DensePlainHNSWEnum {
    Euclidean(HNSW<DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>, Graph>),
    DotProduct(HNSW<DenseDataset<PlainDenseQuantizer<f32, DotProduct>>, Graph>),
}

#[pyclass]
pub struct DensePlainHNSW {
    inner: DensePlainHNSWEnum,
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
                let dataset = read_npy_dataset::<SquaredEuclideanDistance>(data_path)?;
                DensePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset = read_npy_dataset::<DotProduct>(data_path)?;
                DensePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(DensePlainHNSW { inner })
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
        let data_vec = data_vec.as_slice()?.to_vec();
        let n_vecs = data_vec.len() / dim;
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(DensePlainHNSW { inner })
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
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f32, SquaredEuclideanDistance>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                DensePlainHNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f32, DotProduct>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f32, DotProduct>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                DensePlainHNSWEnum::DotProduct(index)
            }
        };
        Ok(DensePlainHNSW { inner })
    }

    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        let query_view = DenseVectorView::new(query_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => {
                let results = index.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let results = index.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        queries_path: &str,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        let mut ids = Vec::new();
        let mut distances = Vec::new();

        match &self.inner {
            DensePlainHNSWEnum::Euclidean(index) => {
                let queries = read_npy_dataset::<SquaredEuclideanDistance>(queries_path)?;
                let num_queries = queries.len();
                ids.reserve(num_queries * k);
                distances.reserve(num_queries * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let queries = read_npy_dataset::<DotProduct>(queries_path)?;
                let num_queries = queries.len();
                ids.reserve(num_queries * k);
                distances.reserve(num_queries * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
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
    pub fn search_filtered(
        &self,
        py: Python<'_>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        predicate: PyObject,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        let query_view = DenseVectorView::new(query_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

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
                let results = index.search_filtered(query_view, k, &search_config, &pred_fn);
                push_results(results, &mut distances, &mut ids);
            }
            DensePlainHNSWEnum::DotProduct(index) => {
                let results = index.search_filtered(query_view, k, &search_config, &pred_fn);
                push_results(results, &mut distances, &mut ids);
            }
        }

        let distances_array = PyArray1::from_vec(py, distances).to_owned();
        let ids_array = PyArray1::from_vec(py, ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }
}

// Dense plain f16

enum DensePlainHNSWf16Enum {
    Euclidean(HNSW<DenseDataset<PlainDenseQuantizer<f16, SquaredEuclideanDistance>>, Graph>),
    DotProduct(HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph>),
}

#[pyclass]
pub struct DensePlainHNSWf16 {
    inner: DensePlainHNSWf16Enum,
}

#[pymethods]
impl DensePlainHNSWf16 {
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
                let data_f32 = read_npy_dataset::<SquaredEuclideanDistance>(data_path)?;
                let dim = data_f32.input_dim();
                let n_vecs = data_f32.len();
                let data_f16: Vec<f16> = data_f32
                    .values()
                    .iter()
                    .map(|v| f16::from_f32(*v))
                    .collect();
                let encoder = PlainDenseQuantizer::<f16, SquaredEuclideanDistance>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWf16Enum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let data_f32 = read_npy_dataset::<DotProduct>(data_path)?;
                let dim = data_f32.input_dim();
                let n_vecs = data_f32.len();
                let data_f16: Vec<f16> = data_f32
                    .values()
                    .iter()
                    .map(|v| f16::from_f32(*v))
                    .collect();
                let encoder = PlainDenseQuantizer::<f16, DotProduct>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWf16Enum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(DensePlainHNSWf16 { inner })
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
        let data_vec = data_vec.as_slice()?.to_vec();
        let n_vecs = data_vec.len() / dim;
        let data_f16: Vec<f16> = data_vec.into_iter().map(f16::from_f32).collect();
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f16, SquaredEuclideanDistance>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWf16Enum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f16, DotProduct>::new(dim);
                let dataset: DenseDataset<_> =
                    DenseDataset::from_raw(data_f16.into_boxed_slice(), n_vecs, encoder);
                DensePlainHNSWf16Enum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(DensePlainHNSWf16 { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            DensePlainHNSWf16Enum::Euclidean(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            DensePlainHNSWf16Enum::DotProduct(index) => index.save_index(path).map_err(|e| {
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
                DensePlainHNSWf16Enum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph> = <HNSW<DenseDataset<PlainDenseQuantizer<f16, DotProduct>>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                DensePlainHNSWf16Enum::DotProduct(index)
            }
        };
        Ok(DensePlainHNSWf16 { inner })
    }

    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        let query_view = DenseVectorView::new(query_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        match &self.inner {
            DensePlainHNSWf16Enum::Euclidean(index) => {
                let results = index.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
            DensePlainHNSWf16Enum::DotProduct(index) => {
                let results = index.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        queries_path: &str,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        let mut ids = Vec::new();
        let mut distances = Vec::new();

        match &self.inner {
            DensePlainHNSWf16Enum::Euclidean(index) => {
                let queries = read_npy_dataset::<SquaredEuclideanDistance>(queries_path)?;
                let num_queries = queries.len();
                ids.reserve(num_queries * k);
                distances.reserve(num_queries * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            DensePlainHNSWf16Enum::DotProduct(index) => {
                let queries = read_npy_dataset::<DotProduct>(queries_path)?;
                let num_queries = queries.len();
                ids.reserve(num_queries * k);
                distances.reserve(num_queries * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
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

// Sparse plain f32

enum SparsePlainHNSWEnum {
    Euclidean(HNSW<PlainSparseDataset<u16, f32, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph>),
}

#[pyclass]
pub struct SparsePlainHNSW {
    inner: SparsePlainHNSWEnum,
}

#[pymethods]
impl SparsePlainHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
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
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
                SparsePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
                SparsePlainHNSWEnum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(SparsePlainHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
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
                SparsePlainHNSWEnum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
                    components_vec,
                    values_vec,
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
                let index: HNSW<PlainSparseDataset<u16, f32, SquaredEuclideanDistance>, Graph> = <HNSW<PlainSparseDataset<u16, f32, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparsePlainHNSWEnum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph> = <HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparsePlainHNSWEnum::DotProduct(index)
            }
        };
        Ok(SparsePlainHNSW { inner })
    }

    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        _d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
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

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut ids = Vec::new();
        let mut distances = Vec::new();

        match &self.inner {
            SparsePlainHNSWEnum::Euclidean(index) => {
                let queries: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            SparsePlainHNSWEnum::DotProduct(index) => {
                let queries: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
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

// Sparse plain f16

enum SparsePlainHNSWf16Enum {
    Euclidean(HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>),
}

#[pyclass]
pub struct SparsePlainHNSWf16 {
    inner: SparsePlainHNSWf16Enum,
}

#[pymethods]
impl SparsePlainHNSWf16 {
    #[staticmethod]
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
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
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
                SparsePlainHNSWf16Enum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset: PlainSparseDataset<u16, f16, DotProduct> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
                SparsePlainHNSWf16Enum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(SparsePlainHNSWf16 { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_vec: Vec<f16> = values
            .as_slice()?
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
        let offsets_vec = offsets
            .as_slice()?
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>();

        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f16, SquaredEuclideanDistance>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                SparsePlainHNSWf16Enum::Euclidean(HNSW::build_index(dataset, &config))
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f16, DotProduct>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                SparsePlainHNSWf16Enum::DotProduct(HNSW::build_index(dataset, &config))
            }
        };

        Ok(SparsePlainHNSWf16 { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparsePlainHNSWf16Enum::Euclidean(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            SparsePlainHNSWf16Enum::DotProduct(index) => index.save_index(path).map_err(|e| {
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
                SparsePlainHNSWf16Enum::Euclidean(index)
            }
            MetricKind::DotProduct => {
                let index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> = <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading index: {:?}", e)))?;
                SparsePlainHNSWf16Enum::DotProduct(index)
            }
        };
        Ok(SparsePlainHNSWf16 { inner })
    }

    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        _d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        match &self.inner {
            SparsePlainHNSWf16Enum::Euclidean(index) => {
                let results = index.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
            SparsePlainHNSWf16Enum::DotProduct(index) => {
                let results = index.search(query_view, k, &search_config);
                push_results(results, &mut distances, &mut ids);
            }
        }

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut ids = Vec::new();
        let mut distances = Vec::new();

        match &self.inner {
            SparsePlainHNSWf16Enum::Euclidean(index) => {
                let queries: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            SparsePlainHNSWf16Enum::DotProduct(index) => {
                let queries: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
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
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
        m: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
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
        if d != dataset.input_dim() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Provided dimension does not match dataset",
            ));
        }

        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
            plain_hnsw.convert_dataset_into();

        Ok(SparseDotVByteHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
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

    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        _d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        let results = self.inner.search(query_view, k, &search_config);
        push_results(results, &mut distances, &mut ids);

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(query_file)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error reading query file: {:?}",
                    e
                ))
            })?;
        if d != queries.input_dim() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Provided dimension does not match query dataset",
            ));
        }

        let mut ids = Vec::with_capacity(queries.len() * k);
        let mut distances = Vec::with_capacity(queries.len() * k);
        for query in queries.iter() {
            let results = self.inner.search(query, k, &search_config);
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
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
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
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
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
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> =
                    plain_hnsw.convert_dataset_into();
                SparseFixedU8HNSWEnum::DotProduct(index)
            }
        };

        Ok(SparseFixedU8HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
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

    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        _d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
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

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut ids = Vec::new();
        let mut distances = Vec::new();

        match &self.inner {
            SparseFixedU8HNSWEnum::Euclidean(index) => {
                let queries: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            SparseFixedU8HNSWEnum::DotProduct(index) => {
                let queries: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
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
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
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
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
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
                if d != dataset.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match dataset",
                    ));
                }
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> =
                    plain_hnsw.convert_dataset_into();
                SparseFixedU16HNSWEnum::DotProduct(index)
            }
        };

        Ok(SparseFixedU16HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200, metric="dotproduct".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
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

    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        _d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
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

        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

        let mut ids = Vec::new();
        let mut distances = Vec::new();

        match &self.inner {
            SparseFixedU16HNSWEnum::Euclidean(index) => {
                let queries: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
                    push_results(results, &mut distances, &mut ids);
                }
            }
            SparseFixedU16HNSWEnum::DotProduct(index) => {
                let queries: PlainSparseDataset<u16, f32, DotProduct> =
                    read_seismic_format(query_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading query file: {:?}",
                            e
                        ))
                    })?;
                if d != queries.input_dim() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Provided dimension does not match query dataset",
                    ));
                }
                ids.reserve(queries.len() * k);
                distances.reserve(queries.len() * k);
                for query in queries.iter() {
                    let results = index.search(query, k, &search_config);
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
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        let query_view = DenseVectorView::new(query_slice);
        let search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);

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
