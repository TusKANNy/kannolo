//! Python bindings.
//!
//! Two clippy lints are allowed module-wide rather than at ~35 individual sites:
//!
//! * `too_many_arguments` — every `#[pyfunction]`/`#[pymethods]` signature here mirrors the
//!   Python-facing keyword arguments one-for-one. Bundling them into a struct would change
//!   the Python API to satisfy a Rust lint.
//! * `type_complexity` — the binding layer names fully monomorphized index types
//!   (`HNSW<DenseDataset<ProductQuantizer<M, D>>, GenericGraph<Ndst>>` and friends) in enum
//!   variants covering every encoder/graph combination. The types are long because they are
//!   explicit, which is the point of the enum dispatch.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::f32;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{
    MAX_NEIGHBORS_PER_NODE, NeighborData, Neighbors, PlainNeighbors, StreamVByteNeighbors,
};
use crate::hnsw::{
    AcornGammaNeighbors, EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration,
    HNSWSearchConfiguration,
};
use half::f16;
use vectorium::IndexSerializer;
use vectorium::core::flat_index::FlatIndex;
use vectorium::core::index::{Index, IndexStats};
use vectorium::dataset::ConvertFrom;
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

#[cfg(feature = "multivec")]
use vectorium::core::rerank_index::RerankIndex;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::{PlainSparseQuantizer, ScalarSparseSupportedDistance};
use vectorium::readers::{read_npy_f32, read_seismic_format};
#[cfg(feature = "multivec")]
use vectorium::vector::DenseMultiVectorView;
use vectorium::vector::{DenseVectorView, SparseVectorView};
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, FixedU8Q, FixedU16Q, Float, FromF32,
    PackedSparseDataset, PlainDenseDataset, PlainSparseDataset, PlainSparseDatasetGrowable,
    ScalarSparseDataset, ValueType,
};
#[cfg(feature = "multivec")]
use vectorium::{MultiVectorDataset, PlainMultiVecQuantizer};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MetricKind {
    Euclidean,
    DotProduct,
}

/// The HNSW graph storage/ordering strategy for build/load.
///
/// `Standard` and `Permuted` load as the same Rust type (`Graph<PlainNeighbors>`) — the
/// EGB permutation, when applied, is baked into the serialized index data itself
/// (`original_ids`), not reflected in the graph's storage type.
///
/// `Compressed` is the Python-facing name for what the CLI calls `streamvbyte`: the variant
/// names the user-visible behaviour, while the `StreamVByteNeighbors` type it selects names
/// the storage that implements it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum GraphTypeKind {
    Standard,
    Permuted,
    Compressed,
}

fn parse_graph_type(graph_type: &str) -> PyResult<GraphTypeKind> {
    match graph_type.to_lowercase().as_str() {
        "standard" => Ok(GraphTypeKind::Standard),
        "permuted" => Ok(GraphTypeKind::Permuted),
        "compressed" => Ok(GraphTypeKind::Compressed),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown graph_type {other:?}; choose \"standard\" (the default), \"compressed\" \
             (smaller index and faster queries), or \"permuted\" (faster queries only). All \
             three return the same search results."
        ))),
    }
}

/// Parses `graph_type` for a *build* call, rejecting parameter combinations the chosen
/// representation cannot express.
///
/// Checked here rather than in `StreamVByteNeighbors::from`, which only runs once the whole
/// index has already been built — and which would abort the interpreter rather than raise,
/// since the release profile sets `panic = "abort"`.
fn parse_build_graph_type(graph_type: &str, m: usize) -> PyResult<GraphTypeKind> {
    let gt = parse_graph_type(graph_type)?;

    if gt == GraphTypeKind::Compressed && 2 * m > MAX_NEIGHBORS_PER_NODE {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "graph_type=\"compressed\" stores at most {MAX_NEIGHBORS_PER_NODE} neighbors per node, \
             but m={m} gives 2 * m = {} on the ground level; use m <= {} or graph_type=\"standard\"",
            2 * m,
            MAX_NEIGHBORS_PER_NODE / 2
        )));
    }

    Ok(gt)
}

/// Wraps a deserialization failure with the cause a caller is most likely to have hit.
///
/// Index files are plain bincode with no header or type tag, so loading one with the wrong
/// `metric` or `graph_type` fails somewhere inside the decoder with a message that says
/// nothing about the actual mistake.
fn load_index_err<E: std::fmt::Debug>(e: E) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
        "Error loading index: {e:?}. Index files are not self-describing: check that `metric` and \
         `graph_type` match the values used when the index was built, and that the file was \
         written by this version of kannolo."
    ))
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
    k: usize,
    distances: &mut Vec<f32>,
    ids: &mut Vec<i64>,
) {
    let mut found = 0;
    for scored in results.into_iter().take(k) {
        distances.push(scored.distance.distance());
        ids.push(scored.vector as i64);
        found += 1;
    }

    for _ in found..k {
        distances.push(f32::INFINITY);
        ids.push(-1);
    }
}

// Dense plain f32 (internally stored as f16)

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
    acorn_gamma: Option<AcornGammaNeighbors>,
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

// Dense plain f16

// Sparse plain f32 (internally stored as f16)

enum SparsePlainHNSWEnum {
    Euclidean(HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>),
    EuclideanStreamVByte(
        HNSW<
            PlainSparseDataset<u16, f16, SquaredEuclideanDistance>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
    DotProductStreamVByte(
        HNSW<PlainSparseDataset<u16, f16, DotProduct>, GenericGraph<StreamVByteNeighbors>>,
    ),
}

#[pyclass]
pub struct SparsePlainHNSW {
    inner: SparsePlainHNSWEnum,
}

#[pymethods]
impl SparsePlainHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_file: &str,
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
                let dataset: PlainSparseDataset<u16, f16, SquaredEuclideanDistance> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => SparsePlainHNSWEnum::Euclidean(plain),
                    GraphTypeKind::Permuted => {
                        SparsePlainHNSWEnum::Euclidean(plain.permute_and_encode::<PlainNeighbors>())
                    }
                    GraphTypeKind::Compressed => SparsePlainHNSWEnum::EuclideanStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
            MetricKind::DotProduct => {
                let dataset: PlainSparseDataset<u16, f16, DotProduct> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => SparsePlainHNSWEnum::DotProduct(plain),
                    GraphTypeKind::Permuted => SparsePlainHNSWEnum::DotProduct(
                        plain.permute_and_encode::<PlainNeighbors>(),
                    ),
                    GraphTypeKind::Compressed => SparsePlainHNSWEnum::DotProductStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
        };

        Ok(SparsePlainHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
    ) -> PyResult<Self> {
        let components_vec = convert_components_to_u16(components.as_slice()?)?;
        let values_f16: Vec<f16> = values
            .as_slice()?
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
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
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f16, SquaredEuclideanDistance>(
                    components_vec,
                    values_f16,
                    offsets_vec,
                    d,
                )?;
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => SparsePlainHNSWEnum::Euclidean(plain),
                    GraphTypeKind::Permuted => {
                        SparsePlainHNSWEnum::Euclidean(plain.permute_and_encode::<PlainNeighbors>())
                    }
                    GraphTypeKind::Compressed => SparsePlainHNSWEnum::EuclideanStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f16, DotProduct>(
                    components_vec,
                    values_f16,
                    offsets_vec,
                    d,
                )?;
                let plain: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => SparsePlainHNSWEnum::DotProduct(plain),
                    GraphTypeKind::Permuted => SparsePlainHNSWEnum::DotProduct(
                        plain.permute_and_encode::<PlainNeighbors>(),
                    ),
                    GraphTypeKind::Compressed => SparsePlainHNSWEnum::DotProductStreamVByte(
                        plain.permute_and_encode::<StreamVByteNeighbors>(),
                    ),
                }
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
            SparsePlainHNSWEnum::EuclideanStreamVByte(index) => {
                index.save_index(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error saving index: {:?}",
                        e
                    ))
                })
            }
            SparsePlainHNSWEnum::DotProductStreamVByte(index) => {
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
                let index: HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph> = <HNSW<PlainSparseDataset<u16, f16, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                SparsePlainHNSWEnum::Euclidean(index)
            }
            (MetricKind::Euclidean, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    PlainSparseDataset<u16, f16, SquaredEuclideanDistance>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    PlainSparseDataset<u16, f16, SquaredEuclideanDistance>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparsePlainHNSWEnum::EuclideanStreamVByte(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> = <HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                SparsePlainHNSWEnum::DotProduct(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    PlainSparseDataset<u16, f16, DotProduct>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    PlainSparseDataset<u16, f16, DotProduct>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparsePlainHNSWEnum::DotProductStreamVByte(index)
            }
        };
        Ok(SparsePlainHNSW { inner })
    }

    /// Search for approximate nearest neighbors for a single sparse query.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices for the query.
    /// * `query_values` – 1-D float32 array of component values for the query.
    /// * `k` – Number of nearest neighbors to return.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut ids = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        let query_view = SparseVectorView::new(&comp_vec, values_slice);

        match &self.inner {
            SparsePlainHNSWEnum::Euclidean(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparsePlainHNSWEnum::DotProduct(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparsePlainHNSWEnum::EuclideanStreamVByte(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparsePlainHNSWEnum::DotProductStreamVByte(index) => {
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

    /// Search a batch of sparse queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of component values (concatenated for batch).
    /// * `offsets` – 1-D int64 array defining query boundaries, e.g. `[0, n1, n1+n2, ...]`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
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

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let query_view =
                SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                SparsePlainHNSWEnum::Euclidean(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparsePlainHNSWEnum::DotProduct(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparsePlainHNSWEnum::EuclideanStreamVByte(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparsePlainHNSWEnum::DotProductStreamVByte(index) => {
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
}

// Sparse DotVByte (dotproduct only)

enum SparseDotVByteHNSWEnum {
    Plain(HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph>),
    StreamVByte(
        HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, GenericGraph<StreamVByteNeighbors>>,
    ),
}

/// Applies `gt` to a plain-built sparse HNSW (permuting via EGB when requested) and
/// converts it into the DotVByte-packed dataset representation.
///
/// **The conversion must run before `permute_and_encode`**, for the reason spelled out on
/// `build_permuted_and_save` in `src/bin/hnsw_build.rs`: the DotVByte component mapping is fitted
/// on a *prefix sample* of the dataset, so permuting first would train it on an EGB-clustered
/// prefix and yield a worse encoding than the `standard` path gets.
fn build_sparse_dotvbyte_inner(
    plain_hnsw: HNSW<PlainSparseDataset<u16, f32, DotProduct>, Graph>,
    gt: GraphTypeKind,
) -> SparseDotVByteHNSWEnum {
    match gt {
        GraphTypeKind::Standard => {
            let packed: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
                plain_hnsw.convert_dataset_into(());
            SparseDotVByteHNSWEnum::Plain(packed)
        }
        GraphTypeKind::Permuted => {
            let packed: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
                plain_hnsw.convert_dataset_into(());
            SparseDotVByteHNSWEnum::Plain(packed.permute_and_encode::<PlainNeighbors>())
        }
        GraphTypeKind::Compressed => {
            let packed: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> =
                plain_hnsw.convert_dataset_into(());
            SparseDotVByteHNSWEnum::StreamVByte(packed.permute_and_encode::<StreamVByteNeighbors>())
        }
    }
}

#[pyclass]
pub struct SparseDotVByteHNSW {
    inner: SparseDotVByteHNSWEnum,
}

#[pymethods]
impl SparseDotVByteHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        m: usize,
        ef_construction: usize,
        graph_type: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let dataset: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(data_file)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error reading dataset: {:?}",
                    e
                ))
            })?;

        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner = build_sparse_dotvbyte_inner(plain_hnsw, gt);

        Ok(SparseDotVByteHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, graph_type="standard".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        graph_type: String,
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
        let gt = parse_build_graph_type(&graph_type, m)?;

        let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
            components_vec,
            values_vec,
            offsets_vec,
            d,
        )?;
        let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
        let inner = build_sparse_dotvbyte_inner(plain_hnsw, gt);

        Ok(SparseDotVByteHNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            SparseDotVByteHNSWEnum::Plain(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
            SparseDotVByteHNSWEnum::StreamVByte(index) => index.save_index(path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
            }),
        }
    }

    /// Loads a previously saved index. `graph_type` must match the value used at build
    /// time (`standard`/`permuted` both load as the same on-disk representation).
    #[staticmethod]
    #[pyo3(signature = (path, graph_type="standard".to_string()))]
    pub fn load(path: &str, graph_type: String) -> PyResult<Self> {
        let inner = match parse_graph_type(&graph_type)? {
            GraphTypeKind::Standard | GraphTypeKind::Permuted => {
                let index: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, Graph> = <HNSW<
                    PackedSparseDataset<DotVByteFixedU8Encoder>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                SparseDotVByteHNSWEnum::Plain(index)
            }
            GraphTypeKind::Compressed => {
                let index: HNSW<
                    PackedSparseDataset<DotVByteFixedU8Encoder>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    PackedSparseDataset<DotVByteFixedU8Encoder>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparseDotVByteHNSWEnum::StreamVByte(index)
            }
        };

        Ok(SparseDotVByteHNSW { inner })
    }

    /// Search for approximate nearest neighbors for a single compressed sparse query.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices for the query.
    /// * `query_values` – 1-D float32 array of component values for the query.
    /// * `k` – Number of nearest neighbors to return.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut ids = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        let query_view = SparseVectorView::new(&comp_vec, values_slice);
        match &self.inner {
            SparseDotVByteHNSWEnum::Plain(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseDotVByteHNSWEnum::StreamVByte(index) => {
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

    /// Search a batch of compressed sparse queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices (concatenated for batch).
    /// * `query_values` – 1-D float32 array of component values (concatenated for batch).
    /// * `offsets` – 1-D int64 array defining query boundaries, e.g. `[0, n1, n1+n2, ...]`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `ef_search` – Candidate list size (higher = better recall, slower). Default: 100.
    /// * `early_exit_threshold` – Early termination threshold. Default: None.
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
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

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let query_view =
                SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                SparseDotVByteHNSWEnum::Plain(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseDotVByteHNSWEnum::StreamVByte(index) => {
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
}

// Sparse scalar fixedu8/fixedu16

enum SparseFixedU8HNSWEnum {
    Euclidean(HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph>),
    EuclideanStreamVByte(
        HNSW<
            ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
    DotProductStreamVByte(
        HNSW<
            ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
}

#[pyclass]
pub struct SparseFixedU8HNSW {
    inner: SparseFixedU8HNSWEnum,
}

#[pymethods]
impl SparseFixedU8HNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_file: &str,
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
                let dataset: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU8HNSWEnum::Euclidean(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::Euclidean(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::EuclideanStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
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
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU8HNSWEnum::DotProduct(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::DotProduct(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::DotProductStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
            }
        };

        Ok(SparseFixedU8HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
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
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f32, SquaredEuclideanDistance>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU8HNSWEnum::Euclidean(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::Euclidean(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::EuclideanStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU8HNSWEnum::DotProduct(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::DotProduct(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU8HNSWEnum::DotProductStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
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
            SparseFixedU8HNSWEnum::EuclideanStreamVByte(index) => {
                index.save_index(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error saving index: {:?}",
                        e
                    ))
                })
            }
            SparseFixedU8HNSWEnum::DotProductStreamVByte(index) => {
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
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                SparseFixedU8HNSWEnum::Euclidean(index)
            }
            (MetricKind::Euclidean, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    ScalarSparseDataset<u16, f32, FixedU8Q, SquaredEuclideanDistance>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparseFixedU8HNSWEnum::EuclideanStreamVByte(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                SparseFixedU8HNSWEnum::DotProduct(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    ScalarSparseDataset<u16, f32, FixedU8Q, DotProduct>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparseFixedU8HNSWEnum::DotProductStreamVByte(index)
            }
        };
        Ok(SparseFixedU8HNSW { inner })
    }

    #[pyo3(signature = (query_components, query_values, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut ids = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        let query_view = SparseVectorView::new(&comp_vec, values_slice);

        match &self.inner {
            SparseFixedU8HNSWEnum::Euclidean(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseFixedU8HNSWEnum::DotProduct(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseFixedU8HNSWEnum::EuclideanStreamVByte(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseFixedU8HNSWEnum::DotProductStreamVByte(index) => {
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

    /// Search a batch of sparse queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
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

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let query_view =
                SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                SparseFixedU8HNSWEnum::Euclidean(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseFixedU8HNSWEnum::DotProduct(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseFixedU8HNSWEnum::EuclideanStreamVByte(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseFixedU8HNSWEnum::DotProductStreamVByte(index) => {
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
}

enum SparseFixedU16HNSWEnum {
    Euclidean(HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>, Graph>),
    DotProduct(HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph>),
    EuclideanStreamVByte(
        HNSW<
            ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
    DotProductStreamVByte(
        HNSW<
            ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
            GenericGraph<StreamVByteNeighbors>,
        >,
    ),
}

#[pyclass]
pub struct SparseFixedU16HNSW {
    inner: SparseFixedU16HNSWEnum,
}

#[pymethods]
impl SparseFixedU16HNSW {
    #[staticmethod]
    #[pyo3(signature = (data_file, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_file: &str,
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
                let dataset: PlainSparseDataset<u16, f32, SquaredEuclideanDistance> =
                    read_seismic_format(data_file).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                            "Error reading dataset: {:?}",
                            e
                        ))
                    })?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU16HNSWEnum::Euclidean(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::Euclidean(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::EuclideanStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
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
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU16HNSWEnum::DotProduct(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::DotProduct(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::DotProductStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
            }
        };

        Ok(SparseFixedU16HNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
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
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset = build_sparse_dataset_from_parts::<f32, SquaredEuclideanDistance>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU16HNSWEnum::Euclidean(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::Euclidean(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::EuclideanStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
            }
            MetricKind::DotProduct => {
                let dataset = build_sparse_dataset_from_parts::<f32, DotProduct>(
                    components_vec,
                    values_vec,
                    offsets_vec,
                    d,
                )?;
                let plain_hnsw: HNSW<_, Graph> = HNSW::build_index(dataset, &config);
                match gt {
                    GraphTypeKind::Standard => {
                        SparseFixedU16HNSWEnum::DotProduct(plain_hnsw.convert_dataset_into(()))
                    }
                    GraphTypeKind::Permuted => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::DotProduct(
                            converted.permute_and_encode::<PlainNeighbors>(),
                        )
                    }
                    GraphTypeKind::Compressed => {
                        let converted: HNSW<
                            ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
                            Graph,
                        > = plain_hnsw.convert_dataset_into(());
                        SparseFixedU16HNSWEnum::DotProductStreamVByte(
                            converted.permute_and_encode::<StreamVByteNeighbors>(),
                        )
                    }
                }
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
            SparseFixedU16HNSWEnum::EuclideanStreamVByte(index) => {
                index.save_index(path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error saving index: {:?}",
                        e
                    ))
                })
            }
            SparseFixedU16HNSWEnum::DotProductStreamVByte(index) => {
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
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                SparseFixedU16HNSWEnum::Euclidean(index)
            }
            (MetricKind::Euclidean, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    ScalarSparseDataset<u16, f32, FixedU16Q, SquaredEuclideanDistance>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparseFixedU16HNSWEnum::EuclideanStreamVByte(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> = <HNSW<ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>, Graph> as IndexSerializer>::load_index(path)
                    .map_err(load_index_err)?;
                SparseFixedU16HNSWEnum::DotProduct(index)
            }
            (MetricKind::DotProduct, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    ScalarSparseDataset<u16, f32, FixedU16Q, DotProduct>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                SparseFixedU16HNSWEnum::DotProductStreamVByte(index)
            }
        };
        Ok(SparseFixedU16HNSW { inner })
    }

    #[pyo3(signature = (query_components, query_values, k, ef_search=100, early_exit_threshold=None))]
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut ids = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        let query_view = SparseVectorView::new(&comp_vec, values_slice);

        match &self.inner {
            SparseFixedU16HNSWEnum::Euclidean(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseFixedU16HNSWEnum::DotProduct(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseFixedU16HNSWEnum::EuclideanStreamVByte(index) => {
                push_results(
                    index.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            SparseFixedU16HNSWEnum::DotProductStreamVByte(index) => {
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

    /// Search a batch of sparse queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    #[pyo3(signature = (query_components, query_values, offsets, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
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

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let query_view =
                SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                SparseFixedU16HNSWEnum::Euclidean(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseFixedU16HNSWEnum::DotProduct(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseFixedU16HNSWEnum::EuclideanStreamVByte(index) => {
                    push_results(
                        index.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                SparseFixedU16HNSWEnum::DotProductStreamVByte(index) => {
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
    PQ8StreamVByte(HNSW<DenseDataset<ProductQuantizer<8, D>>, GenericGraph<StreamVByteNeighbors>>),
    PQ16StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<16, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ32StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<32, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ48StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<48, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ64StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<64, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ96StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<96, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ128StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<128, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ192StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<192, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ256StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<256, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ384StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<384, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
}

/// Builds a plain (uncompressed, original node order) PQ-quantized dense HNSW.
fn build_pq_l2<const M: usize>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, Graph>
where
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Encoder = ProductQuantizer<M, SquaredEuclideanDistance>>,
    for<'a> DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<&'a PlainDenseDataset<f32, SquaredEuclideanDistance>, Config = ()>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    plain_index.convert_dataset_into_ref(())
}

/// Builds an EGB-permuted PQ-quantized dense HNSW, recompressed into `Ndst`
/// (`PlainNeighbors` for `permuted`, `StreamVByteNeighbors` for `streamvbyte`).
fn build_pq_l2_compressed<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, GenericGraph<Ndst>>
where
    Ndst: Neighbors + From<NeighborData>,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Encoder = ProductQuantizer<M, SquaredEuclideanDistance>>,
    for<'a> DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<&'a PlainDenseDataset<f32, SquaredEuclideanDistance>, Config = ()>,
    // `permute_and_encode` hands the permuted dataset back as `Owned`; restated here because the
    // compiler cannot normalize `Owned = Self` through the generic parameter (see
    // `build_permuted_and_save` in `src/bin/hnsw_build.rs`).
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Owned = DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    // Quantize before permuting, never after — see `build_permuted_and_save`.
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    let converted: HNSW<DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, Graph> =
        plain_index.convert_dataset_into_ref(());
    converted.permute_and_encode::<Ndst>()
}

/// Builds a plain (uncompressed, original node order) PQ-quantized dense HNSW.
fn build_pq_ip<const M: usize>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, DotProduct>>, Graph>
where
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    plain_index.convert_dataset_into_ref(())
}

/// Builds an EGB-permuted PQ-quantized dense HNSW, recompressed into `Ndst`
/// (`PlainNeighbors` for `permuted`, `StreamVByteNeighbors` for `streamvbyte`).
fn build_pq_ip_compressed<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, DotProduct>>, GenericGraph<Ndst>>
where
    Ndst: Neighbors + From<NeighborData>,
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    // `permute_and_encode` hands the permuted dataset back as `Owned`; restated here because the
    // compiler cannot normalize `Owned = Self` through the generic parameter (see
    // `build_permuted_and_save` in `src/bin/hnsw_build.rs`).
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Owned = DenseDataset<ProductQuantizer<M, DotProduct>>>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    // Quantize before permuting, never after — see `build_permuted_and_save`.
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    let converted: HNSW<DenseDataset<ProductQuantizer<M, DotProduct>>, Graph> =
        plain_index.convert_dataset_into_ref(());
    converted.permute_and_encode::<Ndst>()
}

impl DensePQHNSWGeneric<DotProduct> {
    fn build_from_dataset(
        dataset: PlainDenseDataset<f32, DotProduct>,
        config: &HNSWBuildConfiguration,
        m_pq: usize,
        gt: GraphTypeKind,
    ) -> PyResult<Self> {
        Ok(match m_pq {
            8 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ8(build_pq_ip::<8>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ8(build_pq_ip_compressed::<8, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ8StreamVByte(build_pq_ip_compressed::<
                        8,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            16 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ16(build_pq_ip::<16>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ16(build_pq_ip_compressed::<16, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ16StreamVByte(build_pq_ip_compressed::<
                        16,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            32 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ32(build_pq_ip::<32>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ32(build_pq_ip_compressed::<32, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ32StreamVByte(build_pq_ip_compressed::<
                        32,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            48 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ48(build_pq_ip::<48>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ48(build_pq_ip_compressed::<48, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ48StreamVByte(build_pq_ip_compressed::<
                        48,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            64 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ64(build_pq_ip::<64>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ64(build_pq_ip_compressed::<64, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ64StreamVByte(build_pq_ip_compressed::<
                        64,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            96 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ96(build_pq_ip::<96>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ96(build_pq_ip_compressed::<96, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ96StreamVByte(build_pq_ip_compressed::<
                        96,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            128 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ128(build_pq_ip::<128>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ128(build_pq_ip_compressed::<128, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ128StreamVByte(build_pq_ip_compressed::<
                        128,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            192 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ192(build_pq_ip::<192>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ192(build_pq_ip_compressed::<192, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ192StreamVByte(build_pq_ip_compressed::<
                        192,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            256 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ256(build_pq_ip::<256>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ256(build_pq_ip_compressed::<256, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ256StreamVByte(build_pq_ip_compressed::<
                        256,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            384 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ384(build_pq_ip::<384>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ384(build_pq_ip_compressed::<384, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ384StreamVByte(build_pq_ip_compressed::<
                        384,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ));
            }
        })
    }
}

impl DensePQHNSWGeneric<SquaredEuclideanDistance> {
    fn build_from_dataset(
        dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
        config: &HNSWBuildConfiguration,
        m_pq: usize,
        gt: GraphTypeKind,
    ) -> PyResult<Self> {
        Ok(match m_pq {
            8 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ8(build_pq_l2::<8>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ8(build_pq_l2_compressed::<8, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ8StreamVByte(build_pq_l2_compressed::<
                        8,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            16 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ16(build_pq_l2::<16>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ16(build_pq_l2_compressed::<16, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ16StreamVByte(build_pq_l2_compressed::<
                        16,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            32 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ32(build_pq_l2::<32>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ32(build_pq_l2_compressed::<32, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ32StreamVByte(build_pq_l2_compressed::<
                        32,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            48 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ48(build_pq_l2::<48>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ48(build_pq_l2_compressed::<48, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ48StreamVByte(build_pq_l2_compressed::<
                        48,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            64 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ64(build_pq_l2::<64>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ64(build_pq_l2_compressed::<64, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ64StreamVByte(build_pq_l2_compressed::<
                        64,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            96 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ96(build_pq_l2::<96>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ96(build_pq_l2_compressed::<96, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ96StreamVByte(build_pq_l2_compressed::<
                        96,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            128 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ128(build_pq_l2::<128>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ128(build_pq_l2_compressed::<128, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ128StreamVByte(build_pq_l2_compressed::<
                        128,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            192 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ192(build_pq_l2::<192>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ192(build_pq_l2_compressed::<192, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ192StreamVByte(build_pq_l2_compressed::<
                        192,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            256 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ256(build_pq_l2::<256>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ256(build_pq_l2_compressed::<256, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ256StreamVByte(build_pq_l2_compressed::<
                        256,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            384 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ384(build_pq_l2::<384>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ384(build_pq_l2_compressed::<384, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ384StreamVByte(build_pq_l2_compressed::<
                        384,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ));
            }
        })
    }
}

impl<D> DensePQHNSWGeneric<D>
where
    D: ProductQuantizerDistance + Distance + ScalarDenseSupportedDistance,
{
    fn load(path: &str, m_pq: usize, gt: GraphTypeKind) -> PyResult<Self> {
        let inner = match (m_pq, gt) {
            (8, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<8, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ8(index)
            }
            (8, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<8, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ8StreamVByte(index)
            }
            (16, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<16, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ16(index)
            }
            (16, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<16, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ16StreamVByte(index)
            }
            (32, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<32, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ32(index)
            }
            (32, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<32, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ32StreamVByte(index)
            }
            (48, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<48, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ48(index)
            }
            (48, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<48, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ48StreamVByte(index)
            }
            (64, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<64, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ64(index)
            }
            (64, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<64, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ64StreamVByte(index)
            }
            (96, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<96, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ96(index)
            }
            (96, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<96, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ96StreamVByte(index)
            }
            (128, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<128, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ128(index)
            }
            (128, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ128StreamVByte(index)
            }
            (192, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<192, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ192(index)
            }
            (192, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ192StreamVByte(index)
            }
            (256, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<256, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ256(index)
            }
            (256, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ256StreamVByte(index)
            }
            (384, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<384, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ384(index)
            }
            (384, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ384StreamVByte(index)
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
            DensePQHNSWGeneric::PQ8StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ16StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ32StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ48StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ64StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ96StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ128StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ192StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ256StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ384StreamVByte(index) => index.save_index(path),
        };

        result.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    fn dim(&self) -> usize {
        match self {
            DensePQHNSWGeneric::PQ8(index) => index.dim(),
            DensePQHNSWGeneric::PQ16(index) => index.dim(),
            DensePQHNSWGeneric::PQ32(index) => index.dim(),
            DensePQHNSWGeneric::PQ48(index) => index.dim(),
            DensePQHNSWGeneric::PQ64(index) => index.dim(),
            DensePQHNSWGeneric::PQ96(index) => index.dim(),
            DensePQHNSWGeneric::PQ128(index) => index.dim(),
            DensePQHNSWGeneric::PQ192(index) => index.dim(),
            DensePQHNSWGeneric::PQ256(index) => index.dim(),
            DensePQHNSWGeneric::PQ384(index) => index.dim(),
            DensePQHNSWGeneric::PQ8StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ16StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ32StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ48StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ64StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ96StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ128StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ192StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ256StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ384StreamVByte(index) => index.dim(),
        }
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
            DensePQHNSWGeneric::PQ8StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ16StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ32StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ48StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ64StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ96StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ128StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ192StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ256StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ384StreamVByte(index) => index.search(query, k, search_config),
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
    #[pyo3(signature = (data_path, m_pq, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_path: &str,
        m_pq: usize,
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
                let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
                    read_npy_dataset::<SquaredEuclideanDistance>(data_path)?;
                DensePQHNSWEnum::Euclidean(
                    DensePQHNSWGeneric::<SquaredEuclideanDistance>::build_from_dataset(
                        dataset, &config, m_pq, gt,
                    )?,
                )
            }
            MetricKind::DotProduct => {
                let dataset: PlainDenseDataset<f32, DotProduct> =
                    read_npy_dataset::<DotProduct>(data_path)?;
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::<DotProduct>::build_from_dataset(
                    dataset, &config, m_pq, gt,
                )?)
            }
        };

        Ok(DensePQHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m_pq, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m_pq: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
    ) -> PyResult<Self> {
        let data_vec = data_vec.as_slice()?.to_vec();
        let n_vecs = data_vec.len() / dim;
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
                let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePQHNSWEnum::Euclidean(
                    DensePQHNSWGeneric::<SquaredEuclideanDistance>::build_from_dataset(
                        dataset, &config, m_pq, gt,
                    )?,
                )
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(dim);
                let dataset: PlainDenseDataset<f32, DotProduct> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::<DotProduct>::build_from_dataset(
                    dataset, &config, m_pq, gt,
                )?)
            }
        };

        Ok(DensePQHNSW { inner })
    }

    /// Loads a previously saved index. `graph_type` must match the value used at build
    /// time (`standard`/`permuted` both load as the same on-disk representation).
    #[staticmethod]
    #[pyo3(signature = (path, m_pq, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn load(path: &str, m_pq: usize, metric: String, graph_type: String) -> PyResult<Self> {
        let gt = parse_graph_type(&graph_type)?;
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                DensePQHNSWEnum::Euclidean(DensePQHNSWGeneric::load(path, m_pq, gt)?)
            }
            MetricKind::DotProduct => {
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::load(path, m_pq, gt)?)
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
        let dim = match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => inner.dim(),
            DensePQHNSWEnum::DotProduct(inner) => inner.dim(),
        };
        let query_slice = query.as_slice()?;
        if query_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "query dimension {} does not match index dimension {}",
                query_slice.len(),
                dim
            )));
        }
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
                push_results(
                    inner.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            DensePQHNSWEnum::DotProduct(inner) => {
                push_results(
                    inner.search(query_view, k, &search_config),
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
            DensePQHNSWEnum::Euclidean(inner) => inner.dim(),
            DensePQHNSWEnum::DotProduct(inner) => inner.dim(),
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
                DensePQHNSWEnum::Euclidean(inner) => {
                    push_results(
                        inner.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                DensePQHNSWEnum::DotProduct(inner) => {
                    push_results(
                        inner.search(query_view, k, &search_config),
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
}

// Multivector Reranking

#[cfg(feature = "multivec")]
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
        let data_f16: Vec<f16> = data_vec
            .as_slice()?
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
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

    /// Exhaustive search over all vectors for exact nearest neighbors (single query).
    ///
    /// # Arguments
    /// * `query` – 1-D float32 numpy array of length `dim`.
    /// * `k` – Number of nearest neighbors to return.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query, k))]
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let dim = match &self.inner {
            DenseFlatIndexEnum::Euclidean(dataset) => dataset.input_dim(),
            DenseFlatIndexEnum::DotProduct(dataset) => dataset.input_dim(),
        };
        let query_slice = query.as_slice()?;
        if query_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "query dimension {} does not match index dimension {}",
                query_slice.len(),
                dim
            )));
        }
        let query_view = DenseVectorView::new(query_slice);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);

        match &self.inner {
            DenseFlatIndexEnum::Euclidean(dataset) => {
                push_results(
                    FlatIndex::from(dataset).search(query_view, k, &()),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            DenseFlatIndexEnum::DotProduct(dataset) => {
                push_results(
                    FlatIndex::from(dataset).search(query_view, k, &()),
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

    /// Exhaustive batch search, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    ///
    /// # Arguments
    /// * `queries` – 1-D float32 numpy array of length `num_queries × dim`.
    /// * `k` – Number of nearest neighbors to return per query.
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length `num_queries × k`.
    #[pyo3(signature = (queries, k, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray1<f32>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let queries_slice = queries.as_slice()?;
        let dim = match &self.inner {
            DenseFlatIndexEnum::Euclidean(dataset) => dataset.input_dim(),
            DenseFlatIndexEnum::DotProduct(dataset) => dataset.input_dim(),
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
                DenseFlatIndexEnum::Euclidean(dataset) => {
                    push_results(
                        FlatIndex::from(dataset).search(query_view, k, &()),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                DenseFlatIndexEnum::DotProduct(dataset) => {
                    push_results(
                        FlatIndex::from(dataset).search(query_view, k, &()),
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
        let values_vec: Vec<f16> = values
            .as_slice()?
            .iter()
            .map(|&x| f16::from_f32(x))
            .collect();
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

    /// Exhaustive search over all vectors for exact nearest neighbors (single query).
    ///
    /// # Arguments
    /// * `query_components` – 1-D int32 array of component indices for the query.
    /// * `query_values` – 1-D float32 array of component values for the query.
    /// * `k` – Number of nearest neighbors to return.
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    pub fn search(
        &self,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        let query_view = SparseVectorView::new(&comp_vec, values_slice);

        match &self.inner {
            SparseFlatIndexEnum::DotProduct(dataset) => {
                push_results(
                    FlatIndex::from(dataset).search(query_view, k, &()),
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

    /// Exhaustive batch search, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    #[pyo3(signature = (query_components, query_values, offsets, k, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        query_components: PyReadonlyArray1<i32>,
        query_values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i64>,
        k: usize,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let comp_vec = convert_components_to_u16(query_components.as_slice()?)?;
        let values_slice = query_values.as_slice()?;
        let offsets_slice = offsets.as_slice()?;
        let num_queries = offsets_slice.len() - 1;

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let start = offsets_slice[i] as usize;
            let end = offsets_slice[i + 1] as usize;
            let query_view =
                SparseVectorView::new(&comp_vec[start..end], &values_slice[start..end]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                SparseFlatIndexEnum::DotProduct(dataset) => {
                    push_results(
                        FlatIndex::from(dataset).search(query_view, k, &()),
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
}

#[cfg(feature = "multivec")]
#[pyclass]
pub struct SparseMultivecRerankIndex {
    inner: RerankIndex<
        HNSW<PlainSparseDataset<u16, f16, DotProduct>, Graph>,
        MultiVectorDataset<PlainMultiVecQuantizer<f32>>,
    >,
}

#[cfg(feature = "multivec")]
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

    /// Search with reranking using plain multivector encoding (single query).
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
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, multivec_query, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None))]
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
        let _ = n_tokens; // token count is implicit from slice length / token_dim

        let results = self.inner.search(
            sparse_query,
            multivec_query_view,
            k_candidates,
            k,
            &search_config,
            &(),
            alpha,
            beta,
        );

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        push_results(results, k, &mut distances, &mut ids);

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search with reranking using plain multivector encoding, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
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
    /// * `num_threads` – Threading model (see above). Default: 0 (all cores).
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of total length ≤ `num_queries × k`.
    #[pyo3(signature = (query_components, query_values, sparse_offsets, multivec_queries, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, num_threads=0))]
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
            let results = self.inner.search(
                sparse_query,
                multivec_query_view,
                k_candidates,
                k,
                &search_config,
                &(),
                alpha,
                beta,
            );
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

// Helper to load two-level PQ multivector dataset
#[cfg(feature = "multivec")]
fn load_multivec_dataset_pq_8(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<8>(data_folder)
}

#[cfg(feature = "multivec")]
fn load_multivec_dataset_pq_16(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<16>(data_folder)
}

#[cfg(feature = "multivec")]
fn load_multivec_dataset_pq_32(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<32>(data_folder)
}

#[cfg(feature = "multivec")]
fn load_multivec_dataset_pq_64(
    data_folder: &str,
) -> PyResult<MultiVectorDataset<PlainMultiVecQuantizer<f32>>> {
    load_multivec_dataset_pq_generic::<64>(data_folder)
}

#[cfg(feature = "multivec")]
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
#[cfg(feature = "multivec")]
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

#[cfg(feature = "multivec")]
#[pyclass]
pub struct SparseMultivecTwoLevelsPQRerankIndex {
    inner: SparseMultivecTwoLevelsPQRerankIndexEnum,
}

#[cfg(feature = "multivec")]
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
    ///
    /// # Returns
    /// `(distances, ids)` – two 1-D numpy arrays of length ≤ `k`.
    #[pyo3(signature = (query_components, query_values, multivec_query, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None))]
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
    #[pyo3(signature = (query_components, query_values, sparse_offsets, multivec_queries, n_tokens, token_dim, k_candidates=100, k=10, ef_search=100, alpha=None, beta=None, early_exit_threshold=None, num_threads=0))]
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
