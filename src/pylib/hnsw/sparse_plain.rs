//! `SparsePlainHNSW`: HNSW over sparse f32 input, stored internally as f16.

use std::f32;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};
use half::f16;
use vectorium::IndexSerializer;
use vectorium::core::index::Index;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::PlainSparseDataset;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::readers::read_seismic_format;
use vectorium::vector::SparseVectorView;

use super::{GraphTypeKind, load_index_err, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{
    MetricKind, build_sparse_dataset_from_parts, convert_components_to_u16, parse_metric,
    push_results,
};

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
