//! `SparseFixedU16HNSW`: HNSW over sparse data with fixed-u16 scalar quantized values.

use std::f32;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::readers::read_seismic_format;
use vectorium::vector::SparseVectorView;
use vectorium::{FixedU16Q, PlainSparseDataset, ScalarSparseDataset};

use super::{GraphTypeKind, load_index_err, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{
    MetricKind, build_sparse_dataset_from_parts, convert_components_to_u16, parse_metric,
    push_results,
};

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
