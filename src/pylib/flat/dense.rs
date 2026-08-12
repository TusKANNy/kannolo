//! `DenseFlatIndex`: exhaustive dense search, for ground truth computation.

use std::f32;

use half::f16;
use vectorium::core::flat_index::FlatIndex;
use vectorium::core::index::Index;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
use vectorium::vector::DenseVectorView;
use vectorium::{Dataset, DenseDataset};

use crate::pylib::common::{MetricKind, parse_metric, push_results, read_npy_dataset_f16};

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
