//! `SparseFlatIndex`: exhaustive sparse search, for ground truth computation.

use std::f32;

use half::f16;
use vectorium::core::flat_index::FlatIndex;
use vectorium::core::index::Index;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::PlainSparseDataset;
use vectorium::distances::DotProduct;
use vectorium::readers::read_seismic_format;
use vectorium::vector::SparseVectorView;

use crate::pylib::common::{
    build_sparse_dataset_from_parts, convert_components_to_u16, push_results,
};

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
