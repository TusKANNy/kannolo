use std::f32;

use crate::datasets::{dense_dataset::DenseDataset, sparse_dataset::SparseDataset};
use crate::hnsw::graph_index::GraphIndex;
use crate::hnsw_utils::config_hnsw::ConfigHnsw;
use crate::index_serializer::IndexSerializer;
use crate::plain_quantizer::PlainQuantizer;
use crate::pq::ProductQuantizer;
use crate::sparse_plain_quantizer::SparsePlainQuantizer;
use crate::{read_numpy_f32_flatten_2d, Dataset, DenseVector1D, DistanceType, Vector1D};
use half::f16;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};

/// A Python-exposed dense index built with a plain quantizer (no quantization).
#[pyclass]
pub struct DensePlainHNSW {
    index: GraphIndex<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>,
}

#[pymethods]
impl DensePlainHNSW {
    /// Build a dense plain index from a 2D NumPy array stored in a npy file.
    /// - `data_path`: path to a 2D f32 array of shape (num_docs, dim) stored in a npy file
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during index construction
    /// - `metric`: either "l2" for Euclidean or "ip" for inner product
    #[staticmethod]
    #[pyo3(signature = (data_path, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_file(
        data_path: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let (data_vec, dim) = read_numpy_f32_flatten_2d(data_path.to_string());

        // Determine the distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Create a plain quantizer.
        let quantizer = PlainQuantizer::<f32>::new(dim, distance);

        // Build the dense dataset.
        let dataset: DenseDataset<PlainQuantizer<f32>> =
            DenseDataset::from_vec(data_vec, dim, quantizer.clone());

        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        let start = std::time::Instant::now();
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
        let elapsed = start.elapsed();
        println!("Time to build index: {:?}", elapsed);

        Ok(DensePlainHNSW { index })
    }

    /// Build a dense plain index from a 1D NumPy array given the dimensionality.
    /// - `data_vec`: a 1D f32 array of len num_docs * dim
    /// - `dim`: dimensionality of the data
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during index construction
    /// - `metric`: either "l2" for Euclidean or "ip" for inner product
    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let data_vec = data_vec.as_slice()?.to_vec();

        // Determine the distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Create a plain quantizer.
        let quantizer = PlainQuantizer::<f32>::new(dim, distance);

        // Build the dense dataset.
        let dataset = DenseDataset::from_vec(data_vec, dim, quantizer.clone());

        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        let start = std::time::Instant::now();
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
        let elapsed = start.elapsed();
        println!("Time to build index: {:?}", elapsed);

        Ok(DensePlainHNSW { index })
    }

    /// Save the index to a file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        IndexSerializer::save_index(path, &self.index).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    /// Load a dense plain index from a file.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let index = IndexSerializer::load_index::<
            GraphIndex<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>,
        >(path);
        Ok(DensePlainHNSW { index })
    }

    /// Search using a single query vector.
    ///
    /// Parameters:
    /// - `query`: a 1D NumPy array (f32) representing the query vector.
    /// - `k`: number of nearest neighbors to return.
    /// - `ef_search`: parameter controlling the candidate pool size during search.
    ///
    /// Returns a tuple (distances, ids) where:
    /// - `distances` is a 1D NumPy array (f32) of scores.
    /// - `ids` is a 1D NumPy array (i64) of document IDs.
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Convert the input query into a Rust slice.
        let query_slice = query.as_slice()?;
        // Determine the dimensionality of the query.
        let _dim = query_slice.len();

        // Set up the HNSW search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Wrap the query slice in a DenseVector1D.
        let query_darray = DenseVector1D::new(query_slice);

        // Perform the search.
        let search_results = self
            .index
            .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                query_darray,
                k,
                &search_config,
            );

        // Collect the results into vectors.
        let mut distances_vec: Vec<f32> = Vec::with_capacity(k);
        let mut ids_vec: Vec<i64> = Vec::with_capacity(k);
        for (score, doc_id) in search_results {
            distances_vec.push(score);
            ids_vec.push(doc_id as i64);
        }

        // Create NumPy arrays within a Python GIL context.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search using a 2D NumPy array of queries.
    ///
    /// Each row in `queries` is a query. For each query, the search returns `k` results.
    /// The method returns a tuple of two 1D NumPy arrays of shape (n_queries * k):
    /// (ids, distances). In Python you can do:
    ///
    ///     ids, distances = index.search(queries, k, ef_search)
    ///
    /// Here, `ids` is an array of doc IDs (as int64) and `distances` is an array of scores (as float32).
    pub fn search_batch(
        &self,
        queries_path: &str,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let (queries_slice, dim) = read_numpy_f32_flatten_2d(queries_path.to_string());
        let num_queries = queries_slice.len() / dim;

        // Set up the search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Pre-allocate vectors to store the results.
        // Each query returns k results; total elements = num_queries * k.
        let mut ids_vec: Vec<i64> = Vec::with_capacity(num_queries * k);
        let mut distances_vec: Vec<f32> = Vec::with_capacity(num_queries * k);

        // Iterate over each query using chunks_exact (each chunk is one query).
        for query in queries_slice.chunks_exact(dim) {
            let query_darray = DenseVector1D::new(query);
            // Search returns a vector of (score, doc_id) pairs.
            let search_results = self
                .index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                );
            for (score, doc_id) in search_results {
                distances_vec.push(score);
                ids_vec.push(doc_id as i64);
            }
        }

        // Create the NumPy arrays within a Python GIL context.
        Python::with_gil(|py| {
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

/// A Python-exposed dense index built with a plain quantizer (no quantization).
#[pyclass]
pub struct DensePlainHNSWf16 {
    index: GraphIndex<DenseDataset<PlainQuantizer<f16>>, PlainQuantizer<f16>>,
}

#[pymethods]
impl DensePlainHNSWf16 {
    /// Build a dense plain index from a 2D NumPy array saved in a npy file.
    /// - `data_path`: path to a 2D f32 array of shape (num_docs, dim) saved in a npy file
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during index construction
    /// - `metric`: either "l2" for Euclidean or "ip" for inner product
    #[staticmethod]
    #[pyo3(signature = (data_path, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_file(
        data_path: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let (data_vec, dim) = read_numpy_f32_flatten_2d(data_path.to_string());
        // Convert the f32 data to f16.
        let data_vec: Vec<f16> = data_vec.iter().map(|&v| f16::from_f32(v)).collect();

        // Determine the distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Create a plain quantizer.
        let quantizer = PlainQuantizer::<f16>::new(dim, distance);

        // Build the dense dataset.
        let dataset = DenseDataset::from_vec(data_vec, dim, quantizer.clone());

        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        let start = std::time::Instant::now();
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
        let elapsed = start.elapsed();
        println!("Time to build index: {:?}", elapsed);

        Ok(DensePlainHNSWf16 { index })
    }

    /// Build a dense plain index from a 1D NumPy array given the dimensionality.
    /// - `data_vec`: a 1D f32 array of len num_docs * dim
    /// - `dim`: dimensionality of the data
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during index construction
    /// - `metric`: either "l2" for Euclidean or "ip" for inner product
    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        // Convert the f32 data to f16.
        let data_vec: Vec<f16> = data_vec
            .as_slice()?
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();

        // Determine the distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Create a plain quantizer.
        let quantizer = PlainQuantizer::<f16>::new(dim, distance);

        // Build the dense dataset.
        let dataset = DenseDataset::from_vec(data_vec, dim, quantizer.clone());

        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        let start = std::time::Instant::now();
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
        let elapsed = start.elapsed();
        println!("Time to build index: {:?}", elapsed);

        Ok(DensePlainHNSWf16 { index })
    }

    /// Save the index to a file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        IndexSerializer::save_index(path, &self.index).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    /// Load a dense plain index from a file.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let index = IndexSerializer::load_index::<
            GraphIndex<DenseDataset<PlainQuantizer<f16>>, PlainQuantizer<f16>>,
        >(path);
        Ok(DensePlainHNSWf16 { index })
    }

    /// Search using a single query vector.
    ///
    /// Parameters:
    /// - `query`: a 1D NumPy array (f32) representing the query vector.
    /// - `k`: number of nearest neighbors to return.
    /// - `ef_search`: parameter controlling the candidate pool size during search.
    ///
    /// Returns a tuple (distances, ids) where:
    /// - `distances` is a 1D NumPy array (f32) of scores.
    /// - `ids` is a 1D NumPy array (i64) of document IDs.
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Convert the input query into a Rust slice.
        let query_slice = query.as_slice()?;
        // Determine the dimensionality of the query.
        let _dim = query_slice.len();

        // Set up the HNSW search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Convert the f32 query to f16.
        let query_slice: Vec<f16> = query_slice.iter().map(|&v| f16::from_f32(v)).collect();
        let query_slice = query_slice.as_slice();
        let query_darray = DenseVector1D::new(query_slice);

        // Perform the search.
        let search_results = self
            .index
            .search::<DenseDataset<PlainQuantizer<f16>>, PlainQuantizer<f16>>(
                query_darray,
                k,
                &search_config,
            );

        // Collect the results into vectors.
        let mut distances_vec: Vec<f32> = Vec::with_capacity(k);
        let mut ids_vec: Vec<i64> = Vec::with_capacity(k);
        for (score, doc_id) in search_results {
            distances_vec.push(score);
            ids_vec.push(doc_id as i64);
        }

        // Create NumPy arrays within a Python GIL context.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search using a 2D NumPy array of queries.
    ///
    /// Each row in `queries` is a query. For each query, the search returns `k` results.
    /// The method returns a tuple of two 1D NumPy arrays of shape (n_queries * k):
    /// (ids, distances). In Python you can do:
    ///
    ///     ids, distances = index.search(queries, k, ef_search)
    ///
    /// Here, `ids` is an array of doc IDs (as int64) and `distances` is an array of scores (as float32).
    pub fn search_batch(
        &self,
        queries_path: &str,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let (queries_slice, dim) = read_numpy_f32_flatten_2d(queries_path.to_string());
        let num_queries = queries_slice.len() / dim;

        // Set up the search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Pre-allocate vectors to store the results.
        // Each query returns k results; total elements = num_queries * k.
        let mut ids_vec: Vec<i64> = Vec::with_capacity(num_queries * k);
        let mut distances_vec: Vec<f32> = Vec::with_capacity(num_queries * k);

        // Iterate over each query using chunks_exact (each chunk is one query).
        for query in queries_slice.chunks_exact(dim) {
            // Convert the f32 query to f16.
            let query: Vec<f16> = query.iter().map(|&v| f16::from_f32(v)).collect();
            let query_slice = query.as_slice();
            let query_darray = DenseVector1D::new(query_slice);
            // Search returns a vector of (score, doc_id) pairs.
            let search_results = self
                .index
                .search::<DenseDataset<PlainQuantizer<f16>>, PlainQuantizer<f16>>(
                    query_darray,
                    k,
                    &search_config,
                );
            for (score, doc_id) in search_results {
                distances_vec.push(score);
                ids_vec.push(doc_id as i64);
            }
        }

        // Create the NumPy arrays within a Python GIL context.
        Python::with_gil(|py| {
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

/// A Python-exposed sparse index built with a plain quantizer.
#[pyclass]
pub struct SparsePlainHNSW {
    // The index type specialized for sparse datasets with SparsePlainQuantizer.
    index: GraphIndex<SparseDataset<SparsePlainQuantizer<f32>>, SparsePlainQuantizer<f32>>,
}

#[pymethods]
impl SparsePlainHNSW {
    /// Build a sparse plain index from a dataset file.
    /// The file is assumed to be in binary format.
    /// - `data_file`: path to the binary dataset file
    /// - `d`: dimensionality of the data
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during construction
    /// - `metric`: either "l2" or "ip"
    #[staticmethod]
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        // Determine distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Read the sparse dataset from file.
        let dataset: SparseDataset<SparsePlainQuantizer<f32>> = SparseDataset::<
            SparsePlainQuantizer<f32>,
        >::read_bin_file(
            data_file, d
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading dataset: {:?}", e))
        })?;

        // Create a quantizer for the sparse dataset.
        let quantizer = SparsePlainQuantizer::<f32>::new(dataset.dim(), distance);

        // Create HNSW configuration.
        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        // Build the index.
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);

        Ok(SparsePlainHNSW { index })
    }

    /// Build a sparse plain index from arrays of components (i32), values (f32) and offsets (i32).
    /// The file is assumed to be in binary format.
    /// - `data_file`: path to the binary dataset file
    /// - `components`: a 1D NumPy array (i32) of component indices.
    /// - `values`: a 1D NumPy array (f32) of values corresponding to the components.
    /// - `offsets`: a 1D NumPy array (i32) of offsets. offsets[i] indicates the start of the i-th document.
    /// - `d`: dimensionality of the vector space
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during construction
    /// - `metric`: either "l2" or "ip"
    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let components_vec = components
            .to_vec()
            .unwrap()
            .iter()
            .map(|x| *x as u16)
            .collect::<Vec<_>>();
        let values_slice = values.as_slice()?;
        let offsets_vec = offsets
            .to_vec()
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();

        // Determine distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Read the sparse dataset from file.
        let dataset = SparseDataset::<SparsePlainQuantizer<f32>>::from_vecs_f32(
            components_vec.as_slice(),
            values_slice,
            offsets_vec.as_slice(),
            d,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading dataset: {:?}", e))
        })?;

        // Create a quantizer for the sparse dataset.
        let quantizer = SparsePlainQuantizer::<f32>::new(dataset.dim(), distance);

        // Create HNSW configuration.
        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        // Build the index.
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);

        Ok(SparsePlainHNSW { index })
    }

    /// Save the sparse index to a file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        IndexSerializer::save_index(path, &self.index).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    /// Load a sparse plain index from a file.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let index = IndexSerializer::load_index::<
            GraphIndex<SparseDataset<SparsePlainQuantizer<f32>>, SparsePlainQuantizer<f32>>,
        >(path);
        Ok(SparsePlainHNSW { index })
    }

    /// Single-query search using sparse query.
    ///
    /// Parameters:
    /// - `query_components`: a 1D NumPy array (i32) of component indices for the query.
    /// - `query_values`: a 1D NumPy array (f32) of values corresponding to the components.
    /// - `d`: dimensionality of the vector space.
    /// - `k`: number of nearest neighbors to return.
    /// - `ef_search`: candidate pool size for the search.
    ///
    /// Returns a tuple (distances, ids) where:
    /// - `distances` is a 1D NumPy array (f32) of scores.
    /// - `ids` is a 1D NumPy array (i64) of document IDs.
    pub fn search(
        &self,
        query_components: numpy::PyReadonlyArray1<i32>,
        query_values: numpy::PyReadonlyArray1<f32>,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Extract slices from the NumPy arrays.
        let comp_vec = query_components
            .to_vec()
            .unwrap()
            .iter()
            .map(|x| *x as u16)
            .collect::<Vec<_>>();
        let values_slice = query_values.as_slice()?;

        // For a single query, the offsets vector is [0, number of values].
        let offsets_vec = vec![0, values_slice.len()];

        // Build a sparse query dataset from the parts.
        let query_dataset = SparseDataset::<SparsePlainQuantizer<f32>>::from_vecs_f32(
            &comp_vec,
            &values_slice,
            &offsets_vec,
            d,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error constructing query dataset: {:?}",
                e
            ))
        })?;

        // Check that we indeed have a single query.
        if query_dataset.len() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected a single query dataset.",
            ));
        }

        // Set up the search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Retrieve the single query from the dataset.
        let query = query_dataset.iter().next().unwrap();

        // Perform the search.
        let search_results = self
            .index
            .search::<SparseDataset<SparsePlainQuantizer<f32>>, SparsePlainQuantizer<f32>>(
                query,
                k,
                &search_config,
            );

        // Collect results.
        let mut distances_vec: Vec<f32> = Vec::with_capacity(k);
        let mut ids_vec: Vec<i64> = Vec::with_capacity(k);
        for (score, doc_id) in search_results {
            distances_vec.push(score);
            ids_vec.push(doc_id as i64);
        }

        // Create NumPy arrays from the results.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Search the sparse index with a batch of queries read from a file.
    ///
    /// Parameters:
    /// - `query_file`: path to the binary dataset file containing the queries.
    /// - `d`: dimensionality of the vector space.
    /// - `k`: number of nearest neighbors to return.
    ///
    /// Returns a list of tuples `(score, doc_id)`.
    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Read the queries from the binary file.
        let queries = SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file(query_file, d)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Error reading query file: {:?}",
                    e
                ))
            })?;
        let num_queries = queries.len();

        // Set up the search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Pre-allocate vectors to store the results.
        let mut ids_vec: Vec<i64> = Vec::with_capacity(num_queries * k);
        let mut distances_vec: Vec<f32> = Vec::with_capacity(num_queries * k);

        // Iterate over each query and perform search.
        for query in queries.iter() {
            let search_results = self
                .index
                .search::<SparseDataset<SparsePlainQuantizer<f32>>, SparsePlainQuantizer<f32>>(
                    query,
                    k,
                    &search_config,
                );
            for (score, doc_id) in search_results {
                distances_vec.push(score);
                ids_vec.push(doc_id as i64);
            }
        }

        // Convert the result vectors into 1D NumPy arrays.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

/// A Python-exposed sparse index built with a plain quantizer.
#[pyclass]
pub struct SparsePlainHNSWf16 {
    // The index type specialized for sparse datasets with SparsePlainQuantizer and f16 values.
    index: GraphIndex<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>>,
}

#[pymethods]
impl SparsePlainHNSWf16 {
    /// Build a sparse plain index from a binary dataset file.
    /// The file is assumed to be in binary format.
    /// - `data_file`: path to the binary dataset file
    /// - `d`: dimensionality of the vector space
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during construction
    /// - `metric`: either "l2" or "ip"
    #[staticmethod]
    #[pyo3(signature = (data_file, d, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_file(
        data_file: &str,
        d: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        // Determine distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Read the sparse dataset from file.
        let dataset =
            SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_f16(data_file, None, d)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error reading dataset: {:?}",
                        e
                    ))
                })?;

        // Create a quantizer for the sparse dataset.
        let quantizer = SparsePlainQuantizer::<f16>::new(dataset.dim(), distance);

        // Create HNSW configuration.
        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        // Build the index.
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);

        Ok(SparsePlainHNSWf16 { index })
    }

    /// Build a sparse plain index from arrays of components (i32), values (f32) and offsets (i32).
    ///
    /// Parameters:
    /// - `components`: a 1D NumPy array (i32) of component indices.
    /// - `values`: a 1D NumPy array (f32) of values corresponding to the components.
    /// - `offsets`: a 1D NumPy array (i32) of offsets. offsets[i] indicates the start of the i-th document.
    /// - `d`: dimensionality of the vector space
    /// - `m`: number of neighbors per node
    /// - `ef_construction`: candidate pool size during construction
    /// - `metric`: either "l2" or "ip"
    #[staticmethod]
    #[pyo3(signature = (components, values, offsets, d, m=32, ef_construction=200, metric="ip".to_string()))]
    pub fn build_from_arrays(
        components: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        offsets: PyReadonlyArray1<i32>,
        d: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
    ) -> PyResult<Self> {
        let components_vec = components
            .to_vec()
            .unwrap()
            .iter()
            .map(|x| *x as u16)
            .collect::<Vec<_>>();
        let values_vec = values
            .to_vec()
            .unwrap()
            .iter()
            .map(|&x| half::f16::from_f32(x))
            .collect::<Vec<_>>();
        let offsets_vec = offsets
            .to_vec()
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect::<Vec<_>>();

        // Determine distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Read the sparse dataset from file.
        let dataset = SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
            components_vec.as_slice(),
            values_vec.as_slice(),
            offsets_vec.as_slice(),
            d,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error reading dataset: {:?}", e))
        })?;

        // Create a quantizer for the sparse dataset.
        let quantizer = SparsePlainQuantizer::<f16>::new(dataset.dim(), distance);

        // Create HNSW configuration.
        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        // Build the index.
        let index = GraphIndex::from_dataset(&dataset, &config, quantizer);

        Ok(SparsePlainHNSWf16 { index })
    }

    /// Save the sparse index to a file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        IndexSerializer::save_index(path, &self.index).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    /// Load a sparse plain index from a file.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let index = IndexSerializer::load_index::<
            GraphIndex<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>>,
        >(path);
        Ok(SparsePlainHNSWf16 { index })
    }

    /// Single-query search using sparse query.
    ///
    /// Parameters:
    /// - `query_components`: a 1D NumPy array (i32) of component indices for the query.
    /// - `query_values`: a 1D NumPy array (f32) of values corresponding to the components.
    /// - `d`: dimensionality of the vector space.
    /// - `k`: number of nearest neighbors to return.
    /// - `ef_search`: candidate pool size for the search.
    ///
    /// Returns a tuple (distances, ids) where:
    /// - `distances` is a 1D NumPy array (f32) of scores.
    /// - `ids` is a 1D NumPy array (i64) of document IDs.
    pub fn search(
        &self,
        query_components: numpy::PyReadonlyArray1<i32>,
        query_values: numpy::PyReadonlyArray1<f32>,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Extract slices from the NumPy arrays.
        let comp_vec = query_components
            .to_vec()
            .unwrap()
            .iter()
            .map(|x| *x as u16)
            .collect::<Vec<_>>();
        let values_slice = query_values.as_slice()?;

        // Convert the f32 values to half precision (f16).
        let values_f16: Vec<half::f16> = values_slice
            .iter()
            .map(|&v| half::f16::from_f32(v))
            .collect();

        // For a single query, the offsets vector is [0, number of values].
        let offsets_vec = vec![0, values_f16.len()];

        // Build a sparse query dataset from the parts.
        let query_dataset = SparseDataset::<SparsePlainQuantizer<half::f16>>::from_vecs_f16(
            &comp_vec,
            &values_f16,
            &offsets_vec,
            d,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Error constructing query dataset: {:?}",
                e
            ))
        })?;

        // Check that we indeed have a single query.
        if query_dataset.len() != 1 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected a single query dataset.",
            ));
        }

        // Set up the search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Retrieve the single query from the dataset.
        let query = query_dataset.iter().next().unwrap();

        // Perform the search.
        let search_results = self.index.search::<SparseDataset<SparsePlainQuantizer<half::f16>>, SparsePlainQuantizer<half::f16>>(query, k, &search_config);

        // Collect results.
        let mut distances_vec: Vec<f32> = Vec::with_capacity(k);
        let mut ids_vec: Vec<i64> = Vec::with_capacity(k);
        for (score, doc_id) in search_results {
            distances_vec.push(score);
            ids_vec.push(doc_id as i64);
        }

        // Create NumPy arrays from the results.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Search the sparse index with a batch of queries read from a file.
    ///
    /// Parameters:
    /// - `query_file`: path to the binary dataset file containing the queries.
    /// - `d`: dimensionality of the vector space.
    /// - `k`: number of nearest neighbors to return.
    ///
    /// Returns a list of tuples `(score, doc_id)`.
    pub fn search_batch(
        &self,
        query_file: &str,
        d: usize,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Read the queries from the binary file.
        let queries =
            SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_f16(query_file, None, d)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error reading query file: {:?}",
                        e
                    ))
                })?;
        let num_queries = queries.len();

        // Set up the search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Pre-allocate vectors to store the results.
        let mut ids_vec: Vec<i64> = Vec::with_capacity(num_queries * k);
        let mut distances_vec: Vec<f32> = Vec::with_capacity(num_queries * k);

        // Iterate over each query and perform search.
        for query in queries.iter() {
            let search_results = self
                .index
                .search::<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>>(
                    query,
                    k,
                    &search_config,
                );
            for (score, doc_id) in search_results {
                distances_vec.push(score);
                ids_vec.push(doc_id as i64);
            }
        }

        // Convert the result vectors into 1D NumPy arrays.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}

// PQ
enum DensePQHNSWEnum {
    PQ8(GraphIndex<DenseDataset<ProductQuantizer<8>>, ProductQuantizer<8>>),
    PQ16(GraphIndex<DenseDataset<ProductQuantizer<16>>, ProductQuantizer<16>>),
    PQ32(GraphIndex<DenseDataset<ProductQuantizer<32>>, ProductQuantizer<32>>),
    PQ48(GraphIndex<DenseDataset<ProductQuantizer<48>>, ProductQuantizer<48>>),
    PQ64(GraphIndex<DenseDataset<ProductQuantizer<64>>, ProductQuantizer<64>>),
    PQ96(GraphIndex<DenseDataset<ProductQuantizer<96>>, ProductQuantizer<96>>),
    PQ128(GraphIndex<DenseDataset<ProductQuantizer<128>>, ProductQuantizer<128>>),
    PQ192(GraphIndex<DenseDataset<ProductQuantizer<192>>, ProductQuantizer<192>>),
    PQ256(GraphIndex<DenseDataset<ProductQuantizer<256>>, ProductQuantizer<256>>),
    PQ384(GraphIndex<DenseDataset<ProductQuantizer<384>>, ProductQuantizer<384>>),
}

/// The Python-exposed PQ index.
/// This type hides the const generics by keeping the internal dispatch logic hidden.
#[pyclass]
pub struct DensePQHNSW {
    inner: DensePQHNSWEnum,
}

#[pymethods]
impl DensePQHNSW {
    /// Build a PQ index from a 2D NumPy array (f32) of shape (num_docs, dim) saved in a npy file.
    ///
    /// Parameters:
    /// - `data_vec`: path to a 2D NumPy array (f32) of shape (num_docs, dim) saved in a npy file.
    /// - `m`: number of neighbors per node (for the HNSW graph)
    /// - `ef_construction`: candidate pool size during construction
    /// - `m_pq`: number of subspaces (const generic for ProductQuantizer). Supported values: 8, 16, or 32.
    /// - `nbits`: number of bits for each subspace.
    /// - `metric`: "l2" for Euclidean or "ip" for inner product.
    /// - `sample_size`: number of training samples for PQ training.
    #[staticmethod]
    #[pyo3(signature = (data_path, m_pq, nbits=8, m=32, ef_construction=200, metric="ip".to_string(), sample_size=100_000))]
    pub fn build_from_file(
        data_path: &str,
        m_pq: usize,
        nbits: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        sample_size: usize,
    ) -> PyResult<Self> {
        let (data_vec, dim) = read_numpy_f32_flatten_2d(data_path.to_string());

        // Choose distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Build the HNSW configuration.
        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        // Create a base dataset (using a plain quantizer as a placeholder).
        let dataset =
            DenseDataset::from_vec(data_vec, dim, PlainQuantizer::<f32>::new(dim, distance));

        // Sample training data from the dataset.
        let mut rng = StdRng::seed_from_u64(523);
        let mut training_vec: Vec<f32> = Vec::new();
        for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
            training_vec.extend(vec.values_as_slice());
        }
        let training_dataset =
            DenseDataset::from_vec(training_vec, dim, PlainQuantizer::<f32>::new(dim, distance));

        // Dispatch based on m_pq.
        let inner = match m_pq {
            8 => {
                let quantizer = ProductQuantizer::<8>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ8(index)
            }
            16 => {
                let quantizer = ProductQuantizer::<16>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ16(index)
            }
            32 => {
                let quantizer = ProductQuantizer::<32>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ32(index)
            }
            48 => {
                let quantizer = ProductQuantizer::<48>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ48(index)
            }
            64 => {
                let quantizer = ProductQuantizer::<64>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ64(index)
            }
            96 => {
                let quantizer = ProductQuantizer::<96>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ96(index)
            }
            128 => {
                let quantizer = ProductQuantizer::<128>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ128(index)
            }
            192 => {
                let quantizer = ProductQuantizer::<192>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ192(index)
            }
            256 => {
                let quantizer = ProductQuantizer::<256>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ256(index)
            }
            384 => {
                let quantizer = ProductQuantizer::<384>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ384(index)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ))
            }
        };

        Ok(DensePQHNSW { inner })
    }

    /// Build a PQ index from a 1D NumPy array given the dimensionality.
    ///
    /// Parameters:
    /// - `data_vec`: a 1D f32 array of len `num_docs * dim`.
    /// - `dim`: dimensionality of the data
    /// - `m`: number of neighbors per node (for the HNSW graph)
    /// - `ef_construction`: candidate pool size during construction
    /// - `m_pq`: number of subspaces (const generic for ProductQuantizer). Supported values: 8, 16, or 32.
    /// - `nbits`: number of bits for each subspace.
    /// - `metric`: "l2" for Euclidean or "ip" for inner product.
    /// - `sample_size`: number of training samples for PQ training.
    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m_pq, nbits=8, m=32, ef_construction=200, metric="ip".to_string(), sample_size=100_000))]
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
        let data_vec = data_vec.as_slice()?.to_vec();

        // Choose distance type.
        let distance = match metric.as_str() {
            "l2" => DistanceType::Euclidean,
            "ip" => DistanceType::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid metric; choose 'l2' or 'ip'",
                ))
            }
        };

        // Build the HNSW configuration.
        let config = ConfigHnsw::new()
            .num_neighbors(m)
            .ef_construction(ef_construction)
            .build();

        // Create a base dataset (using a plain quantizer as a placeholder).
        let dataset =
            DenseDataset::from_vec(data_vec, dim, PlainQuantizer::<f32>::new(dim, distance));

        // Sample training data from the dataset.
        let mut rng = StdRng::seed_from_u64(523);
        let mut training_vec: Vec<f32> = Vec::new();
        for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
            training_vec.extend(vec.values_as_slice());
        }
        let training_dataset =
            DenseDataset::from_vec(training_vec, dim, PlainQuantizer::<f32>::new(dim, distance));

        // Dispatch based on m_pq.
        let inner = match m_pq {
            8 => {
                let quantizer = ProductQuantizer::<8>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ8(index)
            }
            16 => {
                let quantizer = ProductQuantizer::<16>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ16(index)
            }
            32 => {
                let quantizer = ProductQuantizer::<32>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ32(index)
            }
            48 => {
                let quantizer = ProductQuantizer::<48>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ48(index)
            }
            64 => {
                let quantizer = ProductQuantizer::<64>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ64(index)
            }
            96 => {
                let quantizer = ProductQuantizer::<96>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ96(index)
            }
            128 => {
                let quantizer = ProductQuantizer::<128>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ128(index)
            }
            192 => {
                let quantizer = ProductQuantizer::<192>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ192(index)
            }
            256 => {
                let quantizer = ProductQuantizer::<256>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ256(index)
            }
            384 => {
                let quantizer = ProductQuantizer::<384>::train(&training_dataset, nbits, distance);
                let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
                DensePQHNSWEnum::PQ384(index)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ))
            }
        };

        Ok(DensePQHNSW { inner })
    }

    /// Load a PQ index from a file.
    /// The `m_pq` parameter tells the function which const generic instantiation to load.
    /// Supported values: 8, 16, or 32.
    #[staticmethod]
    pub fn load(path: &str, m_pq: usize) -> PyResult<Self> {
        let inner = match m_pq {
            8 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<8>>, ProductQuantizer<8>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ8(index)
            }
            16 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<16>>, ProductQuantizer<16>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ16(index)
            }
            32 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<32>>, ProductQuantizer<32>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ32(index)
            }
            48 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<48>>, ProductQuantizer<48>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ48(index)
            }
            64 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<64>>, ProductQuantizer<64>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ64(index)
            }
            96 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<96>>, ProductQuantizer<96>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ96(index)
            }
            128 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<128>>, ProductQuantizer<128>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ128(index)
            }
            192 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<192>>, ProductQuantizer<192>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ192(index)
            }
            256 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<256>>, ProductQuantizer<256>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ256(index)
            }
            384 => {
                let index: GraphIndex<DenseDataset<ProductQuantizer<384>>, ProductQuantizer<384>> =
                    IndexSerializer::load_index(path);
                DensePQHNSWEnum::PQ384(index)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value for load. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ))
            }
        };
        Ok(DensePQHNSW { inner })
    }

    /// Search using a single query vector.
    ///
    /// Parameters:
    /// - `query`: a 1D NumPy array (f32) representing the query vector.
    /// - `k`: number of nearest neighbors to return.
    /// - `ef_search`: parameter controlling the candidate pool size during search.
    ///
    /// Returns a tuple (distances, ids) where:
    /// - `distances` is a 1D NumPy array (f32) of scores.
    /// - `ids` is a 1D NumPy array (i64) of document IDs.
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        // Convert the input query to a Rust slice.
        let query_slice = query.as_slice()?;
        // (Optionally, you can check that the length matches the expected dimensionality.)

        // Build the HNSW search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Wrap the query slice in a DenseVector1D for search.
        let query_darray = DenseVector1D::new(query_slice);

        // Dispatch the search call based on the internal PQ variant.
        let results = match &self.inner {
            DensePQHNSWEnum::PQ8(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ16(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ32(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ48(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ64(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ96(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ128(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ192(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ256(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
            DensePQHNSWEnum::PQ384(index) => index
                .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                    query_darray,
                    k,
                    &search_config,
                ),
        };

        // Collect the results into vectors.
        let mut distances_vec: Vec<f32> = Vec::with_capacity(k);
        let mut ids_vec: Vec<i64> = Vec::with_capacity(k);
        for (score, doc_id) in results {
            distances_vec.push(score);
            ids_vec.push(doc_id as i64);
        }

        // Create the output NumPy arrays within a Python GIL context.
        Python::with_gil(|py| {
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Batch search using a 2D NumPy array of queries.
    ///
    /// Each row in `queries` is a query. For each query the search returns `k` results.
    /// The method returns a tuple of two 1D NumPy arrays:
    /// - distances (float32 scores)
    /// - ids (int64 document IDs)
    ///
    /// In Python you can do:
    ///
    ///     distances, ids = pq_index.search(queries, k, ef_search)
    pub fn search_batch(
        &self,
        queries_path: &str,
        k: usize,
        ef_search: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let (queries_slice, dim) = read_numpy_f32_flatten_2d(queries_path.to_string());
        let num_queries = queries_slice.len() / dim;

        // Build the HNSW search configuration.
        let mut search_config = ConfigHnsw::new().build();
        search_config.set_ef_search(ef_search);

        // Pre-allocate vectors for results.
        let mut ids_vec: Vec<i64> = Vec::with_capacity(num_queries * k);
        let mut distances_vec: Vec<f32> = Vec::with_capacity(num_queries * k);

        // Iterate over each query (each chunk is one query).
        for query in queries_slice.chunks_exact(dim) {
            let query_darray = DenseVector1D::new(query);
            // Dispatch based on the internal PQ variant.
            let results = match &self.inner {
                DensePQHNSWEnum::PQ8(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ16(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ32(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ48(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ64(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ96(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ128(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ192(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ256(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
                DensePQHNSWEnum::PQ384(index) => index
                    .search::<DenseDataset<PlainQuantizer<f32>>, PlainQuantizer<f32>>(
                        query_darray,
                        k,
                        &search_config,
                    ),
            };
            for (score, doc_id) in results {
                distances_vec.push(score);
                ids_vec.push(doc_id as i64);
            }
        }

        // Create the NumPy arrays within a Python GIL context.
        Python::with_gil(|py| {
            let ids_array = PyArray1::from_vec(py, ids_vec).to_owned();
            let distances_array = PyArray1::from_vec(py, distances_vec).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }
}
