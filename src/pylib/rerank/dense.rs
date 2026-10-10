//! `DenseRerankHNSW`: the Python binding of [`crate::rerank::DenseRerankIndex`]. Its arguments
//! mirror the `hnsw_rerank_build_dense` and `hnsw_rerank_search_dense` binaries.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::vector::DenseVectorView;
use vectorium::{Dataset, Index};

use crate::hnsw::EarlyTerminationStrategy;
use crate::rerank::{
    DenseEncoder, DenseRerankConfig, DenseRerankIndex, GraphLayout, RerankError, RerankSearchParams,
};

use super::super::hnsw::{GraphTypeKind, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{MetricKind, parse_metric, push_pairs, read_npy_dataset};

/// A dense two-stage index: compressed HNSW over `f16`-derived codes, reranked on plain `f16` or
/// on RaBitQ-ext codes (`rerank_bits`).
#[pyclass]
pub struct DenseRerankHNSW {
    inner: Inner,
}

enum Inner {
    DotProduct(DenseRerankIndex<DotProduct>),
    Euclidean(DenseRerankIndex<SquaredEuclideanDistance>),
}

/// Runs `$body` with `$index` bound to whichever metric `$inner` holds.
macro_rules! with_inner {
    ($inner:expr, $index:ident => $body:expr) => {
        match $inner {
            Inner::DotProduct($index) => $body,
            Inner::Euclidean($index) => $body,
        }
    };
}

impl Inner {
    fn dim(&self) -> usize {
        with_inner!(self, index => index.dim())
    }

    /// Searches one query and returns `(score, id)` pairs.
    fn search(&self, query: &[f32], k: usize, params: &RerankSearchParams) -> Vec<(f32, usize)> {
        with_inner!(self, index => index
            .search(DenseVectorView::new(query), k, params)
            .into_iter()
            .map(|r| (r.distance.distance(), r.vector as usize))
            .collect())
    }
}

pub(super) fn value_err(e: RerankError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyValueError, _>(e.0)
}

pub(super) fn io_err(e: RerankError) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyIOError, _>(e.0)
}

pub(super) fn layout(gt: GraphTypeKind) -> GraphLayout {
    match gt {
        GraphTypeKind::Standard => GraphLayout::Standard,
        GraphTypeKind::Permuted => GraphLayout::Permuted,
        GraphTypeKind::Compressed => GraphLayout::Compressed,
    }
}

fn to_encoder(
    name: &str,
    pq_subspaces: usize,
    rabitq_total_bits: u32,
    rabitq_query_bits: u32,
) -> PyResult<DenseEncoder> {
    match name {
        "pq" => Ok(DenseEncoder::Pq {
            subspaces: pq_subspaces,
        }),
        "rabitq" => Ok(DenseEncoder::Rabitq {
            query_bits: rabitq_query_bits,
        }),
        "rabitq-ext" => Ok(DenseEncoder::RabitqExt {
            total_bits: rabitq_total_bits,
        }),
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown encoder {other:?}; choose 'pq', 'rabitq' or 'rabitq-ext'"
        ))),
    }
}

pub(super) fn search_params(
    k_candidates: usize,
    ef_search: usize,
    alpha: Option<f32>,
    beta: Option<usize>,
    early_exit_threshold: Option<f32>,
) -> RerankSearchParams {
    RerankSearchParams {
        k_candidates,
        ef_search,
        alpha,
        beta,
        early_termination: match early_exit_threshold {
            Some(lambda) => EarlyTerminationStrategy::DistanceAdaptive { lambda },
            None => EarlyTerminationStrategy::None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    values: &[f32],
    dim: usize,
    m: usize,
    ef_construction: usize,
    metric: &str,
    graph_type: &str,
    encoder: &str,
    pq_subspaces: usize,
    rabitq_total_bits: u32,
    rabitq_query_bits: u32,
    rerank_bits: u32,
    residual_rerank: bool,
) -> PyResult<DenseRerankHNSW> {
    let cfg = DenseRerankConfig {
        m,
        ef_construction,
        graph: layout(parse_build_graph_type(graph_type, m)?),
        encoder: to_encoder(encoder, pq_subspaces, rabitq_total_bits, rabitq_query_bits)?,
        rerank_bits,
        residual_rerank,
    };
    let inner = match parse_metric(metric)? {
        MetricKind::DotProduct => Inner::DotProduct(
            DenseRerankIndex::<DotProduct>::build(values, dim, &cfg).map_err(value_err)?,
        ),
        MetricKind::Euclidean => Inner::Euclidean(
            DenseRerankIndex::<SquaredEuclideanDistance>::build(values, dim, &cfg)
                .map_err(value_err)?,
        ),
    };
    Ok(DenseRerankHNSW { inner })
}

#[pymethods]
impl DenseRerankHNSW {
    /// Build from a flat row-major `float32` array plus an explicit `dim`.
    ///
    /// `encoder` is one of `pq`, `rabitq`, `rabitq-ext`; `pq_subspaces` applies to `pq`,
    /// `rabitq_total_bits` to `rabitq-ext`, and `rabitq_query_bits` to `rabitq`.
    ///
    /// `rerank_bits` sets the rerank stage's encoding. The default `0` is plain `f16`; `2`, `4`
    /// and `8` use RaBitQ-ext codes instead, which makes the rerank dataset smaller.
    /// `residual_rerank` encodes `x − x̂` instead of `x` and scores a candidate as first-stage
    /// plus rerank score; it needs `rerank_bits > 0`, a `rabitq` or `rabitq-ext` first stage, and
    /// dot product.
    #[staticmethod]
    #[pyo3(signature = (
        data_vec, dim, m=32, ef_construction=200, metric="dotproduct".to_string(),
        graph_type="standard".to_string(), encoder="pq".to_string(), pq_subspaces=32,
        rabitq_total_bits=4, rabitq_query_bits=1, rerank_bits=0, residual_rerank=false
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
        encoder: String,
        pq_subspaces: usize,
        rabitq_total_bits: u32,
        rabitq_query_bits: u32,
        rerank_bits: u32,
        residual_rerank: bool,
    ) -> PyResult<Self> {
        build(
            data_vec.as_slice()?,
            dim,
            m,
            ef_construction,
            &metric,
            &graph_type,
            &encoder,
            pq_subspaces,
            rabitq_total_bits,
            rabitq_query_bits,
            rerank_bits,
            residual_rerank,
        )
    }

    /// Build from a `.npy` collection, read as `f32`.
    ///
    /// `dim` comes from the file's shape. Otherwise identical to
    /// [`build_from_array`](Self::build_from_array).
    #[staticmethod]
    #[pyo3(signature = (
        data_path, m=32, ef_construction=200, metric="dotproduct".to_string(),
        graph_type="standard".to_string(), encoder="pq".to_string(), pq_subspaces=32,
        rabitq_total_bits=4, rabitq_query_bits=1, rerank_bits=0, residual_rerank=false
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_from_file(
        data_path: &str,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
        encoder: String,
        pq_subspaces: usize,
        rabitq_total_bits: u32,
        rabitq_query_bits: u32,
        rerank_bits: u32,
        residual_rerank: bool,
    ) -> PyResult<Self> {
        // The metric only labels the dataset type here; the raw values are the same either way.
        let dataset = read_npy_dataset::<DotProduct>(data_path)?;
        build(
            dataset.values(),
            dataset.output_dim(),
            m,
            ef_construction,
            &metric,
            &graph_type,
            &encoder,
            pq_subspaces,
            rabitq_total_bits,
            rabitq_query_bits,
            rerank_bits,
            residual_rerank,
        )
    }

    /// Writes the first-stage graph and the rerank dataset to one file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        with_inner!(&self.inner, index => index.save(path)).map_err(io_err)
    }

    /// Loads an index written by [`Self::save`].
    ///
    /// Index files carry no header, so `metric`, `graph_type`, `encoder`, `pq_subspaces` and
    /// `residual_rerank` must match the values used at build time. `rerank_bits` must be `0` for
    /// an `f16` rerank stage and `2`, `4` or `8` otherwise. `rabitq_total_bits` and the exact
    /// `rerank_bits` are stored with the index. `rabitq_query_bits` may differ from the build-time
    /// value.
    #[staticmethod]
    #[pyo3(signature = (
        path, metric="dotproduct".to_string(), graph_type="standard".to_string(),
        encoder="pq".to_string(), pq_subspaces=32, rabitq_query_bits=1, rerank_bits=0,
        residual_rerank=false
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        path: &str,
        metric: String,
        graph_type: String,
        encoder: String,
        pq_subspaces: usize,
        rabitq_query_bits: u32,
        rerank_bits: u32,
        residual_rerank: bool,
    ) -> PyResult<Self> {
        // `total_bits` is stored with a RaBitQ-ext encoder and ignored on load.
        let enc = to_encoder(&encoder, pq_subspaces, 0, rabitq_query_bits)?;
        let graph = layout(parse_graph_type(&graph_type)?);
        let inner = match parse_metric(&metric)? {
            MetricKind::DotProduct => Inner::DotProduct(
                DenseRerankIndex::<DotProduct>::load(
                    path,
                    graph,
                    enc,
                    rerank_bits,
                    residual_rerank,
                )
                .map_err(io_err)?,
            ),
            MetricKind::Euclidean => Inner::Euclidean(
                DenseRerankIndex::<SquaredEuclideanDistance>::load(
                    path,
                    graph,
                    enc,
                    rerank_bits,
                    residual_rerank,
                )
                .map_err(io_err)?,
            ),
        };
        Ok(DenseRerankHNSW { inner })
    }

    /// Changes the RaBitQ query bit width on this index.
    ///
    /// The width only affects how the query is encoded, not the stored codes, so one index serves
    /// every width.
    ///
    /// Raises if `bits` is outside `1..=8`, or if the encoder does not quantize the query (`pq`
    /// and `rabitq-ext`).
    pub fn set_query_bits(&mut self, bits: u32) -> PyResult<()> {
        with_inner!(&mut self.inner, index => index.set_query_bits(bits)).map_err(value_err)
    }

    /// Two-stage search for a single query.
    ///
    /// `k_candidates` is the number of first-stage candidates to rerank; `ef_search` sizes the
    /// first-stage candidate list; `alpha` prunes candidates and `beta` enables early exit in
    /// reranking, both off by default.
    #[pyo3(signature = (
        query, k, k_candidates=100, ef_search=100, alpha=None, beta=None,
        early_exit_threshold=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        k_candidates: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let query_slice = query.as_slice()?;
        if query_slice.len() != self.inner.dim() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "query dimension {} does not match index dimension {}",
                query_slice.len(),
                self.inner.dim()
            )));
        }
        let params = search_params(k_candidates, ef_search, alpha, beta, early_exit_threshold);

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        push_pairs(
            self.inner.search(query_slice, k, &params),
            k,
            &mut distances,
            &mut ids,
        );

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Two-stage search over many queries, flattened into one contiguous buffer.
    ///
    /// `queries` holds `n_queries * dim` floats. Results are query-major: `distances` and `ids`
    /// each hold `n_queries * k` entries. `num_threads` is 0 for every core, 1 for serial, or an
    /// explicit count.
    #[pyo3(signature = (
        queries, k, k_candidates=100, ef_search=100, alpha=None, beta=None,
        early_exit_threshold=None, num_threads=0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray1<f32>,
        k: usize,
        k_candidates: usize,
        ef_search: usize,
        alpha: Option<f32>,
        beta: Option<usize>,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let queries_slice = queries.as_slice()?;
        let dim = self.inner.dim();
        if !queries_slice.len().is_multiple_of(dim) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "queries array length {} is not a multiple of index dimension {}",
                queries_slice.len(),
                dim
            )));
        }
        let num_queries = queries_slice.len() / dim;
        let params = search_params(k_candidates, ef_search, alpha, beta, early_exit_threshold);

        let search_one = |i: usize| -> Vec<(f32, usize)> {
            self.inner
                .search(&queries_slice[i * dim..(i + 1) * dim], k, &params)
        };

        let results: Vec<Vec<(f32, usize)>> = py.detach(|| match num_threads {
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
        for r in results {
            push_pairs(r, k, &mut all_distances, &mut all_ids);
        }

        let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
        let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }

    /// Bytes held by the first-stage index plus the rerank dataset.
    pub fn space_usage_bytes(&self) -> usize {
        with_inner!(&self.inner, index => index.space_usage_bytes())
    }
}
