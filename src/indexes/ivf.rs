use std::collections::BinaryHeap;
use std::marker::PhantomData;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::index as rand_index;
use serde::{Deserialize, Serialize};
use vectorium::core::dataset::{ConvertFrom, ScoredItemGeneric, ScoredVector};
use vectorium::core::index::Index;
use vectorium::core::vector::DenseVectorView;
use vectorium::core::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::distances::DotProduct;
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::{
    Dataset, DatasetGrowable, DenseDataset, Distance, Float, FromF32, IndexSerializer, KMeans,
    KMeansBuilder, PlainDenseDataset, PlainDenseDatasetGrowable, QueryEvaluator,
    ScalarDenseQuantizer, SquaredEuclideanDistance, ValueType, VectorId,
};

use crate::graph::Graph;
use crate::hnsw::{HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration};

/// The distance reported by an IVF over centroid/query dataset `DQ`.
type Reported<DQ> = <<DQ as Dataset>::Encoder as VectorEncoder>::Distance;

// ---------------------------------------------------------------------------
// ReportedMetric
// ---------------------------------------------------------------------------

/// The metric [`IVF::search`] reports. The cluster dataset **always** stores
/// dot-product vectors, so every candidate contributes a raw dot product
/// `raw = ⟨q, w⟩` (where `w` is the stored vector `v`, or the residual `r`).
/// The final reported distance is reconstructed from three pieces:
///
/// `score = centroid + correction[pos] + β·raw`
///
/// - `centroid` — the cluster centroid's contribution (`⟨q,c⟩` or `‖q−c‖²`),
///   added only in *residual* mode (see [`IVF::is_residual`]); `0.0` otherwise.
/// - `correction[pos]` — per-vector offline term (L2 only, see below); `0.0`
///   for dot product.
/// - `β` — `+1` for dot product, `−2` for squared Euclidean.
///
/// The metric is a compile-time type (so `β`, the output type, and correction
/// use are resolved at compile time), while *residual-vs-plain* is a runtime
/// flag on the index.
///
/// ## Squared-Euclidean correction & the `‖q‖²` omission
///
/// `‖q − w'‖² = ‖q‖² + ‖w'‖² − 2⟨q,w'⟩`. We drop the query-constant `‖q‖²` (it
/// does not affect ranking) and fold the rest into `correction`:
///
/// - **plain**  (`w' = v`, `centroid = 0`): `correction[pos] = ‖v‖²`  →
///   `score = ‖v‖² − 2⟨q,v⟩`  (equals `‖q−v‖² − ‖q‖²`; add `‖q‖²` per query if
///   the *exact* squared distance is required — ranking is identical without it).
/// - **residual** (`w' = c + r̂`, `centroid = ‖q−c‖²`):
///   `correction[pos] = ‖r̂‖² + 2⟨c,r̂⟩`  →  `score = ‖q−c‖² + ‖r̂‖²+2⟨c,r̂⟩ − 2⟨q,r̂⟩`
///   `= ‖q − (c+r̂)‖²`. The correction must use the *reconstructed* `r̂` (the same
///   residual reconstruction the search scores against); mixing exact `‖r‖²` with
///   the approximate `⟨q,r̂⟩` leaks a `2⟨q, r−r̂⟩` ranking error.
pub trait ReportedMetric: Distance {
    /// Whether this metric consumes the per-vector `correction` term.
    const USES_CORRECTION: bool;

    /// Reconstruct the reported distance from the centroid contribution, the
    /// per-vector correction, and the raw dot product `⟨q, w⟩`.
    fn combine(centroid: f32, correction: f32, raw_dot: f32) -> Self;
}

impl ReportedMetric for SquaredEuclideanDistance {
    const USES_CORRECTION: bool = true;

    #[inline]
    fn combine(centroid: f32, correction: f32, raw_dot: f32) -> Self {
        SquaredEuclideanDistance::from(centroid + correction - 2.0 * raw_dot)
    }
}

impl ReportedMetric for DotProduct {
    const USES_CORRECTION: bool = false;

    #[inline]
    fn combine(centroid: f32, _correction: f32, raw_dot: f32) -> Self {
        DotProduct::from(centroid + raw_dot)
    }
}

// ---------------------------------------------------------------------------
// IVF
// ---------------------------------------------------------------------------

/// Inverted Vector File (IVF) index: a centroid index + a dataset for all clusters.
///
/// Conceptually a pair **(CI, Q)**: a centroid index `CI` and a quantizer `Q`
/// (the encoder of the cluster dataset `D`). Whether the dataset stores the
/// original vectors or residuals `r = v − c` is a **runtime** property
/// ([`is_residual`](Self::is_residual)) — *not* part of the type — so a single build
/// path and a single search cover both. The cluster dataset always uses
/// **dot-product** storage; the reported metric is `DQ`'s metric (see
/// [`ReportedMetric`]).
///
/// Every configuration implements vectorium's [`Index`] over the **query/input
/// dataset** `DQ` (whose metric is the reported metric), and also exposes the
/// inherent [`search`](IVF::search).
///
/// ## Type parameters
/// - `DQ`: centroid / query dataset; `DQ`'s metric is the reported metric.
/// - `D`:  cluster dataset (dot-product storage: f32/f16/PQ).
/// - `CI`: centroid index — e.g. `FlatIndex<DQ>` or `HNSW<DQ, Graph>`.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "CI: Serialize, D: Serialize",
    deserialize = "CI: for<'de2> Deserialize<'de2>, D: for<'de2> Deserialize<'de2>"
))]
pub struct IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    centroid_index: CI,
    dataset: D,
    /// CSR offsets: cluster `c` owns `dataset[cluster_offsets[c]..cluster_offsets[c+1]]`.
    /// Length is `n_clusters + 1`.
    cluster_offsets: Box<[u32]>,
    /// `external_ids[i]` = original dataset ID of the vector at reordered position `i`.
    /// Only accessed to label the final top-`k` results.
    external_ids: Box<[u32]>,
    /// Per-vector L2 correction term (`‖v‖²` for plain, `‖r̂‖²+2⟨c,r̂⟩` for
    /// residual). `Some` for squared-Euclidean indexes, `None` for dot product.
    correction_terms: Option<Box<[f32]>>,
    /// `true` when `dataset` stores residuals `r = v − c` (so the centroid
    /// contribution is added at search time); `false` when it stores `v`.
    residual: bool,
    _phantom: PhantomData<DQ>,
}

impl<DQ, D, CI> IndexSerializer for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
}

/// Search parameters for [`IVF`].
///
/// `P` is the cluster dataset encoder's [`VectorEncoder::QueryParams`], defaulting to `()`
/// for the encoders that take no query-side configuration.
pub struct IVFSearchParams<CI, P = ()>
where
    CI: Index,
{
    /// Number of nearest centroids to probe.
    pub nprobe: usize,
    pub centroid_search_params: CI::SearchParams,
    /// Query-side parameters for the cluster dataset's encoder, forwarded to
    /// [`VectorEncoder::query_evaluator`].
    pub cluster_query_params: P,
}

// ---------------------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------------------

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    pub fn n_clusters(&self) -> usize {
        self.cluster_offsets.len().saturating_sub(1)
    }

    pub fn n_elements(&self) -> usize {
        self.external_ids.len()
    }

    /// Whether the cluster dataset stores residuals `r = v − c` (`true`) or the
    /// original vectors `v` (`false`).
    pub fn is_residual(&self) -> bool {
        self.residual
    }

    pub fn print_space_usage_bytes(&self) {
        let offsets = self.cluster_offsets.len() * std::mem::size_of::<u32>();
        let ids = self.external_ids.len() * std::mem::size_of::<u32>();
        let corr = self.correction_terms.as_ref().map_or(0, |c| c.len() * 4);
        println!("IVF: cluster_offsets={offsets} B  external_ids={ids} B  correction={corr} B");
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
    // The centroid index accepts the same (raw f32) query the IVF does.
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    D::Encoder: VectorEncoder<Distance = DotProduct>,
{
    /// Search the index.
    ///
    /// The reported distance is `DQ`'s metric ([`ReportedMetric`]); the cluster
    /// dataset stores dot products, and residual-vs-plain is handled at runtime
    /// via [`is_residual`](Self::is_residual). The per-candidate `residual` and
    /// `USES_CORRECTION` decisions fold to constants, so the hot loop stays a
    /// single `centroid + correction − 2·raw` (L2) or `centroid + raw` (dot).
    pub fn search<'q>(
        &self,
        query: <DQ::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        params: &IVFSearchParams<CI, <D::Encoder as VectorEncoder>::QueryParams>,
    ) -> Vec<ScoredVector<<DQ::Encoder as VectorEncoder>::Distance>>
    where
        <DQ::Encoder as VectorEncoder>::QueryVector<'q>:
            Into<<D::Encoder as VectorEncoder>::QueryVector<'q>> + Copy,
    {
        let nearest_centroids =
            self.centroid_index
                .search(query, params.nprobe, &params.centroid_search_params);

        let evaluator = self
            .dataset
            .encoder()
            .query_evaluator(query.into(), &params.cluster_query_params);
        let correction: &[f32] = self.correction_terms.as_deref().unwrap_or(&[]);
        let residual = self.residual;
        let uses_correction = <Reported<DQ> as ReportedMetric>::USES_CORRECTION;

        type HeapItem<Dd> = ScoredItemGeneric<Dd, u32>;
        let mut heap: BinaryHeap<HeapItem<Reported<DQ>>> = BinaryHeap::with_capacity(k + 1);

        // Turn the raw dot product into the reported distance.
        let push = |heap: &mut BinaryHeap<HeapItem<Reported<DQ>>>,
                    pos: usize,
                    centroid: f32,
                    raw: DotProduct| {
            let corr = if uses_correction {
                correction[pos]
            } else {
                0.0
            };
            let dist = <Reported<DQ> as ReportedMetric>::combine(centroid, corr, raw.distance());
            let item = HeapItem {
                distance: dist,
                vector: pos as u32,
            };
            if heap.len() < k {
                heap.push(item);
            } else if item < *heap.peek().unwrap() {
                heap.pop();
                heap.push(item);
            }
        };

        for scored_centroid in &nearest_centroids {
            let cluster_id = scored_centroid.vector as usize;
            // Centroid contribution is added only in residual mode.
            let centroid = if residual {
                scored_centroid.distance.distance()
            } else {
                0.0
            };
            let start = self.cluster_offsets[cluster_id] as usize;
            let end = self.cluster_offsets[cluster_id + 1] as usize;

            let batch_end = start + ((end - start) / 6) * 6;
            let mut pos = start;
            while pos < batch_end {
                let dists = evaluator.compute_distances_batch6([
                    self.dataset.get(pos as VectorId),
                    self.dataset.get((pos + 1) as VectorId),
                    self.dataset.get((pos + 2) as VectorId),
                    self.dataset.get((pos + 3) as VectorId),
                    self.dataset.get((pos + 4) as VectorId),
                    self.dataset.get((pos + 5) as VectorId),
                ]);
                for (i, &raw) in dists.iter().enumerate() {
                    push(&mut heap, pos + i, centroid, raw);
                }
                pos += 6;
            }
            for p in batch_end..end {
                let raw = evaluator.compute_distance(self.dataset.get(p as VectorId));
                push(&mut heap, p, centroid, raw);
            }
        }

        heap.into_sorted_vec()
            .into_iter()
            .map(|item| ScoredItemGeneric {
                distance: item.distance,
                vector: self.external_ids[item.vector as usize] as VectorId,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    /// Construct from pre-built components.
    ///
    /// - `residual = false`: `dataset` stores the (dot-product-encoded) original
    ///   vectors `v`. For L2, `correction_terms[i] = ‖v‖²`; for dot product,
    ///   `correction_terms = None`.
    /// - `residual = true`: `dataset` stores residuals `r = v − c`. For L2,
    ///   `correction_terms[i] = ‖r̂‖² + 2⟨c,r̂⟩`; for dot product, `None`.
    ///
    /// `cluster_offsets` must be CSR-format with length `n_clusters + 1`.
    /// `external_ids[i]` is the original ID of the vector at reordered position `i`.
    pub fn from_parts(
        centroid_index: CI,
        dataset: D,
        cluster_offsets: Box<[u32]>,
        external_ids: Box<[u32]>,
        correction_terms: Option<Box<[f32]>>,
        residual: bool,
    ) -> Self {
        debug_assert!(!cluster_offsets.is_empty());
        debug_assert_eq!(
            *cluster_offsets.last().unwrap() as usize,
            external_ids.len()
        );
        debug_assert!(
            correction_terms
                .as_ref()
                .is_none_or(|c| c.len() == external_ids.len())
        );
        Self {
            centroid_index,
            dataset,
            cluster_offsets,
            external_ids,
            correction_terms,
            residual,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Dense build infrastructure
// ---------------------------------------------------------------------------

type DensePlain = PlainDenseDataset<f32, SquaredEuclideanDistance>;

/// Artifacts produced by [`prepare_dense_ivf`], ready to be assembled into an
/// [`IVF`] index of any centroid index / encoding combination.
pub struct DenseBuildArtifacts {
    /// Centroids in f32, as returned by k-means. Convert to the desired
    /// centroid precision (e.g. f16) and build the centroid index externally.
    pub centroids_f32: DensePlain,
    /// Vectors reordered so that each cluster occupies a contiguous range.
    /// If `with_residuals = true`, this contains `v − c` instead of `v`.
    pub reordered_dataset: DensePlain,
    /// CSR offsets, length `n_clusters + 1`.
    pub cluster_offsets: Box<[u32]>,
    /// `external_ids[i]` = original vector ID at reordered position `i`.
    pub external_ids: Box<[u32]>,
}

/// Dispatch k-means training to flat or HNSW assignment based on `use_hnsw`.
fn run_kmeans(
    kmeans: &KMeans,
    dataset: &DensePlain,
    n_clusters: usize,
    use_hnsw: bool,
) -> DensePlain {
    if use_hnsw {
        let build_cfg = HNSWBuildConfiguration::default()
            .with_num_neighbors(KMEANS_HNSW_M)
            .with_ef_construction(KMEANS_HNSW_EF_CONSTRUCTION);
        let search_cfg = HNSWSearchConfiguration::default().with_ef_search(KMEANS_HNSW_EF_SEARCH);
        kmeans.train_with_index::<HNSW<DensePlain, Graph>>(
            dataset,
            n_clusters,
            None,
            |centroids| HNSW::build_index(centroids, &build_cfg),
            &search_cfg,
        )
    } else {
        kmeans.train(dataset, n_clusters, None)
    }
}

/// Auto-threshold: use HNSW for centroid assignment when `n_clusters` reaches
/// this value. Below it a flat exhaustive scan is fast enough; above it
/// building + searching an HNSW is faster.
pub const KMEANS_HNSW_THRESHOLD: usize = 32_768;

/// HNSW parameters used for centroid assignment during k-means.
/// If you want better quality, increase ef_search and eventually ef_construction.
const KMEANS_HNSW_M: usize = 32;
const KMEANS_HNSW_EF_CONSTRUCTION: usize = 200;
const KMEANS_HNSW_EF_SEARCH: usize = 16;

/// Run k-means, assign vectors to clusters, and produce a reordered dataset
/// ready to be assembled into any [`IVF`] variant.
///
/// Call this when you need to control the centroid precision or cluster encoding
/// externally (e.g. f16 centroids, PQ-compressed clusters). The returned
/// [`DenseBuildArtifacts`] owns the f32 centroids; convert and consume them to
/// build your centroid index, then pass the remaining fields to [`IVF::from_parts`].
///
/// If `with_residuals = true`, `artifacts.reordered_dataset` contains residuals
/// `v − c` instead of `v`. The L2 correction term is not produced here — the
/// caller computes it from the *stored* cluster vectors via [`l2_correction`],
/// which is what keeps it exact for lossy f16 / PQ storage.
///
/// `sample_size`: if `Some(s)` and `s < n`, k-means is trained on a random
/// sample of `s` vectors drawn without replacement from `dataset`. Assignment
/// of all `n` vectors to the resulting centroids always uses the full dataset.
/// A sample of ~256 × n_clusters is typically sufficient for good centroid
/// quality and reduces training time from O(n · k) to O(s · k) per iteration.
pub fn prepare_dense_ivf(
    dataset: &DensePlain,
    n_clusters: usize,
    kmeans_builder: KMeansBuilder,
    with_residuals: bool,
    sample_size: Option<usize>,
    kmeans_hnsw: bool,
) -> DenseBuildArtifacts {
    let dim = dataset.input_dim();
    let n = dataset.len();

    let kmeans = kmeans_builder.build();

    let centroids = match sample_size.filter(|&s| s < n) {
        Some(ss) => {
            println!("Sampling {ss} / {n} vectors for k-means training");
            let mut rng = StdRng::from_entropy();
            let indices = rand_index::sample(&mut rng, n, ss);
            let mut builder =
                PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::with_capacity(
                    ScalarDenseQuantizer::new(dim),
                    ss,
                );
            for idx in indices {
                builder.push(dataset.get(idx as VectorId));
            }
            let sampled: DensePlain = builder.into();
            run_kmeans(&kmeans, &sampled, n_clusters, kmeans_hnsw)
        }
        None => run_kmeans(&kmeans, dataset, n_clusters, kmeans_hnsw),
    };

    // Assign all vectors to their nearest centroid.
    // Use HNSW when requested: build once on final centroids, search the full dataset.
    let assignments = if kmeans_hnsw {
        let build_cfg = HNSWBuildConfiguration::default()
            .with_num_neighbors(KMEANS_HNSW_M)
            .with_ef_construction(KMEANS_HNSW_EF_CONSTRUCTION);
        let search_cfg = HNSWSearchConfiguration::default().with_ef_search(KMEANS_HNSW_EF_SEARCH);
        let centroid_index = HNSW::<DensePlain, Graph>::build_index(centroids.clone(), &build_cfg);
        KMeans::assign_with_index(dataset, &centroid_index, &search_cfg)
    } else {
        KMeans::compute_assignments(dataset, &centroids, 0)
    };

    // Centroid raw values, needed to form residuals below. The L2 correction is
    // computed later by the caller from the stored cluster vectors (l2_correction).
    let centroid_raw: Vec<f32> = if with_residuals {
        centroids.values().to_vec()
    } else {
        Vec::new()
    };

    // CSR offsets from per-cluster counts.
    let mut cluster_sizes = vec![0u32; n_clusters];
    for &(_, c) in &assignments {
        cluster_sizes[c] += 1;
    }
    let mut cluster_offsets = vec![0u32; n_clusters + 1];
    for c in 0..n_clusters {
        cluster_offsets[c + 1] = cluster_offsets[c] + cluster_sizes[c];
    }

    // Permutation grouping vectors by cluster. Unstable sort: order within a
    // cluster is arbitrary and irrelevant for correctness.
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by_key(|&i| assignments[i as usize].1);

    // Build reordered data (residuals when requested).
    let mut reordered = Vec::with_capacity(n * dim);
    for &orig in &order {
        let v = dataset.get(orig as VectorId).values();
        if with_residuals {
            let c_id = assignments[orig as usize].1;
            let c = &centroid_raw[c_id * dim..(c_id + 1) * dim];
            reordered.extend(v.iter().zip(c).map(|(&vi, &ci)| vi - ci));
        } else {
            reordered.extend_from_slice(v);
        }
    }

    let reordered_dataset = DensePlain::from_raw(
        reordered.into_boxed_slice(),
        n,
        ScalarDenseQuantizer::new(dim),
    );

    DenseBuildArtifacts {
        centroids_f32: centroids,
        reordered_dataset,
        cluster_offsets: cluster_offsets.into_boxed_slice(),
        external_ids: order.into_boxed_slice(),
    }
}

/// f32 dot-product cluster storage — the canonical L2 IVF stores `v` (or `r`)
/// here and reports squared-Euclidean via the correction terms.
pub type DensePlainDP = PlainDenseDataset<f32, DotProduct>;

// ---------------------------------------------------------------------------
// Cluster encoders + L2 correction (shared by the binaries and the Index impls)
// ---------------------------------------------------------------------------

/// Reinterpret the reordered f32 values as `V`-precision dot-product plain storage.
pub fn plain_dp_cluster<V>(artifacts: &DenseBuildArtifacts) -> PlainDenseDataset<V, DotProduct>
where
    V: Float + FromF32 + ValueType,
    PlainDenseDataset<V, DotProduct>: Dataset,
{
    let n = artifacts.reordered_dataset.len();
    let dim = artifacts.reordered_dataset.input_dim();
    let vals: Vec<V> = artifacts
        .reordered_dataset
        .values()
        .iter()
        .map(|&v| V::from_f32_saturating(v))
        .collect();
    PlainDenseDataset::<V, DotProduct>::from_raw(
        vals.into_boxed_slice(),
        n,
        ScalarDenseQuantizer::new(dim),
    )
}

/// Encode the reordered f32 values as `M`-subspace dot-product PQ storage.
pub fn pq_dp_cluster<const M: usize>(
    artifacts: &DenseBuildArtifacts,
) -> DenseDataset<ProductQuantizer<M, DotProduct>>
where
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
{
    let n = artifacts.reordered_dataset.len();
    let dim = artifacts.reordered_dataset.input_dim();
    let raw_dp = PlainDenseDataset::<f32, DotProduct>::from_raw(
        artifacts
            .reordered_dataset
            .values()
            .to_vec()
            .into_boxed_slice(),
        n,
        ScalarDenseQuantizer::new(dim),
    );
    ConvertFrom::convert_from(&raw_dp, ())
}

/// Per-vector L2 correction from the *stored* (reconstructed) cluster vector ŵ:
/// `‖ŵ‖²` for plain, `‖ŵ‖² + 2⟨c, ŵ⟩` for residual. Computing it from the stored
/// reconstruction (not the exact f32 input) keeps the reported score exactly
/// `‖q − (c + ŵ)‖²` for f16 / PQ storage too.
///
/// `centroids` **must already be at the centroid-index precision** (round to f16
/// first if the centroid index stores f16), so the `2⟨c, ŵ⟩` term matches the
/// `‖q − c‖²` the search reads from the centroid index.
pub fn l2_correction<D>(
    cluster: &D,
    centroids: &[f32],
    offsets: &[u32],
    residual: bool,
) -> Box<[f32]>
where
    D: Dataset,
    D::Encoder: DenseVectorEncoder,
{
    let dim = cluster.input_dim();
    let n_clusters = offsets.len() - 1;
    let mut corr = vec![0.0f32; cluster.len()].into_boxed_slice();
    for cid in 0..n_clusters {
        let c = &centroids[cid * dim..(cid + 1) * dim];
        for pos in offsets[cid] as usize..offsets[cid + 1] as usize {
            let w = cluster.encoder().decode_vector(cluster.get(pos as u64));
            corr[pos] = c
                .iter()
                .zip(w.values())
                .map(|(&cj, &wj)| wj * wj + if residual { 2.0 * cj * wj } else { 0.0 })
                .sum();
        }
    }
    corr
}

// ---------------------------------------------------------------------------
// vectorium Index trait (all configurations)
// ---------------------------------------------------------------------------

/// Build parameters for [`IVF::build_index`]. The centroid index itself is
/// supplied separately, via a `centroid_index_builder` closure passed to
/// `build_index` — see [`IVF`]'s inherent `build_index` impls.
///
/// Carries the k-means configuration (rebuilt into a [`KMeansBuilder`] at build
/// time so `build_index` can take `&self`) plus whether to store residuals.
pub struct IVFBuildParams {
    pub n_clusters: usize,
    pub kmeans_n_iter: usize,
    pub kmeans_n_redo: usize,
    pub kmeans_seed: Option<u64>,
    pub kmeans_spherical: bool,
    /// Store residuals `r = v − c` (`true`) or original vectors (`false`).
    pub residual: bool,
}

/// Run k-means + reorder for a `build_index`, from an f32 (SE-typed) input dataset.
fn prepare_ivf(input: &DensePlain, p: &IVFBuildParams) -> DenseBuildArtifacts {
    let kmeans = KMeansBuilder::new()
        .n_iter(p.kmeans_n_iter)
        .n_redo(p.kmeans_n_redo)
        .seed(p.kmeans_seed)
        .spherical(p.kmeans_spherical);
    prepare_dense_ivf(
        input,
        p.n_clusters,
        kmeans,
        p.residual,
        None,
        p.n_clusters >= KMEANS_HNSW_THRESHOLD,
    )
}

/// Reinterpret f32 dot-product values as an f32 squared-Euclidean input for k-means.
fn as_se_input(d: &DensePlainDP) -> DensePlain {
    DensePlain::from_raw(
        d.values().to_vec().into_boxed_slice(),
        d.len(),
        ScalarDenseQuantizer::new(d.input_dim()),
    )
}

/// Assemble a **squared-Euclidean**-reporting IVF: L2 correction from the stored
/// cluster, f32 centroids kept at SE.
fn assemble_l2<D, CI>(
    artifacts: DenseBuildArtifacts,
    cluster: D,
    centroid_index_builder: impl FnOnce(DensePlain) -> CI,
    residual: bool,
) -> IVF<DensePlain, D, CI>
where
    D: Dataset,
    D::Encoder: DenseVectorEncoder,
    CI: Index,
    for<'q> CI: Index<Query<'q> = DenseVectorView<'q, f32>>,
{
    let correction = l2_correction(
        &cluster,
        artifacts.centroids_f32.values(),
        &artifacts.cluster_offsets,
        residual,
    );
    let ci = centroid_index_builder(artifacts.centroids_f32);
    IVF::from_parts(
        ci,
        cluster,
        artifacts.cluster_offsets,
        artifacts.external_ids,
        Some(correction),
        residual,
    )
}

/// Assemble a **dot-product**-reporting IVF: no correction, centroids reinterpreted as DP.
fn assemble_ip<D, CI>(
    artifacts: DenseBuildArtifacts,
    cluster: D,
    centroid_index_builder: impl FnOnce(DensePlainDP) -> CI,
    residual: bool,
) -> IVF<DensePlainDP, D, CI>
where
    D: Dataset,
    CI: Index,
    for<'q> CI: Index<Query<'q> = DenseVectorView<'q, f32>>,
{
    let centroids_dp = DensePlainDP::from_raw(
        artifacts.centroids_f32.values().to_vec().into_boxed_slice(),
        artifacts.centroids_f32.len(),
        ScalarDenseQuantizer::new(artifacts.centroids_f32.input_dim()),
    );
    let ci = centroid_index_builder(centroids_dp);
    IVF::from_parts(
        ci,
        cluster,
        artifacts.cluster_offsets,
        artifacts.external_ids,
        None,
        residual,
    )
}

impl<DQ, D, CI> vectorium::IndexStats for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
{
    fn n_elements(&self) -> usize {
        self.external_ids.len()
    }
    fn dim(&self) -> usize {
        self.dataset.input_dim()
    }
}

/// A single `Index` impl covering every IVF configuration: the bounds are
/// exactly those the inherent `search` above already needs (`D` generically,
/// via `D::Encoder: VectorEncoder<Distance = DotProduct>`). Unlike
/// `build_index`, `search` never needs to name a concrete encoder like
/// `ScalarDenseQuantizer` or `ProductQuantizer` — that specificity is only
/// needed at build time.
impl<DQ, D, CI> Index for IVF<DQ, D, CI>
where
    DQ: Dataset,
    D: Dataset,
    CI: Index,
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    D::Encoder: VectorEncoder<Distance = DotProduct>,
    for<'q> <DQ::Encoder as VectorEncoder>::QueryVector<'q>:
        Into<<D::Encoder as VectorEncoder>::QueryVector<'q>> + Copy,
    for<'q, 'b> <D::Encoder as VectorEncoder>::Evaluator<'q>:
        QueryEvaluator<<D::Encoder as VectorEncoder>::EncodedVector<'b>, Distance = DotProduct>,
{
    type Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>;
    type Distance = <DQ::Encoder as VectorEncoder>::Distance;
    type SearchParams = IVFSearchParams<CI, <D::Encoder as VectorEncoder>::QueryParams>;

    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<Self::Distance>> {
        // Inherent `search` takes priority, so this is not recursive.
        self.search(query, k, search_params)
    }
}

// ---------------------------------------------------------------------------
// build_index: one impl, dispatched over two independent axes
// ---------------------------------------------------------------------------
//
// `build_index` is always the same three steps — prepare (k-means), encode
// (cluster storage), assemble (final index) — regardless of which reported
// metric or cluster storage is used. [`BuildMetric`] and [`ClusterEncoding`]
// each capture one axis, implemented once per concrete option; `build_index`
// itself is written once, generic over both.

/// The reported-metric axis: how a query/centroid dataset `DQ` maps onto
/// build-time k-means, which internally is always f32/L2. One impl each for
/// [`DensePlain`] (metric = SE, no transform needed) and [`DensePlainDP`]
/// (metric = dot product, reinterpreted to/from SE via [`as_se_input`]).
pub(crate) trait BuildMetric: Dataset + Sized {
    /// Run k-means + reorder, converting to k-means' native f32/L2 shape first if needed.
    fn prepare(&self, p: &IVFBuildParams) -> DenseBuildArtifacts;

    /// Assemble the final IVF, given the already-encoded cluster storage and
    /// a way to build the centroid index from the post-k-means centroids.
    fn assemble<D, CI>(
        artifacts: DenseBuildArtifacts,
        cluster: D,
        centroid_index_builder: impl FnOnce(Self) -> CI,
        residual: bool,
    ) -> IVF<Self, D, CI>
    where
        D: Dataset,
        D::Encoder: DenseVectorEncoder,
        CI: Index,
        for<'q> CI: Index<Query<'q> = DenseVectorView<'q, f32>>;
}

impl BuildMetric for DensePlain {
    fn prepare(&self, p: &IVFBuildParams) -> DenseBuildArtifacts {
        prepare_ivf(self, p)
    }

    fn assemble<D, CI>(
        artifacts: DenseBuildArtifacts,
        cluster: D,
        centroid_index_builder: impl FnOnce(Self) -> CI,
        residual: bool,
    ) -> IVF<Self, D, CI>
    where
        D: Dataset,
        D::Encoder: DenseVectorEncoder,
        CI: Index,
        for<'q> CI: Index<Query<'q> = DenseVectorView<'q, f32>>,
    {
        assemble_l2(artifacts, cluster, centroid_index_builder, residual)
    }
}

impl BuildMetric for DensePlainDP {
    fn prepare(&self, p: &IVFBuildParams) -> DenseBuildArtifacts {
        prepare_ivf(&as_se_input(self), p)
    }

    fn assemble<D, CI>(
        artifacts: DenseBuildArtifacts,
        cluster: D,
        centroid_index_builder: impl FnOnce(Self) -> CI,
        residual: bool,
    ) -> IVF<Self, D, CI>
    where
        D: Dataset,
        D::Encoder: DenseVectorEncoder,
        CI: Index,
        for<'q> CI: Index<Query<'q> = DenseVectorView<'q, f32>>,
    {
        assemble_ip(artifacts, cluster, centroid_index_builder, residual)
    }
}

/// The cluster-storage axis: how k-means artifacts become the vectors an IVF
/// actually scans. One impl each for plain `V`-precision storage and
/// `M`-subspace PQ storage.
pub(crate) trait ClusterEncoding: Dataset {
    fn encode(artifacts: &DenseBuildArtifacts) -> Self;
}

impl<V: Float + FromF32 + ValueType> ClusterEncoding for PlainDenseDataset<V, DotProduct> {
    fn encode(artifacts: &DenseBuildArtifacts) -> Self {
        plain_dp_cluster::<V>(artifacts)
    }
}

impl<const M: usize> ClusterEncoding for DenseDataset<ProductQuantizer<M, DotProduct>>
where
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
{
    fn encode(artifacts: &DenseBuildArtifacts) -> Self {
        pq_dp_cluster::<M>(artifacts)
    }
}

// `BuildMetric`/`ClusterEncoding` are deliberately `pub(crate)`, sealing which
// `DQ`/`D` combinations `build_index` accepts to the ones implemented above.
#[allow(private_bounds)]
impl<DQ, D, CI> IVF<DQ, D, CI>
where
    DQ: BuildMetric,
    D: ClusterEncoding,
    D::Encoder: DenseVectorEncoder,
    CI: Index,
    for<'q> CI: Index<Query<'q> = DenseVectorView<'q, f32>>,
{
    /// Build an IVF. `centroid_index_builder` builds the centroid index from the
    /// post-k-means centroids — how it does so is entirely up to the caller.
    pub fn build_index(
        dataset: DQ,
        p: &IVFBuildParams,
        centroid_index_builder: impl FnOnce(DQ) -> CI,
    ) -> Self {
        let artifacts = dataset.prepare(p);
        let cluster = D::encode(&artifacts);
        DQ::assemble(artifacts, cluster, centroid_index_builder, p.residual)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Sign / formula checks for the reported-metric reconstruction. The cluster
    // dataset always yields a raw dot product `raw = ⟨q,w⟩`.

    #[test]
    fn dot_product_combine_sums_centroid_and_raw() {
        // plain:    ⟨q,v⟩            (centroid 0)
        // residual: ⟨q,c⟩ + ⟨q,r⟩   (centroid = ⟨q,c⟩)
        assert_eq!(DotProduct::combine(0.0, 0.0, 3.0), DotProduct::from(3.0));
        assert_eq!(DotProduct::combine(2.0, 0.0, 3.0), DotProduct::from(5.0));
        const { assert!(!DotProduct::USES_CORRECTION) };
    }

    #[test]
    fn squared_euclidean_combine_applies_correction_and_minus_two() {
        // plain:    ‖v‖² − 2⟨q,v⟩          (centroid 0, correction ‖v‖²)
        //           = 10 − 2·3 = 4
        assert_eq!(
            SquaredEuclideanDistance::combine(0.0, 10.0, 3.0),
            SquaredEuclideanDistance::from(4.0)
        );
        // residual: ‖q−c‖² + (‖r̂‖²+2⟨c,r̂⟩) − 2⟨q,r̂⟩
        //           = 4 + 10 − 2·3 = 8
        assert_eq!(
            SquaredEuclideanDistance::combine(4.0, 10.0, 3.0),
            SquaredEuclideanDistance::from(8.0)
        );
        const { assert!(SquaredEuclideanDistance::USES_CORRECTION) };
    }

    #[test]
    fn ivf_is_a_vectorium_index() {
        use vectorium::core::vector::DenseVectorView;
        use vectorium::{FlatIndex, IndexStats};

        // Two well-separated clusters of 2-D points.
        let dim = 2;
        let enc = ScalarDenseQuantizer::<f32, f32, SquaredEuclideanDistance>::new(dim);
        let mut g = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(enc);
        let pts: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
        ];
        for p in &pts {
            g.push(DenseVectorView::new(p));
        }
        let dataset: DensePlain = g.into();

        type Ivf = IVF<DensePlain, DensePlainDP, FlatIndex<DensePlain>>;

        // Exercise both plain and residual storage through the same Index interface.
        for residual in [false, true] {
            let params = IVFBuildParams {
                n_clusters: 2,
                kmeans_n_iter: 10,
                kmeans_n_redo: 1,
                kmeans_seed: Some(42),
                kmeans_spherical: false,
                residual,
            };
            let index = Ivf::build_index(dataset.clone(), &params, FlatIndex::from);

            assert_eq!(IndexStats::n_elements(&index), 6);
            assert_eq!(IndexStats::dim(&index), 2);
            assert_eq!(index.is_residual(), residual);

            let sp = IVFSearchParams {
                nprobe: 2, // probe both clusters => exact
                centroid_search_params: (),
                cluster_query_params: (),
            };
            let query = DenseVectorView::new(&[10.05f32, 10.0]);
            let res = <Ivf as Index>::search(&index, query, 1, &sp);

            assert_eq!(res.len(), 1);
            // Nearest to (10.05, 10.0) is the original vector id 3 == (10.0, 10.0).
            assert_eq!(res[0].vector, 3, "residual={residual}");
        }
    }

    /// The generic `Index` impls cover **all** configs, not just f32/L2: build an
    /// f16-cluster L2 index and a dot-product index entirely through the trait.
    #[test]
    fn ivf_index_covers_f16_and_dot_product() {
        use half::f16;
        use vectorium::FlatIndex;
        use vectorium::core::vector::DenseVectorView;

        let dim = 2;
        let pts: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 9.9],
            [9.9, 10.2],
        ];
        let mut g = PlainDenseDatasetGrowable::<f32, SquaredEuclideanDistance>::new(
            ScalarDenseQuantizer::new(dim),
        );
        for p in &pts {
            g.push(DenseVectorView::new(p));
        }
        let dataset: DensePlain = g.into();

        let params = IVFBuildParams {
            n_clusters: 2,
            kmeans_n_iter: 10,
            kmeans_n_redo: 1,
            kmeans_seed: Some(42),
            kmeans_spherical: false,
            residual: false,
        };

        // (a) f16 cluster storage, L2 metric.
        type IvfF16 = IVF<DensePlain, PlainDenseDataset<f16, DotProduct>, FlatIndex<DensePlain>>;
        let idx = IvfF16::build_index(dataset.clone(), &params, FlatIndex::from);
        let sp = IVFSearchParams {
            nprobe: 2,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        let res = <IvfF16 as Index>::search(&idx, DenseVectorView::new(&[10.05f32, 10.0]), 1, &sp);
        assert_eq!(res[0].vector, 3); // L2 nearest

        // (b) dot-product metric.
        type IvfDp = IVF<DensePlainDP, DensePlainDP, FlatIndex<DensePlainDP>>;
        let ds_dp = DensePlainDP::from_raw(
            dataset.values().to_vec().into_boxed_slice(),
            dataset.len(),
            ScalarDenseQuantizer::new(dim),
        );
        let idx = IvfDp::build_index(ds_dp, &params, FlatIndex::from);
        let sp = IVFSearchParams {
            nprobe: 2,
            centroid_search_params: (),
            cluster_query_params: (),
        };
        let res = <IvfDp as Index>::search(&idx, DenseVectorView::new(&[10.05f32, 10.0]), 1, &sp);
        // Max ⟨q, v⟩ is id 5 = (9.9, 10.2): 10.05·9.9 + 10·10.2 = 201.5.
        assert_eq!(res[0].vector, 5);
    }
}
