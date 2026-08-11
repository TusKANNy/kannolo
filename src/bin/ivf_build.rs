/// Build an IVF index.
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use serde::Serialize;

use kannolo::graph::Graph;
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use kannolo::ivf::{
    DenseBuildArtifacts, IVF, KMEANS_HNSW_THRESHOLD, l2_correction, plain_dp_cluster,
    pq_dp_cluster, prepare_dense_ivf,
};
use vectorium::core::dataset::ConvertFrom as VConvertFrom;
use vectorium::core::index::Index;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::PlainDenseQuantizer;
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::readers::read_npy_f32;
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::{
    Dataset, DenseDataset, Float, FromF32, KMeansBuilder, PlainDenseDataset, ValueType,
};

/// Centroid + cluster datasets share the same metric (--distance).
#[allow(non_camel_case_types)]
type DQ_SE = PlainDenseDataset<f16, SquaredEuclideanDistance>;
#[allow(non_camel_case_types)]
type DQ_DP = PlainDenseDataset<f16, DotProduct>;
/// Raw f32 data from k-means (always L2 internally).
type DRaw = PlainDenseDataset<f32, SquaredEuclideanDistance>;

const PQ_SUPPORTED_SUBSPACES: [usize; 9] = [4, 8, 16, 32, 48, 64, 96, 128, 192];

#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    #[default]
    F32,
    F16,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum MetricArg {
    #[default]
    Euclidean,
    Dotproduct,
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[clap(short, long)]
    data_file: String,
    #[clap(short, long)]
    output_file: String,
    #[clap(long, value_enum, default_value_t = MetricArg::Euclidean)]
    distance: MetricArg,
    #[clap(long, value_enum, default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,
    #[clap(long, default_value_t = 1024)]
    n_clusters: usize,
    #[clap(long, default_value_t = 25)]
    kmeans_n_iter: usize,
    #[clap(long, default_value_t = 1)]
    kmeans_n_redo: usize,
    #[clap(long)]
    kmeans_sample_size: Option<usize>,
    #[clap(long, default_value_t = false)]
    residuals: bool,
    #[clap(long, default_value_t = false)]
    hnsw: bool,
    #[clap(long, alias = "m", default_value_t = 32)]
    m_hnsw: usize,
    #[clap(long, default_value_t = 200)]
    ef_construction: usize,
    #[clap(long)]
    m_pq: Option<usize>,
    /// Use HNSW for centroid assignment during k-means (default: auto when n_clusters >= 32768).
    #[clap(long, default_value_t = false)]
    kmeans_hnsw: bool,
    /// Spherical k-means: L2-normalize centroids after each update (required for inner-product metrics).
    #[clap(long, default_value_t = false)]
    kmeans_spherical: bool,
}

fn main() {
    let args = Args::parse();
    if matches!(args.value_type, ValueTypeArg::F16) && args.m_pq.is_some() {
        eprintln!("Note: --value-type f16 is ignored with --m-pq.");
    }
    if args.residuals && args.m_pq.is_none() {
        eprintln!("Warning: --residuals gives no benefit without --m-pq.");
    }
    if let Some(m) = args.m_pq
        && !PQ_SUPPORTED_SUBSPACES.contains(&m)
    {
        eprintln!(
            "Error: unsupported --m-pq {m}. Supported: {:?}",
            PQ_SUPPORTED_SUBSPACES
        );
        process::exit(1);
    }

    let raw: DRaw = read_npy_f32(&args.data_file).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {e:?}", args.data_file);
        process::exit(1);
    });
    let dim = raw.input_dim();

    let kmeans_hnsw = args.kmeans_hnsw || args.n_clusters >= KMEANS_HNSW_THRESHOLD;
    println!(
        "Running k-means: n_clusters={} n_iter={} n_redo={} hnsw_assignment={}",
        args.n_clusters, args.kmeans_n_iter, args.kmeans_n_redo, kmeans_hnsw
    );
    let build_start = Instant::now();
    let artifacts = prepare_dense_ivf(
        &raw,
        args.n_clusters,
        KMeansBuilder::new()
            .n_iter(args.kmeans_n_iter)
            .n_redo(args.kmeans_n_redo)
            .verbose(true)
            .spherical(args.kmeans_spherical),
        args.residuals,
        args.kmeans_sample_size,
        kmeans_hnsw,
    );
    println!(
        "K-means + reorder: {:.1}s",
        build_start.elapsed().as_secs_f32()
    );

    let n_centroids = artifacts.centroids_f32.len();
    let centroid_f16: Vec<f16> = artifacts
        .centroids_f32
        .values()
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();

    let t1 = Instant::now();
    match (&args.distance, args.hnsw) {
        (MetricArg::Euclidean, false) => {
            let dq = DQ_SE::from_raw(
                centroid_f16.into_boxed_slice(),
                n_centroids,
                PlainDenseQuantizer::new(dim),
            );
            let q: vectorium::FlatIndex<DQ_SE> = vectorium::FlatIndex::from(dq);
            println!(
                "Centroid FlatIndex/SE built in {:.1}s",
                t1.elapsed().as_secs_f32()
            );
            dispatch_se(q, artifacts, &args, build_start);
        }
        (MetricArg::Euclidean, true) => {
            let dq = DQ_SE::from_raw(
                centroid_f16.into_boxed_slice(),
                n_centroids,
                PlainDenseQuantizer::new(dim),
            );
            let cfg = HNSWBuildConfiguration::default()
                .with_num_neighbors(args.m_hnsw)
                .with_ef_construction(args.ef_construction);
            let q: HNSW<DQ_SE, Graph> = HNSW::build_index(dq, &cfg);
            println!(
                "Centroid HNSW/SE built in {:.1}s",
                t1.elapsed().as_secs_f32()
            );
            dispatch_se(q, artifacts, &args, build_start);
        }
        (MetricArg::Dotproduct, false) => {
            let dq = DQ_DP::from_raw(
                centroid_f16.into_boxed_slice(),
                n_centroids,
                PlainDenseQuantizer::new(dim),
            );
            let q: vectorium::FlatIndex<DQ_DP> = vectorium::FlatIndex::from(dq);
            println!(
                "Centroid FlatIndex/DP built in {:.1}s",
                t1.elapsed().as_secs_f32()
            );
            dispatch_dp(q, artifacts, &args, build_start);
        }
        (MetricArg::Dotproduct, true) => {
            let dq = DQ_DP::from_raw(
                centroid_f16.into_boxed_slice(),
                n_centroids,
                PlainDenseQuantizer::new(dim),
            );
            let cfg = HNSWBuildConfiguration::default()
                .with_num_neighbors(args.m_hnsw)
                .with_ef_construction(args.ef_construction);
            let q: HNSW<DQ_DP, Graph> = HNSW::build_index(dq, &cfg);
            println!(
                "Centroid HNSW/DP built in {:.1}s",
                t1.elapsed().as_secs_f32()
            );
            dispatch_dp(q, artifacts, &args, build_start);
        }
    }
}

// ---------------------------------------------------------------------------
// SE cluster dispatch
// ---------------------------------------------------------------------------

fn dispatch_se<CI: Index + Serialize>(
    q: CI,
    a: DenseBuildArtifacts,
    args: &Args,
    build_start: Instant,
) {
    // `residual` is carried at runtime, so it is not a dispatch dimension.
    match (args.m_pq, &args.value_type) {
        (None, ValueTypeArg::F32) => save_plain_se::<f32, CI>(q, a, args, build_start),
        (None, ValueTypeArg::F16) => save_plain_se::<f16, CI>(q, a, args, build_start),
        (Some(4), _) => save_pq_se::<4, CI>(q, a, args, build_start),
        (Some(8), _) => save_pq_se::<8, CI>(q, a, args, build_start),
        (Some(16), _) => save_pq_se::<16, CI>(q, a, args, build_start),
        (Some(32), _) => save_pq_se::<32, CI>(q, a, args, build_start),
        (Some(48), _) => save_pq_se::<48, CI>(q, a, args, build_start),
        (Some(64), _) => save_pq_se::<64, CI>(q, a, args, build_start),
        (Some(96), _) => save_pq_se::<96, CI>(q, a, args, build_start),
        (Some(128), _) => save_pq_se::<128, CI>(q, a, args, build_start),
        (Some(192), _) => save_pq_se::<192, CI>(q, a, args, build_start),
        (Some(m), _) => unreachable!("m_pq={m}"),
    }
}

// ---------------------------------------------------------------------------
// DP cluster dispatch
// ---------------------------------------------------------------------------

fn dispatch_dp<CI: Index + Serialize>(
    q: CI,
    a: DenseBuildArtifacts,
    args: &Args,
    build_start: Instant,
) {
    match (args.m_pq, &args.value_type) {
        (None, ValueTypeArg::F32) => save_plain_dp::<f32, CI>(q, a, args, build_start),
        (None, ValueTypeArg::F16) => save_plain_dp::<f16, CI>(q, a, args, build_start),
        (Some(4), _) => save_pq_dp::<4, CI>(q, a, args, build_start),
        (Some(8), _) => save_pq_dp::<8, CI>(q, a, args, build_start),
        (Some(16), _) => save_pq_dp::<16, CI>(q, a, args, build_start),
        (Some(32), _) => save_pq_dp::<32, CI>(q, a, args, build_start),
        (Some(48), _) => save_pq_dp::<48, CI>(q, a, args, build_start),
        (Some(64), _) => save_pq_dp::<64, CI>(q, a, args, build_start),
        (Some(96), _) => save_pq_dp::<96, CI>(q, a, args, build_start),
        (Some(128), _) => save_pq_dp::<128, CI>(q, a, args, build_start),
        (Some(192), _) => save_pq_dp::<192, CI>(q, a, args, build_start),
        (Some(m), _) => unreachable!("m_pq={m}"),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn save<I: vectorium::IndexSerializer + serde::Serialize>(index: &I, path: &str) {
    index.save_index(path).unwrap_or_else(|e| {
        eprintln!("{e:?}");
        process::exit(1);
    });
    println!("Saved → {path}");
}

/// Assemble the final IVF from parts and serialise it. `residual` and
/// `correction_terms` are runtime, so one helper covers every configuration.
///
/// The parts are passed individually rather than bundled: each comes from a different
/// stage of the build and bundling them would only move the argument list into a struct
/// used exactly once.
#[allow(clippy::too_many_arguments)]
fn assemble_and_save<DQ, DC, CI>(
    q: CI,
    cluster: DC,
    a_offsets: Box<[u32]>,
    a_ids: Box<[u32]>,
    correction_terms: Option<Box<[f32]>>,
    residual: bool,
    output: &str,
    build_start: Instant,
) where
    DQ: Dataset,
    DC: Dataset + serde::Serialize + for<'de> serde::Deserialize<'de>,
    CI: Index + Serialize,
{
    let t = Instant::now();
    let index: IVF<DQ, DC, CI> =
        IVF::from_parts(q, cluster, a_offsets, a_ids, correction_terms, residual);
    println!("Assembled in {:.1}s", t.elapsed().as_secs_f32());
    println!(
        "Time to build: {} s (before serializing)",
        build_start.elapsed().as_secs()
    );
    save(&index, output);
}

/// f16-round centroids to match the stored (f16) centroid-index precision, so the
/// `2⟨c, ŵ⟩` term in [`l2_correction`] matches the `‖q − c‖²` the search reads.
fn centroids_f16(a: &DenseBuildArtifacts) -> Vec<f32> {
    a.centroids_f32
        .values()
        .iter()
        .map(|&c| f16::from_f32(c).to_f32())
        .collect()
}

fn save_plain_se<
    V: Float + FromF32 + ValueType + serde::Serialize + for<'de> serde::Deserialize<'de>,
    CI: Index + Serialize,
>(
    q: CI,
    a: DenseBuildArtifacts,
    args: &Args,
    build_start: Instant,
) where
    PlainDenseDataset<V, DotProduct>: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let cluster = plain_dp_cluster::<V>(&a);
    let correction = l2_correction(
        &cluster,
        &centroids_f16(&a),
        &a.cluster_offsets,
        args.residuals,
    );
    assemble_and_save::<DQ_SE, _, CI>(
        q,
        cluster,
        a.cluster_offsets,
        a.external_ids,
        Some(correction),
        args.residuals,
        &args.output_file,
        build_start,
    );
}

fn save_plain_dp<
    V: Float + FromF32 + ValueType + serde::Serialize + for<'de> serde::Deserialize<'de>,
    CI: Index + Serialize,
>(
    q: CI,
    a: DenseBuildArtifacts,
    args: &Args,
    build_start: Instant,
) where
    PlainDenseDataset<V, DotProduct>: serde::Serialize + for<'de> serde::Deserialize<'de>,
{
    let cluster = plain_dp_cluster::<V>(&a);
    assemble_and_save::<DQ_DP, _, CI>(
        q,
        cluster,
        a.cluster_offsets,
        a.external_ids,
        None,
        args.residuals,
        &args.output_file,
        build_start,
    );
}

fn save_pq_se<const M: usize, CI: Index + Serialize>(
    q: CI,
    a: DenseBuildArtifacts,
    args: &Args,
    build_start: Instant,
) where
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>> + Serialize,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        VConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>
        + VectorEncoder<Distance = DotProduct>,
{
    let cluster = pq_dp_cluster::<M>(&a);
    let correction = l2_correction(
        &cluster,
        &centroids_f16(&a),
        &a.cluster_offsets,
        args.residuals,
    );
    assemble_and_save::<DQ_SE, _, CI>(
        q,
        cluster,
        a.cluster_offsets,
        a.external_ids,
        Some(correction),
        args.residuals,
        &args.output_file,
        build_start,
    );
}

fn save_pq_dp<const M: usize, CI: Index + Serialize>(
    q: CI,
    a: DenseBuildArtifacts,
    args: &Args,
    build_start: Instant,
) where
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>> + Serialize,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        VConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>
        + VectorEncoder<Distance = DotProduct>,
{
    let cluster = pq_dp_cluster::<M>(&a);
    assemble_and_save::<DQ_DP, _, CI>(
        q,
        cluster,
        a.cluster_offsets,
        a.external_ids,
        None,
        args.residuals,
        &args.output_file,
        build_start,
    );
}
