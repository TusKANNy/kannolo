use std::{fmt::Debug, time::Instant};

use clap::{Parser, ValueEnum};
use half::f16;
use serde::{Deserialize, Serialize};
use std::process;

use kannolo::graph::graph::{Graph, GraphFixedDegree, GrowableGraph};
use kannolo::graph::neighbors::{
    MAX_NEIGHBORS_PER_NODE, NeighborData, Neighbors, PlainNeighbors, StreamVByteNeighbors,
};
use kannolo::hnsw::{HNSW, HNSWBuildConfiguration};
use vectorium::IndexSerializer;
use vectorium::dataset::ConvertFrom;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::{
    Dataset, DenseDataset, FixedU8Q, FixedU16Q, Float, FromF32, PackedSparseDataset,
    PlainDenseDataset, PlainSparseDataset, ScalarSparseDataset, SpaceUsage, ValueType,
};

#[derive(Debug, Clone, ValueEnum)]
enum DatasetType {
    Dense,
    Sparse,
}

/// Value type for stored values.
/// Dense plain: `f32`, `f16`.
/// Sparse plain: `f32`, `f16`, `fixedu8`, `fixedu16`.
/// Ignored for `dotvbyte` and `pq` (they fix their own value representation).
#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    F16,
    #[default]
    F32,
    Fixedu8,
    Fixedu16,
}

impl std::fmt::Display for ValueTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueTypeArg::F16 => write!(f, "f16"),
            ValueTypeArg::F32 => write!(f, "f32"),
            ValueTypeArg::Fixedu8 => write!(f, "fixedu8"),
            ValueTypeArg::Fixedu16 => write!(f, "fixedu16"),
        }
    }
}

/// Encoder type.
/// Dense: `plain`, `pq`.
/// Sparse: `plain`, `dotvbyte`.
#[derive(Debug, Clone, ValueEnum, Default)]
enum EncoderType {
    #[default]
    Plain,
    Pq,
    Dotvbyte,
}

impl std::fmt::Display for EncoderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncoderType::Plain => write!(f, "plain"),
            EncoderType::Pq => write!(f, "pq"),
            EncoderType::Dotvbyte => write!(f, "dotvbyte"),
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum GraphType {
    /// Plain neighbor storage, original (insertion) node order.
    Standard,
    /// Fixed-degree (padded) neighbor storage, original node order.
    FixedDegree,
    /// Plain neighbor storage, nodes reordered by Enhanced Graph Bisection (EGB).
    Permuted,
    /// EGB-reordered nodes, neighbor lists delta-encoded and compressed with StreamVByte.
    Streamvbyte,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ComponentTypeArg {
    #[default]
    U16,
    U32,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the dataset file.
    #[clap(short, long, value_parser)]
    data_file: String,

    /// The output file where to save the index.
    #[clap(short, long, value_parser)]
    output_file: String,

    /// The type of vectors (dense or sparse).
    #[clap(long, value_enum)]
    dataset_type: DatasetType,

    /// Value type for stored values. Dense plain: f32, f16. Sparse plain: f32, f16, fixedu8, fixedu16.
    /// Ignored for dotvbyte and pq (they determine their own value representation).
    #[clap(long = "value-type", value_enum)]
    #[arg(default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,

    /// Component type for sparse datasets (`u16` or `u32`).
    /// DotVByte currently supports only `u16`.
    #[clap(long = "component-type", value_enum)]
    #[arg(default_value_t = ComponentTypeArg::U16)]
    component_type: ComponentTypeArg,

    /// Encoder type. Dense: plain, pq. Sparse: plain, dotvbyte.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EncoderType::Plain)]
    encoder: EncoderType,

    /// The graph type: `standard`, `fixed-degree`, `permuted` (EGB-reordered, uncompressed),
    /// or `streamvbyte` (EGB-reordered, delta+StreamVByte compressed).
    #[clap(long, value_enum)]
    #[arg(default_value_t = GraphType::Standard)]
    graph_type: GraphType,

    /// The number of neighbors per node.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    /// The size of the candidate pool at construction time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    ef_construction: usize,

    /// The type of distance to use. Either 'euclidean' or 'dotproduct'.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("dotproduct"))]
    distance: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    pq_subspaces: usize,

    /// The number of bits per subspace for Product Quantization (ignored; vectorium PQ is fixed).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 8)]
    nbits: usize,

    /// The size of the sample used for training Product Quantization (ignored; vectorium PQ samples automatically).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100_000)]
    sample_size: usize,
}

#[derive(Clone, Copy)]
enum Distance {
    Euclidean,
    DotProduct,
}

fn parse_metric(metric: &str) -> Distance {
    match metric {
        "euclidean" | "l2" => Distance::Euclidean,
        "dotproduct" | "ip" => Distance::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'.");
            process::exit(1);
        }
    }
}

const PQ_SUPPORTED_SUBSPACES: [usize; 9] = [128, 192, 96, 64, 48, 32, 16, 8, 4];

fn choose_pq_subspaces(dim: usize, requested: usize) -> usize {
    if requested == 0 {
        for &m in &PQ_SUPPORTED_SUBSPACES {
            if dim.is_multiple_of(m) {
                return m;
            }
        }
        eprintln!(
            "Error: Could not auto-select pq-subspaces for dimension {dim}. Supported: 4, 8, 16, 32, 48, 64, 96, 128, 192."
        );
        process::exit(1);
    }

    if !PQ_SUPPORTED_SUBSPACES.contains(&requested) {
        eprintln!(
            "Error: Unsupported pq-subspaces value {requested}. Supported: 4, 8, 16, 32, 48, 64, 96, 128, 192."
        );
        process::exit(1);
    }

    if !dim.is_multiple_of(requested) {
        eprintln!("Error: pq-subspaces ({requested}) must divide vector dimension ({dim}).");
        process::exit(1);
    }

    requested
}

fn main() {
    let args: Args = Args::parse();

    // Cross-validation of encoder / dataset-type combinations
    match (&args.dataset_type, &args.encoder) {
        (DatasetType::Sparse, EncoderType::Pq) => {
            eprintln!("Error: PQ encoder is only available for dense vectors.");
            process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Dotvbyte) => {
            eprintln!("Error: DotVByte encoder is only available for sparse vectors.");
            process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Plain)
            if matches!(
                args.value_type,
                ValueTypeArg::Fixedu8 | ValueTypeArg::Fixedu16
            ) =>
        {
            eprintln!("Error: fixedu8/fixedu16 value types are only available for sparse vectors.");
            process::exit(1);
        }
        (DatasetType::Dense, _) if !matches!(args.component_type, ComponentTypeArg::U16) => {
            eprintln!("Error: component-type is only applicable to sparse datasets.");
            process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte)
            if !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!("Error: DotVByte encoder supports only component-type u16.");
            process::exit(1);
        }
        _ => {}
    }

    // Checked here rather than in `StreamVByteNeighbors::from`, which only runs once the whole
    // index (and the EGB permutation) has already been built.
    if matches!(args.graph_type, GraphType::Streamvbyte) && 2 * args.m > MAX_NEIGHBORS_PER_NODE {
        eprintln!(
            "Error: --graph-type streamvbyte caps the ground level at {MAX_NEIGHBORS_PER_NODE} neighbors per node, but --m {} gives 2 * m = {}.",
            args.m,
            2 * args.m
        );
        process::exit(1);
    }

    if matches!(args.encoder, EncoderType::Pq) {
        if args.nbits != 8 {
            eprintln!("Warning: vectorium PQ ignores --nbits (fixed codebook size).");
        }
        if args.sample_size != 100_000 {
            eprintln!("Warning: vectorium PQ ignores --sample-size and uses automatic sampling.");
        }
    }

    let distance = parse_metric(&args.distance);

    let config = HNSWBuildConfiguration::default()
        .with_num_neighbors(args.m)
        .with_ef_construction(args.ef_construction);

    println!(
        "Building Index with M: {}, ef_construction: {}",
        args.m, args.ef_construction
    );

    match (
        &args.dataset_type,
        &args.encoder,
        &args.value_type,
        &args.graph_type,
    ) {
        // Dense plain f32
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Standard) => {
            build_dense_plain::<f32, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::FixedDegree) => {
            build_dense_plain::<f32, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Permuted) => {
            build_dense_plain_permuted::<f32, PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Streamvbyte) => {
            build_dense_plain_permuted::<f32, StreamVByteNeighbors>(&args, distance, &config);
        }
        // Dense plain f16
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Standard) => {
            build_dense_plain::<f16, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::FixedDegree) => {
            build_dense_plain::<f16, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Permuted) => {
            build_dense_plain_permuted::<f16, PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Streamvbyte) => {
            build_dense_plain_permuted::<f16, StreamVByteNeighbors>(&args, distance, &config);
        }
        // Dense PQ (value-type ignored)
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Standard) => {
            build_dense_pq::<Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::FixedDegree) => {
            build_dense_pq::<GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Permuted) => {
            build_dense_pq_permuted::<PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Streamvbyte) => {
            build_dense_pq_permuted::<StreamVByteNeighbors>(&args, distance, &config);
        }
        // Sparse plain f32
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32, GraphType::Standard) => {
            build_sparse_plain::<f32, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32, GraphType::FixedDegree) => {
            build_sparse_plain::<f32, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32, GraphType::Permuted) => {
            build_sparse_plain_permuted::<f32, PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32, GraphType::Streamvbyte) => {
            build_sparse_plain_permuted::<f32, StreamVByteNeighbors>(&args, distance, &config);
        }
        // Sparse plain f16
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16, GraphType::Standard) => {
            build_sparse_plain::<f16, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16, GraphType::FixedDegree) => {
            build_sparse_plain::<f16, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16, GraphType::Permuted) => {
            build_sparse_plain_permuted::<f16, PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16, GraphType::Streamvbyte) => {
            build_sparse_plain_permuted::<f16, StreamVByteNeighbors>(&args, distance, &config);
        }
        // Sparse plain fixedu8
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu8, GraphType::Standard) => {
            build_sparse_scalar::<FixedU8Q, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (
            DatasetType::Sparse,
            EncoderType::Plain,
            ValueTypeArg::Fixedu8,
            GraphType::FixedDegree,
        ) => {
            build_sparse_scalar::<FixedU8Q, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu8, GraphType::Permuted) => {
            build_sparse_scalar_permuted::<FixedU8Q, PlainNeighbors>(&args, distance, &config);
        }
        (
            DatasetType::Sparse,
            EncoderType::Plain,
            ValueTypeArg::Fixedu8,
            GraphType::Streamvbyte,
        ) => {
            build_sparse_scalar_permuted::<FixedU8Q, StreamVByteNeighbors>(
                &args, distance, &config,
            );
        }
        // Sparse plain fixedu16
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu16, GraphType::Standard) => {
            build_sparse_scalar::<FixedU16Q, Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (
            DatasetType::Sparse,
            EncoderType::Plain,
            ValueTypeArg::Fixedu16,
            GraphType::FixedDegree,
        ) => {
            build_sparse_scalar::<FixedU16Q, GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu16, GraphType::Permuted) => {
            build_sparse_scalar_permuted::<FixedU16Q, PlainNeighbors>(&args, distance, &config);
        }
        (
            DatasetType::Sparse,
            EncoderType::Plain,
            ValueTypeArg::Fixedu16,
            GraphType::Streamvbyte,
        ) => {
            build_sparse_scalar_permuted::<FixedU16Q, StreamVByteNeighbors>(
                &args, distance, &config,
            );
        }
        // Sparse dotvbyte (value-type ignored)
        (DatasetType::Sparse, EncoderType::Dotvbyte, _, GraphType::Standard) => {
            build_sparse_dotvbyte::<Graph<PlainNeighbors>>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte, _, GraphType::FixedDegree) => {
            build_sparse_dotvbyte::<GraphFixedDegree>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte, _, GraphType::Permuted) => {
            build_sparse_dotvbyte_permuted::<PlainNeighbors>(&args, distance, &config);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte, _, GraphType::Streamvbyte) => {
            build_sparse_dotvbyte_permuted::<StreamVByteNeighbors>(&args, distance, &config);
        }
        // Unreachable: caught by earlier validation
        (DatasetType::Dense, EncoderType::Dotvbyte, _, _)
        | (DatasetType::Sparse, EncoderType::Pq, _, _)
        | (
            DatasetType::Dense,
            EncoderType::Plain,
            ValueTypeArg::Fixedu8 | ValueTypeArg::Fixedu16,
            _,
        ) => {
            unreachable!()
        }
    }
}

fn build_dense_plain<V, G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            build_dense_plain_with_distance::<V, SquaredEuclideanDistance, G>(args, config)
        }
        Distance::DotProduct => build_dense_plain_with_distance::<V, DotProduct, G>(args, config),
    }
}

fn build_dense_plain_with_distance<V, D, G>(args: &Args, config: &HNSWBuildConfiguration)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarDenseSupportedDistance,
    G: GraphBound,
{
    let dataset = read_dense_v_dataset::<V, D>(args);
    build_and_save::<_, _, G>(dataset, config, &args.output_file, |i| i);
}

fn build_dense_plain_permuted<V, Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize + Default,
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => {
            build_dense_plain_permuted_with_distance::<V, SquaredEuclideanDistance, Ndst>(
                args, config,
            )
        }
        Distance::DotProduct => {
            build_dense_plain_permuted_with_distance::<V, DotProduct, Ndst>(args, config)
        }
    }
}

fn build_dense_plain_permuted_with_distance<V, D, Ndst>(
    args: &Args,
    config: &HNSWBuildConfiguration,
) where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize + Default,
    D: ScalarDenseSupportedDistance,
    Ndst: NeighborsBound,
{
    let dataset = read_dense_v_dataset::<V, D>(args);
    build_permuted_and_save::<_, _, Ndst>(dataset, config, &args.output_file, |i| i);
}

fn build_dense_pq<G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => build_dense_pq_l2::<G>(args, config),
        Distance::DotProduct => build_dense_pq_ip::<G>(args, config),
    }
}

fn build_dense_pq_permuted<Ndst>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => build_dense_pq_permuted_l2::<Ndst>(args, config),
        Distance::DotProduct => build_dense_pq_permuted_ip::<Ndst>(args, config),
    }
}

fn build_dense_pq_l2<G>(args: &Args, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
        read_dense_plain_dataset::<SquaredEuclideanDistance>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_l2::<4, G>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_l2::<8, G>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_l2::<16, G>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_l2::<32, G>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_l2::<48, G>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_l2::<64, G>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_l2::<96, G>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_l2::<128, G>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_l2::<192, G>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn build_dense_pq_ip<G>(args: &Args, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, DotProduct> = read_dense_plain_dataset::<DotProduct>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_ip::<4, G>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_ip::<8, G>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_ip::<16, G>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_ip::<32, G>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_ip::<48, G>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_ip::<64, G>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_ip::<96, G>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_ip::<128, G>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_ip::<192, G>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn build_dense_pq_permuted_l2<Ndst>(args: &Args, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
        read_dense_plain_dataset::<SquaredEuclideanDistance>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_l2_permuted::<4, Ndst>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_l2_permuted::<8, Ndst>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_l2_permuted::<16, Ndst>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_l2_permuted::<32, Ndst>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_l2_permuted::<48, Ndst>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_l2_permuted::<64, Ndst>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_l2_permuted::<96, Ndst>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_l2_permuted::<128, Ndst>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_l2_permuted::<192, Ndst>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn build_dense_pq_permuted_ip<Ndst>(args: &Args, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    let dataset: PlainDenseDataset<f32, DotProduct> = read_dense_plain_dataset::<DotProduct>(args);
    let pq_subspaces = choose_pq_subspaces(dataset.input_dim(), args.pq_subspaces);
    match pq_subspaces {
        4 => build_dense_pq_with_m_ip_permuted::<4, Ndst>(dataset, config, &args.output_file),
        8 => build_dense_pq_with_m_ip_permuted::<8, Ndst>(dataset, config, &args.output_file),
        16 => build_dense_pq_with_m_ip_permuted::<16, Ndst>(dataset, config, &args.output_file),
        32 => build_dense_pq_with_m_ip_permuted::<32, Ndst>(dataset, config, &args.output_file),
        48 => build_dense_pq_with_m_ip_permuted::<48, Ndst>(dataset, config, &args.output_file),
        64 => build_dense_pq_with_m_ip_permuted::<64, Ndst>(dataset, config, &args.output_file),
        96 => build_dense_pq_with_m_ip_permuted::<96, Ndst>(dataset, config, &args.output_file),
        128 => build_dense_pq_with_m_ip_permuted::<128, Ndst>(dataset, config, &args.output_file),
        192 => build_dense_pq_with_m_ip_permuted::<192, Ndst>(dataset, config, &args.output_file),
        _ => unreachable!(),
    }
}

fn read_dense_plain_dataset<D>(args: &Args) -> PlainDenseDataset<f32, D>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(&args.data_file).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        process::exit(1);
    })
}

/// Reads the `.npy` dataset and re-quantizes it into a `DenseDataset` with value type `V`.
fn read_dense_v_dataset<V, D>(args: &Args) -> DenseDataset<PlainDenseQuantizer<V, D>>
where
    V: ValueType + Float + FromF32,
    D: ScalarDenseSupportedDistance,
{
    let dataset_f32 = read_npy_f32::<D>(&args.data_file).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        process::exit(1);
    });
    let d = dataset_f32.input_dim();
    let n_vecs = dataset_f32.len();
    let data: Vec<V> = dataset_f32
        .values()
        .iter()
        .map(|v| V::from_f32_saturating(*v))
        .collect();

    let encoder = PlainDenseQuantizer::<V, D>::new(d);
    DenseDataset::from_raw(data.into_boxed_slice(), n_vecs, encoder)
}

fn build_sparse_plain<V, G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => {
            build_sparse_plain_with_component::<u16, V, G>(args, distance, config)
        }
        ComponentTypeArg::U32 => {
            build_sparse_plain_with_component::<u32, V, G>(args, distance, config)
        }
    }
}

fn build_sparse_plain_with_component<C, V, G>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            build_sparse_plain_with_distance::<C, V, SquaredEuclideanDistance, G>(args, config)
        }
        Distance::DotProduct => {
            build_sparse_plain_with_distance::<C, V, DotProduct, G>(args, config)
        }
    }
}

fn build_sparse_plain_with_distance<C, V, D, G>(args: &Args, config: &HNSWBuildConfiguration)
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
    G: GraphBound,
{
    let dataset: PlainSparseDataset<C, V, D> =
        read_seismic_format::<C, V, D>(&args.data_file).unwrap();
    build_and_save::<_, _, G>(dataset, config, &args.output_file, |i| i);
}

fn build_sparse_plain_permuted<V, Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    Ndst: NeighborsBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => {
            build_sparse_plain_permuted_with_component::<u16, V, Ndst>(args, distance, config)
        }
        ComponentTypeArg::U32 => {
            build_sparse_plain_permuted_with_component::<u32, V, Ndst>(args, distance, config)
        }
    }
}

fn build_sparse_plain_permuted_with_component<C, V, Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => {
            build_sparse_plain_permuted_with_distance::<C, V, SquaredEuclideanDistance, Ndst>(
                args, config,
            )
        }
        Distance::DotProduct => {
            build_sparse_plain_permuted_with_distance::<C, V, DotProduct, Ndst>(args, config)
        }
    }
}

fn build_sparse_plain_permuted_with_distance<C, V, D, Ndst>(
    args: &Args,
    config: &HNSWBuildConfiguration,
) where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
    Ndst: NeighborsBound,
{
    let dataset: PlainSparseDataset<C, V, D> =
        read_seismic_format::<C, V, D>(&args.data_file).unwrap();
    build_permuted_and_save::<_, _, Ndst>(dataset, config, &args.output_file, |i| i);
}

// --- DotVByte (encoder = dotvbyte, DotProduct only) ---

fn build_sparse_dotvbyte<G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            eprintln!("Error: DotVByte encoder only supports dotproduct distance.");
            process::exit(1);
        }
        Distance::DotProduct => build_sparse_dotvbyte_dp::<G>(args, config),
    }
}

fn build_sparse_dotvbyte_dp<G>(args: &Args, config: &HNSWBuildConfiguration)
where
    G: GraphBound,
{
    let dataset: PlainSparseDataset<u16, f32, DotProduct> =
        read_seismic_format::<u16, f32, DotProduct>(&args.data_file).unwrap_or_else(|e| {
            eprintln!("Error reading dataset: {e:?}");
            process::exit(1);
        });
    build_and_save::<_, PackedSparseDataset<DotVByteFixedU8Encoder>, G>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_sparse_dotvbyte_permuted<Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => {
            eprintln!("Error: DotVByte encoder only supports dotproduct distance.");
            process::exit(1);
        }
        Distance::DotProduct => build_sparse_dotvbyte_dp_permuted::<Ndst>(args, config),
    }
}

fn build_sparse_dotvbyte_dp_permuted<Ndst>(args: &Args, config: &HNSWBuildConfiguration)
where
    Ndst: NeighborsBound,
{
    let dataset: PlainSparseDataset<u16, f32, DotProduct> =
        read_seismic_format::<u16, f32, DotProduct>(&args.data_file).unwrap_or_else(|e| {
            eprintln!("Error reading dataset: {e:?}");
            process::exit(1);
        });
    build_permuted_and_save::<_, PackedSparseDataset<DotVByteFixedU8Encoder>, Ndst>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_sparse_scalar<V, G>(args: &Args, distance: Distance, config: &HNSWBuildConfiguration)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => {
            build_sparse_scalar_with_component::<u16, V, G>(args, distance, config)
        }
        ComponentTypeArg::U32 => {
            build_sparse_scalar_with_component::<u32, V, G>(args, distance, config)
        }
    }
}

fn build_sparse_scalar_with_component<C, V, G>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match distance {
        Distance::Euclidean => {
            build_sparse_scalar_with_distance::<C, V, SquaredEuclideanDistance, G>(args, config)
        }
        Distance::DotProduct => {
            build_sparse_scalar_with_distance::<C, V, DotProduct, G>(args, config)
        }
    }
}

fn build_sparse_scalar_with_distance<C, V, D, G>(args: &Args, config: &HNSWBuildConfiguration)
where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
    G: GraphBound,
    ScalarSparseDataset<C, f32, V, D>: ConvertFrom<PlainSparseDataset<C, f32, D>>,
{
    let dataset: PlainSparseDataset<C, f32, D> = read_seismic_format::<C, f32, D>(&args.data_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading dataset: {e:?}");
            process::exit(1);
        });
    build_and_save::<_, ScalarSparseDataset<C, f32, V, D>, G>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_sparse_scalar_permuted<V, Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    Ndst: NeighborsBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => {
            build_sparse_scalar_permuted_with_component::<u16, V, Ndst>(args, distance, config)
        }
        ComponentTypeArg::U32 => {
            build_sparse_scalar_permuted_with_component::<u32, V, Ndst>(args, distance, config)
        }
    }
}

fn build_sparse_scalar_permuted_with_component<C, V, Ndst>(
    args: &Args,
    distance: Distance,
    config: &HNSWBuildConfiguration,
) where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    Ndst: NeighborsBound,
{
    match distance {
        Distance::Euclidean => {
            build_sparse_scalar_permuted_with_distance::<C, V, SquaredEuclideanDistance, Ndst>(
                args, config,
            )
        }
        Distance::DotProduct => {
            build_sparse_scalar_permuted_with_distance::<C, V, DotProduct, Ndst>(args, config)
        }
    }
}

fn build_sparse_scalar_permuted_with_distance<C, V, D, Ndst>(
    args: &Args,
    config: &HNSWBuildConfiguration,
) where
    C: vectorium::ComponentType + num_traits::FromPrimitive + SpaceUsage + Serialize,
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
    Ndst: NeighborsBound,
    ScalarSparseDataset<C, f32, V, D>: ConvertFrom<PlainSparseDataset<C, f32, D>>,
    ScalarSparseDataset<C, f32, V, D>: Dataset<Owned = ScalarSparseDataset<C, f32, V, D>>,
{
    let dataset: PlainSparseDataset<C, f32, D> = read_seismic_format::<C, f32, D>(&args.data_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading dataset: {e:?}");
            process::exit(1);
        });
    build_permuted_and_save::<_, ScalarSparseDataset<C, f32, V, D>, Ndst>(
        dataset,
        config,
        &args.output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_dense_pq_with_m_l2<const M: usize, G>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    G: GraphBound,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Encoder = ProductQuantizer<M, SquaredEuclideanDistance>>,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<PlainDenseDataset<f32, SquaredEuclideanDistance>>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    build_and_save::<_, DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, G>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_dense_pq_with_m_ip<const M: usize, G>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    G: GraphBound,
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    DenseDataset<ProductQuantizer<M, DotProduct>>: ConvertFrom<PlainDenseDataset<f32, DotProduct>>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    build_and_save::<_, DenseDataset<ProductQuantizer<M, DotProduct>>, G>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_dense_pq_with_m_l2_permuted<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    Ndst: NeighborsBound,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>: Dataset<
            Encoder = ProductQuantizer<M, SquaredEuclideanDistance>,
            Owned = DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>,
        >,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<PlainDenseDataset<f32, SquaredEuclideanDistance>>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    build_permuted_and_save::<_, DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, Ndst>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into(),
    );
}

fn build_dense_pq_with_m_ip_permuted<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
    output_file: &str,
) where
    Ndst: NeighborsBound,
    DenseDataset<ProductQuantizer<M, DotProduct>>: Dataset<
            Encoder = ProductQuantizer<M, DotProduct>,
            Owned = DenseDataset<ProductQuantizer<M, DotProduct>>,
        >,
    DenseDataset<ProductQuantizer<M, DotProduct>>: ConvertFrom<PlainDenseDataset<f32, DotProduct>>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    build_permuted_and_save::<_, DenseDataset<ProductQuantizer<M, DotProduct>>, Ndst>(
        dataset,
        config,
        output_file,
        |i| i.convert_dataset_into(),
    );
}

trait GraphBound:
    kannolo::graph::GraphTrait + Serialize + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{
}
impl<T> GraphBound for T where
    T: kannolo::graph::GraphTrait
        + Serialize
        + for<'de> serde::Deserialize<'de>
        + From<GrowableGraph>
{
}

/// Bound satisfied by [`PlainNeighbors`] and [`StreamVByteNeighbors`], the two destination
/// backends of the `permuted`/`streamvbyte` graph types.
trait NeighborsBound: Neighbors + From<NeighborData> + Serialize + for<'de> Deserialize<'de> {}
impl<T> NeighborsBound for T where
    T: Neighbors + From<NeighborData> + Serialize + for<'de> Deserialize<'de>
{
}

/// Builds a `standard`/`fixed-degree` index, applies `convert`, then reports timing and space
/// usage and serializes it to `output_file`.
///
/// `convert` is the only per-encoder variation: identity (`|i| i`) when the built index is
/// already final, or `|i| i.convert_dataset_into()` to re-encode the dataset (PQ, scalar,
/// DotVByte) while keeping the graph unchanged.
fn build_and_save<Dsrc, Dfinal, G>(
    dataset: Dsrc,
    config: &HNSWBuildConfiguration,
    output_file: &str,
    convert: impl FnOnce(HNSW<Dsrc, G>) -> HNSW<Dfinal, G>,
) where
    Dsrc: Dataset + Sync + SpaceUsage,
    <Dsrc::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    Dfinal: Dataset + Sync + SpaceUsage + Serialize,
    G: GraphBound,
{
    let start_time = Instant::now();
    let plain_index: HNSW<Dsrc, G> = HNSW::build_index(dataset, config);
    let index = convert(plain_index);
    println!(
        "Time to build: {} s (before serializing)",
        start_time.elapsed().as_secs()
    );

    index.print_space_usage_bytes();
    let _ = index.save_index(output_file);
}

/// Like [`build_and_save`], but for the `permuted`/`streamvbyte` graph types: `convert` runs
/// first, then the index is rebuilt under an EGB node reordering into a fresh `Ndst` backend.
///
/// **`convert` must run before `permute_and_encode`.** Encoders that are fitted to the data — the
/// DotVByte component mapping in particular — train on a *prefix sample* of the dataset
/// (`SAMPLE_RATE` in vectorium's `PackedSparseDataset` conversion). Permuting first would hand
/// that sampler an EGB-clustered prefix instead of a representative one, yielding a different,
/// worse encoding than the `standard` index gets. Converting first keeps the encoder identical
/// to the `standard` path, and `Dataset::permute` then copies the already-encoded rows verbatim,
/// so the permuted dataset is byte-identical to the unpermuted one up to row order.
fn build_permuted_and_save<Dsrc, Dfinal, Ndst>(
    dataset: Dsrc,
    config: &HNSWBuildConfiguration,
    output_file: &str,
    convert: impl FnOnce(HNSW<Dsrc, Graph<PlainNeighbors>>) -> HNSW<Dfinal, Graph<PlainNeighbors>>,
) where
    Dsrc: Dataset + Sync + SpaceUsage,
    <Dsrc::Encoder as VectorEncoder>::Distance: vectorium::distances::Distance,
    // `permute_and_encode` hands the permuted dataset back as `Dfinal::Owned`. Every concrete dataset
    // sets `Owned = Self`, but the compiler cannot normalize that through the callers' generic
    // parameters, so those restate it in their own `where` clauses.
    Dfinal: Dataset<Owned = Dfinal> + Sync + SpaceUsage + Serialize,
    Ndst: NeighborsBound,
{
    let start_time = Instant::now();
    let plain_index: HNSW<Dsrc, Graph<PlainNeighbors>> = HNSW::build_index(dataset, config);
    let converted_index = convert(plain_index);
    let index = converted_index.permute_and_encode::<Ndst>();
    println!(
        "Time to build (incl. EGB permutation + compression): {} s (before serializing)",
        start_time.elapsed().as_secs()
    );

    index.print_space_usage_bytes();
    let _ = index.save_index(output_file);
}
