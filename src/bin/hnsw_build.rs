use std::{fmt::Debug, time::Instant};

use clap::{Parser, ValueEnum};
use half::f16;
use serde::Serialize;
use std::process;

use kannolo::graph::{Graph, GraphFixedDegree, GrowableGraph};
use kannolo::hnsw::{HNSWBuildParams, HNSW};
use kannolo::index::Index;
use vectorium::IndexSerializer;
use vectorium::dataset::{ConvertFrom, ConvertInto};
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::{Dataset, DenseDataset, PlainDenseDataset, PlainSparseDataset, ValueType, FromF32, Float, SpaceUsage};

#[derive(Debug, Clone, ValueEnum)]
enum VectorRepresentation {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, ValueEnum)]
enum Precision {
    F16,
    F32,
}

#[derive(Debug, Clone, ValueEnum)]
enum QuantizerType {
    Plain,
    Pq,
}

#[derive(Debug, Clone, ValueEnum)]
enum GraphType {
    Standard,
    FixedDegree,
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
    vector_representation: VectorRepresentation,

    /// The precision (f16 or f32). Note: PQ always uses f32.
    #[clap(long, value_enum)]
    #[arg(default_value_t = Precision::F32)]
    precision: Precision,

    /// The quantizer type (plain or pq). Note: PQ is only available for dense vectors.
    #[clap(long, value_enum)]
    #[arg(default_value_t = QuantizerType::Plain)]
    quantizer: QuantizerType,

    /// The graph type (standard or fixed-degree).
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
    efc: usize,

    /// The type of distance to use. Either 'euclidean' or 'dotproduct'.
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("dotproduct"))]
    metric: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m_pq: usize,

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
enum Metric {
    L2,
    Ip,
}

fn parse_metric(metric: &str) -> Metric {
    match metric {
        "euclidean" | "l2" => Metric::L2,
        "dotproduct" | "ip" => Metric::Ip,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'.");
            process::exit(1);
        }
    }
}

fn main() {
    let args: Args = Args::parse();

    match (&args.vector_representation, &args.quantizer) {
        (VectorRepresentation::Sparse, QuantizerType::Pq) => {
            eprintln!("Error: PQ quantizer is only available for dense vectors.");
            process::exit(1);
        }
        (VectorRepresentation::Dense, QuantizerType::Pq) if matches!(args.precision, Precision::F16) => {
            eprintln!("Warning: PQ always uses f32 precision, ignoring f16 specification.");
        }
        _ => {}
    }

    if matches!(args.quantizer, QuantizerType::Pq) {
        if args.nbits != 8 {
            eprintln!("Warning: vectorium PQ ignores --nbits (fixed codebook size)." );
        }
        if args.sample_size != 100_000 {
            eprintln!("Warning: vectorium PQ ignores --sample-size and uses automatic sampling.");
        }
    }

    let metric = parse_metric(&args.metric);

    let config = HNSWBuildParams::new(args.m, args.efc, 4, 320);

    println!(
        "Building Index with M: {}, ef_construction: {}",
        args.m, args.efc
    );

    match (
        &args.vector_representation,
        &args.quantizer,
        &args.precision,
        &args.graph_type,
    ) {
        // Dense vectors with plain quantizer
        (VectorRepresentation::Dense, QuantizerType::Plain, Precision::F32, GraphType::Standard) => {
            build_dense_plain::<f32, Graph>(&args, metric, &config);
        }
        (VectorRepresentation::Dense, QuantizerType::Plain, Precision::F32, GraphType::FixedDegree) => {
            build_dense_plain::<f32, GraphFixedDegree>(&args, metric, &config);
        }
        (VectorRepresentation::Dense, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            build_dense_plain::<f16, Graph>(&args, metric, &config);
        }
        (VectorRepresentation::Dense, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            build_dense_plain::<f16, GraphFixedDegree>(&args, metric, &config);
        }
        // Dense vectors with PQ quantizer (always f32)
        (VectorRepresentation::Dense, QuantizerType::Pq, _, GraphType::Standard) => {
            build_dense_pq::<Graph>(&args, metric, &config);
        }
        (VectorRepresentation::Dense, QuantizerType::Pq, _, GraphType::FixedDegree) => {
            build_dense_pq::<GraphFixedDegree>(&args, metric, &config);
        }
        // Sparse vectors with plain quantizer
        (VectorRepresentation::Sparse, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            build_sparse_plain::<f16, Graph>(&args, metric, &config);
        }
        (VectorRepresentation::Sparse, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            build_sparse_plain::<f16, GraphFixedDegree>(&args, metric, &config);
        }
        (VectorRepresentation::Sparse, QuantizerType::Plain, Precision::F32, GraphType::Standard) => {
            build_sparse_plain::<f32, Graph>(&args, metric, &config);
        }
        (VectorRepresentation::Sparse, QuantizerType::Plain, Precision::F32, GraphType::FixedDegree) => {
            build_sparse_plain::<f32, GraphFixedDegree>(&args, metric, &config);
        }
        (VectorRepresentation::Sparse, QuantizerType::Pq, _, _) => unreachable!(),
    }
}

fn build_dense_plain<V, G>(args: &Args, metric: Metric, config: &HNSWBuildParams)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match metric {
        Metric::L2 => build_dense_plain_with_distance::<V, SquaredEuclideanDistance, G>(args, config),
        Metric::Ip => build_dense_plain_with_distance::<V, DotProduct, G>(args, config),
    }
}

fn build_dense_plain_with_distance<V, D, G>(args: &Args, config: &HNSWBuildParams)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarDenseSupportedDistance,
    G: GraphBound,
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
    let dataset: DenseDataset<_> =
        DenseDataset::from_raw(data.into_boxed_slice(), n_vecs, encoder);

    let start_time = Instant::now();
    let index: HNSW<_, G> = HNSW::build_index(dataset, config);
    let duration = start_time.elapsed();
    println!("Time to build: {} s (before serializing)", duration.as_secs());

    let _ = index.save_index(&args.output_file);
}

fn build_dense_pq<G>(args: &Args, metric: Metric, config: &HNSWBuildParams)
where
    G: GraphBound,
{
    match metric {
        Metric::L2 => build_dense_pq_l2::<G>(args, config),
        Metric::Ip => build_dense_pq_ip::<G>(args, config),
    }
}

fn build_dense_pq_l2<G>(args: &Args, config: &HNSWBuildParams)
where
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
        read_dense_plain_dataset::<SquaredEuclideanDistance>(args);
    match args.m_pq {
        4 => build_dense_pq_with_m::<4, SquaredEuclideanDistance, G>(dataset, args, config),
        8 => build_dense_pq_with_m::<8, SquaredEuclideanDistance, G>(dataset, args, config),
        16 => build_dense_pq_with_m::<16, SquaredEuclideanDistance, G>(dataset, args, config),
        32 => build_dense_pq_with_m::<32, SquaredEuclideanDistance, G>(dataset, args, config),
        48 => build_dense_pq_with_m::<48, SquaredEuclideanDistance, G>(dataset, args, config),
        64 => build_dense_pq_with_m::<64, SquaredEuclideanDistance, G>(dataset, args, config),
        96 => build_dense_pq_with_m::<96, SquaredEuclideanDistance, G>(dataset, args, config),
        128 => build_dense_pq_with_m::<128, SquaredEuclideanDistance, G>(dataset, args, config),
        192 => build_dense_pq_with_m::<192, SquaredEuclideanDistance, G>(dataset, args, config),
        256 => build_dense_pq_with_m::<256, SquaredEuclideanDistance, G>(dataset, args, config),
        384 => build_dense_pq_with_m::<384, SquaredEuclideanDistance, G>(dataset, args, config),
        _ => {
            eprintln!("Error: Invalid m_pq value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.");
            process::exit(1);
        }
    }
}

fn build_dense_pq_ip<G>(args: &Args, config: &HNSWBuildParams)
where
    G: GraphBound,
{
    let dataset: PlainDenseDataset<f32, DotProduct> =
        read_dense_plain_dataset::<DotProduct>(args);
    match args.m_pq {
        4 => build_dense_pq_with_m::<4, DotProduct, G>(dataset, args, config),
        8 => build_dense_pq_with_m::<8, DotProduct, G>(dataset, args, config),
        16 => build_dense_pq_with_m::<16, DotProduct, G>(dataset, args, config),
        32 => build_dense_pq_with_m::<32, DotProduct, G>(dataset, args, config),
        48 => build_dense_pq_with_m::<48, DotProduct, G>(dataset, args, config),
        64 => build_dense_pq_with_m::<64, DotProduct, G>(dataset, args, config),
        96 => build_dense_pq_with_m::<96, DotProduct, G>(dataset, args, config),
        128 => build_dense_pq_with_m::<128, DotProduct, G>(dataset, args, config),
        192 => build_dense_pq_with_m::<192, DotProduct, G>(dataset, args, config),
        256 => build_dense_pq_with_m::<256, DotProduct, G>(dataset, args, config),
        384 => build_dense_pq_with_m::<384, DotProduct, G>(dataset, args, config),
        _ => {
            eprintln!("Error: Invalid m_pq value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.");
            process::exit(1);
        }
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

fn build_sparse_plain<V, G>(args: &Args, metric: Metric, config: &HNSWBuildParams)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    G: GraphBound,
{
    match metric {
        Metric::L2 => build_sparse_plain_with_distance::<V, SquaredEuclideanDistance, G>(args, config),
        Metric::Ip => build_sparse_plain_with_distance::<V, DotProduct, G>(args, config),
    }
}

fn build_dense_pq_with_m<const M: usize, D, G>(
    dataset: PlainDenseDataset<f32, D>,
    args: &Args,
    config: &HNSWBuildParams,
) where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + 'static,
    G: GraphBound,
    DenseDataset<ProductQuantizer<M, D>>: ConvertFrom<PlainDenseDataset<f32, D>>,
{
    let pq_dataset: DenseDataset<ProductQuantizer<M, D>> = dataset.convert_into();
    let start_time = Instant::now();
    let index: HNSW<_, G> = HNSW::build_index(pq_dataset, config);
    let duration = start_time.elapsed();
    println!("Time to build: {} s (before serializing)", duration.as_secs());

    let _ = index.save_index(&args.output_file);
}

fn build_sparse_plain_with_distance<V, D, G>(args: &Args, config: &HNSWBuildParams)
where
    V: ValueType + Float + FromF32 + SpaceUsage + Serialize,
    D: ScalarSparseSupportedDistance,
    G: GraphBound,
{
    let dataset: PlainSparseDataset<u16, V, D> =
        read_seismic_format::<u16, V, D>(&args.data_file).unwrap();

    let start_time = Instant::now();
    let index: HNSW<_, G> = HNSW::build_index(dataset, config);
    let duration = start_time.elapsed();
    println!("Time to build: {} s (before serializing)", duration.as_secs());

    let _ = index.save_index(&args.output_file);
}

trait GraphBound:
    kannolo::graph::GraphTrait + Serialize + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{
}
impl<T> GraphBound for T where
    T: kannolo::graph::GraphTrait + Serialize + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{
}
