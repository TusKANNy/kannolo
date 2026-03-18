use std::io::Write;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use std::fs::File;

use kannolo::graph::{Graph, GraphFixedDegree, GraphTrait, GrowableGraph};
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::index::Index;
use vectorium::IndexSerializer;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::dotvbyte_fixedu8::DotVByteFixedU8Encoder;
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::{
    Dataset, DenseDataset, FixedU8Q, FixedU16Q, PackedSparseDataset, PlainDenseDataset,
    PlainSparseDataset, ScalarSparseDataset,
};

#[derive(Debug, Clone, ValueEnum)]
enum DatasetType {
    Dense,
    Sparse,
}

/// Value type for stored values.
/// Dense plain: `f32`, `f16`.
/// Sparse plain: `f32`, `f16`, `fixedu8`, `fixedu16`.
/// Ignored for `dotvbyte` and `pq`.
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
    Standard,
    FixedDegree,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum ComponentTypeArg {
    #[default]
    U16,
    U32,
}

#[derive(Debug, Clone, ValueEnum)]
enum EarlyTerminationMethod {
    None,
    DistanceAdaptive,
}

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
    Euclidean,
    DotProduct,
}

trait GraphBound: GraphTrait + for<'de> serde::Deserialize<'de> + From<GrowableGraph> {}
impl<T> GraphBound for T where T: GraphTrait + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{}

fn parse_metric(metric: &str) -> DistanceKind {
    match metric {
        "euclidean" | "l2" => DistanceKind::Euclidean,
        "dotproduct" | "ip" => DistanceKind::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'euclidean' and 'dotproduct'.");
            std::process::exit(1);
        }
    }
}

fn read_npy_queries<D>(path: &str) -> PlainDenseDataset<f32, D>
where
    D: ScalarDenseSupportedDistance,
{
    read_npy_f32::<D>(path).unwrap_or_else(|e| {
        eprintln!("Error reading .npy file: {e:?}");
        std::process::exit(1);
    })
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The type of vectors (dense or sparse).
    #[clap(long, value_enum)]
    dataset_type: DatasetType,

    /// Value type for stored values. Dense plain: f32, f16. Sparse plain: f32, f16, fixedu8, fixedu16.
    /// Ignored for dotvbyte and pq.
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

    /// The graph type (standard or fixed-degree).
    #[clap(long, value_enum)]
    #[arg(default_value_t = GraphType::Standard)]
    graph_type: GraphType,

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser)]
    distance: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    pq_subspaces: usize,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    /// Early termination strategy for search.
    #[clap(long, value_enum)]
    #[arg(default_value_t = EarlyTerminationMethod::None)]
    early_termination: EarlyTerminationMethod,

    /// Lambda parameter for DistanceAdaptive strategy.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1.0)]
    lambda: f32,

    /// Number of runs for timing.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    num_runs: usize,
}

fn main() {
    let args: Args = Args::parse();

    // Cross-validation of encoder / dataset-type combinations
    match (&args.dataset_type, &args.encoder) {
        (DatasetType::Sparse, EncoderType::Pq) => {
            eprintln!("Error: PQ encoder is only available for dense vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Dotvbyte) => {
            eprintln!("Error: DotVByte encoder is only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, EncoderType::Plain)
            if matches!(
                args.value_type,
                ValueTypeArg::Fixedu8 | ValueTypeArg::Fixedu16
            ) =>
        {
            eprintln!("Error: fixedu8/fixedu16 value types are only available for sparse vectors.");
            std::process::exit(1);
        }
        (DatasetType::Dense, _) if !matches!(args.component_type, ComponentTypeArg::U16) => {
            eprintln!("Error: component-type is only applicable to sparse datasets.");
            std::process::exit(1);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte)
            if !matches!(args.component_type, ComponentTypeArg::U16) =>
        {
            eprintln!("Error: DotVByte encoder supports only component-type u16.");
            std::process::exit(1);
        }
        _ => {}
    }

    let metric = parse_metric(&args.distance);

    match (
        &args.dataset_type,
        &args.encoder,
        &args.value_type,
        &args.graph_type,
    ) {
        // Dense plain f32
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::Standard) => {
            search_dense_plain_f32::<Graph>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F32, GraphType::FixedDegree) => {
            search_dense_plain_f32::<GraphFixedDegree>(&args, metric);
        }
        // Dense plain f16
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::Standard) => {
            search_dense_plain_f16::<Graph>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Plain, ValueTypeArg::F16, GraphType::FixedDegree) => {
            search_dense_plain_f16::<GraphFixedDegree>(&args, metric);
        }
        // Dense PQ (value-type ignored)
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::Standard) => {
            search_dense_pq::<Graph>(&args, metric);
        }
        (DatasetType::Dense, EncoderType::Pq, _, GraphType::FixedDegree) => {
            search_dense_pq::<GraphFixedDegree>(&args, metric);
        }
        // Sparse plain f16
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16, GraphType::Standard) => {
            search_sparse_plain_f16::<Graph>(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F16, GraphType::FixedDegree) => {
            search_sparse_plain_f16::<GraphFixedDegree>(&args, metric);
        }
        // Sparse plain f32
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32, GraphType::Standard) => {
            search_sparse_plain_f32::<Graph>(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::F32, GraphType::FixedDegree) => {
            search_sparse_plain_f32::<GraphFixedDegree>(&args, metric);
        }
        // Sparse plain fixedu8
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu8, GraphType::Standard) => {
            search_sparse_scalar::<FixedU8Q, Graph>(&args, metric);
        }
        (
            DatasetType::Sparse,
            EncoderType::Plain,
            ValueTypeArg::Fixedu8,
            GraphType::FixedDegree,
        ) => {
            search_sparse_scalar::<FixedU8Q, GraphFixedDegree>(&args, metric);
        }
        // Sparse plain fixedu16
        (DatasetType::Sparse, EncoderType::Plain, ValueTypeArg::Fixedu16, GraphType::Standard) => {
            search_sparse_scalar::<FixedU16Q, Graph>(&args, metric);
        }
        (
            DatasetType::Sparse,
            EncoderType::Plain,
            ValueTypeArg::Fixedu16,
            GraphType::FixedDegree,
        ) => {
            search_sparse_scalar::<FixedU16Q, GraphFixedDegree>(&args, metric);
        }
        // Sparse dotvbyte (value-type ignored)
        (DatasetType::Sparse, EncoderType::Dotvbyte, _, GraphType::Standard) => {
            search_sparse_dotvbyte::<Graph>(&args, metric);
        }
        (DatasetType::Sparse, EncoderType::Dotvbyte, _, GraphType::FixedDegree) => {
            search_sparse_dotvbyte::<GraphFixedDegree>(&args, metric);
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

fn create_search_config(args: &Args) -> HNSWSearchConfiguration {
    let early_termination = match args.early_termination {
        EarlyTerminationMethod::None => EarlyTerminationStrategy::None,
        EarlyTerminationMethod::DistanceAdaptive => EarlyTerminationStrategy::DistanceAdaptive {
            lambda: args.lambda,
        },
    };

    HNSWSearchConfiguration::default()
        .with_ef_search(args.ef_search)
        .with_early_termination(early_termination)
}

fn search_dense_plain_f32<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_plain_f32_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_plain_f32_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_plain_f32_with_distance<D, G>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();
    let index: HNSW<DenseDataset<PlainDenseQuantizer<f32, D>>, G> =
        <HNSW<DenseDataset<PlainDenseQuantizer<f32, D>>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f16<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_plain_f16_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_plain_f16_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_plain_f16_with_distance<D, G>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();
    let index: HNSW<DenseDataset<PlainDenseQuantizer<f16, D>>, G> =
        <HNSW<DenseDataset<PlainDenseQuantizer<f16, D>>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();
    let config = create_search_config(args);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_pq<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_dense_pq_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_dense_pq_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_pq_with_distance<D, G>(args: &Args)
where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();

    let config = create_search_config(args);

    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    match args.pq_subspaces {
        4 => {
            let index: HNSW<DenseDataset<ProductQuantizer<4, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<4, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        8 => {
            let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<8, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        16 => {
            let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<16, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        32 => {
            let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<32, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        48 => {
            let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<48, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        64 => {
            let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<64, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        96 => {
            let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<96, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        128 => {
            let index: HNSW<DenseDataset<ProductQuantizer<128, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<128, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        192 => {
            let index: HNSW<DenseDataset<ProductQuantizer<192, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<192, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        256 => {
            let index: HNSW<DenseDataset<ProductQuantizer<256, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<256, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        384 => {
            let index: HNSW<DenseDataset<ProductQuantizer<384, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<384, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.num_runs {
                for query in queries.iter() {
                    let start_time = Instant::now();
                    let res = index.search(query, args.k, &config);
                    results.extend(
                        res.into_iter()
                            .map(|scored| (scored.distance.distance(), scored.vector as usize)),
                    );
                    total_time_search += start_time.elapsed().as_micros();
                }
            }
            index.print_space_usage_bytes();
        }
        _ => {
            eprintln!(
                "Error: Invalid pq-subspaces value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384."
            );
            std::process::exit(1);
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f16<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => search_sparse_plain_f16_with_component::<u16, G>(args, metric),
        ComponentTypeArg::U32 => search_sparse_plain_f16_with_component::<u32, G>(args, metric),
    }
}

fn search_sparse_plain_f16_with_component<C, G>(args: &Args, metric: DistanceKind)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_sparse_plain_f16_with_distance::<C, SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_sparse_plain_f16_with_distance::<C, DotProduct, G>(args),
    }
}

fn search_sparse_plain_f16_with_distance<C, D, G>(args: &Args)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();
    let index: HNSW<PlainSparseDataset<C, f16, D>, G> =
        <HNSW<PlainSparseDataset<C, f16, D>, G> as IndexSerializer>::load_index(&args.index_file)
            .unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f32<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => search_sparse_plain_f32_with_component::<u16, G>(args, metric),
        ComponentTypeArg::U32 => search_sparse_plain_f32_with_component::<u32, G>(args, metric),
    }
}

fn search_sparse_plain_f32_with_component<C, G>(args: &Args, metric: DistanceKind)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_sparse_plain_f32_with_distance::<C, SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_sparse_plain_f32_with_distance::<C, DotProduct, G>(args),
    }
}

fn search_sparse_plain_f32_with_distance<C, D, G>(args: &Args)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();
    let index: HNSW<PlainSparseDataset<C, f32, D>, G> =
        <HNSW<PlainSparseDataset<C, f32, D>, G> as IndexSerializer>::load_index(&args.index_file)
            .unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

// --- DotVByte (encoder = dotvbyte, DotProduct only) ---

fn search_sparse_dotvbyte<G>(args: &Args, metric: DistanceKind)
where
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            eprintln!("Error: DotVByte encoder only supports dotproduct distance.");
            std::process::exit(1);
        }
        DistanceKind::DotProduct => search_sparse_dotvbyte_dp::<G>(args),
    }
}

fn search_sparse_dotvbyte_dp<G>(args: &Args)
where
    G: GraphBound,
{
    let config = create_search_config(args);
    let queries: PlainSparseDataset<u16, f32, DotProduct> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();

    let index: HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, G> =
        <HNSW<PackedSparseDataset<DotVByteFixedU8Encoder>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_scalar<V, G>(args: &Args, metric: DistanceKind)
where
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    G: GraphBound,
{
    match args.component_type {
        ComponentTypeArg::U16 => search_sparse_scalar_with_component::<u16, V, G>(args, metric),
        ComponentTypeArg::U32 => search_sparse_scalar_with_component::<u32, V, G>(args, metric),
    }
}

fn search_sparse_scalar_with_component<C, V, G>(args: &Args, metric: DistanceKind)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    G: GraphBound,
{
    match metric {
        DistanceKind::Euclidean => {
            search_sparse_scalar_with_distance::<C, V, SquaredEuclideanDistance, G>(args)
        }
        DistanceKind::DotProduct => search_sparse_scalar_with_distance::<C, V, DotProduct, G>(args),
    }
}

fn search_sparse_scalar_with_distance<C, V, D, G>(args: &Args)
where
    C: vectorium::ComponentType
        + num_traits::FromPrimitive
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    V: vectorium::ValueType
        + vectorium::Float
        + vectorium::FromF32
        + vectorium::SpaceUsage
        + for<'de> serde::Deserialize<'de>,
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = create_search_config(args);
    let queries: PlainSparseDataset<C, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();

    let index: HNSW<ScalarSparseDataset<C, f32, V, D>, G> =
        <HNSW<ScalarSparseDataset<C, f32, V, D>, G> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.num_runs {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut file = File::create(output_path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}
