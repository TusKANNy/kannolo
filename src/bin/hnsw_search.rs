use std::io::Write;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;
use std::fs::File;

use kannolo::graph::{Graph, GraphFixedDegree, GraphTrait, GrowableGraph};
use kannolo::hnsw::{HNSW, HNSWSearchConfiguration};
use kannolo::index::Index;
use vectorium::IndexSerializer;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::{read_npy_f32, read_seismic_format};
use vectorium::{Dataset, DenseDataset, PlainDenseDataset, PlainSparseDataset};

#[derive(Debug, Clone, ValueEnum)]
enum VectorType {
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

#[derive(Clone, Copy, Debug)]
enum MetricKind {
    Euclidean,
    DotProduct,
}

trait GraphBound: GraphTrait + for<'de> serde::Deserialize<'de> + From<GrowableGraph> {}
impl<T> GraphBound for T where T: GraphTrait + for<'de> serde::Deserialize<'de> + From<GrowableGraph>
{}

fn parse_metric(metric: &str) -> MetricKind {
    match metric {
        "euclidean" | "l2" => MetricKind::Euclidean,
        "dotproduct" | "ip" => MetricKind::DotProduct,
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
    vector_type: VectorType,

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

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser)]
    metric: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m_pq: usize,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    /// Number of runs for timing.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    n_run: usize,
}

fn main() {
    let args: Args = Args::parse();

    match (&args.vector_type, &args.quantizer) {
        (VectorType::Sparse, QuantizerType::Pq) => {
            eprintln!("Error: PQ quantizer is only available for dense vectors.");
            std::process::exit(1);
        }
        (VectorType::Dense, QuantizerType::Pq) if matches!(args.precision, Precision::F16) => {
            eprintln!("Warning: PQ always uses f32 precision, ignoring f16 specification.");
        }
        _ => {}
    }

    let metric = parse_metric(&args.metric);

    match (
        &args.vector_type,
        &args.quantizer,
        &args.precision,
        &args.graph_type,
    ) {
        (VectorType::Dense, QuantizerType::Plain, Precision::F32, GraphType::Standard) => {
            search_dense_plain_f32::<Graph>(&args, metric);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F32, GraphType::FixedDegree) => {
            search_dense_plain_f32::<GraphFixedDegree>(&args, metric);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            search_dense_plain_f16::<Graph>(&args, metric);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            search_dense_plain_f16::<GraphFixedDegree>(&args, metric);
        }
        (VectorType::Dense, QuantizerType::Pq, _, GraphType::Standard) => {
            search_dense_pq::<Graph>(&args, metric);
        }
        (VectorType::Dense, QuantizerType::Pq, _, GraphType::FixedDegree) => {
            search_dense_pq::<GraphFixedDegree>(&args, metric);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            search_sparse_plain_f16::<Graph>(&args, metric);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            search_sparse_plain_f16::<GraphFixedDegree>(&args, metric);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F32, GraphType::Standard) => {
            search_sparse_plain_f32::<Graph>(&args, metric);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F32, GraphType::FixedDegree) => {
            search_sparse_plain_f32::<GraphFixedDegree>(&args, metric);
        }
        (VectorType::Sparse, QuantizerType::Pq, _, _) => unreachable!(),
    }
}

fn search_dense_plain_f32<G>(args: &Args, metric: MetricKind)
where
    G: GraphBound,
{
    match metric {
        MetricKind::Euclidean => {
            search_dense_plain_f32_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        MetricKind::DotProduct => search_dense_plain_f32_with_distance::<DotProduct, G>(args),
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
    let config = HNSWSearchConfiguration::default().with_ef_search(args.ef_search);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(res.into_iter().map(|scored| (scored.distance.distance(), scored.vector as usize)));
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f16<G>(args: &Args, metric: MetricKind)
where
    G: GraphBound,
{
    match metric {
        MetricKind::Euclidean => {
            search_dense_plain_f16_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        MetricKind::DotProduct => search_dense_plain_f16_with_distance::<DotProduct, G>(args),
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
    let config = HNSWSearchConfiguration::default().with_ef_search(args.ef_search);

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(res.into_iter().map(|scored| (scored.distance.distance(), scored.vector as usize)));
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_pq<G>(args: &Args, metric: MetricKind)
where
    G: GraphBound,
{
    match metric {
        MetricKind::Euclidean => search_dense_pq_with_distance::<SquaredEuclideanDistance, G>(args),
        MetricKind::DotProduct => search_dense_pq_with_distance::<DotProduct, G>(args),
    }
}

fn search_dense_pq_with_distance<D, G>(args: &Args)
where
    D: ProductQuantizerDistance + ScalarDenseSupportedDistance + Distance,
    G: GraphBound,
{
    let queries = read_npy_queries::<D>(&args.query_file);
    let num_queries = queries.len();

    let config = HNSWSearchConfiguration::default().with_ef_search(args.ef_search);

    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    match args.m_pq {
        8 => {
            let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, G> =
                <HNSW<DenseDataset<ProductQuantizer<8, D>>, G> as IndexSerializer>::load_index(
                    &args.index_file,
                )
                .unwrap();
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
            for _ in 0..args.n_run {
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
                "Error: Invalid m_pq value. Choose between 8, 16, 32, 48, 64, 96, 128, 192, 256, 384."
            );
            std::process::exit(1);
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f16<G>(args: &Args, metric: MetricKind)
where
    G: GraphBound,
{
    match metric {
        MetricKind::Euclidean => {
            search_sparse_plain_f16_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        MetricKind::DotProduct => search_sparse_plain_f16_with_distance::<DotProduct, G>(args),
    }
}

fn search_sparse_plain_f16_with_distance<D, G>(args: &Args)
where
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = HNSWSearchConfiguration::default().with_ef_search(args.ef_search);
    let queries: PlainSparseDataset<u16, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();
    let index: HNSW<PlainSparseDataset<u16, f16, D>, G> =
        <HNSW<PlainSparseDataset<u16, f16, D>, G> as IndexSerializer>::load_index(&args.index_file)
            .unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(res.into_iter().map(|scored| (scored.distance.distance(), scored.vector as usize)));
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f32<G>(args: &Args, metric: MetricKind)
where
    G: GraphBound,
{
    match metric {
        MetricKind::Euclidean => {
            search_sparse_plain_f32_with_distance::<SquaredEuclideanDistance, G>(args)
        }
        MetricKind::DotProduct => search_sparse_plain_f32_with_distance::<DotProduct, G>(args),
    }
}

fn search_sparse_plain_f32_with_distance<D, G>(args: &Args)
where
    D: ScalarSparseSupportedDistance + Distance,
    G: GraphBound,
{
    let config = HNSWSearchConfiguration::default().with_ef_search(args.ef_search);
    let queries: PlainSparseDataset<u16, f32, D> = read_seismic_format(&args.query_file)
        .unwrap_or_else(|e| {
            eprintln!("Error reading query file: {e:?}");
            std::process::exit(1);
        });
    let num_queries = queries.len();
    let index: HNSW<PlainSparseDataset<u16, f32, D>, G> =
        <HNSW<PlainSparseDataset<u16, f32, D>, G> as IndexSerializer>::load_index(&args.index_file)
            .unwrap();

    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            let res = index.search(query, args.k, &config);
            results.extend(res.into_iter().map(|scored| (scored.distance.distance(), scored.vector as usize)));
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
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
