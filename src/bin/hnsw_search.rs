use std::io::Write;
use std::{fmt::Debug, time::Instant};

use clap::{Parser, ValueEnum};
use half::f16;
use kannolo::graph::{Graph, GraphFixedDegree};
use kannolo::pq::ProductQuantizer;
use kannolo::sparse_plain_quantizer::SparsePlainQuantizer;
use std::fs::File;

use kannolo::{
    graph_index::GraphIndex,
    hnsw::{HNSWSearchParams, HNSW},
    plain_quantizer::PlainQuantizer,
    read_numpy_f32_flatten_2d, Dataset, DenseDataset, DistanceType, IndexSerializer, SparseDataset,
};

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
    // Parse command line arguments
    let args: Args = Args::parse();

    // Validate arguments
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

    println!("Starting search");

    match (
        &args.vector_type,
        &args.quantizer,
        &args.precision,
        &args.graph_type,
    ) {
        // Dense vectors with plain quantizer
        (VectorType::Dense, QuantizerType::Plain, Precision::F32, GraphType::Standard) => {
            search_dense_plain_f32_standard(&args);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F32, GraphType::FixedDegree) => {
            search_dense_plain_f32_fixed(&args);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            search_dense_plain_f16_standard(&args);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            search_dense_plain_f16_fixed(&args);
        }
        // Dense vectors with PQ quantizer (always f32)
        (VectorType::Dense, QuantizerType::Pq, _, GraphType::Standard) => {
            search_dense_pq_standard(&args);
        }
        (VectorType::Dense, QuantizerType::Pq, _, GraphType::FixedDegree) => {
            search_dense_pq_fixed(&args);
        }
        // Sparse vectors with plain quantizer (f16 only)
        (VectorType::Sparse, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            search_sparse_plain_f16_standard(&args);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            search_sparse_plain_f16_fixed(&args);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F32, _) => {
            eprintln!("Error: Sparse vectors currently only support f16 precision.");
            std::process::exit(1);
        }
        // This case is already handled by validation above
        (VectorType::Sparse, QuantizerType::Pq, _, _) => unreachable!(),
    }
}

fn search_dense_plain_f32_standard(args: &Args) {
    println!("Reading Queries");
    let (queries_vec, d) = read_numpy_f32_flatten_2d(args.query_file.clone());
    let queries = DenseDataset::from_vec(
        queries_vec,
        d,
        PlainQuantizer::<f32>::new(d, DistanceType::Euclidean),
    );

    let index: HNSW<DenseDataset<PlainQuantizer<f32>, Vec<f32>>, PlainQuantizer<f32>, Graph> =
        IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index.search::<DenseDataset<PlainQuantizer<f32>, Vec<f32>>, PlainQuantizer<f32>>(
                    query, args.k, &config,
                ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f32_fixed(args: &Args) {
    println!("Reading Queries");
    let (queries_vec, d) = read_numpy_f32_flatten_2d(args.query_file.clone());
    let queries = DenseDataset::from_vec(
        queries_vec,
        d,
        PlainQuantizer::<f32>::new(d, DistanceType::Euclidean),
    );

    let index: HNSW<
        DenseDataset<PlainQuantizer<f32>, Vec<f32>>,
        PlainQuantizer<f32>,
        GraphFixedDegree,
    > = IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index.search::<DenseDataset<PlainQuantizer<f32>, Vec<f32>>, PlainQuantizer<f32>>(
                    query, args.k, &config,
                ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f16_standard(args: &Args) {
    println!("Reading Queries");
    let (queries_vec, d) = read_numpy_f32_flatten_2d(args.query_file.clone());
    let queries_vec = queries_vec.into_iter().map(f16::from_f32).collect();
    let queries = DenseDataset::from_vec(
        queries_vec,
        d,
        PlainQuantizer::<f16>::new(d, DistanceType::Euclidean),
    );

    let index: HNSW<DenseDataset<PlainQuantizer<f16>, Vec<f16>>, PlainQuantizer<f16>, Graph> =
        IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index.search::<DenseDataset<PlainQuantizer<f16>, Vec<f16>>, PlainQuantizer<f16>>(
                    query, args.k, &config,
                ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_plain_f16_fixed(args: &Args) {
    println!("Reading Queries");
    let (queries_vec, d) = read_numpy_f32_flatten_2d(args.query_file.clone());
    let queries_vec = queries_vec.into_iter().map(f16::from_f32).collect();
    let queries = DenseDataset::from_vec(
        queries_vec,
        d,
        PlainQuantizer::<f16>::new(d, DistanceType::Euclidean),
    );

    let index: HNSW<
        DenseDataset<PlainQuantizer<f16>, Vec<f16>>,
        PlainQuantizer<f16>,
        GraphFixedDegree,
    > = IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index.search::<DenseDataset<PlainQuantizer<f16>, Vec<f16>>, PlainQuantizer<f16>>(
                    query, args.k, &config,
                ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_pq_standard(args: &Args) {
    match args.m_pq {
        4 => search_dense_pq_typed_standard::<4>(args),
        8 => search_dense_pq_typed_standard::<8>(args),
        16 => search_dense_pq_typed_standard::<16>(args),
        32 => search_dense_pq_typed_standard::<32>(args),
        48 => search_dense_pq_typed_standard::<48>(args),
        64 => search_dense_pq_typed_standard::<64>(args),
        96 => search_dense_pq_typed_standard::<96>(args),
        128 => search_dense_pq_typed_standard::<128>(args),
        192 => search_dense_pq_typed_standard::<192>(args),
        256 => search_dense_pq_typed_standard::<256>(args),
        384 => search_dense_pq_typed_standard::<384>(args),
        _ => {
            // This should never happen due to proper CLI validation, but keeping error message for safety
            eprintln!("Error: Invalid m_pq value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.");
            std::process::exit(1);
        }
    }
}

fn search_dense_pq_fixed(args: &Args) {
    match args.m_pq {
        4 => search_dense_pq_typed_fixed::<4>(args),
        8 => search_dense_pq_typed_fixed::<8>(args),
        16 => search_dense_pq_typed_fixed::<16>(args),
        32 => search_dense_pq_typed_fixed::<32>(args),
        48 => search_dense_pq_typed_fixed::<48>(args),
        64 => search_dense_pq_typed_fixed::<64>(args),
        96 => search_dense_pq_typed_fixed::<96>(args),
        128 => search_dense_pq_typed_fixed::<128>(args),
        192 => search_dense_pq_typed_fixed::<192>(args),
        256 => search_dense_pq_typed_fixed::<256>(args),
        384 => search_dense_pq_typed_fixed::<384>(args),
        _ => {
            // This should never happen due to proper CLI validation, but keeping error message for safety
            eprintln!("Error: Invalid m_pq value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.");
            std::process::exit(1);
        }
    }
}

fn search_dense_pq_typed_standard<const M: usize>(args: &Args) {
    println!("Reading Queries");
    let (queries_vec, d) = read_numpy_f32_flatten_2d(args.query_file.clone());
    let queries = DenseDataset::from_vec(
        queries_vec,
        d,
        PlainQuantizer::<f32>::new(d, DistanceType::Euclidean),
    );

    let index: HNSW<DenseDataset<ProductQuantizer<M>, Vec<u8>>, ProductQuantizer<M>, Graph> =
        IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index.search::<DenseDataset<PlainQuantizer<f32>, Vec<f32>>, PlainQuantizer<f32>>(
                    query, args.k, &config,
                ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_dense_pq_typed_fixed<const M: usize>(args: &Args) {
    println!("Reading Queries");
    let (queries_vec, d) = read_numpy_f32_flatten_2d(args.query_file.clone());
    let queries = DenseDataset::from_vec(
        queries_vec,
        d,
        PlainQuantizer::<f32>::new(d, DistanceType::Euclidean),
    );

    let index: HNSW<
        DenseDataset<ProductQuantizer<M>, Vec<u8>>,
        ProductQuantizer<M>,
        GraphFixedDegree,
    > = IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index.search::<DenseDataset<PlainQuantizer<f32>, Vec<f32>>, PlainQuantizer<f32>>(
                    query, args.k, &config,
                ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f16_standard(args: &Args) {
    println!("Reading Queries");
    let (components, values, offsets) =
        SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
            args.query_file.as_str(),
            None,
        )
        .unwrap();

    let d = *components.iter().max().unwrap() as usize + 1;

    let queries: SparseDataset<SparsePlainQuantizer<f16>> = SparseDataset::<
        SparsePlainQuantizer<f16>,
    >::from_vecs_f16(
        &components, &values, &offsets, d
    )
    .unwrap();

    let index: HNSW<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>, Graph> =
        IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index
                    .search::<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>>(
                        query, args.k, &config,
                    ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn search_sparse_plain_f16_fixed(args: &Args) {
    println!("Reading Queries");
    let (components, values, offsets) =
        SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
            args.query_file.as_str(),
            None,
        )
        .unwrap();

    let d = *components.iter().max().unwrap() as usize + 1;

    let queries: SparseDataset<SparsePlainQuantizer<f16>> = SparseDataset::<
        SparsePlainQuantizer<f16>,
    >::from_vecs_f16(
        &components, &values, &offsets, d
    )
    .unwrap();

    let index: HNSW<
        SparseDataset<SparsePlainQuantizer<f16>>,
        SparsePlainQuantizer<f16>,
        GraphFixedDegree,
    > = IndexSerializer::load_index(&args.index_file);

    let num_queries = queries.len();
    let config = HNSWSearchParams::new(args.ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for _ in 0..args.n_run {
        for query in queries.iter() {
            let start_time = Instant::now();
            results.extend(
                index
                    .search::<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>>(
                        query, args.k, &config,
                    ),
            );
            let duration_search = start_time.elapsed();
            total_time_search += duration_search.as_micros();
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
    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.chunks_exact(k).enumerate() {
        // Writes results to a file in a parsable format
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1,
            )
            .unwrap();
        }
    }
}
