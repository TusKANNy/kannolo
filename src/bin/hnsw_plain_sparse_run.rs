use std::io::Write;
use std::{fmt::Debug, time::Instant};

use clap::Parser;
use std::fs::File;

use half::f16;

use kannolo::{
    hnsw::graph_index::GraphIndex, hnsw_utils::config_hnsw::ConfigHnsw,
    sparse_plain_quantizer::SparsePlainQuantizer, Dataset, IndexSerializer, SparseDataset,
};

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

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The list of ef_search values to test separated by commas.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    n_run: usize,
}

fn main() {
    // Parse command line arguments
    let args: Args = Args::parse();

    let query_path = args.query_file;

    let index_path = args.index_file;

    let k = args.k;
    let ef_search = args.ef_search;

    let index: GraphIndex<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>> =
        IndexSerializer::load_index(index_path.as_str());

    let d = index.dim();

    println!("Reading Queries");
    let (components, values, offsets) =
        SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
            query_path.as_str(),
            None,
        )
        .unwrap();

    let queries = SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
        &components,
        &values,
        &offsets,
        d,
    )
    .unwrap();

    println!("Starting search");
    let num_queries = queries.len();

    let mut config = ConfigHnsw::new().build();
    config.set_ef_search(ef_search);

    println!("N queries {num_queries}");

    // Search
    let mut total_time_search = 0;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries);

    for query in queries.iter() {
        let start_time = Instant::now();
        results.extend(
            index.search::<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>>(
                query, k, &config,
            ),
        );
        let duration_search = start_time.elapsed();
        total_time_search += duration_search.as_micros();
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.n_run) as u128;

    println!("[######] Average Query Time: {avg_time_search_per_query} μs");

    index.print_space_usage_byte();

    let output_path = args.output_path.unwrap();
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
