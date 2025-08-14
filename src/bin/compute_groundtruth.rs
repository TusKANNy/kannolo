use kannolo::plain_quantizer::PlainQuantizer;
use kannolo::topk_selectors::{OnlineTopKSelector, TopkHeap};

use kannolo::{read_numpy_f32_flatten_2d, Dataset, DenseDataset, DistanceType};

use clap::Parser;
use indicatif::ProgressIterator;
use std::fs::File;
use std::io::Write;
use std::process;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the dataset file.
    #[clap(short, long, value_parser)]
    input_file: String,

    /// The path of the query file.
    #[clap(short, long, value_parser)]
    queries_file: String,

    /// The number of neihbors to retrieve.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The type of distance to use. Either 'l2' (Euclidean) or 'ip' (Inner product).
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("l2"))]
    metric: String,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,
}

fn main() {
    // Parse command line arguments
    let args: Args = Args::parse();

    let data_path = args.input_file;
    let queries_path = args.queries_file;
    let output_path = args.output_path.unwrap();
    let k = args.k;

    let distance = match args.metric.as_str() {
        "l2" => DistanceType::Euclidean,
        "ip" => DistanceType::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'l2' and 'ip'.");
            process::exit(1);
        }
    };

    let (docs_vec, d) = read_numpy_f32_flatten_2d(data_path.to_string());
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));

    let (queries_vec, d) = read_numpy_f32_flatten_2d(queries_path.to_string());
    let queries = DenseDataset::from_vec(queries_vec, d, PlainQuantizer::<f32>::new(d, distance));

    println!("N documents: {}", dataset.len());
    println!("N dims: {}", dataset.dim());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.dim());

    let results: Vec<_> = queries
        .iter()
        .progress_count(queries.len() as u64)
        .map(|query| {
            let mut heap = TopkHeap::new(k);
            dataset.search(query, &mut heap);
            heap.topk()
        })
        .collect();

    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.iter().enumerate() {
        // Writes results to a file in a parsable format
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1
            )
            .unwrap();
        }
    }
}
