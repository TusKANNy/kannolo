use std::{fmt::Debug, time::Instant};

use clap::Parser;
use kannolo::plain_quantizer::PlainQuantizer;
use std::process;

use kannolo::{
    hnsw::graph_index::GraphIndex, hnsw_utils::config_hnsw::ConfigHnsw, Dataset, DistanceType,
    IndexSerializer,
};
use kannolo::{read_numpy_f32_flatten_2d, DenseDataset};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the dataset file. Only one between data_file and index_file must be provided.
    /// If both are provided, the index_file will be used.
    #[clap(short, long, value_parser)]
    data_file: String,

    /// The output file where to save the index.
    #[clap(short, long, value_parser)]
    output_file: String,

    /// The number of neihbors per node.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    /// The size of the candidate pool at construction time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    efc: usize,

    /// The type of distance to use. Either 'l2' (Euclidean) or 'ip' (Inner product).
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("ip"))]
    metric: String,
}

fn main() {
    // Parse command line arguments
    let args: Args = Args::parse();

    let data_path = args.data_file;

    let num_neighbors = args.m;
    let ef_construction = args.efc;

    println!("Building Index with M: {num_neighbors}, ef_construction: {ef_construction}");

    let distance = match args.metric.as_str() {
        "l2" => DistanceType::Euclidean,
        "ip" => DistanceType::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'l2' and 'ip'.");
            process::exit(1);
        }
    };

    // Set parameters for the HNSW index
    let config = ConfigHnsw::new()
        .num_neighbors(num_neighbors)
        .ef_construction(ef_construction)
        .build();

    let (docs_vec, d) = read_numpy_f32_flatten_2d(data_path.to_string());
    let docs_vec = docs_vec
        .into_iter()
        .map(|x| half::f16::from_f32(x))
        .collect();
    let dataset =
        DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<half::f16>::new(d, distance));

    let quantizer: PlainQuantizer<half::f16> = PlainQuantizer::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}
