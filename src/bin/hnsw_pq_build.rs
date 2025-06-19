use std::{fmt::Debug, time::Instant};

use clap::Parser;
use kannolo::plain_quantizer::PlainQuantizer;
use kannolo::pq::ProductQuantizer;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use std::process;

use kannolo::{
    hnsw::graph_index::GraphIndex, hnsw_utils::config_hnsw::ConfigHnsw, Dataset, DistanceType,
    IndexSerializer,
};
use kannolo::{read_numpy_f32_flatten_2d, DenseDataset, Vector1D};

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

    /// The number of subspaces of Product Quantization.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m_pq: usize,

    /// The number of bits of each subspace of Product Quantization.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 8)]
    nbits: usize,

    /// The type of distance to use. Either 'l2' (Euclidean) or 'ip' (Inner product).
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("ip"))]
    metric: String,

    /// The size of the sample of the dataset used for the training of Product Quantization.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100_000)]
    sample_size: usize,
}

fn main() {
    // Parse command line arguments
    let args: Args = Args::parse();

    let data_path = args.data_file;

    let num_neighbors = args.m;
    let ef_construction = args.efc;
    let m_pq = args.m_pq;
    let sample_size = args.sample_size;
    let nbits = args.nbits;

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
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));

    match m_pq {
        8 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<8>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        16 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<16>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        32 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<32>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        48 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<48>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        64 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<64>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        96 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<96>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        128 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<128>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        192 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<192>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        256 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<256>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        384 => {
            let mut rng = StdRng::seed_from_u64(523);
            let mut training_vec: Vec<f32> = Vec::new();
            for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
                training_vec.extend(vec.values_as_slice());
            }
            let training_dataset = DenseDataset::from_vec(
                training_vec,
                dataset.dim(),
                PlainQuantizer::<f32>::new(dataset.dim(), distance),
            );
            let quantizer = ProductQuantizer::<384>::train(&training_dataset, nbits, distance);
            let start_time = Instant::now();
            let index = GraphIndex::from_dataset(&dataset, &config, quantizer);
            let duration = start_time.elapsed();
            println!(
                "Time to build: {} s (before serializing)",
                duration.as_secs()
            );
            let _ = IndexSerializer::save_index(&args.output_file, &index);
        }
        _ => {
            eprintln!("Error: Invalid number of subspaces for Product Quantization. Available values are 8, 16, 32, 64, 96, 128, 192, 256, 384.");
            process::exit(1);
        }
    };
}
