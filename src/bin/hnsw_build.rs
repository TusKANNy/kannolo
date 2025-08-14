use std::{fmt::Debug, time::Instant};

use clap::{Parser, ValueEnum};
use half::f16;
use kannolo::graph::{Graph, GraphFixedDegree};
use kannolo::graph_index::GraphIndex;
use kannolo::hnsw::{HNSWBuildParams, HNSW};
use kannolo::plain_quantizer::PlainQuantizer;
use kannolo::pq::ProductQuantizer;
use kannolo::sparse_plain_quantizer::SparsePlainQuantizer;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use std::process;

use kannolo::{read_numpy_f32_flatten_2d, DenseDataset, SparseDataset, Vector1D};
use kannolo::{Dataset, DistanceType, IndexSerializer};

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
    /// The path of the dataset file.
    #[clap(short, long, value_parser)]
    data_file: String,

    /// The output file where to save the index.
    #[clap(short, long, value_parser)]
    output_file: String,

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

    /// The number of neighbors per node.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m: usize,

    /// The size of the candidate pool at construction time.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 150)]
    efc: usize,

    /// The type of distance to use. Either 'l2' (Euclidean) or 'ip' (Inner product).
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("ip"))]
    metric: String,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 16)]
    m_pq: usize,

    /// The number of bits per subspace for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 8)]
    nbits: usize,

    /// The size of the sample used for training Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100_000)]
    sample_size: usize,
}

fn main() {
    // Parse command line arguments
    let args: Args = Args::parse();

    // Validate arguments
    match (&args.vector_type, &args.quantizer) {
        (VectorType::Sparse, QuantizerType::Pq) => {
            eprintln!("Error: PQ quantizer is only available for dense vectors.");
            process::exit(1);
        }
        (VectorType::Dense, QuantizerType::Pq) if matches!(args.precision, Precision::F16) => {
            eprintln!("Warning: PQ always uses f32 precision, ignoring f16 specification.");
        }
        _ => {}
    }

    // Validate m_pq parameter for PQ quantizer
    if matches!(args.quantizer, QuantizerType::Pq) {
        match args.m_pq {
            4 | 8 | 16 | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 384 => {
                // Valid value, continue
            }
            _ => {
                eprintln!("Error: Invalid m_pq value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.");
                process::exit(1);
            }
        }
    }

    let distance = match args.metric.as_str() {
        "l2" => DistanceType::Euclidean,
        "ip" => DistanceType::DotProduct,
        _ => {
            eprintln!("Error: Invalid distance type. Choose between 'l2' and 'ip'.");
            process::exit(1);
        }
    };

    let config = HNSWBuildParams::new(args.m, args.efc, 4, 320);

    println!(
        "Building Index with M: {}, ef_construction: {}",
        args.m, args.efc
    );

    match (
        &args.vector_type,
        &args.quantizer,
        &args.precision,
        &args.graph_type,
    ) {
        // Dense vectors with plain quantizer
        (VectorType::Dense, QuantizerType::Plain, Precision::F32, GraphType::Standard) => {
            build_dense_plain_f32_standard(&args, distance, &config);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F32, GraphType::FixedDegree) => {
            build_dense_plain_f32_fixed(&args, distance, &config);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            build_dense_plain_f16_standard(&args, distance, &config);
        }
        (VectorType::Dense, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            build_dense_plain_f16_fixed(&args, distance, &config);
        }
        // Dense vectors with PQ quantizer (always f32)
        (VectorType::Dense, QuantizerType::Pq, _, GraphType::Standard) => {
            build_dense_pq_standard(&args, distance, &config);
        }
        (VectorType::Dense, QuantizerType::Pq, _, GraphType::FixedDegree) => {
            build_dense_pq_fixed(&args, distance, &config);
        }
        // Sparse vectors with plain quantizer (f16 only)
        (VectorType::Sparse, QuantizerType::Plain, Precision::F16, GraphType::Standard) => {
            build_sparse_plain_f16_standard(&args, distance, &config);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F16, GraphType::FixedDegree) => {
            build_sparse_plain_f16_fixed(&args, distance, &config);
        }
        (VectorType::Sparse, QuantizerType::Plain, Precision::F32, _) => {
            eprintln!("Error: Sparse vectors currently only support f16 precision.");
            process::exit(1);
        }
        // This case is already handled by validation above
        (VectorType::Sparse, QuantizerType::Pq, _, _) => unreachable!(),
    }
}

fn build_dense_plain_f32_standard(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (docs_vec, d) = read_numpy_f32_flatten_2d(args.data_file.clone());
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));
    let quantizer: PlainQuantizer<f32> = PlainQuantizer::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index: HNSW<DenseDataset<PlainQuantizer<f32>, Vec<f32>>, PlainQuantizer<f32>, Graph> =
        HNSW::build_from_dataset(&dataset, quantizer, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_dense_plain_f32_fixed(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (docs_vec, d) = read_numpy_f32_flatten_2d(args.data_file.clone());
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));
    let quantizer: PlainQuantizer<f32> = PlainQuantizer::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index: HNSW<
        DenseDataset<PlainQuantizer<f32>, Vec<f32>>,
        PlainQuantizer<f32>,
        GraphFixedDegree,
    > = HNSW::build_from_dataset(&dataset, quantizer, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_dense_plain_f16_standard(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (docs_vec, d) = read_numpy_f32_flatten_2d(args.data_file.clone());
    let docs_vec = docs_vec.into_iter().map(f16::from_f32).collect();
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f16>::new(d, distance));
    let quantizer: PlainQuantizer<f16> = PlainQuantizer::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index: HNSW<DenseDataset<PlainQuantizer<f16>, Vec<f16>>, PlainQuantizer<f16>, Graph> =
        HNSW::build_from_dataset(&dataset, quantizer, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_dense_plain_f16_fixed(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (docs_vec, d) = read_numpy_f32_flatten_2d(args.data_file.clone());
    let docs_vec = docs_vec.into_iter().map(f16::from_f32).collect();
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f16>::new(d, distance));
    let quantizer: PlainQuantizer<f16> = PlainQuantizer::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index: HNSW<
        DenseDataset<PlainQuantizer<f16>, Vec<f16>>,
        PlainQuantizer<f16>,
        GraphFixedDegree,
    > = HNSW::build_from_dataset(&dataset, quantizer, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_dense_pq_standard(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (docs_vec, d) = read_numpy_f32_flatten_2d(args.data_file.clone());
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));

    match args.m_pq {
        4 => build_dense_pq_typed_standard::<4>(args, &dataset, distance, config),
        8 => build_dense_pq_typed_standard::<8>(args, &dataset, distance, config),
        16 => build_dense_pq_typed_standard::<16>(args, &dataset, distance, config),
        32 => build_dense_pq_typed_standard::<32>(args, &dataset, distance, config),
        48 => build_dense_pq_typed_standard::<48>(args, &dataset, distance, config),
        64 => build_dense_pq_typed_standard::<64>(args, &dataset, distance, config),
        96 => build_dense_pq_typed_standard::<96>(args, &dataset, distance, config),
        128 => build_dense_pq_typed_standard::<128>(args, &dataset, distance, config),
        192 => build_dense_pq_typed_standard::<192>(args, &dataset, distance, config),
        256 => build_dense_pq_typed_standard::<256>(args, &dataset, distance, config),
        384 => build_dense_pq_typed_standard::<384>(args, &dataset, distance, config),
        _ => {
            // This should never happen due to early validation, but keep for safety
            unreachable!("Invalid m_pq value should have been caught earlier");
        }
    }
}

fn build_dense_pq_fixed(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (docs_vec, d) = read_numpy_f32_flatten_2d(args.data_file.clone());
    let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));

    match args.m_pq {
        4 => build_dense_pq_typed_fixed::<4>(args, &dataset, distance, config),
        8 => build_dense_pq_typed_fixed::<8>(args, &dataset, distance, config),
        16 => build_dense_pq_typed_fixed::<16>(args, &dataset, distance, config),
        32 => build_dense_pq_typed_fixed::<32>(args, &dataset, distance, config),
        48 => build_dense_pq_typed_fixed::<48>(args, &dataset, distance, config),
        64 => build_dense_pq_typed_fixed::<64>(args, &dataset, distance, config),
        96 => build_dense_pq_typed_fixed::<96>(args, &dataset, distance, config),
        128 => build_dense_pq_typed_fixed::<128>(args, &dataset, distance, config),
        192 => build_dense_pq_typed_fixed::<192>(args, &dataset, distance, config),
        256 => build_dense_pq_typed_fixed::<256>(args, &dataset, distance, config),
        384 => build_dense_pq_typed_fixed::<384>(args, &dataset, distance, config),
        _ => {
            // This should never happen due to early validation, but keep for safety
            unreachable!("Invalid m_pq value should have been caught earlier");
        }
    }
}

fn build_dense_pq_typed_standard<const M: usize>(
    args: &Args,
    dataset: &DenseDataset<PlainQuantizer<f32>, Vec<f32>>,
    distance: DistanceType,
    config: &HNSWBuildParams,
) {
    let mut rng = StdRng::seed_from_u64(523);
    let mut training_vec: Vec<f32> = Vec::new();
    for vec in dataset.iter().choose_multiple(&mut rng, args.sample_size) {
        training_vec.extend(vec.values_as_slice());
    }
    let training_dataset = DenseDataset::from_vec(
        training_vec,
        dataset.dim(),
        PlainQuantizer::<f32>::new(dataset.dim(), distance),
    );

    let pq = ProductQuantizer::<M>::train(&training_dataset, args.nbits, distance);

    let start_time = Instant::now();
    let index: HNSW<DenseDataset<ProductQuantizer<M>, Vec<u8>>, ProductQuantizer<M>, Graph> =
        HNSW::build_from_dataset(dataset, pq, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_dense_pq_typed_fixed<const M: usize>(
    args: &Args,
    dataset: &DenseDataset<PlainQuantizer<f32>, Vec<f32>>,
    distance: DistanceType,
    config: &HNSWBuildParams,
) {
    let mut rng = StdRng::seed_from_u64(523);
    let mut training_vec: Vec<f32> = Vec::new();
    for vec in dataset.iter().choose_multiple(&mut rng, args.sample_size) {
        training_vec.extend(vec.values_as_slice());
    }
    let training_dataset = DenseDataset::from_vec(
        training_vec,
        dataset.dim(),
        PlainQuantizer::<f32>::new(dataset.dim(), distance),
    );

    let pq = ProductQuantizer::<M>::train(&training_dataset, args.nbits, distance);

    let start_time = Instant::now();
    let index: HNSW<
        DenseDataset<ProductQuantizer<M>, Vec<u8>>,
        ProductQuantizer<M>,
        GraphFixedDegree,
    > = HNSW::build_from_dataset(dataset, pq, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_sparse_plain_f16_standard(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (components, values, offsets) =
        SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
            args.data_file.as_str(),
            None,
        )
        .unwrap();

    let d = *components.iter().max().unwrap() as usize + 1;

    let dataset: SparseDataset<SparsePlainQuantizer<f16>> = SparseDataset::<
        SparsePlainQuantizer<f16>,
    >::from_vecs_f16(
        &components, &values, &offsets, d
    )
    .unwrap();

    let quantizer = SparsePlainQuantizer::<f16>::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index: HNSW<SparseDataset<SparsePlainQuantizer<f16>>, SparsePlainQuantizer<f16>, Graph> =
        HNSW::build_from_dataset(&dataset, quantizer, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}

fn build_sparse_plain_f16_fixed(args: &Args, distance: DistanceType, config: &HNSWBuildParams) {
    let (components, values, offsets) =
        SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
            args.data_file.as_str(),
            None,
        )
        .unwrap();

    let d = *components.iter().max().unwrap() as usize + 1;

    let dataset: SparseDataset<SparsePlainQuantizer<f16>> = SparseDataset::<
        SparsePlainQuantizer<f16>,
    >::from_vecs_f16(
        &components, &values, &offsets, d
    )
    .unwrap();

    let quantizer = SparsePlainQuantizer::<f16>::new(dataset.dim(), distance);

    let start_time = Instant::now();
    let index: HNSW<
        SparseDataset<SparsePlainQuantizer<f16>>,
        SparsePlainQuantizer<f16>,
        GraphFixedDegree,
    > = HNSW::build_from_dataset(&dataset, quantizer, config);
    let duration = start_time.elapsed();
    println!(
        "Time to build: {} s (before serializing)",
        duration.as_secs()
    );

    let _ = IndexSerializer::save_index(&args.output_file, &index);
}
