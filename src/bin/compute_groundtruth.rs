// compute_groundtruth: exhaustive search binary
// ---------------------------------------------
// This binary performs an exact (exhaustive) search over a dataset. It can be
// used directly for retrieval (exact k-NN) or to compute a ground-truth file
// against which approximate retrieval methods (e.g. HNSW or PQ) can be
// evaluated.

use kannolo::plain_quantizer::PlainQuantizer;
use kannolo::pq::ProductQuantizer;
use kannolo::topk_selectors::{OnlineTopKSelector, TopkHeap};

use half::f16;
use kannolo::sparse_plain_quantizer::SparsePlainQuantizer;
use kannolo::{
    read_numpy_f32_flatten_2d, Dataset, DenseDataset, DistanceType, GrowableDataset, SparseDataset,
    Vector1D,
};

use clap::{Parser, ValueEnum};
use indicatif::ParallelProgressIterator;
use rand::{rngs::StdRng, seq::IteratorRandom, SeedableRng};
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::Write;
use std::process;

#[derive(Debug, Clone, ValueEnum)]
enum VectorType {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, ValueEnum)]
enum QuantizerType {
    Plain,
    Pq,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the dataset file.
    #[clap(short, long, value_parser)]
    input_file: String,

    /// The path of the query file.
    #[clap(short, long, value_parser)]
    queries_file: String,

    /// The type of vectors (dense or sparse).
    #[clap(long, value_enum)]
    #[arg(default_value_t = VectorType::Dense)]
    vector_type: VectorType,

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

    /// The quantizer type (plain or pq).
    #[clap(long, value_enum)]
    #[arg(default_value_t = QuantizerType::Plain)]
    quantizer: QuantizerType,

    /// The number of subspaces for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    m_pq: Option<usize>,

    /// The number of bits per subspace for Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    nbits: Option<usize>,

    /// The size of the sample used for training Product Quantization (only for PQ).
    #[clap(long, value_parser)]
    sample_size: Option<usize>,
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

    // If sparse, only dot-product is supported
    if matches!(args.vector_type, VectorType::Sparse) && matches!(distance, DistanceType::Euclidean)
    {
        eprintln!("Error: Euclidean distance is not supported for sparse datasets.");
        process::exit(1);
    }

    if matches!(args.quantizer, QuantizerType::Pq) {
        if matches!(args.vector_type, VectorType::Sparse) {
            eprintln!("Error: PQ is not supported for sparse vectors.");
            process::exit(1);
        }
        if args.m_pq.is_none() || args.nbits.is_none() || args.sample_size.is_none() {
            eprintln!("Error: m_pq, nbits, and sample_size must be provided for PQ.");
            process::exit(1);
        }
    }

    if matches!(args.vector_type, VectorType::Dense) {
        let (docs_vec, d) = read_numpy_f32_flatten_2d(data_path.to_string());
        let dataset = DenseDataset::from_vec(docs_vec, d, PlainQuantizer::<f32>::new(d, distance));

        let (queries_vec, d) = read_numpy_f32_flatten_2d(queries_path.to_string());
        let queries =
            DenseDataset::from_vec(queries_vec, d, PlainQuantizer::<f32>::new(d, distance));

        println!("N documents: {}", dataset.len());
        println!("N dims: {}", dataset.dim());
        println!("N queries: {}", queries.len());
        println!("N dims: {}", queries.dim());
        match args.quantizer {
            QuantizerType::Plain => {
                let results: Vec<_> = queries
                    .par_iter()
                    .progress_count(queries.len() as u64)
                    .map(|query| {
                        let mut heap = TopkHeap::new(k);
                        dataset.search(query, &mut heap)
                    })
                    .collect();

                let mut output_file = File::create(output_path).unwrap();

                for (query_id, result) in results.iter().enumerate() {
                    // Writes results to a file in a parsable format
                    for (idx, (score, doc_id)) in result.iter().enumerate() {
                        let out_score = *score;
                        writeln!(
                            &mut output_file,
                            "{query_id}\t{doc_id}\t{}\t{score}",
                            idx + 1,
                            score = out_score
                        )
                        .unwrap();
                    }
                }
            }
            QuantizerType::Pq => {
                let m_pq = args.m_pq.unwrap();
                let nbits = args.nbits.unwrap();
                let sample_size = args.sample_size.unwrap();

                match m_pq {
                    4 => compute_groundtruth_pq::<4>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    8 => compute_groundtruth_pq::<8>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    16 => compute_groundtruth_pq::<16>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    32 => compute_groundtruth_pq::<32>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    48 => compute_groundtruth_pq::<48>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    64 => compute_groundtruth_pq::<64>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    96 => compute_groundtruth_pq::<96>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    128 => compute_groundtruth_pq::<128>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    192 => compute_groundtruth_pq::<192>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    256 => compute_groundtruth_pq::<256>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    384 => compute_groundtruth_pq::<384>(
                        &dataset,
                        &queries,
                        k,
                        distance,
                        nbits,
                        sample_size,
                        output_path,
                    ),
                    _ => {
                        eprintln!("Error: Invalid m_pq value. Choose between 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.");
                        process::exit(1);
                    }
                }
            }
        }
    } else {
        // Sparse
        if matches!(args.quantizer, QuantizerType::Pq) {
            unreachable!();
        }

        let (components, values, offsets) =
            SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
                data_path.as_str(),
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

        let (q_components, q_values, q_offsets) =
            SparseDataset::<SparsePlainQuantizer<f16>>::read_bin_file_parts_f16(
                queries_path.as_str(),
                None,
            )
            .unwrap();

        let qd = *q_components.iter().max().unwrap() as usize + 1;

        let queries: SparseDataset<SparsePlainQuantizer<f16>> =
            SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                &q_components,
                &q_values,
                &q_offsets,
                qd,
            )
            .unwrap();

        println!("N documents: {}", dataset.len());
        println!("N dims: {}", dataset.dim());
        println!("N queries: {}", queries.len());
        println!("N dims: {}", queries.dim());

        let results: Vec<_> = queries
            .par_iter()
            .progress_count(queries.len() as u64)
            .map(|query| {
                let mut heap = TopkHeap::new(k);
                dataset.search(query, &mut heap)
            })
            .collect();

        let mut output_file = File::create(output_path).unwrap();

        for (query_id, result) in results.iter().enumerate() {
            // Writes results to a file in a parsable format
            for (idx, (score, doc_id)) in result.iter().enumerate() {
                let out_score = *score;
                writeln!(
                    &mut output_file,
                    "{query_id}\t{doc_id}\t{}\t{score}",
                    idx + 1,
                    score = out_score
                )
                .unwrap();
            }
        }
    }
}

fn compute_groundtruth_pq<const M: usize>(
    dataset: &DenseDataset<PlainQuantizer<f32>, Vec<f32>>,
    queries: &DenseDataset<PlainQuantizer<f32>, Vec<f32>>,
    k: usize,
    metric: DistanceType,
    nbits: usize,
    sample_size: usize,
    output_path: String,
) {
    let mut rng = StdRng::seed_from_u64(523);
    let mut training_vec: Vec<f32> = Vec::new();
    for vec in dataset.iter().choose_multiple(&mut rng, sample_size) {
        training_vec.extend(vec.values_as_slice());
    }
    let training_dataset = DenseDataset::from_vec(
        training_vec,
        dataset.dim(),
        PlainQuantizer::<f32>::new(dataset.dim(), metric),
    );

    let pq = ProductQuantizer::<M>::train(&training_dataset, nbits, metric);

    let mut pq_dataset = DenseDataset::new(pq, dataset.dim());
    for i in 0..dataset.len() {
        pq_dataset.push(&dataset.get(i));
    }

    println!("N documents: {}", pq_dataset.len());
    println!("N dims: {}", pq_dataset.dim());
    println!("N queries: {}", queries.len());
    println!("N dims: {}", queries.dim());

    let results: Vec<_> = queries
        .par_iter()
        .progress_count(queries.len() as u64)
        .map(|query| {
            let mut heap = TopkHeap::new(k);
            pq_dataset.search(query, &mut heap)
        })
        .collect();

    let mut output_file = File::create(output_path).unwrap();

    for (query_id, result) in results.iter().enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            let out_score = *score;
            writeln!(
                &mut output_file,
                "{query_id}\t{doc_id}\t{}\t{score}",
                idx + 1,
                score = out_score
            )
            .unwrap();
        }
    }
}
