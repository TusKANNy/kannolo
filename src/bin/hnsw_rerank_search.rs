use std::io::Write;
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use half::f16;
use std::fs::File;

use kannolo::graph::Graph;
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::core::rerank_index::RerankIndex;
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::ScalarDenseSupportedDistance;
use vectorium::encoders::sparse_scalar::ScalarSparseSupportedDistance;
use vectorium::readers::read_seismic_format;
use vectorium::{
    Dataset, DenseMultiVectorView, MultiVectorDataset, PlainMultiVecQuantizer, PlainSparseDataset,
};

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::ReadNpyExt;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The path of the sparse HNSW index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// The sparse query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// The folder path containing multivector data.
    /// For "plain": expects documents.npy, doclens.npy, queries.npy
    /// For "two-levels": additionally expects centroids.npy, pq_centroids.npy, residuals.npy, index_assignment.npy
    #[clap(long, value_parser)]
    multivec_data_folder: String,

    /// The multivector quantizer type: "plain" or "two-levels".
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("plain"))]
    multivector_quantizer: String,

    /// Number of PQ subspaces (M) - required only for "two-levels" quantizer.
    #[clap(long, value_parser)]
    pq_subspaces: Option<usize>,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter for first-stage index.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 40)]
    ef_search: usize,

    /// The distance metric ("euclidean" or "dotproduct").
    #[clap(long, value_parser)]
    distance: String,

    /// Number of candidates to retrieve from first stage.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 100)]
    k_candidates: usize,

    /// Alpha parameter for candidate pruning (optional, range 0-1).
    #[clap(long, value_parser)]
    alpha: Option<f32>,

    /// Beta parameter for early exit during reranking (optional).
    #[clap(long, value_parser)]
    beta: Option<usize>,

    /// Early termination strategy for first-stage search ("none", "distance-adaptive", etc).
    #[clap(long, value_parser)]
    #[arg(default_value_t = String::from("none"))]
    early_termination: String,

    /// Lambda parameter for early termination (optional, used with distance-adaptive).
    #[clap(long, value_parser)]
    lambda: Option<f32>,

    /// Number of runs for timing.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 1)]
    num_runs: usize,
}

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

#[derive(Clone, Copy, Debug)]
enum DistanceKind {
    Euclidean,
    DotProduct,
}

/// Load multivector dataset for plain quantizer
fn load_multivec_dataset_plain(
    data_folder: &str,
) -> MultiVectorDataset<PlainMultiVecQuantizer<f32>> {
    let documents_path = Path::new(data_folder).join("documents.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");

    // Load documents as u16 array (shape: n_tokens x token_dim) and reinterpret as f16
    let documents_file = std::fs::File::open(&documents_path).unwrap_or_else(|e| {
        eprintln!(
            "Error opening documents file at {:?}: {}",
            documents_path, e
        );
        std::process::exit(1);
    });
    let documents_reader = std::io::BufReader::new(documents_file);
    let documents_u16: Array2<u16> = Array2::read_npy(documents_reader).unwrap_or_else(|e| {
        eprintln!("Error reading documents array: {}", e);
        std::process::exit(1);
    });

    let (_n_tokens, token_dim) = documents_u16.dim();

    // Reinterpret u16 as f16 and convert to f32
    let documents_raw = documents_u16.into_raw_vec_and_offset().0;
    let mut documents_flat: Vec<f32> = Vec::with_capacity(documents_raw.len());
    for u16_val in documents_raw {
        let f16_val = f16::from_bits(u16_val);
        documents_flat.push(f32::from(f16_val));
    }

    // Load doclens as int32 array (shape: n_docs)
    let doclens_file = std::fs::File::open(&doclens_path).unwrap_or_else(|e| {
        eprintln!("Error opening doclens file at {:?}: {}", doclens_path, e);
        std::process::exit(1);
    });
    let doclens_reader = std::io::BufReader::new(doclens_file);
    let doclens_array: Array1<i32> = Array1::read_npy(doclens_reader).unwrap_or_else(|e| {
        eprintln!("Error reading doclens array: {}", e);
        std::process::exit(1);
    });

    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Build offsets array from doclens
    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * token_dim);
    }

    // Create encoder
    let encoder = PlainMultiVecQuantizer::new(token_dim);

    MultiVectorDataset::from_raw(documents_flat.into(), offsets.into(), encoder)
}

/// Load multivector dataset for two-level PQ quantizer
fn load_multivec_dataset_pq<const M: usize>(
    data_folder: &str,
) -> MultiVectorDataset<PlainMultiVecQuantizer<f32>> {
    use std::path::Path;

    let coarse_path = Path::new(data_folder).join("centroids.npy");
    let pq_centroids_path = Path::new(data_folder).join("pq_centroids.npy");
    let residuals_path = Path::new(data_folder).join("residuals.npy");
    let doclens_path = Path::new(data_folder).join("doclens.npy");
    let assignment_path = Path::new(data_folder).join("index_assignment.npy");

    // Load coarse centroids (n_centroids, dim) to determine token_dim
    let coarse_file = File::open(&coarse_path).unwrap_or_else(|e| {
        eprintln!("Error opening centroids.npy at {:?}: {}", coarse_path, e);
        std::process::exit(1);
    });
    let coarse_reader = std::io::BufReader::new(coarse_file);
    let coarse_array: Array2<f32> =
        Array2::read_npy(coarse_reader).expect("Cannot read centroids.npy");
    let (n_coarse, token_dim) = coarse_array.dim();
    let coarse_flat: Vec<f32> = coarse_array.into_iter().collect();

    // Load PQ centroids
    let pq_file = File::open(&pq_centroids_path).unwrap_or_else(|e| {
        eprintln!(
            "Error opening pq_centroids.npy at {:?}: {}",
            pq_centroids_path, e
        );
        std::process::exit(1);
    });
    let pq_reader = std::io::BufReader::new(pq_file);
    let pq_array: Array1<f32> = Array1::read_npy(pq_reader).expect("Cannot read pq_centroids.npy");
    let pq_flat = pq_array.to_vec();

    let dsub = token_dim / M;
    const KSUB: usize = 256;

    let mut pq_reconstruction_centroids = Vec::new();
    for m in 0..M {
        let offset = m * KSUB * dsub;
        pq_reconstruction_centroids.extend_from_slice(&pq_flat[offset..offset + KSUB * dsub]);
    }

    // Load doclens
    let doclens_file = File::open(&doclens_path).unwrap_or_else(|e| {
        eprintln!("Error opening doclens.npy at {:?}: {}", doclens_path, e);
        std::process::exit(1);
    });
    let doclens_reader = std::io::BufReader::new(doclens_file);
    let doclens_array: Array1<i32> =
        Array1::read_npy(doclens_reader).expect("Cannot read doclens.npy");
    let doclens: Vec<usize> = doclens_array.iter().map(|&x| x as usize).collect();

    // Load residuals
    let residuals_file = File::open(&residuals_path).unwrap_or_else(|e| {
        eprintln!("Error opening residuals.npy at {:?}: {}", residuals_path, e);
        std::process::exit(1);
    });
    let residuals_reader = std::io::BufReader::new(residuals_file);
    let residuals_array: Array2<u8> =
        Array2::read_npy(residuals_reader).expect("Cannot read residuals.npy");
    let (n_tokens, m_check) = residuals_array.dim();
    assert_eq!(
        m_check, M,
        "residuals.npy has {} subspaces, expected {}",
        m_check, M
    );

    // Load index assignments
    let assignment_file = File::open(&assignment_path).unwrap_or_else(|e| {
        eprintln!(
            "Error opening index_assignment.npy at {:?}: {}",
            assignment_path, e
        );
        std::process::exit(1);
    });
    let assignment_reader = std::io::BufReader::new(assignment_file);
    let assignment_array: Array1<u64> =
        Array1::read_npy(assignment_reader).expect("Cannot read index_assignment.npy");
    assert_eq!(assignment_array.len(), n_tokens);

    // Reconstruct documents from two-level PQ
    let mut reconstructed_tokens = Vec::with_capacity(n_tokens * token_dim);
    for token_idx in 0..n_tokens {
        let coarse_idx = assignment_array[token_idx] as usize;
        assert!(coarse_idx < n_coarse);
        let coarse_offset = coarse_idx * token_dim;

        for subspace_idx in 0..M {
            let code = residuals_array[[token_idx, subspace_idx]];
            let pq_offset = subspace_idx * KSUB * dsub + (code as usize) * dsub;

            for d in 0..dsub {
                let coarse_val = coarse_flat[coarse_offset + subspace_idx * dsub + d];
                let residual_val = pq_reconstruction_centroids[pq_offset + d];
                reconstructed_tokens.push(coarse_val + residual_val);
            }
        }
    }

    let mut offsets = vec![0];
    for &doclen in &doclens {
        offsets.push(offsets.last().unwrap() + doclen * token_dim);
    }

    let encoder = PlainMultiVecQuantizer::new(token_dim);
    MultiVectorDataset::from_raw(
        reconstructed_tokens.into_boxed_slice(),
        offsets.into(),
        encoder,
    )
}

fn load_multivec_queries(data_folder: &str) -> Array3<f32> {
    let queries_path = Path::new(data_folder).join("queries.npy");
    let queries_file = std::fs::File::open(&queries_path).unwrap_or_else(|e| {
        eprintln!("Error opening queries file at {:?}: {}", queries_path, e);
        std::process::exit(1);
    });
    let queries_reader = std::io::BufReader::new(queries_file);

    Array3::read_npy(queries_reader).unwrap_or_else(|e| {
        eprintln!("Error reading queries array: {}", e);
        std::process::exit(1);
    })
}

fn read_seismic_queries<D>(path: &str) -> PlainSparseDataset<u16, f32, D>
where
    D: ScalarSparseSupportedDistance,
{
    read_seismic_format::<u16, f32, D>(path).unwrap_or_else(|e| {
        eprintln!("Error reading sparse queries: {:?}", e);
        std::process::exit(1);
    })
}

fn create_search_config(args: &Args) -> HNSWSearchConfiguration {
    let strategy = match args.early_termination.as_str() {
        "distance-adaptive" => {
            let lambda = args.lambda.unwrap_or(0.0);
            EarlyTerminationStrategy::DistanceAdaptive { lambda }
        }
        "none" | _ => EarlyTerminationStrategy::None,
    };

    HNSWSearchConfiguration::default()
        .with_ef_search(args.ef_search)
        .with_early_termination(strategy)
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut file = File::create(output_path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(file, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}

fn main() {
    let args: Args = Args::parse();

    // Validate quantizer type
    match args.multivector_quantizer.as_str() {
        "plain" => {
            let metric = parse_metric(&args.distance);
            match metric {
                DistanceKind::Euclidean => {
                    if try_search_plain_f16::<SquaredEuclideanDistance>(&args).is_ok() {
                        return;
                    }
                    search_plain_f32::<SquaredEuclideanDistance>(&args);
                }
                DistanceKind::DotProduct => {
                    if try_search_plain_f16::<DotProduct>(&args).is_ok() {
                        return;
                    }
                    search_plain_f32::<DotProduct>(&args);
                }
            }
        }
        "two-levels" => {
            if args.pq_subspaces.is_none() {
                eprintln!("Error: --pq-subspaces is required for 'two-levels' quantizer");
                std::process::exit(1);
            }
            let metric = parse_metric(&args.distance);
            match metric {
                DistanceKind::Euclidean => match args.pq_subspaces.unwrap() {
                    8 => {
                        if try_search_pq_f16::<8, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<8, SquaredEuclideanDistance>(&args)
                    }
                    16 => {
                        if try_search_pq_f16::<16, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<16, SquaredEuclideanDistance>(&args)
                    }
                    32 => {
                        if try_search_pq_f16::<32, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<32, SquaredEuclideanDistance>(&args)
                    }
                    64 => {
                        if try_search_pq_f16::<64, SquaredEuclideanDistance>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<64, SquaredEuclideanDistance>(&args)
                    }
                    m => {
                        eprintln!(
                            "Error: Unsupported number of PQ subspaces: {}. Supported: 8, 16, 32, 64",
                            m
                        );
                        std::process::exit(1);
                    }
                },
                DistanceKind::DotProduct => match args.pq_subspaces.unwrap() {
                    8 => {
                        if try_search_pq_f16::<8, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<8, DotProduct>(&args)
                    }
                    16 => {
                        if try_search_pq_f16::<16, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<16, DotProduct>(&args)
                    }
                    32 => {
                        if try_search_pq_f16::<32, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<32, DotProduct>(&args)
                    }
                    64 => {
                        if try_search_pq_f16::<64, DotProduct>(&args).is_ok() {
                            return;
                        }
                        search_pq_f32::<64, DotProduct>(&args)
                    }
                    m => {
                        eprintln!(
                            "Error: Unsupported number of PQ subspaces: {}. Supported: 8, 16, 32, 64",
                            m
                        );
                        std::process::exit(1);
                    }
                },
            }
        }
        q => {
            eprintln!(
                "Error: Invalid quantizer type '{}'. Choose 'plain' or 'two-levels'.",
                q
            );
            std::process::exit(1);
        }
    }
}

fn try_search_plain_f16<D>(args: &Args) -> Result<(), String>
where
    D: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f16, D>, Graph> =
        <HNSW<PlainSparseDataset<u16, f16, D>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .map_err(|e| format!("Failed to load f16 index: {:?}", e))?;

    let sparse_queries = read_seismic_queries::<D>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_plain(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        return Err(format!(
            "Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        ));
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                args.alpha,
                args.beta,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }

    Ok(())
}

fn search_plain_f32<D>(args: &Args)
where
    D: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f32, D>, Graph> =
        <HNSW<PlainSparseDataset<u16, f32, D>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading HNSW index: {:?}", e);
            std::process::exit(1);
        });

    let sparse_queries = read_seismic_queries::<D>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_plain(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        eprintln!(
            "Error: Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        );
        std::process::exit(1);
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                args.alpha,
                args.beta,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}

fn try_search_pq_f16<const M: usize, D1>(args: &Args) -> Result<(), String>
where
    D1: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f16, D1>, Graph> =
        <HNSW<PlainSparseDataset<u16, f16, D1>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .map_err(|e| format!("Failed to load f16 index for PQ: {:?}", e))?;

    let sparse_queries = read_seismic_queries::<D1>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_pq::<M>(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        return Err(format!(
            "Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        ));
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                args.alpha,
                args.beta,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }

    Ok(())
}

fn search_pq_f32<const M: usize, D1>(args: &Args)
where
    D1: ScalarDenseSupportedDistance + Distance + ScalarSparseSupportedDistance + 'static,
{
    let sparse_index: HNSW<PlainSparseDataset<u16, f32, D1>, Graph> =
        <HNSW<PlainSparseDataset<u16, f32, D1>, Graph> as IndexSerializer>::load_index(
            &args.index_file,
        )
        .unwrap_or_else(|e| {
            eprintln!("Error loading HNSW index: {:?}", e);
            std::process::exit(1);
        });

    let sparse_queries = read_seismic_queries::<D1>(&args.query_file);
    let num_queries = sparse_queries.len();

    let multivec_dataset = load_multivec_dataset_pq::<M>(&args.multivec_data_folder);
    let multivec_queries_3d = load_multivec_queries(&args.multivec_data_folder);
    let (n_queries_mv, _n_tokens_per_query, token_dim) = multivec_queries_3d.dim();

    if num_queries != n_queries_mv {
        eprintln!(
            "Error: Number of sparse queries ({}) does not match multivec queries ({})",
            num_queries, n_queries_mv
        );
        std::process::exit(1);
    }

    let rerank_index = RerankIndex::new(sparse_index, multivec_dataset);
    let search_config = create_search_config(args);
    let mut total_time_search = 0u128;
    let mut results = Vec::<(f32, usize)>::with_capacity(num_queries * args.k);

    for query_idx in 0..num_queries {
        for _ in 0..args.num_runs {
            let sparse_query = sparse_queries.get(query_idx as vectorium::VectorId);
            let multivec_query_2d = multivec_queries_3d.slice(ndarray::s![query_idx, .., ..]);
            let multivec_query_flat: Vec<f32> = multivec_query_2d.iter().copied().collect();
            let multivec_query_view = DenseMultiVectorView::new(&multivec_query_flat, token_dim);

            let start_time = Instant::now();
            let res = rerank_index.search(
                sparse_query,
                multivec_query_view,
                args.k_candidates,
                args.k,
                &search_config,
                args.alpha,
                args.beta,
            );

            results.extend(
                res.into_iter()
                    .map(|scored| (scored.distance.distance(), scored.vector as usize)),
            );
            total_time_search += start_time.elapsed().as_micros();
        }
    }

    let avg_time_search_per_query = total_time_search / (num_queries * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg_time_search_per_query} μs");
    rerank_index.first_stage_index().print_space_usage_bytes();

    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &results, args.k);
    }
}
