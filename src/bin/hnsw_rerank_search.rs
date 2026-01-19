// IMPORTANT NOTE: graph queries and rerank queries MUST HAVE THE SAME ORDERING

use std::fs::File;
use std::io::Write;
use std::path::Path;

use clap::{Parser, ValueEnum};
use half::f16;
use kannolo::graph::{Graph, GraphFixedDegree};
use kannolo::hnsw::{HNSWSearchParams, HNSW};
use kannolo::multivector_plain_quantizer::MultiVectorPlainQuantizer;
use kannolo::quantizers::multivector_product_quantizer::MultiVectorProductQuantizer;
use kannolo::quantizers::pq::ProductQuantizer;
use kannolo::quantizers::two_level_quantizer::TwoLevelQuantizer;
use kannolo::rerank_index::RerankIndex;
use kannolo::sparse_plain_quantizer::SparsePlainQuantizer;
use kannolo::{
    read_numpy_1d_usize, read_numpy_f32_flatten_3d, read_numpy_u16_as_f16_flatten_2d, Dataset,
    DistanceType, GrowableDataset, IndexSerializer, MultiVector, MultiVectorDataset, SparseDataset,
    VectorType,
};
use ndarray::{Array2, Array3};
use ndarray_npy::ReadNpyExt;

// First-level graph is always sparse and uses f16 precision.

#[derive(Debug, Clone, ValueEnum)]
enum QuantizerType {
    Plain,
    Pq,
}

#[derive(Parser, Debug)]
#[clap(author, about, long_about = None)]
struct Args {
    /// The path of the index.
    #[clap(short, long, value_parser)]
    index_file: String,

    /// Path to the multivector data folder. Files are inferred from this folder depending on `--multivec-type`.
    ///
    /// Plain expects: documents.npy, doclens.npy, queries.npy
    /// PQ expects: documents_codes.npy, centroids.npy, queries.npy, doclens.npy
    /// TwoLevels expects: centroids.npy, index_assignment.npy, pq_centroids.npy, residuals.npy, queries.npy, doclens.npy
    #[clap(long, value_parser)]
    multivec_data_dir: String,

    /// The type of vectors (dense or sparse).
    // First-level graph is always sparse f16; no CLI option needed.

    /// Type of multivector rerank data: plain, pq, two_levels
    /// Accepts string values such as: "plain", "pq", "two_levels" (also accepts "two-levels" or "twolevels").
    #[clap(long, value_parser, default_value = "plain")]
    multivec_type: String,

    /// The query file.
    #[clap(short, long, value_parser)]
    query_file: String,

    /// Vector dimension (for multivector search).
    #[clap(long, value_parser)]
    #[arg(default_value_t = 128)]
    vector_dim: usize,

    /// Number of candidates to retrieve from graph index
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 100)]
    k_candidates: usize,

    /// The number of top-k results to retrieve.
    #[clap(short, long, value_parser)]
    #[arg(default_value_t = 10)]
    k: usize,

    /// The ef_search parameter.
    #[clap(long, value_parser)]
    #[arg(default_value_t = 800)]
    ef_search: usize,

    /// The output file to write the results.
    #[clap(short, long, value_parser)]
    output_path: Option<String>,

    /// Optional pruning parameter alpha (float). Passed to rerank evaluator.
    #[clap(long, value_parser)]
    alpha: Option<f32>,

    /// Optional early-exit parameter beta (integer). Passed to rerank evaluator.
    #[clap(long, value_parser)]
    beta: Option<usize>,

    /// The graph type, either "variable" or "fixed" (default: variable)
    #[clap(long, value_parser, default_value = "variable")]
    graph_type: String,
}

fn write_results_to_file(output_path: &str, results: &[(f32, usize)], k: usize) {
    let mut output_file = File::create(output_path).unwrap();

    for (query_idx, result) in results.chunks_exact(k).enumerate() {
        for (idx, (score, doc_id)) in result.iter().enumerate() {
            writeln!(
                &mut output_file,
                "{}\t{}\t{}\t{}",
                query_idx,
                doc_id,
                idx + 1,
                score
            )
            .unwrap();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== kANNolo Rerank Index Search ===");
    println!("Graph: Sparse f16 (fixed)");
    println!("Graph type: {:?}", args.graph_type);
    println!("Multivector: {}", args.multivec_type);
    println!("k_candidates: {}, k: {}", args.k_candidates, args.k);
    println!("ef_search: {}", args.ef_search);
    println!();

    // Load multivector queries
    println!("Loading multivector queries...");
    // Determine multivector queries file: always use `<multivec_data_dir>/queries.npy`.
    let queries_3d_path = Path::new(&args.multivec_data_dir)
        .join("queries.npy")
        .to_str()
        .unwrap()
        .to_string();
    let (queries_vec, query_total_dim, num_queries) = read_numpy_f32_flatten_3d(queries_3d_path);
    let num_vectors_per_query = query_total_dim / args.vector_dim;

    // Build multivector queries from owned Vec data
    let multivec_queries: Vec<MultiVector<Vec<f32>>> = queries_vec
        .chunks_exact(query_total_dim)
        .map(|chunk| MultiVector::new(chunk.to_vec(), num_vectors_per_query, args.vector_dim))
        .collect();

    println!("✓ Loaded {} multivector queries", num_queries);

    // Two-level will use the same query for both levels (no separate residual query file needed)

    // Setup search parameters
    let search_params = HNSWSearchParams::new(args.ef_search);

    println!("\n=== Starting Search ===");

    let results: Vec<Vec<(f32, usize)>> = match args.multivec_type.as_str() {
        // Plain multivector rerank (first-level graph is sparse f16)
        "plain" => {
            println!("Loading sparse F16 graph index...");

            println!("Loading multivector documents (u16 numpy array reinterpreted as f16)...");
            let multivec_docs_path = Path::new(&args.multivec_data_dir)
                .join("documents.npy")
                .to_str()
                .unwrap()
                .to_string();
            let (multivec_data_f16, vector_dim_from_file) =
                read_numpy_u16_as_f16_flatten_2d(multivec_docs_path);
            if vector_dim_from_file != args.vector_dim {
                return Err(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    args.vector_dim, vector_dim_from_file
                )
                .into());
            }

            println!("Loading document lengths...");
            let doc_lens_path = Path::new(&args.multivec_data_dir)
                .join("doclens.npy")
                .to_str()
                .unwrap()
                .to_string();
            let doc_lens = read_numpy_1d_usize(doc_lens_path);

            println!("Creating multivector dataset...");
            let quantizer =
                MultiVectorPlainQuantizer::<f16>::new(args.vector_dim, DistanceType::DotProduct);
            let mut rerank_dataset = MultiVectorDataset::new(quantizer, args.vector_dim);

            let mut vector_start = 0;
            for &doc_len in &doc_lens {
                if doc_len > 0 {
                    let vector_end = vector_start + doc_len;
                    let doc_start = vector_start * args.vector_dim;
                    let doc_end = vector_end * args.vector_dim;
                    let doc_vectors = &multivec_data_f16[doc_start..doc_end];

                    let multivec = MultiVector::new(doc_vectors, doc_len, args.vector_dim);
                    rerank_dataset.push(&multivec);
                    vector_start = vector_end;
                }
            }

            println!("Dataset created with {} documents", rerank_dataset.len());

            println!("Creating RerankIndex...");

            // Create index & rerank index depending on graph type
            match args.graph_type.as_str() {
                "fixed" => {
                    let graph_index: HNSW<
                        SparseDataset<SparsePlainQuantizer<f16>>,
                        SparsePlainQuantizer<f16>,
                        GraphFixedDegree,
                    > = IndexSerializer::load_index(&args.index_file);
                    let rerank_index: RerankIndex<_, _, _, GraphFixedDegree, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                    // Load sparse queries
                    println!("Loading sparse queries...");
                    let (components, values, offsets) =
                        SparseDataset::<SparsePlainQuantizer<f32>>::read_bin_file_parts_f32(
                            &args.query_file,
                            None,
                        )?;
                    let d = *components.iter().max().unwrap() as usize + 1;
                    let values_f16: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
                    let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                        SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                            &components,
                            &values_f16,
                            &offsets,
                            d,
                        )?;

                    // Perform rerank search for each query
                    let mut total_search_time = std::time::Duration::new(0, 0);
                    let results: Vec<_> = (0..num_queries)
                        .map(|i| {
                            let graph_query = query_dataset.get(i);
                            let multivec_query = &multivec_queries[i];

                            // Convert f32 multivec query to f16 to match dataset
                            let query_f16_data: Vec<f16> = multivec_query
                                .values_as_slice()
                                .iter()
                                .map(|&x| f16::from_f32(x))
                                .collect();
                            let query_f16 = MultiVector::new(
                                query_f16_data.as_slice(),
                                multivec_query.num_vectors(),
                                args.vector_dim,
                            );

                            let search_start = std::time::Instant::now();
                            let (result, _, _) = rerank_index.search::<SparseDataset<
                                SparsePlainQuantizer<f16>,
                            >, SparsePlainQuantizer<f16>, _>(
                                graph_query,
                                query_f16,
                                args.k_candidates,
                                args.k,
                                &search_params,
                                args.alpha,
                                args.beta,
                            );
                            total_search_time += search_start.elapsed();
                            result
                        })
                        .collect();

                    println!(
                        "[######] Average Query Time: {} μs",
                        total_search_time.as_micros() as u128 / num_queries as u128
                    );

                    results
                }
                "variable" => {
                    let graph_index: HNSW<
                        SparseDataset<SparsePlainQuantizer<f16>>,
                        SparsePlainQuantizer<f16>,
                        Graph,
                    > = IndexSerializer::load_index(&args.index_file);
                    let rerank_index: RerankIndex<_, _, _, Graph, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                    // Load sparse queries
                    println!("Loading sparse queries...");
                    let (components, values, offsets) =
                        SparseDataset::<SparsePlainQuantizer<f32>>::read_bin_file_parts_f32(
                            &args.query_file,
                            None,
                        )?;
                    let d = *components.iter().max().unwrap() as usize + 1;
                    let values_f16: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
                    let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                        SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                            &components,
                            &values_f16,
                            &offsets,
                            d,
                        )?;

                    // Perform rerank search for each query
                    let mut total_search_time = std::time::Duration::new(0, 0);
                    let results: Vec<_> = (0..num_queries)
                        .map(|i| {
                            let graph_query = query_dataset.get(i);
                            let multivec_query = &multivec_queries[i];

                            // Convert f32 multivec query to f16 to match dataset
                            let query_f16_data: Vec<f16> = multivec_query
                                .values_as_slice()
                                .iter()
                                .map(|&x| f16::from_f32(x))
                                .collect();
                            let query_f16 = MultiVector::new(
                                query_f16_data.as_slice(),
                                multivec_query.num_vectors(),
                                args.vector_dim,
                            );

                            let search_start = std::time::Instant::now();
                            let (result, _, _) = rerank_index.search::<SparseDataset<
                                SparsePlainQuantizer<f16>,
                            >, SparsePlainQuantizer<f16>, _>(
                                graph_query,
                                query_f16,
                                args.k_candidates,
                                args.k,
                                &search_params,
                                args.alpha,
                                args.beta,
                            );
                            total_search_time += search_start.elapsed();
                            result
                        })
                        .collect();

                    println!(
                        "[######] Average Query Time: {} μs",
                        total_search_time.as_micros() as u128 / num_queries as u128
                    );

                    results
                }
                // Branch if not fixed/variable graph type
                _ => {
                    return Err(format!(
                        "Unknown graph_type: {} (expected 'variable' or 'fixed')",
                        args.graph_type
                    )
                    .into());
                }
            }
        }

        // PQ multivector rerank (first-level graph is sparse f16)
        "pq" => {
            println!("Loading sparse F16 graph index...");

            // Use data dir for PQ components
            let pq_dir = Path::new(&args.multivec_data_dir);
            println!("Loading PQ components from: {}", pq_dir.display());

            // Load PQ centroids
            let centroids_path = pq_dir.join("centroids.npy").to_str().unwrap().to_string();
            println!("  Loading centroids from: {}", centroids_path);
            let centroids_file = File::open(&centroids_path)?;
            let centroids_3d: Array3<f32> = Array3::<f32>::read_npy(centroids_file)?;

            // Calculate PQ parameters from centroids shape
            let centroids_shape = centroids_3d.shape();
            let m = centroids_shape[0]; // number of subspaces
            let k = centroids_shape[1]; // codebook size per subspace
            let subvector_dim = centroids_shape[2]; // dimension per subspace
            println!(
                "  PQ parameters: m={}, k={}, subvector_dim={}",
                m, k, subvector_dim
            );

            // Flatten centroids to the format expected by ProductQuantizer
            let centroids_flat = centroids_3d.into_raw_vec();

            // Load PQ codes
            let codes_path = pq_dir
                .join("documents_codes.npy")
                .to_str()
                .unwrap()
                .to_string();
            println!("  Loading codes from: {}", codes_path);
            let codes_file = File::open(&codes_path)?;
            let codes_2d: Array2<u8> = Array2::<u8>::read_npy(codes_file)?;
            let codes_flat = codes_2d.into_raw_vec();

            // Load document lengths
            println!("Loading document lengths...");
            let doc_lens_path = Path::new(&args.multivec_data_dir)
                .join("doclens.npy")
                .to_str()
                .unwrap()
                .to_string();
            let doc_lens = read_numpy_1d_usize(doc_lens_path);

            // Create ProductQuantizer from pretrained centroids
            let product_quantizer = ProductQuantizer::<64>::from_pretrained(
                args.vector_dim,
                8, // nbits, typically 8 for u8 codes
                centroids_flat,
                DistanceType::DotProduct,
            );

            // Create MultiVectorProductQuantizer
            let multivector_quantizer = MultiVectorProductQuantizer::<64, f16>::from_pretrained(
                product_quantizer,
                args.vector_dim,
                DistanceType::DotProduct,
            );
            let mut rerank_dataset =
                MultiVectorDataset::new(multivector_quantizer, args.vector_dim);

            // Populate the dataset with pre-quantized codes
            let mut vector_start = 0;
            for &doc_len in &doc_lens {
                let vector_end = vector_start + doc_len;

                // Extract PQ codes for this document
                let code_start = vector_start * m;
                let code_end = vector_end * m;
                let doc_codes = &codes_flat[code_start..code_end];

                // Add this document to the dataset using pre-quantized codes
                rerank_dataset.push_quantized(doc_codes, doc_len);
                vector_start = vector_end;
            }

            println!("Dataset created with {} documents", rerank_dataset.len());

            println!("Creating RerankIndex...");

            match args.graph_type.as_str() {
                "fixed" => {
                    let graph_index: HNSW<
                        SparseDataset<SparsePlainQuantizer<f16>>,
                        SparsePlainQuantizer<f16>,
                        GraphFixedDegree,
                    > = IndexSerializer::load_index(&args.index_file);
                    let rerank_index: RerankIndex<_, _, _, GraphFixedDegree, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                    // Load sparse queries
                    println!("Loading sparse queries...");
                    let (components, values, offsets) =
                        SparseDataset::<SparsePlainQuantizer<f32>>::read_bin_file_parts_f32(
                            &args.query_file,
                            None,
                        )?;
                    let d = *components.iter().max().unwrap() as usize + 1;
                    let values_f16: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
                    let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                        SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                            &components,
                            &values_f16,
                            &offsets,
                            d,
                        )?;

                    // Perform rerank search for each query
                    let mut total_search_time = std::time::Duration::new(0, 0);
                    let results: Vec<_> = (0..num_queries)
                        .map(|i| {
                            let graph_query = query_dataset.get(i);
                            let multivec_query = &multivec_queries[i];

                            // Convert f32 multivec query to f16 to match dataset
                            let query_f16_data: Vec<f16> = multivec_query
                                .values_as_slice()
                                .iter()
                                .map(|&x| f16::from_f32(x))
                                .collect();
                            let query_f16 = MultiVector::new(
                                query_f16_data.as_slice(),
                                multivec_query.num_vectors(),
                                args.vector_dim,
                            );

                            let search_start = std::time::Instant::now();
                            let (result, _, _) = rerank_index.search::<SparseDataset<
                                SparsePlainQuantizer<f16>,
                            >, SparsePlainQuantizer<f16>, _>(
                                graph_query,
                                query_f16,
                                args.k_candidates,
                                args.k,
                                &search_params,
                                args.alpha,
                                args.beta,
                            );
                            total_search_time += search_start.elapsed();
                            result
                        })
                        .collect();

                    println!(
                        "[######] Average Query Time: {} μs",
                        total_search_time.as_micros() as u128 / num_queries as u128
                    );

                    results
                }
                "variable" => {
                    let graph_index: HNSW<
                        SparseDataset<SparsePlainQuantizer<f16>>,
                        SparsePlainQuantizer<f16>,
                        Graph,
                    > = IndexSerializer::load_index(&args.index_file);
                    let rerank_index: RerankIndex<_, _, _, Graph, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                    // Load sparse queries
                    println!("Loading sparse queries...");
                    let (components, values, offsets) =
                        SparseDataset::<SparsePlainQuantizer<f32>>::read_bin_file_parts_f32(
                            &args.query_file,
                            None,
                        )?;
                    let d = *components.iter().max().unwrap() as usize + 1;
                    let values_f16: Vec<f16> = values.iter().map(|&x| f16::from_f32(x)).collect();
                    let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                        SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                            &components,
                            &values_f16,
                            &offsets,
                            d,
                        )?;

                    // Perform rerank search for each query
                    let mut total_search_time = std::time::Duration::new(0, 0);
                    let results: Vec<_> = (0..num_queries)
                        .map(|i| {
                            let graph_query = query_dataset.get(i);
                            let multivec_query = &multivec_queries[i];

                            // Convert f32 multivec query to f16 to match dataset
                            let query_f16_data: Vec<f16> = multivec_query
                                .values_as_slice()
                                .iter()
                                .map(|&x| f16::from_f32(x))
                                .collect();
                            let query_f16 = MultiVector::new(
                                query_f16_data.as_slice(),
                                multivec_query.num_vectors(),
                                args.vector_dim,
                            );

                            let search_start = std::time::Instant::now();
                            let (result, _, _) = rerank_index.search::<SparseDataset<
                                SparsePlainQuantizer<f16>,
                            >, SparsePlainQuantizer<f16>, _>(
                                graph_query,
                                query_f16,
                                args.k_candidates,
                                args.k,
                                &search_params,
                                args.alpha,
                                args.beta,
                            );
                            total_search_time += search_start.elapsed();
                            result
                        })
                        .collect();

                    println!(
                        "[######] Average Query Time: {} μs",
                        total_search_time.as_micros() as u128 / num_queries as u128
                    );

                    results
                }
                // Branch if not fixed/variable graph type
                _ => {
                    return Err(format!(
                        "Unknown graph_type: {} (expected 'variable' or 'fixed')",
                        args.graph_type
                    )
                    .into());
                }
            }
        }

        // Two-level multivector rerank (first-level coarse + PQ residuals)
        "two_levels" | "two-levels" | "twolevels" => {
            println!("Loading two-level rerank components...");

            // Use data dir for PQ centroids and residuals
            let data_dir_path = Path::new(&args.multivec_data_dir);
            let pq_centroids_path = data_dir_path
                .join("pq_centroids.npy")
                .to_str()
                .unwrap()
                .to_string();
            let pq_codes_path = data_dir_path
                .join("residuals.npy")
                .to_str()
                .unwrap()
                .to_string();

            // Load PQ codes (residuals)
            println!("  Loading PQ codes from: {}", pq_codes_path);
            let codes_file = File::open(&pq_codes_path)?;
            let codes_2d: Array2<u8> = Array2::<u8>::read_npy(codes_file)?;
            let codes_flat = codes_2d.clone().into_raw_vec();
            let m = codes_2d.ncols();

            // Load first-level centroids from data dir
            let fl_path = data_dir_path
                .join("centroids.npy")
                .to_str()
                .unwrap()
                .to_string();
            println!("  Loading first-level centroids from: {}", fl_path);
            let fl_file = File::open(&fl_path)?;
            let centroids_2d: Array2<f32> = Array2::<f32>::read_npy(fl_file)?;
            let (num_centroids, _centroid_dim) = (centroids_2d.nrows(), centroids_2d.ncols());
            let first_level_centroids_flat = centroids_2d.clone().into_raw_vec();

            // Load PQ centroids (support 3D/2D/1D formats)
            println!("  Loading PQ centroids from: {}", pq_centroids_path);
            let pq_centroids_flat: Vec<f32> = {
                let f = File::open(&pq_centroids_path)?;
                match Array3::<f32>::read_npy(f) {
                    Ok(arr3) => arr3.into_raw_vec(),
                    Err(_) => {
                        let f2 = File::open(&pq_centroids_path)?;
                        match Array2::<f32>::read_npy(f2) {
                            Ok(arr2) => arr2.into_raw_vec(),
                            Err(_) => {
                                let f3 = File::open(&pq_centroids_path)?;
                                let arr1: ndarray::Array1<f32> =
                                    ndarray::Array1::<f32>::read_npy(f3)?;
                                arr1.into_raw_vec()
                            }
                        }
                    }
                }
            };

            // Infer PQ parameters
            if args.vector_dim % m != 0 {
                return Err(format!(
                    "vector_dim {} is not divisible by PQ m={}",
                    args.vector_dim, m
                )
                .into());
            }
            let subvector_dim = args.vector_dim / m;
            let total_centroid_vals = pq_centroids_flat.len();
            if total_centroid_vals % (m * subvector_dim) != 0 {
                return Err(format!(
                    "PQ centroids length {} is not divisible by m*subvector_dim {}",
                    total_centroid_vals,
                    m * subvector_dim
                )
                .into());
            }
            let k = total_centroid_vals / (m * subvector_dim);
            println!(
                "  PQ parameters: m={}, k={}, subvector_dim={}",
                m, k, subvector_dim
            );

            // Load document lengths
            println!("Loading document lengths...");
            let doc_lens_path = data_dir_path
                .join("doclens.npy")
                .to_str()
                .unwrap()
                .to_string();
            let doc_lens = read_numpy_1d_usize(doc_lens_path);

            // Load assignments (u32/u64) from data dir
            let assignments_vec: Vec<u32> = {
                let assign_path = data_dir_path
                    .join("index_assignment.npy")
                    .to_str()
                    .unwrap()
                    .to_string();
                let f = File::open(&assign_path)?;
                match ndarray::Array1::<u32>::read_npy(f) {
                    Ok(arr) => arr.into_raw_vec(),
                    Err(_) => {
                        let f2 = File::open(&assign_path)?;
                        let arr64: ndarray::Array1<u64> = ndarray::Array1::<u64>::read_npy(f2)?;
                        let mut v: Vec<u32> = Vec::with_capacity(arr64.len());
                        for &val in arr64.iter() {
                            if val > u64::from(u32::MAX) {
                                return Err(
                                    format!("assignment value {} exceeds u32::MAX", val).into()
                                );
                            }
                            v.push(val as u32);
                        }
                        v
                    }
                }
            };

            // Build TwoLevelQuantizer and dataset. Support PQ m == 16 or m == 32.
            if m == 16 {
                let product_quantizer = ProductQuantizer::<16>::from_pretrained(
                    args.vector_dim,
                    8,
                    pq_centroids_flat.clone(),
                    DistanceType::DotProduct,
                );

                let two_level_quantizer = TwoLevelQuantizer::from_pretrained(
                    first_level_centroids_flat.clone(),
                    num_centroids,
                    args.vector_dim,
                    product_quantizer,
                    DistanceType::DotProduct,
                );

                let mut rerank_dataset =
                    MultiVectorDataset::new(two_level_quantizer, args.vector_dim);
                let encoded_size_per_vector = m + 4;
                let mut encoded_flat: Vec<u8> =
                    Vec::with_capacity(doc_lens.iter().sum::<usize>() * encoded_size_per_vector);
                let mut vector_start = 0usize;
                for &doc_len in &doc_lens {
                    if doc_len > 0 {
                        let vector_end = vector_start + doc_len;
                        for vec_idx in vector_start..vector_end {
                            let coarse_id = assignments_vec[vec_idx] as u32;
                            encoded_flat.push((coarse_id & 0xFF) as u8);
                            encoded_flat.push(((coarse_id >> 8) & 0xFF) as u8);
                            encoded_flat.push(((coarse_id >> 16) & 0xFF) as u8);
                            encoded_flat.push(((coarse_id >> 24) & 0xFF) as u8);
                            let code_start = vec_idx * m;
                            let code_end = code_start + m;
                            encoded_flat.extend_from_slice(&codes_flat[code_start..code_end]);
                        }
                        let quantized_slice = &encoded_flat[encoded_flat.len()
                            - doc_len * encoded_size_per_vector
                            ..encoded_flat.len()];
                        rerank_dataset.push_quantized(quantized_slice, doc_len);
                        vector_start = vector_end;
                    }
                }

                println!("Dataset created with {} documents", rerank_dataset.len());

                println!("Creating RerankIndex...");

                match args.graph_type.as_str() {
                    "fixed" => {
                        let graph_index: HNSW<
                            SparseDataset<SparsePlainQuantizer<f16>>,
                            SparsePlainQuantizer<f16>,
                            GraphFixedDegree,
                        > = IndexSerializer::load_index(&args.index_file);
                        let rerank_index: RerankIndex<_, _, _, GraphFixedDegree, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                        // Load sparse queries
                        println!("Loading sparse queries...");
                        let (components, values, offsets) = SparseDataset::<
                            SparsePlainQuantizer<f32>,
                        >::read_bin_file_parts_f32(
                            &args.query_file, None
                        )?;
                        let d = *components.iter().max().unwrap() as usize + 1;
                        let values_f16: Vec<f16> =
                            values.iter().map(|&x| f16::from_f32(x)).collect();
                        let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                            SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                                &components,
                                &values_f16,
                                &offsets,
                                d,
                            )?;

                        // Perform rerank search for each query
                        let mut total_search_time = std::time::Duration::new(0, 0);
                        let results: Vec<_> = (0..num_queries)
                            .map(|i| {
                                let graph_query = query_dataset.get(i);
                                let multivec_query = &multivec_queries[i];

                                let first_level_mv = MultiVector::new(
                                    multivec_query.values_as_slice(),
                                    multivec_query.num_vectors(),
                                    args.vector_dim,
                                );

                                let search_start = std::time::Instant::now();
                                let (result, _, _) = rerank_index.search::<SparseDataset<
                                    SparsePlainQuantizer<f16>,
                                >, SparsePlainQuantizer<f16>, _>(
                                    graph_query,
                                    first_level_mv,
                                    args.k_candidates,
                                    args.k,
                                    &search_params,
                                    args.alpha,
                                    args.beta,
                                );
                                total_search_time += search_start.elapsed();
                                result
                            })
                            .collect();

                        println!(
                            "[######] Average Query Time: {} μs",
                            total_search_time.as_micros() as u128 / num_queries as u128
                        );

                        results
                    }
                    "variable" => {
                        let graph_index: HNSW<
                            SparseDataset<SparsePlainQuantizer<f16>>,
                            SparsePlainQuantizer<f16>,
                            Graph,
                        > = IndexSerializer::load_index(&args.index_file);
                        let rerank_index: RerankIndex<_, _, _, Graph, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                        // Load sparse queries
                        println!("Loading sparse queries...");
                        let (components, values, offsets) = SparseDataset::<
                            SparsePlainQuantizer<f32>,
                        >::read_bin_file_parts_f32(
                            &args.query_file, None
                        )?;
                        let d = *components.iter().max().unwrap() as usize + 1;
                        let values_f16: Vec<f16> =
                            values.iter().map(|&x| f16::from_f32(x)).collect();
                        let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                            SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                                &components,
                                &values_f16,
                                &offsets,
                                d,
                            )?;

                        // Perform rerank search for each query
                        let mut total_search_time = std::time::Duration::new(0, 0);
                        let results: Vec<_> = (0..num_queries)
                            .map(|i| {
                                let graph_query = query_dataset.get(i);
                                let multivec_query = &multivec_queries[i];

                                let first_level_mv = MultiVector::new(
                                    multivec_query.values_as_slice(),
                                    multivec_query.num_vectors(),
                                    args.vector_dim,
                                );

                                let search_start = std::time::Instant::now();
                                let (result, _, _) = rerank_index.search::<SparseDataset<
                                    SparsePlainQuantizer<f16>,
                                >, SparsePlainQuantizer<f16>, _>(
                                    graph_query,
                                    first_level_mv,
                                    args.k_candidates,
                                    args.k,
                                    &search_params,
                                    args.alpha,
                                    args.beta,
                                );
                                total_search_time += search_start.elapsed();
                                result
                            })
                            .collect();

                        println!(
                            "[######] Average Query Time: {} μs",
                            total_search_time.as_micros() as u128 / num_queries as u128
                        );

                        results
                    }
                    // Branch if not fixed/variable graph type
                    _ => {
                        return Err(format!(
                            "Unknown graph_type: {} (expected 'variable' or 'fixed')",
                            args.graph_type
                        )
                        .into());
                    }
                }
            } else if m == 32 {
                let product_quantizer = ProductQuantizer::<32>::from_pretrained(
                    args.vector_dim,
                    8,
                    pq_centroids_flat.clone(),
                    DistanceType::DotProduct,
                );

                let two_level_quantizer = TwoLevelQuantizer::from_pretrained(
                    first_level_centroids_flat.clone(),
                    num_centroids,
                    args.vector_dim,
                    product_quantizer,
                    DistanceType::DotProduct,
                );

                let mut rerank_dataset =
                    MultiVectorDataset::new(two_level_quantizer, args.vector_dim);
                let encoded_size_per_vector = m + 4;
                let mut encoded_flat: Vec<u8> =
                    Vec::with_capacity(doc_lens.iter().sum::<usize>() * encoded_size_per_vector);
                let mut vector_start = 0usize;
                for &doc_len in &doc_lens {
                    if doc_len > 0 {
                        let vector_end = vector_start + doc_len;
                        for vec_idx in vector_start..vector_end {
                            let coarse_id = assignments_vec[vec_idx] as u32;
                            encoded_flat.push((coarse_id & 0xFF) as u8);
                            encoded_flat.push(((coarse_id >> 8) & 0xFF) as u8);
                            encoded_flat.push(((coarse_id >> 16) & 0xFF) as u8);
                            encoded_flat.push(((coarse_id >> 24) & 0xFF) as u8);
                            let code_start = vec_idx * m;
                            let code_end = code_start + m;
                            encoded_flat.extend_from_slice(&codes_flat[code_start..code_end]);
                        }
                        let quantized_slice = &encoded_flat[encoded_flat.len()
                            - doc_len * encoded_size_per_vector
                            ..encoded_flat.len()];
                        rerank_dataset.push_quantized(quantized_slice, doc_len);
                        vector_start = vector_end;
                    }
                }

                println!("Dataset created with {} documents", rerank_dataset.len());

                println!("Creating RerankIndex...");

                // Choose graph type when loading index to avoid deserializing into the wrong
                // HNSW specialization (which causes bincode/tag errors).
                match args.graph_type.as_str() {
                    "fixed" => {
                        let graph_index: HNSW<
                            SparseDataset<SparsePlainQuantizer<f16>>,
                            SparsePlainQuantizer<f16>,
                            GraphFixedDegree,
                        > = IndexSerializer::load_index(&args.index_file);
                        let rerank_index: RerankIndex<_, _, _, GraphFixedDegree, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                        // Load sparse queries
                        println!("Loading sparse queries...");
                        let (components, values, offsets) = SparseDataset::<
                            SparsePlainQuantizer<f32>,
                        >::read_bin_file_parts_f32(
                            &args.query_file, None
                        )?;
                        let d = *components.iter().max().unwrap() as usize + 1;
                        let values_f16: Vec<f16> =
                            values.iter().map(|&x| f16::from_f32(x)).collect();
                        let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                            SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                                &components,
                                &values_f16,
                                &offsets,
                                d,
                            )?;

                        // Perform rerank search for each query
                        let mut total_search_time = std::time::Duration::new(0, 0);
                        let results: Vec<_> = (0..num_queries)
                            .map(|i| {
                                let graph_query = query_dataset.get(i);
                                let multivec_query = &multivec_queries[i];

                                let first_level_mv = MultiVector::new(
                                    multivec_query.values_as_slice(),
                                    multivec_query.num_vectors(),
                                    args.vector_dim,
                                );

                                let search_start = std::time::Instant::now();
                                let (result, _, _) = rerank_index.search::<SparseDataset<
                                    SparsePlainQuantizer<f16>,
                                >, SparsePlainQuantizer<f16>, _>(
                                    graph_query,
                                    first_level_mv,
                                    args.k_candidates,
                                    args.k,
                                    &search_params,
                                    args.alpha,
                                    args.beta,
                                );
                                total_search_time += search_start.elapsed();
                                result
                            })
                            .collect();

                        println!(
                            "[######] Average Query Time: {} μs",
                            total_search_time.as_micros() as u128 / num_queries as u128
                        );

                        results
                    }
                    "variable" => {
                        let graph_index: HNSW<
                            SparseDataset<SparsePlainQuantizer<f16>>,
                            SparsePlainQuantizer<f16>,
                            Graph,
                        > = IndexSerializer::load_index(&args.index_file);
                        let rerank_index: RerankIndex<_, _, _, Graph, _, _> = RerankIndex::new(graph_index, rerank_dataset);

                        // Load sparse queries
                        println!("Loading sparse queries...");
                        let (components, values, offsets) = SparseDataset::<
                            SparsePlainQuantizer<f32>,
                        >::read_bin_file_parts_f32(
                            &args.query_file, None
                        )?;
                        let d = *components.iter().max().unwrap() as usize + 1;
                        let values_f16: Vec<f16> =
                            values.iter().map(|&x| f16::from_f32(x)).collect();
                        let query_dataset: SparseDataset<SparsePlainQuantizer<f16>> =
                            SparseDataset::<SparsePlainQuantizer<f16>>::from_vecs_f16(
                                &components,
                                &values_f16,
                                &offsets,
                                d,
                            )?;

                        // Perform rerank search for each query
                        let mut total_search_time = std::time::Duration::new(0, 0);
                        let results: Vec<_> = (0..num_queries)
                            .map(|i| {
                                let graph_query = query_dataset.get(i);
                                let multivec_query = &multivec_queries[i];

                                let first_level_mv = MultiVector::new(
                                    multivec_query.values_as_slice(),
                                    multivec_query.num_vectors(),
                                    args.vector_dim,
                                );

                                let search_start = std::time::Instant::now();
                                let (result, _, _) = rerank_index.search::<SparseDataset<
                                    SparsePlainQuantizer<f16>,
                                >, SparsePlainQuantizer<f16>, _>(
                                    graph_query,
                                    first_level_mv,
                                    args.k_candidates,
                                    args.k,
                                    &search_params,
                                    args.alpha,
                                    args.beta,
                                );
                                total_search_time += search_start.elapsed();
                                result
                            })
                            .collect();

                        println!(
                            "[######] Average Query Time: {} μs",
                            total_search_time.as_micros() as u128 / num_queries as u128
                        );

                        results
                    }
                    _ => {
                        return Err(format!(
                            "Unknown graph_type: {} (expected 'variable' or 'fixed')",
                            args.graph_type
                        )
                        .into());
                    }
                }
            } else {
                return Err(format!(
                    "Unsupported PQ m={} for two-level quantizer; supported: 16 or 32",
                    m
                )
                .into());
            }
        }
        other => {
            return Err(format!(
                "Unknown multivec_type: {} (expected 'plain', 'pq' or 'two_levels')",
                other
            )
            .into());
        }
    };

    // Flatten results for output
    let mut all_results = Vec::new();
    for result in results {
        all_results.extend(result);
    }

    // Write results if output path is provided
    if let Some(output_path) = &args.output_path {
        write_results_to_file(output_path, &all_results, args.k);
    }

    println!("==================================================================");

    Ok(())
}
