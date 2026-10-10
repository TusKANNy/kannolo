//! Builds a dense two-stage index ([`kannolo::rerank::DenseRerankIndex`]) from a `.npy`
//! collection and saves it to one file, for `hnsw_rerank_search_dense`.
//!
//! The first stage is an HNSW graph built over `f16`, whose dataset is then encoded with PQ,
//! RaBitQ or RaBitQ-ext. The rerank stage is the collection at plain `f16`, or RaBitQ-ext codes
//! with `--rerank-bits`.
//!
//! Example:
//! ```text
//! hnsw_rerank_build_dense --data-file train.npy --output-file idx.bin \
//!   --encoder rabitq-ext --rabitq-total-bits 4 --rerank-bits 8 \
//!   --distance dotproduct --graph-type permuted --m 32 --ef-construction 200
//! ```

use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use kannolo::rerank::{DenseEncoder, DenseRerankConfig, DenseRerankIndex, GraphLayout, Result};
use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::readers::read_npy_f32;
use vectorium::{Dataset, PlainDenseDataset};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum EncoderArg {
    Pq,
    Rabitq,
    RabitqExt,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum GraphTypeArg {
    Standard,
    Permuted,
    Streamvbyte,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// The collection, an `f32` `.npy` of shape (n, dim).
    #[clap(short, long, value_parser)]
    data_file: String,

    /// Where to write the two-stage index.
    #[clap(short, long, value_parser)]
    output_file: String,

    /// The first-stage encoder.
    #[clap(long, value_enum)]
    encoder: EncoderArg,

    /// PQ subspace count, for `--encoder pq`.
    #[clap(long, value_parser, default_value_t = 32)]
    pq_subspaces: usize,

    /// Bits per component, for `--encoder rabitq-ext` (2, 4 or 8).
    #[clap(long, value_parser, default_value_t = 4)]
    rabitq_total_bits: u32,

    /// The rerank stage's encoding: 0 for plain f16, or 2, 4 or 8 for RaBitQ-ext codes.
    #[clap(long, value_parser, default_value_t = 0)]
    rerank_bits: u32,

    /// Encode `x − x̂` in the rerank stage and sum the two scores. Needs `--rerank-bits` 2, 4 or 8,
    /// a RaBitQ or RaBitQ-ext first stage, and dot product.
    #[clap(long, action)]
    residual_rerank: bool,

    #[clap(long, value_enum, default_value_t = GraphTypeArg::Standard)]
    graph_type: GraphTypeArg,

    /// `dotproduct` or `euclidean`.
    #[clap(long, value_parser, default_value_t = String::from("dotproduct"))]
    distance: String,

    #[clap(long, value_parser, default_value_t = 32)]
    m: usize,

    #[clap(long, value_parser, default_value_t = 200)]
    ef_construction: usize,
}

fn main() {
    let args = Args::parse();

    let cfg = DenseRerankConfig {
        m: args.m,
        ef_construction: args.ef_construction,
        graph: match args.graph_type {
            GraphTypeArg::Standard => GraphLayout::Standard,
            GraphTypeArg::Permuted => GraphLayout::Permuted,
            GraphTypeArg::Streamvbyte => GraphLayout::Compressed,
        },
        encoder: match args.encoder {
            EncoderArg::Pq => DenseEncoder::Pq {
                subspaces: args.pq_subspaces,
            },
            // The query width is a search parameter; the stored codes are 1 bit.
            EncoderArg::Rabitq => DenseEncoder::Rabitq { query_bits: 1 },
            EncoderArg::RabitqExt => DenseEncoder::RabitqExt {
                total_bits: args.rabitq_total_bits,
            },
        },
        rerank_bits: args.rerank_bits,
        residual_rerank: args.residual_rerank,
    };

    // The metric only labels the dataset type here; the raw values are the same either way.
    let data: PlainDenseDataset<f32, DotProduct> =
        read_npy_f32(&args.data_file).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {e:?}", args.data_file);
            process::exit(1);
        });

    let (values, dim) = (data.values(), data.output_dim());
    let start_time = Instant::now();
    match args.distance.as_str() {
        "dotproduct" | "ip" => finish(
            DenseRerankIndex::<DotProduct>::build(values, dim, &cfg),
            start_time,
            &args.output_file,
        ),
        "euclidean" | "l2" => finish(
            DenseRerankIndex::<SquaredEuclideanDistance>::build(values, dim, &cfg),
            start_time,
            &args.output_file,
        ),
        d => {
            eprintln!("Error: invalid --distance {d}. Choose 'euclidean' or 'dotproduct'.");
            process::exit(1);
        }
    }
}

fn finish<D: Distance>(built: Result<DenseRerankIndex<D>>, start_time: Instant, output_file: &str) {
    let index = built.unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });
    println!(
        "Time to build: {} s (before serializing)",
        start_time.elapsed().as_secs()
    );
    println!("[######] Space usage: {} bytes", index.space_usage_bytes());

    index.save(output_file).unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        process::exit(1);
    });
}
