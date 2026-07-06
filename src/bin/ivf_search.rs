/// Search an IVF index.
use std::fs::File;
use std::io::Write as IoWrite;
use std::process;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use half::f16;

use kannolo::graph::Graph;
use kannolo::hnsw::{EarlyTerminationStrategy, HNSW, HNSWSearchConfiguration};
use kannolo::ivf::{IVF, IVFSearchParams, ReportedMetric};
use vectorium::IndexSerializer;
use vectorium::core::index::Index;
use vectorium::core::vector::DenseVectorView;
use vectorium::distances::{DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::pq::ProductQuantizer;
use vectorium::readers::read_npy_f32;
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};
use vectorium::{Dataset, DenseDataset, Distance, FlatIndex, PlainDenseDataset, QueryEvaluator};

#[allow(non_camel_case_types)]
type DQ_SE = PlainDenseDataset<f16, SquaredEuclideanDistance>;
#[allow(non_camel_case_types)]
type DQ_DP = PlainDenseDataset<f16, DotProduct>;
type Queries = PlainDenseDataset<f32, SquaredEuclideanDistance>;

#[derive(Debug, Clone, ValueEnum, Default)]
enum ValueTypeArg {
    #[default]
    F32,
    F16,
}

#[derive(Debug, Clone, ValueEnum, Default)]
enum MetricArg {
    #[default]
    Euclidean,
    Dotproduct,
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    #[clap(short, long)]
    index_file: String,
    #[clap(short, long)]
    query_file: String,
    #[clap(short, long)]
    output_path: Option<String>,
    #[clap(short, long, default_value_t = 10)]
    k: usize,
    #[clap(long, default_value_t = 32)]
    n_probe: usize,
    #[clap(long, value_enum, default_value_t = MetricArg::Euclidean)]
    distance: MetricArg,
    #[clap(long, value_enum, default_value_t = ValueTypeArg::F32)]
    value_type: ValueTypeArg,
    #[clap(long, default_value_t = false)]
    hnsw: bool,
    #[clap(long)]
    m_pq: Option<usize>,
    #[clap(long, default_value_t = false)]
    residuals: bool,
    #[clap(long, default_value_t = 40)]
    ef_search: usize,
    #[clap(long, default_value_t = 0.0)]
    lambda: f32,
    #[clap(long, default_value_t = 1)]
    num_runs: usize,
}

fn main() {
    let args = Args::parse();
    let queries: Queries = read_npy_f32(&args.query_file).unwrap_or_else(|e| {
        eprintln!("Error reading queries: {e:?}");
        process::exit(1);
    });
    if args.m_pq.is_some() {
        dispatch_pq(&args, &queries);
    } else {
        dispatch_plain(&args, &queries);
    }
}

// ---------------------------------------------------------------------------
// MakeParams: centroid metric matches DQ type
// ---------------------------------------------------------------------------

trait MakeParams<DQ: Dataset>: Index + Sized {
    fn make_params(nprobe: usize, args: &Args) -> IVFSearchParams<Self>;
}
impl MakeParams<DQ_SE> for FlatIndex<DQ_SE> {
    fn make_params(nprobe: usize, _args: &Args) -> IVFSearchParams<Self> {
        IVFSearchParams {
            nprobe,
            centroid_search_params: (),
        }
    }
}
impl MakeParams<DQ_SE> for HNSW<DQ_SE, Graph> {
    fn make_params(nprobe: usize, args: &Args) -> IVFSearchParams<Self> {
        let ef = args.ef_search.max(nprobe);
        let cfg = HNSWSearchConfiguration::default()
            .with_ef_search(ef)
            .with_early_termination(if args.lambda > 0.0 {
                EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: args.lambda,
                }
            } else {
                EarlyTerminationStrategy::None
            });
        IVFSearchParams {
            nprobe,
            centroid_search_params: cfg,
        }
    }
}
impl MakeParams<DQ_DP> for FlatIndex<DQ_DP> {
    fn make_params(nprobe: usize, _args: &Args) -> IVFSearchParams<Self> {
        IVFSearchParams {
            nprobe,
            centroid_search_params: (),
        }
    }
}
impl MakeParams<DQ_DP> for HNSW<DQ_DP, Graph> {
    fn make_params(nprobe: usize, args: &Args) -> IVFSearchParams<Self> {
        let ef = args.ef_search.max(nprobe);
        let cfg = HNSWSearchConfiguration::default()
            .with_ef_search(ef)
            .with_early_termination(if args.lambda > 0.0 {
                EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: args.lambda,
                }
            } else {
                EarlyTerminationStrategy::None
            });
        IVFSearchParams {
            nprobe,
            centroid_search_params: cfg,
        }
    }
}

// ---------------------------------------------------------------------------
// Plain cluster dispatch
// ---------------------------------------------------------------------------

fn dispatch_plain(args: &Args, queries: &Queries) {
    // Cluster storage is always dot-product; only the centroid metric (DQ) and
    // value type differ. Residual-vs-plain is read from the index at runtime.
    #[allow(non_camel_case_types)]
    type DC32 = PlainDenseDataset<f32, DotProduct>;
    #[allow(non_camel_case_types)]
    type DC16 = PlainDenseDataset<f16, DotProduct>;
    match (&args.distance, &args.value_type, args.hnsw) {
        (MetricArg::Euclidean, ValueTypeArg::F32, false) => {
            run_plain::<DQ_SE, DC32, FlatIndex<DQ_SE>>(args, queries)
        }
        (MetricArg::Euclidean, ValueTypeArg::F32, true) => {
            run_plain::<DQ_SE, DC32, HNSW<DQ_SE, Graph>>(args, queries)
        }
        (MetricArg::Euclidean, ValueTypeArg::F16, false) => {
            run_plain::<DQ_SE, DC16, FlatIndex<DQ_SE>>(args, queries)
        }
        (MetricArg::Euclidean, ValueTypeArg::F16, true) => {
            run_plain::<DQ_SE, DC16, HNSW<DQ_SE, Graph>>(args, queries)
        }
        (MetricArg::Dotproduct, ValueTypeArg::F32, false) => {
            run_plain::<DQ_DP, DC32, FlatIndex<DQ_DP>>(args, queries)
        }
        (MetricArg::Dotproduct, ValueTypeArg::F32, true) => {
            run_plain::<DQ_DP, DC32, HNSW<DQ_DP, Graph>>(args, queries)
        }
        (MetricArg::Dotproduct, ValueTypeArg::F16, false) => {
            run_plain::<DQ_DP, DC16, FlatIndex<DQ_DP>>(args, queries)
        }
        (MetricArg::Dotproduct, ValueTypeArg::F16, true) => {
            run_plain::<DQ_DP, DC16, HNSW<DQ_DP, Graph>>(args, queries)
        }
    }
}

// ---------------------------------------------------------------------------
// PQ cluster dispatch
// ---------------------------------------------------------------------------

fn dispatch_pq(args: &Args, queries: &Queries) {
    // PQ storage is always dot-product; residual-vs-plain is read at runtime.
    // Dispatch only on (centroid metric, HNSW, M).
    macro_rules! pq_arms {
        ($($m:literal),* $(,)?) => {
            match (&args.distance, args.hnsw, args.m_pq) {
                $(
                    (MetricArg::Euclidean, false, Some($m)) => {
                        run_pq::<$m, DQ_SE, FlatIndex<DQ_SE>>(args, queries)
                    }
                    (MetricArg::Euclidean, true, Some($m)) => {
                        run_pq::<$m, DQ_SE, HNSW<DQ_SE, Graph>>(args, queries)
                    }
                    (MetricArg::Dotproduct, false, Some($m)) => {
                        run_pq::<$m, DQ_DP, FlatIndex<DQ_DP>>(args, queries)
                    }
                    (MetricArg::Dotproduct, true, Some($m)) => {
                        run_pq::<$m, DQ_DP, HNSW<DQ_DP, Graph>>(args, queries)
                    }
                )*
                (_, _, Some(m)) => {
                    eprintln!("Unsupported --m-pq {m}");
                    process::exit(1);
                }
                (_, _, None) => unreachable!(),
            }
        };
    }
    pq_arms!(4, 8, 16, 32, 48, 64, 96, 128, 192);
}

// ---------------------------------------------------------------------------
// Plain cluster search
// ---------------------------------------------------------------------------

fn run_plain<DQ, DC, CI>(args: &Args, queries: &Queries)
where
    DQ: Dataset + serde::Serialize + for<'de> serde::Deserialize<'de>,
    DC: Dataset + serde::Serialize + for<'de> serde::Deserialize<'de>,
    CI: Index + MakeParams<DQ> + serde::Serialize + for<'de> serde::Deserialize<'de>,
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    IVF<DQ, DC, CI>: IndexSerializer,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    DC::Encoder: VectorEncoder<Distance = DotProduct>,
    for<'q> <DQ::Encoder as VectorEncoder>::QueryVector<'q>: From<DenseVectorView<'q, f32>>
        + Into<<DC::Encoder as VectorEncoder>::QueryVector<'q>>
        + Copy,
    for<'q, 'b> <DC::Encoder as VectorEncoder>::Evaluator<'q>:
        QueryEvaluator<<DC::Encoder as VectorEncoder>::EncodedVector<'b>, Distance = DotProduct>,
{
    let index: IVF<DQ, DC, CI> = <IVF<DQ, DC, CI> as IndexSerializer>::load_index(&args.index_file)
        .unwrap_or_else(|e| {
            eprintln!("Error loading index: {e:?}");
            process::exit(1);
        });
    index.print_space_usage_bytes();
    let params = CI::make_params(args.n_probe, args);
    search_loop(&index, queries, &params, args);
}

// ---------------------------------------------------------------------------
// PQ cluster search (dot-product PQ storage; centroid metric = DQ)
// ---------------------------------------------------------------------------

fn run_pq<const M: usize, DQ, CI>(args: &Args, queries: &Queries)
where
    DQ: Dataset + serde::Serialize + for<'de> serde::Deserialize<'de>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    CI: Index + MakeParams<DQ> + serde::Serialize + for<'de> serde::Deserialize<'de>,
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    DenseDataset<ProductQuantizer<M, DotProduct>>: Dataset<Encoder = ProductQuantizer<M, DotProduct>>
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>,
    IVF<DQ, DenseDataset<ProductQuantizer<M, DotProduct>>, CI>: IndexSerializer,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>
        + VectorEncoder<Distance = DotProduct>,
    for<'q> <DQ::Encoder as VectorEncoder>::QueryVector<'q>: From<DenseVectorView<'q, f32>>
        + Into<<ProductQuantizer<M, DotProduct> as VectorEncoder>::QueryVector<'q>>
        + Copy,
    for<'q, 'b> <ProductQuantizer<M, DotProduct> as VectorEncoder>::Evaluator<'q>: QueryEvaluator<
            <ProductQuantizer<M, DotProduct> as VectorEncoder>::EncodedVector<'b>,
            Distance = DotProduct,
        >,
{
    type DP_<const N: usize> = DenseDataset<ProductQuantizer<N, DotProduct>>;

    let index: IVF<DQ, DP_<M>, CI> = <IVF<DQ, DP_<M>, CI> as IndexSerializer>::load_index(
        &args.index_file,
    )
    .unwrap_or_else(|e| {
        eprintln!("Error loading index: {e:?}");
        process::exit(1);
    });
    index.print_space_usage_bytes();
    let params = CI::make_params(args.n_probe, args);
    search_loop(&index, queries, &params, args);
}

// ---------------------------------------------------------------------------
// Shared search loop
// ---------------------------------------------------------------------------

fn search_loop<DQ, DC, CI>(
    index: &IVF<DQ, DC, CI>,
    queries: &Queries,
    params: &IVFSearchParams<CI>,
    args: &Args,
) where
    DQ: Dataset,
    DC: Dataset,
    CI: Index,
    for<'q> CI: Index<Query<'q> = <DQ::Encoder as VectorEncoder>::QueryVector<'q>>,
    <DQ::Encoder as VectorEncoder>::Distance: ReportedMetric,
    DC::Encoder: VectorEncoder<Distance = DotProduct>,
    for<'q> <DQ::Encoder as VectorEncoder>::QueryVector<'q>: From<DenseVectorView<'q, f32>>
        + Into<<DC::Encoder as VectorEncoder>::QueryVector<'q>>
        + Copy,
    for<'q, 'b> <DC::Encoder as VectorEncoder>::Evaluator<'q>:
        QueryEvaluator<<DC::Encoder as VectorEncoder>::EncodedVector<'b>, Distance = DotProduct>,
{
    let nq = queries.len();
    let mut results = Vec::<(f32, usize)>::with_capacity(nq * args.k);
    let mut total_us = 0u128;
    for _ in 0..args.num_runs {
        results.clear();
        for query in queries.iter() {
            let t = Instant::now();
            let res = index.search(query.into(), args.k, params);
            total_us += t.elapsed().as_micros();
            results.extend(
                res.iter()
                    .map(|s| (s.distance.distance(), s.vector as usize)),
            );
        }
    }
    let avg = total_us / (nq * args.num_runs) as u128;
    println!("[######] Average Query Time: {avg} μs");
    if let Some(path) = &args.output_path {
        write_results(path, &results, args.k);
    }
}

fn write_results(path: &str, results: &[(f32, usize)], k: usize) {
    let mut f = File::create(path).unwrap();
    for (i, (score, doc_id)) in results.iter().enumerate() {
        let query_id = i / k;
        let rank = (i % k) + 1;
        writeln!(f, "{}\t{}\t{}\t{}", query_id, doc_id, rank, score).unwrap();
    }
}
