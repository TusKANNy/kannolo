# Unified Build, Search, and Convert Binaries

This document describes the current CLI surface for:
- `hnsw_build`
- `hnsw_search`
- `ivf_build`
- `ivf_search`

`hnsw_rerank_search` is documented separately in [MultiVectorUsage.md](MultiVectorUsage.md).

All binaries are behind the `cli` feature (`hnsw_rerank_search` additionally needs `multivec`),
so a plain `cargo build` produces none of them:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features cli
```

All examples and option names below are aligned with the current binaries.

## `hnsw_build`

```bash
Usage: hnsw_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE> --dataset-type <DATASET_TYPE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, dotvbyte]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
      --m <M>                            [default: 16]
      --ef-construction <EF_CONSTRUCTION> [default: 150]
      --distance <DISTANCE>              [default: dotproduct]
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
      --nbits <NBITS>                    [default: 8] (ignored by vectorium PQ)
      --sample-size <SAMPLE_SIZE>        [default: 100000] (ignored by vectorium PQ)
```

## `hnsw_search`

```bash
Usage: hnsw_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE> --dataset-type <DATASET_TYPE> --distance <DISTANCE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, dotvbyte]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree, permuted, streamvbyte]
      --distance <DISTANCE>
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
  -k, --k <K>                            [default: 10]
      --ef-search <EF_SEARCH>            [default: 40]
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>                  [default: 1]
      --num-runs <NUM_RUNS>              [default: 1]
```

## `ivf_build`

Builds an inverted-file index. Dense only.

```bash
Usage: ivf_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE>

Options:
  -d, --data-file <DATA_FILE>
  -o, --output-file <OUTPUT_FILE>
      --distance <DISTANCE>              [default: euclidean] [possible values: euclidean, dotproduct]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --n-clusters <N_CLUSTERS>          [default: 1024]
      --kmeans-n-iter <KMEANS_N_ITER>    [default: 25]
      --kmeans-n-redo <KMEANS_N_REDO>    [default: 1]
      --kmeans-sample-size <SIZE>        (optional; defaults to the whole dataset)
      --kmeans-hnsw                      Use an HNSW index to speed up k-means assignment
      --kmeans-spherical                 L2-normalize centroids each iteration
      --residuals                        Encode vectors as residuals from their centroid
      --hnsw                             Index the centroids with HNSW instead of scanning them
      --m-hnsw <M_HNSW>                  [default: 32] (alias: --m) Only with --hnsw
      --ef-construction <EF_CONSTRUCTION> [default: 200] Only with --hnsw
      --m-pq <M_PQ>                      (optional) PQ-encode the vectors with this many subspaces
```

## `ivf_search`

```bash
Usage: ivf_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE>

Options:
  -i, --index-file <INDEX_FILE>
  -q, --query-file <QUERY_FILE>
  -o, --output-path <OUTPUT_PATH>
  -k, --k <K>                            [default: 10]
      --n-probe <N_PROBE>                [default: 32] Clusters visited per query
      --distance <DISTANCE>              [default: euclidean] [possible values: euclidean, dotproduct]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --hnsw                             Must match the index
      --m-pq <M_PQ>                      Must match the index
      --residuals                        Must match the index
      --ef-search <EF_SEARCH>            [default: 40] Only with --hnsw
      --lambda <LAMBDA>                  [default: 0]
      --num-runs <NUM_RUNS>              [default: 1]
```

As with HNSW, `--distance`, `--value-type`, `--hnsw`, `--m-pq` and `--residuals` are properties
baked into the index at build time and must be repeated identically at search time.

## Examples

Dense plain:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder plain --value-type f32 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Dense PQ:

```bash
./hnsw_build --data-file data.npy --output-file index.bin \
  --dataset-type dense --encoder pq --pq-subspaces 16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse plain with explicit sparse component type:

```bash
./hnsw_build --data-file data.bin --output-file index.bin \
  --dataset-type sparse --encoder plain --value-type f16 --component-type u16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse DotVByte:

```bash
./hnsw_build --data-file data.bin --output-file index.bin \
  --dataset-type sparse --encoder dotvbyte --component-type u16 \
  --m 16 --ef-construction 150 --distance dotproduct
```

Sparse DotVByte search:

```bash
./hnsw_search --index-file index.bin --query-file queries.bin \
  --dataset-type sparse --encoder dotvbyte --component-type u16 \
  --distance dotproduct --k 10 --ef-search 40 --output-path results.tsv
```

Compressed graph (see [GraphCompression.md](GraphCompression.md)):

```bash
./hnsw_build --data-file data.npy --output-file index_svb.bin \
  --dataset-type dense --encoder plain --value-type f32 --graph-type streamvbyte \
  --m 16 --ef-construction 150 --distance dotproduct

./hnsw_search --index-file index_svb.bin --query-file queries.npy \
  --dataset-type dense --encoder plain --value-type f32 --graph-type streamvbyte \
  --distance dotproduct --k 10 --ef-search 40 --output-path results.tsv
```

IVF:

```bash
./ivf_build --data-file data.npy --output-file ivf.bin \
  --distance euclidean --value-type f32 --n-clusters 1024

./ivf_search --index-file ivf.bin --query-file queries.npy \
  --distance euclidean --value-type f32 --n-probe 32 --k 10 --output-path results.tsv
```

## Validation Rules

The binaries reject invalid combinations:

1. `pq` is dense-only.
2. `dotvbyte` is sparse-only.
3. `fixedu8` and `fixedu16` value types are sparse-only.
4. `component-type` is sparse-only.
5. `dotvbyte` requires `component-type = u16`.
6. `pq-subspaces` must be one of `4, 8, 16, 32, 48, 64, 96, 128, 192` and must divide the vector dimensionality.
7. For PQ, `--nbits` and `--sample-size` are accepted for compatibility but ignored by vectorium.
8. `--graph-type streamvbyte` caps the ground level at 256 neighbors per node, so it requires `--m` of at most 128. This is checked before the build starts, not after.

Beyond these, `dataset-type`, `value-type`, `component-type`, `encoder`, `distance` and `graph-type` are all baked into the index at build time. `hnsw_search` cannot detect them — index files carry no header — so passing a different value produces a decode error rather than a helpful message. Repeat the build flags exactly.
