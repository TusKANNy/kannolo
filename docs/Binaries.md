# Unified Build, Search, and Convert Binaries

This document describes the current CLI surface for:
- `hnsw_build`
- `hnsw_search`
- `hnsw_convert`

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
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree]
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
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree]
      --distance <DISTANCE>
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
  -k, --k <K>                            [default: 10]
      --ef-search <EF_SEARCH>            [default: 40]
      --early-termination <EARLY_TERMINATION> [default: none] [possible values: none, distance-adaptive]
      --lambda <LAMBDA>                  [default: 1]
      --num-runs <NUM_RUNS>              [default: 1]
```

## `hnsw_convert`

```bash
Usage: hnsw_convert [OPTIONS] --index-file <INDEX_FILE> --output-file <OUTPUT_FILE> --dataset-type <DATASET_TYPE>

Options:
  -i, --index-file <INDEX_FILE>
  -o, --output-file <OUTPUT_FILE>
      --dataset-type <DATASET_TYPE>      [possible values: dense, sparse]
      --value-type <VALUE_TYPE>          [default: f32] [possible values: f16, f32, fixedu8, fixedu16]
      --component-type <COMPONENT_TYPE>  [default: u16] [possible values: u16, u32]
      --encoder <ENCODER>                [default: plain] [possible values: plain, pq, dotvbyte]
      --graph-type <GRAPH_TYPE>          [default: standard] [possible values: standard, fixed-degree]
      --m <M>                            [default: 16] (compatibility option, ignored)
      --ef-construction <EF_CONSTRUCTION> [default: 150] (compatibility option, ignored)
      --distance <DISTANCE>              [default: dotproduct]
      --pq-subspaces <PQ_SUBSPACES>      [default: 16]
      --nbits <NBITS>                    [default: 8] (compatibility option, ignored)
      --sample-size <SAMPLE_SIZE>        [default: 100000] (compatibility option, ignored)
```

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

Convert a dense plain-f32 index to PQ:

```bash
./hnsw_convert --index-file plain_f32.bin --output-file pq.bin \
  --dataset-type dense --encoder pq --distance dotproduct --pq-subspaces 16
```

Convert a sparse plain-f32 index to fixedu8 scalar:

```bash
./hnsw_convert --index-file plain_sparse_f32.bin --output-file sparse_fixedu8.bin \
  --dataset-type sparse --component-type u16 --encoder plain --value-type fixedu8 --distance dotproduct
```

Convert a sparse plain-f32 index to DotVByte:

```bash
./hnsw_convert --index-file plain_sparse_f32.bin --output-file sparse_dotvbyte.bin \
  --dataset-type sparse --component-type u16 --encoder dotvbyte --distance dotproduct
```

## Validation Rules

The binaries reject invalid combinations:

1. `pq` is dense-only.
2. `dotvbyte` is sparse-only.
3. `fixedu8` and `fixedu16` value types are sparse-only.
4. `component-type` is sparse-only.
5. `dotvbyte` requires `component-type = u16`.
6. `pq-subspaces` must be one of `4, 8, 16, 32, 64, 96, 128` and must divide the vector dimensionality.
7. For PQ, `--nbits` and `--sample-size` are accepted for compatibility but ignored by vectorium.
8. `hnsw_convert` expects a plain source index matching `dataset-type`, `graph-type`, `distance`, and `component-type` (for sparse).
