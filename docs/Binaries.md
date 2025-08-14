# Unified Build and Search Binaries

This document explains the unified build and search binaries.

## Overview

There are two binaries for HNSW:
- `hnsw_build`: Index construction binary
- `hnsw_search`: Search binary

## Command-Line Options

### hnsw_build

```bash
Usage: hnsw_build [OPTIONS] --data-file <DATA_FILE> --output-file <OUTPUT_FILE> --vector-type <VECTOR_TYPE>

Options:
  -d, --data-file <DATA_FILE>      The path of the dataset file
  -o, --output-file <OUTPUT_FILE>  The output file where to save the index
      --vector-type <VECTOR_TYPE>  The type of vectors (dense or sparse) [possible values: dense, sparse]
      --precision <PRECISION>      The precision (f16 or f32). Note: PQ always uses f32 [default: f32] [possible values: f16, f32]
      --quantizer <QUANTIZER>      The quantizer type (plain or pq). Note: PQ is only available for dense vectors [default: plain] [possible values: plain, pq]
      --graph-type <GRAPH_TYPE>    The graph type (standard or fixed-degree) [default: standard] [possible values: standard, fixed-degree]
      --m <M>                      The number of neighbors per node [default: 16]
      --efc <EFC>                  The size of the candidate pool at construction time [default: 40]
      --metric <METRIC>            The type of distance to use. Either 'l2' (Euclidean) or 'ip' (Inner product) [default: ip]
      --m-pq <M_PQ>                The number of subspaces for Product Quantization (only for PQ). Supported values: 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384 [default: 16]
      --nbits <NBITS>              The number of bits per subspace for Product Quantization (only for PQ) [default: 8]
      --sample-size <SAMPLE_SIZE>  The size of the sample used for training Product Quantization (only for PQ) [default: 100000]
```

### hnsw_search

```bash
Usage: hnsw_search [OPTIONS] --index-file <INDEX_FILE> --query-file <QUERY_FILE> --vector-type <VECTOR_TYPE>

Options:
  -i, --index-file <INDEX_FILE>    The path of the index
  -q, --query-file <QUERY_FILE>    The query file
  -o, --output-path <OUTPUT_PATH>  The output file to write the results
      --vector-type <VECTOR_TYPE>  The type of vectors (dense or sparse) [possible values: dense, sparse]
      --precision <PRECISION>      The precision (f16 or f32). Note: PQ always uses f32 [default: f32] [possible values: f16, f32]
      --quantizer <QUANTIZER>      The quantizer type (plain or pq). Note: PQ is only available for dense vectors [default: plain] [possible values: plain, pq]
      --graph-type <GRAPH_TYPE>    The graph type (standard or fixed-degree) [default: standard] [possible values: standard, fixed-degree]
      --m-pq <M_PQ>                The number of subspaces for Product Quantization (only for PQ). Supported values: 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384 [default: 16]
  -k, --k <K>                      The number of top-k results to retrieve [default: 10]
      --ef-search <EF_SEARCH>      The ef_search parameter [default: 40]
      --n-run <N_RUN>              Number of runs for timing [default: 1]
```

## Examples

### Building Indexes

#### Dense vectors with plain quantizer (f32)
```bash
./hnsw_build --data-file data.npy --output-file index.bin --vector-type dense --precision f32 --quantizer plain --m 16 --efc 40
```

#### Dense vectors with plain quantizer (f16)
```bash
./hnsw_build --data-file data.npy --output-file index.bin --vector-type dense --precision f16 --quantizer plain --m 16 --efc 40
```

#### Dense vectors with PQ quantizer
```bash
./hnsw_build --data-file data.npy --output-file index.bin --vector-type dense --quantizer pq --m-pq 16 --nbits 8 --m 16 --efc 40
```

#### Sparse vectors with plain quantizer
```bash
./hnsw_build --data-file data.bin --output-file index.bin --vector-type sparse --precision f16 --quantizer plain --m 16 --efc 40
```

#### Using fixed-degree graph
```bash
./hnsw_build --data-file data.npy --output-file index.bin --vector-type dense --graph-type fixed-degree --m 16 --efc 40
```

### Searching

#### Dense vectors with plain quantizer (f32)
```bash
./hnsw_search --index-file index.bin --query-file queries.npy --vector-type dense --precision f32 --quantizer plain --k 10 --ef-search 40 --output-path results.txt
```

#### Dense vectors with PQ quantizer
```bash
./hnsw_search --index-file index.bin --query-file queries.npy --vector-type dense --quantizer pq --m-pq 16 --k 10 --ef-search 40 --output-path results.txt
```

#### Sparse vectors with plain quantizer
```bash
./hnsw_search --index-file index.bin --query-file queries.bin --vector-type sparse --precision f16 --quantizer plain --k 10 --ef-search 40 --output-path results.txt
```

## Validation Rules

The binaries include validation to prevent invalid combinations:

1. **PQ quantizer is only available for dense vectors**: Attempting to use `--quantizer pq` with `--vector-type sparse` will result in an error.

2. **PQ always uses f32 precision**: When using `--quantizer pq`, the precision is automatically set to f32, regardless of the `--precision` setting.

3. **Sparse vectors currently only support f16 precision**: Attempting to use `--precision f32` with `--vector-type sparse` will result in an error.

4. **Supported m_pq values**: The Product Quantization parameter `--m-pq` supports the following values: 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384. Using any other value will result in an error.


