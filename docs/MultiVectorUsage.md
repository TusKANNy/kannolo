# Multivector Reranking in kANNolo

## Overview

kANNolo provides a flexible two-stage retrieval framework for efficient search:
1. **First stage**: Fast candidate retrieval (e.g., HNSW on single-vector representations)
2. **Second stage**: High-quality reranking with arbitrary data (e.g., multivector representations)

This architecture enables super-fast approximate retrieval followed by accurate reranking, allowing you to combine efficiency with quality.

### General Reranking Framework

The reranking pattern is generic and extensible:
```
First-Stage Index (fast, approximate) → Top-k candidates → Reranking Dataset (accurate) → Final Results
```

### Binary Specialized for Sparse + Multivector

Our provided binary `hnsw_rerank_search` specializes in a common and effective pattern:
- **First stage**: HNSW on sparse data 
- **Reranking**: Multivector dataset (token-level dense vectors)

This combination leverages sparse retrieval speed with multivector quality for optimal results.

## Architecture

```
Sparse Queries → Sparse HNSW Index → Top-k candidates → Multivector Reranking → Final Results
```

The sparse HNSW index efficiently retrieves candidates, which are then reranked using dense multivector representations (token-level vectors).

## ⚠️ Critical Constraint

**Document and Query Alignment**: The documents and queries in the reranking dataset must be in the **exact same order** as in the first-stage index. Document IDs from the first stage are used directly to access the reranking data. It is **your responsibility** to ensure this alignment—the binary will not validate it.

Misaligned data will silently produce incorrect results.


## Supported Configurations

### Plain Multivector Reranking
- **Data**: Documents as token vectors (n_tokens × token_dim)
- **Format**: Stored as u16, reinterpreted as f16 after read
- **Best for**: Smaller datasets or when memory is less critical

### Two-Level Product Quantization (PQ)
- **Data**: Coarse centroids + PQ-encoded residuals
- **Format**: 
  - Coarse centroids: F32 array
  - PQ residuals: u8 codes per subspace
  - Index assignments: u64 (maps tokens to coarse centroids)
- **Best for**: Large-scale deployments with memory constraints
- **Reconstruction**: Combines coarse centroid + PQ residuals per token

## Binary: `hnsw_rerank_search`

Unified binary supporting both plain and two-level PQ quantization for sparse first-stage + multivector reranking.

### Basic Usage

```bash
./target/release/hnsw_rerank_search \
  --index-file /path/to/sparse_hnsw_index \
  --query-file /path/to/sparse_queries.bin \
  --multivec-data-folder /path/to/multivec_data \
  --multivector-quantizer [plain|two-levels] \
  --k 10 \
  --ef-search 400 \
  --distance dotproduct
```

### Command-Line Parameters

#### Required
- `--index-file`: Path to sparse HNSW index
- `--query-file`: Path to sparse queries (seismic format)
- `--multivec-data-folder`: Folder containing multivector data
  - **For "plain"**: Must contain `documents.npy`, `doclens.npy`, `queries.npy`
  - **For "two-levels"**: Additionally requires `centroids.npy`, `pq_centroids.npy`, `residuals.npy`, `index_assignment.npy`
- `--distance`: Distance metric (`dotproduct` or `euclidean`)

#### First-Stage Search (HNSW)
- `--ef-search`: Dynamic candidate list size (default: 40)
- `--early-termination`: Strategy: `none` or `distance-adaptive` (default: `none`)
- `--lambda`: Relaxation parameter for distance-adaptive termination (0-1)

#### Candidate Retrieval
- `--k-candidates`: Number of candidates from first stage (default: 100)

#### Reranking Control
- `--alpha`: Candidate pruning threshold (0-1, optional)
  - Filters candidates based on distance before reranking
- `--beta`: Early exit parameter (optional)
  - Stops reranking after `beta` consecutive non-improving results

#### Quantizer-Specific
- `--multivector-quantizer`: `plain` or `two-levels` (default: `plain`)
- `--pq-subspaces`: Number of PQ subspaces M (required for `two-levels`)

#### Output
- `--output-path`: Path to write results (TSV format)
- `--k`: Top-k results to return (default: 10)
- `--num-runs`: Number of runs for timing measurement (default: 1)

## Configuration via TOML

Use `scripts/run_experiments.py` with TOML config files for automated experiments.

### Required Sections

```toml
[settings]
k = 10
metric = "RR@10"  # Evaluation metric

[folder]
data = "/path/to/sparse_data"
index = "/path/to/index"
multivec_data = "/path/to/multivec_data"
experiment = "."

[filename]
dataset = "documents.bin" # First stage document vectors
queries = "queries.bin" # First stage query vectors
groundtruth = "groundtruth.tsv" # First stage groundtruth
index = "index_name"

[indexing_parameters]
m = 32
ef-construction = 2000
metric = "dotproduct"

[multivector]
quantizer = "plain"  # or "two-levels"
pq-subspaces = 32    # Required for "two-levels"
```

### Query Configurations

Define multiple search configurations as subsections of `[query]`:

```toml
[query]
    [query.config_1]
    ef-search = 15
    k_candidates = 30
    alpha = 0.05
    early-termination = "distance-adaptive"
    lambda = 0.1

    [query.config_2]
    ef-search = 400
    k_candidates = 35
    alpha = 0.1
    beta = 2
    early-termination = "none"
```

Each subsection is executed separately, allowing parameter sweeps within a single config file.

## Running Experiments

### Using run_experiments.py

```bash
python3 scripts/run_experiments.py --exp experiments/best_configs/msmarco-v1/rerank_ms_marco_cocondenser_two_levels_pq.toml
```

The script will:
1. Compile the binary (if needed)
2. Build the sparse HNSW index
3. Execute queries for each `[query]` subsection
4. Compute metrics (e.g. RR@k, Success@5)
5. Output results to `report.tsv`

### Example Configuration

See `experiments/best_configs/msmarco-v1/rerank_ms_marco_cocondenser_two_levels_pq.toml` for a complete working example using:
- **First stage**: SPLADE CoCondenser sparse vectors
- **Second stage**: ColBERTv2 dense token vectors with two-level PQ
- **Dataset**: MS MARCO v1 Passage

## Input Data Format

### Document Ordering

**Critical**: Documents in the multivector reranking dataset must appear in the **exact same order** as in the first-stage index. The binary uses document IDs directly from the first stage to index into the reranking data—there is no ID mapping or validation.

Example:
```
First-stage index: doc_ids = [42, 128, 7, 203, ...]
Reranking data:   doclens[0] must be for doc_id=42
                  doclens[1] must be for doc_id=128
                  doclens[2] must be for doc_id=7
                  etc.
```

If your first-stage and reranking data are built independently, you **must reorder** the reranking data to match the first-stage document order.

### Plain Multivector Reranking

```
documents.npy: shape (n_tokens, token_dim), dtype uint16 (reinterpreted as f16)
doclens.npy: shape (n_docs,), dtype int32 (tokens per document)
queries.npy: shape (n_queries, n_tokens_per_query, token_dim), dtype float32
```

### Two-Level PQ

```
centroids.npy: shape (n_coarse_centroids, token_dim), dtype float32
pq_centroids.npy: flattened PQ centroids (M * 256 * dsub,), dtype float32
residuals.npy: shape (n_tokens, M), dtype uint8 (PQ codes)
index_assignment.npy: shape (n_tokens,), dtype uint64 (coarse centroid indices)
doclens.npy: shape (n_docs,), dtype int32
queries.npy: shape (n_queries, n_tokens_per_query, token_dim), dtype float32
```


## Output Format

Results are written as TSV (tab-separated values):

```
query_id    doc_id    rank    distance
0           42        1       10.95
0           128       2       8.12
```

## Performance Notes

### Parameter Tuning

- **`ef-search`**: Higher values improve recall but increase latency. Typical range: 100-1000 without early-exit, 10-30 with distance-adaptive early-exit.
- **`lambda`** (distance-adaptive): Higher means less aggressive early exit. Values 0.005-0.25 shown to maintain quality while improving speed.
- **`k-candidates`**: Balance between reranking quality and speed. Usually 2-5x the final `k` is sufficient.
- **`alpha`**: Candidates Pruning parameter. Higher means more aggressive pruning. Values 0.015-0.05 shown to maintain quality while improving speed.
- **`beta`**: Early termination after `beta` non-improving candidates, stop reranking. Higher means less aggressive early exit. Values 2-4 shown to maintain quality while improving speed.


