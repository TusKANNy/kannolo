## Rust API

kANNolo provides three command-line binaries for index building and search.

**Binaries:**
- `hnsw_build` – Build an HNSW index from dense or sparse data
- `hnsw_search` – Search an existing HNSW index
- `hnsw_rerank_search` – Two-stage search: sparse first-stage + multivector reranking

**Data Formats:**
- Dense: `.npy` files (float32 concatenated vectors)
- Sparse: Binary seismic format (see [PythonUsage.md](PythonUsage.md) for format details). Use scripts/convert_npy_arrays_to_bin.py and scripts/convert_bin_to_npy_arrays.py for conversion from bin to npy or viceversa.

---


### hnsw_build

Build an HNSW index from dense or sparse data.

**Required arguments:**
```bash
--data-file <path>        Input data file (.npy for dense, .bin for sparse)
--output-file <path>      Output index file
--dataset-type <type>     dense or sparse
```

**Index parameters:**
```bash
--m <int>                 Neighbors per node (default: 16)
--ef-construction <int>   Construction effort (default: 150)
```

**Data type parameters:**
```bash
--encoder <type>          plain (default), pq (dense only), dotvbyte (sparse only).
--value-type <type>       f32 (default), f16, fixedu8 (sparse only), fixedu16 (sparse only).
--component-type <type>   u16 (default), u32. (sparse only; dotvbyte requires u16)
```

**Search parameters:**
```bash
--distance <type>         euclidean or dotproduct (default: dotproduct)
```

**PQ-specific (when --encoder pq):**
```bash
--pq-subspaces <int>      Number of subspaces. Supported: 4, 8, 16, 32, 48, 64, 96, 128, 192
```

**Graph structure:**
```bash
--graph-type <type>       standard (default) or fixed-degree
```

**Examples:**

```bash
# Dense plain index
./target/release/hnsw_build \
  --data-file documents.npy \
  --output-file index.bin \
  --dataset-type dense \
  --encoder plain \
  --value-type f32 \
  --m 32 \
  --ef-construction 200 \
  --distance euclidean

# Sparse plain index
./target/release/hnsw_build \
  --data-file documents.bin \
  --output-file index.bin \
  --dataset-type sparse \
  --encoder plain \
  --component-type u16 \
  --m 32 \
  --ef-construction 2000 \
  --distance dotproduct

# Dense PQ index
./target/release/hnsw_build \
  --data-file documents.npy \
  --output-file index.bin \
  --dataset-type dense \
  --encoder pq \
  --pq-subspaces 64 \
  --m 32 \
  --ef-construction 200 \
  --distance dotproduct

# Sparse DotVByte index
./target/release/hnsw_build \
  --data-file documents.bin \
  --output-file index.bin \
  --dataset-type sparse \
  --encoder dotvbyte \
  --component-type u16 \
  --m 32 \
  --ef-construction 2000 \
  --distance dotproduct
```

---

### hnsw_search

Search an HNSW index with dense or sparse queries.

**Required arguments:**
```bash
--index-file <path>       HNSW index file (from hnsw_build)
--query-file <path>       Query file (.npy for dense, .bin for sparse)
--dataset-type <type>     dense or sparse (must match index)
```

**Search parameters:**
```bash
--k <int>                 Top-k results (default: 10)
--ef-search <int>         Search effort (default: 40)
--early-termination <type>  none (default) or distance-adaptive
--lambda <float>          Threshold for distance-adaptive (default: 1.0, range: 0.005-0.25)
```

**Data type parameters:**
```bash
--encoder <type>          plain (default), pq, dotvbyte (must match index)
--value-type <type>       f32 (default), f16, fixedu8, fixedu16
--component-type <type>   u16 (default), u32
```

**Distance & search:**
```bash
--distance <type>         euclidean or dotproduct (must match index)
--pq-subspaces <int>      Required only if --encoder pq was used during build
```

**Other:**
```bash
--output-path <path>      Output file for results (optional)
--graph-type <type>       standard (default) or fixed-degree (must match index)
--num-runs <int>          Number of runs for timing (default: 1)
```

**Output format:** TSV with columns: `query_id`, `document_id`, `rank`, `score`

**Examples:**

```bash
# Search dense plain index
./target/release/hnsw_search \
  --index-file index.bin \
  --query-file queries.npy \
  --dataset-type dense \
  --encoder plain \
  --value-type f32 \
  --distance euclidean \
  --k 10 \
  --ef-search 200 \
  --output-path results.tsv

# Search sparse plain index
./target/release/hnsw_search \
  --index-file index.bin \
  --query-file queries.bin \
  --dataset-type sparse \
  --encoder plain \
  --component-type u16 \
  --distance dotproduct \
  --k 10 \
  --ef-search 100 \
  --output-path results.tsv

# Search with early termination
./target/release/hnsw_search \
  --index-file index.bin \
  --query-file queries.bin \
  --dataset-type sparse \
  --encoder plain \
  --component-type u16 \
  --distance dotproduct \
  --k 10 \
  --ef-search 100 \
  --early-termination distance-adaptive \
  --lambda 0.1 \
  --output-path results.tsv

# Search dense PQ index
./target/release/hnsw_search \
  --index-file index.bin \
  --query-file queries.npy \
  --dataset-type dense \
  --encoder pq \
  --pq-subspaces 64 \
  --distance dotproduct \
  --k 10 \
  --ef-search 200 \
  --output-path results.tsv
```

---

### hnsw_rerank_search

Two-stage search: first-stage sparse HNSW + second-stage dense multivector reranking.

**Required arguments:**
```bash
--index-file <path>              Pre-built sparse HNSW index
--query-file <path>              Sparse queries (binary format)
--multivec-data-folder <path>    Folder with multivector data
```

**Multivector data folder structure:**

For **plain quantizer**:
```
multivec_data/
  documents.npy              [n_docs, n_tokens, token_dim]
  queries.npy                [n_queries, n_tokens, token_dim]
  doclens.npy                [n_docs]
```

For **two-levels (PQ) quantizer**:
```
multivec_data/
  documents.npy              [n_docs, n_tokens, token_dim]
  queries.npy                [n_queries, n_tokens, token_dim]
  doclens.npy                [n_docs]
  centroids.npy              [n_centroids, token_dim]
  pq_centroids.npy           [n_centroids, M, subspace_dim]
  residuals.npy              [n_docs, n_tokens, token_dim]
  index_assignment.npy       [n_docs, n_tokens]
```

**Search parameters:**
```bash
--k <int>                 Final top-k results (default: 10)
--k-candidates <int>      First-stage candidates (default: 100)
--ef-search <int>         HNSW search effort (default: 40)
```

**Reranking:**
```bash
--alpha <float>           First-stage weight (optional)
--beta <int>              Second-stage early exit (optional)
```

**Multivector quantizer:**
```bash
--multivector-quantizer <type>  plain (default) or two-levels
--pq-subspaces <int>            Required if --multivector-quantizer two-levels. Values: 8, 16, 32, 64
```

**Other:**
```bash
--output-path <path>      Output file for results
--num-runs <int>          Number of runs for timing (default: 1)
```

**Example:**

```bash
# Plain multivector reranking
./target/release/hnsw_rerank_search \
  --index-file sparse_index.bin \
  --query-file queries.bin \
  --multivec-data-folder multivec_data/ \
  --multivector-quantizer plain \
  --k-candidates 100 \
  --k 10 \
  --ef-search 100 \
  --alpha 0.5 \
  --output-path results.tsv

# Two-level PQ multivector reranking
./target/release/hnsw_rerank_search \
  --index-file sparse_index.bin \
  --query-file queries.bin \
  --multivec-data-folder multivec_data/ \
  --multivector-quantizer two-levels \
  --pq-subspaces 32 \
  --k-candidates 100 \
  --k 10 \
  --ef-search 100 \
  --alpha 0.5 \
  --beta 10 \
  --output-path results.tsv
```
