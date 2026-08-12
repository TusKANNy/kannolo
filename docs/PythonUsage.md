# Kannolo Python API Reference

Kannolo provides a Python interface for vector search with support for:
- **Dense indexes**: Plain and PQ-encoded HNSW
- **Sparse indexes**: Multiple encoding schemes (plain, DotVByte, FixedU8, FixedU16)
- **Flat (brute-force) indexes**: Dense and sparse exhaustive search
- **Multivector reranking**: Two-stage sparse + multivector retrieval

## Imports

```python
from kannolo import (
    # Dense HNSW
    DensePlainHNSW,      # Dense plain HNSW (no quantization)
    DensePQHNSW,         # Dense product-quantized HNSW
    DenseFlatIndex,      # Dense exhaustive (brute-force) search
    
    # Sparse HNSW
    SparsePlainHNSW,     # Sparse plain HNSW
    SparseDotVByteHNSW,  # Sparse DotVByte-encoded HNSW
    SparseFixedU8HNSW,   # Sparse fixed u8-encoded HNSW
    SparseFixedU16HNSW,  # Sparse fixed u16-encoded HNSW
    SparseFlatIndex,     # Sparse exhaustive (brute-force) search
    
    # Multivector reranking
    SparseMultivecRerankIndex,            # Sparse HNSW + Plain multivector rerank
    SparseMultivecTwoLevelsPQRerankIndex, # Sparse HNSW + Two-level PQ multivector rerank
)
import numpy as np
```

## Index Construction

### HNSW Parameters

All HNSW indexes support:
- `m` (int): Neighbors per node. Typical: 16–64. Default: 32
- `ef_construction` (int): Graph construction effort. Higher = better quality, slower. Default: 200
- `metric` (str): Distance metric. Options: `"euclidean"`, `"dotproduct"` (default)
- `graph_type` (str): How the graph is stored. Leave it out and you get `"standard"`.
  See [GraphCompression.md](GraphCompression.md)
  - `"standard"` (default) — original node order, uncompressed
  - `"compressed"` — reordered *and* compressed: roughly halves the graph portion of the
    index and keeps the query speed-up below. Requires `m <= 128`
  - `"permuted"` — reordered only: faster queries, because nodes that are searched together
    end up stored together, but no size saving (marginally larger than `"standard"`)

  All three return identical search results; they differ in index size and query speed.
  Whatever you pass to the builder must be passed again to `load()`: index files are not
  self-describing, so `load()` cannot detect it for you.

### Dense Plain HNSW

```python
# From .npy file (dtype=float32)
index = DensePlainHNSW.build_from_file(
    "data.npy",
    m=32,
    ef_construction=200,
    metric="dotproduct",
    graph_type="standard"   # or "compressed" / "permuted"
)

# From numpy array. The array must be flattened, and `dim` is required:
# vectors are passed as one contiguous 1-D buffer of n_vectors * dim floats.
data = np.random.randn(10000, 768).astype(np.float32)
index = DensePlainHNSW.build_from_array(data.flatten(), dim=768, m=32, ef_construction=200)
```

### Dense PQ HNSW

```python
# Product-quantized dense index
index = DensePQHNSW.build_from_file(
    "data.npy",
    m_pq=32,           # PQ subspaces: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384
    m=32,              # HNSW neighbors
    ef_construction=200,
    metric="dotproduct"
)
```

`m_pq` must divide the vector dimensionality. Note that the CLI's `--pq-subspaces` accepts a
different set (`4` … `192`); `256` and `384` are Python-only, and `4` is CLI-only.

### Dense Flat Index

```python
# Exhaustive search (no HNSW index, just linear scan)
index = DenseFlatIndex.build_from_file("data.npy", metric="dotproduct")

# Or from array (flattened, as above)
data = np.random.randn(10000, 768).astype(np.float32)
index = DenseFlatIndex.build_from_array(data.flatten(), dim=768, metric="dotproduct")
```

### Sparse Plain HNSW

```python
# From binary file (seismic format)
index = SparsePlainHNSW.build_from_file(
    "data.bin",
    m=32,
    ef_construction=200,
    metric="dotproduct"
)

# From numpy arrays (components and values)
components = np.array([0, 5, 10, 15], dtype=np.int32)
values = np.array([0.5, 0.3, 0.8, 0.2], dtype=np.float32)
offsets = np.array([0, 4], dtype=np.int64)  # one document starting at 0, ending at 4

# Build with these vectors
index = SparsePlainHNSW.build_from_arrays(
    components, values, offsets,
    m=32,
    ef_construction=200,
    metric="dotproduct"
)
```

### Sparse Variants (DotVByte, FixedU8, FixedU16)

```python
# All follow the same interface as SparsePlainHNSW
index = SparseDotVByteHNSW.build_from_file("data.bin", m=32, ef_construction=200)
index = SparseFixedU8HNSW.build_from_file("data.bin", m=32, ef_construction=200, metric="dotproduct")
index = SparseFixedU16HNSW.build_from_file("data.bin", m=32, ef_construction=200, metric="dotproduct")
```

### Sparse Flat Index

```python
# Exhaustive sparse search
index = SparseFlatIndex.build_from_file("data.bin")

# Or from arrays
components = np.array([0, 5, 10], dtype=np.int32)
values = np.array([0.5, 0.3, 0.8], dtype=np.float32)
offsets = np.array([0, 3], dtype=np.int64)
index = SparseFlatIndex.build_from_arrays(components, values, offsets)
```

### Multivector Reranking

#### Plain Multivector

```python
# Expects multivec_data_folder with:
# - documents.npy (shape: [n_docs, n_tokens, token_dim], dtype: float32)
# - queries.npy (shape: [n_queries, n_tokens, token_dim], dtype: float32)
# - doclens.npy (shape: [n_docs], dtype: int32/int64)

index = SparseMultivecRerankIndex.build_from_file(
    sparse_index_path="sparse_index_file",
    multivec_data_folder="/path/to/multivec_data_folder"
)
```

#### Two-Level PQ Multivector

```python
# Expects all files from plain, except documents.npy, plus:
# - centroids.npy, pq_centroids.npy, residuals.npy, index_assignment.npy

index = SparseMultivecTwoLevelsPQRerankIndex.build_from_file(
    sparse_index_path="sparse_index.bin",
    multivec_data_folder="/path/to/multivec_data",
    pq_subspaces=32  # Must be 8, 16, 32, or 64
)
```

---

## Save / Load

```python
# Save any index
index.save("my_index.bin")

# Load (must specify metric for types that support it)
index = DensePlainHNSW.load("my_index.bin", metric="dotproduct")
index = SparsePlainHNSW.load("my_index.bin", metric="dotproduct")
index = DensePQHNSW.load("my_index.bin", m_pq=32, metric="dotproduct")

# HNSW indexes built with a non-default graph type must be loaded with the same one
index = DensePlainHNSW.load("my_index.bin", metric="dotproduct", graph_type="compressed")

# Flat indexes
index = DenseFlatIndex.load("my_index.bin")  # Requires nothing extra
index = SparseFlatIndex.load("my_index.bin")
```

---

## Search Operations

Every index exposes **two** methods, and they are not interchangeable:

- **`search(...)`** — one query. Dense indexes take a 1-D array of exactly `dim` floats and
  raise `ValueError` otherwise; sparse indexes take one `(components, values)` pair and take
  **no** `offsets` argument.
- **`batch_search(...)`** — many queries at once, optionally in parallel. Dense indexes take
  all queries concatenated into one flat array; sparse indexes take the concatenated
  `components`/`values` plus an `offsets` array delimiting each query.

Both return `(distances, ids)` as two 1-D numpy arrays. **Results are always padded to `k`**:
if fewer than `k` neighbors are found, the remainder is filled with `inf` distances and `-1`
ids. `batch_search` returns `num_queries * k` entries, query-major.

Returned ids are always ids into your original dataset, whatever `graph_type` you built with.

### `num_threads` (batch only)

- `0` (default) — rayon's default pool, typically every core.
- `1` — a plain serial loop with no rayon involvement. Use this to reproduce single-threaded
  benchmarks that pin the process with `numactl --physcpubind`.
- `n` — a temporary pool of `n` threads for the duration of the call.

### Dense indexes (`DensePlainHNSW`, `DensePQHNSW`)

```python
# Single query: a 1-D array of exactly `dim` floats.
query = np.random.randn(768).astype(np.float32)
dists, ids = index.search(query, k=10, ef_search=200)

# With early exit threshold (optional, suggested 0.005-0.25)
dists, ids = index.search(query, k=10, ef_search=200, early_exit_threshold=0.1)

# Many queries: flatten them into a single contiguous buffer.
queries = np.random.randn(100, 768).astype(np.float32)
dists, ids = index.batch_search(queries.flatten(), k=10, ef_search=200, num_threads=0)
# dists and ids each hold 100 * 10 entries, query-major.
```

`DensePQHNSW.search` has no default arguments: pass `ef_search` and `early_exit_threshold`
explicitly.

### Dense Flat Index

```python
query = np.random.randn(768).astype(np.float32)
dists, ids = index.search(query, k=10)

queries = np.random.randn(100, 768).astype(np.float32)
dists, ids = index.batch_search(queries.flatten(), k=10, num_threads=0)
```

### Sparse HNSW (`SparsePlainHNSW`, `SparseDotVByteHNSW`, `SparseFixedU8HNSW`, `SparseFixedU16HNSW`)

```python
# Single query: components and values only, no offsets.
query_components = np.array([0, 5, 10], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3], dtype=np.float32)
dists, ids = index.search(query_components, query_values, k=10, ef_search=200)

# Many queries: concatenate them and delimit with offsets.
query_components = np.array([0, 5, 10, 100, 200], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
offsets = np.array([0, 3, 5], dtype=np.int64)  # Two queries: [0,3) and [3,5)

dists, ids = index.batch_search(
    query_components, query_values, offsets,
    k=10, ef_search=200,
    early_exit_threshold=0.1,   # optional
    num_threads=0,
)
```

### Sparse Flat Index

```python
query_components = np.array([0, 5, 10], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3], dtype=np.float32)
dists, ids = index.search(query_components, query_values, k=10)

# Batch form takes offsets.
offsets = np.array([0, 3], dtype=np.int64)
dists, ids = index.batch_search(query_components, query_values, offsets, k=10, num_threads=0)
```

### Sparse Multivector Reranking

The single-query form takes `multivec_query` (singular) and no offsets:

```python
query_components = np.array([0, 5, 10, 100], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7], dtype=np.float32)

# One query: 8 tokens of 768 dimensions, flattened.
multivec_query = np.random.randn(8, 768).astype(np.float32).reshape(-1)

dists, ids = index.search(
    query_components, query_values, multivec_query,
    n_tokens=8,
    token_dim=768,
    k_candidates=25,   # First-stage candidates
    k=10,              # Final results
    ef_search=100,
    alpha=0.05,        # First-stage weight (optional)
    beta=2,            # Second-stage early exit (optional)
    residuals=False,   # Sum first- and second-stage scores (optional)
)
```

`residuals=True` scores each candidate as `first_stage_score + rerank_score` instead of the rerank
score alone, for setups where the rerank dataset holds the residual part of a decomposed
representation. Nothing checks that the two scores are summable — that is on the caller.

The batch form adds `sparse_offsets` and takes `multivec_queries` (plural):

```python
sparse_offsets = np.array([0, 4], dtype=np.int64)  # One query
multivec_queries = np.random.randn(1, 8, 768).astype(np.float32).reshape(-1)

dists, ids = index.batch_search(
    query_components, query_values, sparse_offsets, multivec_queries,
    n_tokens=8, token_dim=768,
    k_candidates=25, k=10, ef_search=100,
    num_threads=0,
    residuals=False,   # Sum first- and second-stage scores (optional)
)
```

---

## Filtered Search (ACORN)

Predicate-aware search — "the `k` nearest neighbors for which `predicate(id)` is true" — is
available on **`DensePlainHNSW` only**. It is not a post-filter: the predicate steers the
graph traversal, so recall stays usable even when few vectors qualify.

The predicate is a Python callable receiving an original dataset id and returning `bool`.
It is invoked during traversal, so keep it cheap.

```python
# ACORN-1: two-hop expansion computed on the fly. Nothing to prepare.
dists, ids = index.search_filtered(query, k=10, predicate=lambda i: i % 2 == 0, ef_search=100)

# ACORN-gamma: precompute the expansion once, then search repeatedly.
# Trades memory for speed; gamma of 2-4 is a sensible starting point.
index.build_acorn_gamma(gamma=3)
dists, ids = index.search_filtered_gamma(query, k=10, predicate=lambda i: i % 2 == 0, ef_search=100)
```

`build_acorn_gamma` must be called before `search_filtered_gamma`, which otherwise raises
`RuntimeError`. The expanded lists live on the index object in memory and are **not** written
by `save()`: call `build_acorn_gamma` again after `load()`.

---

## Index Selection Guide

| Use Case | Index | Notes |
|----------|-------|-------|
| Dense vectors, high accuracy | `DensePlainHNSW` | Default choice |
| Dense vectors, memory limited | `DensePQHNSW` | Quantized, faster search |
| Dense vectors, ground truth/exhaustive search | `DenseFlatIndex` | Exhaustive, exact neighbors |
| Sparse vectors, standard | `SparsePlainHNSW` | Plain encoding, good recall |
| Sparse vectors, memory limited | `SparseFixedU8HNSW` or `SparseDotVByteHNSW` | Compressed |
| Sparse vectors, ground truth/exhaustive search | `SparseFlatIndex` | Exhaustive, exact |
| Multivector retrieval | `SparseMultivecRerankIndex` | Sparse first-stage + multivec rerank |
| Multivector + quantization | `SparseMultivecTwoLevelsPQRerankIndex` | Sparse + PQ rerank |

---

## Additional Resources

- **Notebooks**: See [notebooks/](../notebooks/) for end-to-end examples
- **Rust API**: See [RustUsage.md](RustUsage.md)
- **Running Experiments**: See [RunExperiments.md](RunExperiments.md)
