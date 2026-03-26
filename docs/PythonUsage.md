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

### Dense Plain HNSW

```python
# From .npy file (dtype=float32)
index = DensePlainHNSW.build_from_file(
    "data.npy",
    m=32,
    ef_construction=200,
    metric="dotproduct"
)

# From numpy array
data = np.random.randn(10000, 768).astype(np.float32)
index = DensePlainHNSW.build_from_array(data, m=32, ef_construction=200)
```

### Dense PQ HNSW

```python
# Product-quantized dense index
index = DensePQHNSW.build_from_file(
    "data.npy",
    m_pq=32,           # PQ subspaces: 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384
    m=32,              # HNSW neighbors
    ef_construction=200,
    metric="dotproduct"
)
```

### Dense Flat Index

```python
# Exhaustive search (no HNSW index, just linear scan)
index = DenseFlatIndex.build_from_file("data.npy", metric="dotproduct")

# Or from array
data = np.random.randn(10000, 768).astype(np.float32)
index = DenseFlatIndex.build_from_array(data, dim=768, metric="dotproduct")
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

# Flat indexes
index = DenseFlatIndex.load("my_index.bin")  # Requires nothing extra
index = SparseFlatIndex.load("my_index.bin")
```

---

## Search Operations

### Dense Plain HNSW

```python
# Batch search (automatically detects # of queries from array length)
queries = np.random.randn(100, 768).astype(np.float32)  # 100 queries
dists, ids = index.search(queries, k=10, ef_search=200)
# Returns: distances of shape [≤100*10], ids of shape [≤100*10]

# With early exit threshold (optional, suggested 0.005-0.25)
dists, ids = index.search(queries, k=10, ef_search=200, early_exit_threshold=0.1)
```

### Dense Flat Index

```python
queries = np.random.randn(100, 768).astype(np.float32)
dists, ids = index.search(queries, k=10)
```

### Sparse Plain HNSW / Other Sparse Variants

```python
# Batch search for multiple sparse queries
query_components = np.array([0, 5, 10, 100, 200], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
offsets = np.array([0, 3, 5], dtype=np.int64)  # Two queries: [0,3) and [3,5)

dists, ids = index.search(
    query_components,
    query_values,
    offsets,
    k=10,
    ef_search=200
)

# With early exit
dists, ids = index.search(
    query_components, query_values, offsets,
    k=10, ef_search=200,
    early_exit_threshold=0.1
)
```

### Sparse Flat Index

```python
query_components = np.array([0, 5, 10, 100, 200], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7, 0.2], dtype=np.float32)
offsets = np.array([0, 3, 5], dtype=np.int64)  # Two queries

dists, ids = index.search(query_components, query_values, offsets, k=10)
```

### Dense PQ HNSW

```python
queries = np.random.randn(100, 768).astype(np.float32)
dists, ids = index.search(queries, k=10, ef_search=200)

# With early exit
dists, ids = index.search(queries, k=10, ef_search=200, early_exit_threshold=0.1)
```

### Sparse Multivector Reranking

```python
# Sparse + dense two-stage search
query_components = np.array([0, 5, 10, 100], dtype=np.int32)
query_values = np.array([0.8, 0.5, 0.3, 0.7], dtype=np.float32)
sparse_offsets = np.array([0, 4], dtype=np.int64)  # One query

# Dense multivector queries (must match n_queries from sparse_offsets)
multivec_queries = np.random.randn(1, 8, 768).astype(np.float32)  # 1 query, 8 tokens, 768-dim each
multivec_flat = multivec_queries.reshape(-1).astype(np.float32)

dists, ids = index.search(
    query_components=query_components,
    query_values=query_values,
    sparse_offsets=sparse_offsets,
    multivec_queries=multivec_flat,
    n_tokens=8,
    token_dim=768,
    k_candidates=25,  # First-stage candidates
    k=10,               # Final results
    ef_search=100,
    alpha=0.05,          # First-stage weight (optional)
    beta=2,            # Second-stage early exit (optional)
    early_exit_threshold=None  # HNSW early exit (optional)
)
```

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
