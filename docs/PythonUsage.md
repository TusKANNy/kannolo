## Usage Example in Python
```python
from kannolo import DensePlainHNSW
from kannolo import DensePlainHNSWf16
from kannolo import SparsePlainHNSW
from kannolo import SparsePlainHNSWf16
from kannolo import DensePQHNSW
import numpy as np
```

The functioning of f16 indexes is the same as that of f32 ones, we outline examples for f32 indexes.

### Index Construction

Set index construction parameters.

```python
efConstruction = 200
m = 32 # n. neighbors per node
metric = "ip" # Inner product
```

Build HNSW index on dense, plain data.

```python
npy_input_file = "" # your input file

index = DensePlainHNSW.build(npy_input_file, m, efConstruction, "ip")
```

Build HNSW index on dense, PQ-encoded data.

```python
npy_input_file = "" # your input file

# Set PQ's parameters
m_pq = 192 # Number of subspaces of PQ
nbits = 8 # Number of bits to represent a centroid of a PQ's subspace
sample_size = 500_000 # Size of the sample of the dataset for training PQ

index = DensePQHNSW.build(data_path, m_pq, nbits, m, efConstruction, "ip", sample_size)
```


Build HNSW index on sparse data.

```python
"""
Binary File Format:
- First 4 bytes: Unsigned 32-bit integer (little-endian) indicating the total number of sparse vectors.
- For each vector:
    - 4 bytes: Unsigned 32-bit integer (little-endian) representing the number of nonzero components.
    - Next (4 * n) bytes: Array of n unsigned 32-bit integers (little-endian) for component indices (cast to int32).
    - Following (4 * n) bytes: Array of n 32-bit floating point values (little-endian) for the nonzero components.
"""
bin_input_file = "" # your input file

index = SparsePlainHNSW.build(data_path, m, efConstruction, "ip")
```

### Search

Set search parameters
```python
k = 10 # Number of results to be retrieved
efSearch = 200 # Search parameter for regulating the accuracy
```

#### Batch Search

Search multiple queries saved in a file.

```python
query_file = "" # your queries file, .npy for dense, .bin for sparse
dists, ids = index.search_batch(query_file, k, efSearch)
```

#### Single query search

Search for a single query.

##### Dense

Search for a dense query `my_query` stored in a numpy array.

```python
dists, ids = index.search(my_query, k, efSearch)
```

##### Sparse

Search for a sparse query represented by two numpy arrays: `components`, containing the component IDs (i32) of the sparse query vector, and `values`, containing the non-zero floating point values (f32) associated with the components.

Conversion between numpy arrays and binary format for sparse data and queries can be performed with the `convert_bin_to_npy_arrays.py` and `convert_npy_arrays_to_bin.py` scripts.

```python
dists, ids = index.search(components, values, k, efSearch)
```

### Evaluation

For evaluation, see the demo notebooks in the `notebooks` folder.
