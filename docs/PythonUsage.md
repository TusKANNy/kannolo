## Usage Example in Python
```python
from kannolo import DensePlainHNSW
from kannolo import DensePlainHNSWf16
from kannolo import SparsePlainHNSW
from kannolo import SparsePlainHNSWf16
from kannolo import DensePQHNSW
import numpy as np
import ir_measures
import ir_datasets
from ir_measures import *
```

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

index = DensePQHNSW.build(data_path, m, efConstruction, m_pq, nbits, "ip", sample_size)
```

<!--
Load queries

```python
queries_path = "" # your query file

queries = []
with open(queries_path, 'r') as f:
    for line in f:
        queries.append(json.loads(line))

MAX_TOKEN_LEN = 30
string_type  = f'U{MAX_TOKEN_LEN}'

queries_ids = np.array([q['id'] for q in queries], dtype=string_type)

query_components = []
query_values = []

for query in queries:
    vector = query['vector']
    query_components.append(np.array(list(vector.keys()), dtype=string_type))
    query_values.append(np.array(list(vector.values()), dtype=np.float32))
```
-->
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

```python
dists, ids = index.search(components, values, k, efSearch)
```
<!--

Evaluation

```python
ir_results = [ir_measures.ScoredDoc(query_id, doc_id, score) for r in results for (query_id, score, doc_id) in r]
qrels = ir_datasets.load('msmarco-passage/dev/small').qrels

ir_measures.calc_aggregate([RR@10], qrels, ir_results)
```
