## Using the Rust Code

This guide explains how to use kANNolo's Rust code independently ([standalone](#itself)) or integrate it into your own Rust project ([via Cargo](#notitsef)).

kANNolo uses the HNSW graph for indexing vectors. It supports three main configurations: dense plain vectors, Product Quantization-encoded dense vectors, and sparse vectors. All these kind of vectors can be seamlessly indexed using HNSW index.

#### Dense Data
In kANNolo, dense vectors are read as 1-dimensional numpy arrays (`.npy` format) given by the concatenation of all the vectors in the collection. This holds for dataset, queries, and ground truth.

#### Dense Plain Data
To build an index over dense plain vectors, you only need to load data, create a `DenseDataset` and a `PlainQuantizer`, and call the `build` function of the `GraphIndex`. 

Examples of usage can be found in `src/bin/hnsw_plain_dense_build` and `src/bin/hnsw_plain_dense_run`.

#### Dense PQ Data
To build an index over dense PQ vectors, you need to load data, create a `DenseDataset`, create and train a `ProductQuantizer`, and call the `build` function of the `GraphIndex`.

Product Quantization needs two additional parameters:
- `M`: Const parameter indicating the number of subspaces of Product Quantization. Higher values mean more accurate approximation but higher memory occupancy.
- `nbits`: Number of bits to store the centroids of a subspace. The number of centroids per subspace is thus 2^{nbits}. Higher values mean more accurate approximation but higher memory occupancy.

Examples of usage can be found in `src/bin/hnsw_pq_build` and `src/bin/hnsw_pq_run`.

#### Sparse Data
In kANNolo, sparse documents and queries should be in a binary format. For more details on this format see the [`docs/PythonUsage.md`](docs/PythonUsage.md)

To build an index over sparse vectors, you need to load data, create a `SparseDataset` and a `SparsePlainQuantizer`, and call the `build` function of the `GraphIndex`.

Examples of usage can be found in `src/bin/hnsw_plain_sparse_build` and `src/bin/hnsw_plain_sparse_run`.

---

### **Index Parameters**
kANNolo's HNSW index structure is very simple, consisting in only two parameters to control the index quality during construction:

- `--num_neighbors_per_vec`: Upper bound for the number of neighbors of each node in the graph in the upper levels of HNSW. The upper bound on ground level is doubled.
- `--ef_construction`: Size of the candidates pool during the construction process. Higher values translate into a more precise construction.

To create and serialize an HNSW index on sparse data `documents.bin` with `num_neighbors_per_vec` (`m`) equal to 32 and `ef_construction` (efc) equal to 2000 for Maximum Inner Product Search (MIPS), run:

```bash
./target/release/hnsw_plain_sparse_build \
--data-file documents.bin \
--output-file kannolo_sparse_index \
--m 32 \
--efc 2000 \
--metric ip
```


To create and serialize an HNSW index on dense data `documents.npy` with `num_neighbors_per_vec` (`m`) equal to 16 and `ef_construction` (efc) equal to 200 for Euclidean Nearest Neighbors Search, run:

```bash
./target/release/hnsw_plain_dense_build \
--data-file documents.npy \
--output-file kannolo_dense_index \
--m 16 \
--efc 200 \
--metric l2
```

To create and serialize an HNSW index on dense PQ-encoded data with 64 subspaces and 256 centroids per subspace, obtained from plain vectors `documents.npy`, with `num_neighbors_per_vec` (`m`) equal to 16 and `ef_construction` (efc) equal to 200 for Maximum Inner Product Search (MIPS), run:

```bash
./target/release/hnsw_plain_dense_build \
--data-file documents.npy \
--output-file kannolo_dense_pq_index \
--m 16 \
--efc 200 \
--metric ip \
--m-pq 64 \
--nbits 8 \
--sample-size 100000
```

The product quantizer is trained on a sample of size `sample_size` of the dataset.

---

### **Executing Queries**
One parameter trades off efficiency and accuracy:

- `--ef_search`: Size of the candidates pool during the search process. Higher values translate into a more precise search.

To search for top `k=10` results of sparse queries `queries.bin` with an HNSW index on sparse data saved in `kannolo_sparse_index` with `ef_search` parameter equal to 40, and save results in `results_sparse`, run:

Example command:

```bash
./target/release/hnsw_plain_sparse_run \
--index-file kannolo_sparse_index \
 --query-file queries.bin \
--k 10 \
--ef-search 40 \
--output-path results_sparse 
```

To search for top `k=10` results of dense queries `queries.npy` with an HNSW index on dense data saved in `kannolo_dense_index` with `ef_search` parameter equal to 200, and save results in `results_dense`, run:

Example command:

```bash
./target/release/hnsw_plain_dense_run \
--index-file kannolo_dense_index \
 --query-file queries.npy \
--k 10 \
--ef-search 200 \
--output-path results_dense 
```


To search for top `k=10` results of dense queries `queries.npy` with an HNSW index on dense PQ-encoded data saved in `kannolo_dense_pq_index` with `ef_search` parameter equal to 200, and save results in `results_pq`, run:

Example command:

```bash
./target/release/hnsw_pq_run \
--index-file kannolo_dense_pq_index \
 --query-file queries.npy \
--k 10 \
--ef-search 200 \
--output-path results_pq 
```

Queries are executed in **single-thread mode** by default. To enable multithreading, modify the Rust code:

```rust
queries.iter() 
// Change to:
queries.par_iter()
```

The results are written to `results.tsv`. Each query produces `k` lines in the following format:

```text
query_id\tdocument_id\tresult_rank\tscore_value
```

Where:
- `query_id`: A progressive identifier for the query.
- `document_id`: The document ID from the indexed dataset.
- `result_rank`: The ranking of the result by dot product.
- `score_value`: The score of the document-query pair. It can be (squared) Euclidean distance or inner product.


<!-- 

---

## **<a name="notitsef">Use Seismic in Your Rust Code</a>**

To incorporate the Seismic library into your Rust project, navigate to your project directory and run:

```bash
cargo add seismic
```

-->
