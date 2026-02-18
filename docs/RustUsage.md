## Using the Rust Code

This guide explains how to use kANNolo's Rust code independently ([standalone](#itself)) or integrate it into your own Rust project ([via Cargo](#notitsef)).

kANNolo uses the HNSW graph for indexing vectors. It supports dense plain vectors, Product Quantization-encoded dense vectors, and sparse vectors. The Rust API is vectorium-first: datasets and encoders come from `vectorium`, and HNSW operates over `vectorium::Dataset`.

#### Dense Data
In kANNolo, dense vectors are read as 1-dimensional numpy arrays (`.npy` format) given by the concatenation of all the vectors in the collection. This holds for dataset, queries, and ground truth.

#### Dense Plain Data
To build an index over dense plain vectors, load data into a `vectorium::PlainDenseDataset` (or `DenseDataset` + a plain encoder) and call `HNSW::build_index`. 

Examples of usage can be found in `src/bin/hnsw_build.rs` and `src/bin/hnsw_search.rs`.

#### Dense PQ Data
To build an index over dense PQ vectors, load data into a `vectorium` dataset, create a `vectorium::encoders::pq::ProductQuantizer`, and call `HNSW::build_index`.

Product Quantization needs two additional parameters:
- `M`: Const parameter indicating the number of subspaces of Product Quantization. Supported values: 4, 8, 16, 32, 64, 96, 128. Higher values mean more accurate approximation but higher memory occupancy. M must divide the dimensionality of the original vectors.
- The codebook size is fixed by vectorium (256 centroids per subspace); `nbits` is not configurable.

Examples of usage can be found in `src/bin/hnsw_build.rs` and `src/bin/hnsw_search.rs`.

#### Sparse Data
In kANNolo, sparse documents and queries should be in a binary format. For more details on this format see the [`docs/PythonUsage.md`](docs/PythonUsage.md)

To build an index over sparse vectors, load data into a `vectorium::PlainSparseDataset` (or `SparseDataset` + a plain encoder) and call `HNSW::build_index`.
For sparse vectors, `--component-type` controls sparse index component width (`u16` or `u32`), and `dotvbyte` encoding requires `--component-type u16`.

Examples of usage can be found in `src/bin/hnsw_build.rs` and `src/bin/hnsw_search.rs`.

---

### **Index Parameters**
kANNolo's HNSW index structure is very simple, consisting in only two parameters to control the index quality during construction:

- `--m`: Upper bound for the number of neighbors of each node in the graph in the upper levels of HNSW. The upper bound on ground level is doubled. Higher values result in a better connected graph. As a side effect, construction becomes slower, and above a certain threshold also search can become slower due to the high number of distances computed at each hop in the graph.
- `--ef-construction`: Size of the candidates pool during the construction process. Higher values translate into a more precise construction and thus a more accurate search. As a downside, higher values lead to a slower construction.

Two additional parameters, `initial_build_batch_size` and `max_build_batch_size`, regulate the parallelization of the construction process. We decided to not allow for the tuning of these parameters in our current binary files in which they are less prone to adjustments and they are not part of the original HNSW structure.

To create and serialize an HNSW index on sparse data `documents.bin` with `num_neighbors_per_vec` (`m`) equal to 32 and `ef_construction` (efc) equal to 2000 for Maximum Inner Product Search (MIPS), run:

```bash
./target/release/hnsw_build \
--data-file documents.bin \
--output-file kannolo_sparse_index \
--dataset-type sparse \
--value-type f16 \
--component-type u16 \
--encoder plain \
--m 32 \
--ef-construction 2000 \
--distance dotproduct
```

To create and serialize an HNSW index on sparse data using DotVByte encoding, run:

```bash
./target/release/hnsw_build \
--data-file documents.bin \
--output-file kannolo_sparse_dotvbyte_index \
--dataset-type sparse \
--component-type u16 \
--encoder dotvbyte \
--m 32 \
--ef-construction 2000 \
--distance dotproduct
```


To create and serialize an HNSW index on dense data `documents.npy` with `num_neighbors_per_vec` (`m`) equal to 16 and `ef_construction` (efc) equal to 200 for Euclidean Nearest Neighbors Search, run:

```bash
./target/release/hnsw_build \
--data-file documents.npy \
--output-file kannolo_dense_index \
--dataset-type dense \
--value-type f32 \
--encoder plain \
--m 16 \
--ef-construction 200 \
--distance euclidean
```

To create and serialize an HNSW index on dense PQ-encoded data with 64 subspaces and 256 centroids per subspace, obtained from plain vectors `documents.npy`, with `num_neighbors_per_vec` (`m`) equal to 16 and `ef_construction` (efc) equal to 200 for Maximum Inner Product Search (MIPS), run:

```bash
./target/release/hnsw_build \
--data-file documents.npy \
--output-file kannolo_dense_pq_index \
--dataset-type dense \
--encoder pq \
--m 16 \
--ef-construction 200 \
--distance dotproduct \
--pq-subspaces 64
```

Vectorium PQ selects its training samples internally. The CLI flags `--nbits` and `--sample-size` are accepted for compatibility but ignored.

---

### **Executing Queries**
One parameter trades off efficiency and accuracy:

- `--ef_search`: Size of the candidates pool during the search process. Higher values translate into a more precise search.

To search for top `k=10` results of sparse queries `queries.bin` with an HNSW index on sparse data saved in `kannolo_sparse_index` with `ef_search` parameter equal to 40, and save results in `results_sparse`, run:

Example command:

```bash
./target/release/hnsw_search \
--index-file kannolo_sparse_index \
--query-file queries.bin \
--dataset-type sparse \
--value-type f16 \
--encoder plain \
--distance dotproduct \
--k 10 \
--ef-search 40 \
--output-path results_sparse 
```

To search for top `k=10` results of dense queries `queries.npy` with an HNSW index on dense data saved in `kannolo_dense_index` with `ef_search` parameter equal to 200, and save results in `results_dense`, run:

Example command:

```bash
./target/release/hnsw_search \
--index-file kannolo_dense_index \
--query-file queries.npy \
--dataset-type dense \
--value-type f32 \
--encoder plain \
--distance euclidean \
--k 10 \
--ef-search 200 \
--output-path results_dense 
```


To search for top `k=10` results of dense queries `queries.npy` with an HNSW index on dense PQ-encoded data saved in `kannolo_dense_pq_index` with `ef_search` parameter equal to 200, and save results in `results_pq`, run:

Example command:

```bash
./target/release/hnsw_search \
--index-file kannolo_dense_pq_index \
--query-file queries.npy \
--dataset-type dense \
--encoder pq \
--pq-subspaces 64 \
--distance dotproduct \
--k 10 \
--ef-search 200 \
--output-path results_pq 
```

To search with a sparse DotVByte index, run:

```bash
./target/release/hnsw_search \
--index-file kannolo_sparse_dotvbyte_index \
--query-file queries.bin \
--dataset-type sparse \
--component-type u16 \
--encoder dotvbyte \
--distance dotproduct \
--k 10 \
--ef-search 40 \
--output-path results_dotvbyte
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
