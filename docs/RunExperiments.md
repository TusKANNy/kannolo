## Replicate Results

We provide a quick way to replicate the results of our paper. 

Use the [`scripts/run_experiments.py`](scripts/run_experiments.py) script to quickly reproduce a result from the paper. 
This script is configurable via TOML files, which specify the parameters to build the index and execute queries on it.  
The script measures average query time (in microseconds), recall with respect to the true closest vectors of the query (accuracy@k), MRR or other metrics with respect to judged qrels if specified, and index space usage (bytes).

TOML files to reproduce the experiments of our paper can be found in [`experiments/ecir2025`](experiments/ecir2025).

Datasets can be found at [`Hugging Face`](https://huggingface.co/collections/tuskanny/kannolo-datasets-67f2527781f4f7a1b4c9fe54).

As an example, let's now run the experiments using the TOML file [`experiments/ecir2025/dense_sift1m.toml`](experiments/ecir2025/dense_sift1m.toml), which replicates the results of kANNolo on the SIFT1M dataset.

### <a name="bin_data">Setting up for the Experiment</a>
Let's start by creating a working directory for the data and indexes.

```bash
mkdir -p ~/knn_datasets/dense_datasets/sift1M
mkdir -p ~/knn_indexes/dense_datasets/sift1M
```

We need to download datasets, queries, ground truth (and, eventually, qrels and query IDs) as follows. Here, we are downloading SIFT1M vectors.  

```bash
cd ~/knn_datasets/dense_datasets/sift1M
wget https://huggingface.co/datasets/tuskanny/kannolo-sift1M/resolve/main/dataset.npy
wget https://huggingface.co/datasets/tuskanny/kannolo-sift1M/resolve/main/groundtruth.npy
wget https://huggingface.co/datasets/tuskanny/kannolo-sift1M/resolve/main/queries.npy

```


### Running the Experiment
We are now ready to run the experiment.

First, clone the kANNolo Git repository and compile kANNolo:

```bash
cd ~
git clone git@github.com:TusKANNy/kannolo.git
cd kannolo
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

If needed, install Rust on your machine with the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Now we can run the experiment with the following command:

```bash
python3 scripts/run_experiments.py --exp experiments/ecir2025/dense_sift1m.toml
```

Please install the required Python's libraries with the following command:
```bash
pip install -r scripts/requirements.txt
```

The script will build an index using the unified binary parameters specified at the top level of the TOML file (`build-command`, `vector-type`, `precision`, `quantizer`, `graph-type`) and the traditional indexing parameters in the `[indexing_parameters]` section (`m`, `efc`, `metric`).  
The index is saved in the directory `~/knn_indexes/dense_datasets/sift1M`.  
You can change directory names by modifying the `[folder]` section in the TOML file.

Next, the script will query the same index with different parameters, as specified in the `[query]` section.  
These parameters provide different trade-offs between query time and accuracy.

**Important**: if your machine is NUMA, the NUMA setting in the TOML file should be UNcommented and should be configured according to your hardware for better performance. 

## TOML Configuration Structure

The TOML configuration files have been updated to work with the unified binaries. Here's the structure:

### Top-level Parameters
- `build-command`: Path to the unified build binary (e.g., `"./target/release/hnsw_build"`)
- `query-command`: Path to the unified search binary (e.g., `"./target/release/hnsw_search"`)
- `vector-type`: Type of vectors - `"dense"` or `"sparse"`
- `precision`: Precision - `"f32"` or `"f16"` (Note: PQ always uses f32)
- `quantizer`: Quantizer type - `"plain"` or `"pq"`
- `graph-type`: Graph type - `"standard"` or `"fixed-degree"`

### Sections
- `[indexing_parameters]`: Traditional HNSW parameters (`m`, `efc`, `metric`)
- `[pq_parameters]`: PQ-specific parameters (`m-pq`, `nbits`, `sample-size`) when using PQ quantizer
- `[folder]`: Directory paths for data, indexes, and experiments
- `[filename]`: Filenames for dataset, queries, groundtruth, etc.
- `[settings]`: Runtime settings (k, NUMA, build flag, evaluation metric)
- `[query]`: Different ef-search values for query experiments

### Example TOML Structure

Here's an example of the complete TOML structure for a dense PQ experiment:

```toml
name = "example_hnsw_pq"
title = "Example HNSW PQ Experiment"
description = "Example experiment with Product Quantization"
dataset = "Example Dataset"
build-command = "./target/release/hnsw_build"
query-command = "./target/release/hnsw_search"
vector-type = "dense"
precision = "f32"
quantizer = "pq"
graph-type = "standard"

[settings]
k = 10
n-runs = 1
NUMA = "numactl --physcpubind='0-15' --localalloc"
build = true
metric = ""

[folder]
data = "~/knn_datasets/dense_datasets/example"
index = "~/knn_indexes/dense_datasets/example"
experiment = "."

[filename]
dataset = "dataset.npy"
queries = "queries.npy"
groundtruth = "groundtruth.npy"
index = "example_index"

[indexing_parameters]
m = 16
efc = 150
metric = "ip"

[pq_parameters]  # Only needed when quantizer = "pq"
m-pq = 64
nbits = 8
sample-size = 100000

[query]
    [query.efs_40]
    ef-search = 40
    [query.efs_80]
    ef-search = 80
``` 

### Getting the Results
The script creates a folder named `sift_hnsw_XXX`, where `XXX` encodes the datetime at which the script was executed. This ensures that each run creates a unique directory.

Inside the folder, you can find the data collected during the experiment.

The most important file is `report.tsv`, which reports *query time* and *accuracy*.

