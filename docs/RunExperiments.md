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

The script will build an index using the parameters in the `[indexing_parameters]` section of the TOML file.  
The index is saved in the directory `~/knn_indexes/dense_datasets/sift1M`.  
You can change directory names by modifying the `[folders]` section in the TOML file.

Next, the script will query the same index with different parameters, as specified in the `[query]` section.  
These parameters provide different trade-offs between query time and accuracy.

**Important**: if your machine is NUMA, the NUMA setting in the TOML file should be UNcommented and should be configured according to your hardware for better performance. 

### Getting the Results
The script creates a folder named `sift_hnsw_XXX`, where `XXX` encodes the datetime at which the script was executed. This ensures that each run creates a unique directory.

Inside the folder, you can find the data collected during the experiment.

The most important file is `report.tsv`, which reports *query time* and *accuracy*.

