## Replicate Results

We provide a quick way to replicate the results of our paper. 

Use the scripts in `scripts_experiments` to quickly reproduce a result from the paper. 

Use `scripts_experiments/build_index.py` to build the indexes. Use `scripts_experiments/run_search.py` to perform a search.

Use the files in `build_toml_files_pub/` and `search_toml_files_pub` to specify the experiment to reproduce.

Here is an example

```bash
python3 scripts_experiments/run_search.py --exp search_toml_files_pub/run_search_dense_dragon.toml  
```

Make sure to provide the correct paths in the `toml` files.
