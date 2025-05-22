

<h1 align="center">kANNolo</h1>
<p align="center">
    <img width="300px" src="/imgs/kannolo_with_text.jpeg" />
</p>

<p align="center">
    <a href="https://rdcu.be/eii62"><img src="https://badgen.net/static/paper/ECIR 2025/yellow"" /></a>  
    <a href="https://arxiv.org/abs/2501.06121"><img src="https://badgen.net/static/arXiv/2501.06121/red" /></a>
</p>

<p align="center">    
    <a href="https://crates.io/crates/kannolo"><img src="https://badgen.infra.medigy.com/crates/v/kannolo" /></a>
    <a href="https://crates.io/crates/kannolo"><img src="https://badgen.infra.medigy.com/crates/d/kannolo" /></a>
    <a href="LICENSE"><img src="https://badgen.net/static/license/MIT/blue" /></a>
</p>

<!--
<p align="center">    
    <a href="https://crates.io/crates/seismic"><img src="https://badgen.infra.medigy.com/crates/v/seismic" /></a>
    <a href="https://crates.io/crates/seismic"><img src="https://badgen.infra.medigy.com/crates/d/seismic" /></a>
    <a href="LICENSE.md"><img src="https://badgen.net/static/license/MIT/blue" /></a>
</p>

-->

kANNolo is a research-oriented library for Approximate Nearest Neighbors (ANN) search written in Rust 🦀. It is explicitly designed to combine usability with performance effectively. Designed with modularity and researchers in mind, kANNolo makes prototyping new ANN search algorithms and data structures easy. kANNolo supports both dense and sparse embeddings seamlessly. It implements the HNSW graph index and Product Quantization.


### Python - Maximum performance
If you want to compile the package optimized for your CPU, you need to install the package from the Source Distribution.
In order to do that you need to have the Rust toolchain installed. Use the following commands:
#### Prerequisites
Install Rust (via `rustup`):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
#### Installation
```bash
RUSTFLAGS="-C target-cpu=native" pip install --no-binary :all: kannolo
```
This will compile the Rust code tailored for your machine, providing maximum performance.

### Python - Easy installation
If you are not interested in obtaining the maximum performance, you can install the package from a prebuilt Wheel.
If a compatible wheel exists for your platform, `pip` will download and install it directly, avoiding the compilation phase.
If no compatible wheel exists, pip will download the source distribution and attempt to compile it using the Rust compiler (rustc).
```bash
pip install kannolo
```

Prebuilt wheels are available for Linux platforms (x86_64, i686, aarch64) with different Python implementation (CPython, PyPy) for linux distros using glibc 2.17 or later.
Wheels are also available x86_64 platforms with linux distros using musl 1.2 or later.

### Rust 

This command allows you to compile all the Rust binaries contained in `src/bin`

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Details on how to use kANNolo's core engine in Rust 🦀 can be found in [`docs/RustUsage.md`](docs/RustUsage.md).

Details on how to use kANNolo's Python interface can be found in [`docs/PythonUsage.md`](docs/PythonUsage.md).


### Resources
Check out our `docs` folder for a more detailed guide on how to use kANNolo directly in Rust, replicate the results of our paper, or use kANNolo with your custom collection. 

Disclaimer: The results in the paper are obtained with a direct-access table shared among threads to keep track of visited nodes. In the current version, this is substituted with a hash set, with the double goal of simplifying the code for users and to make it independent of the size of the dataset, a feature that one would like to enable when dealing with large datasets. This may affect performance.

### <a name="bib">📚 Bibliography</a>
Leonardo Delfino, Domenico Erriquez, Silvio Martinico, Franco Maria Nardini, Cosimo Rulli and Rossano Venturini. "*kANNolo: Sweet and Smooth Approximate k-Nearest Neighbors Search*." Proc. ECIR. 2025.


### Citation License
The source code in this repository is subject to the following citation license:

By downloading and using this software, you agree to cite the under-noted paper in any kind of material you produce where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, article, poster, presentation, or documentation. By using this software, you have agreed to the citation license.


ECIR 2025
```bibtex
@InProceedings{10.1007/978-3-031-88717-8_29,
author="Leonardo Delfino and
        Domenico Erriquez and
        Silvio Martinico and
        Franco Maria Nardini and
        Cosimo Rulli and
        Rossano Venturini",
title="kANNolo: Sweet and Smooth Approximate k-Nearest Neighbors Search",
booktitle="Advances in Information Retrieval",
year="2025",
publisher="Springer Nature Switzerland",
pages="400--406",
isbn="978-3-031-88717-8"
}
```

