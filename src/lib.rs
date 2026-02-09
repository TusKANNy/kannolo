#![feature(iter_array_chunks)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_mm_shuffle))]
#![feature(portable_simd)]
#![feature(thread_id_value)]

use pyo3::types::PyModuleMethods;

pub mod pylib;
use crate::pylib::DensePQHNSW as DensePQIndexPy;
use crate::pylib::DensePlainHNSW as DensePlainIndexPy;
use crate::pylib::DensePlainHNSWf16 as DensePlainIndexPyf16;
use crate::pylib::SparsePlainHNSW as SparsePlainIndexPy;
use crate::pylib::SparsePlainHNSWf16 as SparsePlainIndexPyf16;
use pyo3::prelude::PyModule;
use pyo3::{pymodule, Bound, PyResult};

pub mod graph;
pub mod visited_set;

pub mod indexes;
pub use indexes::{hnsw, hnsw_utils, index};

/// A Python module implemented in Rust. The name of this function must match the `lib.name`
/// setting in the `Cargo.toml`, otherwise Python will not be able to import the module.
#[pymodule]
pub fn kannolo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DensePlainIndexPy>()?;
    m.add_class::<DensePlainIndexPyf16>()?;
    m.add_class::<SparsePlainIndexPy>()?;
    m.add_class::<SparsePlainIndexPyf16>()?;
    m.add_class::<DensePQIndexPy>()?;
    Ok(())
}
