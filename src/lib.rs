#![feature(iter_array_chunks)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_mm_shuffle))]
#![feature(portable_simd)]
#![feature(thread_id_value)]

use pyo3::types::PyModuleMethods;

pub mod pylib;
use crate::pylib::DenseFlatIndex;
use crate::pylib::DensePQHNSW as DensePQIndexPy;
use crate::pylib::DensePlainHNSW as DensePlainIndexPy;
use crate::pylib::SparseDotVByteHNSW as SparseDotVByteIndexPy;
use crate::pylib::SparseFixedU8HNSW as SparseFixedU8IndexPy;
use crate::pylib::SparseFixedU16HNSW as SparseFixedU16IndexPy;
use crate::pylib::SparseFlatIndex;
use crate::pylib::SparseMultivecRerankIndex;
use crate::pylib::SparseMultivecTwoLevelsPQRerankIndex;
use crate::pylib::SparsePlainHNSW as SparsePlainIndexPy;
use pyo3::prelude::PyModule;
use pyo3::{Bound, PyResult, pymodule};

pub mod graph;
pub mod visited_set;

pub mod indexes;
pub use indexes::{hnsw, hnsw_utils};

/// A Python module implemented in Rust. The name of this function must match the `lib.name`
/// setting in the `Cargo.toml`, otherwise Python will not be able to import the module.
#[pymodule]
pub fn kannolo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DensePlainIndexPy>()?;
    m.add_class::<SparsePlainIndexPy>()?;
    m.add_class::<SparseDotVByteIndexPy>()?;
    m.add_class::<SparseFixedU8IndexPy>()?;
    m.add_class::<SparseFixedU16IndexPy>()?;
    m.add_class::<DensePQIndexPy>()?;
    m.add_class::<DenseFlatIndex>()?;
    m.add_class::<SparseFlatIndex>()?;
    m.add_class::<SparseMultivecRerankIndex>()?;
    m.add_class::<SparseMultivecTwoLevelsPQRerankIndex>()?;
    Ok(())
}
