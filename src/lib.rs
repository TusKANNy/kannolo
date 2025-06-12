#![feature(iter_array_chunks)]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_mm_shuffle))]
#![feature(portable_simd)]
#![feature(thread_id_value)]
#![feature(array_chunks)]
#![feature(allocator_api)]

pub mod topk_selectors;

use pyo3::types::PyModuleMethods;

pub mod pylib;
use crate::pylib::DensePQHNSW as DensePQIndexPy;
use crate::pylib::DensePlainHNSW as DensePlainIndexPy;
use crate::pylib::DensePlainHNSWf16 as DensePlainIndexPyf16;
use crate::pylib::SparsePlainHNSW as SparsePlainIndexPy;
use crate::pylib::SparsePlainHNSWf16 as SparsePlainIndexPyf16;
use num_traits::{ToPrimitive, Zero};
use pyo3::prelude::PyModule;
use pyo3::{pymodule, Bound, PyResult};

pub mod clustering {
    pub mod kmeans;
    pub use kmeans::KMeans;
    pub use kmeans::KMeansBuilder;
}

pub mod quantizers;
pub mod visited_set;
pub use quantizers::decoder;
pub use quantizers::encoder;
pub use quantizers::plain_quantizer;
pub use quantizers::pq;
pub use quantizers::quantizer;
pub use quantizers::sparse_plain_quantizer;

pub mod datasets {
    pub mod dataset;
    pub mod dense_dataset;
    pub mod sparse_dataset;
    pub mod utils;
}

pub use datasets::dataset::Dataset;
pub use datasets::dataset::GrowableDataset;
pub use datasets::dense_dataset::DenseDataset;
pub use datasets::dense_dataset::DenseDatasetIter;
pub use datasets::sparse_dataset::ParSparseDatasetIter;
pub use datasets::sparse_dataset::SparseDataset;
pub use datasets::sparse_dataset::SparseDatasetIter;
pub use datasets::utils::*;

type PlainDenseDataset<T> = DenseDataset<plain_quantizer::PlainQuantizer<T>>;

pub mod distances;
pub use distances::dot_product::*;
pub use distances::euclidean_distance::*;
pub use distances::simd::distances as simd_distances;
pub use distances::simd::transpose as simd_transpose;
pub use distances::simd::utils as simd_utils;

pub mod utils;

pub mod hnsw {
    pub mod graph_index;
}

pub mod hnsw_utils;

pub mod index_serializer;
pub use index_serializer::IndexSerializer;

use half::f16;

use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
pub enum DistanceType {
    #[default]
    Euclidean,
    DotProduct,
}

pub trait Float: Copy + Default + ToPrimitive + PartialOrd + Zero + Send + Sync {}

//impl Float for f64 {}
impl Float for f32 {}
impl Float for f16 {}

/// A trait for a 1D array, that contains elements of type `Item`.
pub trait AsRefItem {
    type Item;

    fn as_ref_item(&self) -> &[Self::Item];
}

impl<U> AsRefItem for Vec<U> {
    type Item = U;

    #[inline(always)]
    fn as_ref_item(&self) -> &[Self::Item] {
        self.as_slice()
    }
}

impl<U> AsRefItem for Box<[U]> {
    type Item = U;

    #[inline(always)]
    fn as_ref_item(&self) -> &[Self::Item] {
        self.as_ref()
    }
}

impl<'a, U> AsRefItem for &'a [U] {
    type Item = U;

    #[inline(always)]
    fn as_ref_item(&self) -> &[Self::Item] {
        self
    }
}

impl<'a, U> AsRefItem for &'a mut [U] {
    type Item = U;

    #[inline(always)]
    fn as_ref_item(&self) -> &[Self::Item] {
        self
    }
}

pub trait Vector1D {
    type ComponentsType;
    type ValuesType;

    fn len(&self) -> usize;
    fn components_as_slice(&self) -> &[Self::ComponentsType];
    fn values_as_slice(&self) -> &[Self::ValuesType];
}

static EMPTY_COMPONENTS: [(); 0] = [];

#[derive(Debug, Clone, PartialEq)]
pub struct DenseVector1D<T: AsRefItem> {
    components: (),
    values: T,
}

impl<T: AsRefItem> DenseVector1D<T> {
    #[inline]
    pub fn new(values: T) -> Self {
        DenseVector1D {
            components: (),
            values,
        }
    }
}

impl<T: AsRefItem> Vector1D for DenseVector1D<T> {
    type ComponentsType = ();
    type ValuesType = T::Item;

    #[inline(always)]
    fn len(&self) -> usize {
        self.values.as_ref_item().len()
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::ComponentsType] {
        &EMPTY_COMPONENTS
    }

    #[inline(always)]
    fn values_as_slice(&self) -> &[Self::ValuesType] {
        self.values.as_ref_item()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector1D<V, T>
where
    V: AsRefItem<Item = u16>,
    T: AsRefItem,
{
    components: V,
    values: T,
    max_component_id: u16,
    d: usize, // dimensionality of the vector space
}

impl<V, T> SparseVector1D<V, T>
where
    V: AsRefItem<Item = u16>,
    T: AsRefItem,
{
    #[inline]
    pub fn new(components: V, values: T, d: usize) -> Self {
        let max_component_id = components
            .as_ref_item()
            .iter()
            .max()
            .map(|&x| x)
            .unwrap_or(0);
        SparseVector1D {
            components,
            values,
            max_component_id,
            d,
        }
    }
}

impl<V, T> Vector1D for SparseVector1D<V, T>
where
    V: AsRefItem<Item = u16>,
    T: AsRefItem,
{
    type ComponentsType = V::Item;
    type ValuesType = T::Item;

    /// Returns the length of the sparse array.
    ///
    /// The length is defined as the value of the highest component index plus one.
    /// This length is precomputed during initialization and stored in the `max_component` field.
    /// Therefore, this method provides an O(1) access time.
    ///
    /// The length is computed as the maximum component index plus one because the components
    /// represent indices that are zero-based. Therefore, if the highest index in the components
    /// is `n`, the total length of the array must be `n + 1` to account for the zero index.
    ///
    /// # Returns
    ///
    /// The length of the array, which is the maximum component index plus one.
    ///
    #[inline(always)]
    fn len(&self) -> usize {
        (self.max_component_id as usize) + 1
    }

    #[inline(always)]
    fn components_as_slice(&self) -> &[Self::ComponentsType] {
        self.components.as_ref_item()
    }

    #[inline(always)]
    fn values_as_slice(&self) -> &[Self::ValuesType] {
        self.values.as_ref_item()
    }
}

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
