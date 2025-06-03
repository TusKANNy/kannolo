use crate::distances::sparse_dot_product_with_merge;
use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::quantizers::sparse_plain_quantizer::SparsePlainQuantizer;
use crate::topk_selectors::OnlineTopKSelector;
use crate::{Dataset, DistanceType, Float, GrowableDataset};
use crate::{DotProduct, EuclideanDistance};

use half::f16;

use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::prelude::*;

// Reading files
use std::fs::File;
use std::io::{BufReader, Read, Result as IoResult};
use std::ops::Range;
use std::path::Path;

use crate::{SparseVector1D, Vector1D};

use serde::{Deserialize, Serialize};

pub trait Container {
    type Type<T>: AsRef<[T]> + Default;

    #[inline]
    fn default<T>() -> Self::Type<T>
    where
        Self::Type<T>: Default,
    {
        Self::Type::<T>::default()
    }

    #[inline]
    fn with_initial_value<T>(initial_value: T) -> Self::Type<T>
    where
        Self::Type<T>: Default,
        T: Clone,
    {
        let mut container = Self::default::<T>();
        if let Some(vec) = Self::as_mut_vec(&mut container) {
            vec.push(initial_value);
        }
        container
    }

    #[inline]
    fn as_mut_vec<T>(_container: &mut Self::Type<T>) -> Option<&mut Vec<T>> {
        None
    }
}

impl<U> Container for Vec<U> {
    type Type<T> = Vec<T>;

    #[inline]
    fn as_mut_vec<T>(container: &mut Vec<T>) -> Option<&mut Vec<T>> {
        Some(container)
    }
}

impl<U> Container for Box<[U]> {
    type Type<T> = Box<[T]>;
}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparseDataset<Q, C: Container = Vec<()>>
where
    Q: Quantizer,
{
    values: C::Type<Q::OutputItem>,
    components: C::Type<u16>,
    offsets: C::Type<usize>,
    n_vecs: usize,
    d: usize,
    quantizer: Q,
}

impl<Q, C: Container> Dataset<Q> for SparseDataset<Q, C>
where
    Q: Quantizer<DatasetType = Self>,
    C::Type<Q::OutputItem>: AsRef<[Q::OutputItem]>,
    C::Type<u16>: AsRef<[u16]>,
    C::Type<usize>: AsRef<[usize]>,
{
    type DataType<'a>
        = SparseVector1D<&'a [u16], &'a [Q::OutputItem]>
    where
        Q: 'a,
        C: 'a,
        Q::OutputItem: 'a;

    #[inline]
    fn new(quantizer: Q, _d: usize) -> Self {
        SparseDataset {
            values: C::default::<Q::OutputItem>(),
            components: C::default::<u16>(),
            offsets: C::with_initial_value(0),
            n_vecs: 0,
            d: quantizer.m(),
            quantizer,
        }
    }

    #[inline]
    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.n_vecs, self.d)
    }

    #[inline]
    fn data<'a>(&'a self) -> Self::DataType<'a> {
        SparseVector1D::new(
            self.components.as_ref(),
            self.values.as_ref(),
            self.d as usize,
        )
    }

    #[inline]
    fn dim(&self) -> usize {
        // id maximum component + 1
        self.d
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_vecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.components.as_ref().len()
    }

    fn get_space_usage_bytes(&self) -> usize {
        let components = self.components.as_ref().len() * std::mem::size_of::<u16>();
        let values = self.values.as_ref().len() * std::mem::size_of::<Q::OutputItem>();
        let offsets = self.offsets.as_ref().len() * std::mem::size_of::<usize>();
        components + values + offsets + self.quantizer.get_space_usage_bytes()
    }

    #[inline]
    fn get<'a>(&'a self, index: usize) -> Self::DataType<'a> {
        assert!(index < self.len(), "Index out of bounds.");

        let v_components =
            &self.components.as_ref()[Self::vector_range(self.offsets.as_ref(), index)];
        let v_values = &self.values.as_ref()[Self::vector_range(self.offsets.as_ref(), index)];

        SparseVector1D::new(v_components, v_values, self.d)
    }

    #[inline]
    fn compute_distance_by_id(&self, idx1: usize, idx2: usize) -> f32
    where
        Q::OutputItem: Float,
    {
        let document1 = self.get(idx1);
        let document2 = self.get(idx2);
        match self.quantizer().distance() {
            // Euclidean distance Not supported, raise error if called
            DistanceType::Euclidean => {
                panic!("Euclidean distance is not supported for sparse datasets.")
            }
            DistanceType::DotProduct => {
                -1.0 * sparse_dot_product_with_merge(&document1, &document2)
            }
        }
    }

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = Self::DataType<'a>>
    where
        Q::OutputItem: 'a,
    {
        SparseDatasetIter::new(self)
    }

    #[inline]
    fn search<'a, H: OnlineTopKSelector>(
        &self,
        query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
        heap: &mut H,
    ) -> Vec<(f32, usize)>
    where
        Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>,
    {
        assert_eq!(
            query.components_as_slice().len(),
            query.values_as_slice().len(),
            "Query components and values length must match."
        );

        if self.data().values_as_slice().is_empty() {
            return Vec::new();
        }

        let evaluator = self.query_evaluator(query);
        let distances = evaluator.compute_distances(&self, 0..self.len());
        evaluator.topk_retrieval(distances, heap)
    }
}

impl<Q, C: Container> SparseDataset<Q, C>
where
    Q: Quantizer,
    C::Type<Q::OutputItem>: AsRef<[Q::OutputItem]>,
    C::Type<u16>: AsRef<[u16]>,
    C::Type<usize>: AsRef<[usize]>,
{
    #[inline]
    pub fn values(&self) -> &[Q::OutputItem] {
        self.values.as_ref()
    }

    #[inline]
    pub fn components(&self) -> &[u16] {
        self.components.as_ref()
    }

    #[inline]
    pub fn offsets(&self) -> &[usize] {
        self.offsets.as_ref()
    }

    #[must_use]
    #[inline(always)]
    fn vector_range(offsets: &[usize], id: usize) -> Range<usize> {
        assert!(id <= offsets.len(), "{id} is out of range");

        // Safety: safe accesses due to the check above
        unsafe {
            Range {
                start: *offsets.get_unchecked(id),
                end: *offsets.get_unchecked(id + 1),
            }
        }
    }

    #[must_use]
    #[inline]
    pub fn get_with_offset(
        &self,
        offset: usize,
        len: usize,
    ) -> SparseVector1D<&[u16], &[Q::OutputItem]> {
        assert!(
            offset + len <= self.components.as_ref().len(),
            "The id is out of range"
        );

        let v_components = &self.components.as_ref()[offset..offset + len];
        let v_values = &self.values.as_ref()[offset..offset + len];

        SparseVector1D::new(v_components, v_values, self.d)
    }

    #[must_use]
    #[inline]
    pub fn offset_to_id(&self, offset: usize) -> usize {
        self.offsets.as_ref().binary_search(&offset).unwrap()
    }

    #[must_use]
    #[inline]
    pub fn vector_len(&self, id: usize) -> usize {
        assert!(
            id < self.offsets.as_ref().len() - 1,
            "The id is out of range"
        );

        self.offsets.as_ref()[id + 1] - self.offsets.as_ref()[id]
    }
}

impl<Q, C> GrowableDataset<Q> for SparseDataset<Q, C>
where
    Q: Quantizer<DatasetType = Self>,
    Q::OutputItem: Copy + Default,
    C: Container<Type<Q::OutputItem> = Vec<Q::OutputItem>>,
    C: Container<Type<u16> = Vec<u16>>,
    C: Container<Type<usize> = Vec<usize>>,
{
    type InputDataType<'a>
        = SparseVector1D<&'a [u16], &'a [Q::InputItem]>
    where
        Q::InputItem: 'a;

    #[inline]
    fn push<'a>(&mut self, vec: &Self::InputDataType<'a>) {
        let (components, values) = (vec.components_as_slice(), vec.values_as_slice());
        assert_eq!(
            components.len(),
            values.len(),
            "Vectors have different sizes"
        );
        assert!(!components.is_empty());
        assert!(
            components.windows(2).all(|w| w[0] <= w[1]),
            "Components must be given in sorted order"
        );

        if *components.last().unwrap() as usize >= self.d {
            self.d = *components.last().unwrap() as usize + 1;
        }

        self.components.extend(components);

        let old_size = self.values.len();
        let new_size = old_size + values.len();
        self.values.resize(new_size, Default::default());

        self.quantizer
            .encode(values, &mut self.values[old_size..new_size]);

        // self.values.extend(values);

        self.offsets
            .push(*self.offsets.last().unwrap() + values.len());
        self.n_vecs += 1;
    }
}

impl<Q, C> SparseDataset<Q, C>
where
    Q: Quantizer<DatasetType = Self>,
    C: Container<Type<Q::OutputItem> = Vec<Q::OutputItem>>,
    C: Container<Type<u16> = Vec<u16>>,
    C: Container<Type<usize> = Vec<usize>>,
{
    /// Reads a binary file and returns a SparseDataset.
    /// Arguments:
    /// * `fname`: The name of the file to read.
    /// * `d`: The dimensionality of the dataset.
    pub fn read_bin_file(
        fname: &str,
        d: usize,
    ) -> IoResult<SparseDataset<SparsePlainQuantizer<f32>>> {
        Self::read_bin_file_limit(fname, None, d)
    }

    pub fn read_bin_file_f16(
        fname: &str,
        limit: Option<usize>,
        d: usize,
    ) -> IoResult<SparseDataset<SparsePlainQuantizer<f16>>> {
        let path = Path::new(fname);
        let f = File::open(path)?;
        // let f_size = f.metadata().unwrap().len() as usize;

        let mut br = BufReader::new(f);

        let mut buffer_d = [0u8; std::mem::size_of::<u32>()];
        let mut buffer = [0u8; std::mem::size_of::<f32>()];

        br.read_exact(&mut buffer_d)?;
        let mut n_vecs = u32::from_le_bytes(buffer_d) as usize;

        if let Some(n) = limit {
            n_vecs = n.min(n_vecs);
        }

        let quantizer = SparsePlainQuantizer::<f16>::new(n_vecs, DistanceType::DotProduct);
        let mut data = SparseDataset::new(quantizer, 0);

        for _ in 0..n_vecs {
            br.read_exact(&mut buffer_d)?;
            let n = u32::from_le_bytes(buffer_d) as usize;

            let mut components = Vec::with_capacity(n);
            let mut values = Vec::<f16>::with_capacity(n);

            for _ in 0..n {
                br.read_exact(&mut buffer_d)?;
                let c = u32::from_le_bytes(buffer_d) as u16;
                components.push(c);
            }
            for _ in 0..n {
                br.read_exact(&mut buffer)?;
                let v = f32::from_le_bytes(buffer);
                values.push(f16::from_f32(v));
            }

            data.push(&SparseVector1D::new(&components, &values, d));
        }

        Ok(data)
    }

    pub fn read_bin_file_limit(
        fname: &str,
        limit: Option<usize>,
        d: usize,
    ) -> IoResult<SparseDataset<SparsePlainQuantizer<f32>>> {
        let path = Path::new(fname);
        let f = File::open(path)?;
        // let f_size = f.metadata().unwrap().len() as usize;

        let mut br = BufReader::new(f);

        let mut buffer_d = [0u8; std::mem::size_of::<u32>()];
        let mut buffer = [0u8; std::mem::size_of::<f32>()];

        br.read_exact(&mut buffer_d)?;
        let mut n_vecs = u32::from_le_bytes(buffer_d) as usize;

        if let Some(n) = limit {
            n_vecs = n.min(n_vecs);
        }

        let quantizer = SparsePlainQuantizer::<f32>::new(n_vecs, DistanceType::DotProduct);
        let mut data = SparseDataset::new(quantizer, 0);

        for _ in 0..n_vecs {
            br.read_exact(&mut buffer_d)?;
            let n = u32::from_le_bytes(buffer_d) as usize;

            let mut components = Vec::with_capacity(n);
            let mut values = Vec::<f32>::with_capacity(n);

            for _ in 0..n {
                br.read_exact(&mut buffer_d)?;
                let c = u32::from_le_bytes(buffer_d) as u16;
                components.push(c);
            }
            for _ in 0..n {
                br.read_exact(&mut buffer)?;
                let v = f32::from_le_bytes(buffer);
                values.push(v);
            }

            data.push(&SparseVector1D::new(&components, &values, d));
        }

        Ok(data)
    }

    /// Reads the binary file and returns a tuple of:
    /// (components: Vec<u16>, values: Vec<f16>, offsets: Vec<usize>)
    pub fn read_bin_file_parts_f16(
        fname: &str,
        limit: Option<usize>,
    ) -> IoResult<(Vec<u16>, Vec<f16>, Vec<usize>)> {
        let path = Path::new(fname);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read the total number of vectors.
        let mut buffer_u32 = [0u8; std::mem::size_of::<u32>()];
        reader.read_exact(&mut buffer_u32)?;
        let mut n_vecs = u32::from_le_bytes(buffer_u32) as usize;
        if let Some(limit) = limit {
            n_vecs = n_vecs.min(limit);
        }

        let mut components = Vec::new();
        let mut values = Vec::new();
        // Offsets: first element is 0. Each new vector's data starts at the last offset.
        let mut offsets = Vec::with_capacity(n_vecs + 1);
        offsets.push(0);

        let mut buffer_f32 = [0u8; std::mem::size_of::<f32>()];

        for _ in 0..n_vecs {
            // Read number of components/values for this vector.
            reader.read_exact(&mut buffer_u32)?;
            let n = u32::from_le_bytes(buffer_u32) as usize;

            // Read components (as u32, then cast to u16).
            for _ in 0..n {
                reader.read_exact(&mut buffer_u32)?;
                let comp = u32::from_le_bytes(buffer_u32) as u16;
                components.push(comp);
            }
            // Read values (f32 converted to f16).
            for _ in 0..n {
                reader.read_exact(&mut buffer_f32)?;
                let val_f32 = f32::from_le_bytes(buffer_f32);
                values.push(f16::from_f32(val_f32));
            }
            // Update the offsets. The next vector starts after the current one.
            let last_offset = *offsets.last().unwrap();
            offsets.push(last_offset + n);
        }

        Ok((components, values, offsets))
    }

    /// Reads the binary file and returns a tuple of:
    /// (components: Vec<u16>, values: Vec<f16>, offsets: Vec<usize>)
    pub fn read_bin_file_parts_f32(
        fname: &str,
        limit: Option<usize>,
    ) -> IoResult<(Vec<u16>, Vec<f32>, Vec<usize>)> {
        let path = Path::new(fname);
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read the total number of vectors.
        let mut buffer_u32 = [0u8; std::mem::size_of::<u32>()];
        reader.read_exact(&mut buffer_u32)?;
        let mut n_vecs = u32::from_le_bytes(buffer_u32) as usize;
        if let Some(limit) = limit {
            n_vecs = n_vecs.min(limit);
        }

        let mut components = Vec::new();
        let mut values = Vec::new();
        // Offsets: first element is 0. Each new vector's data starts at the last offset.
        let mut offsets = Vec::with_capacity(n_vecs + 1);
        offsets.push(0);

        let mut buffer_f32 = [0u8; std::mem::size_of::<f32>()];

        for _ in 0..n_vecs {
            // Read number of components/values for this vector.
            reader.read_exact(&mut buffer_u32)?;
            let n = u32::from_le_bytes(buffer_u32) as usize;

            // Read components (as u32, then cast to u16).
            for _ in 0..n {
                reader.read_exact(&mut buffer_u32)?;
                let comp = u32::from_le_bytes(buffer_u32) as u16;
                components.push(comp);
            }
            // Read values (f32 converted to f16).
            for _ in 0..n {
                reader.read_exact(&mut buffer_f32)?;
                let val_f32 = f32::from_le_bytes(buffer_f32);
                values.push(val_f32);
            }
            // Update the offsets. The next vector starts after the current one.
            let last_offset = *offsets.last().unwrap();
            offsets.push(last_offset + n);
        }

        Ok((components, values, offsets))
    }

    /// Constructs a SparseDataset from the given vectors.
    /// For each vector i, the components are in `components[offsets[i]..offsets[i+1]]`,
    /// and similarly for values.
    pub fn from_vecs_f16(
        components: &[u16],
        values: &[f16],
        offsets: &[usize],
        d: usize,
    ) -> IoResult<SparseDataset<SparsePlainQuantizer<f16>>> {
        let n_vecs = offsets.len() - 1;
        let quantizer = SparsePlainQuantizer::<f16>::new(d, DistanceType::DotProduct);
        let mut dataset = SparseDataset::new(quantizer, d);

        for i in 0..n_vecs {
            let start = offsets[i];
            let end = offsets[i + 1];
            let vec_components = &components[start..end];
            let vec_values = &values[start..end];

            dataset.push(&SparseVector1D::new(vec_components, vec_values, d));
        }

        Ok(dataset)
    }

    /// Constructs a SparseDataset from the given vectors.
    /// For each vector i, the components are in `components[offsets[i]..offsets[i+1]]`,
    /// and similarly for values.
    pub fn from_vecs_f32(
        components: &[u16],
        values: &[f32],
        offsets: &[usize],
        d: usize,
    ) -> IoResult<SparseDataset<SparsePlainQuantizer<f32>>> {
        let n_vecs = offsets.len() - 1;
        let quantizer = SparsePlainQuantizer::<f32>::new(d, DistanceType::DotProduct);
        let mut dataset = SparseDataset::new(quantizer, d);

        for i in 0..n_vecs {
            let start = offsets[i];
            let end = offsets[i + 1];
            let vec_components = &components[start..end];
            let vec_values = &values[start..end];

            dataset.push(&SparseVector1D::new(vec_components, vec_values, d));
        }

        Ok(dataset)
    }
}

impl<Q> From<SparseDataset<Q, Vec<()>>> for SparseDataset<Q, Box<[()]>>
where
    Q: Quantizer,
{
    fn from(mutable_dataset: SparseDataset<Q, Vec<()>>) -> Self {
        SparseDataset {
            values: mutable_dataset.values.into_boxed_slice(),
            components: mutable_dataset.components.into_boxed_slice(),
            offsets: mutable_dataset.offsets.into_boxed_slice(),
            n_vecs: mutable_dataset.n_vecs,
            d: mutable_dataset.d,
            quantizer: mutable_dataset.quantizer,
        }
    }
}

impl<Q, C> AsRef<[Q::OutputItem]> for SparseDataset<Q, C>
where
    Q: Quantizer,
    C: Container<Type<Q::OutputItem> = Vec<Q::OutputItem>>,
{
    #[inline]
    fn as_ref(&self) -> &[Q::OutputItem] {
        self.values.as_ref()
    }
}

impl<Q, C> AsMut<[Q::OutputItem]> for SparseDataset<Q, C>
where
    Q: Quantizer,
    C: Container<Type<Q::OutputItem> = Vec<Q::OutputItem>>,
{
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [Q::OutputItem] {
        self.values.as_mut()
    }
}

// Sparse
#[derive(Clone)]
pub struct SparseDatasetIter<'a, Q>
where
    Q: Quantizer,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [u16],
    values: &'a [Q::OutputItem],
    d: usize,
}

impl<'a, Q> SparseDatasetIter<'a, Q>
where
    Q: Quantizer,
{
    #[inline]
    pub fn new<C>(dataset: &'a SparseDataset<Q, C>) -> Self
    where
        C: Container,
        C::Type<<Q as Quantizer>::OutputItem>: AsRef<[Q::OutputItem]>,
        C::Type<u16>: AsRef<[u16]>,
        C::Type<usize>: AsRef<[usize]>,
    {
        Self {
            last_offset: 0,
            offsets: &dataset.offsets()[1..],
            components: dataset.components(),
            values: dataset.values(),
            d: dataset.d,
        }
    }
}

impl<'a, Q> Iterator for SparseDatasetIter<'a, Q>
where
    Q: Quantizer,
{
    type Item = SparseVector1D<&'a [u16], &'a [Q::OutputItem]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&next_offset, rest) = self.offsets.split_first()?;
        self.offsets = rest;

        let (cur_components, rest) = self.components.split_at(next_offset - self.last_offset);
        self.components = rest;

        let (cur_values, rest) = self.values.split_at(next_offset - self.last_offset);
        self.values = rest;

        self.last_offset = next_offset;

        Some(SparseVector1D::new(cur_components, cur_values, self.d))
    }
}

/// Parallel iterator over the sparse dataset
#[derive(Clone)]
pub struct ParSparseDatasetIter<'a, Q>
where
    Q: Quantizer,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [u16],
    values: &'a [Q::OutputItem],
    d: usize,
}

impl<'a, Q, C> IntoParallelIterator for &'a SparseDataset<Q, C>
where
    Q: Quantizer + Sync,
    Q::OutputItem: Sync,
    C: Container<Type<Q::OutputItem> = Vec<Q::OutputItem>>,
    C: Container<Type<u16> = Vec<u16>>,
    C: Container<Type<usize> = Vec<usize>>,
{
    type Iter = ParSparseDatasetIter<'a, Q>;
    type Item = SparseVector1D<&'a [u16], &'a [Q::OutputItem]>;

    fn into_par_iter(self) -> Self::Iter {
        ParSparseDatasetIter {
            last_offset: self.offsets()[0],
            offsets: &self.offsets()[1..],
            components: &self.components(),
            values: &self.values(),
            d: self.d,
        }
    }
}

impl<'a, Q> ParallelIterator for ParSparseDatasetIter<'a, Q>
where
    Q: Quantizer,
    Q::OutputItem: Sync,
{
    type Item = SparseVector1D<&'a [u16], &'a [Q::OutputItem]>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.offsets.len())
    }
}

impl<'a, Q> IndexedParallelIterator for ParSparseDatasetIter<'a, Q>
where
    Q: Quantizer,
    Q::OutputItem: Sync,
{
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        let producer = SparseDatasetProducer::from(self);
        callback.callback(producer)
    }

    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<'a, Q> ExactSizeIterator for SparseDatasetIter<'a, Q>
where
    Q: Quantizer,
{
    fn len(&self) -> usize {
        self.offsets.len()
    }
}

impl<'a, Q> DoubleEndedIterator for SparseDatasetIter<'a, Q>
where
    Q: Quantizer,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let (&last_offset, rest) = self.offsets.split_last()?;
        self.offsets = rest;

        let (rest, cur_components) = self
            .components
            .split_at(self.components.len() - (last_offset - self.last_offset));
        self.components = rest;

        let (rest, cur_values) = self
            .values
            .split_at(self.values.len() - (last_offset - self.last_offset));
        self.values = rest;

        self.last_offset = last_offset;

        Some(SparseVector1D::new(cur_components, cur_values, self.d))
    }
}

struct SparseDatasetProducer<'a, Q>
where
    Q: Quantizer,
{
    last_offset: usize,
    offsets: &'a [usize],
    components: &'a [u16],
    values: &'a [Q::OutputItem],
    d: usize,
}

impl<'a, Q> Producer for SparseDatasetProducer<'a, Q>
where
    Q: Quantizer,
    Q::OutputItem: Sync,
{
    type Item = SparseVector1D<&'a [u16], &'a [Q::OutputItem]>;
    type IntoIter = SparseDatasetIter<'a, Q>;

    fn into_iter(self) -> Self::IntoIter {
        SparseDatasetIter {
            last_offset: self.last_offset,
            offsets: self.offsets,
            components: self.components,
            values: self.values,
            d: self.d,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let left_last_offset = self.last_offset;

        let (left_offsets, right_offsets) = self.offsets.split_at(index);
        let right_last_offset = *left_offsets.last().unwrap();

        let (left_components, right_components) = self
            .components
            .split_at(right_last_offset - left_last_offset);
        let (left_values, right_values) =
            self.values.split_at(right_last_offset - left_last_offset);

        (
            SparseDatasetProducer {
                last_offset: left_last_offset,
                offsets: left_offsets,
                components: left_components,
                values: left_values,
                d: self.d,
            },
            SparseDatasetProducer {
                last_offset: right_last_offset,
                offsets: right_offsets,
                components: right_components,
                values: right_values,
                d: self.d,
            },
        )
    }
}

impl<'a, Q> From<ParSparseDatasetIter<'a, Q>> for SparseDatasetProducer<'a, Q>
where
    Q: Quantizer,
{
    fn from(iter: ParSparseDatasetIter<'a, Q>) -> Self {
        SparseDatasetProducer {
            last_offset: iter.last_offset,
            offsets: iter.offsets,
            components: iter.components,
            values: iter.values,
            d: iter.d,
        }
    }
}
