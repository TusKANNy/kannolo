use crate::distances::dot_product_dense_sparse;
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{DArray1, DenseDArray1, SparseDArray1};
use crate::{Dataset, DistanceType, Float};
use crate::{DotProduct, EuclideanDistance};

use crate::datasets::sparse_dataset::SparseDataset;

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparsePlainQuantizer<T> {
    d: usize,
    distance: DistanceType,
    _phantom: PhantomData<T>,
}

impl<T> SparsePlainQuantizer<T> {
    #[inline]
    pub fn new(d: usize, distance: DistanceType) -> Self {
        SparsePlainQuantizer {
            d,
            distance,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + Default + PartialOrd + Sync + Send> Quantizer for SparsePlainQuantizer<T> {
    type InputItem = T;
    type OutputItem = T;

    type DatasetType<'a>
        = SparseDataset<Self>
    where
        T: 'a;
    type Evaluator<'a>
        = SparseQueryEvaluatorPlain<'a, Self::InputItem>
    where
        Self::InputItem: Float + EuclideanDistance<T> + DotProduct<T> + 'a;

    #[inline]
    fn encode(&self, input_vectors: &[Self::InputItem], output_vectors: &mut [Self::OutputItem]) {
        output_vectors.copy_from_slice(input_vectors);
    }

    #[inline]
    fn m(&self) -> usize {
        self.d
    }

    #[inline]
    fn distance(&self) -> DistanceType {
        self.distance
    }
}

pub struct SparseQueryEvaluatorPlain<'a, T: Float + 'a> {
    dataset: &'a <<Self as QueryEvaluator<'a>>::Q as Quantizer>::DatasetType<'a>,
    dense_query: DenseDArray1<Vec<T>>,
}

impl<'a, T: Float> QueryEvaluator<'a> for SparseQueryEvaluatorPlain<'a, T> {
    type Q = SparsePlainQuantizer<T>;
    type QueryType = SparseDArray1<&'a [u16], &'a [T]>;

    #[inline]
    fn new(dataset: &'a <Self::Q as Quantizer>::DatasetType<'a>, query: Self::QueryType) -> Self {
        let mut dense_query = vec![T::zero(); dataset.dim()];
        for (&i, &v) in query
            .components_as_slice()
            .iter()
            .zip(query.values_as_slice())
        {
            dense_query[i as usize] = v;
        }

        let dense_query = DenseDArray1::new(dense_query);

        Self {
            dataset,
            dense_query,
        }
    }

    #[inline]
    fn compute_distance(&self, index: usize) -> f32 {
        let document = self.dataset.get(index);

        -1.0 * dot_product_dense_sparse(&self.dense_query, &document)
    }

    #[inline]
    fn topk_retrieval<I, H>(&self, distances: I, heap: &mut H) -> Vec<(f32, usize)>
    where
        I: Iterator<Item = f32>,
        H: OnlineTopKSelector,
    {
        for distance in distances {
            heap.push(distance);
        }

        heap.topk()
    }
}
