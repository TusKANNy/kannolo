use crate::distances::dot_product_dense_sparse;
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{Vector1D, DenseVector1D, SparseVector1D};
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

    type DatasetType
        = SparseDataset<Self>;

    type Evaluator
        = SparseQueryEvaluatorPlain<Self::InputItem>
    where
        Self::InputItem: Float + EuclideanDistance<T> + DotProduct<T>;

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

    fn get_space_usage_bytes(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}

pub struct SparseQueryEvaluatorPlain<T: Float> {
    dense_query: DenseVector1D<Vec<T>>,
}

impl<T: Float> QueryEvaluator for SparseQueryEvaluatorPlain<T> {
    type Q = SparsePlainQuantizer<T>;
    type QueryType = SparseVector1D<Vec<u16>, Vec<T>>;

    #[inline]
    fn new(query: Self::QueryType) -> Self {
        let mut dense_query = vec![T::zero(); query.d as usize];
        for (&i, &v) in query
            .components_as_slice()
            .iter()
            .zip(query.values_as_slice())
        {
            dense_query[i as usize] = v;
        }

        let dense_query = DenseVector1D::new(dense_query);

        Self {
            dense_query,
        }
    }

    #[inline]
    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32 {
        let document = dataset.get(index);

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
