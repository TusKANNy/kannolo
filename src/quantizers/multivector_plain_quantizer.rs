use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{multivector_maxsim_query_batch_4, Dataset, DistanceType, Float, MultiVector};
use crate::{DotProduct, EuclideanDistance};

use crate::datasets::multivector_dataset::MultiVectorDataset;

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiVectorPlainQuantizer<T> {
    vector_dim: usize,
    distance: DistanceType,
    _phantom: PhantomData<T>,
}

impl<T> MultiVectorPlainQuantizer<T> {
    #[inline]
    pub fn new(vector_dim: usize, distance: DistanceType) -> Self {
        MultiVectorPlainQuantizer {
            vector_dim,
            distance,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + Default + PartialOrd + Sync + Send> Quantizer for MultiVectorPlainQuantizer<T> {
    type InputItem = T;
    type OutputItem = T;
    type DatasetType = MultiVectorDataset<Self>;

    type Evaluator<'a>
        = MultiVectorQueryEvaluatorPlain<'a, Self::InputItem>
    where
        Self::InputItem: Float + EuclideanDistance<T> + DotProduct<T> + 'a;

    #[inline]
    fn encode(&self, input_vectors: &[Self::InputItem], output_vectors: &mut [Self::OutputItem]) {
        output_vectors.copy_from_slice(input_vectors);
    }

    #[inline]
    fn m(&self) -> usize {
        self.vector_dim
    }

    #[inline]
    fn distance(&self) -> DistanceType {
        self.distance
    }

    fn get_space_usage_bytes(&self) -> usize {
        std::mem::size_of::<usize>()
    }
}

pub struct MultiVectorQueryEvaluatorPlain<'a, T: Float> {
    query: MultiVector<&'a [T]>,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Float + DotProduct<T>> QueryEvaluator<'a> for MultiVectorQueryEvaluatorPlain<'a, T> {
    type Q = MultiVectorPlainQuantizer<T>;
    type QueryType = MultiVector<&'a [T]>;

    #[inline]
    fn new(query: Self::QueryType, _dataset: &<Self::Q as Quantizer>::DatasetType) -> Self {
        Self {
            query,
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32 {
        let document = dataset.get(index);

        match dataset.quantizer().distance() {
            DistanceType::Euclidean => {
                panic!("Euclidean distance not supported for multivectors")
            }
            DistanceType::DotProduct => {
                // Return negative MaxSim for min-heap behavior
                -multivector_maxsim_query_batch_4(&self.query, &document)
            }
        }
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
