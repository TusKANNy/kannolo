use crate::distances;
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{
    dot_product_batch_4_simd, dot_product_simd, euclidean_distance_batch_4_simd,
    euclidean_distance_simd, DenseVector1D, Vector1D,
};
use crate::{Dataset, DistanceType, Float};

use crate::datasets::dense_dataset::DenseDataset;

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlainQuantizer<T> {
    d: usize,
    distance: DistanceType,
    _phantom: PhantomData<T>,
}

impl<T> PlainQuantizer<T> {
    pub fn new(d: usize, distance: DistanceType) -> Self {
        PlainQuantizer {
            d,
            distance,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy + Default + PartialOrd + Sync + Send> Quantizer for PlainQuantizer<T> {
    type InputItem = T;
    type OutputItem = T;

    type DatasetType = DenseDataset<Self>;

    type Evaluator<'a>
        = QueryEvaluatorPlain<'a, Self::InputItem>
    where
        Self::InputItem: Float
            + distances::euclidean_distance::EuclideanDistance<T>
            + distances::dot_product::DotProduct<T>
            + 'a;

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

pub struct QueryEvaluatorPlain<
    'a,
    T: Float
        + distances::euclidean_distance::EuclideanDistance<T>
        + distances::dot_product::DotProduct<T>
        + 'a,
> {
    query: <Self as QueryEvaluator<'a>>::QueryType,
}

impl<'a, T: Float> QueryEvaluator<'a> for QueryEvaluatorPlain<'a, T>
where
    T: Float
        + distances::euclidean_distance::EuclideanDistance<T>
        + distances::dot_product::DotProduct<T>,
{
    type Q = PlainQuantizer<T>;
    type QueryType = DenseVector1D<&'a [T]>;

    #[inline]
    fn new(query: Self::QueryType, _dataset: &<Self::Q as Quantizer>::DatasetType) -> Self {
        Self { query }
    }

    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32 {
        let document_slice = dataset.get(index);
        let document_slice = document_slice.values_as_slice();
        let query_slice = self.query.values_as_slice();
        match dataset.quantizer().distance() {
            DistanceType::Euclidean => euclidean_distance_simd(query_slice, document_slice),
            DistanceType::DotProduct => -dot_product_simd(query_slice, document_slice),
        }
    }

    #[inline]
    fn compute_four_distances(
        &self,
        dataset: &<Self::Q as Quantizer>::DatasetType,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        let chunk: Vec<_> = indexes.into_iter().map(|id| dataset.get(id)).collect();
        let query_slice = self.query.values_as_slice();
        let quantizer = dataset.quantizer();

        // Process exactly 4 vectors
        let v0 = chunk[0].values_as_slice();
        let v1 = chunk[1].values_as_slice();
        let v2 = chunk[2].values_as_slice();
        let v3 = chunk[3].values_as_slice();
        let vector_batch = [&v0[..], &v1[..], &v2[..], &v3[..]];

        let dist = match quantizer.distance() {
            DistanceType::Euclidean => euclidean_distance_batch_4_simd(query_slice, vector_batch),
            DistanceType::DotProduct => {
                let dps = dot_product_batch_4_simd(query_slice, vector_batch); // Negate distances
                [-dps[0], -dps[1], -dps[2], -dps[3]]
            }
        };

        dist.into_iter()
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
