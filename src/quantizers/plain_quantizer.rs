use crate::distances::{self, dense_dot_product_unrolled, dense_euclidean_distance_unrolled};
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{dot_product_batch_4, euclidean_distance_batch_4, DArray1, DenseDArray1};
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

    type DatasetType<'a>
        = DenseDataset<Self>
    where
        T: 'a;
    type Evaluator<'a>
        = QueryEvaluatorPlain<'a, Self::InputItem>
    where
        Self::InputItem: Float
            + 'a
            + distances::euclidean_distance::EuclideanDistance<T>
            + distances::dot_product::DotProduct<T>;

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

pub struct QueryEvaluatorPlain<
    'a,
    T: Float
        + 'a
        + distances::euclidean_distance::EuclideanDistance<T>
        + distances::dot_product::DotProduct<T>,
> {
    dataset: &'a <<Self as QueryEvaluator<'a>>::Q as Quantizer>::DatasetType<'a>,
    query: <Self as QueryEvaluator<'a>>::QueryType,
}

impl<'a, T: Float> QueryEvaluator<'a> for QueryEvaluatorPlain<'a, T>
where
    T: Float
        + distances::euclidean_distance::EuclideanDistance<T>
        + distances::dot_product::DotProduct<T>,
{
    type Q = PlainQuantizer<T>;
    type QueryType = DenseDArray1<&'a [T]>;

    #[inline]
    fn new(dataset: &'a <Self::Q as Quantizer>::DatasetType<'a>, query: Self::QueryType) -> Self {
        Self { dataset, query }
    }

    fn compute_distance(&self, index: usize) -> f32 {
        let document = self.dataset.get(index);
        match self.dataset.quantizer().distance() {
            DistanceType::Euclidean => dense_euclidean_distance_unrolled(&self.query, &document),
            DistanceType::DotProduct => -dense_dot_product_unrolled(&self.query, &document),
        }
    }

    #[inline]
    fn compute_four_distances(
        &self,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        let chunk: Vec<_> = indexes.into_iter().map(|id| self.dataset.get(id)).collect();
        let query_slice = self.query.values_as_slice();
        let quantizer = self.dataset.quantizer();

        // Process exactly 4 vectors
        let v0 = chunk[0].values_as_slice();
        let v1 = chunk[1].values_as_slice();
        let v2 = chunk[2].values_as_slice();
        let v3 = chunk[3].values_as_slice();
        let vector_batch = [&v0[..], &v1[..], &v2[..], &v3[..]];

        let dist = match quantizer.distance() {
            DistanceType::Euclidean => euclidean_distance_batch_4(query_slice, vector_batch),
            DistanceType::DotProduct => {
                let dps = dot_product_batch_4(query_slice, vector_batch); // Negate distances
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
