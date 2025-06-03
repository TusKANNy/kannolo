use crate::topk_selectors::OnlineTopKSelector;
use crate::{Dataset, DistanceType, Float, Vector1D};
use crate::{DotProduct, EuclideanDistance};

pub trait IdentityQuantizer: Quantizer<InputItem = Self::T, OutputItem = Self::T> {
    type T: Float + EuclideanDistance<Self::T> + DotProduct<Self::T>;
}

impl<T, Q> IdentityQuantizer for Q
where
    Q: Quantizer<InputItem = T, OutputItem = T>,
    T: Float + EuclideanDistance<T> + DotProduct<T>,
{
    type T = T;
}

pub trait Quantizer: Sized {
    type InputItem;
    type OutputItem;
    type DatasetType: Dataset<Self>;

    type Evaluator<'a>: QueryEvaluator<'a, Q = Self>
    where
        Self::InputItem:
            Float + EuclideanDistance<Self::InputItem> + DotProduct<Self::InputItem> + 'a;

    fn encode(&self, input_vectors: &[Self::InputItem], output_vectors: &mut [Self::OutputItem]);

    fn m(&self) -> usize;

    fn distance(&self) -> DistanceType;

    fn get_space_usage_bytes(&self) -> usize;
}

pub trait QueryEvaluator<'a> {
    type Q: Quantizer;
    type QueryType: Vector1D;

    fn new(query: Self::QueryType, dataset: &<Self::Q as Quantizer>::DatasetType) -> Self;

    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32;

    #[inline]
    fn compute_distances(
        &self,
        dataset: &<Self::Q as Quantizer>::DatasetType,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        indexes
            .into_iter()
            .map(|index| self.compute_distance(dataset, index))
    }

    #[inline]
    fn compute_four_distances(
        &self,
        dataset: &<Self::Q as Quantizer>::DatasetType,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        indexes
            .into_iter()
            .map(|index| self.compute_distance(dataset, index))
    }

    fn topk_retrieval<I, H>(&self, distances: I, heap: &mut H) -> Vec<(f32, usize)>
    where
        I: Iterator<Item = f32>,
        H: OnlineTopKSelector;
}
