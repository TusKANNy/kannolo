use crate::topk_selectors::OnlineTopKSelector;
use crate::{DArray1, Dataset, DistanceType, Float};
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
    type DatasetType<'a>: Dataset<'a, Self>
    where
        Self: 'a;

    type Evaluator<'a>: QueryEvaluator<'a, Q = Self>
    where
        Self::InputItem:
            Float + EuclideanDistance<Self::InputItem> + DotProduct<Self::InputItem> + 'a;

    fn encode(&self, input_vectors: &[Self::InputItem], output_vectors: &mut [Self::OutputItem]);

    fn m(&self) -> usize;

    fn distance(&self) -> DistanceType;
}

pub trait QueryEvaluator<'a> {
    type Q: Quantizer;
    type QueryType: DArray1;

    fn new(dataset: &'a <Self::Q as Quantizer>::DatasetType<'a>, query: Self::QueryType) -> Self;

    fn compute_distance(&self, index: usize) -> f32;

    #[inline]
    fn compute_distances(
        &self,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        indexes
            .into_iter()
            .map(|index| self.compute_distance(index))
    }

    #[inline]
    fn compute_four_distances(
        &self,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        indexes
            .into_iter()
            .map(|index| self.compute_distance(index))
    }

    fn topk_retrieval<I, H>(&self, distances: I, heap: &mut H) -> Vec<(f32, usize)>
    where
        I: Iterator<Item = f32>,
        H: OnlineTopKSelector;
}
