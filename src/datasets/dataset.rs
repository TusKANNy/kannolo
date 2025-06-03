use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{DotProduct, EuclideanDistance};
use crate::{Float, Vector1D};

pub trait Dataset<Q>
where
    Q: Quantizer<DatasetType = Self>,
{
    type DataType<'a>: Vector1D<ValuesType = Q::OutputItem>
    where
        Q::OutputItem: 'a,
        Self: 'a;

    fn new(quantizer: Q, d: usize) -> Self;

    #[inline]
    fn query_evaluator<'a>(
        &self,
        query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
    ) -> Q::Evaluator<'a>
    where
        Q::Evaluator<'a>: QueryEvaluator<'a, Q = Q>,
        Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>,
    {
        <Q::Evaluator<'a>>::new(query, &self)
    }

    fn quantizer(&self) -> &Q;

    fn shape(&self) -> (usize, usize);

    fn dim(&self) -> usize;

    fn len(&self) -> usize;

    fn get_space_usage_bytes(&self) -> usize;

    #[inline]
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn nnz(&self) -> usize;

    fn data<'a>(&'a self) -> Self::DataType<'a>;

    fn get<'a>(&'a self, index: usize) -> Self::DataType<'a>;

    fn compute_distance_by_id(&self, idx1: usize, idx2: usize) -> f32
    where
        Q::OutputItem: Float;

    fn iter<'a>(&'a self) -> impl Iterator<Item = Self::DataType<'a>>
    where
        Q::OutputItem: 'a;

    fn search<'a, H: OnlineTopKSelector>(
        &self,
        query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
        heap: &mut H,
    ) -> Vec<(f32, usize)>
    where
        Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>;
}

pub trait GrowableDataset<Q>: Dataset<Q>
where
    Q: Quantizer<DatasetType = Self>,
{
    type InputDataType<'a>: Vector1D<ValuesType = Q::InputItem>
    where
        Q::InputItem: 'a;
    fn push<'a>(&mut self, vec: &Self::InputDataType<'a>);
}
