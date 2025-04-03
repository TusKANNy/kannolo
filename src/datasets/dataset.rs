use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{DArray1, Float};
use crate::{DotProduct, EuclideanDistance};

pub trait Dataset<'a, Q>
where
    Q: Quantizer<DatasetType<'a> = Self> + 'a,
{
    type DataType: DArray1<ValuesType = Q::OutputItem>;

    fn new(quantizer: Q, d: usize) -> Self;

    #[inline]
    fn query_evaluator(
        &'a self,
        query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
    ) -> Q::Evaluator<'a>
    where
        Q::Evaluator<'a>: QueryEvaluator<'a, Q = Q>,
        Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>,
    {
        <Q::Evaluator<'a>>::new(self, query)
    }

    fn quantizer(&self) -> &Q;

    fn shape(&self) -> (usize, usize);

    fn dim(&self) -> usize;

    fn len(&self) -> usize;

    #[inline]
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn nnz(&self) -> usize;

    fn data(&'a self) -> Self::DataType;

    fn get(&'a self, index: usize) -> Self::DataType;

    fn compute_distance_by_id(&'a self, idx1: usize, idx2: usize) -> f32
    where
        Q::OutputItem: Float;

    fn iter(&'a self) -> impl Iterator<Item = Self::DataType>;

    fn search<H: OnlineTopKSelector>(
        &'a self,
        query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
        heap: &mut H,
    ) -> Vec<(f32, usize)>
    where
        Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>;
}

pub trait GrowableDataset<'a, Q>: Dataset<'a, Q>
where
    Q: Quantizer<DatasetType<'a> = Self> + 'a,
{
    type InputDataType: DArray1<ValuesType = Q::InputItem>;
    fn push(&mut self, vec: &Self::InputDataType);
}
