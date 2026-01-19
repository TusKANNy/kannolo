use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{DotProduct, EuclideanDistance};
use crate::{Float, VectorType};

pub trait Dataset<Q>
where
    Q: Quantizer<DatasetType = Self>,
{
    type DataType<'a>: VectorType<ValuesType = Q::OutputItem>
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
        <Q::Evaluator<'a>>::new(query, self)
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
        Q::OutputItem: Float,
        <Q as Quantizer>::OutputItem: DotProduct<<Q as Quantizer>::OutputItem>;

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

    /// Sample a subset of vectors from the dataset uniformly at random.
    /// Returns a new dataset containing the sampled vectors.
    /// If `sample_size` >= dataset length, returns all vectors.
    fn sample(&self, sample_size: usize) -> Self
    where
        Self: Sized;
}

pub trait GrowableDataset<Q>: Dataset<Q>
where
    Q: Quantizer<DatasetType = Self>,
{
    type InputDataType<'a>: VectorType<ValuesType = Q::InputItem>
    where
        Q::InputItem: 'a;

    fn push<'a>(&mut self, vec: &Self::InputDataType<'a>);
}

// Implement the `Index` trait for any `Dataset` so that datasets can be used as
// flat indices.
impl<D, Q> crate::index::Index<D, Q> for D
where
    D: Dataset<Q> + Sync,
    Q: Quantizer<InputItem: Float, DatasetType = D> + Sync,
{
    type BuildParams = ();
    type SearchParams = ();

    #[inline]
    fn n_vectors(&self) -> usize {
        self.len()
    }

    #[inline]
    fn dim(&self) -> usize {
        self.dim()
    }

    fn print_space_usage_bytes(&self) {
        let dataset_size = self.get_space_usage_bytes();
        println!("[######] Space usage: Dataset: {dataset_size} bytes");
    }

    fn build_index<'a, BD, IQ>(
        source_dataset: &'a BD,
        quantizer: Q,
        _build_params: &Self::BuildParams,
    ) -> Self
    where
        BD: Dataset<IQ> + Sync + 'a,
        IQ: crate::quantizer::IdentityQuantizer<DatasetType = BD, T: Float> + Sync + 'a,
        <IQ as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <BD as Dataset<IQ>>::DataType<'a>>,
        D: GrowableDataset<Q, InputDataType<'a> = <BD as Dataset<IQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: 'a,
    {
        // By default, just create a new dataset from the source dataset and quantizer.
        let mut dataset = D::new(quantizer, source_dataset.dim());
        for id in 0..source_dataset.len() {
            // Encode and add each vector to the final dataset.
            dataset.push(&source_dataset.get(id));
        }
        dataset
    }

    fn search<'a, QD, QQ>(
        &'a self,
        query: QD::DataType<'a>,
        k: usize,
        _search_params: &Self::SearchParams,
    ) -> Vec<(f32, usize)>
    where
        QD: Dataset<QQ> + Sync + 'a,
        QQ: Quantizer<DatasetType = QD> + Sync + 'a,
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <QD as Dataset<QQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>,
        <Q as Quantizer>::InputItem: 'a,
    {
        let mut heap = crate::topk_selectors::TopkHeap::new(k);
        Dataset::search(self, query, &mut heap)
    }
}
