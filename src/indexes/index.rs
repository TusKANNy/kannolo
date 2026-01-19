use crate::quantizer::{IdentityQuantizer, Quantizer, QueryEvaluator};
use crate::{Dataset, DotProduct, EuclideanDistance, Float, GrowableDataset};

pub trait Index<D, Q>
where
    D: Dataset<Q>,
    Q: Quantizer<DatasetType = D>,
    Q: Quantizer<InputItem: Float, DatasetType = D> + Sync,
{
    type BuildParams; // Type for graph build parameters
    type SearchParams; // Type for search parameters

    /// Returns the number of vectors in the graph index.
    fn n_vectors(&self) -> usize;

    /// Returns the dimensionality of the vectors in the graph index.
    fn dim(&self) -> usize;

    /// Prints the space usage of the graph index in bytes,
    /// including the dataset and the graph structure.
    fn print_space_usage_bytes(&self);

    /// Builds an index from a dataset.
    ///
    /// For graph-based indexes this will perform the graph build. For datasets
    /// acting as indices (flat indices), this can simply return the dataset
    /// that was passed as `dataset_for_search`.
    fn build_index<'a, BD, IQ>(
        dataset: &'a BD,
        quantizer: Q,
        build_params: &Self::BuildParams,
    ) -> Self
    where
        BD: Dataset<IQ> + Sync + 'a,
        IQ: IdentityQuantizer<DatasetType = BD, T: Float> + Sync + 'a,
        // This constraint is necessary because the vector returned by the dataset's get function is of type Datatype.
        // The query evaluator, however, requires a vector of type Querytype.
        <IQ as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <BD as Dataset<IQ>>::DataType<'a>>,
        D: GrowableDataset<Q, InputDataType<'a> = <BD as Dataset<IQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: 'a;

    fn search<'a, QD, QQ>(
        &'a self,
        query: QD::DataType<'a>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<(f32, usize)>
    where
        // The query dataset type (QD) could be directly of type D, but this would not work if D is a Dataset
        // with a ProductQuantizer, this because queries is a dataset with a PlainQuantizer.
        QD: Dataset<QQ> + Sync + 'a,
        QQ: Quantizer<DatasetType = QD> + Sync + 'a,
        // This constraint is necessary because the find_k_nearest_neighbors function takes an input parameter
        // of type QueryType, which is an associated type of the QueryEvaluator associated with the quantizer Q.
        // However, the queries are of type DataType, which is an associated type of the dataset QD.
        <Q as Quantizer>::Evaluator<'a>:
            QueryEvaluator<'a, QueryType = <QD as Dataset<QQ>>::DataType<'a>>,
        <Q as Quantizer>::InputItem: EuclideanDistance<<Q as Quantizer>::InputItem>
            + DotProduct<<Q as Quantizer>::InputItem>,
        <Q as Quantizer>::InputItem: 'a;

}
