use vectorium::dataset::ScoredVector;
use vectorium::vector_encoder::VectorEncoder;
use vectorium::Dataset;

pub trait Index<D>
where
    D: Dataset,
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

    /// Builds an index from an already-encoded dataset.
    fn build_index(dataset: D, build_params: &Self::BuildParams) -> Self;

    fn search<'q>(
        &'q self,
        query: <D::Encoder as VectorEncoder>::QueryVector<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<<D::Encoder as VectorEncoder>::Distance>>;

}
