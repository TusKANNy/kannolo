use crate::datasets::dataset::{Dataset, GrowableDataset};
use crate::distances::multivector_maxsim;
use crate::quantizer::{Quantizer, QueryEvaluator};
use crate::topk_selectors::OnlineTopKSelector;
use crate::{DistanceType, Float, MultiVector, VectorType};
use crate::{DotProduct, EuclideanDistance};

use serde::{Deserialize, Serialize};

#[derive(Default, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorDataset<Q, B = Vec<<Q as Quantizer>::OutputItem>>
where
    Q: Quantizer,
{
    data: B,
    offsets: Vec<usize>,       // Stores the starting position of each multivector
    vector_counts: Vec<usize>, // Number of vectors in each multivector
    n_multivecs: usize,
    vector_dim: usize, // Fixed dimension of each individual vector
    quantizer: Q,
}

impl<Q, B> Dataset<Q> for MultiVectorDataset<Q, B>
where
    Q: Quantizer<DatasetType = Self>,
    B: AsRef<[Q::OutputItem]> + Default,
{
    type DataType<'a>
        = MultiVector<&'a [Q::OutputItem]>
    where
        Q: 'a,
        B: 'a,
        Q::OutputItem: 'a;

    #[inline]
    fn new(quantizer: Q, vector_dim: usize) -> Self {
        Self {
            data: B::default(),
            offsets: vec![0], // Start with offset 0
            vector_counts: Vec::new(),
            n_multivecs: 0,
            vector_dim,
            quantizer,
        }
    }

    #[inline]
    fn quantizer(&self) -> &Q {
        &self.quantizer
    }

    #[inline]
    fn shape(&self) -> (usize, usize) {
        // For multivectors, the "dimension" is more complex
        // Return (num_multivectors, vector_dim)
        (self.n_multivecs, self.vector_dim)
    }

    #[inline]
    fn data<'a>(&'a self) -> Self::DataType<'a> {
        // Return a view of all data as one large multivector
        let total_vectors: usize = self.vector_counts.iter().sum();
        MultiVector::new(self.data.as_ref(), total_vectors, self.vector_dim)
    }

    #[inline]
    fn dim(&self) -> usize {
        self.vector_dim
    }

    #[inline]
    fn len(&self) -> usize {
        self.n_multivecs
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.data.as_ref().len()
    }

    fn get_space_usage_bytes(&self) -> usize {
        let data_size = std::mem::size_of_val(self.data.as_ref());
        let offsets_size = self.offsets.len() * std::mem::size_of::<usize>();
        let counts_size = self.vector_counts.len() * std::mem::size_of::<usize>();
        data_size + offsets_size + counts_size + self.quantizer.get_space_usage_bytes()
    }

    #[inline]
    fn get<'a>(&'a self, index: usize) -> Self::DataType<'a> {
        assert!(index < self.len(), "Index out of bounds.");

        let start = self.offsets[index];
        let end = self.offsets[index + 1];
        let num_vectors = self.vector_counts[index];
        let quantized_vector_dim = self.quantizer.m(); // Use quantized dimension

        MultiVector::new(
            &self.data.as_ref()[start..end],
            num_vectors,
            quantized_vector_dim,
        )
    }

    fn compute_distance_by_id(&self, idx1: usize, idx2: usize) -> f32
    where
        Q::OutputItem: Float,
        <Q as Quantizer>::OutputItem: DotProduct<<Q as Quantizer>::OutputItem>,
    {
        let multivec1 = self.get(idx1);
        let multivec2 = self.get(idx2);

        match self.quantizer().distance() {
            DistanceType::Euclidean => {
                panic!("Euclidean distance is not typically used for multivectors")
            }
            DistanceType::DotProduct => {
                // Implement MaxSim distance
                multivector_maxsim(&multivec1, &multivec2)
            }
        }
    }

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = Self::DataType<'a>>
    where
        Q::OutputItem: 'a,
    {
        (0..self.len()).map(move |i| self.get(i))
    }

    #[inline]
    fn search<'a, H: OnlineTopKSelector>(
        &self,
        query: <Q::Evaluator<'a> as QueryEvaluator<'a>>::QueryType,
        heap: &mut H,
    ) -> Vec<(f32, usize)>
    where
        Q::InputItem: Float + EuclideanDistance<Q::InputItem> + DotProduct<Q::InputItem>,
    {
        if self.data().values_as_slice().is_empty() {
            return Vec::new();
        }

        let evaluator = self.query_evaluator(query);
        let distances = evaluator.compute_distances(self, 0..self.len());
        evaluator.topk_retrieval(distances, heap)
    }

    fn sample(&self, _sample_size: usize) -> Self {
        unimplemented!("sample() is only implemented for MultiVectorDataset with Vec buffer")
    }
}

impl<Q> MultiVectorDataset<Q, Vec<Q::OutputItem>>
where
    Q: Quantizer<DatasetType = MultiVectorDataset<Q>> + Clone + Sync,
    Q::InputItem: Sync,
    Q::OutputItem: Copy + Default + Send,
{
    pub fn sample(&self, sample_size: usize) -> Self {
        let dataset_len = self.n_multivecs;
        
        // If sample size >= dataset length, clone the entire dataset
        if sample_size >= dataset_len {
            return Self {
                data: self.data.clone(),
                offsets: self.offsets.clone(),
                vector_counts: self.vector_counts.clone(),
                n_multivecs: self.n_multivecs,
                vector_dim: self.vector_dim,
                quantizer: self.quantizer.clone(),
            };
        }

        // Sample indices uniformly at random
        use rand::seq::index::sample;
        use rand::{SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(525);
        let sampled_indices = sample(&mut rng, dataset_len, sample_size);

        // Extract sampled multivectors
        let mut sampled_data = Vec::new();
        let mut sampled_offsets = vec![0];
        let mut sampled_vector_counts = Vec::new();
        
        for idx in sampled_indices.into_vec() {
            let start = self.offsets[idx];
            let end = self.offsets[idx + 1];
            let num_vecs = self.vector_counts[idx];
            sampled_vector_counts.push(num_vecs);
            
            // Copy the multivector's quantized data
            sampled_data.extend_from_slice(&self.data[start..end]);
            sampled_offsets.push(sampled_data.len());
        }

        Self {
            data: sampled_data,
            offsets: sampled_offsets,
            vector_counts: sampled_vector_counts,
            n_multivecs: sample_size,
            vector_dim: self.vector_dim,
            quantizer: self.quantizer.clone(),
        }
    }

    /// Push a batch of multivectors with parallel encoding.
    /// This is more efficient than calling push repeatedly as it parallelizes the encoding step.
    pub fn push_batch<'a>(&mut self, multivecs: &[MultiVector<&'a [Q::InputItem]>]) {
        use rayon::prelude::*;

        if multivecs.is_empty() {
            return;
        }

        let quantized_size_per_vector = self.quantizer.m();

        // Parallel encoding: encode all multivectors in parallel
        let encoded_batch: Vec<(Vec<Q::OutputItem>, usize)> = multivecs
            .par_iter()
            .map(|multivec| {
                let num_vectors = multivec.num_vectors();
                let total_quantized_size = num_vectors * quantized_size_per_vector;
                let mut quantized_codes = vec![Default::default(); total_quantized_size];
                
                self.quantizer.encode(
                    multivec.values_as_slice(),
                    &mut quantized_codes,
                );
                
                (quantized_codes, num_vectors)
            })
            .collect();

        // Sequential insertion: append to dataset
        for (quantized_codes, num_vectors) in encoded_batch {
            self.push_quantized(&quantized_codes, num_vectors);
        }
    }

    /// Append a batch of raw input vectors, encoding them directly into the dataset.
    /// This avoids intermediate buffer allocation and copying.
    /// 
    /// `input_flat`: Flattened input vectors for the entire batch.
    /// `doc_lens`: Number of vectors for each document in the batch.
    pub fn append_batch_and_encode(&mut self, input_flat: &[Q::InputItem], doc_lens: &[usize]) {
        let n_vectors = input_flat.len() / self.vector_dim;
        let m = self.quantizer.m();
        let total_code_len = n_vectors * m;

        let start_idx = self.data.len();
        self.data.resize(start_idx + total_code_len, Default::default());

        // Encode directly into the tail of data
        self.quantizer.encode(input_flat, &mut self.data[start_idx..]);

        let mut current_offset = start_idx;
        for &len in doc_lens {
            let code_len = len * m;
            self.vector_counts.push(len);
            current_offset += code_len;
            self.offsets.push(current_offset);
            self.n_multivecs += 1;
        }
    }
}

impl<Q> GrowableDataset<Q> for MultiVectorDataset<Q, Vec<Q::OutputItem>>
where
    Q: Quantizer<DatasetType = MultiVectorDataset<Q>>,
    Q::OutputItem: Copy + Default,
{
    type InputDataType<'a>
        = MultiVector<&'a [Q::InputItem]>
    where
        Q::InputItem: 'a;

    #[inline]
    fn push<'a>(&mut self, multivec: &Self::InputDataType<'a>) {
        assert_eq!(
            multivec.vector_dim(),
            self.vector_dim,
            "MultiVector vector dimension must match dataset dimension"
        );

        let old_size = self.data.len();
        // For quantizers: each vector becomes quantizer.m() output elements
        let num_vectors = multivec.num_vectors();
        let quantized_size_per_vector = self.quantizer.m();
        let total_quantized_size = num_vectors * quantized_size_per_vector;
        let new_size = old_size + total_quantized_size;

        self.data.resize(new_size, Default::default());

        self.quantizer.encode(
            multivec.values_as_slice(),
            &mut self.data[old_size..new_size],
        );

        self.vector_counts.push(multivec.num_vectors());
        self.offsets.push(new_size);
        self.n_multivecs += 1;
    }
}

impl<Q, B> MultiVectorDataset<Q, B>
where
    Q: Quantizer<DatasetType = Self>,
    B: AsRef<[Q::OutputItem]> + AsMut<[Q::OutputItem]> + Extend<Q::OutputItem>,
    Q::OutputItem: Copy + Default,
{
    /// Manually set pre-quantized data for a multivector document
    /// This bypasses the encoding step and directly stores quantized codes
    #[inline]
    pub fn push_quantized(&mut self, quantized_codes: &[Q::OutputItem], num_vectors: usize) {
        let old_size = self.data.as_ref().len();
        let quantized_size_per_vector = self.quantizer.m();
        let expected_size = num_vectors * quantized_size_per_vector;

        assert_eq!(
            quantized_codes.len(),
            expected_size,
            "Quantized codes size mismatch: expected {} ({}x{}), got {}",
            expected_size,
            num_vectors,
            quantized_size_per_vector,
            quantized_codes.len()
        );

        // Extend the data vector with the new codes
        self.data.extend(quantized_codes.iter().copied());

        self.vector_counts.push(num_vectors);
        self.offsets.push(old_size + quantized_codes.len());
        self.n_multivecs += 1;
    }
}
