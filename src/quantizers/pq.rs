use crate::clustering::KMeansBuilder;
use crate::datasets::dense_dataset::DenseDataset;
use crate::quantizers::encoder::{Encoder, PQEncoder8};
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
#[cfg(target_arch = "x86_64")]
use crate::simd_distances::{
    compute_distance_table_avx2_d2, compute_distance_table_avx2_d4, compute_distance_table_avx2_d8, compute_distance_table_avx2_d16,
    compute_distance_table_ip_d2, compute_distance_table_ip_d4, compute_distance_table_ip_d8, compute_distance_table_ip_d16,
};

use crate::simd_distances::find_nearest_centroid_idx;
use crate::topk_selectors::OnlineTopKSelector;
use crate::utils::{sgemm, MatrixLayout};
use crate::{euclidean_distance_simd, Dataset, DistanceType};
use crate::{Float, PlainDenseDataset};
use itertools::izip;

use crate::{AsRefItem, DenseVector1D, VectorType};

use serde::{Deserialize, Serialize};

const BLOCK_SIZE: usize = 256 * 1024;

/// A struct representing a Product Quantizer, implemented as described in the paper
/// "Product quantization for nearest neighbor search.", Jegou et al.
///
/// A Product Quantizer is a data structure used in quantization and indexing applications.
/// It partitions high-dimensional data into `M` smaller subspaces of size `dsub` and quantizes each subspace
/// separately using a `ksub` centroids. The value of `ksub` can be controlled using `nbits`, as `ksub = 2^nbits`
///
/// # Fields
///
/// - `d`: The data dimension, representing the total number of dimensions in the high-dimensional space.
/// - `ksub`: The number of centroids per subspace, indicating the quantization level for each subspace.
/// - `nbits`: The number of bits used to store the centroids for each subspace.
/// - `dsub`: The subspace dimension, representing the number of dimensions in each subspace.
/// - `centroids`: A vector containing the quantization centroids. It has a shape of M x ksub x dsub
///   or equivalently d x ksub, where M represents the number of subspaces.
///
/// The `ProductQuantizer` struct is typically used for efficient retrieval and search operations
/// in high-dimensional spaces.
///
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProductQuantizer<const M: usize> {
    d: usize,            // data dimension
    ksub: usize,         // number of centroids per subspace
    nbits: usize,        // number of bits to store the ksub centroids per subspace
    dsub: usize,         // subspace dimension (d = M * dsub)
    centroids: Vec<f32>, // tensor of shape m x ksub x dsub (or equivalently d x ksub)
    distance: DistanceType,
}

impl<const M: usize> Quantizer for ProductQuantizer<M> {
    type InputItem = f32;
    type OutputItem = u8;

    type DatasetType = DenseDataset<Self>;

    type Evaluator<'a>
        = QueryEvaluatorPQ<'a, M>
    where
        Self::InputItem: Float;

    fn encode(&self, input_vectors: &[Self::InputItem], output_vectors: &mut [Self::OutputItem]) {
        let n = input_vectors.len() / self.d();
        let code_size = (self.nbits() * M + 7) / 8;

        assert!(
            output_vectors.len() >= M * n,
            "Not enough space allocated for output vector. Required {}, given {}",
            M * n,
            output_vectors.len()
        );

        if n > BLOCK_SIZE {
            for i0 in (0..n).step_by(BLOCK_SIZE) {
                let i1 = std::cmp::min(i0 + BLOCK_SIZE, n);
                let block_input = &input_vectors[i0 * self.d()..i1 * self.d()];
                let block_output = &mut output_vectors[i0 * code_size..i1 * code_size];
                self.encode(block_input, block_output);
            }
            return;
        }

        // Use parallel SIMD-optimized encoding for all dsub values.
        // For dsub=4,8,16 we have specialized SIMD functions.
        // For other values, find_nearest_centroid_general uses SIMD-friendly distance computation.
        use rayon::prelude::*;

        // Parallelize over vectors: each vector encodes to M bytes independently.
        input_vectors
            .par_chunks(self.d())
            .zip(output_vectors.par_chunks_mut(M))
            .for_each(|(query_vec, out_code)| {
                let mut encoder = PQEncoder8::new(out_code);

                for m in 0..M {
                    let qvec_slice = &query_vec[m * self.dsub()..(m + 1) * self.dsub()];
                    let start = m * self.ksub() * self.dsub();
                    let end = (m + 1) * self.ksub() * self.dsub();
                    let centroids_slice = &self.centroids()[start..end];

                    let nearest_centroid_idx = find_nearest_centroid_idx(
                        qvec_slice,
                        centroids_slice,
                        self.dsub(),
                        self.ksub(),
                    );

                    encoder.encode(nearest_centroid_idx);
                }
            });
    }

    #[inline]
    fn m(&self) -> usize {
        M
    }

    #[inline]
    fn distance(&self) -> DistanceType {
        self.distance
    }

    fn get_space_usage_bytes(&self) -> usize {
        4 * std::mem::size_of::<usize>() + self.centroids.len() * std::mem::size_of::<f32>()
    }
}

impl<const M: usize> ProductQuantizer<M> {
    #[inline]
    pub fn from_pretrained(
        d: usize,
        nbits: usize,
        centroids: Vec<f32>,
        distance: DistanceType,
    ) -> Self {
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);
        let dsub = d / M;
        let ksub: usize = 2_usize.pow(nbits as u32);

        assert_eq!(centroids.len(), M * ksub * dsub, "Wrong centroids shape");

        ProductQuantizer {
            d,
            ksub,
            nbits,
            dsub,
            centroids,
            distance,
        }
    }

    /// Train a ProductQuantizer on the full dataset without sampling.
    ///
    /// # Arguments
    /// * `training_data` - Dataset to train on (all vectors will be used)
    /// * `nbits` - Number of bits per subspace (ksub = 2^nbits)
    /// * `distance` - Distance type
    #[inline]
    pub fn train(
        training_data: &PlainDenseDataset<f32>,
        nbits: usize,
        distance: DistanceType,
    ) -> Self {
        let d = training_data.dim();
        let n = training_data.len();
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);

        let dsub = d / M;
        let ksub: usize = 2_usize.pow(nbits as u32);

        println!("Training PQ on all {} vectors", n);
        let centroids = ProductQuantizer::<M>::train_centroids(training_data, ksub, dsub);

        ProductQuantizer {
            d,
            ksub,
            nbits,
            dsub,
            centroids,
            distance,
        }
    }

    /// Train a ProductQuantizer with optional sampling.
    ///
    /// # Arguments
    /// * `training_data` - Dataset to train on
    /// * `nbits` - Number of bits per subspace (ksub = 2^nbits)
    /// * `distance` - Distance type
    /// * `sample_size` - Optional sample size:
    ///   - `None`: compute sample size using formula (min(10^7, N, max(10^6, 2*39*ksub, N/20)))
    ///   - `Some(size)`: use the specified size
    ///
    /// If a sample size is determined, extracts a sample and calls `train()` on it.
    #[inline]
    pub fn train_with_sample_size(
        training_data: &PlainDenseDataset<f32>,
        nbits: usize,
        distance: DistanceType,
        sample_size: Option<usize>,
    ) -> Self {
        let dataset_len = training_data.len();
        let ksub: usize = 2_usize.pow(nbits as u32);

        // Determine the sample size
        let n_samples = match sample_size {
            Some(size) => size.min(dataset_len),
            None => {
                if dataset_len > 1_000_000 {
                    // Auto-sampling logic for large datasets
                    let min_points_per_centroid = 39;
                    let n_iter_for_sampling = 10;
                    let min_by_cluster = 2 * min_points_per_centroid * ksub;
                    let min_by_iter = dataset_len / (2 * n_iter_for_sampling);
                    let candidate = std::cmp::max(std::cmp::max(1_000_000, min_by_cluster), min_by_iter);
                    std::cmp::min(std::cmp::min(10_000_000, dataset_len), candidate)
                } else {
                    dataset_len
                }
            }
        };

        // If we're using all data, just call train directly
        if n_samples >= dataset_len {
            return Self::train(training_data, nbits, distance);
        }

        // Sample the dataset using the Dataset::sample method
        println!("Training PQ on {} sampled vectors (out of {})", n_samples, dataset_len);
        let sampled_dataset = training_data.sample(n_samples);
        Self::train(&sampled_dataset, nbits, distance)
    }

    fn train_centroids(
        training_data: &PlainDenseDataset<f32>,
        ksub: usize,
        dsub: usize,
    ) -> Vec<f32> {
        let d = training_data.dim();
        let dataset_len = training_data.len();

        println!("Running K-Means for {} subspaces", M);
        
        let run_kmeans = |i: usize| -> Vec<f32> {
            // Extract the i-th subspace from ALL vectors
            let mut current_slice = Vec::<f32>::with_capacity(dataset_len * dsub);
            for vec_idx in 0..dataset_len {
                for j in 0..dsub {
                    current_slice
                        .push(training_data.data().values_as_slice()[vec_idx * d + i * dsub + j]);
                }
            }

            let temp_dataset = PlainDenseDataset::<f32>::from_vec_plain(current_slice, dsub);
            // Train k-means on all the data (no sampling)
            let kmeans = KMeansBuilder::new()
                .sample_size(None)
                .build();
            let current_centroids = kmeans.train(&temp_dataset, ksub, None);
            current_centroids.data().values_as_slice().to_vec()
        };

        // Run kmeans for each subspace in parallel, but let the kmeans implementation
        // use its own internal parallelism. We spawn M tasks (one per subspace) and
        // collect their results into a shared vector protected by a Mutex. This avoids
        // constraining the inner parallelism to a fixed per-subspace thread count.
        use std::sync::{Arc, Mutex};

        let results: Arc<Mutex<Vec<Option<Vec<f32>>>>> = Arc::new(Mutex::new(vec![None; M]));

        rayon::scope(|s| {
            for i in 0..M {
                let results = Arc::clone(&results);
                s.spawn(move |_| {
                    let cent = run_kmeans(i);
                    let mut guard = results.lock().unwrap();
                    guard[i] = Some(cent);
                });
            }
        });

        // Flatten results into a single centroids Vec<f32>
        let mut centroids: Vec<f32> = Vec::with_capacity(M * ksub * dsub);
        let guard = Arc::try_unwrap(results)
            .ok()
            .expect("Arc still has multiple owners")
            .into_inner()
            .expect("Mutex poisoned");

        for opt in guard.into_iter() {
            match opt {
                Some(mut v) => centroids.append(&mut v),
                None => panic!("KMeans did not produce centroids for a subspace"),
            }
        }

        println!("K-Means finished");

        centroids
    }

    #[inline]
    pub fn ksub(&self) -> usize {
        self.ksub
    }

    #[inline]
    pub fn dsub(&self) -> usize {
        self.dsub
    }

    #[inline]
    fn nbits(&self) -> usize {
        self.nbits
    }

    #[inline]
    pub fn centroids(&self) -> &Vec<f32> {
        &self.centroids
    }

    #[inline]
    fn d(&self) -> usize {
        self.d
    }

    #[inline]
    pub fn compute_distance(&self, distance_table: &[f32], code: &[u8]) -> f32 {
        assert_eq!(M % 4, 0, "M is not a multiple of 4");
        // Assumes that the distances table has already been computed.
        let mut distance = [0.0; 4];
        let mut pointer = 0;

        for subcode in code.chunks_exact(4) {
            unsafe {
                distance[0] += *distance_table.get_unchecked(pointer + subcode[0] as usize);
                distance[1] +=
                    *distance_table.get_unchecked(pointer + self.ksub() + subcode[1] as usize);
                distance[2] +=
                    *distance_table.get_unchecked(pointer + 2 * self.ksub() + subcode[2] as usize);
                distance[3] +=
                    *distance_table.get_unchecked(pointer + 3 * self.ksub() + subcode[3] as usize);
                pointer += 4 * self.ksub();
            }
        }

        let final_distance = distance[0] + distance[1] + distance[2] + distance[3];
        final_distance
    }

    #[inline]
    fn get_centroids(&self, m: usize) -> &[f32] {
        let index = m * self.ksub() * self.dsub();
        &self.centroids()[index..index + self.ksub() * self.dsub()]
    }

    #[inline]
    fn compute_euclidean_distance_table<T>(&self, query: &DenseVector1D<T>) -> Vec<f32>
    where
        T: AsRefItem<Item = f32>,
    {
        let mut distance_table = vec![0.0_f32; self.ksub() * M];

        for m in 0..M {
            let query_subvector = &query.values_as_slice()[m * self.dsub()..(m + 1) * self.dsub()];
            let centroids = self.get_centroids(m);
            let distance_table_slice = &mut distance_table[m * self.ksub()..(m + 1) * self.ksub()];

            #[cfg(target_arch = "x86_64")]
            {
                match self.dsub() {
                    2 => unsafe {
                        compute_distance_table_avx2_d2(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },

                    4 => unsafe {
                        compute_distance_table_avx2_d4(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    8 => unsafe {
                        compute_distance_table_avx2_d8(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    16 => unsafe {
                        compute_distance_table_avx2_d16(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    _ => {
                        for i in 0..self.ksub() {
                            distance_table_slice[i] = euclidean_distance_simd(
                                query_subvector,
                                &centroids[i * self.dsub()..(i + 1) * self.dsub()],
                            );
                        }
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                for i in 0..self.ksub() {
                    distance_table_slice[i] = euclidean_distance_simd(
                        query_subvector,
                        &centroids[i * self.dsub()..(i + 1) * self.dsub()],
                    );
                }
            }
        }

        distance_table
    }

    #[inline]
    pub fn compute_dot_product_table<T>(&self, query: &DenseVector1D<T>) -> Vec<f32>
    where
        T: AsRefItem<Item = f32>,
    {
        let mut dot_product_table = vec![0.0_f32; self.ksub() * M];

        for m in 0..M {
            let query_subvector = &query.values_as_slice()[m * self.dsub()..(m + 1) * self.dsub()];
            let centroids = self.get_centroids(m);
            let distance_table_slice =
                &mut dot_product_table[m * self.ksub()..(m + 1) * self.ksub()];

            #[cfg(target_arch = "x86_64")]
            {
                match self.dsub() {
                    2 => unsafe {
                        compute_distance_table_ip_d2(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    4 => unsafe {
                        compute_distance_table_ip_d4(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    8 => unsafe {
                        compute_distance_table_ip_d8(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    16 => unsafe {
                        compute_distance_table_ip_d16(
                            distance_table_slice,
                            query_subvector,
                            centroids,
                            self.ksub(),
                        )
                    },
                    _ => {
                        let alpha = 1.0;
                        let beta = 0.0;
                        let m = 1;
                        let k = self.dsub;
                        let n = self.ksub;

                        for (x_subspace, centroids_subspace, dot_product) in izip!(
                            query.values_as_slice().chunks_exact(self.dsub()),
                            self.centroids().chunks_exact(self.ksub() * self.dsub()),
                            dot_product_table.chunks_exact_mut(self.ksub())
                        ) {
                            sgemm(
                                MatrixLayout::RowMajor,
                                false,
                                true,
                                alpha,
                                beta,
                                m,
                                k,
                                n,
                                x_subspace.as_ptr(),
                                k as isize,
                                centroids_subspace.as_ptr(),
                                k as isize,
                                dot_product.as_mut_ptr(),
                                n as isize,
                            );
                        }
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                let alpha = 1.0;
                let beta = 0.0;
                let m = 1;
                let k = self.dsub;
                let n = self.ksub;

                for (x_subspace, centroids_subspace, dot_product) in izip!(
                    query.values_as_slice().chunks_exact(self.dsub()),
                    self.centroids().chunks_exact(self.ksub() * self.dsub()),
                    dot_product_table.chunks_exact_mut(self.ksub())
                ) {
                    sgemm(
                        MatrixLayout::RowMajor,
                        false,
                        true,
                        alpha,
                        beta,
                        m,
                        k,
                        n,
                        x_subspace.as_ptr(),
                        k as isize,
                        centroids_subspace.as_ptr(),
                        k as isize,
                        dot_product.as_mut_ptr(),
                        n as isize,
                    );
                }
            }
        }

        dot_product_table
    }
}

pub struct QueryEvaluatorPQ<'a, const M: usize> {
    _query: <Self as QueryEvaluator<'a>>::QueryType,
    distance_table: Vec<f32>,
}

impl<'a, const M: usize> QueryEvaluator<'a> for QueryEvaluatorPQ<'a, M> {
    type Q = ProductQuantizer<M>;
    type QueryType = DenseVector1D<&'a [f32]>;

    #[inline]
    fn new(query: Self::QueryType, dataset: &<Self::Q as Quantizer>::DatasetType) -> Self {
        let distance_table = match dataset.quantizer().distance() {
            DistanceType::Euclidean => dataset.quantizer().compute_euclidean_distance_table(&query),
            DistanceType::DotProduct => dataset.quantizer().compute_dot_product_table(&query),
        };

        Self {
            _query: query,
            distance_table: distance_table,
        }
    }

    #[inline]
    fn compute_distance(&self, dataset: &<Self::Q as Quantizer>::DatasetType, index: usize) -> f32 {
        let code = dataset.get(index);

        let distance = dataset
            .quantizer()
            .compute_distance(&self.distance_table, code.values_as_slice());

        match dataset.quantizer().distance() {
            DistanceType::DotProduct => -distance,
            _ => distance,
        }
    }

    #[inline]
    fn compute_distances(
        &self,
        dataset: &<Self::Q as Quantizer>::DatasetType,
        indexes: impl IntoIterator<Item = usize>,
    ) -> impl Iterator<Item = f32> {
        let codes: Vec<_> = indexes.into_iter().map(|id| dataset.get(id)).collect();

        let mut accs = vec![0.0; codes.len()];

        for (j, four_codes) in codes.chunks_exact(4).enumerate() {
            let code1 = four_codes[0].values_as_slice();
            let code2 = four_codes[1].values_as_slice();
            let code3 = four_codes[2].values_as_slice();
            let code4 = four_codes[3].values_as_slice();
            let mut pointer = 0;

            for i in 0..M {
                unsafe {
                    accs[4 * j] += self
                        .distance_table
                        .get_unchecked(pointer + *code1.get_unchecked(i) as usize);
                    accs[4 * j + 1] += self
                        .distance_table
                        .get_unchecked(pointer + *code2.get_unchecked(i) as usize);
                    accs[4 * j + 2] += self
                        .distance_table
                        .get_unchecked(pointer + *code3.get_unchecked(i) as usize);
                    accs[4 * j + 3] += self
                        .distance_table
                        .get_unchecked(pointer + *code4.get_unchecked(i) as usize);
                }
                pointer += dataset.quantizer().ksub();
            }
        }

        let reminder = codes.len() % 4;
        let n_processed = codes.len() - reminder;

        for (j, code) in codes.iter().skip(n_processed).enumerate() {
            let mut pointer = 0;
            for i in 0..M {
                unsafe {
                    accs[n_processed + j] += self
                        .distance_table
                        .get_unchecked(pointer + *code.values_as_slice().get_unchecked(i) as usize);
                }
                pointer += dataset.quantizer().ksub();
            }
        }

        if dataset.quantizer().distance() == DistanceType::DotProduct {
            accs.iter_mut().for_each(|d| *d = -*d);
        }

        accs.into_iter()
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
