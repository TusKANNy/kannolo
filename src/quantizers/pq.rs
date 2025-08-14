use crate::clustering::KMeansBuilder;
use crate::datasets::dense_dataset::DenseDataset;
use crate::quantizers::encoder::{Encoder, PQEncoder8};
use crate::quantizers::quantizer::{Quantizer, QueryEvaluator};
#[cfg(target_arch = "x86_64")]
use crate::simd_distances::{
    compute_distance_table_avx2_d2, compute_distance_table_avx2_d4, compute_distance_table_ip_d4,
    compute_distance_table_ip_d8,
};

use crate::simd_distances::find_nearest_centroid_idx;
use crate::topk_selectors::OnlineTopKSelector;
use crate::utils::{compute_vector_norm_squared, sgemm, MatrixLayout};
use crate::{euclidean_distance_simd, Dataset, DistanceType};
use crate::{Float, PlainDenseDataset};
use itertools::izip;
use rayon::prelude::*;

use crate::{AsRefItem, DenseVector1D, Vector1D};

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

        if self.dsub() < 16 {
            for i in 0..n {
                let mut encoder = PQEncoder8::new(&mut output_vectors[i * M..(i + 1) * M]);

                let query_vec = &input_vectors[i * self.d()..(i + 1) * self.d()];
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
            }
        } else {
            let mut dis_tables = vec![0.0f32; n * self.ksub() * M];
            self.compute_distance_tables(n, input_vectors, &mut dis_tables);

            for i in 0..n {
                let code_slice = &mut output_vectors[i * code_size..];
                let tab_slice = &dis_tables[i * self.ksub() * M..];
                self.compute_code_from_distance_table(tab_slice, code_slice);
            }
        }
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

    #[inline]
    pub fn train(
        training_data: &PlainDenseDataset<f32>,
        nbits: usize,
        distance: DistanceType,
    ) -> Self {
        let d = training_data.dim();
        assert_eq!(M % 4, 0, "M ({}) is not divisible by 4", M);
        assert_eq!(d % M, 0, "d ({}) is not divisible by M ({})", d, M);

        let dsub = d / M;
        let ksub: usize = 2_usize.pow(nbits as u32);

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

    fn train_centroids(
        training_data: &PlainDenseDataset<f32>,
        ksub: usize,
        dsub: usize,
    ) -> Vec<f32> {
        let d = training_data.dim();
        let n_samples = training_data.len();

        println!("Running K-Means for {} subspaces", M);
        let run_kmeans = |i: usize| -> Vec<f32> {
            let mut current_slice = Vec::<f32>::with_capacity(n_samples * dsub);
            for ns in 0..n_samples {
                for j in 0..dsub {
                    current_slice
                        .push(training_data.data().values_as_slice()[ns * d + i * dsub + j]);
                }
            }

            let temp_dataset = PlainDenseDataset::<f32>::from_vec_plain(current_slice, dsub);
            let kmeans = KMeansBuilder::new().build();
            let current_centroids = kmeans.train(&temp_dataset, ksub, None);
            current_centroids.data().values_as_slice().to_vec()
        };

        let centroids = (0..M).into_par_iter().map(|i| run_kmeans(i)).reduce(
            || Vec::new(),
            |mut acc, x| {
                acc.extend_from_slice(&x);
                acc
            },
        );

        centroids
    }

    #[inline]
    fn ksub(&self) -> usize {
        self.ksub
    }

    #[inline]
    fn dsub(&self) -> usize {
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
    fn compute_code_from_distance_table(&self, distance_table: &[f32], code: &mut [u8]) {
        let mut encoder = PQEncoder8::new(code);

        for m in 0..M {
            let mut min_distance = f32::MAX;
            let mut closest_centroid_idx = 0;

            for j in 0..self.ksub() {
                let distance = distance_table[m * self.ksub() + j];
                if distance < min_distance {
                    min_distance = distance;
                    closest_centroid_idx = j;
                }
            }

            encoder.encode(closest_centroid_idx);
        }
    }

    #[inline]
    fn compute_distance_tables(&self, n_queries: usize, query_vec: &[f32], dis_tables: &mut [f32]) {
        for m in 0..M {
            let qvec_slice = &query_vec[m * self.dsub()..];
            let centroids_slice = &self.centroids()[m * self.dsub() * self.ksub()..];
            let dis_tables_slice = &mut dis_tables[m * self.ksub()..];

            if n_queries == 0 || self.ksub() == 0 {
                return;
            }

            self.compute_pairwise_distances(
                n_queries,
                qvec_slice,
                centroids_slice,
                dis_tables_slice,
            );
        }
    }

    #[inline]
    fn compute_pairwise_distances(
        &self,
        n_queries: usize,
        query_vec: &[f32],
        centroids: &[f32],
        distances: &mut [f32],
    ) {
        // Compute the squared L2 norm for each centroid segment and store in the beginning of the distances array.
        for i in 0..self.ksub() {
            distances[i] = compute_vector_norm_squared(
                &centroids[i * self.dsub()..i * self.dsub() + self.dsub()],
                self.dsub(),
            );
        }

        // For each query vector (except the first one), compute the distances to all centroids.
        for i in 1..n_queries {
            let query_norm = compute_vector_norm_squared(&query_vec[i * self.d()..], self.dsub());
            for j in 0..self.ksub() {
                distances[i * (self.ksub() * M) + j] = query_norm + distances[j];
            }
        }

        // Compute the squared L2 norm for the first query vector segment.
        let query_norm = compute_vector_norm_squared(&query_vec, self.dsub());

        for j in 0..self.ksub() {
            distances[j] += query_norm;
        }

        let transpose_a = true;
        let transpose_b = false;

        let alpha = -2.0;
        let beta = 1.0;
        let lda = self.dsub() as isize;
        let ldb = self.d() as isize;
        let ldc = (self.ksub() * M) as isize;

        // Perform matrix multiplication using sgemm function.
        // This operation computes the final pairwise distances between the query vectors and centroids.
        sgemm(
            MatrixLayout::ColMajor,
            transpose_a,
            transpose_b,
            alpha,
            beta,
            self.ksub(),
            self.dsub(),
            n_queries,
            centroids.as_ptr(),
            lda,
            query_vec.as_ptr(),
            ldb,
            distances.as_mut_ptr(),
            ldc,
        );
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
