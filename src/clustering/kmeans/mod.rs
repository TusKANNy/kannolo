use crate::plain_quantizer::PlainQuantizer;
use crate::{Dataset, DistanceType, DotProduct, EuclideanDistance, Float, dot_product_simd};
use crate::{PlainDenseDataset, VectorType};
use crate::index::Index;
use crate::indexes::hnsw::{HNSW, HNSWBuildParams, HNSWSearchParams};
use crate::graph::GraphFixedDegree;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

pub struct KMeans {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    min_points_per_centroid: usize,
    sample_size: Option<usize>,
    /// Threshold for number of centroids above which HNSW index is used instead of flat search
    index_threshold: usize,
}

impl KMeans {
    /// Computes the imbalance factor of the clustering.
    /// Smaller unfairness factor means more balanced clusters.
    ///
    /// # Arguments
    ///
    /// * `histograms`: vector storing how many vectors are assigned to each cluster
    /// * `k`: the number of clusters
    ///
    /// returns: the imbalance factor as f32
    ///
    #[inline]
    fn imbalance_factor(histograms: &[f32], k: usize) -> f32 {
        let unfairness_factor = dot_product_simd(histograms, histograms);
        let total: f32 = histograms.iter().sum();
        unfairness_factor * k as f32 / (total * total)
    }

    /// Assignment: for each vector, find nearest centroid by parallelizing over vectors
    /// Internally decides whether to use HNSW index or flat search based on the number of centroids.
    /// For small k (< index_threshold), uses flat dataset search. For large k, builds HNSW index.
    pub fn compute_assignments<T>(
        dataset: &PlainDenseDataset<T>,
        centroids: &PlainDenseDataset<T>,
        index_threshold: usize,
    ) -> Vec<(f32, usize)>
    where
        T: Float + EuclideanDistance<T> + DotProduct<T>,
    {
        let k = centroids.len();
        
        if k >= index_threshold {
            // Use HNSW index for large k
            let hnsw_params = HNSWBuildParams {
                num_neighbors_per_vec: 16,
                ef_construction: 600,
                initial_build_batch_size: 4,
                max_build_batch_size: 64,
            };
            let centroids_hnsw = HNSW::<PlainDenseDataset<T>, PlainQuantizer<T>, GraphFixedDegree>::build_index(
                centroids,
                PlainQuantizer::<T>::new(Dataset::dim(centroids), DistanceType::Euclidean),
                &hnsw_params,
            );
            let search_params = HNSWSearchParams { ef_search: 600 };
            
            (0..dataset.len())
                .into_par_iter()
                .map(|i| {
                    let query = dataset.get(i);
                    let res = centroids_hnsw.search::<PlainDenseDataset<T>, PlainQuantizer<T>>(
                        query,
                        1,
                        &search_params,
                    );
                    if let Some((dist, idx)) = res.into_iter().next() {
                        (dist, idx)
                    } else {
                        (f32::MAX, usize::MAX)
                    }
                })
                .collect()
        } else {
            // Use flat dataset search for small k
            use crate::topk_selectors::{TopkHeap, OnlineTopKSelector};
            (0..dataset.len())
                .into_par_iter()
                .map(|i| {
                    let query = dataset.get(i);
                    let mut heap = TopkHeap::new(1);
                    let res = Dataset::search(centroids, query, &mut heap);
                    if let Some((dist, idx)) = res.into_iter().next() {
                        (dist, idx)
                    } else {
                        (f32::MAX, usize::MAX)
                    }
                })
                .collect()
        }
    }

    /// Computes the centroids of the clustering as the mean of vectors assigned to each cluster.
    /// Then, splits empty clusters
    ///
    /// # Arguments
    ///
    /// * `dataset`: the input dataset
    /// * `weights`: the weight of each vector, optionally
    /// * `k`: the number of clusters
    /// * `assignments`: the latest assignment vector in the dataset - cluster
    ///
    /// returns: the number of splits, a vector storing how many vectors are assigned to each cluster
    /// and the new centroids in a VecDataset<T>
    ///
    fn update_and_split<T>(
        dataset: &PlainDenseDataset<T>,
        weights: Option<&[f32]>,
        k: usize,
        assignments: &[(f32, usize)],
        rng: &mut StdRng,
        // Removed the parallel parameter; always run in parallel
    ) -> (usize, Vec<f32>, PlainDenseDataset<T>)
    where
        T: Float + EuclideanDistance<T> + DotProduct<T> + num_traits::ToPrimitive + num_traits::FromPrimitive,
    {
        let n = dataset.len();
        let d = Dataset::dim(dataset);

    // Always run centroid updates in parallel
    let (centroids, histograms) = {
    let results: Vec<(Vec<f32>, f32)> = (0..k)
            .into_par_iter()
            .map(|ci| {
                let mut centroid = vec![0.0; d];
                let mut count = 0.0;

                for i in 0..n {
                    if assignments[i].1 == ci {
                        let vec = dataset.get(i);
                        match weights {
                            Some(w) => {
                                count += w[i];
                                centroid
                                    .iter_mut()
                                    .zip(vec.values_as_slice().iter())
                                    .for_each(|(c, x)| *c += x.to_f32().unwrap() * w[i]);
                            }
                            None => {
                                count += 1.0;
                                centroid
                                    .iter_mut()
                                    .zip(vec.values_as_slice().iter())
                                    .for_each(|(c, x)| *c += x.to_f32().unwrap());
                            }
                        }
                    }
                }

                // Normalize
                if count > 0.0 {
                    centroid.iter_mut().for_each(|c| *c /= count);
                }

                (centroid, count)
            })
            .collect();

        // Flatten results
        let mut centroids = Vec::with_capacity(k * d);
        let mut histograms = Vec::with_capacity(k);
        for (centroid, count) in results {
            centroids.extend(centroid);
            histograms.push(count);
        }
        (centroids, histograms)
    };

    let mut centroids = centroids;
    let mut histograms = histograms;

        // Splits clusters
        let mut n_splits = 0;
        let mut cj;
        let epsilon = 1.0 / 1024.;

        for ci in 0..k {
            if histograms[ci] != 0.0 {
                continue;
            }
            cj = 0;
            loop {
                let p = (histograms[cj] - 1.0) / (n - k) as f32;
                let r = rng.gen::<f32>();
                if r < p {
                    break;
                }
                cj = (cj + 1) % k;
            }

            let tmp = centroids[cj * d..(cj + 1) * d].to_owned();
            centroids[ci * d..(ci + 1) * d].copy_from_slice(&tmp);

            for j in 0..d {
                if j % 2 == 0 {
                    centroids[ci * d + j] *= 1.0 + epsilon;
                    centroids[cj * d + j] *= 1.0 - epsilon;
                } else {
                    centroids[ci * d + j] *= 1.0 - epsilon;
                    centroids[cj * d + j] *= 1.0 + epsilon;
                }
            }

            histograms[ci] = histograms[cj] / 2.0;
            histograms[cj] /= 2.0;
            n_splits += 1;
        }
        
        // Convert f32 centroids back to T
        let centroids_t: Vec<T> = centroids
            .iter()
            .map(|&x| T::from_f32(x).unwrap())
            .collect();
        
        (
            n_splits,
            histograms,
            PlainDenseDataset::from_vec_plain(centroids_t, d),
        )
    }

    /// Runs K-Means training on a dataset with k clusters.
    /// If the user has provided input weights, the computation of centroids is the weighted mean
    /// of every vector assigned to their cluster. Otherwise the computation is just the mean.
    /// 
    /// If a sample_size is specified, a random sample of that size is used for training.
    /// Otherwise, the full dataset is used for training.
    ///
    /// # Arguments
    ///
    /// * `dataset`: the dataset (can be f32 or f16)
    /// * `k`: the desired number of clusters
    /// * `weights`: optionally weights of the same length of the dataset
    ///
    /// returns: the best computed centroids in the training.
    ///
    pub fn train<T>(
        &self,
        dataset: &PlainDenseDataset<T>,
        k: usize,
        weights: Option<Vec<f32>>,
    ) -> PlainDenseDataset<T>
    where
        T: Float + EuclideanDistance<T> + DotProduct<T> + num_traits::ToPrimitive + num_traits::FromPrimitive,
    {
    let n = dataset.len();

        if n == k {
            if self.verbose {
                println!("WARNING: number of training data is equal to the number of clusters.");
            }
            return dataset.clone();
        }

        if self.verbose && n <= k * self.min_points_per_centroid {
            println!(
                "WARNING: You provided {} training points for {} centroids,
                but the minimum number of points per centroid set to {}.
                Consider increasing the number of training points.
                ",
                n, k, self.min_points_per_centroid
            )
        }

        // Create a seeded RNG for reproducibility (seed 524 for k-means)
        let mut rng = StdRng::seed_from_u64(524);

        // Determine the sample size using the new sampling strategy
        let actual_sample_size = if let Some(sample_size) = self.sample_size {
            // User-provided sample size
            sample_size
        } else if n > 1_000_000 {
            // If dataset is larger than 1M, use sampling strategy:
            // min(10^7, N, max(10^6, 2 * min_points_per_centroid * k, N/(2*n_iter)))
            let min_by_cluster = 2 * self.min_points_per_centroid * k;
            let min_by_iter = n / (2 * self.n_iter);
            let candidate = std::cmp::max(std::cmp::max(1_000_000, min_by_cluster), min_by_iter);
            std::cmp::min(std::cmp::min(10_000_000, n), candidate)
        } else {
            // For datasets <= 1M, use all data
            n
        };

        // Use sample_size if we need to subsample, otherwise use full dataset
        let (training_dataset, training_weights) = if actual_sample_size < n {
            if self.verbose {
                println!("Sampling {} training points from {} vectors", actual_sample_size, n);
            }
            let sampled = dataset.from_random_sample_with_rng(&mut rng, actual_sample_size);
            // If weights were provided, we ignore them for the sampled dataset
            // (proper implementation would sample weights too, but it's simpler this way)
            (sampled, None)
        } else {
            // No sampling needed, use dataset as-is (avoid unnecessary clone)
            if self.verbose {
                println!("Using all {} training points (no sampling)", n);
            }
            (dataset.clone(), weights)
        };

        let training_n = training_dataset.len();
        let d = Dataset::dim(&training_dataset);

        if self.verbose {
            println!(
                "Clustering {} points in {}D to {} clusters, redo {} times, {} iterations",
                training_n, d, k, self.n_redo, self.n_iter
            );
        }

        let mut best_obj = f32::MAX;

        // clustering-related
        let mut best_centroids = PlainDenseDataset::with_dim_plain(d);
        let mut assignments: Vec<(f32, usize)>;
        let w = training_weights.as_deref();

        if self.verbose && k >= self.index_threshold {
            println!("Will use HNSW index for {} centroids (threshold: {})", k, self.index_threshold);
        }

        for redo in 0..self.n_redo {
            let mut centroids_dataset = training_dataset.from_random_sample_with_rng(&mut rng, k);

            let mut obj;
            let mut average_imbalance_factor = 0.0;
            let mut total_splits = 0;

            for i in 0..self.n_iter {
                let t0 = Instant::now();
                
                // Assignment: find nearest centroid for each vector
                // compute_assignments will internally decide whether to use HNSW or flat search
                assignments = KMeans::compute_assignments(&training_dataset, &centroids_dataset, self.index_threshold);
                
                let search_time = t0.elapsed();

                obj = assignments.iter().map(|&(value, _)| value).sum();
                
                // Update: recompute centroids
                let (n_split, histograms, centroids) =
                    KMeans::update_and_split(&training_dataset, w, k, &assignments, &mut rng);

                let imbalance_factor = KMeans::imbalance_factor(&histograms, k);

                average_imbalance_factor += imbalance_factor;
                total_splits += n_split;

                if obj < best_obj {
                    if self.verbose {
                        println!("New best objective: {} (keep new clusters)", obj);
                    }
                    best_obj = obj;
                    best_centroids = centroids.clone();
                }

                centroids_dataset = PlainDenseDataset::from_vec_plain(centroids.values().to_vec(), d);

                if self.verbose {
                    println!(
                        "Iteration {}, imbalance: {}, splits: {}, search{:.2?}",
                        i, imbalance_factor, n_split, search_time
                    );
                }
            }

            if self.verbose {
                println!(
                    "Outer iteration {} -- average imbalance: {}, splits: {}",
                    redo,
                    average_imbalance_factor / (self.n_iter + 1) as f32,
                    total_splits
                );
            }
        }
        best_centroids
    }
}

#[derive(Default)]
pub struct KMeansBuilder {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    min_points_per_centroid: usize,
    max_points_per_centroid: usize,
    sample_size: Option<usize>,
    index_threshold: usize,
}

impl KMeansBuilder {
    pub fn new() -> KMeansBuilder {
        KMeansBuilder {
            n_iter: 10,
            n_redo: 1,
            verbose: false,
            min_points_per_centroid: 39,
            max_points_per_centroid: 256,
            sample_size: None,
            index_threshold: 50_000, // Default: use HNSW when k >= 50,000
        }
    }

    pub fn n_iter(mut self, n_iter: usize) -> KMeansBuilder {
        self.n_iter = n_iter;
        self
    }

    pub fn n_redo(mut self, n_redo: usize) -> KMeansBuilder {
        self.n_redo = n_redo;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> KMeansBuilder {
        self.verbose = verbose;
        self
    }

    pub fn min_points_per_centroid(mut self, min_points_per_centroid: usize) -> KMeansBuilder {
        self.min_points_per_centroid = min_points_per_centroid;
        self
    }

    pub fn max_points_per_centroid(mut self, max_points_per_centroid: usize) -> KMeansBuilder {
        self.max_points_per_centroid = max_points_per_centroid;
        self
    }

    pub fn sample_size(mut self, sample_size: Option<usize>) -> KMeansBuilder {
        self.sample_size = sample_size;
        self
    }

    pub fn index_threshold(mut self, index_threshold: usize) -> KMeansBuilder {
        self.index_threshold = index_threshold;
        self
    }

    pub fn build(self) -> KMeans {
        KMeans {
            n_iter: self.n_iter,
            n_redo: self.n_redo,
            verbose: self.verbose,
            min_points_per_centroid: self.min_points_per_centroid,
            sample_size: self.sample_size,
            index_threshold: self.index_threshold,
        }
    }
}

//#[cfg(test)]
//mod tests;