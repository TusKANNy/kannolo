use crate::{dot_product_simd, Dataset};
use crate::{PlainDenseDataset, Vector1D};
use std::time::Instant;

pub struct KMeans {
    n_iter: usize,
    n_redo: usize,
    verbose: bool,
    min_points_per_centroid: usize,
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
        return unfairness_factor * k as f32 / (total * total);
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
    /// and the new centroids in a VecDataset<f32>
    ///
    fn update_and_split(
        dataset: &PlainDenseDataset<f32>,
        weights: Option<&[f32]>,
        k: usize,
        assignments: &[(f32, usize)],
    ) -> (usize, Vec<f32>, PlainDenseDataset<f32>) {
        let n = dataset.len();
        let d = dataset.dim();

        let mut centroids = vec![0.0; k * d];
        let mut histograms = vec![0.0; k];

        // update histogram of assignments with a weighted sum of vectors, if weights is not None
        // otherwise we assume each vector weights 1.0
        // then, compute the sum all vectors assigned to each centroid
        match weights {
            Some(w) => {
                for (i, ((_, ci), current_vector)) in
                    assignments.iter().zip(dataset.iter()).enumerate()
                {
                    histograms[*ci] += w[i];
                    centroids[*ci * d..(*ci + 1) * d]
                        .iter_mut()
                        .zip(current_vector.values_as_slice().iter())
                        .for_each(|(c, x)| *c += x * w[*ci]);
                }
            }
            None => {
                for ((_, ci), current_vector) in assignments.iter().zip(dataset.iter()) {
                    histograms[*ci] += 1.0;
                    centroids[*ci * d..(*ci + 1) * d]
                        .iter_mut()
                        .zip(current_vector.values_as_slice().iter())
                        .for_each(|(c, x)| *c += x);
                }
            }
        }

        // normalize centroids
        for (histogram, centroid) in histograms.iter().zip(centroids.chunks_exact_mut(d)) {
            if *histogram == 0.0 {
                continue;
            }
            centroid.iter_mut().for_each(|c| *c /= *histogram);
        }

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
                let r = rand::random::<f32>();
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
        //(n_splits, histograms, VecDataset::from_vec(centroids, k, d))
        (
            n_splits,
            histograms,
            PlainDenseDataset::from_vec_plain(centroids, d),
        )
    }

    /// Runs K-Means training on a dataset with k clusters.
    /// If the user has provided input weights, the computation of centroids is the weighted mean
    /// of every vector assigned to their cluster. Otherwise the computation is just the mean.
    ///
    /// # Arguments
    ///
    /// * `dataset`: the dataset
    /// * `k`: the desired number of clusters
    /// * `weights`: optionally weights of the same length of the dataset
    ///
    /// returns: the best computed centroids in the training.
    ///
    pub fn train(
        &self,
        dataset: &PlainDenseDataset<f32>,
        k: usize,
        weights: Option<Vec<f32>>,
    ) -> PlainDenseDataset<f32> {
        let n = dataset.len();

        if n == k {
            if self.verbose {
                println!("WARNING: number of training data is equal to the number of clusters.");
            }
            return dataset.clone();
        }

        if n <= k * self.min_points_per_centroid {
            println!(
                "WARNING: You provided {} training points for {} centroids,
                but the minimum number of points per centroid set to {}.
                Consider increasing the number of training points.
                ",
                n, k, self.min_points_per_centroid
            )
        }

        // assert!(
        //     n >= k * self.min_points_per_centroid,
        //     "The number of input vectors {} must be greater than the number of clusters",
        //     n,
        // );

        // assert!(
        //     n <= k * self.max_points_per_centroid,
        //     "The number of input vectors must be smaller than the maximum number \
        //     of points per centroid times the number of clusters"
        // );

        let d = dataset.dim();

        if self.verbose {
            println!(
                "Clustering {} points in {}D to {} clusters, redo {} times, {} iterations",
                n, d, k, self.n_redo, self.n_iter
            );
        }

        let mut best_obj = f32::MAX;

        // clustering-related
        let mut best_centroids = PlainDenseDataset::with_dim_plain(d);
        let mut assignments: Vec<(f32, usize)>;
        let w = weights.as_ref().map(|vec| vec.as_slice());

        for redo in 0..self.n_redo {
            let mut index = PlainDenseDataset::from_random_sample(&dataset, k);

            let mut obj;
            let mut average_imbalance_factor = 0.0;
            let mut total_splits = 0;

            for i in 0..self.n_iter {
                let t0 = Instant::now();
                assignments = index.top1(dataset.as_ref(), n);
                let search_time = t0.elapsed();

                obj = assignments.iter().map(|&(value, _)| value).sum();
                let (n_split, histograms, centroids) =
                    KMeans::update_and_split(&dataset, w, k, &assignments);

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

                //index = ScanL2::from(centroids);
                index = PlainDenseDataset::from_vec_plain(centroids.values().to_vec(), d);

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
}

impl KMeansBuilder {
    pub fn new() -> KMeansBuilder {
        KMeansBuilder {
            n_iter: 25,
            n_redo: 1,
            verbose: false,
            min_points_per_centroid: 39,
            max_points_per_centroid: 256,
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

    pub fn build(self) -> KMeans {
        KMeans {
            n_iter: self.n_iter,
            n_redo: self.n_redo,
            verbose: self.verbose,
            min_points_per_centroid: self.min_points_per_centroid,
        }
    }
}

//#[cfg(test)]
//mod tests;
