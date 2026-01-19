use crate::VectorType;
use crate::distances::dot_product_batch_4_simd;
use crate::distances::dot_product_simd;
use crate::{Float, MultiVector};

/// Computes MaxSim between two multivectors.
/// MaxSim is the sum over query vectors of the maximum dot product
/// with any document vector.
#[inline]
pub fn multivector_maxsim<T: Float>(
    multivec1: &MultiVector<&[T]>,
    multivec2: &MultiVector<&[T]>,
) -> f32
where
    T: crate::DotProduct<T>,
{
    multivector_maxsim_query(multivec1, multivec2)
}

/// Computes MaxSim between a query multivector and a document multivector.
/// For each vector in the query, finds the maximum dot product with any vector in the document.
/// This is the ORIGINAL implementation.
#[inline]
pub fn multivector_maxsim_query<T: Float>(
    query: &MultiVector<&[T]>,
    document: &MultiVector<&[T]>,
) -> f32
where
    T: crate::DotProduct<T>,
{
    let mut total_sim = 0.0f32;

    // For each query vector
    for query_vec in query.iter_vectors() {
        let mut max_sim = f32::NEG_INFINITY;

        // Find maximum similarity with any document vector
        for doc_vec in document.iter_vectors() {
            let sim = dot_product_simd(query_vec, doc_vec);
            if sim > max_sim {
                max_sim = sim;
            }
        }

        total_sim += max_sim;
    }

    total_sim
}

/// Computes MaxSim between a query multivector and a document multivector using batched dot products (batch size 4).
/// For each query vector, finds the maximum dot product with any vector in the document, using SIMD batch processing.
pub fn multivector_maxsim_query_batch_4<T: Float>(
    query: &MultiVector<&[T]>,
    document: &MultiVector<&[T]>,
) -> f32
where
    T: crate::DotProduct<T>,
{
    let mut total_sim = 0.0f32;
    
    // 1. Get raw access to document memory
    let doc_slice = document.values_as_slice();
    let dim = document.vector_dim();
    
    // Calculate the size of a "batch" in raw elements (4 vectors * dimension)
    let batch_size = dim * 4;

    // Loop 1: Query Vectors
    for query_vec in query.iter_vectors() {
        let mut max_sim = f32::NEG_INFINITY;

        // 2. Create an iterator over "Super Chunks" (blocks containing 4 vectors)
        let mut batch_iter = doc_slice.chunks_exact(batch_size);

        // Loop 2: Process full batches
        for super_chunk in batch_iter.by_ref() {
            // We know super_chunk has exactly 4 * dim elements.
            // We split it manually into 4 references without bounds checks.
            let (v0, rest) = super_chunk.split_at(dim);
            let (v1, rest) = rest.split_at(dim);
            let (v2, v3)   = rest.split_at(dim);

            // Construct the batch array
            let batch = [v0, v1, v2, v3];

            let sims = dot_product_batch_4_simd(query_vec, batch);

            // Reduction
            let batch_max = sims[0].max(sims[1]).max(sims[2]).max(sims[3]);
            max_sim = max_sim.max(batch_max);
        }

        // Loop 3: Process the remainder (the vectors that didn't fit in a batch of 4)
        // batch_iter.remainder() returns the leftover slice (size < 4 * dim)
        for doc_vec in batch_iter.remainder().chunks_exact(dim) {
            let sim = crate::distances::dot_product_simd(query_vec, doc_vec);
            max_sim = max_sim.max(sim);
        }

        total_sim += max_sim;
    }

    total_sim
}

