pub mod dot_product;
pub mod euclidean_distance;
pub mod multivector;

pub mod simd {
    pub mod distances;
    pub mod transpose;
    pub mod utils;
}

pub use dot_product::*;
pub use euclidean_distance::*;
pub use multivector::*;
