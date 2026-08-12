//! Flat (exhaustive search) index bindings.

mod dense;
mod sparse;

pub use dense::DenseFlatIndex;
pub use sparse::SparseFlatIndex;
