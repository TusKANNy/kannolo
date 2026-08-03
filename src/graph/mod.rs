pub mod egb;
#[allow(clippy::module_inception)]
pub mod graph;
pub mod neighbors;

pub use graph::{GraphFixedDegree, GraphTrait, GrowableGraph};
pub use neighbors::{NeighborData, Neighbors, PlainNeighbors, StreamVByteNeighbors};

/// Default graph type using plain (uncompressed) neighbor storage.
pub type Graph = graph::Graph<PlainNeighbors>;
