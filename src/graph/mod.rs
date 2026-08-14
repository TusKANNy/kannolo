pub mod egb;
#[allow(clippy::module_inception)]
pub mod graph;
pub mod neighbors;

pub use graph::{GraphTrait, GrowableGraph};
pub use neighbors::{
    FixedDegreeNeighbors, NeighborData, Neighbors, PlainNeighbors, StreamVByteNeighbors,
};

/// Default graph type using plain (uncompressed) neighbor storage.
pub type Graph = graph::Graph<PlainNeighbors>;

/// Graph type using the fixed-stride, offsets-free neighbor storage.
pub type GraphFixedDegree = graph::Graph<FixedDegreeNeighbors>;
