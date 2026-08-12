//! `DensePQHNSW`: HNSW over product-quantized dense data, with the exhaustive
//! `(metric, graph_type, m_pq)` monomorphization table it dispatches into.

use std::f32;

use crate::graph::Graph;
use crate::graph::graph::Graph as GenericGraph;
use crate::graph::neighbors::{NeighborData, Neighbors, PlainNeighbors, StreamVByteNeighbors};
use crate::hnsw::{
    EarlyTerminationStrategy, HNSW, HNSWBuildConfiguration, HNSWSearchConfiguration,
};
use vectorium::IndexSerializer;
use vectorium::core::index::{Index, IndexStats};
use vectorium::dataset::ConvertFrom;
use vectorium::vector_encoder::{DenseVectorEncoder, VectorEncoder};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use vectorium::distances::{Distance, DotProduct, SquaredEuclideanDistance};
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::encoders::pq::{ProductQuantizer, ProductQuantizerDistance};
use vectorium::vector::DenseVectorView;
use vectorium::{Dataset, DenseDataset, PlainDenseDataset};

use super::{GraphTypeKind, load_index_err, parse_build_graph_type, parse_graph_type};
use crate::pylib::common::{MetricKind, parse_metric, push_results, read_npy_dataset};

enum DensePQHNSWGeneric<D>
where
    D: ProductQuantizerDistance,
{
    PQ8(HNSW<DenseDataset<ProductQuantizer<8, D>>, Graph>),
    PQ16(HNSW<DenseDataset<ProductQuantizer<16, D>>, Graph>),
    PQ32(HNSW<DenseDataset<ProductQuantizer<32, D>>, Graph>),
    PQ48(HNSW<DenseDataset<ProductQuantizer<48, D>>, Graph>),
    PQ64(HNSW<DenseDataset<ProductQuantizer<64, D>>, Graph>),
    PQ96(HNSW<DenseDataset<ProductQuantizer<96, D>>, Graph>),
    PQ128(HNSW<DenseDataset<ProductQuantizer<128, D>>, Graph>),
    PQ192(HNSW<DenseDataset<ProductQuantizer<192, D>>, Graph>),
    PQ256(HNSW<DenseDataset<ProductQuantizer<256, D>>, Graph>),
    PQ384(HNSW<DenseDataset<ProductQuantizer<384, D>>, Graph>),
    PQ8StreamVByte(HNSW<DenseDataset<ProductQuantizer<8, D>>, GenericGraph<StreamVByteNeighbors>>),
    PQ16StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<16, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ32StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<32, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ48StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<48, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ64StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<64, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ96StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<96, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ128StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<128, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ192StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<192, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ256StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<256, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
    PQ384StreamVByte(
        HNSW<DenseDataset<ProductQuantizer<384, D>>, GenericGraph<StreamVByteNeighbors>>,
    ),
}

/// Builds a plain (uncompressed, original node order) PQ-quantized dense HNSW.
fn build_pq_l2<const M: usize>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, Graph>
where
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Encoder = ProductQuantizer<M, SquaredEuclideanDistance>>,
    for<'a> DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<&'a PlainDenseDataset<f32, SquaredEuclideanDistance>, Config = ()>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    plain_index.convert_dataset_into_ref(())
}

/// Builds an EGB-permuted PQ-quantized dense HNSW, recompressed into `Ndst`
/// (`PlainNeighbors` for `permuted`, `StreamVByteNeighbors` for `streamvbyte`).
fn build_pq_l2_compressed<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, GenericGraph<Ndst>>
where
    Ndst: Neighbors + From<NeighborData>,
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Encoder = ProductQuantizer<M, SquaredEuclideanDistance>>,
    for<'a> DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        ConvertFrom<&'a PlainDenseDataset<f32, SquaredEuclideanDistance>, Config = ()>,
    // `permute_and_encode` hands the permuted dataset back as `Owned`; restated here because the
    // compiler cannot normalize `Owned = Self` through the generic parameter (see
    // `build_permuted_and_save` in `src/bin/hnsw_build.rs`).
    DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>:
        Dataset<Owned = DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, SquaredEuclideanDistance>:
        VectorEncoder<Distance = SquaredEuclideanDistance>,
    <ProductQuantizer<M, SquaredEuclideanDistance> as VectorEncoder>::Distance:
        vectorium::distances::Distance,
{
    // Quantize before permuting, never after — see `build_permuted_and_save`.
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    let converted: HNSW<DenseDataset<ProductQuantizer<M, SquaredEuclideanDistance>>, Graph> =
        plain_index.convert_dataset_into_ref(());
    converted.permute_and_encode::<Ndst>()
}

/// Builds a plain (uncompressed, original node order) PQ-quantized dense HNSW.
fn build_pq_ip<const M: usize>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, DotProduct>>, Graph>
where
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    plain_index.convert_dataset_into_ref(())
}

/// Builds an EGB-permuted PQ-quantized dense HNSW, recompressed into `Ndst`
/// (`PlainNeighbors` for `permuted`, `StreamVByteNeighbors` for `streamvbyte`).
fn build_pq_ip_compressed<const M: usize, Ndst>(
    dataset: PlainDenseDataset<f32, DotProduct>,
    config: &HNSWBuildConfiguration,
) -> HNSW<DenseDataset<ProductQuantizer<M, DotProduct>>, GenericGraph<Ndst>>
where
    Ndst: Neighbors + From<NeighborData>,
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Encoder = ProductQuantizer<M, DotProduct>>,
    for<'a> DenseDataset<ProductQuantizer<M, DotProduct>>:
        ConvertFrom<&'a PlainDenseDataset<f32, DotProduct>, Config = ()>,
    // `permute_and_encode` hands the permuted dataset back as `Owned`; restated here because the
    // compiler cannot normalize `Owned = Self` through the generic parameter (see
    // `build_permuted_and_save` in `src/bin/hnsw_build.rs`).
    DenseDataset<ProductQuantizer<M, DotProduct>>:
        Dataset<Owned = DenseDataset<ProductQuantizer<M, DotProduct>>>,
    ProductQuantizer<M, DotProduct>: DenseVectorEncoder<InputValueType = f32, OutputValueType = u8>,
    ProductQuantizer<M, DotProduct>: VectorEncoder<Distance = DotProduct>,
    <ProductQuantizer<M, DotProduct> as VectorEncoder>::Distance: vectorium::distances::Distance,
{
    // Quantize before permuting, never after — see `build_permuted_and_save`.
    let plain_index: HNSW<_, Graph> = HNSW::build_index(dataset, config);
    let converted: HNSW<DenseDataset<ProductQuantizer<M, DotProduct>>, Graph> =
        plain_index.convert_dataset_into_ref(());
    converted.permute_and_encode::<Ndst>()
}

impl DensePQHNSWGeneric<DotProduct> {
    fn build_from_dataset(
        dataset: PlainDenseDataset<f32, DotProduct>,
        config: &HNSWBuildConfiguration,
        m_pq: usize,
        gt: GraphTypeKind,
    ) -> PyResult<Self> {
        Ok(match m_pq {
            8 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ8(build_pq_ip::<8>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ8(build_pq_ip_compressed::<8, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ8StreamVByte(build_pq_ip_compressed::<
                        8,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            16 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ16(build_pq_ip::<16>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ16(build_pq_ip_compressed::<16, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ16StreamVByte(build_pq_ip_compressed::<
                        16,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            32 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ32(build_pq_ip::<32>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ32(build_pq_ip_compressed::<32, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ32StreamVByte(build_pq_ip_compressed::<
                        32,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            48 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ48(build_pq_ip::<48>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ48(build_pq_ip_compressed::<48, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ48StreamVByte(build_pq_ip_compressed::<
                        48,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            64 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ64(build_pq_ip::<64>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ64(build_pq_ip_compressed::<64, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ64StreamVByte(build_pq_ip_compressed::<
                        64,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            96 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ96(build_pq_ip::<96>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ96(build_pq_ip_compressed::<96, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ96StreamVByte(build_pq_ip_compressed::<
                        96,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            128 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ128(build_pq_ip::<128>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ128(build_pq_ip_compressed::<128, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ128StreamVByte(build_pq_ip_compressed::<
                        128,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            192 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ192(build_pq_ip::<192>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ192(build_pq_ip_compressed::<192, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ192StreamVByte(build_pq_ip_compressed::<
                        192,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            256 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ256(build_pq_ip::<256>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ256(build_pq_ip_compressed::<256, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ256StreamVByte(build_pq_ip_compressed::<
                        256,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            384 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ384(build_pq_ip::<384>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ384(build_pq_ip_compressed::<384, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ384StreamVByte(build_pq_ip_compressed::<
                        384,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ));
            }
        })
    }
}

impl DensePQHNSWGeneric<SquaredEuclideanDistance> {
    fn build_from_dataset(
        dataset: PlainDenseDataset<f32, SquaredEuclideanDistance>,
        config: &HNSWBuildConfiguration,
        m_pq: usize,
        gt: GraphTypeKind,
    ) -> PyResult<Self> {
        Ok(match m_pq {
            8 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ8(build_pq_l2::<8>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ8(build_pq_l2_compressed::<8, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ8StreamVByte(build_pq_l2_compressed::<
                        8,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            16 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ16(build_pq_l2::<16>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ16(build_pq_l2_compressed::<16, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ16StreamVByte(build_pq_l2_compressed::<
                        16,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            32 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ32(build_pq_l2::<32>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ32(build_pq_l2_compressed::<32, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ32StreamVByte(build_pq_l2_compressed::<
                        32,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            48 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ48(build_pq_l2::<48>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ48(build_pq_l2_compressed::<48, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ48StreamVByte(build_pq_l2_compressed::<
                        48,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            64 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ64(build_pq_l2::<64>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ64(build_pq_l2_compressed::<64, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ64StreamVByte(build_pq_l2_compressed::<
                        64,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            96 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ96(build_pq_l2::<96>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ96(build_pq_l2_compressed::<96, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ96StreamVByte(build_pq_l2_compressed::<
                        96,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            128 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ128(build_pq_l2::<128>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ128(build_pq_l2_compressed::<128, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ128StreamVByte(build_pq_l2_compressed::<
                        128,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            192 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ192(build_pq_l2::<192>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ192(build_pq_l2_compressed::<192, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ192StreamVByte(build_pq_l2_compressed::<
                        192,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            256 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ256(build_pq_l2::<256>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ256(build_pq_l2_compressed::<256, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ256StreamVByte(build_pq_l2_compressed::<
                        256,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            384 => match gt {
                GraphTypeKind::Standard => {
                    DensePQHNSWGeneric::PQ384(build_pq_l2::<384>(dataset, config))
                }
                GraphTypeKind::Permuted => {
                    DensePQHNSWGeneric::PQ384(build_pq_l2_compressed::<384, PlainNeighbors>(
                        dataset, config,
                    ))
                }
                GraphTypeKind::Compressed => {
                    DensePQHNSWGeneric::PQ384StreamVByte(build_pq_l2_compressed::<
                        384,
                        StreamVByteNeighbors,
                    >(dataset, config))
                }
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ));
            }
        })
    }
}

impl<D> DensePQHNSWGeneric<D>
where
    D: ProductQuantizerDistance + Distance + ScalarDenseSupportedDistance,
{
    fn load(path: &str, m_pq: usize, gt: GraphTypeKind) -> PyResult<Self> {
        let inner = match (m_pq, gt) {
            (8, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<8, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ8(index)
            }
            (8, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<8, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<8, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ8StreamVByte(index)
            }
            (16, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<16, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ16(index)
            }
            (16, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<16, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<16, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ16StreamVByte(index)
            }
            (32, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<32, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ32(index)
            }
            (32, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<32, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<32, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ32StreamVByte(index)
            }
            (48, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<48, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ48(index)
            }
            (48, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<48, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<48, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ48StreamVByte(index)
            }
            (64, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<64, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ64(index)
            }
            (64, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<64, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<64, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ64StreamVByte(index)
            }
            (96, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<96, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ96(index)
            }
            (96, GraphTypeKind::Compressed) => {
                let index: HNSW<DenseDataset<ProductQuantizer<96, D>>, GenericGraph<StreamVByteNeighbors>> = <HNSW<
                    DenseDataset<ProductQuantizer<96, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ96StreamVByte(index)
            }
            (128, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<128, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ128(index)
            }
            (128, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<128, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ128StreamVByte(index)
            }
            (192, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<192, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ192(index)
            }
            (192, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<192, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ192StreamVByte(index)
            }
            (256, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<256, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ256(index)
            }
            (256, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<256, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ256StreamVByte(index)
            }
            (384, GraphTypeKind::Standard | GraphTypeKind::Permuted) => {
                let index: HNSW<DenseDataset<ProductQuantizer<384, D>>, Graph> = <HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    Graph,
                > as IndexSerializer>::load_index(
                    path
                )
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ384(index)
            }
            (384, GraphTypeKind::Compressed) => {
                let index: HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > = <HNSW<
                    DenseDataset<ProductQuantizer<384, D>>,
                    GenericGraph<StreamVByteNeighbors>,
                > as IndexSerializer>::load_index(path)
                .map_err(load_index_err)?;
                DensePQHNSWGeneric::PQ384StreamVByte(index)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported m_pq value for load. Supported values: 8, 16, 32, 48, 64, 96, 128, 192, 256, 384.",
                ));
            }
        };
        Ok(inner)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let result = match self {
            DensePQHNSWGeneric::PQ8(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ16(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ32(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ48(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ64(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ96(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ128(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ192(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ256(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ384(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ8StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ16StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ32StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ48StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ64StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ96StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ128StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ192StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ256StreamVByte(index) => index.save_index(path),
            DensePQHNSWGeneric::PQ384StreamVByte(index) => index.save_index(path),
        };

        result.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error saving index: {:?}", e))
        })
    }

    fn dim(&self) -> usize {
        match self {
            DensePQHNSWGeneric::PQ8(index) => index.dim(),
            DensePQHNSWGeneric::PQ16(index) => index.dim(),
            DensePQHNSWGeneric::PQ32(index) => index.dim(),
            DensePQHNSWGeneric::PQ48(index) => index.dim(),
            DensePQHNSWGeneric::PQ64(index) => index.dim(),
            DensePQHNSWGeneric::PQ96(index) => index.dim(),
            DensePQHNSWGeneric::PQ128(index) => index.dim(),
            DensePQHNSWGeneric::PQ192(index) => index.dim(),
            DensePQHNSWGeneric::PQ256(index) => index.dim(),
            DensePQHNSWGeneric::PQ384(index) => index.dim(),
            DensePQHNSWGeneric::PQ8StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ16StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ32StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ48StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ64StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ96StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ128StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ192StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ256StreamVByte(index) => index.dim(),
            DensePQHNSWGeneric::PQ384StreamVByte(index) => index.dim(),
        }
    }

    fn search(
        &self,
        query: DenseVectorView<'_, f32>,
        k: usize,
        search_config: &HNSWSearchConfiguration,
    ) -> Vec<vectorium::dataset::ScoredVector<D>> {
        match self {
            DensePQHNSWGeneric::PQ8(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ16(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ32(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ48(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ64(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ96(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ128(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ192(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ256(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ384(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ8StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ16StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ32StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ48StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ64StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ96StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ128StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ192StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ256StreamVByte(index) => index.search(query, k, search_config),
            DensePQHNSWGeneric::PQ384StreamVByte(index) => index.search(query, k, search_config),
        }
    }
}

enum DensePQHNSWEnum {
    Euclidean(DensePQHNSWGeneric<SquaredEuclideanDistance>),
    DotProduct(DensePQHNSWGeneric<DotProduct>),
}

#[pyclass]
pub struct DensePQHNSW {
    inner: DensePQHNSWEnum,
}

#[pymethods]
impl DensePQHNSW {
    #[staticmethod]
    #[pyo3(signature = (data_path, m_pq, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_file(
        data_path: &str,
        m_pq: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
    ) -> PyResult<Self> {
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
                    read_npy_dataset::<SquaredEuclideanDistance>(data_path)?;
                DensePQHNSWEnum::Euclidean(
                    DensePQHNSWGeneric::<SquaredEuclideanDistance>::build_from_dataset(
                        dataset, &config, m_pq, gt,
                    )?,
                )
            }
            MetricKind::DotProduct => {
                let dataset: PlainDenseDataset<f32, DotProduct> =
                    read_npy_dataset::<DotProduct>(data_path)?;
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::<DotProduct>::build_from_dataset(
                    dataset, &config, m_pq, gt,
                )?)
            }
        };

        Ok(DensePQHNSW { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (data_vec, dim, m_pq, m=32, ef_construction=200, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn build_from_array(
        data_vec: PyReadonlyArray1<f32>,
        dim: usize,
        m_pq: usize,
        m: usize,
        ef_construction: usize,
        metric: String,
        graph_type: String,
    ) -> PyResult<Self> {
        let data_vec = data_vec.as_slice()?.to_vec();
        let n_vecs = data_vec.len() / dim;
        let config = HNSWBuildConfiguration::default()
            .with_num_neighbors(m)
            .with_ef_construction(ef_construction);
        let gt = parse_build_graph_type(&graph_type, m)?;

        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                let encoder = PlainDenseQuantizer::<f32, SquaredEuclideanDistance>::new(dim);
                let dataset: PlainDenseDataset<f32, SquaredEuclideanDistance> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePQHNSWEnum::Euclidean(
                    DensePQHNSWGeneric::<SquaredEuclideanDistance>::build_from_dataset(
                        dataset, &config, m_pq, gt,
                    )?,
                )
            }
            MetricKind::DotProduct => {
                let encoder = PlainDenseQuantizer::<f32, DotProduct>::new(dim);
                let dataset: PlainDenseDataset<f32, DotProduct> =
                    DenseDataset::from_raw(data_vec.into_boxed_slice(), n_vecs, encoder);
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::<DotProduct>::build_from_dataset(
                    dataset, &config, m_pq, gt,
                )?)
            }
        };

        Ok(DensePQHNSW { inner })
    }

    /// Loads a previously saved index. `graph_type` must match the value used at build
    /// time (`standard`/`permuted` both load as the same on-disk representation).
    #[staticmethod]
    #[pyo3(signature = (path, m_pq, metric="dotproduct".to_string(), graph_type="standard".to_string()))]
    pub fn load(path: &str, m_pq: usize, metric: String, graph_type: String) -> PyResult<Self> {
        let gt = parse_graph_type(&graph_type)?;
        let inner = match parse_metric(&metric)? {
            MetricKind::Euclidean => {
                DensePQHNSWEnum::Euclidean(DensePQHNSWGeneric::load(path, m_pq, gt)?)
            }
            MetricKind::DotProduct => {
                DensePQHNSWEnum::DotProduct(DensePQHNSWGeneric::load(path, m_pq, gt)?)
            }
        };
        Ok(DensePQHNSW { inner })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => inner.save(path),
            DensePQHNSWEnum::DotProduct(inner) => inner.save(path),
        }
    }

    pub fn search(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let dim = match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => inner.dim(),
            DensePQHNSWEnum::DotProduct(inner) => inner.dim(),
        };
        let query_slice = query.as_slice()?;
        if query_slice.len() != dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "query dimension {} does not match index dimension {}",
                query_slice.len(),
                dim
            )));
        }
        let query_view = DenseVectorView::new(query_slice);
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let mut distances = Vec::with_capacity(k);
        let mut ids = Vec::with_capacity(k);
        match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => {
                push_results(
                    inner.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
            DensePQHNSWEnum::DotProduct(inner) => {
                push_results(
                    inner.search(query_view, k, &search_config),
                    k,
                    &mut distances,
                    &mut ids,
                );
            }
        }

        Python::attach(|py| {
            let distances_array = PyArray1::from_vec(py, distances).to_owned();
            let ids_array = PyArray1::from_vec(py, ids).to_owned();
            Ok((distances_array.into(), ids_array.into()))
        })
    }

    /// Search a batch of queries, optionally in parallel.
    ///
    /// `num_threads` controls the threading model:
    /// - `0` — use rayon's default thread pool (typically all available cores).
    /// - `1` — serial loop, no rayon involvement. Use this to reproduce single-thread
    ///   benchmarks that pin the process via `numactl --physcpubind`.
    /// - `n` — build a temporary rayon pool with `n` threads for the duration of this call.
    #[pyo3(signature = (queries, k, ef_search=100, early_exit_threshold=None, num_threads=0))]
    pub fn batch_search(
        &self,
        py: Python<'_>,
        queries: PyReadonlyArray1<f32>,
        k: usize,
        ef_search: usize,
        early_exit_threshold: Option<f32>,
        num_threads: usize,
    ) -> PyResult<(Py<PyArray1<f32>>, Py<PyArray1<i64>>)> {
        let mut search_config = HNSWSearchConfiguration::default().with_ef_search(ef_search);
        if let Some(threshold) = early_exit_threshold {
            search_config =
                search_config.with_early_termination(EarlyTerminationStrategy::DistanceAdaptive {
                    lambda: threshold,
                });
        }

        let queries_slice = queries.as_slice()?;
        let dim = match &self.inner {
            DensePQHNSWEnum::Euclidean(inner) => inner.dim(),
            DensePQHNSWEnum::DotProduct(inner) => inner.dim(),
        };
        if queries_slice.len() % dim != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "queries array length {} is not a multiple of index dimension {}",
                queries_slice.len(),
                dim
            )));
        }
        let num_queries = queries_slice.len() / dim;

        let search_one = |i: usize| -> (Vec<f32>, Vec<i64>) {
            let query_view = DenseVectorView::new(&queries_slice[i * dim..(i + 1) * dim]);
            let mut distances = Vec::with_capacity(k);
            let mut ids = Vec::with_capacity(k);
            match &self.inner {
                DensePQHNSWEnum::Euclidean(inner) => {
                    push_results(
                        inner.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
                DensePQHNSWEnum::DotProduct(inner) => {
                    push_results(
                        inner.search(query_view, k, &search_config),
                        k,
                        &mut distances,
                        &mut ids,
                    );
                }
            }
            (distances, ids)
        };

        let results: Vec<(Vec<f32>, Vec<i64>)> = py.detach(|| match num_threads {
            1 => (0..num_queries).map(search_one).collect(),
            0 => (0..num_queries).into_par_iter().map(search_one).collect(),
            n => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("failed to build rayon thread pool")
                .install(|| (0..num_queries).into_par_iter().map(search_one).collect()),
        });

        let mut all_distances = Vec::with_capacity(num_queries * k);
        let mut all_ids = Vec::with_capacity(num_queries * k);
        for (d, i) in results {
            all_distances.extend(d);
            all_ids.extend(i);
        }

        let distances_array = PyArray1::from_vec(py, all_distances).to_owned();
        let ids_array = PyArray1::from_vec(py, all_ids).to_owned();
        Ok((distances_array.into(), ids_array.into()))
    }
}
