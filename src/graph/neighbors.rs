use serde::{Deserialize, Serialize};
use toolkit::stream_vbyte::StreamVByteBlocks;

/// Raw neighbor data and per-node offsets, used as the common input format
/// when building any [`Neighbors`] backend.
#[derive(Clone)]
pub struct NeighborData {
    pub data: Box<[u32]>,      // neighbor IDs
    pub offsets: Box<[usize]>, // segment offsets
}

/// Trait for neighbor-list storage backends (plain or compressed).
#[allow(clippy::len_without_is_empty)]
pub trait Neighbors {
    /// Whether this backend needs each adjacency list sorted ascending on construction.
    /// Delta-coding backends do, since they reconstruct by prefix sum; backends that store
    /// neighbor IDs verbatim do not care and must not have their list order disturbed.
    /// Conversions that feed a backend consult this before touching list order.
    const REQUIRES_SORTED: bool = false;

    fn len(&self) -> usize;
    fn n_nodes(&self) -> usize;

    /// Returns the neighbor IDs of `node_id`, in ascending order.
    ///
    /// `scratch` is a work buffer: backends that can hand out their storage directly ignore it,
    /// the others decode into it. **Callers must read only the returned slice, never `scratch`**,
    /// whose length and contents afterwards are unspecified.
    fn get<'a>(&'a self, node_id: usize, scratch: &'a mut Vec<u32>) -> &'a [u32];

    fn space_usage_bytes(&self) -> usize;
}

/// Plain neighbors stored as a flat `Box<[u32]>` sliced by `offsets`.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct PlainNeighbors {
    data: Box<[u32]>,
    offsets: Box<[usize]>,
}

impl From<NeighborData> for PlainNeighbors {
    fn from(nd: NeighborData) -> Self {
        PlainNeighbors {
            data: nd.data,
            offsets: nd.offsets,
        }
    }
}

impl Neighbors for PlainNeighbors {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn n_nodes(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Hands out the stored slice; `scratch` is left untouched.
    #[inline]
    fn get<'a>(&'a self, node_id: usize, _scratch: &'a mut Vec<u32>) -> &'a [u32] {
        let start = self.offsets[node_id];
        let end = self.offsets[node_id + 1];
        &self.data[start..end]
    }

    fn space_usage_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u32>()
            + self.offsets.len() * std::mem::size_of::<usize>()
    }
}

/// Neighbors stored at a fixed stride, with no offsets array.
///
/// Every node owns exactly `stride` contiguous slots and unused ones hold the `u32::MAX`
/// sentinel, so a node's list starts at `node_id * stride` — computed arithmetically, with no
/// dependent load on an offsets array. The trade is space: the padding of partially-filled
/// lists is retained rather than compacted away, which is why this backend is an alternative to
/// [`PlainNeighbors`] rather than a complement to it. It is also mutually exclusive with
/// compression, since padding is exactly what a compressed layout would remove.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct FixedDegreeNeighbors {
    /// `stride`-sized slots per node, sentinel-padded.
    slots: Box<[u32]>,
    stride: usize,
    /// Held explicitly rather than derived as `slots.len() / stride`, which is undefined for a
    /// level whose nodes all have empty lists — a real case for a one-node top level.
    n_nodes: usize,
    /// Number of real (non-sentinel) neighbors, i.e. the logical edge count.
    logical_len: usize,
}

impl From<NeighborData> for FixedDegreeNeighbors {
    /// Re-pads variable-length lists to the longest one, which is the stride this backend needs.
    fn from(nd: NeighborData) -> Self {
        let n_nodes = nd.offsets.len().saturating_sub(1);
        let stride = (0..n_nodes)
            .map(|v| nd.offsets[v + 1] - nd.offsets[v])
            .max()
            .unwrap_or(0);

        let mut slots = vec![u32::MAX; n_nodes * stride];
        for v in 0..n_nodes {
            let (start, end) = (nd.offsets[v], nd.offsets[v + 1]);
            slots[v * stride..v * stride + (end - start)].copy_from_slice(&nd.data[start..end]);
        }

        FixedDegreeNeighbors {
            slots: slots.into_boxed_slice(),
            stride,
            n_nodes,
            logical_len: nd.data.len(),
        }
    }
}

impl Neighbors for FixedDegreeNeighbors {
    fn len(&self) -> usize {
        self.logical_len
    }

    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Hands out the non-sentinel prefix of the node's slots; `scratch` is left untouched.
    /// Neighbors occupy a contiguous prefix, so the first sentinel ends the list.
    #[inline]
    fn get<'a>(&'a self, node_id: usize, _scratch: &'a mut Vec<u32>) -> &'a [u32] {
        let start = node_id * self.stride;
        let slots = &self.slots[start..start + self.stride];
        let len = slots
            .iter()
            .position(|&v| v == u32::MAX)
            .unwrap_or(self.stride);
        &slots[..len]
    }

    fn space_usage_bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<u32>()
    }
}

/// Longest adjacency list a StreamVByte block can hold. The ground level's max degree is
/// `2 * M`, so callers must keep `M` under half of this to use [`StreamVByteNeighbors`].
pub const MAX_NEIGHBORS_PER_NODE: usize = 256;

/// Neighbors stored delta-encoded and compressed with StreamVByte, one block per node.
///
/// Adjacency lists must be sorted in ascending order before construction: the encoder
/// stores `value[i] - value[i-1]` deltas (first delta relative to 0) and reconstructs the
/// original IDs on read via a running prefix sum.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct StreamVByteNeighbors {
    blocks: StreamVByteBlocks<u32>,
    empty_nodes: Box<[u64]>,
    logical_len: usize,
}

impl From<NeighborData> for StreamVByteNeighbors {
    fn from(nd: NeighborData) -> Self {
        let mut deltas = Vec::with_capacity(nd.data.len().max(nd.offsets.len().saturating_sub(1)));
        let mut offsets = Vec::with_capacity(nd.offsets.len());
        let n_nodes = nd.offsets.len().saturating_sub(1);
        let mut empty_nodes = vec![0u64; n_nodes.div_ceil(64)];
        let mut has_empty_nodes = false;

        offsets.push(0);
        for node_id in 0..n_nodes {
            let start = nd.offsets[node_id];
            let end = nd.offsets[node_id + 1];
            if start == end {
                has_empty_nodes = true;
                empty_nodes[node_id / 64] |= 1u64 << (node_id % 64);
                deltas.push(0);
                offsets.push(deltas.len());
                continue;
            }

            assert!(
                end - start <= MAX_NEIGHBORS_PER_NODE,
                "StreamVByteBlocks supports adjacency lists up to {MAX_NEIGHBORS_PER_NODE} neighbors"
            );

            let mut prev = 0u32;
            for &value in &nd.data[start..end] {
                debug_assert!(
                    value >= prev,
                    "Neighbor list must be sorted in ascending order"
                );
                deltas.push(value - prev);
                prev = value;
            }
            offsets.push(deltas.len());
        }

        StreamVByteNeighbors {
            blocks: StreamVByteBlocks::new(&deltas, &offsets),
            empty_nodes: if has_empty_nodes {
                empty_nodes.into_boxed_slice()
            } else {
                Box::new([])
            },
            logical_len: nd.data.len(),
        }
    }
}

impl StreamVByteNeighbors {
    #[inline]
    fn is_empty_node(&self, node_id: usize) -> bool {
        !self.empty_nodes.is_empty()
            && self
                .empty_nodes
                .get(node_id / 64)
                .is_some_and(|word| (word & (1u64 << (node_id % 64))) != 0)
    }
}

impl Neighbors for StreamVByteNeighbors {
    /// Delta coding reconstructs by prefix sum, so the input lists must be ascending.
    const REQUIRES_SORTED: bool = true;

    fn len(&self) -> usize {
        self.logical_len
    }

    fn n_nodes(&self) -> usize {
        self.blocks.num_blocks()
    }

    /// Decodes the node's block into `scratch` and returns the decoded prefix.
    ///
    /// `scratch` is grown once to the largest block a [`StreamVByteBlocks`] can hold and then
    /// reused in place: the block is decoded straight into it and the delta prefix sum runs over
    /// the same buffer, so no intermediate copy is made. Slots past the returned length keep
    /// values from earlier calls, which is why callers must read only the returned slice.
    #[inline]
    fn get<'a>(&'a self, node_id: usize, scratch: &'a mut Vec<u32>) -> &'a [u32] {
        if self.is_empty_node(node_id) {
            return &[];
        }

        if scratch.len() < MAX_NEIGHBORS_PER_NODE {
            scratch.resize(MAX_NEIGHBORS_PER_NODE, 0);
        }

        let len = self.blocks.get_block(node_id, scratch);

        let mut acc = 0u32;
        for slot in &mut scratch[..len] {
            acc = acc.wrapping_add(*slot);
            *slot = acc;
        }
        &scratch[..len]
    }

    fn space_usage_bytes(&self) -> usize {
        postcard::to_allocvec(&self.blocks).unwrap().len()
            + self.empty_nodes.len() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(lists: &[&[u32]]) -> NeighborData {
        let mut data = Vec::new();
        let mut offsets = vec![0usize];
        for list in lists {
            data.extend_from_slice(list);
            offsets.push(data.len());
        }
        NeighborData {
            data: data.into_boxed_slice(),
            offsets: offsets.into_boxed_slice(),
        }
    }

    fn roundtrip<N: Neighbors + From<NeighborData>>(lists: &[&[u32]]) {
        let nd = build(lists);
        let n: N = N::from(nd.clone());
        for (i, list) in lists.iter().enumerate() {
            // A dirty scratch must not leak into the result: `get` owns the buffer's state.
            let mut scratch: Vec<u32> = vec![u32::MAX];
            assert_eq!(n.get(i, &mut scratch), *list, "node {i} mismatch");
        }
        assert_eq!(n.len(), nd.data.len());
    }

    #[test]
    fn plain_neighbors_roundtrip() {
        roundtrip::<PlainNeighbors>(&[&[1, 2, 3], &[], &[0, 5], &[7]]);
    }

    #[test]
    fn streamvbyte_neighbors_roundtrip() {
        roundtrip::<StreamVByteNeighbors>(&[&[1, 2, 3], &[], &[0, 5], &[7]]);
    }

    #[test]
    fn fixed_degree_neighbors_roundtrip() {
        roundtrip::<FixedDegreeNeighbors>(&[&[1, 2, 3], &[], &[0, 5], &[7]]);
    }

    /// The stride comes from the longest list, so a level whose nodes all have empty lists has
    /// no slots at all. Node count must still be reported from the input, not divided out of an
    /// empty buffer.
    #[test]
    fn fixed_degree_neighbors_all_lists_empty() {
        let n: FixedDegreeNeighbors = FixedDegreeNeighbors::from(build(&[&[], &[], &[]]));
        assert_eq!(n.n_nodes(), 3);
        assert_eq!(n.len(), 0);
        assert_eq!(n.space_usage_bytes(), 0);
        let mut scratch = Vec::new();
        for v in 0..3 {
            assert!(n.get(v, &mut scratch).is_empty());
        }
    }

    /// Padding is retained, so the footprint is `n_nodes * longest_list`, and a node's list is
    /// still handed out as a borrowed prefix rather than copied into the scratch.
    #[test]
    fn fixed_degree_neighbors_pads_to_longest_list() {
        let n: FixedDegreeNeighbors = FixedDegreeNeighbors::from(build(&[&[1, 2, 3], &[4], &[]]));
        assert_eq!(n.n_nodes(), 3);
        assert_eq!(n.len(), 4);
        assert_eq!(n.space_usage_bytes(), 3 * 3 * std::mem::size_of::<u32>());
        let mut scratch = vec![99u32];
        assert_eq!(n.get(1, &mut scratch), &[4]);
        assert_eq!(scratch, vec![99u32], "scratch must be left untouched");
    }

    #[test]
    fn streamvbyte_neighbors_large_block() {
        let list: Vec<u32> = (0..200).collect();
        roundtrip::<StreamVByteNeighbors>(&[&list]);
    }

    /// `get` decodes in place and leaves stale values past the returned length, so a scratch
    /// reused across nodes must never leak a longer previous list into a shorter one.
    #[test]
    fn streamvbyte_neighbors_reused_scratch() {
        let long: Vec<u32> = (0..200).collect();
        let lists: Vec<&[u32]> = vec![&long, &[1, 2, 3], &[], &[9]];
        let n = StreamVByteNeighbors::from(build(&lists));

        let mut scratch: Vec<u32> = Vec::new();
        for (i, list) in lists.iter().enumerate() {
            assert_eq!(n.get(i, &mut scratch), *list, "node {i} mismatch");
        }
    }
}
