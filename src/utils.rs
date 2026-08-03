use serde::{Deserialize, Serialize};
use tantivy_bitpacker::{BitPacker, BitUnpacker};

/// A compact, bit-packed array of `usize` values.
///
/// Values are stored using the minimum number of bits required to represent the
/// maximum value in the array, via `tantivy_bitpacker`. Suitable for large ID
/// mappings where memory footprint matters: for a 1M-vector index the stored IDs
/// need 20 bits each rather than the 64 a `usize` would occupy.
#[derive(Clone, Serialize, Deserialize, Default)]
pub(crate) struct CompactArray {
    data: Box<[u8]>,
    bit_width: u8,
    logical_len: usize,
}

impl CompactArray {
    #[inline]
    pub(crate) fn get(&self, index: usize) -> usize {
        assert!(index < self.logical_len, "index out of bounds: {index}");
        BitUnpacker::new(self.bit_width).get(index as u32, &self.data) as usize
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.logical_len
    }

    pub(crate) fn to_vec(&self) -> Vec<usize> {
        let unpacker = BitUnpacker::new(self.bit_width);
        (0..self.logical_len)
            .map(|i| unpacker.get(i as u32, &self.data) as usize)
            .collect()
    }

    #[inline]
    pub(crate) fn space_usage_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<u8>() + std::mem::size_of::<usize>()
    }
}

impl From<Vec<usize>> for CompactArray {
    fn from(values: Vec<usize>) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let logical_len = values.len();
        let max_value = values.iter().copied().max().unwrap();
        let max_value_u32 = u32::try_from(max_value)
            .expect("CompactArray value does not fit in u32 required by bitpacking");
        let bit_width = ((u32::BITS - max_value_u32.leading_zeros()) as u8).max(1);

        let mut packed = Vec::new();
        let mut packer = BitPacker::new();
        for value in values {
            let value_u32 = u32::try_from(value)
                .expect("CompactArray value does not fit in u32 required by bitpacking");
            packer
                .write(value_u32 as u64, bit_width, &mut packed)
                .unwrap();
        }
        packer.close(&mut packed).unwrap();

        Self {
            data: packed.into_boxed_slice(),
            bit_width,
            logical_len,
        }
    }
}

impl From<Box<[usize]>> for CompactArray {
    fn from(values: Box<[usize]>) -> Self {
        Self::from(values.into_vec())
    }
}

/// Inverts a permutation: given `p` where `p[old_id] = new_id`, returns `inv` where
/// `inv[new_id] = old_id`.
pub fn invert_mapping(p: &[usize]) -> Vec<usize> {
    let n = p.len();
    let mut inv = vec![0usize; n];

    for (i, &j) in p.iter().enumerate() {
        inv[j] = i;
    }

    inv
}

/// Validates that `p` is a permutation of `0..p.len()`: every value is in range and
/// appears exactly once.
pub fn validate_permutation(p: &[usize]) -> Result<(), String> {
    let n = p.len();
    let mut seen = vec![false; n];

    for (i, &x) in p.iter().enumerate() {
        if x >= n {
            return Err(format!(
                "invalid permutation: value {} at position {} out of range",
                x, i
            ));
        }
        if seen[x] {
            return Err(format!("invalid permutation: duplicate value {}", x));
        }
        seen[x] = true;
    }

    Ok(())
}

#[cfg(test)]
mod tests_permutation_helpers {
    use super::*;

    #[test]
    fn invert_mapping_round_trips() {
        let p = vec![2, 0, 3, 1];
        let inv = invert_mapping(&p);
        for (old_id, &new_id) in p.iter().enumerate() {
            assert_eq!(inv[new_id], old_id);
        }
    }

    #[test]
    fn validate_permutation_accepts_valid() {
        assert!(validate_permutation(&[2, 0, 3, 1]).is_ok());
        assert!(validate_permutation(&[]).is_ok());
    }

    #[test]
    fn validate_permutation_rejects_out_of_range() {
        assert!(validate_permutation(&[0, 5, 2]).is_err());
    }

    #[test]
    fn validate_permutation_rejects_duplicate() {
        assert!(validate_permutation(&[0, 1, 1]).is_err());
    }
}

#[cfg(test)]
mod tests_compact_array {
    use super::CompactArray;

    /// Round-trips every value: a packed array must read back exactly what went in.
    #[test]
    fn round_trips_values() {
        let values: Vec<usize> = vec![0, 1, 7, 255, 256, 1_000_000, 999_999];
        let packed = CompactArray::from(values.clone());

        assert_eq!(packed.len(), values.len());
        assert_eq!(packed.to_vec(), values);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(packed.get(i), expected, "mismatch at index {i}");
        }
    }

    /// A permutation of 0..n is the actual use case (inverse permutation / ids mapping).
    #[test]
    fn round_trips_a_permutation() {
        let n = 5000;
        let values: Vec<usize> = (0..n).map(|i| (i * 7919) % n).collect();
        let packed = CompactArray::from(values.clone());
        assert_eq!(packed.to_vec(), values);
    }

    /// The whole point: storing 20-bit IDs must cost far less than 64 bits each.
    #[test]
    fn packs_smaller_than_unpacked_usize() {
        let n = 100_000;
        let values: Vec<usize> = (0..n).collect();
        let packed = CompactArray::from(values);

        let unpacked_bytes = n * std::mem::size_of::<usize>();
        // 17 bits suffice for 99_999, so expect well under a third of the usize cost.
        assert!(
            packed.space_usage_bytes() < unpacked_bytes / 3,
            "packed {} bytes vs unpacked {} bytes",
            packed.space_usage_bytes(),
            unpacked_bytes
        );
    }

    #[test]
    fn handles_empty_and_single_element() {
        let empty = CompactArray::from(Vec::<usize>::new());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.to_vec(), Vec::<usize>::new());

        let single = CompactArray::from(vec![0usize]);
        assert_eq!(single.len(), 1);
        assert_eq!(single.get(0), 0);
    }

    #[test]
    fn survives_serde_round_trip() {
        let values: Vec<usize> = (0..1000).map(|i| (i * 13) % 1000).collect();
        let packed = CompactArray::from(values.clone());

        let bytes = postcard::to_allocvec(&packed).unwrap();
        let decoded: CompactArray = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.to_vec(), values);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_past_end_panics() {
        let packed = CompactArray::from(vec![1usize, 2, 3]);
        let _ = packed.get(3);
    }
}
