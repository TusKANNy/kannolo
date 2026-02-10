use bitvec::prelude::BitVec;
use nohash_hasher::BuildNoHashHasher;
use std::collections::HashSet;

pub trait VisitedSet {
    fn insert(&mut self, val: usize) -> bool;
    fn contains(&self, val: usize) -> bool;
}

impl VisitedSet for HashSet<usize, BuildNoHashHasher<usize>> {
    fn insert(&mut self, val: usize) -> bool {
        self.insert(val)
    }

    fn contains(&self, val: usize) -> bool {
        self.contains(&val)
    }
}

impl VisitedSet for BitVec {
    fn insert(&mut self, val: usize) -> bool {
        if val >= self.len() {
            self.resize(val + 1, false);
        }
        let already = self[val];
        self.set(val, true);
        !already
    }

    fn contains(&self, val: usize) -> bool {
        val < self.len() && self[val]
    }
}

/// Stack-allocated visited set that chooses between BitVec or HashSet at construction time.
/// Eliminates vtable indirection on the hot path by using an enum instead of Box<dyn VisitedSet>.
pub enum VisitedSetImpl {
    BitVec(BitVec),
    HashSet(HashSet<usize, BuildNoHashHasher<usize>>),
}

impl VisitedSet for VisitedSetImpl {
    #[inline]
    fn insert(&mut self, val: usize) -> bool {
        match self {
            VisitedSetImpl::BitVec(bv) => {
                if val >= bv.len() {
                    bv.resize(val + 1, false);
                }
                let already = bv[val];
                bv.set(val, true);
                !already
            }
            VisitedSetImpl::HashSet(hs) => hs.insert(val),
        }
    }

    #[inline]
    fn contains(&self, val: usize) -> bool {
        match self {
            VisitedSetImpl::BitVec(bv) => val < bv.len() && bv[val],
            VisitedSetImpl::HashSet(hs) => hs.contains(&val),
        }
    }
}

/// Creates a visited set optimized for the given dataset and search parameters.
///
/// Returns the appropriate variant based on dataset size and ef:
/// - BitVec for smaller datasets or high-ef searches (dense access pattern)
/// - HashSet for larger datasets with small ef (sparse access pattern)
///
/// The enum-based approach eliminates vtable indirection on hot paths.
pub fn create_visited_set(dataset_size: usize, ef: usize) -> VisitedSetImpl {
    if dataset_size <= 2_000_000 || (dataset_size <= 10_000_000 && ef >= 400) {
        VisitedSetImpl::BitVec(BitVec::repeat(false, dataset_size))
    } else {
        VisitedSetImpl::HashSet(HashSet::with_capacity_and_hasher(
            200 + 32 * ef,
            BuildNoHashHasher::default(),
        ))
    }
}
