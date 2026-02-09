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

pub fn create_visited_set(dataset_size: usize, ef: usize) -> Box<dyn VisitedSet> {
    if dataset_size <= 2_000_000 || (dataset_size <= 10_000_000 && ef >= 400) {
        Box::new(BitVec::repeat(false, dataset_size))
    } else {
        Box::new(HashSet::with_capacity_and_hasher(
            200 + 32 * ef,
            BuildNoHashHasher::default(),
        ))
    }
}
