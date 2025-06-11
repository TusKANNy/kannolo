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
