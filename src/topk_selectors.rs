pub trait OnlineTopKSelector {
    fn new(k: usize) -> Self;

    fn push(&mut self, distance: f32);

    fn push_with_id(&mut self, distance: f32, id: usize);

    fn extend(&mut self, distances: &[f32]);

    fn topk(&self) -> Vec<(f32, usize)>;
}

pub mod topk_heap;
pub use topk_heap::TopkHeap;
