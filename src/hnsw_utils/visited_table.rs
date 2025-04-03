/// A `VisitedTable` is used to track the visit status of elements in a graph or other data structures.
/// It keeps track of whether an element has been visited during a specific iteration or search process.
///
/// This structure is useful for avoiding revisiting nodes or elements within a single iteration of an algorithm,
/// and it efficiently resets the visit status when the iteration number overflows.
///
/// # Fields
///
/// - `visit_status`: A boxed slice of `u8` values, where each entry corresponds to an element's visit status.
///   The value indicates the iteration during which the element was visited.
/// - `iteration`: An `u8` counter that tracks the current iteration of the process. It starts at 1 and increments
///   with each iteration. When it reaches `u8::MAX - 1`, it wraps around to 1, resetting the visit status of all elements.

pub struct VisitedTable {
    visit_status: Box<[u8]>,
    iteration: u8,
}

impl VisitedTable {
    /// Creates a new `VisitedTable` with a given size.
    ///
    /// # Parameters
    ///
    /// - `n`: The number of elements to track. This determines the size of the `visit_status` vector.
    ///
    /// # Returns
    ///
    /// A `VisitedTable` instance initialized with all elements unvisited.
    ///
    /// # Example
    ///
    /// ```
    /// use struttura_kANNolo::hnsw_utils::visited_table::VisitedTable;
    ///
    /// let table = VisitedTable::new(10);
    /// assert_eq!(table.get_visit_status(), &[0; 10]);
    /// ```
    pub fn new(n: usize) -> Self {
        Self {
            visit_status: vec![0; n].into_boxed_slice(),
            iteration: 1,
        }
    }
    /// Returns a reference to the internal `visit_status` vector.
    ///
    /// # Returns
    ///
    /// A slice of `u8` values representing the visit status of each element.
    ///
    /// # Example
    ///
    /// ```
    /// use struttura_kANNolo::hnsw_utils::visited_table::VisitedTable;
    ///
    /// let table = VisitedTable::new(5);
    /// let status = table.get_visit_status();
    /// assert_eq!(status.len(), 5);
    /// ```
    #[inline]
    pub fn get_visit_status(&self) -> &[u8] {
        &self.visit_status
    }
    /// Marks an element as visited in the current iteration.
    ///
    /// # Parameters
    ///
    /// - `id`: The index of the element to mark as visited.
    ///
    /// # Panics
    ///
    /// This function will panic if `id` is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use struttura_kANNolo::hnsw_utils::visited_table::VisitedTable;
    ///
    /// let mut table = VisitedTable::new(5);
    /// table.set(2);
    /// assert_eq!(table.get(2), true);
    /// ```
    #[inline]
    pub fn set(&mut self, id: usize) {
        self.visit_status[id] = self.iteration;
    }
    /// Checks if an element was visited in the current iteration.
    ///
    /// # Parameters
    ///
    /// - `id`: The index of the element to check.
    ///
    /// # Returns
    ///
    /// `true` if the element was visited in the current iteration, `false` otherwise.
    ///
    /// # Panics
    ///
    /// This function will panic if `id` is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use struttura_kANNolo::hnsw_utils::visited_table::VisitedTable;
    ///
    /// let mut table = VisitedTable::new(5);
    /// assert_eq!(table.get(2), false);
    /// table.set(2);
    /// assert_eq!(table.get(2), true);
    /// ```
    #[inline]
    pub fn get(&self, id: usize) -> bool {
        self.visit_status[id] == self.iteration
    }

    /// Advances to the next iteration, resetting visit status if necessary.
    ///
    /// When the iteration counter reaches `u8::MAX - 1`, it wraps around to 1 and resets all visit statuses to 0.
    /// This ensures that the `VisitedTable` can continue to be used without manual reset.
    ///
    /// # Example
    ///
    /// ```
    /// use struttura_kANNolo::hnsw_utils::visited_table::VisitedTable;
    ///
    /// let mut table = VisitedTable::new(3);
    /// table.set(1);
    /// assert_eq!(table.get(1), true);
    /// table.advance();
    /// assert_eq!(table.get(1), false);
    /// ```
    pub fn advance(&mut self) {
        self.iteration += 1;

        if self.iteration == u8::MAX - 1 {
            self.visit_status.fill(0);
            self.iteration = 1;
        }
    }
}
#[cfg(test)]
mod tests_visited_table {
    use super::*;

    /// Tests the creation of a new `VisitedTable`.
    ///
    /// This test verifies that a new `VisitedTable` is initialized correctly with a specified
    /// number of elements. It ensures that all visit statuses are initially zero and the iteration
    /// count starts at 1.
    #[test]
    fn test_new_visited_table() {
        let table = VisitedTable::new(10);
        assert_eq!(table.get_visit_status(), &[0; 10]);
        assert_eq!(table.iteration, 1);
    }

    /// Tests setting and getting visit status.
    ///
    /// This test checks the functionality of the `set` and `get` methods. It verifies that setting
    /// a visit status for a specific index works as expected, and that other indices remain unaffected.
    #[test]
    fn test_set_and_get() {
        let mut table = VisitedTable::new(5);
        assert_eq!(table.get(2), false); // Initially, the status should be false
        table.set(2);
        assert_eq!(table.get(2), true); // After setting, the status should be true
        assert_eq!(table.get(3), false); // Other indices should remain unaffected
    }

    /// Tests advancing the iteration count.
    ///
    /// This test ensures that advancing the iteration resets the status of previously set elements
    /// to false, as they should no longer be considered visited after the iteration changes.
    #[test]
    fn test_advance_iteration() {
        let mut table = VisitedTable::new(3);
        table.set(1);
        assert_eq!(table.get(1), true);

        table.advance(); // Advance the iteration
        assert_eq!(table.get(1), false); // After advancing, the status should be false
    }

    /// Tests setting and advancing multiple times.
    ///
    /// This test verifies that setting a visit status and advancing the iteration multiple times
    /// works correctly. It ensures that after an iteration is advanced, previously set statuses are
    /// reset, and statuses can be set again for the new iteration.
    #[test]
    fn test_set_multiple_times() {
        let mut table = VisitedTable::new(5);

        table.set(3);
        assert_eq!(table.get(3), true);

        table.advance();
        assert_eq!(table.get(3), false); // Status should be false after advancing

        table.set(3);
        assert_eq!(table.get(3), true); // Status should be true after setting again
    }

    /// Tests setting a visit status out of bounds.
    ///
    /// This test ensures that the `set` method panics when trying to set a visit status at an index
    /// that is out of bounds of the allocated array.
    #[test]
    #[should_panic]
    fn test_set_out_of_bounds() {
        let mut table = VisitedTable::new(2);
        table.set(5); // Should panic due to out of bounds
    }

    /// Tests getting a visit status out of bounds.
    ///
    /// This test ensures that the `get` method panics when trying to retrieve a visit status at an
    /// index that is out of bounds of the allocated array.
    #[test]
    #[should_panic]
    fn test_get_out_of_bounds() {
        let table = VisitedTable::new(2);
        let _ = table.get(5); // Should panic due to out of bounds
    }

    /// Tests advancing the iteration count when it overflows.
    ///
    /// This test checks the behavior of the `advance` method when the iteration count is close to
    /// the maximum value of `u8`. It ensures that the iteration count wraps around to 1 and that
    /// the visit statuses are reset appropriately.
    #[test]
    fn test_advance_overflow() {
        let mut table = VisitedTable::new(2);

        table.iteration = u8::MAX - 2;
        table.advance();
        assert_eq!(table.iteration, 1);

        // The visit status should be reset
        assert_eq!(table.get_visit_status(), vec![0; 2].as_slice());
    }
}
