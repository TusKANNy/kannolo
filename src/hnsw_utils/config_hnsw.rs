/// Represents the configuration parameters for the HNSW (Hierarchical Navigable Small World) algorithm.
///
/// The `ConfigHnsw` struct holds various parameters that control the behavior of the HNSW algorithm used for
/// approximate nearest neighbor search.
///  
/// Fields:
/// - `num_neighbors_per_vec`: The number of neighbors to keep per vector in the HNSW graph.
/// - `ef_construction`: The size of the dynamic candidate set during the construction phase.
/// - `ef_search`: The size of the candidate set for the search phase.
/// - `check_relative_distance`: If enabled, the search terminates early if a sufficient number of close candidates
///   have been found, avoiding unnecessary further exploration.
/// - `bounded_exploration`: If enabled, the search stops after `ef_search` steps. If disabled, the search continues
///   until no closer candidates remain in the queue.
pub struct ConfigHnsw {
    num_neighbors_per_vec: usize,
    ef_construction: usize,
    ef_search: usize,
}

/// A builder for constructing `ConfigHnsw` instances with customizable parameters.
/// This struct provides a flexible way to set configuration parameters for the HNSW (Hierarchical Navigable Small World)
/// algorithm.
/// After configuring the desired parameters, call `build()` to create an instance of `ConfigHnsw`.
pub struct ConfigHnswBuilder {
    num_neighbors: Option<usize>,
    ef_construction: Option<usize>,
    ef_search: Option<usize>,
    check_relative_distance: Option<bool>,
}

impl ConfigHnsw {
    /// Initializes a `ConfigHnswBuilder` with all parameters unset by default.
    ///
    /// The `new` function creates a `ConfigHnswBuilder` with all configuration parameters set to `None`.
    /// This allows for the customization of the HNSW (Hierarchical Navigable Small World) algorithm's configuration
    /// by setting specific parameters through the builder's methods.
    ///
    /// After setting the desired parameters, the `build()` method finalizes the configuration by creating a
    /// `ConfigHnsw` instance.
    ///
    /// # Returns
    /// A `ConfigHnswBuilder` instance with all parameters unset, ready for customization.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // Create a new builder instance
    /// let mut builder = ConfigHnsw::new();
    ///
    /// // Customize the configuration and build it
    /// let config = builder
    ///     .num_neighbors(16)
    ///     .ef_construction(200)
    ///     .build();
    /// ```
    pub fn new() -> ConfigHnswBuilder {
        ConfigHnswBuilder {
            num_neighbors: None,
            ef_construction: None,
            ef_search: None,
            check_relative_distance: None,
        }
    }

    /// **Sets the `ef_search` parameter.**
    ///
    /// The `ef_search` parameter controls the candidate heap size during the search process in HNSW.
    /// This heap holds potential nearest neighbors to the query vector, and its size is determined as
    /// the maximum of the `k` parameter (the number of results to return) and `ef_search`.
    ///
    /// Additionally, if `bounded_exploration` is enabled and `check_relative_distance` is disabled,
    /// `ef_search` also sets the maximum number of search steps. A search step consists of selecting
    /// a candidate vector from the heap and exploring its neighbors.
    ///
    /// # Parameters
    /// - `ef_search` (`usize`): The size of the candidate heap, must be at least 1.
    ///
    /// # Default
    /// The default value is 16.
    ///
    /// # Panics
    /// Panics if `ef_search` is set to a value less than 1.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // create config
    /// let mut config = ConfigHnsw::new().build();
    /// config.set_ef_search(12);
    /// ```
    pub fn set_ef_search(&mut self, ef_search: usize) {
        assert!(ef_search > 0, "The ef_search must be at least 1");
        self.ef_search = ef_search;
    }

    /// **Retrieves the value of the `ef_search` parameter.**
    ///
    /// The `ef_search` parameter controls the candidate heap size during the search process in HNSW.
    /// This heap holds potential nearest neighbors to the query vector, and its size is determined as
    /// the maximum of the `k` parameter (the number of results to return) and `ef_search`.
    ///
    /// Additionally, if `bounded_exploration` is enabled and `check_relative_distance` is disabled,
    /// `ef_search` also sets the maximum number of search steps. A search step consists of selecting
    /// a candidate vectorfrom the heap and exploring its neighbors.
    ///
    /// # Returns
    /// - `usize`: The current value of `ef_search`.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // Create a new HNSW configuration and get the `ef_search` value
    /// let config = ConfigHnsw::new().build();
    /// let ef_search = config.get_ef_search();
    /// assert_eq!(ef_search,16); // the default value is 16.
    /// ```
    pub fn get_ef_search(&self) -> usize {
        self.ef_search
    }

    /// **Retrieves the number of neighbors per vector.**
    ///
    /// This method returns the number of neighbors that each vector has in the HNSW index. This parameter
    /// affects the connectivity of vectors within the index. A higher number of neighbors generally
    /// improves the accuracy of nearest neighbor searches but increases the construction time and memory usage.
    ///
    /// # Returns
    /// - `usize`: The current number of neighbors per vector.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // Create a new HNSW configuration and get the number of neighbors per vector
    /// let config = ConfigHnsw::new().build();
    /// let num_neighbors = config.get_num_neighbors_per_vec();
    /// ```
    pub fn get_num_neighbors_per_vec(&self) -> usize {
        self.num_neighbors_per_vec
    }

    /// **Retrieves the `ef_construction` parameter.**
    ///
    /// This method returns the `ef_construction` value, which determines the size of the heap used
    /// during the construction phase of the HNSW index. A higher value generally leads to more accurate
    /// neighbor determination for each vector but may also increase the construction time.
    ///
    /// # Returns
    /// - `usize`: The current `ef_construction` value.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // Create a new HNSW configuration and get the `ef_construction` value
    /// let config = ConfigHnsw::new().build();
    /// let ef_construction = config.get_ef_construction();
    /// ```
    pub fn get_ef_construction(&self) -> usize {
        self.ef_construction
    }
}

impl ConfigHnswBuilder {
    /// **Sets the number of neighbors for each vector**
    ///
    /// This function configures the number of neighbors that each vector should have in the HNSW index.
    /// The provided `num_neighbors` parameter determines the connectivity of vectors within the index.
    /// A higher value generally results in increased accuracy at the cost of higher construction time
    /// and memory usage.
    ///
    /// # Parameters
    /// - `num_neighbors` (`usize`)
    ///
    /// # Defaults
    /// The default value is 32.
    ///
    /// # Panics
    /// This function will panic if the specified `num_neighbors` is less than 2
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // create config
    /// let mut config = ConfigHnsw::new().num_neighbors(16).build();
    /// ```
    pub fn num_neighbors(&mut self, num_neighbors: usize) -> &mut Self {
        assert!(
            num_neighbors >= 2,
            "The number of neighbors must be at least 2"
        );

        self.num_neighbors = Some(num_neighbors);
        self
    }
    /// **Sets the `ef_construction` parameter.**
    ///
    /// This parameter controls the size of the candidate set used during the construction of the HNSW index.
    /// Specifically, `ef_construction` determines the number of candidates considered when adding a new vector
    /// to the index. A larger value may improve the accuracy of the neighbors determined during construction,
    /// but may increase the construction time and memory usage.
    ///
    /// # Parameters
    /// - `ef_construction` (`usize`): The number of candidates to consider during the graph construction.
    ///
    /// # Defaults
    /// The default value is 40.
    ///
    /// # Panics
    /// This function will panic if `ef_construction` is set to less than 1.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // Create a new HNSW configuration and set `ef_construction`
    /// let mut config = ConfigHnsw::new().ef_construction(30).build();
    /// ```
    pub fn ef_construction(&mut self, ef_construction: usize) -> &mut Self {
        assert!(
            ef_construction > 0,
            "The ef_construction must be at least 1"
        );
        self.ef_construction = Some(ef_construction);
        self
    }

    /// **Sets the ef_search parameter**
    ///
    /// The `ef_search` parameter controls the candidate heap size during the search process in HNSW.
    /// This heap holds potential nearest neighbors to the query vector, and its size is determined as
    /// the maximum of the `k` parameter (the number of results to return) and `ef_search`.
    ///
    /// Additionally, if `bounded_exploration` is enabled and `check_relative_distance` is disabled,
    /// `ef_search` also sets the maximum number of search steps. A search step consists of selecting
    ///  a candidate vector from the heap and exploring its neighbors.
    ///
    /// # Parameters
    /// - `ef_search` (`usize`)
    ///
    /// # Defaults
    /// The default value is 16, which is used if this function is not called before calling `build()`.
    ///
    /// # Panics
    /// This function will panic if the specified `ef_search` is less than 1
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // create config
    /// let mut config = ConfigHnsw::new().ef_search(512).build();
    /// ```
    pub fn ef_search(&mut self, ef_search: usize) -> &mut Self {
        assert!(ef_search > 0, "The ef_search must be at least 1");
        self.ef_search = Some(ef_search);
        self
    }
    /// **Sets the check_relative_distance parameter**
    ///
    /// The `check_relative_distance` parameter determines how the search process handles the termination condition:
    ///
    /// - If `check_relative_distance` is set to `true`, the search will stop if the number of candidates
    ///   with distances less than the current node's distance is greater than or equal to `ef_search`.
    ///   This helps to ensure that the search terminates only when a sufficient number of close candidates have been found.
    /// - If set to `false`, the search will terminate after `ef_search` iterations, regardless of the number
    ///   of candidates or their distances.
    ///
    /// # Parameters
    /// - `check_relative_distance` (`bool`)
    ///
    /// # Defaults
    /// The default value is true, which is used if this function is not called before calling `build()`.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // create config
    /// let mut config = ConfigHnsw::new().chec_relative_distance(false).build();
    /// ```
    pub fn check_relative_distance(&mut self, check_relative_distance: bool) -> &mut Self {
        self.check_relative_distance = Some(check_relative_distance);
        self
    }

    /// Finalizes the configuration and constructs the ConfigHnsw struct.
    ///
    /// The `build` function concludes the setup of the config struct based on the specified parameters,
    /// creating an instance of `ConfigHnsw` ready for use. If certain values are not explicitly provided,
    /// default parameters are applied.
    ///
    /// # Defaults
    /// - `num_neighbors`: 32
    /// - `ef_construction`: 40
    /// - `ef_search`: 16
    /// - `bounded_exploration`: true
    /// - `check_relative_distace`: true
    ///
    /// # Returns
    /// A fully configured and constructed configHnsw.
    ///
    /// # Examples
    /// ```rust
    /// use struttura_kANNolo::hnsw_utils::config_hnsw::ConfigHnsw;
    ///
    /// // create config
    /// let mut config = ConfigHnsw::new().num_neighbors(16).ef_search(128).bounded_exploration(false).build();
    /// ```
    pub fn build(&mut self) -> ConfigHnsw {
        ConfigHnsw {
            num_neighbors_per_vec: self.num_neighbors.unwrap_or(32),
            ef_construction: self.ef_construction.unwrap_or(40),
            ef_search: self.ef_search.unwrap_or(16),
        }
    }
}

#[cfg(test)]
mod tests_confighsnw {
    use super::*;

    /// Tests partial configuration of `ConfigHnsw`.
    ///
    /// This test verifies that when only some parameters are set using the builder, the remaining parameters
    /// are correctly assigned their default values. Specifically, it sets `num_neighbors` and `ef_construction`
    /// while letting the other parameters use their defaults.
    #[test]
    fn test_partial_config_build() {
        // Set only some parameters and let others use defaults
        let config = ConfigHnsw::new()
            .num_neighbors(15)
            .ef_construction(100)
            .build();

        assert_eq!(config.get_num_neighbors_per_vec(), 15);
        assert_eq!(config.get_ef_construction(), 100);
        assert_eq!(config.get_ef_search(), 16); // default
    }

    /// Tests setting multiple parameters after building the configuration.
    ///
    /// This test verifies that configuration parameters can be modified after the initial build of `ConfigHnsw`.
    /// It ensures that both the initial builder settings and subsequent modifications are applied correctly.
    #[test]
    fn test_multiple_set_operations() {
        let mut config = ConfigHnsw::new()
            .num_neighbors(32)
            .ef_construction(200)
            .build();

        // Modify after build
        config.set_ef_search(60);

        assert_eq!(config.get_num_neighbors_per_vec(), 32);
        assert_eq!(config.get_ef_construction(), 200);
        assert_eq!(config.get_ef_search(), 60);
    }

    /// Tests `ConfigHnsw` with minimum valid values.
    ///
    /// This test verifies that the configuration can handle cases where the minimum valid values are provided
    /// for the various parameters, ensuring that the system behaves as expected without panicking.
    #[test]
    fn test_minimum_valid_values() {
        let config = ConfigHnsw::new()
            .num_neighbors(2) // Minimum valid value
            .ef_construction(1) // Minimum valid value
            .ef_search(1) // Minimum valid value
            .build();

        assert_eq!(config.get_num_neighbors_per_vec(), 2);
        assert_eq!(config.get_ef_construction(), 1);
        assert_eq!(config.get_ef_search(), 1);
    }

    #[test]
    #[should_panic(expected = "The number of neighbors must be at least 2")]
    /// Tests that `ConfigHnsw` panics when `num_neighbors` is set to an invalid value.
    ///
    /// This test checks that attempting to set `num_neighbors` below the minimum valid value causes the system
    /// to panic with the appropriate error message.
    fn test_num_neighbors_panic() {
        let _config = ConfigHnsw::new().num_neighbors(1).build(); // Should panic
    }

    /// Tests that `ConfigHnsw` panics when `ef_construction` is set to an invalid value.
    ///
    /// This test verifies that setting `ef_construction` below the minimum valid value causes the system
    /// to panic with the appropriate error message.
    #[test]
    #[should_panic(expected = "The ef_construction must be at least 1")]
    fn test_ef_construction_panic() {
        let _config = ConfigHnsw::new().ef_construction(0).build(); // Should panic
    }

    /// Tests `ConfigHnsw` with extreme values.
    ///
    /// This test checks that the system can handle extremely large values for the configuration parameters
    /// without crashing or producing incorrect results. It uses the maximum possible values for `usize`.
    #[test]
    fn test_extreme_values() {
        let config = ConfigHnsw::new()
            .num_neighbors(usize::MAX)
            .ef_construction(usize::MAX)
            .ef_search(usize::MAX)
            .build();

        assert_eq!(config.get_num_neighbors_per_vec(), usize::MAX);
        assert_eq!(config.get_ef_construction(), usize::MAX);
        assert_eq!(config.get_ef_search(), usize::MAX);
    }
}
