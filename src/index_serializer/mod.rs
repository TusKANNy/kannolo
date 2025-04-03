use std::{fs, io::Error, path::PathBuf};

use serde::{de::DeserializeOwned, Serialize};

pub struct IndexSerializer {
    // We could have a `mode` field here that determines which serializer is used.
}

impl IndexSerializer {
    pub fn save_index<T: Serialize>(filename: &str, index: &T) -> Result<(), Error> {
        let filepath = PathBuf::from(filename);
        let serialized = bincode::serialize(&index).unwrap();
        fs::write(filepath, serialized)
    }

    pub fn load_index<T: DeserializeOwned>(filename: &str) -> T {
        let filepath = PathBuf::from(filename);
        let serialized: Vec<u8> = fs::read(filepath).unwrap();
        let index = bincode::deserialize::<T>(&serialized).unwrap();
        index
    }
}
