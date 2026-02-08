use half::f16;
use ndarray::{Array1, Array2, Array3};
use ndarray_npy::ReadNpyExt;
use npyz::NpyFile;
use std::fs::File;
use std::io::Result as IoResult;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

macro_rules! read_numpy_flatten {
    ($func_name:ident, $arr_type:ty, $elem_type:ty, $dim:ty) => {
        #[inline]
        pub fn $func_name(filepath: String) -> (Vec<$elem_type>, usize) {
            let reader = File::open(filepath).unwrap();
            let arr: $arr_type = <$arr_type>::read_npy(reader).unwrap();

            let second_dim = arr.shape()[1];

            let mut result = Vec::new();
            for row_nd in arr.rows() {
                let mut row = row_nd.to_vec();
                result.append(&mut row);
            }

            (result, second_dim)
        }
    };
}

read_numpy_flatten!(read_numpy_f32_flatten_1d, Array1<f32>, f32, unused);
read_numpy_flatten!(read_numpy_f32_flatten_2d, Array2<f32>, f32, unused);
read_numpy_flatten!(read_numpy_u8_flatten, Array1<u8>, u8, unused);
read_numpy_flatten!(read_numpy_u32_flatten, Array2<u32>, u32, unused);
read_numpy_flatten!(read_numpy_u64_flatten_1d, Array1<u64>, u64, unused);
read_numpy_flatten!(read_numpy_i64_flatten_1d, Array1<i64>, i64, unused);
read_numpy_flatten!(read_numpy_u16_flatten_2d, Array2<u16>, u16, unused);

// Specialized function for reading document lengths as usize
#[inline]
pub fn read_numpy_1d_usize(filepath: String) -> Vec<usize> {
    let reader = File::open(&filepath).unwrap();
    // Try to read as i64 first (most common format for document lengths)
    match Array1::<i64>::read_npy(reader) {
        Ok(arr) => {
            // Convert i64 to usize
            arr.into_raw_vec().into_iter().map(|x| x as usize).collect()
        }
        Err(_) => {
            // If i64 fails, try u64
            let reader2 = File::open(&filepath).unwrap();
            match Array1::<u64>::read_npy(reader2) {
                Ok(arr) => arr.into_raw_vec().into_iter().map(|x| x as usize).collect(),
                Err(_) => {
                    // If u64 fails, try i32
                    let reader3 = File::open(&filepath).unwrap();
                    match Array1::<i32>::read_npy(reader3) {
                        Ok(arr) => arr.into_raw_vec().into_iter().map(|x| x as usize).collect(),
                        Err(_) => {
                            // If i32 fails, try u32
                            let reader4 = File::open(&filepath).unwrap();
                            let arr: Array1<u32> = Array1::<u32>::read_npy(reader4).unwrap();
                            arr.into_raw_vec().into_iter().map(|x| x as usize).collect()
                        }
                    }
                }
            }
        }
    }
}

// Functions to read custom f16 binary format (supports 1D, 2D, 3D arrays)
// Binary format: ndim (u64) + shape[ndim] (u64 each) + data (f16 values)

#[inline]
pub fn read_f16_bin_1d(filepath: String) -> Vec<f16> {
    let mut file =
        File::open(&filepath).expect(&format!("Failed to open f16 binary file: {}", filepath));

    let mut buffer = [0u8; 8];

    // Read number of dimensions
    file.read_exact(&mut buffer).unwrap();
    let ndim = u64::from_le_bytes(buffer) as usize;

    if ndim != 1 {
        panic!(
            "Expected 1D array, got {}D array in file: {}",
            ndim, filepath
        );
    }

    // Read shape
    file.read_exact(&mut buffer).unwrap();
    let len = u64::from_le_bytes(buffer) as usize;

    // Read f16 data efficiently
    let total_bytes = len * 2; // 2 bytes per f16
    let mut raw_data = vec![0u8; total_bytes];
    file.read_exact(&mut raw_data).unwrap();

    // Convert bytes to f16 efficiently
    let mut f16_data = Vec::with_capacity(len);
    for chunk in raw_data.chunks_exact(2) {
        let u16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
        f16_data.push(f16::from_bits(u16_val));
    }

    f16_data
}

#[inline]
pub fn read_f16_bin_2d(filepath: String) -> (Vec<f16>, usize) {
    let mut file =
        File::open(&filepath).expect(&format!("Failed to open f16 binary file: {}", filepath));

    let mut buffer = [0u8; 8];

    // Read number of dimensions
    file.read_exact(&mut buffer).unwrap();
    let ndim = u64::from_le_bytes(buffer) as usize;

    if ndim != 2 {
        panic!(
            "Expected 2D array, got {}D array in file: {}",
            ndim, filepath
        );
    }

    // Read shape
    file.read_exact(&mut buffer).unwrap();
    let rows = u64::from_le_bytes(buffer) as usize;

    file.read_exact(&mut buffer).unwrap();
    let cols = u64::from_le_bytes(buffer) as usize;

    // Read f16 data efficiently in large chunks
    let total_elements = rows * cols;
    let total_bytes = total_elements * 2; // 2 bytes per f16

    println!(
        "Reading {} elements ({} GB) from binary file...",
        total_elements,
        total_bytes as f64 / 1e9
    );

    // Read all data at once
    let mut raw_data = vec![0u8; total_bytes];
    file.read_exact(&mut raw_data).unwrap();

    println!("Raw data read, converting to f16...");

    // Convert bytes to f16 efficiently
    let mut f16_data = Vec::with_capacity(total_elements);
    for chunk in raw_data.chunks_exact(2) {
        let u16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
        f16_data.push(f16::from_bits(u16_val));
    }

    (f16_data, cols)
}

#[inline]
pub fn read_f16_bin_3d(filepath: String) -> (Vec<f16>, usize, usize) {
    let mut file =
        File::open(&filepath).expect(&format!("Failed to open f16 binary file: {}", filepath));

    let mut buffer = [0u8; 8];

    // Read number of dimensions
    file.read_exact(&mut buffer).unwrap();
    let ndim = u64::from_le_bytes(buffer) as usize;

    if ndim != 3 {
        panic!(
            "Expected 3D array, got {}D array in file: {}",
            ndim, filepath
        );
    }

    // Read shape
    file.read_exact(&mut buffer).unwrap();
    let dim0 = u64::from_le_bytes(buffer) as usize;

    file.read_exact(&mut buffer).unwrap();
    let dim1 = u64::from_le_bytes(buffer) as usize;

    file.read_exact(&mut buffer).unwrap();
    let dim2 = u64::from_le_bytes(buffer) as usize;

    // Read f16 data efficiently in large chunks
    let total_elements = dim0 * dim1 * dim2;
    let total_bytes = total_elements * 2; // 2 bytes per f16

    // Read all data at once
    let mut raw_data = vec![0u8; total_bytes];
    file.read_exact(&mut raw_data).unwrap();

    // Convert bytes to f16 efficiently
    let mut f16_data = Vec::with_capacity(total_elements);
    for chunk in raw_data.chunks_exact(2) {
        let u16_val = u16::from_le_bytes([chunk[0], chunk[1]]);
        f16_data.push(f16::from_bits(u16_val));
    }

    // Return (flattened_data, total_dim_per_item, num_items)
    // where total_dim_per_item = dim1 * dim2
    (f16_data, dim1 * dim2, dim0)
}

// Legacy function that now uses the new binary format for 2D arrays
#[inline]
pub fn read_numpy_f16_flatten_2d(filepath: String) -> (Vec<f16>, usize) {
    // Check if it's a .bin file (our custom format) or .npy file
    if filepath.ends_with(".bin") {
        read_f16_bin_2d(filepath)
    } else {
        // Fallback to old npyz-based implementation for .npy files
        let file =
            File::open(&filepath).expect(&format!("Failed to open numpy file: {}", filepath));

        match NpyFile::new(file) {
            Ok(npy_file) => {
                let shape = npy_file.shape();

                if shape.len() != 2 {
                    panic!("Expected 2D array, got {}D array", shape.len());
                }

                let second_dim = shape[1] as usize;

                // Read as f32 and convert to f16 (npyz doesn't directly support f16)
                let file2 = File::open(&filepath).unwrap();
                let npy_file2 = NpyFile::new(file2).unwrap();

                match npy_file2.into_vec::<f32>() {
                    Ok(f32_data) => {
                        let f16_data: Vec<f16> =
                            f32_data.into_iter().map(|x| f16::from_f32(x)).collect();

                        (f16_data, second_dim)
                    }
                    Err(e) => {
                        panic!("Could not read numpy file {} as f32: {}", filepath, e);
                    }
                }
            }
            Err(e) => {
                panic!("Failed to parse numpy file {}: {}", filepath, e);
            }
        }
    }
}

// Function to read 3D numpy arrays and flatten them to 2D
#[inline]
pub fn read_numpy_f32_flatten_3d(filepath: String) -> (Vec<f32>, usize, usize) {
    let reader = File::open(filepath).unwrap();
    let arr: Array3<f32> = Array3::<f32>::read_npy(reader).unwrap();

    let shape = arr.shape();
    let first_dim = shape[0]; // e.g., 323 queries
    let second_dim = shape[1]; // e.g., 32 vectors per query
    let third_dim = shape[2]; // e.g., 128 dimensions per vector

    // Flatten to a single vector, maintaining the order
    let flattened = arr.into_raw_vec();

    // Return (flattened_data, total_dim_per_item, num_items)
    // where total_dim_per_item = second_dim * third_dim
    (flattened, second_dim * third_dim, first_dim)
}

// Function to read 3D numpy u16 arrays and reinterpret as f16, then flatten them to 2D
#[inline]
pub fn read_numpy_u16_as_f16_flatten_3d(filepath: String) -> (Vec<f16>, usize, usize) {
    let reader = File::open(filepath).unwrap();
    let arr: Array3<u16> = Array3::<u16>::read_npy(reader).unwrap();

    let shape = arr.shape();
    let first_dim = shape[0]; // e.g., 323 queries
    let second_dim = shape[1]; // e.g., 32 vectors per query
    let third_dim = shape[2]; // e.g., 128 dimensions per vector

    // Flatten to a single vector of u16, maintaining the order
    let flattened_u16 = arr.into_raw_vec();

    // Reinterpret u16 values as f16
    let flattened_f16: Vec<f16> = flattened_u16
        .into_iter()
        .map(|u16_val| f16::from_bits(u16_val))
        .collect();

    // Return (flattened_data, total_dim_per_item, num_items)
    // where total_dim_per_item = second_dim * third_dim
    (flattened_f16, second_dim * third_dim, first_dim)
}

// Function to read 2D numpy u16 arrays and reinterpret as f16, then flatten them
#[inline]
pub fn read_numpy_u16_as_f16_flatten_2d(filepath: String) -> (Vec<f16>, usize) {
    let reader = File::open(filepath).unwrap();
    let arr: Array2<u16> = Array2::<u16>::read_npy(reader).unwrap();

    let second_dim = arr.shape()[1];

    // Flatten to a single vector of u16, maintaining the order
    let flattened_u16 = arr.into_raw_vec();

    // Reinterpret u16 values as f16
    let flattened_f16: Vec<f16> = flattened_u16
        .into_iter()
        .map(|u16_val| f16::from_bits(u16_val))
        .collect();

    (flattened_f16, second_dim)
}

// Function for loading with token pruning - skips pruned vectors during reading
#[inline]
pub fn read_numpy_u16_as_f16_flatten_2d_with_pruning(
    filepath: String,
    exclude_indices: &[usize],
    vector_dim: usize,
) -> (Vec<f16>, usize) {
    let reader = File::open(filepath).unwrap();
    let arr: Array2<u16> = Array2::<u16>::read_npy(reader).unwrap();

    let second_dim = arr.shape()[1];

    // Flatten to a single vector of u16, maintaining the order
    let flattened_u16 = arr.into_raw_vec();

    println!(
        "Loading and filtering {} elements with {} exclusions...",
        flattened_u16.len(),
        exclude_indices.len()
    );

    // Convert u16 to f16 while skipping pruned vectors
    let mut filtered_data = Vec::new();
    let mut exclude_idx = 0;
    let mut current_element_idx = 0;

    while current_element_idx < flattened_u16.len() {
        // Calculate which vector we're currently in
        let current_vector_idx = current_element_idx / vector_dim;

        // Check if current vector should be excluded
        let should_exclude = exclude_idx < exclude_indices.len()
            && exclude_indices[exclude_idx] == current_vector_idx;

        if should_exclude {
            // Skip this entire vector
            exclude_idx += 1;
            current_element_idx += vector_dim;
        } else {
            // Keep this vector - convert and add all its elements
            let vector_end = (current_element_idx + vector_dim).min(flattened_u16.len());
            for i in current_element_idx..vector_end {
                filtered_data.push(f16::from_bits(flattened_u16[i]));
            }
            current_element_idx = vector_end;
        }
    }

    println!("Filtered data size: {} elements", filtered_data.len());

    (filtered_data, second_dim)
}

#[cfg(test)]
mod tests {
    use half::f16;

    #[test]
    fn test_u16_to_f16_conversion() {
        // Test basic u16 to f16 conversion
        let u16_value: u16 = 0x3c00; // This represents 1.0 in f16 format
        let f16_value = f16::from_bits(u16_value);
        assert_eq!(f16_value, f16::from_f32(1.0));

        let u16_value: u16 = 0x4000; // This represents 2.0 in f16 format
        let f16_value = f16::from_bits(u16_value);
        assert_eq!(f16_value, f16::from_f32(2.0));

        // Test zero
        let u16_value: u16 = 0x0000; // This represents 0.0 in f16 format
        let f16_value = f16::from_bits(u16_value);
        assert_eq!(f16_value, f16::from_f32(0.0));

        // Test negative value
        let u16_value: u16 = 0xbc00; // This represents -1.0 in f16 format
        let f16_value = f16::from_bits(u16_value);
        assert_eq!(f16_value, f16::from_f32(-1.0));
    }

    #[test]
    fn test_f16_memory_efficiency() {
        // Demonstrate that f16 uses half the memory of f32
        assert_eq!(std::mem::size_of::<f16>(), 2);
        assert_eq!(std::mem::size_of::<f32>(), 4);

        // For a million elements
        let num_elements = 1_000_000;
        let f32_bytes = num_elements * std::mem::size_of::<f32>();
        let f16_bytes = num_elements * std::mem::size_of::<f16>();

        assert_eq!(f32_bytes, 4_000_000);
        assert_eq!(f16_bytes, 2_000_000);
        assert_eq!(f32_bytes / f16_bytes, 2); // f32 uses exactly 2x memory
    }
}

macro_rules! read_vecs_file {
    ($fname:expr, $elem_type:ty, $from_le_bytes:expr) => {{
        let path = Path::new($fname);
        let f = File::open(path)?;
        let f_size = f.metadata().unwrap().len() as usize;

        let mut br = BufReader::new(f);

        let mut buffer_d = [0u8; std::mem::size_of::<u32>()];
        let mut buffer = [0u8; std::mem::size_of::<$elem_type>()];

        br.read_exact(&mut buffer_d)?;
        let d = u32::from_le_bytes(buffer_d) as usize;

        let n_rows = f_size / (d * std::mem::size_of::<$elem_type>() + 4);
        let mut data = Vec::with_capacity(n_rows * d);

        for row in 0..n_rows {
            if row != 0 {
                br.read_exact(&mut buffer_d)?;
            }
            for _ in 0..d {
                br.read_exact(&mut buffer)?;
                data.push($from_le_bytes(buffer));
            }
        }

        Ok((data, d, n_rows))
    }};
}

#[inline]
pub fn read_fvecs_file(fname: &str) -> IoResult<(Vec<f32>, usize, usize)> {
    read_vecs_file!(fname, f32, f32::from_le_bytes)
}

#[inline]
pub fn read_ivecs_file(fname: &str) -> IoResult<(Vec<u32>, usize, usize)> {
    read_vecs_file!(fname, u32, u32::from_le_bytes)
}

pub fn read_tsv_file(fname: &str) -> IoResult<(Vec<Vec<u32>>, usize)> {
    let path = Path::new(fname);
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data: Vec<Vec<u32>> = Vec::new();
    let mut current_query_id = None;
    let mut current_query_docs = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split('\t').collect();

        if parts.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid number of columns found in line: {}", line),
            ));
        }

        let query_id = parts[0].parse::<u32>().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Parse error: {:?}", e))
        })?;
        let document_id = parts[1].parse::<u32>().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("Parse error: {:?}", e))
        })?;

        if current_query_id.is_none() {
            current_query_id = Some(query_id);
        }

        if current_query_id.unwrap() != query_id {
            data.push(current_query_docs);
            current_query_docs = Vec::new();
            current_query_id = Some(query_id);
        }

        current_query_docs.push(document_id);
    }

    if !current_query_docs.is_empty() {
        data.push(current_query_docs);
    }

    let dimension = data.iter().map(|v| v.len()).max().unwrap_or(0);

    Ok((data, dimension))
}
