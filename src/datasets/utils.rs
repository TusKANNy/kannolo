use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
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

read_numpy_flatten!(read_numpy_f32_flatten_1d, Array1<f32>, f32, Ix2);
read_numpy_flatten!(read_numpy_f32_flatten_2d, Array2<f32>, f32, Ix2);
read_numpy_flatten!(read_numpy_u8_flatten, Array1<u8>, u8, Ix1);
read_numpy_flatten!(read_numpy_u32_flatten, Array2<u32>, u32, Ix2);

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
