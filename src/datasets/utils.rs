use crate::plain_quantizer::PlainQuantizer;
use crate::{DenseDataset, DistanceType};
use anyhow::Result as AnyResult;
use hdf5::types::H5Type;
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
    read_vecs_file!(fname, f32, |bytes| f32::from_le_bytes(bytes))
}

#[inline]
pub fn read_ivecs_file(fname: &str) -> IoResult<(Vec<u32>, usize, usize)> {
    read_vecs_file!(fname, u32, |bytes| u32::from_le_bytes(bytes))
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

pub fn read_dataset(
    hdf5_path: &str,
    data_label: &str,
    query_label: &str,
    train_label: &str,
    groundtruth_label: &str,
    distance_type: DistanceType,
) -> AnyResult<(
    DenseDataset<PlainQuantizer<f32>, Vec<f32>>, // data
    DenseDataset<PlainQuantizer<f32>, Vec<f32>>, // query
    DenseDataset<PlainQuantizer<f32>, Vec<f32>>, // train
    DenseDataset<PlainQuantizer<u32>, Vec<u32>>, // gt
)> {
    fn read_data<T: H5Type>(path: &str, label: &str) -> AnyResult<(Vec<T>, usize)> {
        let file = hdf5::File::open(path)?;
        let dataset = file.dataset(label)?;
        let shape = dataset.shape();
        let data: Vec<T> = dataset.read_raw::<T>()?.into();

        Ok((data, shape[1]))
    }

    let (data, data_dim) = read_data(hdf5_path, data_label)?;
    let data_dataset = DenseDataset::from_vec(
        data,
        data_dim,
        PlainQuantizer::<f32>::new(data_dim, distance_type),
    );

    let (query, query_dim) = read_data(hdf5_path, query_label)?;
    let query_dataset = DenseDataset::from_vec(
        query,
        query_dim,
        PlainQuantizer::<f32>::new(query_dim, distance_type),
    );

    let (train, train_dim) = read_data(hdf5_path, train_label)?;
    let train_dataset = DenseDataset::from_vec(
        train,
        train_dim,
        PlainQuantizer::<f32>::new(train_dim, distance_type),
    );

    let (groundtruth, gt_dim) = read_data(hdf5_path, groundtruth_label)?;
    let groundtruth_dataset = DenseDataset::from_vec(
        groundtruth,
        gt_dim,
        PlainQuantizer::<u32>::new(gt_dim, distance_type),
    );

    Ok((
        data_dataset,
        query_dataset,
        train_dataset,
        groundtruth_dataset,
    ))
}
