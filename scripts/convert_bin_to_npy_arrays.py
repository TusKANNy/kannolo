import numpy as np
import struct

def read_bin_file_limit(input_fname, output_path, limit=None):
    """
    Reads a binary file where the format is:
      - u32: total number of sparse vectors (n_vecs)
      - For each vector:
          - u32: number of nonzero components (n)
          - n × u32: component indices (to be cast to int32)
          - n × f32: corresponding float values

    This function saves the data into three NumPy arrays:
      - components: a NumPy array of int32 indices
      - values: a NumPy array of float32 values
      - offsets: a NumPy array (length n_vecs+1) where offsets[i] is the starting
                 index in components/values for the i-th vector, so that the i-th vector's
                 data is components[offsets[i]:offsets[i+1]] and similarly for values.

    Parameters:
        input_fname (str): Path to the binary file.
        output_path (str): Path to save the output NumPy files.
        limit (int or None): If provided, limits the number of vectors read to the given value.

    Returns:
        None: The function saves the components, values, and offsets to the specified output path.
    """
    with open(input_fname, 'rb') as f:
        # Read the total number of vectors (u32 little-endian)
        n_vecs = struct.unpack('<I', f.read(4))[0]
        if limit is not None:
            n_vecs = min(limit, n_vecs)
        
        # Initialize lists to hold the parts of the data.
        components_list = []
        values_list = []
        offsets = [0]
        
        # Process each vector.
        for _ in range(n_vecs):
            # Read the number of nonzero components (u32 little-endian)
            n = struct.unpack('<I', f.read(4))[0]
            
            # Read the component indices (each stored as u32, then cast to uint16)
            comps_data = f.read(4 * n)
            # Use np.frombuffer to quickly convert the bytes to a numpy array,
            # then cast to uint16.
            comps = np.frombuffer(comps_data, dtype='<u4').astype(np.int32)
            
            # Read the corresponding float values (f32 little-endian)
            vals_data = f.read(4 * n)
            vals = np.frombuffer(vals_data, dtype='<f4')
            
            # Append to our lists
            components_list.append(comps)
            values_list.append(vals)
            offsets.append(offsets[-1] + n)
        
        # Concatenate lists into single NumPy arrays.
        components = np.concatenate(components_list) if components_list else np.array([], dtype=np.int32)
        values = np.concatenate(values_list) if values_list else np.array([], dtype=np.float32)
        offsets = np.array(offsets, dtype=np.int64)
        
        # Save to a file
        np.save(output_path + "/components.npy", components)
        np.save(output_path + "/values.npy", values)
        np.save(output_path + "/offsets.npy", offsets)
        print(f"Saved components, values, and offsets to {output_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert binary file to NumPy arrays.")
    parser.add_argument("input_fname", type=str, help="Path to the input binary file.")
    parser.add_argument("output_path", type=str, help="Path to save the output NumPy files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of vectors read.")

    args = parser.parse_args()

    read_bin_file_limit(args.input_fname, args.output_path, args.limit)
    # Example usage:
    # read_bin_file_limit("data.bin", "output", limit=1000)
    # This will read the first 1000 vectors from "data.bin" and save them as
    # components.npy, values.npy, and offsets.npy in the "output" directory.
    # Note: The output directory must exist before running this script.
    # The script will create the files in the specified output directory.
    # Ensure the output directory exists.