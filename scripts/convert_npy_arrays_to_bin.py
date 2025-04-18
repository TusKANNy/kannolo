import numpy as np
import struct
import os

def write_bin_file_from_npy(npy_path, output_fname):
    """
    Writes a binary file from pre-saved npy arrays with the following format:
      - 4 bytes: Unsigned 32-bit integer (little-endian) indicating the total number of vectors.
      - For each vector:
          - 4 bytes: Unsigned 32-bit integer (little-endian) representing the number of nonzero components.
          - Next (4 * n) bytes: Array of component indices stored as unsigned 32-bit integers (little-endian).
          - Following (4 * n) bytes: Array of 32-bit float values (little-endian) for the nonzero components.
    
    Parameters:
        npy_path (str): Directory path containing components.npy, values.npy, and offsets.npy.
        output_fname (str): Output binary file name.
    """
    # Load the npy arrays.
    components = np.load(os.path.join(npy_path, "components.npy"))
    values = np.load(os.path.join(npy_path, "values.npy"))
    offsets = np.load(os.path.join(npy_path, "offsets.npy"))
    
    # The number of vectors is the length of offsets minus one.
    n_vecs = offsets.shape[0] - 1
    
    with open(output_fname, "wb") as f:
        # Write total number of vectors (u32 little-endian)
        f.write(struct.pack("<I", n_vecs))
        
        # Write each vector's data.
        for i in range(n_vecs):
            start = offsets[i]
            end = offsets[i+1]
            n = end - start
            
            # Write number of nonzero components for this vector.
            f.write(struct.pack("<I", n))
            
            # Write component indices (cast to unsigned 32-bit integers).
            comps = components[start:end].astype(np.uint32)
            f.write(comps.tobytes())
            
            # Write corresponding float values (as 32-bit floats).
            vals = values[start:end].astype(np.float32)
            f.write(vals.tobytes())

    print(f"Binary file saved to {output_fname}")

# Example usage:
# write_bin_file_from_npy("output_directory", "data.bin")

# Parse command line arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert npy arrays to binary file.")
    parser.add_argument("npy_path", type=str, help="Path to the directory containing npy files.")
    parser.add_argument("output_fname", type=str, help="Output binary file name.")
    
    args = parser.parse_args()
    
    write_bin_file_from_npy(args.npy_path, args.output_fname)
