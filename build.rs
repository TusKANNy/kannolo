use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(use_cblas)");

    if cblas_is_available() {
        println!("cargo:rustc-cfg=use_cblas");
        println!("Detected MKL/CBLAS");
    } else {
        println!("Did not detect MKL/CBLAS");
    }
}

fn cblas_is_available() -> bool {
    if let Ok(mklroot) = std::env::var("MKLROOT") {
        if check_mkl_directory(&mklroot) {
            return true;
        }
    } else {
        println!("MKLROOT is not set");
        // MKLROOT is not set
        return false;
    }

    println!("Checking for MKL/CBLAS");
    let base_path = PathBuf::from("/opt/intel/oneapi/mkl/");
    if base_path.exists() {
        if let Ok(entries) = fs::read_dir(base_path) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_dir() && check_mkl_directory(&path.to_string_lossy()) {
                    println!("using blas!");
                    return true;
                }
            }
        }
    }

    false
}

fn check_mkl_directory(mklroot: &str) -> bool {
    let lib_path = PathBuf::from(mklroot).join("lib/intel64/libmkl_rt.so");
    Path::new(&lib_path).exists()
}
