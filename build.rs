fn main() {
    // On macOS, Python symbols are resolved at runtime when the extension is loaded.
    // The Apple linker requires explicit permission to leave them undefined.
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
