fn main() {
    println!(
        "cargo:rustc-link-search=native={}",
        std::env::current_dir()
            .unwrap()
            .parent()
            .unwrap()
            .display()
    );
    println!("cargo:rustc-link-lib=static=simple_mlp");
    println!("cargo:rustc-link-lib=dylib=m");
}
