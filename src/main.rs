mod cli;
mod diag;
mod driver;

mod codegen;
mod ir;
mod sema;
mod syntax;
mod weights;

fn main() {
    let cli = cli::parse();
    let exit_code = driver::run(&cli);
    std::process::exit(exit_code);
}
