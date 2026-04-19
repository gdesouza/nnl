use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nnc", about = "NNL compiler — compile neural network definitions to native binaries")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Compile an NNL model to a native artifact
    Compile {
        /// Path to the .nnl source file
        source: PathBuf,

        /// Output format
        #[arg(long, default_value = "exe")]
        emit: EmitFormat,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Target triple for cross-compilation
        #[arg(long)]
        target_triple: Option<String>,
    },

    /// Inspect a model: print graph, shapes, parameter counts, and memory estimates
    Inspect {
        /// Path to the .nnl source file
        source: PathBuf,
    },

    /// Import an ONNX model into NNL format
    Import {
        /// Path to the ONNX model file
        source: PathBuf,

        /// Output .nnl file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Directory to write extracted weight files
        #[arg(long, default_value = "./weights")]
        weights_dir: PathBuf,
    },

    /// Test a compiled model against known input/output pairs
    Test {
        /// Path to the .nnl source file
        source: PathBuf,

        /// Path to input tensor (.npy)
        #[arg(long)]
        input: PathBuf,

        /// Path to expected output tensor (.npy)
        #[arg(long)]
        expected: PathBuf,

        /// Element-wise tolerance for comparison
        #[arg(long, default_value = "1e-5")]
        tolerance: f64,
    },
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum EmitFormat {
    Exe,
    Obj,
    Lib,
    Shared,
    Header,
}

pub fn parse() -> Cli {
    Cli::parse()
}
