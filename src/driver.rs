use std::path::{Path, PathBuf};

use crate::cli::{Cli, Command, EmitFormat};
use crate::codegen::{emit, toolchain};
use crate::diag::NncError;
use crate::ir::graph::{self, GraphInfo};
use crate::ir::lower;
use crate::ir::model::Model;
use crate::sema::{memory, shapes, validate};
use crate::syntax::{lexer, parser};
use crate::weights;

/// Result of the frontend pipeline (lex → parse → lower → validate → graph → shapes).
struct FrontendResult {
    model: Model,
    graph_info: GraphInfo,
    shape_info: shapes::ShapeInfo,
}

pub fn run(cli: &Cli) -> i32 {
    match &cli.command {
        Command::Compile {
            source,
            emit,
            output,
            ..
        } => run_compile(source, emit, output.as_deref()),
        Command::Inspect { source } => run_inspect(source),
        Command::Test { source, .. } => {
            eprintln!("nnc: test not yet implemented for {:?}", source);
            1
        }
    }
}

/// Run the frontend pipeline shared by inspect and compile.
fn run_frontend(source: &Path) -> Result<FrontendResult, i32> {
    let filename = source.display().to_string();
    let content = std::fs::read_to_string(source).map_err(|e| {
        eprintln!("nnc: cannot read {filename}: {e}");
        1
    })?;

    let tokens = lexer::tokenize(&content).map_err(|e| {
        let err = NncError::lex(e.span, &filename, &content);
        eprintln!("{:?}", miette::Report::new(err));
        1
    })?;

    let file = parser::parse(&tokens, &content).map_err(|e| {
        let err = NncError::syntax(e.message, e.span, &filename, &content);
        eprintln!("{:?}", miette::Report::new(err));
        1
    })?;

    let model = lower::lower(&file).map_err(|e| {
        let err = NncError::syntax(e.message, e.span, &filename, &content);
        eprintln!("{:?}", miette::Report::new(err));
        1
    })?;

    let warnings = validate::validate(&model).map_err(|e| {
        eprintln!("{}: {}: {}", filename, e.code, e.message);
        1
    })?;
    for w in &warnings {
        eprintln!("{filename}: {w}");
    }

    let graph_info = graph::build_graph(&model).map_err(|e| {
        eprintln!("{}: {}: {}", filename, e.code, e.message);
        1
    })?;
    for w in &graph_info.warnings {
        eprintln!("{filename}: {w}");
    }

    let shape_info = shapes::infer_shapes(&model, &graph_info.topo_order).map_err(|e| {
        eprintln!("{}: {}: {}", filename, e.code, e.message);
        1
    })?;

    Ok(FrontendResult {
        model,
        graph_info,
        shape_info,
    })
}

fn run_inspect(source: &Path) -> i32 {
    let fr = match run_frontend(source) {
        Ok(r) => r,
        Err(code) => return code,
    };

    let mem_info = memory::estimate_memory(&fr.model, &fr.shape_info.shapes);
    print_inspect(&fr.model, &fr.shape_info, &mem_info);
    0
}

fn run_compile(source: &Path, emit_fmt: &EmitFormat, output: Option<&Path>) -> i32 {
    let fr = match run_frontend(source) {
        Ok(r) => r,
        Err(code) => return code,
    };

    // Load weights
    let weight_set = match weights::load_and_validate(&fr.model, &fr.shape_info) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{}: {}: {}", source.display(), e.code, e.message);
            return 1;
        }
    };

    // Generate C source and header
    let c_header = emit::emit_header(&fr.model, &fr.shape_info);
    let c_source = emit::emit_source(
        &fr.model,
        &fr.shape_info,
        &weight_set,
        &fr.graph_info.topo_order,
    );

    // Determine output path
    let default_output = default_output_path(source, emit_fmt);
    let output_path = output.map(|p| p.to_path_buf()).unwrap_or(default_output);

    // Invoke toolchain
    let model_name = fr.model.name.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    if let Err(e) = toolchain::compile(&c_source, &c_header, emit_fmt, &output_path, &model_name) {
        eprintln!("nnc: {}", e.message);
        return 1;
    }

    eprintln!("nnc: wrote {}", output_path.display());
    0
}

fn default_output_path(source: &Path, emit: &EmitFormat) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    match emit {
        EmitFormat::Exe => PathBuf::from(stem),
        EmitFormat::Obj => PathBuf::from(format!("{stem}.o")),
        EmitFormat::Lib => PathBuf::from(format!("lib{stem}.a")),
        EmitFormat::Shared => PathBuf::from(format!("lib{stem}.so")),
        EmitFormat::Header => PathBuf::from(format!("{stem}.h")),
    }
}

fn print_inspect(
    model: &crate::ir::model::Model,
    shape_info: &shapes::ShapeInfo,
    mem_info: &memory::MemoryInfo,
) {
    let version_str = model
        .version
        .map(|v| format!("version {v}"))
        .unwrap_or_default();
    if !version_str.is_empty() {
        println!(
            "Model: {} ({version_str})",
            model.name
        );
    } else {
        println!("Model: {}", model.name);
    }
    println!(
        "Precision: {} | Target: {} | Batch: {}",
        model.config.precision, model.config.target, model.config.batch
    );
    println!();

    // Table header
    println!(
        "{:<16}{:<12}{:<18}{:>8}",
        "Layer", "Type", "Output Shape", "Params"
    );
    println!("{}", "─".repeat(54));

    for layer in &model.layers {
        let type_name = layer.kind.type_name();
        let shape_str = shape_info
            .shapes
            .get(&layer.id)
            .map(|s| format_shape(s))
            .unwrap_or_else(|| "?".to_string());
        let params = mem_info
            .layer_params
            .get(&layer.id)
            .copied()
            .unwrap_or(0);

        println!(
            "{:<16}{:<12}{:<18}{:>8}",
            layer.id, type_name, shape_str, format_number(params)
        );
    }

    println!("{}", "─".repeat(54));
    println!("Total params:    {}", format_number(mem_info.total_params));
    println!(
        "Weight memory:   {}",
        format_bytes(mem_info.weight_bytes)
    );
    println!(
        "Workspace:       {} (static buffer)",
        format_bytes(mem_info.workspace_bytes)
    );
}

fn format_shape(shape: &[usize]) -> String {
    let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

fn format_number(n: usize) -> String {
    if n == 0 {
        return "0".to_string();
    }
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}
