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
    if cli.version {
        println!("nnc {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }

    let command = match &cli.command {
        Some(cmd) => cmd,
        None => {
            eprintln!("nnc: no command specified. Run `nnc --help` for usage.");
            return 1;
        }
    };
    match command {
        Command::Compile {
            source,
            emit,
            output,
            target_triple,
        } => run_compile(source, emit, output.as_deref(), target_triple.as_deref()),
        Command::Inspect { source } => run_inspect(source),
        Command::Import {
            source,
            output,
            weights_dir,
        } => run_import(source, output.as_deref(), weights_dir),
        Command::Test {
            source,
            input,
            expected,
            tolerance,
        } => run_test(source, input, expected, *tolerance),
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

    match memory::check_memory_limit(&mem_info, fr.model.config.memory_limit) {
        memory::MemoryCheck::Ok => {}
        memory::MemoryCheck::Warning(msg) => eprintln!("{}: {msg}", source.display()),
        memory::MemoryCheck::Error(msg) => {
            eprintln!("{}: {msg}", source.display());
            return 1;
        }
    }
    0
}

fn run_compile(
    source: &Path,
    emit_fmt: &EmitFormat,
    output: Option<&Path>,
    target_triple: Option<&str>,
) -> i32 {
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

    // Memory check
    let mem_info = memory::estimate_memory(&fr.model, &fr.shape_info.shapes);
    match memory::check_memory_limit(&mem_info, fr.model.config.memory_limit) {
        memory::MemoryCheck::Ok => {}
        memory::MemoryCheck::Warning(msg) => eprintln!("{}: {msg}", source.display()),
        memory::MemoryCheck::Error(msg) => {
            eprintln!("{}: {msg}", source.display());
            return 1;
        }
    }

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
    let model_name = fr
        .model
        .name
        .replace(|c: char| !c.is_ascii_alphanumeric(), "_");
    let opts = toolchain::CompileOptions {
        target: fr.model.config.target,
        target_triple,
    };
    if let Err(e) = toolchain::compile(
        &c_source,
        &c_header,
        emit_fmt,
        &output_path,
        &model_name,
        &opts,
    ) {
        eprintln!("nnc: {}", e.message);
        return 1;
    }

    eprintln!("nnc: wrote {}", output_path.display());
    0
}

fn run_import(source: &Path, output: Option<&Path>, weights_dir: &Path) -> i32 {
    let output_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| source.with_extension("nnl"));

    if let Err(e) = crate::import::import_onnx(source, &output_path, weights_dir) {
        eprintln!("nnc: import failed: {e}");
        return 1;
    }

    eprintln!(
        "nnc: imported {} -> {} (weights in {})",
        source.display(),
        output_path.display(),
        weights_dir.display()
    );
    0
}

fn run_test(source: &Path, input_path: &Path, expected_path: &Path, tolerance: f64) -> i32 {
    // Compile to a temp executable
    let tmp_dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nnc: failed to create temp directory: {e}");
            return 1;
        }
    };
    let exe_path = tmp_dir.path().join("nnc_test_binary");

    let compile_code = run_compile(source, &EmitFormat::Exe, Some(&exe_path), None);
    if compile_code != 0 {
        return compile_code;
    }

    // Read input .npy
    let input_data = match read_npy_f32(input_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nnc: cannot read input: {e}");
            return 1;
        }
    };

    // Read expected .npy
    let expected_data = match read_npy_f32(expected_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("nnc: cannot read expected output: {e}");
            return 1;
        }
    };

    // Run inference
    let input_bytes: Vec<u8> = input_data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let output = match std::process::Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        }) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("nnc: failed to run compiled binary: {e}");
            return 1;
        }
    };

    if !output.status.success() {
        eprintln!(
            "nnc: inference binary failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return 1;
    }

    // Parse output
    let output_data: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    // Compare
    if output_data.len() != expected_data.len() {
        eprintln!(
            "nnc: output size mismatch: got {} elements, expected {}",
            output_data.len(),
            expected_data.len()
        );
        return 1;
    }

    let mut max_diff: f64 = 0.0;
    let mut fail_count = 0;
    for (i, (got, exp)) in output_data.iter().zip(expected_data.iter()).enumerate() {
        let diff = (*got as f64 - *exp as f64).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance {
            if fail_count < 10 {
                eprintln!("  mismatch at [{i}]: got {got:.8}, expected {exp:.8}, diff {diff:.2e}");
            }
            fail_count += 1;
        }
    }

    if fail_count > 0 {
        if fail_count > 10 {
            eprintln!("  ... and {} more mismatches", fail_count - 10);
        }
        eprintln!(
            "FAIL: {fail_count}/{} elements exceed tolerance {tolerance:.1e} (max diff: {max_diff:.2e})",
            output_data.len()
        );
        return 1;
    }

    eprintln!(
        "PASS: {}/{} elements within tolerance {tolerance:.1e} (max diff: {max_diff:.2e})",
        output_data.len(),
        output_data.len()
    );
    0
}

fn read_npy_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
    let npy = npyz::NpyFile::new(&bytes[..])
        .map_err(|e| format!("cannot parse `{}`: {e}", path.display()))?;
    npy.into_vec::<f32>()
        .map_err(|e| format!("cannot read float32 data from `{}`: {e}", path.display()))
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
        EmitFormat::C => PathBuf::from(format!("{stem}.c")),
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
        println!("Model: {} ({version_str})", model.name);
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
        let params = mem_info.layer_params.get(&layer.id).copied().unwrap_or(0);

        println!(
            "{:<16}{:<12}{:<18}{:>8}",
            layer.id,
            type_name,
            shape_str,
            format_number(params)
        );
    }

    println!("{}", "─".repeat(54));
    println!("Total params:    {}", format_number(mem_info.total_params));
    println!("Weight memory:   {}", format_bytes(mem_info.weight_bytes));
    println!(
        "Workspace:       {} (static buffer)",
        format_bytes(mem_info.workspace_bytes)
    );
    println!("Total memory:    {}", format_bytes(mem_info.total_bytes));
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
