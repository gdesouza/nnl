use crate::cli::{Cli, Command};
use crate::diag::NncError;
use crate::ir::{graph, lower};
use crate::sema::{memory, shapes, validate};
use crate::syntax::{lexer, parser};

pub fn run(cli: &Cli) -> i32 {
    match &cli.command {
        Command::Compile { source, .. } => {
            eprintln!("nnc: compile not yet implemented for {:?}", source);
            1
        }
        Command::Inspect { source } => run_inspect(source),
        Command::Test { source, .. } => {
            eprintln!("nnc: test not yet implemented for {:?}", source);
            1
        }
    }
}

fn run_inspect(source: &std::path::Path) -> i32 {
    let filename = source.display().to_string();
    let content = match std::fs::read_to_string(source) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("nnc: cannot read {filename}: {e}");
            return 1;
        }
    };

    // Phase 1: Lex + Parse
    let tokens = match lexer::tokenize(&content) {
        Ok(t) => t,
        Err(e) => {
            let err = NncError::lex(e.span, &filename, &content);
            eprintln!("{:?}", miette::Report::new(err));
            return 1;
        }
    };

    let file = match parser::parse(&tokens, &content) {
        Ok(f) => f,
        Err(e) => {
            let err = NncError::syntax(e.message, e.span, &filename, &content);
            eprintln!("{:?}", miette::Report::new(err));
            return 1;
        }
    };

    // Phase 2: Lower to IR
    let model = match lower::lower(&file) {
        Ok(m) => m,
        Err(e) => {
            let err = NncError::syntax(e.message, e.span, &filename, &content);
            eprintln!("{:?}", miette::Report::new(err));
            return 1;
        }
    };

    // Semantic validation
    let warnings = match validate::validate(&model) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{}: {}: {}", filename, e.code, e.message);
            return 1;
        }
    };
    for w in &warnings {
        eprintln!("{filename}: {w}");
    }

    // Build graph + topo sort
    let graph_info = match graph::build_graph(&model) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{}: {}: {}", filename, e.code, e.message);
            return 1;
        }
    };
    for w in &graph_info.warnings {
        eprintln!("{filename}: {w}");
    }

    // Shape inference
    let shape_info = match shapes::infer_shapes(&model, &graph_info.topo_order) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {}: {}", filename, e.code, e.message);
            return 1;
        }
    };

    // Memory estimation
    let mem_info = memory::estimate_memory(&model, &shape_info.shapes);

    // Print inspect output
    print_inspect(&model, &shape_info, &mem_info);
    0
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
