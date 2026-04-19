use crate::cli::{Cli, Command};
use crate::diag::NncError;
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

    print_inspect(&file);
    0
}

fn print_inspect(file: &crate::syntax::ast::File) {
    if let Some(v) = &file.version {
        println!("Version: {}", v.number);
    }

    let model = &file.model;
    println!("Model: {}", model.name.name);
    println!();

    println!("Config:");
    for setting in &model.config.settings {
        println!("  {}: {}", setting.key.name, format_value(&setting.value));
    }
    println!();

    println!("Layers:");
    for layer in &model.layers {
        let params: Vec<String> = layer
            .params
            .iter()
            .map(|p| format!("{}: {}", p.key.name, format_value(&p.value)))
            .collect();
        if params.is_empty() {
            println!("  {} = {}()", layer.name.name, layer.layer_type);
        } else {
            println!(
                "  {} = {}({})",
                layer.name.name,
                layer.layer_type,
                params.join(", ")
            );
        }
    }

    if let Some(conns) = &model.connections {
        println!();
        println!("Connections:");
        for conn in &conns.connections {
            if conn.sources.len() == 1 {
                println!("  {} -> {}", conn.sources[0].name, conn.target.name);
            } else {
                let srcs: Vec<&str> = conn.sources.iter().map(|s| s.name.as_str()).collect();
                println!("  [{}] -> {}", srcs.join(", "), conn.target.name);
            }
        }
    }
}

fn format_value(value: &crate::syntax::ast::Value) -> String {
    use crate::syntax::ast::Value;
    match value {
        Value::String(s, _) => format!("\"{s}\""),
        Value::Integer(n, _) => n.to_string(),
        Value::Float(f, _) => f.to_string(),
        Value::Bool(b, _) => b.to_string(),
        Value::Shape(nums, _) => {
            let parts: Vec<String> = nums.iter().map(|n| format_shape_num(*n)).collect();
            format!("[{}]", parts.join(", "))
        }
    }
}

fn format_shape_num(n: f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}
