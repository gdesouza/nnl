use crate::ir::model::*;
use crate::syntax::ast;
use crate::syntax::lexer::Span;

#[derive(Debug)]
pub struct LowerError {
    pub message: String,
    pub span: Span,
}

pub fn lower(file: &ast::File) -> Result<Model, LowerError> {
    let version = file.version.as_ref().map(|v| v.number);
    let config = lower_config(&file.model.config)?;
    let layers = lower_layers(&file.model.layers)?;
    let edges = build_edges(&file.model)?;

    Ok(Model {
        name: file.model.name.name.clone(),
        version,
        config,
        layers,
        edges,
    })
}

fn lower_config(config: &ast::ConfigBlock) -> Result<Config, LowerError> {
    let mut precision = Precision::Float32;
    let mut weights: Option<String> = None;
    let mut target = Target::Generic;
    let mut align: usize = 64;
    let mut batch: usize = 1;
    let mut preprocess = Preprocess::None;
    let mut preprocess_mean: Option<Vec<f64>> = None;
    let mut preprocess_std: Option<Vec<f64>> = None;
    let mut io = IoMode::Stdio;
    let mut memory_limit: Option<usize> = None;

    let known_keys = [
        "precision",
        "weights",
        "target",
        "align",
        "batch",
        "preprocess",
        "preprocess_mean",
        "preprocess_std",
        "io",
        "memory_limit",
    ];

    for setting in &config.settings {
        let key = setting.key.name.as_str();
        if !known_keys.contains(&key) {
            return Err(LowerError {
                message: format!("unknown config key `{key}`"),
                span: setting.key.span.clone(),
            });
        }
        match key {
            "precision" => {
                precision = match get_string(&setting.value)? {
                    "float32" => Precision::Float32,
                    "float64" => Precision::Float64,
                    "int8" => Precision::Int8,
                    other => {
                        return Err(LowerError {
                            message: format!(
                                "invalid precision `{other}`, expected \"float32\", \"float64\", or \"int8\""
                            ),
                            span: setting.value.span().clone(),
                        });
                    }
                };
            }
            "weights" => {
                weights = Some(get_string(&setting.value)?.to_string());
            }
            "target" => {
                target = match get_string(&setting.value)? {
                    "generic" => Target::Generic,
                    "avx2" => Target::Avx2,
                    "avx512" => Target::Avx512,
                    "arm_neon" => Target::ArmNeon,
                    other => {
                        return Err(LowerError {
                            message: format!(
                                "invalid target `{other}`, expected \"generic\", \"avx2\", \"avx512\", or \"arm_neon\""
                            ),
                            span: setting.value.span().clone(),
                        });
                    }
                };
            }
            "align" => {
                align = get_usize(&setting.value)?;
            }
            "batch" => {
                batch = get_usize(&setting.value)?;
            }
            "preprocess" => {
                preprocess = match get_string(&setting.value)? {
                    "none" => Preprocess::None,
                    "normalize_0_1" => Preprocess::Normalize01,
                    "standardize" => Preprocess::Standardize,
                    other => {
                        return Err(LowerError {
                            message: format!(
                                "invalid preprocess `{other}`, expected \"none\", \"normalize_0_1\", or \"standardize\""
                            ),
                            span: setting.value.span().clone(),
                        });
                    }
                };
            }
            "preprocess_mean" => {
                preprocess_mean = Some(get_shape(&setting.value)?);
            }
            "preprocess_std" => {
                preprocess_std = Some(get_shape(&setting.value)?);
            }
            "io" => {
                io = match get_string(&setting.value)? {
                    "stdio" => IoMode::Stdio,
                    "none" => IoMode::None,
                    other => {
                        return Err(LowerError {
                            message: format!(
                                "unsupported io mode `{other}`, expected \"stdio\" or \"none\""
                            ),
                            span: setting.value.span().clone(),
                        });
                    }
                };
            }
            "memory_limit" => {
                let s = get_string(&setting.value)?;
                memory_limit = Some(parse_memory_limit(s).map_err(|msg| LowerError {
                    message: msg,
                    span: setting.value.span().clone(),
                })?);
            }
            _ => unreachable!(),
        }
    }

    let weights = weights.ok_or_else(|| LowerError {
        message: "missing required config key `weights`".to_string(),
        span: config.span.clone(),
    })?;

    Ok(Config {
        precision,
        weights,
        target,
        align,
        batch,
        preprocess,
        preprocess_mean,
        preprocess_std,
        io,
        memory_limit,
    })
}

fn lower_layers(layers: &[ast::LayerDecl]) -> Result<Vec<Layer>, LowerError> {
    layers.iter().map(lower_layer).collect()
}

fn lower_layer(layer: &ast::LayerDecl) -> Result<Layer, LowerError> {
    let kind = match layer.layer_type {
        ast::LayerType::Input => {
            let shape = get_required_shape_param(&layer.params, "shape", &layer.span)?;
            let shape: Vec<usize> = shape.iter().map(|v| *v as usize).collect();
            LayerKind::Input { shape }
        }
        ast::LayerType::Dense => {
            let units = get_required_usize_param(&layer.params, "units", &layer.span)?;
            let activation = match get_optional_string_param(&layer.params, "activation")? {
                Some("relu") => Activation::ReLU,
                Some("sigmoid") => Activation::Sigmoid,
                Some("softmax") => Activation::Softmax,
                Some("none") | None => Activation::None,
                Some(other) => {
                    let p = find_param(&layer.params, "activation").unwrap();
                    return Err(LowerError {
                        message: format!("invalid activation `{other}`"),
                        span: p.value.span().clone(),
                    });
                }
            };
            LayerKind::Dense { units, activation }
        }
        ast::LayerType::Conv2D => {
            let filters = get_required_usize_param(&layer.params, "filters", &layer.span)?;
            let kernel = get_required_kernel_param(&layer.params, "kernel", &layer.span)?;
            let stride = get_optional_usize_param(&layer.params, "stride")?.unwrap_or(1);
            let groups = get_optional_usize_param(&layer.params, "groups")?.unwrap_or(1);
            let padding = match get_optional_string_param(&layer.params, "padding")? {
                Some("valid") | None => Padding::Valid,
                Some("same") => Padding::Same,
                Some(other) => {
                    let p = find_param(&layer.params, "padding").unwrap();
                    return Err(LowerError {
                        message: format!(
                            "invalid padding `{other}`, expected \"valid\" or \"same\""
                        ),
                        span: p.value.span().clone(),
                    });
                }
            };
            LayerKind::Conv2D {
                filters,
                kernel,
                stride,
                padding,
                groups,
            }
        }
        ast::LayerType::MaxPool2D => {
            let kernel = get_required_kernel_param(&layer.params, "kernel", &layer.span)?;
            let stride = get_optional_usize_param(&layer.params, "stride")?;
            let padding = get_optional_pool_padding_param(&layer.params, "padding")?;
            LayerKind::MaxPool2D {
                kernel,
                stride,
                padding,
            }
        }
        ast::LayerType::AvgPool2D => {
            let kernel = get_required_kernel_param(&layer.params, "kernel", &layer.span)?;
            let stride = get_optional_usize_param(&layer.params, "stride")?;
            let padding = get_optional_pool_padding_param(&layer.params, "padding")?;
            LayerKind::AvgPool2D {
                kernel,
                stride,
                padding,
            }
        }
        ast::LayerType::Flatten => LayerKind::Flatten,
        ast::LayerType::BatchNorm => {
            let epsilon = get_optional_float_param(&layer.params, "epsilon")?.unwrap_or(1e-5);
            LayerKind::BatchNorm { epsilon }
        }
        ast::LayerType::Dropout => {
            let rate = get_optional_float_param(&layer.params, "rate")?.unwrap_or(0.5);
            LayerKind::Dropout { rate }
        }
        ast::LayerType::Add => LayerKind::Add,
        ast::LayerType::Concat => {
            let axis = get_optional_int_param(&layer.params, "axis")?.unwrap_or(-1);
            LayerKind::Concat { axis }
        }
        ast::LayerType::ReLU => LayerKind::ReLU,
        ast::LayerType::Sigmoid => LayerKind::Sigmoid,
        ast::LayerType::Softmax => {
            let axis = get_optional_int_param(&layer.params, "axis")?.unwrap_or(-1);
            LayerKind::Softmax { axis }
        }
        ast::LayerType::GlobalAvgPool2D => LayerKind::GlobalAvgPool2D,
        ast::LayerType::ReLU6 => LayerKind::ReLU6,
        ast::LayerType::LeakyReLU => {
            let alpha = get_optional_float_param(&layer.params, "alpha")?.unwrap_or(0.01);
            LayerKind::LeakyReLU { alpha }
        }
        ast::LayerType::SiLU => LayerKind::SiLU,
        ast::LayerType::Mul => LayerKind::Mul,
        ast::LayerType::Hardswish => LayerKind::Hardswish,
        ast::LayerType::Upsample => {
            let scale = get_required_usize_param(&layer.params, "scale", &layer.span)?;
            LayerKind::Upsample {
                scale_h: scale,
                scale_w: scale,
            }
        }
        ast::LayerType::Conv1D => {
            let filters = get_required_usize_param(&layer.params, "filters", &layer.span)?;
            let kernel = get_required_usize_param(&layer.params, "kernel", &layer.span)?;
            let stride = get_optional_usize_param(&layer.params, "stride")?.unwrap_or(1);
            let padding = match get_optional_string_param(&layer.params, "padding")? {
                Some("valid") | None => Padding::Valid,
                Some("same") => Padding::Same,
                Some(other) => {
                    let p = find_param(&layer.params, "padding").unwrap();
                    return Err(LowerError {
                        message: format!(
                            "invalid padding `{other}`, expected \"valid\" or \"same\""
                        ),
                        span: p.value.span().clone(),
                    });
                }
            };
            LayerKind::Conv1D {
                filters,
                kernel,
                stride,
                padding,
            }
        }
        ast::LayerType::MaxPool1D => {
            let kernel = get_required_usize_param(&layer.params, "kernel", &layer.span)?;
            let stride = get_optional_usize_param(&layer.params, "stride")?;
            LayerKind::MaxPool1D { kernel, stride }
        }
        ast::LayerType::LayerNorm => {
            let epsilon = get_optional_float_param(&layer.params, "epsilon")?.unwrap_or(1e-5);
            LayerKind::LayerNorm { epsilon }
        }
        ast::LayerType::Lrn => {
            let size = get_required_usize_param(&layer.params, "size", &layer.span)?;
            let alpha = get_optional_float_param(&layer.params, "alpha")?.unwrap_or(1e-4);
            let beta = get_optional_float_param(&layer.params, "beta")?.unwrap_or(0.75);
            let bias = get_optional_float_param(&layer.params, "bias")?.unwrap_or(1.0);
            LayerKind::Lrn {
                size,
                alpha,
                beta,
                bias,
            }
        }
        ast::LayerType::FakeQuant => {
            let scale = get_required_float_param(&layer.params, "scale", &layer.span)?;
            let zero_point = get_required_int_param(&layer.params, "zero_point", &layer.span)?;
            let qmin = get_required_int_param(&layer.params, "qmin", &layer.span)?;
            let qmax = get_required_int_param(&layer.params, "qmax", &layer.span)?;
            LayerKind::FakeQuant {
                scale,
                zero_point,
                qmin,
                qmax,
            }
        }
    };

    Ok(Layer {
        id: layer.name.name.clone(),
        kind,
        span: layer.span.clone(),
    })
}

fn build_edges(model: &ast::ModelDecl) -> Result<Vec<Edge>, LowerError> {
    match &model.connections {
        Some(block) => {
            let mut edges = Vec::new();
            for conn in &block.connections {
                for src in &conn.sources {
                    edges.push(Edge {
                        source: src.name.clone(),
                        target: conn.target.name.clone(),
                    });
                }
            }
            Ok(edges)
        }
        None => {
            // Implicit sequential: connect each layer to the next.
            let mut edges = Vec::new();
            for pair in model.layers.windows(2) {
                edges.push(Edge {
                    source: pair[0].name.name.clone(),
                    target: pair[1].name.name.clone(),
                });
            }
            Ok(edges)
        }
    }
}

// --- Helper functions for extracting typed values from AST params/values ---

fn get_string(value: &ast::Value) -> Result<&str, LowerError> {
    match value {
        ast::Value::String(s, _) => Ok(s.as_str()),
        other => Err(LowerError {
            message: "expected string value".to_string(),
            span: other.span().clone(),
        }),
    }
}

fn get_usize(value: &ast::Value) -> Result<usize, LowerError> {
    match value {
        ast::Value::Integer(n, _) => Ok(*n as usize),
        other => Err(LowerError {
            message: "expected integer value".to_string(),
            span: other.span().clone(),
        }),
    }
}

fn get_shape(value: &ast::Value) -> Result<Vec<f64>, LowerError> {
    match value {
        ast::Value::Shape(nums, _) => Ok(nums.clone()),
        other => Err(LowerError {
            message: "expected shape value".to_string(),
            span: other.span().clone(),
        }),
    }
}

fn find_param<'a>(params: &'a [ast::Param], name: &str) -> Option<&'a ast::Param> {
    params.iter().find(|p| p.key.name == name)
}

fn get_required_usize_param(
    params: &[ast::Param],
    name: &str,
    layer_span: &Span,
) -> Result<usize, LowerError> {
    match find_param(params, name) {
        Some(p) => get_usize(&p.value),
        None => Err(LowerError {
            message: format!("missing required parameter `{name}`"),
            span: layer_span.clone(),
        }),
    }
}

fn get_required_shape_param(
    params: &[ast::Param],
    name: &str,
    layer_span: &Span,
) -> Result<Vec<f64>, LowerError> {
    match find_param(params, name) {
        Some(p) => get_shape(&p.value),
        None => Err(LowerError {
            message: format!("missing required parameter `{name}`"),
            span: layer_span.clone(),
        }),
    }
}

fn get_required_kernel_param(
    params: &[ast::Param],
    name: &str,
    layer_span: &Span,
) -> Result<KernelSize, LowerError> {
    match find_param(params, name) {
        Some(p) => match &p.value {
            ast::Value::Integer(n, _) => Ok(KernelSize::Square(*n as usize)),
            ast::Value::Shape(nums, span) => {
                if nums.len() == 2 {
                    Ok(KernelSize::Rect(nums[0] as usize, nums[1] as usize))
                } else {
                    Err(LowerError {
                        message: "kernel shape must have exactly 2 dimensions".to_string(),
                        span: span.clone(),
                    })
                }
            }
            other => Err(LowerError {
                message: "expected integer or shape for kernel".to_string(),
                span: other.span().clone(),
            }),
        },
        None => Err(LowerError {
            message: format!("missing required parameter `{name}`"),
            span: layer_span.clone(),
        }),
    }
}

fn get_optional_usize_param(
    params: &[ast::Param],
    name: &str,
) -> Result<Option<usize>, LowerError> {
    match find_param(params, name) {
        Some(p) => Ok(Some(get_usize(&p.value)?)),
        None => Ok(None),
    }
}

fn get_optional_string_param<'a>(
    params: &'a [ast::Param],
    name: &str,
) -> Result<Option<&'a str>, LowerError> {
    match find_param(params, name) {
        Some(p) => Ok(Some(get_string(&p.value)?)),
        None => Ok(None),
    }
}

fn get_optional_float_param(params: &[ast::Param], name: &str) -> Result<Option<f64>, LowerError> {
    match find_param(params, name) {
        Some(p) => match &p.value {
            ast::Value::Float(v, _) => Ok(Some(*v)),
            ast::Value::Integer(v, _) => Ok(Some(*v as f64)),
            other => Err(LowerError {
                message: "expected number value".to_string(),
                span: other.span().clone(),
            }),
        },
        None => Ok(None),
    }
}

fn get_optional_int_param(params: &[ast::Param], name: &str) -> Result<Option<i64>, LowerError> {
    match find_param(params, name) {
        Some(p) => match &p.value {
            ast::Value::Integer(v, _) => Ok(Some(*v as i64)),
            other => Err(LowerError {
                message: "expected integer value".to_string(),
                span: other.span().clone(),
            }),
        },
        None => Ok(None),
    }
}

fn get_required_float_param(
    params: &[ast::Param],
    name: &str,
    layer_span: &Span,
) -> Result<f64, LowerError> {
    match get_optional_float_param(params, name)? {
        Some(value) => Ok(value),
        None => Err(LowerError {
            message: format!("missing required parameter `{name}`"),
            span: layer_span.clone(),
        }),
    }
}

fn get_required_int_param(
    params: &[ast::Param],
    name: &str,
    layer_span: &Span,
) -> Result<i64, LowerError> {
    match get_optional_int_param(params, name)? {
        Some(value) => Ok(value),
        None => Err(LowerError {
            message: format!("missing required parameter `{name}`"),
            span: layer_span.clone(),
        }),
    }
}

fn get_optional_pool_padding_param(
    params: &[ast::Param],
    name: &str,
) -> Result<Option<PoolPadding>, LowerError> {
    let Some(param) = find_param(params, name) else {
        return Ok(None);
    };
    match &param.value {
        ast::Value::Shape(nums, span) => {
            if nums.len() != 4 {
                return Err(LowerError {
                    message: "pool padding must have exactly 4 values [top, left, bottom, right]"
                        .to_string(),
                    span: span.clone(),
                });
            }
            Ok(Some(PoolPadding {
                top: nums[0] as usize,
                left: nums[1] as usize,
                bottom: nums[2] as usize,
                right: nums[3] as usize,
            }))
        }
        other => Err(LowerError {
            message: "expected shape value for pool padding".to_string(),
            span: other.span().clone(),
        }),
    }
}

fn parse_memory_limit(s: &str) -> Result<usize, String> {
    let s = s.trim();
    if let Some(num) = s.strip_suffix("GB") {
        let n: f64 = num
            .trim()
            .parse()
            .map_err(|_| format!("invalid memory_limit `{s}`, expected e.g. \"256MB\""))?;
        Ok((n * 1024.0 * 1024.0 * 1024.0) as usize)
    } else if let Some(num) = s.strip_suffix("MB") {
        let n: f64 = num
            .trim()
            .parse()
            .map_err(|_| format!("invalid memory_limit `{s}`, expected e.g. \"256MB\""))?;
        Ok((n * 1024.0 * 1024.0) as usize)
    } else if let Some(num) = s.strip_suffix("KB") {
        let n: f64 = num
            .trim()
            .parse()
            .map_err(|_| format!("invalid memory_limit `{s}`, expected e.g. \"256MB\""))?;
        Ok((n * 1024.0) as usize)
    } else {
        Err(format!(
            "invalid memory_limit `{s}`, expected a value with KB, MB, or GB suffix (e.g. \"256MB\")"
        ))
    }
}
