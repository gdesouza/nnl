use crate::ir::model::*;

/// Run semantic validation rules on a lowered model.
/// Returns a list of warnings (non-fatal). Errors are returned as Err.
pub fn validate(model: &Model) -> Result<Vec<String>, ValidationError> {
    let mut warnings = Vec::new();

    // W002: missing version declaration
    if model.version.is_none() {
        warnings.push("W002: no version declared, assuming 0.2".to_string());
    }

    // E008: only float32 is supported by codegen in v0.2
    if model.config.precision != Precision::Float32 {
        return Err(ValidationError {
            code: "E008",
            message: format!(
                "precision \"{}\" is not yet supported by codegen; only \"float32\" is available in v0.2",
                model.config.precision
            ),
        });
    }

    // E007: unsupported target/precision combination
    if model.config.precision == Precision::Int8 && model.config.target != Target::Generic {
        return Err(ValidationError {
            code: "E007",
            message: format!(
                "unsupported target `{}` for precision `{}`",
                model.config.target, model.config.precision
            ),
        });
    }

    // Check for duplicate layer IDs
    let mut seen = std::collections::HashSet::new();
    for layer in &model.layers {
        if !seen.insert(&layer.id) {
            return Err(ValidationError {
                code: "E001",
                message: format!("duplicate layer identifier `{}`", layer.id),
            });
        }
    }

    // Validate preprocess config consistency
    if model.config.preprocess == Preprocess::Standardize {
        if model.config.preprocess_mean.is_none() {
            return Err(ValidationError {
                code: "E007",
                message: "preprocess \"standardize\" requires preprocess_mean".to_string(),
            });
        }
        if model.config.preprocess_std.is_none() {
            return Err(ValidationError {
                code: "E007",
                message: "preprocess \"standardize\" requires preprocess_std".to_string(),
            });
        }
    }

    Ok(warnings)
}

#[derive(Debug)]
pub struct ValidationError {
    pub code: &'static str,
    pub message: String,
}
