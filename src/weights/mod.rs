mod load;
mod require;
mod tensor;

use std::path::Path;

pub use load::load_weights;
pub use require::required_weights;
pub use tensor::WeightTensor;

use indexmap::IndexMap;

use crate::ir::model::Model;
use crate::sema::shapes::ShapeInfo;

/// All loaded and validated weights for a model.
pub type WeightSet = IndexMap<String, WeightTensor>;

/// Load and validate all weights for a model.
pub fn load_and_validate(
    model: &Model,
    shape_info: &ShapeInfo,
    source_path: &Path,
) -> Result<WeightSet, WeightError> {
    let required = required_weights(model, shape_info);
    let mut loaded = load_weights(&model.config.weights, model.config.precision)?;
    let mut result = IndexMap::new();
    let mut missing = Vec::new();

    for req in &required {
        let key = &req.name;
        match loaded.remove(key.as_str()) {
            Some(tensor) => {
                // Validate shape
                if tensor.shape != req.expected_shape {
                    return Err(WeightError {
                        code: "E003",
                        message: format!(
                            "shape mismatch for weight `{key}`: expected {:?}, got {:?}",
                            req.expected_shape, tensor.shape
                        ),
                    });
                }
                result.insert(key.clone(), tensor);
            }
            None => missing.push((key.clone(), req.expected_shape.clone())),
        }
    }

    if !missing.is_empty() {
        return Err(missing_weight_error(
            &model.config.weights,
            source_path,
            &missing,
        ));
    }

    Ok(result)
}

fn missing_weight_error(
    weights_path: &str,
    source_path: &Path,
    missing: &[(String, Vec<usize>)],
) -> WeightError {
    let path = Path::new(weights_path);
    let mut message = String::new();

    if path.is_dir() {
        message.push_str(&format!(
            "missing required weight files in `{weights_path}`\nsearched directory: `{weights_path}`\nexpected files:\n"
        ));
        for (name, shape) in missing {
            message.push_str(&format!(
                "  - `{}.npy` for `{}` with shape {:?}\n",
                name, name, shape
            ));
        }
    } else if weights_path.ends_with(".npz") {
        message.push_str(&format!(
            "missing required weight arrays in `{weights_path}`\nsearched archive: `{weights_path}`\nexpected arrays:\n"
        ));
        for (name, shape) in missing {
            message.push_str(&format!("  - `{}` with shape {:?}\n", name, shape));
        }
    } else {
        message.push_str(&format!(
            "missing required weights from `{weights_path}`\nexpected tensors:\n"
        ));
        for (name, shape) in missing {
            message.push_str(&format!("  - `{}` with shape {:?}\n", name, shape));
        }
    }

    message.push_str(&format!(
        "hint: run `nnc inspect {}` to view expected tensors and shapes",
        source_path.display()
    ));

    WeightError {
        code: "E003",
        message,
    }
}

#[derive(Debug)]
pub struct WeightError {
    pub code: &'static str,
    pub message: String,
}
