mod load;
mod require;
mod tensor;

pub use load::load_weights;
pub use require::required_weights;
pub use tensor::WeightTensor;

use indexmap::IndexMap;

use crate::ir::model::{Model, Precision};
use crate::sema::shapes::ShapeInfo;

/// All loaded and validated weights for a model.
pub type WeightSet = IndexMap<String, WeightTensor>;

/// Load and validate all weights for a model.
pub fn load_and_validate(
    model: &Model,
    shape_info: &ShapeInfo,
) -> Result<WeightSet, WeightError> {
    let required = required_weights(model, shape_info);
    let mut loaded = load_weights(&model.config.weights, model.config.precision)?;
    let mut result = IndexMap::new();

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
            None => {
                return Err(WeightError {
                    code: "E003",
                    message: format!(
                        "missing weight `{key}`, expected shape {:?}",
                        req.expected_shape
                    ),
                });
            }
        }
    }

    Ok(result)
}

#[derive(Debug)]
pub struct WeightError {
    pub code: &'static str,
    pub message: String,
}
