use indexmap::IndexMap;

use crate::ir::graph;
use crate::ir::model::*;

/// Shape assigned to each layer after inference.
#[derive(Debug, Clone)]
pub struct ShapeInfo {
    /// Output shape per layer ID (excludes batch dimension).
    pub shapes: IndexMap<String, Vec<usize>>,
}

/// Propagate shapes through the graph in topological order.
pub fn infer_shapes(model: &Model, topo_order: &[String]) -> Result<ShapeInfo, ShapeError> {
    let mut shapes: IndexMap<String, Vec<usize>> = IndexMap::new();

    for layer_id in topo_order {
        let layer = model.layers.iter().find(|l| l.id == *layer_id).unwrap();
        let output_shape = compute_output_shape(layer, model, &shapes)?;
        shapes.insert(layer_id.clone(), output_shape);
    }

    Ok(ShapeInfo { shapes })
}

fn compute_output_shape(
    layer: &Layer,
    model: &Model,
    shapes: &IndexMap<String, Vec<usize>>,
) -> Result<Vec<usize>, ShapeError> {
    match &layer.kind {
        LayerKind::Input { shape } => Ok(shape.clone()),

        LayerKind::Dense { units, .. } => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            // Input must be 1D (flattened)
            if input.len() != 1 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: Dense expects 1D input, got {:?}",
                        layer.id, input
                    ),
                });
            }
            Ok(vec![*units])
        }

        LayerKind::Conv2D {
            filters,
            kernel,
            stride,
            padding,
            ..
        } => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            // Expect HWC: [H, W, C]
            if input.len() != 3 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: Conv2D expects 3D input [H, W, C], got {:?}",
                        layer.id, input
                    ),
                });
            }
            let (h, w) = (input[0], input[1]);
            let kh = kernel.height();
            let kw = kernel.width();

            let (oh, ow) = match padding {
                Padding::Valid => {
                    let oh = (h - kh) / stride + 1;
                    let ow = (w - kw) / stride + 1;
                    (oh, ow)
                }
                Padding::Same => {
                    let oh = h.div_ceil(*stride);
                    let ow = w.div_ceil(*stride);
                    (oh, ow)
                }
            };
            Ok(vec![oh, ow, *filters])
        }

        LayerKind::MaxPool2D { kernel, stride } | LayerKind::AvgPool2D { kernel, stride } => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            if input.len() != 3 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: pooling expects 3D input [H, W, C], got {:?}",
                        layer.id, input
                    ),
                });
            }
            let (h, w, c) = (input[0], input[1], input[2]);
            let kh = kernel.height();
            let kw = kernel.width();
            let s = stride.unwrap_or(kh); // default stride = kernel size
            let oh = (h - kh) / s + 1;
            let ow = (w - kw) / s + 1;
            Ok(vec![oh, ow, c])
        }

        LayerKind::Flatten => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            let total: usize = input.iter().product();
            Ok(vec![total])
        }

        LayerKind::GlobalAvgPool2D => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            if input.len() != 3 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: GlobalAvgPool2D expects 3D input [H, W, C], got {:?}",
                        layer.id, input
                    ),
                });
            }
            Ok(vec![input[2]])
        }

        LayerKind::BatchNorm { .. }
        | LayerKind::Dropout { .. }
        | LayerKind::ReLU
        | LayerKind::ReLU6
        | LayerKind::LeakyReLU { .. }
        | LayerKind::SiLU
        | LayerKind::Hardswish
        | LayerKind::LayerNorm { .. }
        | LayerKind::Sigmoid
        | LayerKind::Softmax { .. } => {
            // Identity shape
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            Ok(input.clone())
        }

        LayerKind::Conv1D {
            filters,
            kernel,
            stride,
            padding,
            ..
        } => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            // Expect [L, C]
            if input.len() != 2 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: Conv1D expects 2D input [L, C], got {:?}",
                        layer.id, input
                    ),
                });
            }
            let l = input[0];
            let ol = match padding {
                Padding::Valid => (l - kernel) / stride + 1,
                Padding::Same => l.div_ceil(*stride),
            };
            Ok(vec![ol, *filters])
        }

        LayerKind::MaxPool1D { kernel, stride } => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            if input.len() != 2 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: MaxPool1D expects 2D input [L, C], got {:?}",
                        layer.id, input
                    ),
                });
            }
            let l = input[0];
            let s = stride.unwrap_or(*kernel);
            let ol = (l - kernel) / s + 1;
            Ok(vec![ol, input[1]])
        }

        LayerKind::Upsample { scale_h, scale_w } => {
            let input = get_single_input_shape(&layer.id, model, shapes)?;
            if input.len() != 3 {
                return Err(ShapeError {
                    code: "E002",
                    message: format!(
                        "shape mismatch at `{}`: Upsample expects 3D input [H, W, C], got {:?}",
                        layer.id, input
                    ),
                });
            }
            Ok(vec![input[0] * scale_h, input[1] * scale_w, input[2]])
        }

        LayerKind::Add | LayerKind::Mul => {
            let inputs = get_multi_input_shapes(&layer.id, model, shapes)?;
            if inputs.len() < 2 {
                return Err(ShapeError {
                    code: "E004",
                    message: format!(
                        "{} `{}` requires at least 2 inputs",
                        layer.kind.type_name(),
                        layer.id
                    ),
                });
            }
            let first = inputs[0];
            for (i, shape) in inputs.iter().enumerate().skip(1) {
                if *shape != first {
                    return Err(ShapeError {
                        code: "E004",
                        message: format!(
                            "shape mismatch in {} `{}`: input 0 has shape {:?}, input {} has shape {:?}",
                            layer.kind.type_name(),
                            layer.id,
                            first,
                            i,
                            shape
                        ),
                    });
                }
            }
            Ok(first.to_vec())
        }

        LayerKind::Concat { axis } => {
            let inputs = get_multi_input_shapes(&layer.id, model, shapes)?;
            if inputs.len() < 2 {
                return Err(ShapeError {
                    code: "E005",
                    message: format!("Concat `{}` requires at least 2 inputs", layer.id),
                });
            }
            let ndim = inputs[0].len();
            let axis_normalized = normalize_axis(*axis, ndim)?;

            // Check all dims match except concat axis
            let first = inputs[0];
            for (i, shape) in inputs.iter().enumerate().skip(1) {
                if shape.len() != ndim {
                    return Err(ShapeError {
                        code: "E005",
                        message: format!(
                            "incompatible shapes for Concat `{}`: input 0 has {} dims, input {} has {} dims",
                            layer.id,
                            ndim,
                            i,
                            shape.len()
                        ),
                    });
                }
                for (d, (a, b)) in first.iter().zip(shape.iter()).enumerate() {
                    if d != axis_normalized && a != b {
                        return Err(ShapeError {
                            code: "E005",
                            message: format!(
                                "incompatible shapes for Concat `{}` on axis {}: input shapes {:?}",
                                layer.id, axis_normalized, inputs
                            ),
                        });
                    }
                }
            }

            // Build output shape: sum along concat axis
            let mut output = first.to_vec();
            for shape in inputs.iter().skip(1) {
                output[axis_normalized] += shape[axis_normalized];
            }
            Ok(output)
        }
    }
}

fn get_single_input_shape<'a>(
    layer_id: &str,
    model: &Model,
    shapes: &'a IndexMap<String, Vec<usize>>,
) -> Result<&'a Vec<usize>, ShapeError> {
    let inputs = graph::get_input_layers(model, layer_id);
    if inputs.is_empty() {
        return Err(ShapeError {
            code: "E001",
            message: format!("layer `{layer_id}` has no input connection"),
        });
    }
    // Use the first (and typically only) input
    let input_id = &inputs[0].id;
    shapes.get(input_id).ok_or_else(|| ShapeError {
        code: "E002",
        message: format!("cannot resolve input shape for `{layer_id}`"),
    })
}

fn get_multi_input_shapes<'a>(
    layer_id: &str,
    model: &Model,
    shapes: &'a IndexMap<String, Vec<usize>>,
) -> Result<Vec<&'a Vec<usize>>, ShapeError> {
    let inputs = graph::get_input_layers(model, layer_id);
    let mut result = Vec::new();
    for input in &inputs {
        let shape = shapes.get(&input.id).ok_or_else(|| ShapeError {
            code: "E002",
            message: format!("cannot resolve input shape for `{layer_id}`"),
        })?;
        result.push(shape);
    }
    Ok(result)
}

fn normalize_axis(axis: i64, ndim: usize) -> Result<usize, ShapeError> {
    let ndim_i = ndim as i64;
    if axis >= 0 && axis < ndim_i {
        Ok(axis as usize)
    } else if axis < 0 && axis >= -ndim_i {
        Ok((ndim_i + axis) as usize)
    } else {
        Err(ShapeError {
            code: "E005",
            message: format!("axis {axis} is out of range for {ndim} dimensions"),
        })
    }
}

#[derive(Debug)]
pub struct ShapeError {
    pub code: &'static str,
    pub message: String,
}
