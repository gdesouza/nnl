use crate::ir::graph;
use crate::ir::model::*;
use crate::sema::shapes::ShapeInfo;

/// Describes a required weight tensor.
#[derive(Debug)]
pub struct RequiredWeight {
    pub name: String,
    pub expected_shape: Vec<usize>,
}

/// Enumerate all required weights for a model based on layer types and inferred shapes.
pub fn required_weights(model: &Model, shape_info: &ShapeInfo) -> Vec<RequiredWeight> {
    let mut required = Vec::new();

    for layer in &model.layers {
        let input_shape = get_input_shape(&layer.id, model, shape_info);
        match &layer.kind {
            LayerKind::Dense { units, .. } => {
                let input_dim: usize = input_shape.map(|s| s.iter().product()).unwrap_or(0);
                // weight: [input_dim, units]
                required.push(RequiredWeight {
                    name: format!("{}.weight", layer.id),
                    expected_shape: vec![input_dim, *units],
                });
                // bias: [units]
                required.push(RequiredWeight {
                    name: format!("{}.bias", layer.id),
                    expected_shape: vec![*units],
                });
            }
            LayerKind::Conv2D {
                filters, kernel, groups, ..
            } => {
                let in_channels = input_shape.and_then(|s| s.last().copied()).unwrap_or(0);
                let kh = kernel.height();
                let kw = kernel.width();
                let ci_per_group = in_channels / groups;
                // weight: [filters, in_channels/groups, kH, kW]
                required.push(RequiredWeight {
                    name: format!("{}.weight", layer.id),
                    expected_shape: vec![*filters, ci_per_group, kh, kw],
                });
                // bias: [filters]
                required.push(RequiredWeight {
                    name: format!("{}.bias", layer.id),
                    expected_shape: vec![*filters],
                });
            }
            LayerKind::BatchNorm { .. } => {
                let channels = input_shape.and_then(|s| s.last().copied()).unwrap_or(0);
                for param in &["gamma", "beta", "running_mean", "running_var"] {
                    required.push(RequiredWeight {
                        name: format!("{}.{param}", layer.id),
                        expected_shape: vec![channels],
                    });
                }
            }
            _ => {}
        }
    }

    required
}

fn get_input_shape<'a>(
    layer_id: &str,
    model: &Model,
    shape_info: &'a ShapeInfo,
) -> Option<&'a Vec<usize>> {
    let inputs = graph::get_input_layers(model, layer_id);
    inputs.first().and_then(|l| shape_info.shapes.get(&l.id))
}
