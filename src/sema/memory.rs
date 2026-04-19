use indexmap::IndexMap;

use crate::ir::model::*;

/// Memory statistics for a model.
#[derive(Debug)]
pub struct MemoryInfo {
    /// Parameter count per layer ID.
    pub layer_params: IndexMap<String, usize>,
    /// Total parameter count.
    pub total_params: usize,
    /// Total weight memory in bytes.
    pub weight_bytes: usize,
    /// Estimated static workspace in bytes (peak activation memory).
    pub workspace_bytes: usize,
}

/// Compute parameter counts and memory estimates.
pub fn estimate_memory(model: &Model, shapes: &IndexMap<String, Vec<usize>>) -> MemoryInfo {
    let elem_size = model.config.precision.byte_size();
    let mut layer_params = IndexMap::new();
    let mut total_params: usize = 0;

    for layer in &model.layers {
        let input_shape = get_input_shape(&layer.id, model, shapes);
        let count = param_count(&layer.kind, input_shape.as_deref());
        layer_params.insert(layer.id.clone(), count);
        total_params += count;
    }

    let weight_bytes = total_params * elem_size;

    // Estimate workspace: sum of the two largest activation buffers (ping-pong).
    // For a more precise estimate, we'd need liveness analysis (Phase 4).
    let mut activation_sizes: Vec<usize> = Vec::new();
    for layer in &model.layers {
        if let Some(shape) = shapes.get(&layer.id) {
            let elems: usize = shape.iter().product::<usize>() * model.config.batch;
            activation_sizes.push(elems * elem_size);
        }
    }
    activation_sizes.sort_unstable_by(|a, b| b.cmp(a));
    let workspace_bytes = activation_sizes.iter().take(2).sum();

    MemoryInfo {
        layer_params,
        total_params,
        weight_bytes,
        workspace_bytes,
    }
}

fn param_count(kind: &LayerKind, input_shape: Option<&[usize]>) -> usize {
    match kind {
        LayerKind::Dense { units, .. } => {
            // weight: input_dim × units, bias: units
            let input_dim = input_shape
                .map(|s| s.iter().product::<usize>())
                .unwrap_or(0);
            input_dim * units + units
        }
        LayerKind::Conv2D {
            filters, kernel, ..
        } => {
            // weight: filters × in_channels × kH × kW, bias: filters
            let in_channels = input_shape.and_then(|s| s.last().copied()).unwrap_or(0);
            let kh = kernel.height();
            let kw = kernel.width();
            filters * in_channels * kh * kw + filters
        }
        LayerKind::BatchNorm { .. } => {
            // gamma, beta, running_mean, running_var (4 × channels)
            let channels = input_shape.and_then(|s| s.last().copied()).unwrap_or(0);
            channels * 4
        }
        _ => 0,
    }
}

fn get_input_shape(
    layer_id: &str,
    model: &Model,
    shapes: &IndexMap<String, Vec<usize>>,
) -> Option<Vec<usize>> {
    // Find first incoming edge
    let source_id = model
        .edges
        .iter()
        .find(|e| e.target == layer_id)
        .map(|e| &e.source)?;
    shapes.get(source_id.as_str()).cloned()
}
