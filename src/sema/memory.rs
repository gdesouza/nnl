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
    /// Total static memory (weights + workspace).
    pub total_bytes: usize,
}

/// Result of a memory limit check.
pub enum MemoryCheck {
    /// Total memory is within limits, no issues.
    Ok,
    /// Total memory exceeds 256 MB default threshold (warning).
    Warning(String),
    /// Total memory exceeds the user-specified limit (hard error).
    Error(String),
}

/// Check memory against optional user limit and default 256 MB warning threshold.
pub fn check_memory_limit(mem_info: &MemoryInfo, limit: Option<usize>) -> MemoryCheck {
    let total = mem_info.total_bytes;
    if let Some(max_bytes) = limit
        && total > max_bytes
    {
        return MemoryCheck::Error(format!(
            "E009: total static memory ({}) exceeds memory_limit ({})",
            format_bytes(total),
            format_bytes(max_bytes),
        ));
    }
    const DEFAULT_WARN_THRESHOLD: usize = 256 * 1024 * 1024;
    if total > DEFAULT_WARN_THRESHOLD {
        return MemoryCheck::Warning(format!(
            "W003: this model requires {} of static memory — verify your target supports this",
            format_bytes(total),
        ));
    }
    MemoryCheck::Ok
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
    let total_bytes = weight_bytes + workspace_bytes;

    MemoryInfo {
        layer_params,
        total_params,
        weight_bytes,
        workspace_bytes,
        total_bytes,
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
