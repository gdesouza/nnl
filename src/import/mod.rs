pub mod onnx;

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io;
use std::path::Path;

use npyz::WriterBuilder;
use prost::Message;

use crate::import::onnx::{ModelProto, NodeProto, TensorProto};

#[derive(Debug)]
pub struct ImportError {
    pub message: String,
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl From<io::Error> for ImportError {
    fn from(e: io::Error) -> Self {
        ImportError {
            message: e.to_string(),
        }
    }
}

impl From<prost::DecodeError> for ImportError {
    fn from(e: prost::DecodeError) -> Self {
        ImportError {
            message: format!("failed to decode ONNX protobuf: {e}"),
        }
    }
}

pub fn import_onnx(
    onnx_path: &Path,
    output_path: &Path,
    weights_dir: &Path,
) -> Result<(), ImportError> {
    let bytes = fs::read(onnx_path).map_err(|e| ImportError {
        message: format!("cannot read `{}`: {e}", onnx_path.display()),
    })?;

    let model = ModelProto::decode(&bytes[..])?;

    let graph = model.graph.as_ref().ok_or_else(|| ImportError {
        message: "ONNX model has no graph".to_string(),
    })?;

    let model_name = sanitize_id(if graph.name.is_empty() {
        "imported_model"
    } else {
        &graph.name
    });

    // Build initializer lookup
    let initializers: HashMap<&str, &TensorProto> = graph
        .initializer
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    // Resolve the directory containing the ONNX file (for external data)
    let onnx_dir = onnx_path.parent().unwrap_or(Path::new(".")).to_path_buf();

    let resolved_tensors = build_resolved_tensors(&graph.node, &initializers, &onnx_dir)?;

    // Detect input shape from graph inputs (skip initializers)
    let input_shape = detect_input_shape(graph);

    // Map nodes to NNL layers
    let mut layers: Vec<LayerDef> = Vec::new();
    let mut connections: Vec<Connection> = Vec::new();
    // Track which ONNX output name maps to which NNL layer id
    let mut output_map: HashMap<String, String> = HashMap::new();

    // Register graph inputs
    for inp in &graph.input {
        if !initializers.contains_key(inp.name.as_str()) {
            output_map.insert(inp.name.clone(), "input".to_string());
        }
    }

    // Add input layer
    layers.push(LayerDef {
        id: "input".to_string(),
        def: format!("Input(shape: [{}])", input_shape.join(", ")),
    });

    // Create weights dir
    fs::create_dir_all(weights_dir)?;

    let mut layer_counter: HashMap<String, usize> = HashMap::new();

    // Track ONNX output names that come from Flatten nodes and the CHW shape
    // before Flatten (needed for CHW→HWC weight permutation).
    let mut flatten_shapes: HashMap<String, (usize, usize, usize)> = HashMap::new();

    // Build a map of ONNX output name → tensor shape from initializers and graph
    // value_info, so we can infer the pre-Flatten shape.
    let mut tensor_shapes: HashMap<String, Vec<i64>> = HashMap::new();
    // Seed from graph inputs
    for inp in &graph.input {
        if let Some(tp) = &inp.r#type
            && let Some(tt) = &tp.tensor_type
            && let Some(shape) = &tt.shape
        {
            let dims: Vec<i64> = shape.dim.iter().map(|d| d.dim_value).collect();
            tensor_shapes.insert(inp.name.clone(), dims);
        }
    }

    // First pass: propagate shapes and identify Flatten→Gemm boundaries
    for node in &graph.node {
        match node.op_type.as_str() {
            "Flatten" => {
                if let Some(inp_name) = node.input.first()
                    && let Some(in_dims) = tensor_shapes.get(inp_name.as_str()).cloned()
                    && in_dims.len() == 4
                {
                    // ONNX shape is [N, C, H, W]
                    let (c, h, w) = (
                        in_dims[1] as usize,
                        in_dims[2] as usize,
                        in_dims[3] as usize,
                    );
                    for out in &node.output {
                        flatten_shapes.insert(out.clone(), (c, h, w));
                        tensor_shapes.insert(out.clone(), vec![in_dims[0], (c * h * w) as i64]);
                    }
                }
            }
            "Reshape" => {
                if should_lower_reshape_to_flatten(node, &tensor_shapes, &resolved_tensors)
                    && let Some(inp_name) = node.input.first()
                    && let Some(in_dims) = tensor_shapes.get(inp_name.as_str()).cloned()
                    && !in_dims.is_empty()
                {
                    let flattened: i64 = in_dims.iter().skip(1).product();
                    for out in &node.output {
                        flatten_shapes.insert(
                            out.clone(),
                            (
                                in_dims.get(1).copied().unwrap_or(1) as usize,
                                in_dims.get(2).copied().unwrap_or(1) as usize,
                                in_dims.get(3).copied().unwrap_or(1) as usize,
                            ),
                        );
                        tensor_shapes.insert(out.clone(), vec![in_dims[0], flattened]);
                    }
                }
            }
            "Conv" => {
                // Propagate output shape: [N, filters, OH, OW]
                if let Some(w_name) = node.input.get(1)
                    && let Some(tensor) = resolved_tensors.get(w_name.as_str())
                    && let Some(inp_name) = node.input.first()
                    && let Some(in_dims) = tensor_shapes.get(inp_name.as_str()).cloned()
                    && in_dims.len() == 4
                {
                    let filters = tensor.dims[0];
                    let kh = tensor.dims[2];
                    let kw = tensor.dims[3];
                    let mut stride: i64 = 1;
                    let mut pad_h: i64 = 0;
                    let mut pad_w: i64 = 0;
                    for attr in &node.attribute {
                        match attr.name.as_str() {
                            "strides" => {
                                if let Some(&s) = attr.ints.first() {
                                    stride = s;
                                }
                            }
                            "pads" if attr.ints.len() >= 2 => {
                                pad_h = attr.ints[0];
                                pad_w = attr.ints[1];
                            }
                            _ => {}
                        }
                    }
                    let oh = (in_dims[2] + 2 * pad_h - kh) / stride + 1;
                    let ow = (in_dims[3] + 2 * pad_w - kw) / stride + 1;
                    for out in &node.output {
                        tensor_shapes.insert(out.clone(), vec![in_dims[0], filters, oh, ow]);
                    }
                }
            }
            "MaxPool" | "AveragePool" => {
                if let Some(inp_name) = node.input.first()
                    && let Some(in_dims) = tensor_shapes.get(inp_name.as_str()).cloned()
                    && in_dims.len() == 4
                {
                    let mut kernel: i64 = 2;
                    let mut stride: i64 = 2;
                    let mut pad_h: i64 = 0;
                    let mut pad_w: i64 = 0;
                    for attr in &node.attribute {
                        match attr.name.as_str() {
                            "kernel_shape" => {
                                if let Some(&k) = attr.ints.first() {
                                    kernel = k;
                                }
                            }
                            "strides" => {
                                if let Some(&s) = attr.ints.first() {
                                    stride = s;
                                }
                            }
                            "pads" if attr.ints.len() >= 2 => {
                                pad_h = attr.ints[0];
                                pad_w = attr.ints[1];
                            }
                            _ => {}
                        }
                    }
                    let oh = (in_dims[2] + 2 * pad_h - kernel) / stride + 1;
                    let ow = (in_dims[3] + 2 * pad_w - kernel) / stride + 1;
                    for out in &node.output {
                        tensor_shapes.insert(out.clone(), vec![in_dims[0], in_dims[1], oh, ow]);
                    }
                }
            }
            "Concat" => {
                let Some(first_input) = node.input.first() else {
                    continue;
                };
                let Some(mut out_dims) = tensor_shapes.get(first_input.as_str()).cloned() else {
                    continue;
                };
                let Some(axis) = remap_axis_for_shape(node, &out_dims) else {
                    continue;
                };
                let axis = normalize_axis(axis, out_dims.len());
                for inp_name in node.input.iter().skip(1) {
                    let Some(shape) = tensor_shapes.get(inp_name.as_str()) else {
                        continue;
                    };
                    if shape.len() == out_dims.len() {
                        out_dims[axis] += shape[axis];
                    }
                }
                for out in &node.output {
                    tensor_shapes.insert(out.clone(), out_dims.clone());
                }
            }
            "GlobalAveragePool" => {
                if let Some(inp_name) = node.input.first()
                    && let Some(in_dims) = tensor_shapes.get(inp_name.as_str()).cloned()
                    && in_dims.len() == 4
                {
                    for out in &node.output {
                        tensor_shapes.insert(out.clone(), vec![in_dims[0], in_dims[1], 1, 1]);
                    }
                }
            }
            "Relu" | "Sigmoid" | "BatchNormalization" | "Dropout" | "Add" | "Mul" | "Softmax"
            | "QuantizeLinear" | "DequantizeLinear" => {
                // Shape-preserving ops: propagate input shape
                if let Some(inp_name) = node.input.first()
                    && let Some(dims) = tensor_shapes.get(inp_name.as_str()).cloned()
                {
                    for out in &node.output {
                        tensor_shapes.insert(out.clone(), dims.clone());
                    }
                }
            }
            _ => {}
        }
    }

    let mut skip_nodes = HashSet::new();
    for (node_index, node) in graph.node.iter().enumerate() {
        if skip_nodes.contains(&node_index) {
            continue;
        }

        let fused_bias = find_fusable_add_bias(node_index, &graph.node, &resolved_tensors);
        let (layer_id, layer_def, weight_info) = map_node(
            node,
            &resolved_tensors,
            &mut layer_counter,
            &flatten_shapes,
            &tensor_shapes,
            fused_bias.as_deref(),
        )?;

        if let Some(def) = layer_def {
            // Write weights
            for wi in &weight_info {
                let tensor = resolved_tensors
                    .get(wi.tensor_name.as_str())
                    .ok_or_else(|| ImportError {
                        message: format!(
                            "missing resolved tensor '{}' for layer '{}'",
                            wi.tensor_name, layer_id
                        ),
                    })?;
                let data = tensor.data.clone();
                if data.is_empty() {
                    return Err(ImportError {
                        message: format!(
                            "tensor '{}' has no data (external data not found?)",
                            wi.tensor_name
                        ),
                    });
                }
                let dims: Vec<u64> = tensor.dims.iter().map(|&d| d as u64).collect();
                let path = weights_dir.join(format!("{}.{}.npy", layer_id, wi.param_name));
                if wi.transpose && dims.len() == 2 {
                    // Transpose from [out, in] to [in, out]
                    let (rows, cols) = (dims[0] as usize, dims[1] as usize);
                    let mut transposed = vec![0.0f32; data.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            transposed[c * rows + r] = data[r * cols + c];
                        }
                    }
                    let final_data = if let Some((ch, h, w)) = wi.chw_to_hwc {
                        permute_dense_weight_chw_to_hwc(&transposed, ch, h, w, rows)
                    } else {
                        transposed
                    };
                    write_weight_npy(&path, &[dims[1], dims[0]], &final_data)?;
                } else if let Some((ch, h, w)) = wi.chw_to_hwc {
                    let permuted =
                        permute_dense_weight_chw_to_hwc(&data, ch, h, w, dims[1] as usize);
                    write_weight_npy(&path, &dims, &permuted)?;
                } else {
                    write_weight_npy(&path, &dims, &data)?;
                }
            }

            // Map all ONNX outputs of this node to the layer id
            for out in &node.output {
                output_map.insert(out.clone(), layer_id.clone());
            }
            if let Some(next_index) = fused_bias.as_ref().and_then(|_| node_index.checked_add(1))
                && let Some(add_node) = graph.node.get(next_index)
            {
                for out in &add_node.output {
                    output_map.insert(out.clone(), layer_id.clone());
                }
                skip_nodes.insert(next_index);
            }

            // Build connections from ONNX inputs
            let mut sources: Vec<String> = Vec::new();
            for inp_name in &node.input {
                if inp_name.is_empty() || initializers.contains_key(inp_name.as_str()) {
                    continue;
                }
                if let Some(src) = output_map.get(inp_name)
                    && !sources.contains(src)
                {
                    sources.push(src.clone());
                }
            }

            if !sources.is_empty() {
                connections.push(Connection {
                    from: sources,
                    to: layer_id.clone(),
                });
            }

            layers.push(LayerDef { id: layer_id, def });
        } else {
            // Unsupported op — pass through outputs
            for (i, out) in node.output.iter().enumerate() {
                if let Some(inp) = node.input.get(i)
                    && let Some(src) = output_map.get(inp)
                {
                    output_map.insert(out.clone(), src.clone());
                }
            }
            if !matches!(node.op_type.as_str(), "QuantizeLinear" | "DequantizeLinear") {
                // Add as comment
                layers.push(LayerDef {
                    id: String::new(),
                    def: format!("// UNSUPPORTED: {}({})", node.op_type, node.name),
                });
            }
        }
    }

    // Compute relative weights path for the .nnl file
    let weights_rel = pathdiff(output_path, weights_dir);

    // Generate .nnl source
    let mut nnl = String::new();
    nnl.push_str("version 0.2;\n\n");
    nnl.push_str(&format!("model {model_name} {{\n"));
    nnl.push_str("    config {\n");
    nnl.push_str("        precision: \"float32\";\n");
    nnl.push_str(&format!("        weights: \"{}\";\n", weights_rel));
    nnl.push_str("        io: \"stdio\";\n");
    nnl.push_str("    }\n\n");

    // Determine max id width for alignment
    let max_id_len = layers
        .iter()
        .filter(|l| !l.id.is_empty())
        .map(|l| l.id.len())
        .max()
        .unwrap_or(0);

    for layer in &layers {
        if layer.id.is_empty() {
            nnl.push_str(&format!("    {}\n", layer.def));
        } else {
            nnl.push_str(&format!(
                "    layer {:<width$} = {};\n",
                layer.id,
                layer.def,
                width = max_id_len
            ));
        }
    }

    // Emit connections if there are multi-input layers or non-sequential flow
    let needs_connections = connections
        .iter()
        .any(|c| c.from.len() > 1 || !is_sequential_flow(&connections));

    if needs_connections {
        nnl.push_str("\n    connections {\n");
        for conn in &connections {
            if conn.from.len() == 1 {
                nnl.push_str(&format!("        {} -> {};\n", conn.from[0], conn.to));
            } else {
                let sources = conn.from.join(", ");
                nnl.push_str(&format!("        [{}] -> {};\n", sources, conn.to));
            }
        }
        nnl.push_str("    }\n");
    }

    nnl.push_str("}\n");

    fs::write(output_path, &nnl)?;
    Ok(())
}

struct LayerDef {
    id: String,
    def: String,
}

struct Connection {
    from: Vec<String>,
    to: String,
}

struct WeightRef {
    tensor_name: String,
    param_name: String,
    transpose: bool,
    /// When set, the Dense weight rows are permuted from CHW to HWC flatten order.
    /// Contains (channels, height, width) of the pre-Flatten activation.
    chw_to_hwc: Option<(usize, usize, usize)>,
}

#[derive(Clone)]
struct ResolvedTensor {
    dims: Vec<i64>,
    data: Vec<f32>,
}

fn map_node(
    node: &NodeProto,
    resolved_tensors: &HashMap<String, ResolvedTensor>,
    counter: &mut HashMap<String, usize>,
    flatten_shapes: &HashMap<String, (usize, usize, usize)>,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    fused_bias: Option<&str>,
) -> Result<(String, Option<String>, Vec<WeightRef>), ImportError> {
    let op = node.op_type.as_str();
    let layer_id = make_layer_id(node, op, counter);

    match op {
        "Gemm" | "MatMul" => {
            let mut weights = Vec::new();
            let mut units = 0usize;
            // Check transB attribute (default 0)
            let trans_b = node
                .attribute
                .iter()
                .find(|a| a.name == "transB")
                .map(|a| a.i)
                .unwrap_or(0);
            // Detect if this Gemm follows a Flatten with a known CHW shape
            let chw_permute = node
                .input
                .first()
                .and_then(|inp| flatten_shapes.get(inp.as_str()))
                .copied();
            // Weight is typically the second input
            if let Some(w_name) = node.input.get(1)
                && let Some(tensor) = resolved_tensors.get(w_name.as_str())
            {
                // transB=1: shape [out, in], first dim is units
                // transB=0: shape [in, out], last dim is units
                units = if trans_b != 0 {
                    *tensor.dims.first().unwrap_or(&0) as usize
                } else {
                    *tensor.dims.last().unwrap_or(&0) as usize
                };
                weights.push(WeightRef {
                    tensor_name: w_name.clone(),
                    param_name: "weight".to_string(),
                    transpose: trans_b != 0,
                    chw_to_hwc: chw_permute,
                });
            }
            if let Some(b_name) = node.input.get(2)
                && resolved_tensors.contains_key(b_name.as_str())
            {
                weights.push(WeightRef {
                    tensor_name: b_name.clone(),
                    param_name: "bias".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }
            if let Some(bias_name) = fused_bias
                && resolved_tensors.contains_key(bias_name)
                && !weights.iter().any(|weight| weight.param_name == "bias")
            {
                weights.push(WeightRef {
                    tensor_name: bias_name.to_string(),
                    param_name: "bias".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }
            if weights.iter().any(|w| w.param_name == "weight") && units == 0 {
                return Err(ImportError {
                    message: format!(
                        "cannot infer Dense units for `{}` from ONNX weights",
                        node.name
                    ),
                });
            }
            Ok((layer_id, Some(format!("Dense(units: {units})")), weights))
        }
        "Conv"
            if node
                .input
                .get(1)
                .and_then(|w| resolved_tensors.get(w.as_str()))
                .is_some_and(|t| t.dims.len() == 3) =>
        {
            let mut weights = Vec::new();
            let mut filters = 0usize;
            let mut kernel = 3;
            let mut stride = 1;
            let mut padding = "valid";

            if let Some(w_name) = node.input.get(1)
                && let Some(tensor) = resolved_tensors.get(w_name.as_str())
            {
                filters = tensor.dims.first().copied().unwrap_or(0) as usize;
                if tensor.dims.len() >= 3 {
                    kernel = tensor.dims[2] as usize;
                }
                weights.push(WeightRef {
                    tensor_name: w_name.clone(),
                    param_name: "weight".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }
            if let Some(b_name) = node.input.get(2)
                && resolved_tensors.contains_key(b_name.as_str())
            {
                weights.push(WeightRef {
                    tensor_name: b_name.clone(),
                    param_name: "bias".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }

            if weights.iter().any(|w| w.param_name == "weight") && filters == 0 {
                return Err(ImportError {
                    message: format!(
                        "cannot infer Conv1D filters for `{}` from ONNX weights",
                        node.name
                    ),
                });
            }

            for attr in &node.attribute {
                match attr.name.as_str() {
                    "strides" => {
                        if let Some(&s) = attr.ints.first() {
                            stride = s as usize;
                        }
                    }
                    "pads" if attr.ints.iter().any(|&p| p > 0) => {
                        padding = "same";
                    }
                    "kernel_shape" => {
                        if let Some(&k) = attr.ints.first() {
                            kernel = k as usize;
                        }
                    }
                    _ => {}
                }
            }
            Ok((
                layer_id,
                Some(format!(
                    "Conv1D(filters: {filters}, kernel: {kernel}, stride: {stride}, padding: \"{padding}\")"
                )),
                weights,
            ))
        }
        "Conv" => {
            let mut weights = Vec::new();
            let mut filters = 0usize;
            let mut kernel = 3;
            let mut stride = 1;
            let mut padding = "valid";
            let mut groups = 1usize;

            if let Some(w_name) = node.input.get(1)
                && let Some(tensor) = resolved_tensors.get(w_name.as_str())
            {
                filters = tensor.dims.first().copied().unwrap_or(0) as usize;
                if tensor.dims.len() >= 4 {
                    kernel = tensor.dims[2] as usize;
                }
                weights.push(WeightRef {
                    tensor_name: w_name.clone(),
                    param_name: "weight".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }
            if let Some(b_name) = node.input.get(2)
                && resolved_tensors.contains_key(b_name.as_str())
            {
                weights.push(WeightRef {
                    tensor_name: b_name.clone(),
                    param_name: "bias".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }

            if weights.iter().any(|w| w.param_name == "weight") && filters == 0 {
                return Err(ImportError {
                    message: format!(
                        "cannot infer Conv2D filters for `{}` from ONNX weights",
                        node.name
                    ),
                });
            }

            for attr in &node.attribute {
                match attr.name.as_str() {
                    "strides" => {
                        if let Some(&s) = attr.ints.first() {
                            stride = s as usize;
                        }
                    }
                    "pads" if attr.ints.iter().any(|&p| p > 0) => {
                        padding = "same";
                    }
                    "kernel_shape" => {
                        if let Some(&k) = attr.ints.first() {
                            kernel = k as usize;
                        }
                    }
                    "group" => {
                        groups = attr.i as usize;
                    }
                    _ => {}
                }
            }

            let groups_str = if groups > 1 {
                format!(", groups: {groups}")
            } else {
                String::new()
            };
            Ok((
                layer_id,
                Some(format!(
                    "Conv2D(filters: {filters}, kernel: {kernel}, stride: {stride}, padding: \"{padding}\"{groups_str})"
                )),
                weights,
            ))
        }
        "MaxPool"
            if node
                .attribute
                .iter()
                .any(|a| a.name == "kernel_shape" && a.ints.len() == 1) =>
        {
            let mut kernel = 2;
            for attr in &node.attribute {
                if attr.name == "kernel_shape"
                    && let Some(&k) = attr.ints.first()
                {
                    kernel = k as usize;
                }
            }
            Ok((
                layer_id,
                Some(format!("MaxPool1D(kernel: {kernel})")),
                Vec::new(),
            ))
        }
        "MaxPool" => {
            let mut kernel = 2;
            let mut stride = None;
            for attr in &node.attribute {
                match attr.name.as_str() {
                    "kernel_shape" => {
                        if let Some(&k) = attr.ints.first() {
                            kernel = k as usize;
                        }
                    }
                    "strides" => {
                        if let Some(&s) = attr.ints.first() {
                            stride = Some(s as usize);
                        }
                    }
                    _ => {}
                }
            }
            let stride_str = stride.map(|s| format!(", stride: {s}")).unwrap_or_default();
            Ok((
                layer_id,
                Some(format!("MaxPool2D(kernel: {kernel}{stride_str})")),
                Vec::new(),
            ))
        }
        "AveragePool" => {
            let mut kernel = 2;
            let mut stride = None;
            for attr in &node.attribute {
                match attr.name.as_str() {
                    "kernel_shape" => {
                        if let Some(&k) = attr.ints.first() {
                            kernel = k as usize;
                        }
                    }
                    "strides" => {
                        if let Some(&s) = attr.ints.first() {
                            stride = Some(s as usize);
                        }
                    }
                    _ => {}
                }
            }
            let stride_str = stride.map(|s| format!(", stride: {s}")).unwrap_or_default();
            Ok((
                layer_id,
                Some(format!("AvgPool2D(kernel: {kernel}{stride_str})")),
                Vec::new(),
            ))
        }
        "Flatten" => Ok((layer_id, Some("Flatten()".to_string()), Vec::new())),
        "Reshape" if should_lower_reshape_to_flatten(node, tensor_shapes, resolved_tensors) => {
            Ok((layer_id, Some("Flatten()".to_string()), Vec::new()))
        }
        "BatchNormalization" => {
            let mut weights = Vec::new();
            let param_names = ["scale", "bias", "mean", "var"];
            for (i, pname) in param_names.iter().enumerate() {
                if let Some(inp_name) = node.input.get(i + 1)
                    && resolved_tensors.contains_key(inp_name.as_str())
                {
                    weights.push(WeightRef {
                        tensor_name: inp_name.clone(),
                        param_name: pname.to_string(),
                        transpose: false,
                        chw_to_hwc: None,
                    });
                }
            }
            Ok((layer_id, Some("BatchNorm()".to_string()), weights))
        }
        "Dropout" => {
            let mut ratio = 0.5f32;
            for attr in &node.attribute {
                if attr.name == "ratio" {
                    ratio = attr.f;
                }
            }
            Ok((
                layer_id,
                Some(format!("Dropout(rate: {ratio})")),
                Vec::new(),
            ))
        }
        "Add" => Ok((layer_id, Some("Add()".to_string()), Vec::new())),
        "Concat" => {
            let axis = remap_axis_for_shape(
                node,
                tensor_shapes
                    .get(node.input[0].as_str())
                    .unwrap_or(&Vec::new()),
            )
            .unwrap_or(-1);
            Ok((layer_id, Some(format!("Concat(axis: {axis})")), Vec::new()))
        }
        "Relu" => Ok((layer_id, Some("ReLU()".to_string()), Vec::new())),
        "Sigmoid" => Ok((layer_id, Some("Sigmoid()".to_string()), Vec::new())),
        "Softmax" => {
            let axis = remap_axis_for_shape(
                node,
                tensor_shapes
                    .get(node.input[0].as_str())
                    .unwrap_or(&Vec::new()),
            )
            .unwrap_or(-1);
            Ok((layer_id, Some(format!("Softmax(axis: {axis})")), Vec::new()))
        }
        "GlobalAveragePool" => Ok((layer_id, Some("GlobalAvgPool2D()".to_string()), Vec::new())),
        "LeakyRelu" => {
            let alpha = node
                .attribute
                .iter()
                .find(|a| a.name == "alpha")
                .map(|a| a.f)
                .unwrap_or(0.01);
            Ok((
                layer_id,
                Some(format!("LeakyReLU(alpha: {alpha})")),
                Vec::new(),
            ))
        }
        "Clip" => {
            // Clip(min=0, max=6) → ReLU6
            let min_val = node
                .attribute
                .iter()
                .find(|a| a.name == "min")
                .map(|a| a.f)
                .unwrap_or(f32::MIN);
            let max_val = node
                .attribute
                .iter()
                .find(|a| a.name == "max")
                .map(|a| a.f)
                .unwrap_or(f32::MAX);
            if min_val == 0.0 && max_val == 6.0 {
                Ok((layer_id, Some("ReLU6()".to_string()), Vec::new()))
            } else {
                Ok((layer_id, None, Vec::new()))
            }
        }
        "Mul" => Ok((layer_id, Some("Mul()".to_string()), Vec::new())),
        "HardSwish" => Ok((layer_id, Some("Hardswish()".to_string()), Vec::new())),
        "LayerNormalization" => {
            let epsilon = node
                .attribute
                .iter()
                .find(|a| a.name == "epsilon")
                .map(|a| a.f as f64)
                .unwrap_or(1e-5);
            let mut weights = Vec::new();
            if let Some(s_name) = node.input.get(1)
                && resolved_tensors.contains_key(s_name.as_str())
            {
                weights.push(WeightRef {
                    tensor_name: s_name.clone(),
                    param_name: "scale".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }
            if let Some(b_name) = node.input.get(2)
                && resolved_tensors.contains_key(b_name.as_str())
            {
                weights.push(WeightRef {
                    tensor_name: b_name.clone(),
                    param_name: "bias".to_string(),
                    transpose: false,
                    chw_to_hwc: None,
                });
            }
            Ok((
                layer_id,
                Some(format!("LayerNorm(epsilon: {epsilon})")),
                weights,
            ))
        }
        "Upsample" | "Resize" => {
            let mut scale = 2usize;
            // Try scales from attribute (Upsample opset < 10)
            if let Some(attr) = node.attribute.iter().find(|a| a.name == "scales") {
                // scales is [N, C, H, W]; take H scale (index 2)
                if let Some(&s) = attr.floats.get(2) {
                    scale = s as usize;
                }
            }
            // Try scales from initializer input (Upsample opset 9+, Resize)
            let scales_idx = if op == "Resize" { 2 } else { 1 };
            if let Some(scales_name) = node.input.get(scales_idx)
                && let Some(tensor) = resolved_tensors.get(scales_name.as_str())
                && let Some(&s) = tensor.data.get(2)
            {
                scale = s as usize;
            }
            Ok((
                layer_id,
                Some(format!("Upsample(scale: {scale})")),
                Vec::new(),
            ))
        }
        _ => Ok((layer_id, None, Vec::new())),
    }
}

fn build_resolved_tensors(
    nodes: &[NodeProto],
    initializers: &HashMap<&str, &TensorProto>,
    onnx_dir: &Path,
) -> Result<HashMap<String, ResolvedTensor>, ImportError> {
    let mut resolved = HashMap::new();

    for (&name, tensor) in initializers {
        resolved.insert(
            name.to_string(),
            ResolvedTensor {
                dims: tensor.dims.clone(),
                data: decode_tensor_data(tensor, onnx_dir)?,
            },
        );
    }

    for node in nodes {
        if node.op_type != "DequantizeLinear" {
            continue;
        }
        let (Some(input_name), Some(scale_name), Some(output_name)) =
            (node.input.first(), node.input.get(1), node.output.first())
        else {
            continue;
        };
        let Some(input_tensor) = resolved.get(input_name).cloned() else {
            continue;
        };
        let Some(scale_tensor) = resolved.get(scale_name) else {
            continue;
        };
        let zero_point_tensor = node.input.get(2).and_then(|name| resolved.get(name));
        let axis = node
            .attribute
            .iter()
            .find(|attr| attr.name == "axis")
            .map(|attr| attr.i)
            .unwrap_or(1);
        let data = dequantize_tensor(
            &input_tensor.data,
            &input_tensor.dims,
            &scale_tensor.data,
            zero_point_tensor.map(|tensor| tensor.data.as_slice()),
            axis,
        )?;
        resolved.insert(
            output_name.clone(),
            ResolvedTensor {
                dims: input_tensor.dims,
                data,
            },
        );
    }

    Ok(resolved)
}

fn decode_tensor_data(tensor: &TensorProto, onnx_dir: &Path) -> Result<Vec<f32>, ImportError> {
    match tensor.data_type {
        1 => Ok(tensor.to_f32_vec(onnx_dir)),
        2 => Ok(tensor.raw_data.iter().map(|&v| v as f32).collect()),
        3 => Ok(tensor.raw_data.iter().map(|&v| (v as i8) as f32).collect()),
        6 => Ok(tensor
            .raw_data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()),
        7 => Ok(tensor
            .raw_data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()),
        other => Err(ImportError {
            message: format!(
                "unsupported ONNX tensor data type {other} for `{}`",
                tensor.name
            ),
        }),
    }
}

fn dequantize_tensor(
    data: &[f32],
    dims: &[i64],
    scales: &[f32],
    zero_points: Option<&[f32]>,
    axis: i64,
) -> Result<Vec<f32>, ImportError> {
    if scales.is_empty() {
        return Err(ImportError {
            message: "DequantizeLinear is missing scale data".to_string(),
        });
    }

    let zeros_storage;
    let zeros = if let Some(zp) = zero_points {
        zp
    } else {
        zeros_storage = vec![0.0f32; scales.len()];
        &zeros_storage
    };

    if scales.len() != 1 && scales.len() != zeros.len() {
        return Err(ImportError {
            message: "DequantizeLinear scale and zero-point sizes differ".to_string(),
        });
    }

    if scales.len() == 1 {
        let scale = scales[0];
        let zero = zeros[0];
        return Ok(data.iter().map(|value| (value - zero) * scale).collect());
    }

    let axis = normalize_axis(axis, dims.len());
    let axis_len = dims.get(axis).copied().unwrap_or(0) as usize;
    if axis_len != scales.len() {
        return Err(ImportError {
            message: format!(
                "DequantizeLinear axis length {} does not match scale length {}",
                axis_len,
                scales.len()
            ),
        });
    }

    let inner_stride = dims[axis + 1..]
        .iter()
        .fold(1usize, |acc, dim| acc.saturating_mul(*dim as usize));
    let outer_stride = if axis_len == 0 {
        0
    } else {
        inner_stride * axis_len
    };
    let mut out = Vec::with_capacity(data.len());

    for (index, value) in data.iter().enumerate() {
        let axis_index = if outer_stride == 0 {
            0
        } else {
            (index % outer_stride) / inner_stride
        };
        out.push((value - zeros[axis_index]) * scales[axis_index]);
    }
    Ok(out)
}

fn remap_axis_for_shape(node: &NodeProto, input_shape: &[i64]) -> Option<i64> {
    let axis = node
        .attribute
        .iter()
        .find(|attr| attr.name == "axis")
        .map(|attr| attr.i)
        .unwrap_or(-1);

    if input_shape.is_empty() {
        return Some(axis);
    }

    let rank = input_shape.len();
    let normalized = normalize_axis(axis, rank);
    if rank <= 1 {
        return Some(normalized as i64);
    }
    if normalized == 0 {
        return None;
    }
    if rank == 4 {
        return Some(match normalized {
            1 => 2,
            2 => 0,
            3 => 1,
            _ => normalized as i64 - 1,
        });
    }
    Some((normalized - 1) as i64)
}

fn should_lower_reshape_to_flatten(
    node: &NodeProto,
    tensor_shapes: &HashMap<String, Vec<i64>>,
    resolved_tensors: &HashMap<String, ResolvedTensor>,
) -> bool {
    let Some(input_name) = node.input.first() else {
        return false;
    };
    let Some(input_shape) = tensor_shapes.get(input_name.as_str()) else {
        return false;
    };
    if input_shape.len() < 3 {
        return false;
    }
    let Some(shape_name) = node.input.get(1) else {
        return false;
    };
    let Some(target_shape) = resolved_tensors.get(shape_name.as_str()) else {
        return false;
    };
    let target_dims: Vec<i64> = target_shape
        .data
        .iter()
        .map(|value| *value as i64)
        .collect();
    if target_dims.len() != 2 {
        return false;
    }
    let batch = input_shape[0];
    let flattened: i64 = input_shape.iter().skip(1).product();
    (target_dims[0] == batch || target_dims[0] == 1) && target_dims[1] == flattened
}

fn find_fusable_add_bias(
    node_index: usize,
    nodes: &[NodeProto],
    resolved_tensors: &HashMap<String, ResolvedTensor>,
) -> Option<String> {
    let node = nodes.get(node_index)?;
    if !matches!(node.op_type.as_str(), "Gemm" | "MatMul") {
        return None;
    }
    let next = nodes.get(node_index + 1)?;
    if next.op_type != "Add" || node.output.len() != 1 {
        return None;
    }
    let output_name = node.output.first()?;
    let left = next.input.first()?;
    let right = next.input.get(1)?;
    if left == output_name && resolved_tensors.contains_key(right.as_str()) {
        return Some(right.clone());
    }
    if right == output_name && resolved_tensors.contains_key(left.as_str()) {
        return Some(left.clone());
    }
    None
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
}

fn make_layer_id(node: &NodeProto, op: &str, counter: &mut HashMap<String, usize>) -> String {
    if !node.name.is_empty() {
        return sanitize_id(&node.name);
    }
    let base = match op {
        "Gemm" | "MatMul" => "dense",
        "Conv" => "conv",
        "MaxPool" => "maxpool",
        "AveragePool" => "avgpool",
        "Flatten" => "flatten",
        "BatchNormalization" => "bn",
        "Dropout" => "dropout",
        "Add" => "add",
        "Concat" => "concat",
        "Relu" => "relu",
        "Sigmoid" => "sigmoid",
        "Softmax" => "softmax",
        "GlobalAveragePool" => "gap",
        "LeakyRelu" => "leaky_relu",
        "Clip" => "relu6",
        "Mul" => "mul",
        "HardSwish" => "hardswish",
        "LayerNormalization" => "ln",
        "Upsample" => "upsample",
        "Resize" => "resize",
        other => other,
    };
    let count = counter.entry(base.to_string()).or_insert(0);
    *count += 1;
    let id = if *count == 1 {
        base.to_string()
    } else {
        format!("{base}{count}")
    };
    sanitize_id(&id)
}

fn sanitize_id(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if i == 0 && c.is_ascii_digit() {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("layer");
    }
    out
}

fn detect_input_shape(graph: &onnx::GraphProto) -> Vec<String> {
    let init_names: HashSet<&str> = graph.initializer.iter().map(|t| t.name.as_str()).collect();

    for inp in &graph.input {
        if init_names.contains(inp.name.as_str()) {
            continue;
        }
        if let Some(tp) = &inp.r#type
            && let Some(tt) = &tp.tensor_type
            && let Some(shape) = &tt.shape
        {
            // Skip the first dimension (batch) — it's either
            // a symbolic param or a concrete 1.
            let dims: Vec<i64> = shape
                .dim
                .iter()
                .skip(1)
                .filter_map(|d| {
                    if d.dim_value > 0 {
                        Some(d.dim_value)
                    } else {
                        None
                    }
                })
                .collect();
            if !dims.is_empty() {
                if dims.len() == 3 {
                    return vec![
                        dims[1].to_string(),
                        dims[2].to_string(),
                        dims[0].to_string(),
                    ];
                }
                return dims.into_iter().map(|dim| dim.to_string()).collect();
            }
        }
    }

    vec!["1".to_string()]
}

fn is_sequential_flow(connections: &[Connection]) -> bool {
    connections.iter().all(|c| c.from.len() == 1)
        && connections.windows(2).all(|w| w[0].to == w[1].from[0])
}

fn pathdiff(output_path: &Path, weights_dir: &Path) -> String {
    // Compute weights_dir relative to the output file's parent directory
    let output_parent = output_path.parent().unwrap_or(Path::new("."));

    // Canonicalize both paths (creating weights_dir first ensures it exists)
    let abs_out = fs::canonicalize(output_parent)
        .unwrap_or_else(|_| fs::canonicalize(".").unwrap_or_else(|_| output_parent.to_path_buf()));
    let abs_w = fs::canonicalize(weights_dir).unwrap_or_else(|_| weights_dir.to_path_buf());

    // Build relative path from output dir to weights dir
    let from_parts: Vec<_> = abs_out.components().collect();
    let to_parts: Vec<_> = abs_w.components().collect();

    // Find common prefix length
    let common = from_parts
        .iter()
        .zip(to_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let ups = from_parts.len() - common;
    let mut rel = String::new();
    if ups == 0 {
        rel.push_str("./");
    } else {
        for _ in 0..ups {
            rel.push_str("../");
        }
    }
    let tail: Vec<_> = to_parts[common..]
        .iter()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    rel.push_str(&tail.join("/"));
    rel
}

/// Permute columns of a Dense weight matrix from CHW flatten order to HWC flatten order.
///
/// In CHW order, the flattened index for (c, h, w) is `c * H * W + h * W + w`.
/// In HWC order, the flattened index for (h, w, c) is `(h * W + w) * C + c`.
///
/// The weight matrix has shape `[in_features, out_units]` (after transpose).
/// We permute the rows (input dimension) so that position `hwc_idx` gets the
/// value from position `chw_idx`.
fn permute_dense_weight_chw_to_hwc(
    data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    out_units: usize,
) -> Vec<f32> {
    let in_features = channels * height * width;
    let mut permuted = vec![0.0f32; data.len()];

    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                let chw_idx = c * height * width + h * width + w;
                let hwc_idx = (h * width + w) * channels + c;
                // Copy the entire row (all output units) for this input feature
                let src_offset = chw_idx * out_units;
                let dst_offset = hwc_idx * out_units;
                permuted[dst_offset..dst_offset + out_units]
                    .copy_from_slice(&data[src_offset..src_offset + out_units]);
            }
        }
    }

    // Copy any remaining data beyond the permuted region (shouldn't happen, but safe)
    if data.len() > in_features * out_units {
        permuted[in_features * out_units..].copy_from_slice(&data[in_features * out_units..]);
    }

    permuted
}

fn write_weight_npy(path: &Path, shape: &[u64], data: &[f32]) -> Result<(), ImportError> {
    let file = File::create(path).map_err(|e| ImportError {
        message: format!("cannot write `{}`: {e}", path.display()),
    })?;
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(file)
        .begin_nd()
        .map_err(|e| ImportError {
            message: format!("npy write error: {e}"),
        })?;
    writer
        .extend(data.iter().copied())
        .map_err(|e| ImportError {
            message: format!("npy write error: {e}"),
        })?;
    writer.finish().map_err(|e| ImportError {
        message: format!("npy write error: {e}"),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::onnx::{
        AttributeProto, Dimension, GraphProto, TensorShapeProto, TensorTypeProto, TypeProto,
        ValueInfoProto,
    };
    use tempfile::tempdir;

    #[test]
    fn import_remaps_nchw_input_concat_and_pool_stride() {
        let model = ModelProto {
            ir_version: 8,
            opset_import: Vec::new(),
            graph: Some(GraphProto {
                name: "cnn".to_string(),
                input: vec![value_info("input", &[1, 3, 8, 8])],
                output: Vec::new(),
                initializer: vec![float_tensor("conv_w", &[4, 3, 1, 1], &[1.0; 12])],
                node: vec![
                    node(
                        "conv",
                        "Conv",
                        &["input", "conv_w"],
                        &["conv_out"],
                        &[ints_attr("kernel_shape", &[1, 1])],
                    ),
                    node(
                        "pool",
                        "MaxPool",
                        &["conv_out"],
                        &["pool_out"],
                        &[
                            ints_attr("kernel_shape", &[3, 3]),
                            ints_attr("strides", &[2, 2]),
                        ],
                    ),
                    node(
                        "cat",
                        "Concat",
                        &["conv_out", "conv_out"],
                        &["cat_out"],
                        &[int_attr("axis", 1)],
                    ),
                ],
            }),
        };

        let imported = import_model_to_string(&model).unwrap();
        assert!(imported.contains("Input(shape: [8, 8, 3])"));
        assert!(imported.contains("MaxPool2D(kernel: 3, stride: 2)"));
        assert!(imported.contains("Concat(axis: 2)"));
    }

    #[test]
    fn import_fails_early_when_conv_weight_shape_is_invalid() {
        let model = ModelProto {
            ir_version: 8,
            opset_import: Vec::new(),
            graph: Some(GraphProto {
                name: "bad_conv".to_string(),
                input: vec![value_info("input", &[1, 3, 8, 8])],
                output: Vec::new(),
                initializer: vec![float_tensor("conv_w", &[0, 3, 1, 1], &[])],
                node: vec![node(
                    "conv",
                    "Conv",
                    &["input", "conv_w"],
                    &["conv_out"],
                    &[ints_attr("kernel_shape", &[1, 1])],
                )],
            }),
        };

        let err = import_model_to_string(&model).unwrap_err();
        assert!(err.message.contains("cannot infer Conv2D filters"));
    }

    #[test]
    fn import_dequantizes_initializer_backed_conv_weights() {
        let model = ModelProto {
            ir_version: 8,
            opset_import: Vec::new(),
            graph: Some(GraphProto {
                name: "qdq_conv".to_string(),
                input: vec![value_info("input", &[1, 3, 8, 8])],
                output: Vec::new(),
                initializer: vec![
                    uint8_tensor("conv_w_q", &[4, 3, 1, 1], &[2; 12]),
                    float_tensor("conv_scale", &[1], &[0.5]),
                    uint8_tensor("conv_zp", &[1], &[1]),
                ],
                node: vec![
                    node(
                        "dq_w",
                        "DequantizeLinear",
                        &["conv_w_q", "conv_scale", "conv_zp"],
                        &["conv_w"],
                        &[],
                    ),
                    node(
                        "conv",
                        "Conv",
                        &["input", "conv_w"],
                        &["conv_out"],
                        &[ints_attr("kernel_shape", &[1, 1])],
                    ),
                ],
            }),
        };

        let tmp = tempdir().unwrap();
        let onnx_path = tmp.path().join("model.onnx");
        let output_path = tmp.path().join("model.nnl");
        let weights_dir = tmp.path().join("weights");
        fs::write(&onnx_path, model.encode_to_vec()).unwrap();

        import_onnx(&onnx_path, &output_path, &weights_dir).unwrap();
        let imported = fs::read_to_string(&output_path).unwrap();
        assert!(imported.contains("Conv2D(filters: 4, kernel: 1, stride: 1, padding: \"valid\")"));
        assert!(!imported.contains("UNSUPPORTED: DequantizeLinear"));
        assert!(weights_dir.join("conv.weight.npy").exists());
    }

    #[test]
    fn import_lowers_classifier_tail_reshape_and_fuses_bias_add() {
        let model = ModelProto {
            ir_version: 8,
            opset_import: Vec::new(),
            graph: Some(GraphProto {
                name: "classifier_tail".to_string(),
                input: vec![value_info("input", &[1, 4, 1, 1])],
                output: Vec::new(),
                initializer: vec![
                    int64_tensor("shape", &[2], &[1, 4]),
                    float_tensor(
                        "dense_w",
                        &[4, 3],
                        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    ),
                    float_tensor("dense_b", &[3], &[0.5, 1.5, 2.5]),
                ],
                node: vec![
                    node("reshape", "Reshape", &["input", "shape"], &["flat"], &[]),
                    node("matmul", "MatMul", &["flat", "dense_w"], &["mm_out"], &[]),
                    node("bias_add", "Add", &["mm_out", "dense_b"], &["biased"], &[]),
                ],
            }),
        };

        let imported = import_model_to_string(&model).unwrap();
        assert!(imported.contains("Flatten()"));
        assert!(imported.contains("Dense(units: 3)"));
        assert!(!imported.contains("layer bias_add = Add()"));
    }

    fn import_model_to_string(model: &ModelProto) -> Result<String, ImportError> {
        let tmp = tempdir().unwrap();
        let onnx_path = tmp.path().join("model.onnx");
        let output_path = tmp.path().join("model.nnl");
        let weights_dir = tmp.path().join("weights");
        fs::write(&onnx_path, model.encode_to_vec()).unwrap();
        import_onnx(&onnx_path, &output_path, &weights_dir)?;
        Ok(fs::read_to_string(output_path).unwrap())
    }

    fn value_info(name: &str, dims: &[i64]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type: 1,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|&dim_value| Dimension {
                                dim_value,
                                dim_param: String::new(),
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn float_tensor(name: &str, dims: &[i64], data: &[f32]) -> TensorProto {
        TensorProto {
            dims: dims.to_vec(),
            data_type: 1,
            name: name.to_string(),
            raw_data: data.iter().flat_map(|value| value.to_le_bytes()).collect(),
            float_data: Vec::new(),
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    fn uint8_tensor(name: &str, dims: &[i64], data: &[u8]) -> TensorProto {
        TensorProto {
            dims: dims.to_vec(),
            data_type: 2,
            name: name.to_string(),
            raw_data: data.to_vec(),
            float_data: Vec::new(),
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    fn int64_tensor(name: &str, dims: &[i64], data: &[i64]) -> TensorProto {
        TensorProto {
            dims: dims.to_vec(),
            data_type: 7,
            name: name.to_string(),
            raw_data: data.iter().flat_map(|value| value.to_le_bytes()).collect(),
            float_data: Vec::new(),
            external_data: Vec::new(),
            data_location: 0,
        }
    }

    fn node(
        name: &str,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attribute: &[AttributeProto],
    ) -> NodeProto {
        NodeProto {
            input: inputs.iter().map(|s| (*s).to_string()).collect(),
            output: outputs.iter().map(|s| (*s).to_string()).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            attribute: attribute.to_vec(),
        }
    }

    fn ints_attr(name: &str, ints: &[i64]) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: 0.0,
            i: 0,
            s: Vec::new(),
            floats: Vec::new(),
            ints: ints.to_vec(),
            r#type: 0,
        }
    }

    fn int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: 0.0,
            i: value,
            s: Vec::new(),
            floats: Vec::new(),
            ints: Vec::new(),
            r#type: 0,
        }
    }
}
