pub mod onnx;

use std::collections::HashMap;
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

    for node in &graph.node {
        let (layer_id, layer_def, weight_info) = map_node(node, &initializers, &mut layer_counter)?;

        if let Some(def) = layer_def {
            // Write weights
            for wi in &weight_info {
                let tensor = initializers.get(wi.initializer_name.as_str()).unwrap();
                let data = tensor.to_f32_vec();
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
                    write_weight_npy(&path, &[dims[1], dims[0]], &transposed)?;
                } else {
                    write_weight_npy(&path, &dims, &data)?;
                }
            }

            // Map all ONNX outputs of this node to the layer id
            for out in &node.output {
                output_map.insert(out.clone(), layer_id.clone());
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
            // Add as comment
            layers.push(LayerDef {
                id: String::new(),
                def: format!("// UNSUPPORTED: {}({})", node.op_type, node.name),
            });
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
    initializer_name: String,
    param_name: String,
    transpose: bool,
}

fn map_node(
    node: &NodeProto,
    initializers: &HashMap<&str, &TensorProto>,
    counter: &mut HashMap<String, usize>,
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
            // Weight is typically the second input
            if let Some(w_name) = node.input.get(1)
                && let Some(tensor) = initializers.get(w_name.as_str())
            {
                // transB=1: shape [out, in], first dim is units
                // transB=0: shape [in, out], last dim is units
                units = if trans_b != 0 {
                    *tensor.dims.first().unwrap_or(&0) as usize
                } else {
                    *tensor.dims.last().unwrap_or(&0) as usize
                };
                weights.push(WeightRef {
                    initializer_name: w_name.clone(),
                    param_name: "weight".to_string(),
                    transpose: trans_b != 0,
                });
            }
            if let Some(b_name) = node.input.get(2)
                && initializers.contains_key(b_name.as_str())
            {
                weights.push(WeightRef {
                    initializer_name: b_name.clone(),
                    param_name: "bias".to_string(),
                    transpose: false,
                });
            }
            Ok((layer_id, Some(format!("Dense(units: {units})")), weights))
        }
        "Conv" => {
            let mut weights = Vec::new();
            let mut filters = 0usize;
            let mut kernel = 3;
            let mut stride = 1;
            let mut padding = "valid";

            if let Some(w_name) = node.input.get(1)
                && let Some(tensor) = initializers.get(w_name.as_str())
            {
                filters = tensor.dims.first().copied().unwrap_or(0) as usize;
                if tensor.dims.len() >= 4 {
                    kernel = tensor.dims[2] as usize;
                }
                weights.push(WeightRef {
                    initializer_name: w_name.clone(),
                    param_name: "weight".to_string(),
                    transpose: false,
                });
            }
            if let Some(b_name) = node.input.get(2)
                && initializers.contains_key(b_name.as_str())
            {
                weights.push(WeightRef {
                    initializer_name: b_name.clone(),
                    param_name: "bias".to_string(),
                    transpose: false,
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
                    "Conv2D(filters: {filters}, kernel: {kernel}, stride: {stride}, padding: \"{padding}\")"
                )),
                weights,
            ))
        }
        "MaxPool" => {
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
                Some(format!("MaxPool2D(kernel: {kernel})")),
                Vec::new(),
            ))
        }
        "AveragePool" => {
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
                Some(format!("AvgPool2D(kernel: {kernel})")),
                Vec::new(),
            ))
        }
        "Flatten" => Ok((layer_id, Some("Flatten()".to_string()), Vec::new())),
        "BatchNormalization" => {
            let mut weights = Vec::new();
            let param_names = ["scale", "bias", "mean", "var"];
            for (i, pname) in param_names.iter().enumerate() {
                if let Some(inp_name) = node.input.get(i + 1)
                    && initializers.contains_key(inp_name.as_str())
                {
                    weights.push(WeightRef {
                        initializer_name: inp_name.clone(),
                        param_name: pname.to_string(),
                        transpose: false,
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
        "Concat" => Ok((layer_id, Some("Concat()".to_string()), Vec::new())),
        "Relu" => Ok((layer_id, Some("ReLU()".to_string()), Vec::new())),
        "Sigmoid" => Ok((layer_id, Some("Sigmoid()".to_string()), Vec::new())),
        "Softmax" => Ok((layer_id, Some("Softmax()".to_string()), Vec::new())),
        _ => Ok((layer_id, None, Vec::new())),
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
    let init_names: std::collections::HashSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();

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
            let dims: Vec<String> = shape
                .dim
                .iter()
                .skip(1)
                .filter_map(|d| {
                    if d.dim_value > 0 {
                        Some(d.dim_value.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            if !dims.is_empty() {
                return dims;
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
