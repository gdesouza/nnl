use std::fmt::Write;

use crate::ir::model::{Activation, LayerKind, Model, Preprocess};
use crate::sema::shapes::ShapeInfo;
use crate::weights::WeightSet;

/// Sanitize a model name into a valid C identifier.
fn c_ident(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Convert a weight key (e.g. "fc1.weight") to a C variable name ("nnc_fc1_weight").
fn weight_var(key: &str) -> String {
    let sanitized: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("nnc_{sanitized}")
}

/// Product of all elements in a shape (number of elements, excluding batch).
fn shape_elems(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

/// Generate the C header for a model.
pub fn emit_header(model: &Model, shape_info: &ShapeInfo) -> String {
    let name = c_ident(&model.name);
    let guard = name.to_ascii_uppercase();

    let _ = shape_info; // used in future expansions

    let mut h = String::new();
    writeln!(h, "#ifndef {guard}_H").unwrap();
    writeln!(h, "#define {guard}_H").unwrap();
    writeln!(h).unwrap();
    writeln!(h, "#include <stdint.h>").unwrap();
    writeln!(h).unwrap();
    writeln!(h, "int {name}_infer(const void *input, void *output);").unwrap();
    writeln!(h, "int {name}_input_size(void);").unwrap();
    writeln!(h, "int {name}_output_size(void);").unwrap();
    writeln!(h).unwrap();
    writeln!(h, "#endif /* {guard}_H */").unwrap();
    h
}

/// Generate the C source for a model.
pub fn emit_source(
    model: &Model,
    shape_info: &ShapeInfo,
    weights: &WeightSet,
    topo_order: &[String],
) -> String {
    let name = c_ident(&model.name);
    let align = model.config.align;
    let batch = model.config.batch;

    let mut c = String::new();

    // ── Includes ──────────────────────────────────────────────────
    writeln!(c, "#include <math.h>").unwrap();
    writeln!(c, "#include <string.h>").unwrap();
    writeln!(c, "#include <stdint.h>").unwrap();
    writeln!(c).unwrap();

    // ── Weight arrays ────────────────────────────────────────────
    for (key, tensor) in weights.iter() {
        let var = weight_var(key);
        let n = tensor.len();
        let floats = tensor.as_f32_slice();
        write!(
            c,
            "static const float {var}[{n}] __attribute__((aligned({align}))) = {{",
        )
        .unwrap();
        for (i, v) in floats.iter().enumerate() {
            if i > 0 {
                write!(c, ", ").unwrap();
            }
            write!(c, "{v:.8}f").unwrap();
        }
        writeln!(c, "}};").unwrap();
    }
    writeln!(c).unwrap();

    // ── Determine input / output sizes & workspace size ──────────
    let input_layer_id = topo_order.first().expect("empty topo order");
    let output_layer_id = topo_order.last().expect("empty topo order");

    let input_elems = shape_elems(shape_info.shapes.get(input_layer_id).unwrap()) * batch;
    let output_elems = shape_elems(shape_info.shapes.get(output_layer_id).unwrap()) * batch;

    let max_activation: usize = shape_info
        .shapes
        .values()
        .map(|s| shape_elems(s) * batch)
        .max()
        .unwrap_or(1);

    // ── Workspace buffers (ping-pong) ────────────────────────────
    writeln!(
        c,
        "static float nnc_workspace_a[{max_activation}] __attribute__((aligned({align})));"
    )
    .unwrap();
    writeln!(
        c,
        "static float nnc_workspace_b[{max_activation}] __attribute__((aligned({align})));"
    )
    .unwrap();
    writeln!(c).unwrap();

    // ── Size helpers ─────────────────────────────────────────────
    writeln!(
        c,
        "int {name}_input_size(void) {{ return {input_elems}; }}"
    )
    .unwrap();
    writeln!(
        c,
        "int {name}_output_size(void) {{ return {output_elems}; }}"
    )
    .unwrap();
    writeln!(c).unwrap();

    // ── Inference function ───────────────────────────────────────
    writeln!(
        c,
        "int {name}_infer(const void *input, void *output) {{"
    )
    .unwrap();
    writeln!(
        c,
        "    const float *in_ptr = (const float *)input;"
    )
    .unwrap();
    writeln!(c, "    float *out_ptr = (float *)output;").unwrap();
    writeln!(c).unwrap();

    // Track which workspace buffer each layer writes to.
    // 0 = workspace_a, 1 = workspace_b, -1 = special (input param)
    let mut buf_map: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let mut next_buf: i32 = 0; // start writing to workspace_a

    for layer_id in topo_order {
        let layer = model.layers.iter().find(|l| l.id == *layer_id).unwrap();
        let out_shape = shape_info.shapes.get(layer_id).unwrap();
        let out_elems = shape_elems(out_shape) * batch;

        writeln!(c, "    /* layer: {} ({}) */", layer_id, layer.kind.type_name()).unwrap();

        match &layer.kind {
            LayerKind::Input { shape } => {
                let n = shape_elems(shape) * batch;
                let dst = buf_name(next_buf);
                // Copy input into workspace, applying preprocessing
                match model.config.preprocess {
                    Preprocess::None => {
                        writeln!(c, "    memcpy({dst}, in_ptr, {n} * sizeof(float));").unwrap();
                    }
                    Preprocess::Normalize01 => {
                        writeln!(c, "    for (int i = 0; i < {n}; i++) {{").unwrap();
                        writeln!(c, "        {dst}[i] = in_ptr[i] / 255.0f;").unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                    Preprocess::Standardize => {
                        emit_standardize_loop(&mut c, n, &model.config.preprocess_mean, &model.config.preprocess_std, &dst);
                    }
                }
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Dense { units, activation } => {
                let src = src_buf_for(layer_id, model, &buf_map);
                let dst = buf_name(next_buf);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let in_features = shape_elems(&in_shape);

                let w_var = weight_var(&format!("{layer_id}.weight"));
                let b_var = weight_var(&format!("{layer_id}.bias"));

                writeln!(c, "    for (int i = 0; i < {units}; i++) {{").unwrap();
                writeln!(c, "        float sum = {b_var}[i];").unwrap();
                writeln!(
                    c,
                    "        for (int j = 0; j < {in_features}; j++) {{"
                )
                .unwrap();
                writeln!(
                    c,
                    "            sum += {w_var}[j * {units} + i] * {src}[j];"
                )
                .unwrap();
                writeln!(c, "        }}").unwrap();

                // Inline activation
                match activation {
                    Activation::None => {
                        writeln!(c, "        {dst}[i] = sum;").unwrap();
                    }
                    Activation::ReLU => {
                        writeln!(c, "        {dst}[i] = sum > 0.0f ? sum : 0.0f;").unwrap();
                    }
                    Activation::Sigmoid => {
                        writeln!(
                            c,
                            "        {dst}[i] = 1.0f / (1.0f + expf(-sum));"
                        )
                        .unwrap();
                    }
                    Activation::Softmax => {
                        // defer softmax to after the loop
                        writeln!(c, "        {dst}[i] = sum;").unwrap();
                    }
                }
                writeln!(c, "    }}").unwrap();

                if matches!(activation, Activation::Softmax) {
                    emit_softmax_block(&mut c, *units, &dst);
                }

                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Flatten | LayerKind::Dropout { .. } => {
                // Identity: copy src to dst (or alias)
                let src = src_buf_for(layer_id, model, &buf_map);
                let dst = buf_name(next_buf);
                writeln!(
                    c,
                    "    memcpy({dst}, {src}, {out_elems} * sizeof(float));"
                )
                .unwrap();
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::ReLU => {
                let src = src_buf_for(layer_id, model, &buf_map);
                let dst = buf_name(next_buf);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(
                    c,
                    "        {dst}[i] = {src}[i] > 0.0f ? {src}[i] : 0.0f;"
                )
                .unwrap();
                writeln!(c, "    }}").unwrap();
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Sigmoid => {
                let src = src_buf_for(layer_id, model, &buf_map);
                let dst = buf_name(next_buf);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(
                    c,
                    "        {dst}[i] = 1.0f / (1.0f + expf(-{src}[i]));"
                )
                .unwrap();
                writeln!(c, "    }}").unwrap();
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Softmax { .. } => {
                let src = src_buf_for(layer_id, model, &buf_map);
                let dst = buf_name(next_buf);
                // Copy then softmax in-place on dst
                writeln!(
                    c,
                    "    memcpy({dst}, {src}, {out_elems} * sizeof(float));"
                )
                .unwrap();
                emit_softmax_block(&mut c, out_elems, &dst);
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::BatchNorm { epsilon } => {
                let src = src_buf_for(layer_id, model, &buf_map);
                let dst = buf_name(next_buf);
                let gamma_var = weight_var(&format!("{layer_id}.gamma"));
                let beta_var = weight_var(&format!("{layer_id}.beta"));
                let mean_var = weight_var(&format!("{layer_id}.running_mean"));
                let var_var = weight_var(&format!("{layer_id}.running_var"));

                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(
                    c,
                    "        {dst}[i] = {gamma_var}[i] * ({src}[i] - {mean_var}[i]) / sqrtf({var_var}[i] + {epsilon:.10}f) + {beta_var}[i];"
                )
                .unwrap();
                writeln!(c, "    }}").unwrap();
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Add => {
                // Sum all inputs into dst
                let input_ids = get_input_layer_ids(layer_id, model);
                let dst = buf_name(next_buf);
                if let Some(first_id) = input_ids.first() {
                    let first_buf = buf_map.get(*first_id).copied().unwrap_or(0);
                    let first_src = buf_name(first_buf);
                    writeln!(
                        c,
                        "    memcpy({dst}, {first_src}, {out_elems} * sizeof(float));"
                    )
                    .unwrap();
                    for add_id in input_ids.iter().skip(1) {
                        let add_buf = buf_map.get(*add_id).copied().unwrap_or(0);
                        let add_src = buf_name(add_buf);
                        writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                        writeln!(c, "        {dst}[i] += {add_src}[i];").unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                }
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Concat { .. } => {
                let input_ids = get_input_layer_ids(layer_id, model);
                let dst = buf_name(next_buf);
                let mut offset = 0usize;
                for cat_id in &input_ids {
                    let cat_shape = shape_info.shapes.get(*cat_id).unwrap();
                    let cat_elems = shape_elems(cat_shape) * batch;
                    let cat_buf = buf_map.get(*cat_id).copied().unwrap_or(0);
                    let cat_src = buf_name(cat_buf);
                    writeln!(
                        c,
                        "    memcpy({dst} + {offset}, {cat_src}, {cat_elems} * sizeof(float));"
                    )
                    .unwrap();
                    offset += cat_elems;
                }
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }

            LayerKind::Conv2D { .. } | LayerKind::MaxPool2D { .. } | LayerKind::AvgPool2D { .. } => {
                // Placeholder: not yet implemented for MVP
                writeln!(c, "    /* TODO: {layer_id} — layer type not yet emitted */").unwrap();
                buf_map.insert(layer_id.clone(), next_buf);
                next_buf = 1 - next_buf;
            }
        }

        writeln!(c).unwrap();
    }

    // Copy final result to output
    let final_buf = buf_map.get(output_layer_id).copied().unwrap_or(0);
    let final_src = buf_name(final_buf);
    writeln!(
        c,
        "    memcpy(out_ptr, {final_src}, {output_elems} * sizeof(float));"
    )
    .unwrap();
    writeln!(c, "    return 0;").unwrap();
    writeln!(c, "}}").unwrap();
    writeln!(c).unwrap();

    // ── Main (stdio) ─────────────────────────────────────────────
    emit_main(&mut c, &name, input_elems, output_elems);

    c
}

// ── Helpers ──────────────────────────────────────────────────────

fn buf_name(idx: i32) -> String {
    if idx == 0 {
        "nnc_workspace_a".to_string()
    } else {
        "nnc_workspace_b".to_string()
    }
}

fn src_buf_for(
    layer_id: &str,
    model: &Model,
    buf_map: &std::collections::HashMap<String, i32>,
) -> String {
    let input_ids = get_input_layer_ids(layer_id, model);
    if let Some(first) = input_ids.first() {
        let idx = buf_map.get(*first).copied().unwrap_or(0);
        buf_name(idx)
    } else {
        "nnc_workspace_a".to_string()
    }
}

fn get_input_layer_ids<'a>(layer_id: &str, model: &'a Model) -> Vec<&'a str> {
    model
        .edges
        .iter()
        .filter(|e| e.target == layer_id)
        .map(|e| e.source.as_str())
        .collect()
}

fn input_shape_for(
    layer_id: &str,
    model: &Model,
    shape_info: &ShapeInfo,
) -> Vec<usize> {
    let ids = get_input_layer_ids(layer_id, model);
    if let Some(first) = ids.first() {
        shape_info.shapes.get(*first).cloned().unwrap_or_default()
    } else {
        Vec::new()
    }
}

fn emit_softmax_block(c: &mut String, n: usize, dst: &str) {
    writeln!(c, "    {{").unwrap();
    writeln!(c, "        float max_val = {dst}[0];").unwrap();
    writeln!(c, "        for (int i = 1; i < {n}; i++) {{").unwrap();
    writeln!(c, "            if ({dst}[i] > max_val) max_val = {dst}[i];").unwrap();
    writeln!(c, "        }}").unwrap();
    writeln!(c, "        float sum = 0.0f;").unwrap();
    writeln!(c, "        for (int i = 0; i < {n}; i++) {{").unwrap();
    writeln!(c, "            {dst}[i] = expf({dst}[i] - max_val);").unwrap();
    writeln!(c, "            sum += {dst}[i];").unwrap();
    writeln!(c, "        }}").unwrap();
    writeln!(c, "        for (int i = 0; i < {n}; i++) {{").unwrap();
    writeln!(c, "            {dst}[i] /= sum;").unwrap();
    writeln!(c, "        }}").unwrap();
    writeln!(c, "    }}").unwrap();
}

fn emit_standardize_loop(
    c: &mut String,
    n: usize,
    mean: &Option<Vec<f64>>,
    std_dev: &Option<Vec<f64>>,
    dst: &str,
) {
    match (mean, std_dev) {
        (Some(m), Some(s)) if m.len() == 1 && s.len() == 1 => {
            let mu = m[0] as f32;
            let sigma = s[0] as f32;
            writeln!(c, "    for (int i = 0; i < {n}; i++) {{").unwrap();
            writeln!(
                c,
                "        {dst}[i] = (in_ptr[i] - {mu:.8}f) / {sigma:.8}f;"
            )
            .unwrap();
            writeln!(c, "    }}").unwrap();
        }
        (Some(m), Some(s)) => {
            let channels = m.len();
            let per_ch = n / channels;
            writeln!(c, "    for (int i = 0; i < {n}; i++) {{").unwrap();
            writeln!(c, "        int ch = i / {per_ch};").unwrap();
            write!(c, "        const float mu[] = {{").unwrap();
            for (i, v) in m.iter().enumerate() {
                if i > 0 {
                    write!(c, ", ").unwrap();
                }
                write!(c, "{:.8}f", *v as f32).unwrap();
            }
            writeln!(c, "}};").unwrap();
            write!(c, "        const float sd[] = {{").unwrap();
            for (i, v) in s.iter().enumerate() {
                if i > 0 {
                    write!(c, ", ").unwrap();
                }
                write!(c, "{:.8}f", *v as f32).unwrap();
            }
            writeln!(c, "}};").unwrap();
            writeln!(
                c,
                "        {dst}[i] = (in_ptr[i] - mu[ch]) / sd[ch];"
            )
            .unwrap();
            writeln!(c, "    }}").unwrap();
        }
        _ => {
            // Fallback: identity copy
            writeln!(c, "    memcpy({dst}, in_ptr, {n} * sizeof(float));").unwrap();
        }
    }
}

fn emit_main(c: &mut String, name: &str, input_size: usize, output_size: usize) {
    writeln!(c, "#include <stdio.h>").unwrap();
    writeln!(c, "#include <stdlib.h>").unwrap();
    writeln!(c).unwrap();
    writeln!(c, "int main(void) {{").unwrap();
    writeln!(c, "    float input[{input_size}];").unwrap();
    writeln!(c, "    float output[{output_size}];").unwrap();
    writeln!(
        c,
        "    size_t n = fread(input, sizeof(float), {input_size}, stdin);"
    )
    .unwrap();
    writeln!(c, "    if (n != {input_size}) {{").unwrap();
    writeln!(
        c,
        "        fprintf(stderr, \"expected {input_size} floats\\n\");"
    )
    .unwrap();
    writeln!(c, "        return 1;").unwrap();
    writeln!(c, "    }}").unwrap();
    writeln!(c, "    int rc = {name}_infer(input, output);").unwrap();
    writeln!(c, "    if (rc != 0) return rc;").unwrap();
    writeln!(
        c,
        "    fwrite(output, sizeof(float), {output_size}, stdout);"
    )
    .unwrap();
    writeln!(c, "    return 0;").unwrap();
    writeln!(c, "}}").unwrap();
}
