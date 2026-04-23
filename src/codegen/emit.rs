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

    // ── Buffer allocation with liveness analysis ─────────────────
    let buf_plan = plan_buffers(model, topo_order);
    for slot in 0..buf_plan.num_slots {
        writeln!(
            c,
            "static float nnc_buf_{slot}[{max_activation}] __attribute__((aligned({align})));"
        )
        .unwrap();
    }
    writeln!(c).unwrap();

    // ── Size helpers ─────────────────────────────────────────────
    writeln!(c, "int {name}_input_size(void) {{ return {input_elems}; }}").unwrap();
    writeln!(
        c,
        "int {name}_output_size(void) {{ return {output_elems}; }}"
    )
    .unwrap();
    writeln!(c).unwrap();

    // ── Inference function ───────────────────────────────────────
    writeln!(c, "int {name}_infer(const void *input, void *output) {{").unwrap();
    writeln!(c, "    const float *in_ptr = (const float *)input;").unwrap();
    writeln!(c, "    float *out_ptr = (float *)output;").unwrap();
    writeln!(c).unwrap();

    for layer_id in topo_order {
        let layer = model.layers.iter().find(|l| l.id == *layer_id).unwrap();
        let out_shape = shape_info.shapes.get(layer_id).unwrap();
        let out_elems = shape_elems(out_shape) * batch;
        let slot = buf_plan.slot[layer_id];
        let dst = format!("nnc_buf_{slot}");

        writeln!(
            c,
            "    /* layer: {} ({}) */",
            layer_id,
            layer.kind.type_name()
        )
        .unwrap();

        match &layer.kind {
            LayerKind::Input { shape } => {
                let n = shape_elems(shape) * batch;
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
                        emit_standardize_loop(
                            &mut c,
                            n,
                            &model.config.preprocess_mean,
                            &model.config.preprocess_std,
                            &dst,
                        );
                    }
                }
            }

            LayerKind::Dense { units, activation } => {
                let src = src_buf(&buf_plan, layer_id, model);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let in_features = shape_elems(&in_shape);

                let w_var = weight_var(&format!("{layer_id}.weight"));
                let b_var = weight_var(&format!("{layer_id}.bias"));

                writeln!(c, "    for (int i = 0; i < {units}; i++) {{").unwrap();
                writeln!(c, "        float sum = {b_var}[i];").unwrap();
                writeln!(c, "        for (int j = 0; j < {in_features}; j++) {{").unwrap();
                writeln!(c, "            sum += {w_var}[j * {units} + i] * {src}[j];").unwrap();
                writeln!(c, "        }}").unwrap();

                match activation {
                    Activation::None => {
                        writeln!(c, "        {dst}[i] = sum;").unwrap();
                    }
                    Activation::ReLU => {
                        writeln!(c, "        {dst}[i] = sum > 0.0f ? sum : 0.0f;").unwrap();
                    }
                    Activation::Sigmoid => {
                        writeln!(c, "        {dst}[i] = 1.0f / (1.0f + expf(-sum));").unwrap();
                    }
                    Activation::Softmax => {
                        writeln!(c, "        {dst}[i] = sum;").unwrap();
                    }
                }
                writeln!(c, "    }}").unwrap();

                if matches!(activation, Activation::Softmax) {
                    emit_softmax_block(&mut c, *units, &dst);
                }
            }

            LayerKind::Flatten | LayerKind::Dropout { .. } => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    memcpy({dst}, {src}, {out_elems} * sizeof(float));").unwrap();
            }

            LayerKind::ReLU => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(c, "        {dst}[i] = {src}[i] > 0.0f ? {src}[i] : 0.0f;").unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::ReLU6 => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(c, "        {dst}[i] = fminf(fmaxf(0.0f, {src}[i]), 6.0f);").unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::LeakyReLU { alpha } => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(
                    c,
                    "        {dst}[i] = {src}[i] > 0.0f ? {src}[i] : {alpha}f * {src}[i];"
                )
                .unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::SiLU => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(c, "        {dst}[i] = {src}[i] / (1.0f + expf(-{src}[i]));").unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::Sigmoid => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                writeln!(c, "        {dst}[i] = 1.0f / (1.0f + expf(-{src}[i]));").unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::Softmax { .. } => {
                let src = src_buf(&buf_plan, layer_id, model);
                writeln!(c, "    memcpy({dst}, {src}, {out_elems} * sizeof(float));").unwrap();
                emit_softmax_block(&mut c, out_elems, &dst);
            }

            LayerKind::BatchNorm { epsilon } => {
                let src = src_buf(&buf_plan, layer_id, model);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let channels = *in_shape.last().unwrap_or(&out_elems);
                let gamma_var = weight_var(&format!("{layer_id}.gamma"));
                let beta_var = weight_var(&format!("{layer_id}.beta"));
                let mean_var = weight_var(&format!("{layer_id}.running_mean"));
                let var_var = weight_var(&format!("{layer_id}.running_var"));

                writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                if in_shape.len() > 1 {
                    writeln!(c, "        int ch = i % {channels};").unwrap();
                    writeln!(c, "        {dst}[i] = {gamma_var}[ch] * ({src}[i] - {mean_var}[ch]) / sqrtf({var_var}[ch] + {epsilon:.10}f) + {beta_var}[ch];").unwrap();
                } else {
                    writeln!(c, "        {dst}[i] = {gamma_var}[i] * ({src}[i] - {mean_var}[i]) / sqrtf({var_var}[i] + {epsilon:.10}f) + {beta_var}[i];").unwrap();
                }
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::GlobalAvgPool2D => {
                let src = src_buf(&buf_plan, layer_id, model);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let (h, w, channels) = (in_shape[0], in_shape[1], in_shape[2]);
                let spatial = h * w;
                writeln!(c, "    for (int ch = 0; ch < {channels}; ch++) {{").unwrap();
                writeln!(c, "        float s = 0.0f;").unwrap();
                writeln!(c, "        for (int hw = 0; hw < {spatial}; hw++) {{").unwrap();
                writeln!(c, "            s += {src}[hw * {channels} + ch];").unwrap();
                writeln!(c, "        }}").unwrap();
                writeln!(c, "        {dst}[ch] = s / {spatial}.0f;").unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::Add => {
                let input_ids = get_input_layer_ids(layer_id, model);
                if let Some(first_id) = input_ids.first() {
                    let first_src = format!("nnc_buf_{}", buf_plan.slot[*first_id]);
                    writeln!(
                        c,
                        "    memcpy({dst}, {first_src}, {out_elems} * sizeof(float));"
                    )
                    .unwrap();
                    for add_id in input_ids.iter().skip(1) {
                        let add_src = format!("nnc_buf_{}", buf_plan.slot[*add_id]);
                        writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                        writeln!(c, "        {dst}[i] += {add_src}[i];").unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                }
            }

            LayerKind::Mul => {
                let input_ids = get_input_layer_ids(layer_id, model);
                if let Some(first_id) = input_ids.first() {
                    let first_src = format!("nnc_buf_{}", buf_plan.slot[*first_id]);
                    writeln!(
                        c,
                        "    memcpy({dst}, {first_src}, {out_elems} * sizeof(float));"
                    )
                    .unwrap();
                    for mul_id in input_ids.iter().skip(1) {
                        let mul_src = format!("nnc_buf_{}", buf_plan.slot[*mul_id]);
                        writeln!(c, "    for (int i = 0; i < {out_elems}; i++) {{").unwrap();
                        writeln!(c, "        {dst}[i] *= {mul_src}[i];").unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                }
            }

            LayerKind::Concat { axis } => {
                let input_ids = get_input_layer_ids(layer_id, model);
                let ndim = out_shape.len();
                let axis_norm = if *axis >= 0 {
                    *axis as usize
                } else {
                    (ndim as i64 + *axis) as usize
                };

                // Compute strides for the output shape
                let mut out_strides = vec![1usize; ndim];
                for d in (0..ndim - 1).rev() {
                    out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
                }

                // Number of iterations over outer dims (before axis)
                let outer_count: usize = out_shape[..axis_norm].iter().product::<usize>().max(1);
                // Number of iterations over inner dims (after axis)
                let inner_count: usize =
                    out_shape[axis_norm + 1..].iter().product::<usize>().max(1);
                let out_axis_stride = out_strides[axis_norm]; // == inner_count

                let mut axis_offset = 0usize;
                for cat_id in &input_ids {
                    let cat_shape = shape_info.shapes.get(*cat_id).unwrap();
                    let cat_axis_dim = cat_shape[axis_norm];
                    let cat_src = format!("nnc_buf_{}", buf_plan.slot[*cat_id]);

                    // Copy slice: for each outer position, memcpy the
                    // contiguous [cat_axis_dim * inner_count] block.
                    let copy_size = cat_axis_dim * inner_count;
                    let src_row_stride = copy_size; // contiguous in source
                    let dst_row_stride = out_shape[axis_norm] * inner_count;

                    if ndim == 1 || (axis_norm == ndim - 1 && outer_count == 1) {
                        // Simple flat case — single memcpy
                        writeln!(
                            c,
                            "    memcpy({dst} + {axis_offset}, {cat_src}, {copy_size} * sizeof(float));"
                        )
                        .unwrap();
                    } else {
                        writeln!(c, "    for (int oc = 0; oc < {outer_count}; oc++) {{").unwrap();
                        writeln!(
                            c,
                            "        memcpy({dst} + oc * {dst_row_stride} + {axis_offset} * {out_axis_stride}, {cat_src} + oc * {src_row_stride}, {copy_size} * sizeof(float));"
                        )
                        .unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                    axis_offset += cat_axis_dim;
                }
            }

            LayerKind::Conv2D {
                filters,
                kernel,
                stride,
                padding,
                groups,
            } => {
                let src = src_buf(&buf_plan, layer_id, model);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let (ih, iw, ic) = (in_shape[0], in_shape[1], in_shape[2]);
                let kh = kernel.height();
                let kw = kernel.width();
                let (oh, ow) = (out_shape[0], out_shape[1]);
                let w_var = weight_var(&format!("{layer_id}.weight"));
                let b_var = weight_var(&format!("{layer_id}.bias"));
                let ci_per_group = ic / groups;
                let f_per_group = filters / groups;

                match padding {
                    Padding::Valid => {
                        writeln!(c, "    for (int oh = 0; oh < {oh}; oh++) {{").unwrap();
                        writeln!(c, "      for (int ow_ = 0; ow_ < {ow}; ow_++) {{").unwrap();
                        writeln!(c, "        for (int f = 0; f < {filters}; f++) {{").unwrap();
                        writeln!(c, "          float sum = {b_var}[f];").unwrap();
                        if *groups == 1 {
                            writeln!(c, "          for (int kh_ = 0; kh_ < {kh}; kh_++) {{")
                                .unwrap();
                            writeln!(c, "            for (int kw_ = 0; kw_ < {kw}; kw_++) {{")
                                .unwrap();
                            writeln!(c, "              for (int ci = 0; ci < {ic}; ci++) {{")
                                .unwrap();
                            writeln!(c, "                int ih_ = oh * {stride} + kh_;").unwrap();
                            writeln!(c, "                int iw_ = ow_ * {stride} + kw_;").unwrap();
                            writeln!(c, "                sum += {src}[(ih_ * {iw} + iw_) * {ic} + ci] * {w_var}[((f * {ic} + ci) * {kh} + kh_) * {kw} + kw_];").unwrap();
                            writeln!(c, "              }}").unwrap();
                            writeln!(c, "            }}").unwrap();
                            writeln!(c, "          }}").unwrap();
                        } else {
                            writeln!(c, "          int g = f / {f_per_group};").unwrap();
                            writeln!(c, "          int ci_start = g * {ci_per_group};").unwrap();
                            writeln!(c, "          for (int kh_ = 0; kh_ < {kh}; kh_++) {{")
                                .unwrap();
                            writeln!(c, "            for (int kw_ = 0; kw_ < {kw}; kw_++) {{")
                                .unwrap();
                            writeln!(
                                c,
                                "              for (int ci = 0; ci < {ci_per_group}; ci++) {{"
                            )
                            .unwrap();
                            writeln!(c, "                int ih_ = oh * {stride} + kh_;").unwrap();
                            writeln!(c, "                int iw_ = ow_ * {stride} + kw_;").unwrap();
                            writeln!(c, "                sum += {src}[(ih_ * {iw} + iw_) * {ic} + ci_start + ci] * {w_var}[((f * {ci_per_group} + ci) * {kh} + kh_) * {kw} + kw_];").unwrap();
                            writeln!(c, "              }}").unwrap();
                            writeln!(c, "            }}").unwrap();
                            writeln!(c, "          }}").unwrap();
                        }
                        writeln!(
                            c,
                            "          {dst}[(oh * {ow} + ow_) * {filters} + f] = sum;"
                        )
                        .unwrap();
                        writeln!(c, "        }}").unwrap();
                        writeln!(c, "      }}").unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                    Padding::Same => {
                        let pad_h = (kh - 1) / 2;
                        let pad_w = (kw - 1) / 2;
                        writeln!(c, "    for (int oh = 0; oh < {oh}; oh++) {{").unwrap();
                        writeln!(c, "      for (int ow_ = 0; ow_ < {ow}; ow_++) {{").unwrap();
                        writeln!(c, "        for (int f = 0; f < {filters}; f++) {{").unwrap();
                        writeln!(c, "          float sum = {b_var}[f];").unwrap();
                        if *groups == 1 {
                            writeln!(c, "          for (int kh_ = 0; kh_ < {kh}; kh_++) {{")
                                .unwrap();
                            writeln!(c, "            for (int kw_ = 0; kw_ < {kw}; kw_++) {{")
                                .unwrap();
                            writeln!(c, "              int ih_ = oh * {stride} + kh_ - {pad_h};")
                                .unwrap();
                            writeln!(c, "              int iw_ = ow_ * {stride} + kw_ - {pad_w};")
                                .unwrap();
                            writeln!(
                                c,
                                "              if (ih_ >= 0 && ih_ < {ih} && iw_ >= 0 && iw_ < {iw}) {{"
                            )
                            .unwrap();
                            writeln!(c, "                for (int ci = 0; ci < {ic}; ci++) {{")
                                .unwrap();
                            writeln!(c, "                  sum += {src}[(ih_ * {iw} + iw_) * {ic} + ci] * {w_var}[((f * {ic} + ci) * {kh} + kh_) * {kw} + kw_];").unwrap();
                            writeln!(c, "                }}").unwrap();
                            writeln!(c, "              }}").unwrap();
                            writeln!(c, "            }}").unwrap();
                            writeln!(c, "          }}").unwrap();
                        } else {
                            writeln!(c, "          int g = f / {f_per_group};").unwrap();
                            writeln!(c, "          int ci_start = g * {ci_per_group};").unwrap();
                            writeln!(c, "          for (int kh_ = 0; kh_ < {kh}; kh_++) {{")
                                .unwrap();
                            writeln!(c, "            for (int kw_ = 0; kw_ < {kw}; kw_++) {{")
                                .unwrap();
                            writeln!(c, "              int ih_ = oh * {stride} + kh_ - {pad_h};")
                                .unwrap();
                            writeln!(c, "              int iw_ = ow_ * {stride} + kw_ - {pad_w};")
                                .unwrap();
                            writeln!(
                                c,
                                "              if (ih_ >= 0 && ih_ < {ih} && iw_ >= 0 && iw_ < {iw}) {{"
                            )
                            .unwrap();
                            writeln!(
                                c,
                                "                for (int ci = 0; ci < {ci_per_group}; ci++) {{"
                            )
                            .unwrap();
                            writeln!(c, "                  sum += {src}[(ih_ * {iw} + iw_) * {ic} + ci_start + ci] * {w_var}[((f * {ci_per_group} + ci) * {kh} + kh_) * {kw} + kw_];").unwrap();
                            writeln!(c, "                }}").unwrap();
                            writeln!(c, "              }}").unwrap();
                            writeln!(c, "            }}").unwrap();
                            writeln!(c, "          }}").unwrap();
                        }
                        writeln!(
                            c,
                            "          {dst}[(oh * {ow} + ow_) * {filters} + f] = sum;"
                        )
                        .unwrap();
                        writeln!(c, "        }}").unwrap();
                        writeln!(c, "      }}").unwrap();
                        writeln!(c, "    }}").unwrap();
                    }
                }
            }

            LayerKind::MaxPool2D { kernel, stride } => {
                let src = src_buf(&buf_plan, layer_id, model);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let (iw_dim, channels) = (in_shape[1], in_shape[2]);
                let kh = kernel.height();
                let kw = kernel.width();
                let s = stride.unwrap_or(kh);
                let (oh, ow) = (out_shape[0], out_shape[1]);

                writeln!(c, "    for (int oh = 0; oh < {oh}; oh++) {{").unwrap();
                writeln!(c, "      for (int ow_ = 0; ow_ < {ow}; ow_++) {{").unwrap();
                writeln!(c, "        for (int ch = 0; ch < {channels}; ch++) {{").unwrap();
                writeln!(c, "          float mv = -1e38f;").unwrap();
                writeln!(c, "          for (int kh_ = 0; kh_ < {kh}; kh_++) {{").unwrap();
                writeln!(c, "            for (int kw_ = 0; kw_ < {kw}; kw_++) {{").unwrap();
                writeln!(c, "              int ih_ = oh * {s} + kh_;").unwrap();
                writeln!(c, "              int iw_ = ow_ * {s} + kw_;").unwrap();
                writeln!(
                    c,
                    "              float v = {src}[(ih_ * {iw_dim} + iw_) * {channels} + ch];"
                )
                .unwrap();
                writeln!(c, "              if (v > mv) mv = v;").unwrap();
                writeln!(c, "            }}").unwrap();
                writeln!(c, "          }}").unwrap();
                writeln!(
                    c,
                    "          {dst}[(oh * {ow} + ow_) * {channels} + ch] = mv;"
                )
                .unwrap();
                writeln!(c, "        }}").unwrap();
                writeln!(c, "      }}").unwrap();
                writeln!(c, "    }}").unwrap();
            }

            LayerKind::AvgPool2D { kernel, stride } => {
                let src = src_buf(&buf_plan, layer_id, model);
                let in_shape = input_shape_for(layer_id, model, shape_info);
                let (iw_dim, channels) = (in_shape[1], in_shape[2]);
                let kh = kernel.height();
                let kw = kernel.width();
                let s = stride.unwrap_or(kh);
                let (oh, ow) = (out_shape[0], out_shape[1]);
                let pool_size = kh * kw;

                writeln!(c, "    for (int oh = 0; oh < {oh}; oh++) {{").unwrap();
                writeln!(c, "      for (int ow_ = 0; ow_ < {ow}; ow_++) {{").unwrap();
                writeln!(c, "        for (int ch = 0; ch < {channels}; ch++) {{").unwrap();
                writeln!(c, "          float s = 0.0f;").unwrap();
                writeln!(c, "          for (int kh_ = 0; kh_ < {kh}; kh_++) {{").unwrap();
                writeln!(c, "            for (int kw_ = 0; kw_ < {kw}; kw_++) {{").unwrap();
                writeln!(c, "              int ih_ = oh * {s} + kh_;").unwrap();
                writeln!(c, "              int iw_ = ow_ * {s} + kw_;").unwrap();
                writeln!(
                    c,
                    "              s += {src}[(ih_ * {iw_dim} + iw_) * {channels} + ch];"
                )
                .unwrap();
                writeln!(c, "            }}").unwrap();
                writeln!(c, "          }}").unwrap();
                writeln!(
                    c,
                    "          {dst}[(oh * {ow} + ow_) * {channels} + ch] = s / {pool_size}.0f;"
                )
                .unwrap();
                writeln!(c, "        }}").unwrap();
                writeln!(c, "      }}").unwrap();
                writeln!(c, "    }}").unwrap();
            }
        }

        writeln!(c).unwrap();
    }

    // Copy final result to output
    let final_slot = buf_plan.slot[output_layer_id];
    let final_src = format!("nnc_buf_{final_slot}");
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

// ── Buffer planning ──────────────────────────────────────────────

use crate::ir::model::Padding;

struct BufferPlan {
    /// Buffer slot index for each layer.
    slot: std::collections::HashMap<String, usize>,
    /// Total number of buffer slots needed.
    num_slots: usize,
}

fn plan_buffers(model: &Model, topo_order: &[String]) -> BufferPlan {
    // Compute last_use: for each layer, the topo index of its last consumer.
    let mut last_use: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (idx, layer_id) in topo_order.iter().enumerate() {
        let input_ids = get_input_layer_ids(layer_id, model);
        for src_id in input_ids {
            let entry = last_use.entry(src_id).or_insert(idx);
            if idx > *entry {
                *entry = idx;
            }
        }
    }
    // Layers with no consumers (output layer) last_use = their own index
    for (idx, layer_id) in topo_order.iter().enumerate() {
        last_use.entry(layer_id.as_str()).or_insert(idx);
    }

    let mut slot_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut slot_free_after: Vec<usize> = Vec::new(); // slot -> topo index when it becomes free
    let mut num_slots: usize = 0;

    for (idx, layer_id) in topo_order.iter().enumerate() {
        // Find a free slot: one whose free_after < idx
        let reuse = slot_free_after.iter().position(|&free_at| free_at < idx);
        let slot = match reuse {
            Some(s) => {
                slot_free_after[s] = *last_use.get(layer_id.as_str()).unwrap_or(&idx);
                s
            }
            None => {
                let s = num_slots;
                num_slots += 1;
                slot_free_after.push(*last_use.get(layer_id.as_str()).unwrap_or(&idx));
                s
            }
        };
        slot_map.insert(layer_id.clone(), slot);
    }

    BufferPlan {
        slot: slot_map,
        num_slots: num_slots.max(1),
    }
}

fn src_buf(plan: &BufferPlan, layer_id: &str, model: &Model) -> String {
    let input_ids = get_input_layer_ids(layer_id, model);
    if let Some(first) = input_ids.first() {
        let slot = plan.slot[*first];
        format!("nnc_buf_{slot}")
    } else {
        "nnc_buf_0".to_string()
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn get_input_layer_ids<'a>(layer_id: &str, model: &'a Model) -> Vec<&'a str> {
    model
        .edges
        .iter()
        .filter(|e| e.target == layer_id)
        .map(|e| e.source.as_str())
        .collect()
}

fn input_shape_for(layer_id: &str, model: &Model, shape_info: &ShapeInfo) -> Vec<usize> {
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
            writeln!(c, "        {dst}[i] = (in_ptr[i] - mu[ch]) / sd[ch];").unwrap();
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
