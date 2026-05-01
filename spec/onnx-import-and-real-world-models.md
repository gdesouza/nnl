# ONNX Import Fix & Gaps for Real-World Pre-Trained Models

**Date:** 2026-04-22
**Status:** Draft
**Context:** Findings from the [nnl-experiments](https://github.com/gdesouza/nnl-experiments) benchmarking repo, where we attempted to import and run a CIFAR-10 CNN trained in PyTorch via `nnc import`.

---

## 1. Bug: ONNX Import Protobuf Decode Failure

### Symptom

`nnc import` fails on **every** valid ONNX file with a protobuf decoding error:

```
nnc: import failed: failed to decode ONNX protobuf: failed to decode Protobuf message:
  AttributeProto.f: NodeProto.attribute: GraphProto.node: ModelProto.graph:
  invalid wire type: LengthDelimited (expected ThirtyTwoBit)
```

Tested with ONNX files from both PyTorch's legacy TorchScript exporter (opset 11, IR v6) and the new `torch.export`-based exporter (opset 18, IR v10). Both fail.

### Root Cause

The hand-rolled protobuf definition in `src/import/onnx.rs` has **incorrect field tag numbers** for `AttributeProto`. The tags don't match the [official ONNX protobuf schema](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto).

**`AttributeProto` tag comparison:**

| Field  | Official ONNX tag | nnc tag | Status |
|--------|-------------------|---------|--------|
| `name` | 1                 | 1       | ✅ correct |
| `type` | 20                | 2       | ❌ **wrong** — tag 2 is `f` in the spec |
| `f`    | 2                 | 4       | ❌ **wrong** — tag 4 is `s` in the spec |
| `i`    | 3                 | 3       | ✅ correct |
| `s`    | 4                 | 5       | ❌ **wrong** — tag 5 is `t` in the spec |
| `ints` | 8                 | 7       | ❌ **wrong** — tag 7 is `floats` in the spec |

When `prost` encounters tag 2 in the wire data (which is the `f` field, a 32-bit float), it tries to decode it as `type` (a varint `int32`). The wire type mismatch (`ThirtyTwoBit` vs `Varint`) causes the decode to fail immediately.

Additionally, several fields present in real ONNX files are **missing** from the definition:

| Field | Tag | Type | Needed for |
|-------|-----|------|------------|
| `t` | 5 | `TensorProto` | Constant tensor attributes (used by Reshape, Unsqueeze, etc.) |
| `g` | 6 | `GraphProto` | Subgraph attributes (used by If, Loop) |
| `floats` | 7 | repeated float | Multi-value float attributes (e.g., dilations as floats) |
| `strings` | 9 | repeated bytes | Multi-value string attributes |

While `prost` normally skips unknown fields gracefully, the tag *collisions* (nnc's `ints` at tag 7 collides with the spec's `floats` at tag 7) cause type misinterpretation and decode failures.

### Fix

Replace the `AttributeProto` definition in `src/import/onnx.rs` with correct tags:

```rust
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes, tag = "4")]
    pub s: Vec<u8>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
}
```

All other proto definitions (`ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `ValueInfoProto`, `TypeProto`, `TensorShapeProto`, `Dimension`, `OperatorSetIdProto`) were verified against the official spec and have **correct tag numbers**.

### Verification

After fixing, `nnc import` should successfully parse any ONNX file containing the supported ops (Conv, Relu, MaxPool, Gemm, Flatten, BatchNormalization, Softmax, etc.). Test with:

```bash
# Export a simple model from PyTorch
python -c "
import torch, torch.nn as nn
model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
torch.onnx.export(model, torch.randn(1, 4), 'test.onnx', opset_version=11, dynamo=False)
"

# Import into NNL
nnc import test.onnx -o test.nnl --weights-dir ./test_weights
nnc inspect test.nnl
nnc compile test.nnl --emit exe -o test_model
```

---

## 2. Data Layout Mismatch: CHW vs HWC at the Flatten Boundary

### Problem

Every major ML framework (PyTorch, TensorFlow, ONNX) processes images in **NCHW** order. nnc computes internally in **HWC** order. While Conv2D weight tensors have the same `[F, C_in, kH, kW]` layout in both systems, the **Flatten layer** produces different orderings:

- **PyTorch:** flattens `[C, H, W]` → channel-major (all spatial positions for channel 0, then channel 1, ...)
- **nnc:** flattens `[H, W, C]` → spatial-major (all channels for position (0,0), then (0,1), ...)

This means the Dense layer immediately after Flatten receives inputs in a different order. Weights trained in PyTorch's CHW flatten order produce **wrong results** when used with nnc's HWC flatten order.

### Impact

This blocks direct use of **any** pre-trained CNN that has a Flatten→Dense transition — which includes most classification models. The workaround of permuting the fc weight matrix row ordering is fragile and not handled by `nnc import`.

### Recommended Solutions (pick one)

**Option A — Handle in `nnc import`:** When importing an ONNX model, detect the Flatten→Gemm pattern and automatically permute the Dense weight matrix rows from CHW to HWC order. The import code already has access to the tensor shapes from the ONNX graph, so it can compute the permutation indices.

**Option B — Add `GlobalAvgPool2D`:** Most modern CNNs (ResNet, MobileNet, EfficientNet) use Global Average Pooling instead of Flatten+Dense. `GlobalAvgPool2D` reduces `[H, W, C] → [C]`, which produces the same result regardless of spatial layout. This sidesteps the problem entirely for modern architectures.

**Option C — Support NCHW layout:** Add a `data_format: "nchw"` config option. This is the most general solution but the largest implementation effort.

Option B is recommended as the highest-impact, lowest-effort path — it unlocks ResNet and MobileNet families while avoiding the layout problem.

---

## 3. Missing Layers for Real-World Pre-Trained Models

The following table maps commonly deployed embedded models to the nnc features they require. Layers already supported are marked ✅.

### Feature Matrix

| Feature | ResNet-18 | MobileNetV1 | MobileNetV2 | EfficientNet-B0 | YOLO-Tiny |
|---------|-----------|-------------|-------------|-----------------|-----------|
| Conv2D | ✅ | ✅ | ✅ | ✅ | ✅ |
| MaxPool2D | ✅ | | | | ✅ |
| BatchNorm | ✅ | ✅ | ✅ | ✅ | ✅ |
| ReLU | ✅ | | | | |
| Add (skip conn.) | ✅ | | ✅ | ✅ | |
| Dense | ✅ | ✅ | ✅ | ✅ | ✅ |
| Softmax | ✅ | ✅ | ✅ | ✅ | |
| Flatten | ✅ | | | | |
| **GlobalAvgPool2D** | ❌ needed | ❌ needed | ❌ needed | ❌ needed | |
| **ReLU6** | | ❌ needed | ❌ needed | | |
| **Depthwise Conv2D** | | ❌ needed | ❌ needed | ❌ needed | |
| **SiLU / Swish** | | | | ❌ needed | |
| **Hardswish** | | | ❌ (V3) | | |
| **Squeeze-Excite (Mul)** | | | | ❌ needed | |
| **LeakyReLU** | | | | | ❌ needed |
| **Upsample / Resize** | | | | | ❌ needed |
| **Concat** | ✅ | | | | ✅ |

### Prioritized Roadmap

#### Tier 1 — Unlock ResNet-18 (1 new layer)

| Feature | Effort | Impact |
|---------|--------|--------|
| `GlobalAvgPool2D` | Small — average all spatial dims, output `[C]` | Unlocks ResNet-18/34/50, avoids CHW/HWC Flatten issue |

ResNet-18 is the most commonly deployed pre-trained model for embedded vision. With `GlobalAvgPool2D`, every layer it uses is already supported: Conv2D, BatchNorm, ReLU, Add, Dense, Softmax. It would be the first real-world pre-trained model directly importable via ONNX.

#### Tier 2 — Unlock MobileNetV1 (2 new features)

| Feature | Effort | Impact |
|---------|--------|--------|
| Depthwise Conv2D (`groups` param) | Medium — separate codegen path for `groups == in_channels` | Core building block of all efficient mobile architectures |
| `ReLU6` activation | Trivial — `min(max(0, x), 6)` | Required by MobileNetV1/V2, trivial to implement |

MobileNetV1 is the poster child for embedded inference — designed explicitly for mobile/edge devices with ~4.2M parameters and ~569M MACs.

#### Tier 3 — Unlock MobileNetV2 and EfficientNet (2 new features)

| Feature | Effort | Impact |
|---------|--------|--------|
| `SiLU` / `Swish` activation | Trivial — `x * sigmoid(x)` | Required by EfficientNet |
| Element-wise `Mul` layer | Small — pointwise multiply for squeeze-excite blocks | Required by EfficientNet |

#### Tier 4 — Broader Coverage

| Feature | Effort | Impact |
|---------|--------|--------|
| `LeakyReLU` activation | Trivial — `max(alpha * x, x)` | YOLO, many detection models |
| `Upsample` / `Resize` | Medium | U-Net, encoder-decoder models, detection models |
| `Conv1D` / `MaxPool1D` | Medium | Audio, time-series, keyword spotting |
| `LayerNorm` | Medium | Transformer-adjacent models |

---

## 4. ONNX Import: Layout Conversion for Weights

### Current Behavior

`nnc import` extracts Conv2D weights as-is from the ONNX initializers. The weight shape `[F, C_in, kH, kW]` is the same in both PyTorch and nnc, so Conv2D weights work correctly.

However, Dense (Gemm) weights after a Flatten layer are extracted without adjusting for the CHW→HWC reordering that nnc's Flatten produces. This means any imported CNN with Flatten→Dense will produce wrong inference results.

### Recommended Fix

In `src/import/mod.rs`, when processing a `Gemm` node that follows a `Flatten` node:

1. Detect the upstream Conv2D/Pool output shape from the ONNX graph's `value_info` or shape inference
2. Compute the CHW→HWC index permutation for the flattened dimension
3. Permute the weight matrix rows accordingly before writing the `.npy` file

This should be done as part of the `nnc import` pipeline so users don't need to manually adjust weights.

---

## 5. Summary: Minimum Path to Real-World Pre-Trained Models

```
MUST FIX (blocks all ONNX import):
  [1] Fix AttributeProto field tags in src/import/onnx.rs

SHOULD ADD (enables ResNet-18 — first real-world model):
  [2] GlobalAvgPool2D layer
  [3] CHW→HWC weight permutation in nnc import for Flatten→Dense

NEXT (enables MobileNetV1 — the embedded vision workhorse):
  [4] Depthwise Conv2D (groups parameter)
  [5] ReLU6 activation
```

With items [1]–[3], a user could do:

```bash
# Download pre-trained ResNet-18 from torchvision, export to ONNX
python export_resnet18.py

# Import into NNL — weights automatically converted
nnc import resnet18.onnx -o resnet18.nnl --weights-dir ./weights

# Compile to a zero-dependency 45 MB binary (weights embedded)
nnc compile resnet18.nnl --emit exe -o resnet18

# Run inference on an image
cat image.bin | ./resnet18 > predictions.bin
```

This would be a compelling demo: a production-grade ImageNet classifier in a single binary with zero runtime dependencies.
