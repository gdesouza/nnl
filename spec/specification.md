# NNL Specification v0.2

NNL (Neural Network Language) is a declarative language for defining neural network architectures. It is designed to be consumed by the `nnc` compiler that generates native machine code with embedded weights for zero-dependency inference.

## 1. Syntax Overview

### 1.1 Comments

NNL supports line comments and block comments:

```
// This is a line comment
/* This is a
   block comment */
```

### 1.2 Grammar (BNF)

```
<model>            ::= <version_decl>? "model" <identifier> "{" <config_block> <layer_list> <connection_block>? "}"

<version_decl>     ::= "version" <number> ";"

<config_block>     ::= "config" "{" <setting_list> "}"
<setting_list>     ::= <setting> | <setting> <setting_list>
<setting>          ::= <identifier> ":" <value> ";"

<layer_list>       ::= <layer> | <layer> <layer_list>
<layer>            ::= "layer" <identifier> "=" <layer_type> "(" <param_list>? ")" ";"

<layer_type>       ::= "Input" | "Dense" | "Conv2D" | "MaxPool2D" | "AvgPool2D"
                     | "Flatten" | "BatchNorm" | "Dropout"
                     | "Add" | "Concat"
                     | "ReLU" | "Sigmoid" | "Softmax"

<param_list>       ::= <param> | <param> "," <param_list>
<param>            ::= <identifier> ":" <value>

<connection_block> ::= "connections" "{" <connection_list> "}"
<connection_list>  ::= <connection> | <connection> <connection_list>
<connection>       ::= <identifier> "->" <identifier> ";"
                     | "[" <id_list> "]" "->" <identifier> ";"
<id_list>          ::= <identifier> | <identifier> "," <id_list>

<value>            ::= <string> | <number> | <boolean> | <shape>
<shape>            ::= "[" <number_list> "]"
<number_list>      ::= <number> | <number> "," <number_list>

<identifier>       ::= [a-zA-Z_][a-zA-Z0-9_]*
<number>           ::= [0-9]+("." [0-9]+)?
<string>           ::= "\"" [^\"]* "\""
<boolean>          ::= "true" | "false"
```

### 1.3 Connectivity

Layers can be connected in two modes:

- **Implicit sequential.** When no `connections` block is present, layers are connected in declaration order (each layer receives the output of the previous layer). This is the simple case for linear stacks.
- **Explicit graph.** When a `connections` block is present, it fully defines the data flow. This enables skip connections, branches, and multi-input layers.

Multi-input layers (`Add`, `Concat`) require explicit connections with the bracket syntax:

```
connections {
    input -> conv1;
    conv1 -> bn1;
    bn1 -> relu1;
    relu1 -> conv2;
    conv2 -> bn2;
    [relu1, bn2] -> residual;  // skip connection
    residual -> output;
}
```

## 2. Configuration Settings

The config block defines global compilation parameters.

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| precision | String | No | "float32" | Data type for tensors ("float32", "float64", "int8"). |
| weights | String | Yes | — | Path to the weight source (.npy, .onnx, or directory). |
| target | String | No | "generic" | Optimization target ("generic", "avx2", "avx512", "arm_neon"). |
| align | Number | No | 64 | Memory alignment in bytes for weight buffers. |
| batch | Number | No | 1 | Inference batch size. Determines static buffer dimensions. |
| preprocess | String | No | "none" | Input preprocessing ("none", "normalize_0_1", "standardize"). |
| preprocess_mean | Shape | No | — | Per-channel mean for "standardize" (e.g., [0.485, 0.456, 0.406]). |
| preprocess_std | Shape | No | — | Per-channel std for "standardize" (e.g., [0.229, 0.224, 0.225]). |
| io | String | No | "stdio" | I/O mode for `--emit exe` binaries (see §7.4). Currently only "stdio" is supported. |

## 3. Layer Definitions

### 3.1 Input

Defines the entry point of the network.

- Params: `shape` (Shape, required).
- Example: `layer input = Input(shape: [28, 28, 1]);`

### 3.2 Dense (Fully Connected)

Performs $Y = \sigma(WX + B)$.

- Params: `units` (Integer, required), `activation` (String, optional, default: "none").
- Weight mapping: `nnc` expects tensors named `{layer_id}.weight` (shape: input_dim × units) and `{layer_id}.bias` (shape: units) in the weight source.

### 3.3 Conv2D

Performs 2D spatial convolution.

- Params: `filters` (Integer, required), `kernel` (Integer or Shape, required), `stride` (Integer, optional, default: 1), `padding` (String, optional: "valid" | "same", default: "valid").
- Weight mapping: `{layer_id}.weight` (shape: filters × in_channels × kH × kW), `{layer_id}.bias` (shape: filters).

### 3.4 MaxPool2D

Spatial max pooling.

- Params: `kernel` (Integer or Shape, required), `stride` (Integer, optional, defaults to kernel size).

### 3.5 AvgPool2D

Spatial average pooling.

- Params: `kernel` (Integer or Shape, required), `stride` (Integer, optional, defaults to kernel size).

### 3.6 Flatten

Reshapes a multi-dimensional tensor into a 1D vector.

- Params: none.

### 3.7 BatchNorm

Batch normalization (inference mode: uses stored running mean/variance).

- Params: `epsilon` (Number, optional, default: 1e-5).
- Weight mapping: `{layer_id}.gamma`, `{layer_id}.beta`, `{layer_id}.running_mean`, `{layer_id}.running_var`.

### 3.8 Dropout

During inference, Dropout is a no-op (identity pass-through). It exists in the grammar so that models exported from training frameworks can be represented without manual editing.

- Params: `rate` (Number, optional, default: 0.5). Ignored during compilation.

### 3.9 Add

Element-wise addition of two or more inputs. Requires explicit connections.

- Params: none.
- Constraint: all inputs must have identical shapes.

### 3.10 Concat

Channel-wise concatenation of two or more inputs. Requires explicit connections.

- Params: `axis` (Integer, optional, default: -1).
- Constraint: all inputs must have identical shapes except along the concatenation axis.

### 3.11 Activation Layers

Standalone activation layers that can be used as separate nodes in the graph.

- **ReLU**: `max(0, x)`. No params.
- **Sigmoid**: `1 / (1 + exp(-x))`. No params.
- **Softmax**: Normalized exponential. Params: `axis` (Integer, optional, default: -1).

## 4. Weight Mapping

Weights are matched to layers by layer identifier. Given a layer declared as `layer fc1 = Dense(units: 128);`, `nnc` looks up tensors with the prefix `fc1.` in the weight source.

### 4.1 Supported Weight Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| NumPy | `.npy`, `.npz` | Single-tensor or archive. For `.npz`, keys must match `{layer_id}.{param}`. |
| ONNX | `.onnx` | Initializer names must match `{layer_id}.{param}`. |
| Directory | path/ | Directory of `.npy` files named `{layer_id}.{param}.npy`. |

### 4.2 Missing or Mismatched Weights

- If a required tensor is missing, `nnc` emits a compile error listing the expected tensor name, shape, and dtype.
- If a tensor's shape does not match the layer's computed expectation, `nnc` emits a compile error showing both the expected and actual shapes.

## 5. Semantic Analysis

After parsing, `nnc` performs a semantic analysis pass before code generation. This pass catches errors early and provides clear diagnostics.

### 5.1 Shape Inference

`nnc` propagates tensor shapes forward through the graph starting from the Input layer(s). At each layer, the output shape is computed from the input shape and layer parameters. This produces a complete shape annotation for every edge in the graph.

### 5.2 Validation Rules

| Rule | Error |
|------|-------|
| Every non-Input layer must have at least one input. | `E001: layer '{id}' has no input connection` |
| Input dimensions must be compatible with the next layer. | `E002: shape mismatch at '{id}': expected {expected}, got {actual}` |
| Weight tensors must exist and match inferred shapes. | `E003: missing weight '{layer_id}.{param}', expected shape {shape}` |
| Add inputs must have identical shapes. | `E004: shape mismatch in Add '{id}': inputs have shapes {shapes}` |
| Concat inputs must match on all axes except the concat axis. | `E005: incompatible shapes for Concat '{id}' on axis {axis}` |
| No cycles in the connection graph. | `E006: cycle detected involving layer '{id}'` |
| All declared layers must be reachable from an Input. | `W001: layer '{id}' is unreachable` |
| Config references valid precision/target combinations. | `E007: unsupported target '{target}' for precision '{precision}'` |

## 6. Compilation Strategy

### 6.1 Weight Embedding (Baking)

`nnc` performs the following during the build phase:
- Extraction: Parses the external weight file (NPY/ONNX).
- Quantization/Cast: If the source weight precision differs from the config.precision, `nnc` performs a lossless or lossy cast. Lossy casts (e.g., float32 → int8) emit a warning.
- Layout Optimization: Weights are reordered (e.g., from NCHW to NHWC) to match the target architecture's SIMD requirements.
- Binary Mapping: Weights are written into the `.rodata` section of the output binary.

### 6.2 Memory Management

The generated executable must not use dynamic heap allocation (malloc/free) during inference.
- Static Workspace: `nnc` calculates the peak activation memory requirement (informed by shape inference and batch size) and allocates a static global buffer.
- Buffer Swapping: `nnc` generates code that alternates between two regions of the static buffer to minimize total footprint.

### 6.3 Preprocessing Codegen

When `preprocess` is set in the config, `nnc` generates a preprocessing step at the beginning of the inference function:
- `normalize_0_1`: Divides each input element by 255.0.
- `standardize`: Applies `(x - mean) / std` per channel using the provided `preprocess_mean` and `preprocess_std` values.

## 7. Output Formats and Calling Convention

### 7.1 Output Formats

`nnc` supports the following output modes, selected via the `--emit` flag:

| Flag | Output | Description |
|------|--------|-------------|
| `--emit exe` | Executable | Standalone binary with a `main()`. I/O behavior is determined by the `io` config setting (see §7.4). |
| `--emit obj` | `.o` file | Relocatable object file for linking into a larger project. |
| `--emit lib` | `.a` file | Static library archive. |
| `--emit shared` | `.so` / `.dll` | Shared library. |
| `--emit header` | `.h` file | C header declaring the inference function signature. Always generated alongside `obj`, `lib`, and `shared`. |

### 7.2 C Calling Convention

The generated inference function follows a C ABI with the following signature:

```c
// Generated header: model_name.h
#include <stdint.h>

// Run inference. Returns 0 on success.
// - input:  pointer to input tensor, row-major, matching config.precision
// - output: pointer to pre-allocated output buffer
int model_name_infer(const void* input, void* output);

// Query input/output dimensions (optional helpers)
int model_name_input_size(void);   // total elements
int model_name_output_size(void);  // total elements
```

The caller is responsible for allocating the input and output buffers. All internal computation uses the statically allocated workspace — no heap allocation occurs.

### 7.3 Executable I/O Modes

The `io` config setting controls how `--emit exe` binaries receive input and produce output. The `io` setting is ignored for `obj`, `lib`, `shared`, and `header` output formats.

#### 7.3.1 `"stdio"` (default)

The generated binary reads raw tensor bytes from **stdin** and writes raw tensor bytes to **stdout**.

- **Input**: exactly `input_size * sizeof(precision)` bytes, row-major layout. For `float32`, this is `input_size * 4` bytes.
- **Output**: exactly `output_size * sizeof(precision)` bytes, row-major layout.
- **Exit code**: 0 on success, 1 on error (e.g., unexpected input size). Errors are printed to stderr.

Example usage:

```bash
cat input.bin | ./mnist_classifier > output.bin
```

Where `input.bin` contains 784 float32 values (28×28×1) as raw bytes, and `output.bin` will contain 10 float32 values.

#### 7.3.2 Future I/O Modes

The following modes are reserved for future specification and are not yet implemented:

| Mode | Description |
|------|-------------|
| `"file"` | Read/write named files passed as command-line arguments. |
| `"npy"` | Read/write NumPy `.npy` format from files or stdin/stdout. |
| `"tcp"` | Listen on a TCP socket, accept inference requests using a binary framing protocol. |

Using an unimplemented `io` mode produces a compile error.

### 7.4 Cross-Compilation

`nnc` supports cross-compilation via the `--target-triple` flag (e.g., `--target-triple thumbv7em-none-eabi` for ARM Cortex-M). The `target` config setting controls SIMD optimization; `--target-triple` controls the output architecture.

## 8. Error Model

`nnc` categorizes diagnostics into three levels:

| Level | Prefix | Behavior |
|-------|--------|----------|
| Error | `E` | Compilation aborts. Exit code 1. |
| Warning | `W` | Compilation continues. Printed to stderr. |
| Info | `I` | Informational. Printed only with `--verbose`. |

All diagnostics include the source file path, line number, column number, and a human-readable message:

```
model.nnl:12:5: E002: shape mismatch at 'fc1': expected [784], got [256]
```

## 9. CLI and Tooling

### 9.1 Compiler

```
nnc compile model.nnl [--emit exe|obj|lib|shared] [--target-triple <triple>] [-o output]
```

### 9.2 Inspect

Prints the computation graph, shapes at each layer, total parameter count, and estimated memory footprint:

```
nnc inspect model.nnl
```

Example output:
```
Model: mnist_classifier (version 0.2)
Precision: float32 | Target: avx2 | Batch: 1

Layer           Type       Output Shape     Params
──────────────────────────────────────────────────────
input           Input      [28, 28, 1]           0
conv1           Conv2D     [26, 26, 32]        320
pool1           MaxPool2D  [13, 13, 32]          0
flatten         Flatten    [5408]                0
fc1             Dense      [128]           692,352
output          Dense      [10]              1,290
──────────────────────────────────────────────────────
Total params:    693,962
Weight memory:   2.65 MB (float32)
Workspace:       21.2 KB (static buffer)
```

### 9.3 Test

Runs inference using the compiled model against known input/output pairs for correctness verification:

```
nnc test model.nnl --input test_input.npy --expected test_output.npy [--tolerance 1e-5]
```

Exit code 0 if all outputs match within tolerance; exit code 1 with a diff report otherwise.

### 9.4 Import (Future)

Converts a trained model from a standard format into an NNL source file and extracted weights:

```
nnc import model.onnx -o model.nnl --weights-dir ./weights/
```

This lowers the adoption barrier by letting users start from existing trained models rather than hand-writing `.nnl` files. Unsupported layers are emitted as `// UNSUPPORTED: LayerType(...)` comments for manual resolution.

## 10. Versioning

### 10.1 File Version

NNL source files may declare a version at the top of the model:

```
version 0.2;
model mnist_classifier {
    ...
}
```

If omitted, `nnc` assumes the latest version it supports and emits `W002: no version declared, assuming 0.2`.

### 10.2 Compatibility

`nnc` must be able to compile files targeting older NNL versions. When a breaking syntax change is introduced, the version number is incremented and `nnc` uses the declared version to select the appropriate parser rules.

## 11. Example

A complete NNL file for an MNIST classifier:

```
version 0.2;

// MNIST handwritten digit classifier
model mnist_classifier {
    config {
        precision: "float32";
        weights: "./weights/mnist.npz";
        target: "avx2";
        batch: 1;
        preprocess: "normalize_0_1";
        io: "stdio";
    }

    layer input   = Input(shape: [28, 28, 1]);
    layer conv1   = Conv2D(filters: 32, kernel: 3, stride: 1, padding: "valid");
    layer pool1   = MaxPool2D(kernel: 2);
    layer flatten  = Flatten();
    layer fc1     = Dense(units: 128, activation: "relu");
    layer output  = Dense(units: 10, activation: "softmax");
}
```

A ResNet-style residual block using explicit connections:

```
version 0.2;

model resnet_block {
    config {
        precision: "float32";
        weights: "./weights/resnet.npz";
        target: "generic";
        io: "stdio";
    }

    layer input  = Input(shape: [32, 32, 64]);
    layer conv1  = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer bn1    = BatchNorm();
    layer relu1  = ReLU();
    layer conv2  = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer bn2    = BatchNorm();
    layer res    = Add();
    layer relu2  = ReLU();

    connections {
        input -> conv1;
        conv1 -> bn1;
        bn1 -> relu1;
        relu1 -> conv2;
        conv2 -> bn2;
        [input, bn2] -> res;   // skip connection
        res -> relu2;
    }
}
```
