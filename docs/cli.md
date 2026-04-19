# CLI Reference

`nnc` is the NNL compiler. It compiles `.nnl` neural network definitions into standalone, zero-dependency native artifacts.

---

## nnc compile

Compile an NNL model to a native artifact.

```
nnc compile <source.nnl> [--emit exe|obj|lib|shared|header] [-o <output>] [--target-triple <triple>]
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--emit <format>` | Output format: `exe`, `obj`, `lib`, `shared`, `header` | `exe` |
| `-o, --output <path>` | Output file path | Source stem with appropriate extension |
| `--target-triple <triple>` | Target triple for cross-compilation | Host platform |

### Emit formats

| Format | Output | Extension | Notes |
|--------|--------|-----------|-------|
| `exe` | Standalone executable with `main()` (reads stdin, writes stdout) | `<stem>` | Default |
| `obj` | Relocatable object file | `<stem>.o` | Also generates `<stem>.h` |
| `lib` | Static archive | `lib<stem>.a` | Also generates `<stem>.h` |
| `shared` | Shared library | `lib<stem>.so` | Also generates `<stem>.h` |
| `header` | C header only | `<stem>.h` | No compilation step |

For `obj`, `lib`, and `shared`, a `.h` header declaring the public C API is generated alongside the output.

### Examples

```sh
# Compile to a standalone executable (default)
nnc compile mnist.nnl

# Compile to a static library + header
nnc compile mnist.nnl --emit lib -o build/libmnist.a

# Compile to a shared library
nnc compile mnist.nnl --emit shared

# Compile to an object file
nnc compile mnist.nnl --emit obj -o mnist.o

# Generate only the C header
nnc compile mnist.nnl --emit header

# Cross-compile for ARM Cortex-M
nnc compile mnist.nnl --emit obj --target-triple thumbv7em-none-eabi

# Cross-compile for bare-metal ARM
nnc compile model.nnl --emit lib --target-triple arm-none-eabi
```

---

## nnc inspect

Print a model summary: layers, types, output shapes, parameter counts, and memory estimates.

```
nnc inspect <source.nnl>
```

### Example

```sh
nnc inspect mnist.nnl
```

Example output:

```
Model: mnist_classifier (version 0.2)
Precision: float32 | Target: avx2 | Batch: 1

Layer           Type        Output Shape        Params
──────────────────────────────────────────────────────
input           Input       [28, 28, 1]              0
conv1           Conv2D      [26, 26, 32]           320
pool1           MaxPool2D   [13, 13, 32]             0
flatten         Flatten     [5408]                   0
fc1             Dense       [128]                691,328
output          Dense       [10]                   1,290
──────────────────────────────────────────────────────
Total params:    692,938
Weight memory:   2.64 MB
Workspace:       86.5 KB (static buffer)
```

---

## nnc import

Convert an ONNX model into NNL format with extracted weight files.

```
nnc import <model.onnx> [-o <output.nnl>] [--weights-dir <dir>]
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-o, --output <path>` | Output `.nnl` file path | Source name with `.nnl` extension |
| `--weights-dir <dir>` | Directory to write extracted `.npy` weight files | `./weights` |

### Notes

- Each ONNX initializer is extracted as a separate `.npy` file in the weights directory.
- Unsupported ONNX operators are emitted as comments in the generated `.nnl` file.

### Examples

```sh
# Import with defaults (resnet.nnl + ./weights/)
nnc import resnet.onnx

# Specify output path and weights directory
nnc import resnet.onnx -o models/resnet.nnl --weights-dir models/weights
```

---

## nnc test

Compile a model, run inference on a given input, and compare the output element-wise against expected values.

```
nnc test <source.nnl> --input <input.npy> --expected <expected.npy> [--tolerance <tol>]
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--input <path>` | Path to input tensor (`.npy`, float32) | Required |
| `--expected <path>` | Path to expected output tensor (`.npy`, float32) | Required |
| `--tolerance <tol>` | Maximum allowed absolute difference per element | `1e-5` |

### Behavior

1. Compiles the model to a temporary executable.
2. Feeds the input tensor via stdin as raw float32 bytes.
3. Reads the output tensor from stdout.
4. Compares each element against the expected tensor.
5. Reports up to 10 individual mismatches, then a summary.

### Examples

```sh
# Test with default tolerance (1e-5)
nnc test mnist.nnl --input test_input.npy --expected test_output.npy

# Test with relaxed tolerance
nnc test mnist.nnl --input test_input.npy --expected test_output.npy --tolerance 1e-3
```

Example pass output:

```
PASS: 10/10 elements within tolerance 1.0e-5 (max diff: 3.42e-7)
```

Example fail output:

```
  mismatch at [3]: got 0.72341299, expected 0.72345012, diff 3.71e-5
  mismatch at [7]: got 0.10002345, expected 0.10010000, diff 7.66e-5
FAIL: 2/10 elements exceed tolerance 1.0e-5 (max diff: 7.66e-5)
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success (compilation succeeded, test passed, import/inspect completed) |
| `1` | Error (syntax error, validation failure, compilation error, test mismatch, I/O error) |

---

## Environment

| Requirement | Purpose |
|-------------|---------|
| Rust toolchain | Building `nnc` from source |
| C compiler (`cc`, `gcc`, or `clang`) on `PATH` | Used by `nnc compile` to produce native artifacts |
| Cross-compiler (e.g., `arm-none-eabi-gcc`) | Required when using `--target-triple` for cross-compilation |
