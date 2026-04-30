# NNLang — Neural Network Language

[![CI](https://github.com/gdesouza/nnl/actions/workflows/ci.yml/badge.svg)](https://github.com/gdesouza/nnl/actions/workflows/ci.yml)
[![Release](https://github.com/gdesouza/nnl/actions/workflows/release.yml/badge.svg)](https://github.com/gdesouza/nnl/actions/workflows/release.yml)
[![Security](https://github.com/gdesouza/nnl/actions/workflows/security.yml/badge.svg)](https://github.com/gdesouza/nnl/actions/workflows/security.yml)
[![Latest Release](https://img.shields.io/github/v/release/gdesouza/nnl)](https://github.com/gdesouza/nnl/releases/tag/v0.7.0)

NNL is a declarative language for defining neural network architectures, paired with the `nnc` compiler that produces **standalone, zero-dependency native binaries** with embedded weights.

## Goals

- **No runtime dependencies** — compiled models are self-contained static binaries.
- **No heap allocation** — all memory is statically allocated at compile time.
- **Human-readable source** — model architectures are defined in plain-text `.nnl` files, version-controllable and auditable.
- **Systems-first** — `nnc` targets bare-metal-capable output, treating inference as a systems programming problem.

## Quick Start

```bash
# Install
cargo install nnlang

# Write a model (or import from ONNX)
nnc import model.onnx -o model.nnl --weights-dir ./weights

# Inspect the architecture
nnc inspect model.nnl

# Compile to a native binary
nnc compile model.nnl --emit exe -o inference

# Run inference (stdin/stdout, raw float32 bytes)
cat input.bin | ./inference > output.bin

# Test against known output
nnc test model.nnl --input input.npy --expected output.npy
```

## Example

```
version 0.2;

model mnist_classifier {
    config {
        precision: "float32";
        weights: "./weights";
        target: "avx2";
        preprocess: "normalize_0_1";
        io: "stdio";
    }

    layer input   = Input(shape: [28, 28, 1]);
    layer conv1   = Conv2D(filters: 32, kernel: 3, stride: 1, padding: "valid");
    layer pool1   = MaxPool2D(kernel: 2);
    layer flatten = Flatten();
    layer fc1     = Dense(units: 128, activation: "relu");
    layer output  = Dense(units: 10, activation: "softmax");
}
```

## Target Use Cases

- Embedded / IoT devices (microcontrollers, edge hardware)
- Safety-critical systems (aerospace, automotive, medical — DO-178C, ISO 26262)
- Low-latency inference (real-time control, HFT, robotics)
- Minimal-dependency deployments (air-gapped, hardened containers, serverless)

## Current Scope (v0.6)

NNLang v0.6 targets **small to medium CNN, MLP, and 1D models** for inference.
The following limitations apply:

- **float32 only.** `int8` and `float64` precision are reserved for future versions.
- **Supported layers:** Input, Dense, Conv2D, Conv1D, MaxPool2D, MaxPool1D,
  AvgPool2D, GlobalAvgPool2D, Flatten, BatchNorm, LayerNorm, Dropout, Add,
  Concat, Mul, Upsample, ReLU, ReLU6, LeakyReLU, SiLU, Hardswish, Sigmoid,
  Softmax.
- **No recurrent layers** (LSTM, GRU), **no attention layers**,
  **no Reshape/Transpose**.
- **Single input, single output.** Multi-input/output models are not supported.
- **ONNX import** covers the ops listed above; unsupported ops are emitted as
  comments for manual resolution.
- **SIMD:** the compiler passes target flags (e.g., `-mavx2`) to the C compiler
  for autovectorization but does not emit hand-tuned SIMD intrinsics yet.
- **Platforms:** Linux and macOS. Windows support is not tested.

## Commands

| Command | Description |
|---------|-------------|
| `nnc new` | Generate a starter host-language project around a sample NNL model |
| `nnc compile` | Compile an NNL model to a native artifact (exe, .o, .a, .so, .h) |
| `nnc inspect` | Print graph, shapes, parameter counts, and memory estimates |
| `nnc import` | Convert an ONNX model to NNL format with extracted weights |
| `nnc test` | Verify model output against known input/output pairs |

## Requirements

- **Rust toolchain** (to build `nnc` from source)
- **C compiler** (`cc`, `gcc`, or `clang` on PATH — used by `nnc compile` to produce native code)

## Documentation

- [Language Reference](docs/language-reference.md) — NNLang v0.2 syntax, config keys, layer types, connections
- [CLI Reference](docs/cli.md) — all commands, flags, and examples
- [Weight Files](docs/weights.md) — formats, naming convention, expected shapes
- [ONNX Import](docs/onnx-import.md) — supported ops, round-trip workflow
- [Code Generation](docs/codegen.md) — C backend, output formats, integration, cross-compilation
- [Examples](docs/examples.md) — walkthroughs of included models
- [Specification](spec/specification.md) — formal NNLang v0.2 grammar and semantics

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
