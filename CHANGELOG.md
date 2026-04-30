# Changelog

All notable changes to NNLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.8.0] — 2026-04-30

### Added

- **`nnc new` project scaffolding** — generate a starter host-language project around a sample NNL model. Supports `--project rust`, `go`, `cpp`, and `python`. The scaffold includes a sample `model.nnl` (configured with `io: "none"`), host-language boilerplate wired to the generated C ABI, a build script or build file for compiling the model artifact, and a README with run instructions.

### Changed

- **Improved missing-weight diagnostics (E003)** — `nnc compile` now produces a structured, actionable error when required weights are missing. Errors list every missing tensor with its expected shape, identify whether the source is a directory of `.npy` files, an `.npz` archive, or another path, and include a `hint:` to run `nnc inspect <model>` to view expected tensors and shapes. All missing weights are reported in a single error instead of stopping at the first one.

## [0.7.0] — 2026-04-26

### Added

- **Compile-time memory check with optional `memory_limit` config** — `nnc` now computes total static memory (weights + workspace) and emits a W003 warning when it exceeds 256 MB. Add `memory_limit: "128MB"` to the config block to turn this into a hard compile error (E009). `nnc inspect` now shows a "Total memory" line. Accepted units: KB, MB, GB.
- **`io: "none"` config option** — skips `main()` generation, producing a pure library artifact. Use with `--emit lib`, `--emit shared`, or `--emit obj` for embedding models in host applications. `io: "none"` with `--emit exe` produces a clear compile error.
- **Integration examples** — new `examples/integration/` directory with documented examples showing how to call an NNL-compiled model from C++, Rust, Go, and Python, using static/shared library linking and FFI.

## [0.6.0] — 2026-04-23

### Added

- **New layers: Hardswish, Upsample, Conv1D, MaxPool1D, LayerNorm** — five new layer types across all pipeline stages (lexer, parser, IR, shape inference, codegen, ONNX import), completing the Tier 4 roadmap from the ONNX spec.
- **Hardswish activation** — `Hardswish(x) = x * min(max(0, x+3), 6) / 6`, unlocks MobileNetV3. ONNX `HardSwish` op imported automatically.
- **Upsample layer** — `Upsample(scale: N)` with nearest-neighbor interpolation for spatial upsampling. ONNX `Upsample` and `Resize` ops imported automatically. Unlocks YOLO-Tiny, U-Net, and encoder-decoder models.
- **Conv1D layer** — 1D convolution with `filters`, `kernel`, `stride`, `padding` parameters. ONNX `Conv` ops with 3D weight tensors auto-detected as Conv1D. Enables audio, time-series, and keyword spotting models.
- **MaxPool1D layer** — 1D max pooling with `kernel` and optional `stride`. ONNX `MaxPool` ops with 1D `kernel_shape` auto-detected. Enables audio and time-series models.
- **LayerNorm layer** — Layer normalization with learnable `scale` and `bias` over the last dimension, with configurable `epsilon`. ONNX `LayerNormalization` op imported with epsilon and weights. Enables transformer-adjacent models.

## [0.5.0] — 2026-04-23

### Added

- **New layers: GlobalAvgPool2D, ReLU6, LeakyReLU, SiLU, Mul** — six new layer types across all pipeline stages (lexer, parser, IR, shape inference, codegen, ONNX import), unlocking ResNet-18, MobileNetV1/V2, and EfficientNet model families.
- **Grouped / depthwise Conv2D** — `Conv2D` now accepts a `groups` parameter (default 1) for grouped convolution, including depthwise separable convolution (`groups == in_channels`). ONNX `Conv` `group` attribute is imported automatically.
- **ONNX external tensor data support** — `nnc import` can now load weights stored as external data files (ONNX `data_location = EXTERNAL`) with offset/length support, fixing import failures for models exported with `torch.onnx.export(..., use_external_data_format=True)`.

### Fixed

- **CHW→HWC weight permutation at Flatten→Dense boundary** — `nnc import` now automatically detects the Flatten→Gemm pattern in ONNX graphs and permutes Dense weight matrix rows from PyTorch's CHW flatten order to nnc's HWC order, fixing incorrect inference results for all imported CNNs with Flatten→Dense transitions.
- **ONNX import empty tensor error** — `nnc import` now produces a clear error message (`"tensor '...' has no data"`) instead of a cryptic npy shape mismatch when tensor data is missing.

## [0.4.0] — 2026-04-23

### Added

- **`--version` / `-V` flag** — `nnc --version` now prints the version from `Cargo.toml`.
- **`--emit c` flag** — `nnc compile model.nnl --emit c` writes the generated `.c` and `.h` files directly without invoking the C compiler, useful for debugging and auditing generated code.

### Fixed

- **Concat codegen for multi-dimensional tensors** — fixed incorrect flat `memcpy` in Concat codegen that produced wrong results when concatenating 3D (HWC) tensors along the channel axis. Now generates proper strided copies for arbitrary concat axes.
- **ONNX import protobuf decode failure** — fixed incorrect field tag numbers in `AttributeProto` that caused all ONNX imports to fail with a protobuf wire type error. Added missing `floats` field (tag 7).
- **Unsupported precision silently accepted** — `precision: "int8"` and `precision: "float64"` now produce a compile error instead of silently generating incorrect float32 code.
- **Website hero demo** — the output example now shows the realistic workflow (raw bytes piped through Python) instead of implying the binary outputs formatted text.
- **Website copyright year** — updated from `© 2024` to `© 2024–2025`.
- **README DESIGN.md link** — corrected broken link to point to `docs/src/DESIGN.md`.

## [0.3.0] — 2025-04-23

### Fixed

- **Conv2D rectangular kernel correctness** — fixed a bug where non-square kernels (e.g., `kernel: [3, 5]`) produced incorrect inference results due to a variable shadowing issue in the generated C code. Square kernels were unaffected. The same shadowing fix was applied to MaxPool2D and AvgPool2D codegen for consistency.

## [0.2.0] — 2025-04-20

Initial public release.

### Added

- NNLang DSL with `version 0.2` syntax for defining neural network models
- Layers: Input, Dense, Conv2D, MaxPool2D, AvgPool2D, Flatten, BatchNorm, Dropout, Add, Concat, ReLU, Sigmoid, Softmax
- C code generation backend with static memory allocation (no heap, no runtime dependencies)
- Output formats: `exe`, `obj`, `lib`, `shared`, `header`
- Cross-compilation via `--target-triple` flag
- SIMD target hints: `generic`, `avx2`, `avx512`, `arm_neon`
- Weight loading from `.npy` files and `.npz` archives
- ONNX model import via `nnc import`
- `nnc inspect` command for model summary and shape information
- `nnc test` command for verifying inference correctness against expected outputs
- Explicit graph connections with `connections { }` block and skip connections
- Liveness-based buffer reuse for minimal activation memory footprint
- mdbook documentation site
