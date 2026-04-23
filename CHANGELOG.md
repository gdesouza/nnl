# Changelog

All notable changes to NNLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`--version` / `-V` flag** — `nnc --version` now prints the version from `Cargo.toml`.

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
