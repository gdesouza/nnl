# Changelog

All notable changes to NNLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
