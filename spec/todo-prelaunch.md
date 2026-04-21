# NNLang v0.2 → Public Launch: Prioritized Action Plan

**Date:** 2026-04-20
**Status:** Draft
**Goal:** Bridge the gaps identified during pre-launch review so that NNLang can be announced publicly without misleading users or causing immediate frustration.

---

## Guiding Principles

1. **Correctness over features.** Fix bugs that produce silently wrong results before adding anything new.
2. **Honesty over polish.** Clearly document what works and what doesn't. Users forgive missing features; they don't forgive broken promises.
3. **First-hour experience.** Optimize the path from `cargo install nnlang` to a successful `nnc compile` + correct inference result.
4. **Scope clarity.** NNLang v0.2 targets small CNN/MLP models for embedded inference. Say so explicitly.

---

## Phase 0: Correctness Fixes (BLOCKING — do before any announcement)

These are bugs that can produce **silently wrong inference results**. Ship these or lose all credibility.

### 0.1 Fix Conv2D weight indexing for rectangular kernels

**Severity:** Critical — incorrect output for any model with non-square kernels
**File:** `src/codegen/emit.rs`, lines 306 and 339

The generated C code uses `kh` as both the outer loop variable name and the constant kernel height in the weight index expression:

```c
// Current (wrong for rectangular kernels):
sum += src[...] * w_var[((f * ic + ci) * kh + kh) * kw + kw_];
//                                        ^^   ^^
//                              constant ──┘    └── should be loop variable
```

The loop variable `kh` shadows the constant `kh` in the generated C. When `kh == kw` (square kernels) this is harmless. When `kh != kw`, the weight index is wrong.

**Fix:** Rename the loop variables in the generated C to avoid shadowing (e.g., `kh_` and `kw_` for loop vars, keep `kh`/`kw` as constants). Apply to both `Padding::Valid` and `Padding::Same` branches, and to `MaxPool2D` and `AvgPool2D` which have the same pattern.

**Verification:**
- Add an integration test with a rectangular kernel (e.g., `kernel: [3, 5]`) and known weights/expected output.
- Verify the existing CNN test still passes.

### 0.2 Fix Concat codegen for multi-dimensional tensors

**Severity:** High — incorrect output for channel-axis concat on 3D tensors
**File:** `src/codegen/emit.rs`, lines 264–278

The current Concat implementation copies contiguous blocks:

```c
memcpy(dst + offset, src, cat_elems * sizeof(float));
offset += cat_elems;
```

This is only correct when concatenating along the **last** axis of a **1D** tensor. For 3D HWC tensors concatenated along the channel axis (the primary use case for Concat), the data is interleaved and cannot be copied as contiguous blocks.

**Fix:** Generate nested loops that iterate over the spatial dimensions and copy channel slices to the correct positions. Handle axis=0, axis=1, and axis=-1 (channel) cases.

**Verification:**
- Add an integration test: two `[4, 4, 2]` tensors concatenated along axis -1 → `[4, 4, 4]`.
- Verify element ordering matches NumPy's `np.concatenate(..., axis=-1)`.

### 0.3 Guard against unsupported precision in codegen

**Severity:** High — `precision: "int8"` or `"float64"` parses and validates but generates incorrect `float`-typed C code
**File:** `src/sema/validate.rs` or `src/ir/lower.rs`

**Fix (option A — recommended for v0.2):** Emit a compile error if `precision` is anything other than `"float32"`. Remove `"float64"` and `"int8"` from the accepted values until codegen supports them.

**Fix (option B):** Emit a clear warning: `W003: precision '{p}' is not yet supported by codegen; using float32`.

**Verification:**
- The existing `error_invalid_precision` test covers `"float16"`. Add a test that `"int8"` and `"float64"` produce an error (option A) or warning (option B).

---

## Phase 1: Documentation Honesty (do before announcement)

### 1.1 Add a "Scope & Limitations" section to README.md

Insert after "Target Use Cases", before "Commands". Content:

```markdown
## Current Scope (v0.2)

NNLang v0.2 targets **small to medium CNN and MLP models** for inference.
The following limitations apply:

- **float32 only.** `int8` and `float64` precision are reserved for future versions.
- **Supported layers:** Input, Dense, Conv2D, MaxPool2D, AvgPool2D, Flatten,
  BatchNorm, Dropout, Add, Concat, ReLU, Sigmoid, Softmax.
- **No recurrent layers** (LSTM, GRU), **no transformer layers** (Attention,
  LayerNorm), **no 1D convolutions**, **no Reshape/Transpose**.
- **Single input, single output.** Multi-input/output models are not supported.
- **ONNX import** covers the ops listed above; unsupported ops are emitted as
  comments for manual resolution.
- **SIMD:** the compiler passes target flags (e.g., `-mavx2`) to the C compiler
  for autovectorization but does not emit hand-tuned SIMD intrinsics yet.
- **Platforms:** Linux and macOS. Windows support is not tested.
```

### 1.2 Fix the website hero demo

**File:** `website/index.html`, line 58–59

The demo shows `[0.87, 0.13]` as text output from the compiled binary, but the binary actually writes raw float32 bytes to stdout. Users who run this will see garbled binary data, not formatted numbers.

**Fix:** Change the demo to show the realistic workflow:

```
$ cat input.bin | ./my_model > output.bin
$ python3 -c "import numpy as np; print(np.frombuffer(open('output.bin','rb').read(), dtype=np.float32))"
[0.87 0.13]
```

Or add a note: "Output is raw float32 bytes. Use `nnc test` to verify results."

### 1.3 Fix the website copyright year

**File:** `website/index.html`, line 241

Change `© 2024` to `© 2025` or `© 2024–2025`.

### 1.4 Fix README DESIGN.md link

The README links to `DESIGN.md` at the repo root, but the file lives at `docs/src/DESIGN.md`.

### 1.5 Update spec/docs to clarify SIMD status

In `docs/src/codegen.md`, the "Available targets and their flags" table implies NNLang handles SIMD optimization. Add a note:

> **Note:** Target flags enable the C compiler's autovectorizer. Hand-tuned SIMD
> intrinsics (AVX2, NEON) are planned for a future release. The generated C code
> uses scalar loops that the C compiler may vectorize automatically.

---

## Phase 2: First-Hour Experience (do before or shortly after announcement)

### 2.1 Add `--emit-c` flag to preserve generated C source

**Motivation:** Referenced in DESIGN.md as a debug flag. Users debugging correctness issues need to see the generated code. Compiler developers need it for troubleshooting. This is trivial to implement.

**Implementation:**
- Add `--emit-c` to `EmitFormat` enum in `src/cli.rs`.
- In `src/codegen/toolchain.rs`, when `--emit c` is selected, write the `.c` and `.h` files directly to the output path without invoking the C compiler.

**Verification:** `nnc compile model.nnl --emit c -o model.c` produces readable C source.

### 2.2 Add `--version` flag

**Implementation:** Add `#[command(version)]` to the `Cli` struct in `src/cli.rs`. Clap will use the version from `Cargo.toml`.

### 2.3 Improve error messages for missing weight files

Currently, a missing weights directory produces a generic error. Enhance the error to:
- List exactly which weight files are expected
- Show the directory that was searched
- Suggest the `nnc inspect` command to see expected shapes

### 2.4 Add a scaffold/init command

```
nnc init my_model --input-shape [28,28,1] --output-units 10
```

Generates:
- `my_model.nnl` with a basic sequential model
- `weights/` directory with random `.npy` weight files of the correct shapes
- `test_input.npy` with random input data

This bridges the gap between "write .nnl file" and "successfully compile". Even a minimal version (just generating the `.nnl` file) significantly improves onboarding.

**Alternative (lower effort):** Add a `nnc check model.nnl` command that runs the frontend pipeline without requiring weights, so users can validate their `.nnl` syntax and shapes before creating weight files. (Note: `nnc inspect` already does this — document it more prominently as the "validate your model" step.)

---

## Phase 3: Layer Coverage (post-announcement, driven by user demand)

### Priority order based on frequency in real-world ONNX models:

| Priority | Layer | Rationale |
|----------|-------|-----------|
| P0 | **Reshape** | Appears in virtually every ONNX model. Required for any non-trivial import. |
| P0 | **GlobalAvgPool2D** | Standard in ResNet, MobileNet, EfficientNet. Trivial to implement (average all spatial dims). |
| P1 | **Transpose / Permute** | Needed for layout conversions in imported models. |
| P1 | **Conv1D / MaxPool1D** | Opens up audio, time-series, and 1D signal processing models. |
| P2 | **DepthwiseConv2D** | Required for MobileNet and all efficient architectures. |
| P2 | **Resize / Upsample** | Required for U-Net, encoder-decoder architectures. |
| P2 | **LayerNorm** | Needed for any transformer-adjacent model. |
| P3 | **Pad / Slice / Gather** | Common ONNX ops for model surgery. |
| P3 | **MatMul** (standalone) | Needed for attention mechanisms. |
| P3 | **LSTM / GRU** | Temporal models. Significant implementation effort. |

### Implementation pattern for each new layer:

1. Add variant to `LayerKind` enum in `src/ir/model.rs`
2. Add parser support in `src/syntax/parser.rs` and `src/ir/lower.rs`
3. Add shape inference in `src/sema/shapes.rs`
4. Add C codegen in `src/codegen/emit.rs`
5. Add ONNX op mapping in `src/import/onnx.rs`
6. Add integration test in `tests/compile.rs`
7. Update `docs/src/language-reference.md` and `spec/specification.md`

---

## Phase 4: Performance & Codegen (post-announcement)

### 4.1 SIMD intrinsics for Dense and Conv2D

Start with AVX2 for `Dense` (matrix-vector multiply) since it's the highest-impact, lowest-complexity target. Generate `_mm256_fmadd_ps()` calls conditioned on `config.target`.

### 4.2 Fused operations

Fuse common patterns in the IR before codegen:
- `Conv2D → BatchNorm` → fold BN parameters into conv weights
- `Dense → ReLU` → already handled inline, but formalize as an IR pass
- `Conv2D → ReLU` → add inline activation to conv codegen

### 4.3 Memory planner improvements

The current buffer planner allocates every slot at `max_activation` size. Allocate each slot at its actual required size to reduce total static memory footprint.

---

## Phase 5: Platform & Ecosystem (months 2–6)

### 5.1 Windows support

- Support MSVC (`cl.exe`) in `src/codegen/toolchain.rs`
- Emit `.dll` for `--emit shared` on Windows
- Add Windows to the CI matrix and release workflow
- Test with MinGW-w64 as a fallback

### 5.2 VS Code extension with syntax highlighting

A TextMate grammar for `.nnl` files. No LSP needed initially — just syntax highlighting, bracket matching, and comment toggling. This is high-impact, low-effort.

### 5.3 `nnc bench` command

Compile and run the model N times, report average/p99 inference latency. Essential for users who care about "low-latency inference."

### 5.4 Expand ONNX import coverage

Track which ONNX ops are encountered but unsupported. Emit a summary at the end of import:

```
nnc: imported 12/15 operators
nnc: unsupported: Reshape (2 occurrences), Transpose (1 occurrence)
```

---

## Summary: Pre-Launch Checklist

```
MUST DO (blocks announcement):
  [0.1] Fix Conv2D rectangular kernel weight indexing bug
  [0.2] Fix Concat codegen for multi-dimensional tensors
  [0.3] Guard against unsupported precision in codegen
  [1.1] Add "Scope & Limitations" section to README
  [1.2] Fix website hero demo (raw bytes, not formatted text)

SHOULD DO (same week as announcement):
  [1.3] Fix website copyright year
  [1.4] Fix README DESIGN.md link
  [1.5] Clarify SIMD status in codegen docs
  [2.1] Add --emit-c flag
  [2.2] Add --version flag

NICE TO HAVE (first two weeks):
  [2.3] Improve missing-weights error messages
  [2.4] Add nnc init or document nnc inspect as validation step
```

---

## Appendix: Verification Commands

```bash
# Run full test suite after changes
cargo test

# Verify the examples still work end-to-end
nnc test examples/model/model.nnl \
    --input examples/model/test_input.npy \
    --expected examples/model/expected_output.npy

nnc test examples/mnist/mnist.nnl \
    --input examples/mnist/test_input.npy \
    --expected examples/mnist/expected_output.npy

nnc test examples/resnet_block/resnet_block.nnl \
    --input examples/resnet_block/test_input.npy \
    --expected examples/resnet_block/expected_output.npy
```
