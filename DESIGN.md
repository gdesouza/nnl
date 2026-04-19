# NNL Design Decisions

## MVP Constraints (v0.2)

These constraints scope the first end-to-end release of `nnc`:

- **float32 only.** int8 quantization requires unspecified metadata (scaling, saturation semantics) and is deferred.
- **generic target first.** AVX2/AVX512/ARM NEON specialization is Phase 7 work.
- **Single input tensor, single output tensor.** The C ABI (`model_name_infer(const void*, void*)`) assumes one of each. Multi-input/output is a future spec extension.
- **HWC tensor convention.** Activation shapes exclude batch; `config.batch` is prepended internally. Shapes like `[28, 28, 1]` follow height × width × channels ordering.
- **Weight format priority:** directory of `.npy` files → `.npz` archives → `.onnx` initializers.
- **Bare `.npy`** is valid only when exactly one tensor is expected; otherwise a clear error is emitted.

## ADR-001: C Codegen Backend

**Status:** Accepted  
**Date:** 2026-04-18

### Context

`nnc` must produce standalone native artifacts (executables, object files, static libraries, shared libraries) with embedded weights, zero heap allocation, and a C-ABI inference function. Three backend strategies were evaluated:

1. **Direct machine code emission** — write raw instructions and ELF/Mach-O/PE object files using a crate like `object`.
2. **LLVM or Cranelift** — generate IR for an existing compiler backend.
3. **Emit C source → invoke system C compiler.**

### Decision

**Emit C source code and invoke the host (or cross) C compiler** (`cc`/`gcc`/`clang`) to produce final machine code.

`nnc` generates a `.c` file containing:
- `static const float` weight arrays (placed in `.rodata` via `const`)
- A statically-allocated workspace buffer
- The inference function body with kernel calls in topological order
- A `.h` header declaring the public API

Then it invokes the system C compiler to produce the requested output format.

### Rationale

**Free optimizations.** `gcc -O2` / `clang -O2` provides vectorization, loop unrolling, constant folding, and register allocation. These are professional-grade optimizations that would take months to replicate.

**Trivial output format support.** All `--emit` modes map to standard compiler/archiver invocations:
- `--emit obj` → `cc -c`
- `--emit exe` → `cc` (with a generated `main()`)
- `--emit lib` → `cc -c` + `ar rcs`
- `--emit shared` → `cc -shared`

**Cross-compilation for free.** `--target-triple thumbv7em-none-eabi` simply invokes `arm-none-eabi-gcc`. The entire cross-compilation ecosystem already exists.

**Automatic C ABI compliance.** The spec requires a C calling convention. Emitting actual C guarantees correct struct layout, alignment, and calling convention — no manual ABI bugs.

**SIMD via intrinsics.** Target-specific kernels (Phase 7) emit C intrinsic calls like `_mm256_fmadd_ps()`. The C compiler handles register allocation and instruction scheduling.

**Debuggability.** The generated `.c` file is human-readable and inspectable. An `--emit-c` debug flag can preserve it for verification.

**Small implementation surface.** The C emitter is ~1000–2000 lines of Rust `write!()` calls. An LLVM backend would be 5–10× larger.

### Tradeoff

The only real downside is a **runtime dependency on a C compiler** on the build machine. This is the same requirement as `cargo` (which invokes `cc` for build scripts and `-sys` crates), Go's `cgo`, and Zig's build system. For `nnc`'s embedded/safety-critical audience, a C cross-toolchain is always present.

### When to Reconsider

Replace the C backend only if:
- Embedding very large weight arrays as C literals exceeds C compiler memory limits.
- Fused kernels require control that C intrinsics cannot express.
- `nnc` must be fully self-contained with zero external tool dependencies.

At that point, only the last codegen stage is replaced — the frontend, IR, semantic analysis, and memory planner remain unchanged.
