# NNL Integration Examples

This directory contains examples showing how to call an NNL-compiled model
from **C++**, **Rust**, **Go**, and **Python**. All examples use the same
minimal 2-layer MLP defined in [`model.nnl`](model.nnl).

## Compile the model

Before building any example, compile the model into the artifacts you need:

```sh
# Static library (for C++, Rust, Go)
nnc compile model.nnl --emit lib -o libsimple_mlp.a

# C header (for C++ and any FFI consumer)
nnc compile model.nnl --emit header -o simple_mlp.h

# Shared library (for Python / ctypes)
nnc compile model.nnl --emit shared -o libsimple_mlp.so
```

## Language examples

| Language | Directory | Linkage |
|----------|-----------|---------|
| C++      | [`cpp/`](cpp/) | Static library (`.a`) via Makefile |
| Rust     | [`rust/`](rust/) | Static library (`.a`) via `build.rs` FFI |
| Go       | [`go/`](go/) | Static library (`.a`) via cgo |
| Python   | [`python/`](python/) | Shared library (`.so`) via ctypes + numpy |

Each subdirectory has its own build file or instructions.
