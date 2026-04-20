# Code Generation

## How It Works

`nnc` generates C source code from the NNL model, then invokes the system C compiler (`cc`/`gcc`/`clang`) to produce the final artifact. This approach is documented in [DESIGN.md](../DESIGN.md) as **ADR-001: C Codegen Backend**.

The generated C contains:
- `static const float` weight arrays (placed in `.rodata` via `const`)
- Statically-allocated workspace buffers for activations
- The inference function body with kernel calls in topological order
- A `.h` header declaring the public API

## Pipeline

```
.nnl source → nnc frontend → IR → C source → cc/gcc/clang → native binary
```

1. **Frontend** — parses the `.nnl` file into an AST (`src/syntax/`)
2. **Semantic analysis** — validates layer types, resolves connections, infers shapes (`src/sema/`)
3. **IR** — builds a typed model graph with topological ordering (`src/ir/`)
4. **Weights** — loads `.npy` / `.npz` / ONNX weight tensors (`src/weights/`)
5. **C emitter** — generates a `.c` source file and `.h` header (`src/codegen/emit.rs`)
6. **Toolchain** — invokes `cc`/`gcc`/`clang` and `ar` to produce the requested artifact (`src/codegen/toolchain.rs`)

## Generated C API

For a model named `my_model`, `nnc` generates:

```c
#ifndef MY_MODEL_H
#define MY_MODEL_H

#include <stdint.h>

int my_model_infer(const void *input, void *output);
int my_model_input_size(void);   // total float elements in input tensor
int my_model_output_size(void);  // total float elements in output tensor

#endif /* MY_MODEL_H */
```

- `input` / `output` are raw `float` arrays in row-major (HWC) layout
- Returns `0` on success
- No heap allocation during inference — all buffers are `static`
- All weights are embedded as `static const float` arrays in `.rodata`

## Output Formats

| `--emit` flag | File type | What's generated | Use case |
|---|---|---|---|
| `exe` | Standalone binary | Binary with `main()` that reads `stdin` / writes `stdout` | Quick testing, CLI inference |
| `obj` | `.o` relocatable object | Object file + `.h` header | Linking into a larger C/C++ project |
| `lib` | `.a` static archive | Static library + `.h` header | Distribution as a self-contained library |
| `shared` | `.so` shared library | Shared object + `.h` header | Dynamic linking, plugins |
| `header` | `.h` file only | Header with API declarations | Inspection, IDE integration |

Under the hood, these map to standard compiler/archiver invocations:

- `exe` → `cc -O2 -o output source.c -lm`
- `obj` → `cc -O2 -c -o output.o source.c`
- `lib` → `cc -O2 -c` + `ar rcs output.a output.o`
- `shared` → `cc -O2 -shared -fPIC -o output.so source.c -lm`
- `header` → direct file copy

## Integration Example

Compile a model as a static library:

```sh
nnc compile my_model.nnl --emit lib -o libmy_model.a
```

This produces `libmy_model.a` and `my_model.h` in the same directory. Link them into your C project:

```c
#include "my_model.h"

float input[784], output[10];

int main(void) {
    // ... fill input[] with preprocessed data ...
    int rc = my_model_infer(input, output);
    if (rc != 0) return rc;
    // ... use output[] ...
    return 0;
}
```

Compile and link:

```sh
gcc -O2 -o app app.c -L. -lmy_model -lm
```

Alternatively, link a `.o` object directly:

```sh
nnc compile my_model.nnl --emit obj -o my_model.o
gcc -O2 -o app app.c my_model.o -lm
```

## Cross-Compilation

When `--target-triple` is specified, `nnc` invokes the corresponding cross-compiler instead of `cc`:

```sh
nnc compile model.nnl --emit exe --target-triple arm-none-eabi -o model
# invokes: arm-none-eabi-gcc -O2 -o model model.c -lm
```

Combine with a SIMD target in the model config for architecture-specific optimizations:

```
config {
    target: "arm_neon";
}
```

This adds `-mfpu=neon` to the compiler flags. Available targets and their flags:

| Config `target` | Compiler flag |
|---|---|
| `"generic"` | *(none)* |
| `"avx2"` | `-mavx2` |
| `"avx512"` | `-mavx512f` |
| `"arm_neon"` | `-mfpu=neon` |

## Memory Model

- **Static workspace buffers** — all activation memory is statically allocated (`static float` arrays). No `malloc` is ever called.
- **Liveness-based buffer reuse** — the codegen performs liveness analysis on the layer graph and reuses buffer slots when a layer's output is no longer needed, minimizing total activation memory.
- **Weights in read-only data** — all weight arrays are `static const float` with alignment attributes, placed in the `.rodata` section by the C compiler.
- **Alignment** — buffers and weight arrays use `__attribute__((aligned(N)))` for SIMD-friendly access patterns.
