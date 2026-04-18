# NNL: Motivation

## The Problem

Deploying neural network models today typically requires shipping heavyweight runtime dependencies — Python interpreters, framework libraries (PyTorch, TensorFlow), or inference engines (ONNX Runtime, TensorRT). These runtimes introduce dynamic memory allocation, large binary footprints, and complex dependency chains that are impractical in many target environments.

## Why NNL?

NNL addresses this gap by providing a declarative language paired with the `nnc` compiler that produces **standalone, zero-dependency native binaries** with embedded weights. The key properties of the output — no heap allocation, no external runtime, deterministic memory usage — make it uniquely suited for environments where conventional ML deployment stacks cannot reach.

## Target Use Cases

### Embedded and IoT Devices
Microcontrollers and edge devices often have no operating system, limited memory, and no package manager. A single static binary with baked-in weights is the natural deployment unit for running inference on sensor data, anomaly detection, or on-device classification.

### Safety-Critical Systems
Aerospace, automotive, and medical device software must meet strict certification requirements (DO-178C, ISO 26262, IEC 62304). Deterministic memory behavior — no `malloc`, no garbage collection, no runtime surprises — is a prerequisite. NNL's heap-free, statically-allocated output aligns directly with these constraints.

### Low-Latency Inference
Real-time control loops, high-frequency trading systems, and robotics demand predictable, minimal-overhead execution. Eliminating runtime interpretation and ensuring SIMD-optimized machine code reduces latency to the hardware floor.

### Minimal-Dependency Deployments
Air-gapped environments, hardened containers, and serverless functions benefit from binaries that carry no transitive dependencies. Fewer dependencies mean a smaller attack surface and faster cold starts.

## Landscape

Existing projects validate the "compile the model to native code" approach — Apache TVM, IREE, Glow, and microTVM all target ahead-of-time compilation of neural networks. NNL has two distinctive aspects:

1. **Human-readable DSL as input.** Rather than importing opaque serialized graphs (ONNX, TFLite), the model architecture is defined in a plain-text, version-controllable, auditable source file.
2. **Zero-runtime, zero-heap target.** The compiler is designed from the ground up to produce binaries that require no runtime support and perform no dynamic allocation — a stricter contract than most general-purpose ML compilers offer.

## Design Philosophy

- **Simplicity over generality.** Support a focused set of layer types well, rather than chasing full operator coverage prematurely.
- **Transparency.** The NNL source, the weights, and the compilation target are all explicit and inspectable — no hidden graph transformations.
- **Systems-first.** `nnc` is written in Rust and targets bare-metal-capable output, treating inference as a systems programming problem rather than a data science workflow.
