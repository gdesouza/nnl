# NNL — Neural Network Language

> **⚠️ This project is currently in the specification phase.** No compiler or runtime code has been implemented yet. The language design and tooling interface are being defined and are subject to change.

NNL is a declarative language for defining neural network architectures, paired with the `nnc` compiler that will produce **standalone, zero-dependency native binaries** with embedded weights.

## Goals

- **No runtime dependencies** — compiled models are self-contained static binaries.
- **No heap allocation** — all memory is statically allocated at compile time.
- **Human-readable source** — model architectures are defined in plain-text `.nnl` files, version-controllable and auditable.
- **Systems-first** — `nnc` targets bare-metal-capable output, treating inference as a systems programming problem.

## Target Use Cases

- Embedded / IoT devices (microcontrollers, edge hardware)
- Safety-critical systems (aerospace, automotive, medical — DO-178C, ISO 26262)
- Low-latency inference (real-time control, HFT, robotics)
- Minimal-dependency deployments (air-gapped, hardened containers, serverless)

## Example

```
version 0.2;

model mnist_classifier {
    config {
        precision: "float32";
        weights: "./weights/mnist.npz";
        target: "avx2";
        batch: 1;
        preprocess: "normalize_0_1";
        io: "stdio";
    }

    layer input   = Input(shape: [28, 28, 1]);
    layer conv1   = Conv2D(filters: 32, kernel: 3, stride: 1, padding: "valid");
    layer pool1   = MaxPool2D(kernel: 2);
    layer flatten  = Flatten();
    layer fc1     = Dense(units: 128, activation: "relu");
    layer output  = Dense(units: 10, activation: "softmax");
}
```

## Documentation

- [Motivation](spec/motivation.md) — problem statement and design philosophy
- [Specification](spec/specification.md) — language grammar, layer definitions, compilation strategy, and CLI design

## Status

🔬 **Specification phase** — the language grammar (v0.2), layer types, weight mapping, compilation strategy, and CLI interface are being designed. Contributions to the specification and feedback on the language design are welcome.
