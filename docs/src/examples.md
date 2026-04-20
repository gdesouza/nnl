# Examples

The `examples/` directory contains complete, self-contained models with pre-generated weights and test data. Each example includes:

- A `.nnl` model definition
- A `weights/` directory with `.npy` weight files
- `test_input.npy` and `expected_output.npy` for verification

## Simple MLP (`examples/model/`)

**Architecture:** `[4] → Dense(3) → Dense(2)`

A minimal multi-layer perceptron with no activation functions — useful as a smoke test for the compiler pipeline.

### Model definition

```
version 0.2;
model test_mlp {
    config {
        weights: "./weights";
        io: "stdio";
    }
    layer input = Input(shape: [4]);
    layer fc1   = Dense(units: 3);
    layer fc2   = Dense(units: 2);
}
```

- **Input:** 4 floats
- **fc1:** Dense layer with 3 units (no activation), weights: `fc1.weight.npy` [4×3], `fc1.bias.npy` [3]
- **fc2:** Dense layer with 2 units (no activation), weights: `fc2.weight.npy` [3×2], `fc2.bias.npy` [2]
- **Output:** 2 floats

### Compile and test

```sh
# Compile to a standalone executable
nnc compile examples/model/model.nnl --emit exe -o mlp

# Verify against known test data
nnc test examples/model/model.nnl \
    --input examples/model/test_input.npy \
    --expected examples/model/expected_output.npy
```

## MNIST CNN (`examples/mnist/`)

**Architecture:** `[28,28,1] → Conv2D(32) → MaxPool2D(2) → Flatten → Dense(128, relu) → Dense(10, softmax)`

A convolutional neural network for MNIST handwritten digit classification.

### Model definition

```
version 0.2;

// MNIST handwritten digit classifier
model mnist_classifier {
    config {
        precision: "float32";
        weights: "./weights";
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

### Layer breakdown

| Layer | Operation | Output shape | Notes |
|---|---|---|---|
| `input` | Input | [28, 28, 1] | Single-channel grayscale image (HWC) |
| `conv1` | Conv2D | [26, 26, 32] | 32 filters, 3×3 kernel, valid padding |
| `pool1` | MaxPool2D | [13, 13, 32] | 2×2 pooling window |
| `flatten` | Flatten | [5408] | 13 × 13 × 32 = 5408 |
| `fc1` | Dense + ReLU | [128] | Fully connected with ReLU activation |
| `output` | Dense + Softmax | [10] | 10-class probability distribution |

### Preprocessing

`preprocess: "normalize_0_1"` divides each input pixel by 255.0, mapping raw `[0, 255]` byte values to `[0.0, 1.0]` floats. This is applied automatically in the generated inference code.

### Compile and test

```sh
nnc compile examples/mnist/mnist.nnl --emit exe -o mnist

nnc test examples/mnist/mnist.nnl \
    --input examples/mnist/test_input.npy \
    --expected examples/mnist/expected_output.npy
```

## ResNet Block (`examples/resnet_block/`)

**Architecture:** A residual block with skip connection using explicit `connections` and `Add`.

This example demonstrates non-sequential layer graphs — the `connections` block allows arbitrary wiring between layers, including multi-input layers like `Add`.

### Model definition

```
version 0.2;

model resnet_block {
    config {
        precision: "float32";
        weights: "./weights";
        target: "generic";
        io: "stdio";
    }

    layer input  = Input(shape: [32, 32, 64]);
    layer conv1  = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer bn1    = BatchNorm();
    layer relu1  = ReLU();
    layer conv2  = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer bn2    = BatchNorm();
    layer res    = Add();
    layer relu2  = ReLU();

    connections {
        input -> conv1;
        conv1 -> bn1;
        bn1 -> relu1;
        relu1 -> conv2;
        conv2 -> bn2;
        [input, bn2] -> res;
        res -> relu2;
    }
}
```

### Skip connection explained

The key line is `[input, bn2] -> res;` — this feeds both the original input and the output of `bn2` into the `Add` layer, creating the residual shortcut:

```
input ──→ conv1 → bn1 → relu1 → conv2 → bn2 ──┐
  │                                              │
  └──────────────────────────────────────────→ Add → relu2
```

Without the `connections` block, layers are connected sequentially in declaration order. The `connections` block overrides this default with explicit wiring.

### Weight files

BatchNorm layers require four weight files each:

- `bn1.gamma.npy`, `bn1.beta.npy` — learned scale and shift
- `bn1.running_mean.npy`, `bn1.running_var.npy` — running statistics from training

### Compile and test

```sh
nnc compile examples/resnet_block/resnet_block.nnl --emit exe -o resnet_block

nnc test examples/resnet_block/resnet_block.nnl \
    --input examples/resnet_block/test_input.npy \
    --expected examples/resnet_block/expected_output.npy
```

## ONNX Import (`examples/import_test/`)

Demonstrates the round-trip workflow: generate an ONNX model in Python, import it into NNL, compile, and verify.

**Architecture:** `[4] → Dense(3, relu) → Dense(2)`

### Step 1: Generate the ONNX model

```sh
cd examples/import_test
python3 gen_mlp.py
```

This creates:
- `model.onnx` — the ONNX model with embedded weights
- `input.npy` — test input `[1.0, 2.0, 3.0, 4.0]`
- `expected.npy` — expected output computed from the same weights

### Step 2: Import into NNL

```sh
nnc import examples/import_test/model.onnx \
    -o examples/import_test/model.nnl \
    --weights-dir examples/import_test/weights
```

This produces a `.nnl` file and extracts weight tensors into the `weights/` directory as `.npy` files.

### Step 3: Compile

```sh
nnc compile examples/import_test/model.nnl --emit exe -o import_mlp
```

### Step 4: Test

```sh
nnc test examples/import_test/model.nnl \
    --input examples/import_test/input.npy \
    --expected examples/import_test/expected.npy
```

### What gen_mlp.py does

The script builds a two-layer MLP with fixed weights using the ONNX helper API:

- Layer 1: `Gemm` (matrix multiply + bias) → `Relu`
- Layer 2: `Gemm`

It uses deterministic weights so the expected output can be computed exactly and verified after the NNL round-trip.

## Creating Your Own Model

### 1. Write the `.nnl` file

Define your architecture with layer declarations and an optional `connections` block:

```
version 0.2;
model my_model {
    config {
        weights: "./weights";
        io: "stdio";
    }
    layer input = Input(shape: [784]);
    layer fc1   = Dense(units: 64, activation: "relu");
    layer fc2   = Dense(units: 10, activation: "softmax");
}
```

### 2. Create the weights directory

Each layer expects specific `.npy` files named `<layer_id>.<param>.npy`:

| Layer type | Weight files |
|---|---|
| Dense | `<id>.weight.npy`, `<id>.bias.npy` |
| Conv2D | `<id>.weight.npy`, `<id>.bias.npy` |
| BatchNorm | `<id>.gamma.npy`, `<id>.beta.npy`, `<id>.running_mean.npy`, `<id>.running_var.npy` |

### 3. Generate weights with NumPy

```python
import numpy as np

np.save("weights/fc1.weight.npy", np.random.randn(784, 64).astype(np.float32))
np.save("weights/fc1.bias.npy",   np.zeros(64, dtype=np.float32))
np.save("weights/fc2.weight.npy", np.random.randn(64, 10).astype(np.float32))
np.save("weights/fc2.bias.npy",   np.zeros(10, dtype=np.float32))
```

### 4. Compile

```sh
nnc compile my_model.nnl --emit exe -o my_model
```

### 5. Test

Generate test inputs and expected outputs, then verify:

```sh
nnc test my_model.nnl --input test_input.npy --expected expected_output.npy
```

The default tolerance is `1e-5` (element-wise). Adjust with `--tolerance` if needed.
