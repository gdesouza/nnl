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

## VGG Block (`examples/vgg_block/`)

**Architecture:** `[32,32,3] → Conv2D(64)×2 → AvgPool2D(2) → Flatten → Dense(256, relu) → Dropout(0.5) → Dense(10, softmax)`

A VGG-style CNN block for CIFAR-10 classification. Demonstrates stacked convolutions before pooling, `AvgPool2D`, and `Dropout`.

### Model definition

```
version 0.2;

// VGG-style CNN block for CIFAR-10 classification
model vgg_block {
    config {
        precision: "float32";
        weights: "./weights";
        target: "generic";
        io: "stdio";
    }

    layer input   = Input(shape: [32, 32, 3]);
    layer conv1   = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer conv2   = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer pool    = AvgPool2D(kernel: 2);
    layer flatten = Flatten();
    layer fc1     = Dense(units: 256, activation: "relu");
    layer drop    = Dropout(rate: 0.5);
    layer output  = Dense(units: 10, activation: "softmax");
}
```

### Key features

- **AvgPool2D:** Average pooling instead of max pooling — useful for smoother feature maps.
- **Dropout:** A no-op during inference, but preserved from training frameworks so the model definition stays faithful to the original.
- **Stacked Conv2D:** Two 3×3 convolutions before pooling gives a 5×5 effective receptive field with fewer parameters.

### Compile and test

```sh
nnc compile examples/vgg_block/vgg_block.nnl --emit exe -o vgg_block

nnc test examples/vgg_block/vgg_block.nnl \
    --input examples/vgg_block/test_input.npy \
    --expected examples/vgg_block/expected_output.npy
```

## Binary Classifier (`examples/binary_classifier/`)

**Architecture:** `[16] → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid`

A binary classifier MLP using standalone activation layers instead of inline activations on Dense.

### Model definition

```
version 0.2;

// Binary classifier MLP for tabular data
// Dense layers with standalone ReLU and Sigmoid activations
model binary_classifier {
    config {
        weights: "./weights";
        io: "stdio";
    }

    layer input   = Input(shape: [16]);
    layer fc1     = Dense(units: 64);
    layer relu1   = ReLU();
    layer fc2     = Dense(units: 32);
    layer relu2   = ReLU();
    layer fc3     = Dense(units: 1);
    layer sigmoid = Sigmoid();
}
```

### Key features

- **Standalone activations:** `ReLU()` and `Sigmoid()` as separate layers rather than Dense parameters. This matches the graph structure of many ONNX exports.
- **Sigmoid output:** Produces a single probability value in `[0, 1]` for binary classification.

### Compile and test

```sh
nnc compile examples/binary_classifier/binary_classifier.nnl --emit exe -o binary_classifier

nnc test examples/binary_classifier/binary_classifier.nnl \
    --input examples/binary_classifier/test_input.npy \
    --expected examples/binary_classifier/expected_output.npy
```

## Inception Module (`examples/inception_module/`)

**Architecture:** Three parallel Conv2D branches (1×1, 3×3, 5×5) merged via `Concat`.

A simplified Inception-style module demonstrating parallel branches and channel-wise concatenation.

### Model definition

```
version 0.2;

// Simplified Inception module: three parallel convolution branches
// (1x1, 3x3, 5x5) concatenated along the channel axis.

model inception_module {
    config {
        precision: "float32";
        weights: "./weights";
        target: "generic";
        io: "stdio";
    }

    layer input   = Input(shape: [32, 32, 64]);
    layer conv1x1 = Conv2D(filters: 32, kernel: 1, stride: 1, padding: "same");
    layer conv3x3 = Conv2D(filters: 32, kernel: 3, stride: 1, padding: "same");
    layer conv5x5 = Conv2D(filters: 32, kernel: 5, stride: 1, padding: "same");
    layer concat  = Concat();
    layer bn      = BatchNorm();
    layer relu    = ReLU();

    connections {
        input -> conv1x1;
        input -> conv3x3;
        input -> conv5x5;
        [conv1x1, conv3x3, conv5x5] -> concat;
        concat -> bn;
        bn -> relu;
    }
}
```

### Connection graph

```
           ┌→ conv1x1 (32 filters) ──┐
input ────→├→ conv3x3 (32 filters) ──├→ Concat → BatchNorm → ReLU
           └→ conv5x5 (32 filters) ──┘
```

### Key features

- **Concat:** Channel-wise concatenation of three branches (32+32+32 = 96 output channels).
- **Multi-input bracket syntax:** `[conv1x1, conv3x3, conv5x5] -> concat;` feeds all three branches into the Concat layer.
- **Parallel branches:** The `connections` block wires `input` to all three convolutions independently.

### Compile and test

```sh
nnc compile examples/inception_module/inception_module.nnl --emit exe -o inception_module

nnc test examples/inception_module/inception_module.nnl \
    --input examples/inception_module/test_input.npy \
    --expected examples/inception_module/expected_output.npy
```

## Feature Extractor (`examples/feature_extractor/`)

**Architecture:** `[224,224,3] → Conv2D(32,7) → BN → ReLU → MaxPool → Conv2D(64,3) → BN → ReLU → MaxPool → Flatten → Dense(256) → ReLU → Dense(10) → Softmax`

A CNN feature extractor with ImageNet-style preprocessing and standalone `Softmax`.

### Model definition

```
version 0.2;

// CNN feature extractor with ImageNet-style preprocessing and standalone Softmax
model feature_extractor {
    config {
        precision: "float32";
        weights: "./weights";
        target: "avx2";
        io: "stdio";
        preprocess: "standardize";
        preprocess_mean: [0.485, 0.456, 0.406];
        preprocess_std: [0.229, 0.224, 0.225];
    }

    layer input   = Input(shape: [224, 224, 3]);
    layer conv1   = Conv2D(filters: 32, kernel: 7, stride: 2, padding: "valid");
    layer bn1     = BatchNorm();
    layer relu1   = ReLU();
    layer pool1   = MaxPool2D(kernel: 3, stride: 2);
    layer conv2   = Conv2D(filters: 64, kernel: 3, padding: "valid");
    layer bn2     = BatchNorm();
    layer relu2   = ReLU();
    layer pool2   = MaxPool2D(kernel: 2);
    layer flatten = Flatten();
    layer fc1     = Dense(units: 256);
    layer relu3   = ReLU();
    layer fc2     = Dense(units: 10);
    layer output  = Softmax();
}
```

### Key features

- **Standalone Softmax:** Used as a separate layer rather than a Dense activation parameter.
- **ImageNet preprocessing:** `preprocess: "standardize"` with per-channel mean and std — the generated binary applies `(x - mean) / std` per channel automatically.
- **Strided convolution:** `Conv2D(kernel: 7, stride: 2)` for aggressive spatial downsampling.
- **MaxPool2D with stride:** `MaxPool2D(kernel: 3, stride: 2)` allows kernel/stride to differ.

### Compile and test

```sh
nnc compile examples/feature_extractor/feature_extractor.nnl --emit exe -o feature_extractor

nnc test examples/feature_extractor/feature_extractor.nnl \
    --input examples/feature_extractor/test_input.npy \
    --expected examples/feature_extractor/expected_output.npy
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
