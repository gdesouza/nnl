# NNLang Language Reference (v0.2)

A practical reference for writing `.nnl` model files consumed by the `nnc` compiler.

---

## File Structure

An NNL file has a fixed top-level structure:

```
version 0.2;

model <name> {
    config { ... }

    layer <id> = <LayerType>(<params>);
    ...

    connections { ... }   // optional
}
```

| Section | Required | Purpose |
|---------|----------|---------|
| `version` | No (warns if absent) | Declares the NNL spec version. |
| `model` | Yes | Names the model. Determines the generated C symbols (e.g., `model_name_infer`). |
| `config` | Yes | Global compilation settings (precision, weights path, target, etc.). |
| Layers | Yes | One or more `layer` declarations defining the network. |
| `connections` | No | Explicit data-flow graph. Omit for simple sequential models. |

---

## Comments

```
// Line comment — extends to end of line

/* Block comment —
   can span multiple lines */
```

---

## Config Block

The `config` block sets compilation and runtime parameters.

```
config {
    precision: "float32";
    weights: "./weights/mnist.npz";
    target: "generic";
    align: 64;
    batch: 1;
    preprocess: "normalize_0_1";
    io: "stdio";
}
```

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `precision` | String | No | `"float32"` | Tensor data type. `"float32"`, `"float64"`, `"int8"`. |
| `weights` | String | **Yes** | — | Path to weights: directory of `.npy` files, `.npz` archive, or `.onnx` file. |
| `target` | String | No | `"generic"` | SIMD optimization target. `"generic"`, `"avx2"`, `"avx512"`, `"arm_neon"`. |
| `align` | Number | No | `64` | Memory alignment in bytes for weight and workspace buffers. |
| `batch` | Number | No | `1` | Inference batch size. Determines static buffer dimensions. |
| `preprocess` | String | No | `"none"` | Input preprocessing. `"none"`, `"normalize_0_1"`, `"standardize"`. |
| `preprocess_mean` | Shape | No | — | Per-channel mean for `"standardize"` (e.g., `[0.485, 0.456, 0.406]`). |
| `preprocess_std` | Shape | No | — | Per-channel std for `"standardize"` (e.g., `[0.229, 0.224, 0.225]`). |
| `io` | String | No | `"stdio"` | I/O mode for `--emit exe` binaries. Currently only `"stdio"`. |

**Preprocessing modes:**

- `"normalize_0_1"` — divides each input element by 255.0.
- `"standardize"` — applies `(x - mean) / std` per channel; requires `preprocess_mean` and `preprocess_std`.

---

## Layer Types

Every layer is declared as:

```
layer <id> = <LayerType>(<param>: <value>, ...);
```

The layer `<id>` is used for connections and for matching weight tensors (weights are looked up as `{id}.{param_name}` in the weight source).

---

### Input

Entry point of the network. Defines the input tensor shape (excluding batch dimension).

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `shape` | Shape | Yes | — |

**Output shape:** the declared shape.

```
layer input = Input(shape: [28, 28, 1]);
```

---

### Dense

Fully connected layer: Y = activation(W·X + B).

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `units` | Integer | Yes | — |
| `activation` | String | No | `"none"` |

`activation` accepts `"none"`, `"relu"`, `"sigmoid"`, `"softmax"`.

**Weight files:** `{id}.weight` (shape: input_dim × units), `{id}.bias` (shape: units).

**Output shape:** `[units]`.

```
layer fc1 = Dense(units: 128, activation: "relu");
```

---

### Conv2D

2D spatial convolution.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `filters` | Integer | Yes | — |
| `kernel` | Integer or Shape | Yes | — |
| `stride` | Integer | No | `1` |
| `padding` | String | No | `"valid"` |

`padding` accepts `"valid"` (no padding) or `"same"` (zero-pad to preserve spatial dims).

**Weight files:** `{id}.weight` (shape: filters × in_channels × kH × kW), `{id}.bias` (shape: filters).

**Output shape (HWC):**
- `"valid"`: `[⌊(H - kH) / stride⌋ + 1, ⌊(W - kW) / stride⌋ + 1, filters]`
- `"same"`: `[⌈H / stride⌉, ⌈W / stride⌉, filters]`

```
layer conv1 = Conv2D(filters: 32, kernel: 3, stride: 1, padding: "valid");
```

---

### MaxPool2D

Spatial max pooling.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `kernel` | Integer or Shape | Yes | — |
| `stride` | Integer | No | kernel size |

**Weight files:** none.

**Output shape:** `[⌊(H - kH) / stride⌋ + 1, ⌊(W - kW) / stride⌋ + 1, C]`

```
layer pool1 = MaxPool2D(kernel: 2);
```

---

### AvgPool2D

Spatial average pooling.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `kernel` | Integer or Shape | Yes | — |
| `stride` | Integer | No | kernel size |

**Weight files:** none.

**Output shape:** same formula as MaxPool2D.

```
layer pool1 = AvgPool2D(kernel: 2, stride: 2);
```

---

### Flatten

Reshapes a multi-dimensional tensor into a 1D vector.

No parameters.

**Weight files:** none.

**Output shape:** `[H × W × C]` (product of all input dimensions).

```
layer flat = Flatten();
```

---

### BatchNorm

Batch normalization (inference mode — uses stored running statistics).

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `epsilon` | Number | No | `1e-5` |

**Weight files:** `{id}.gamma`, `{id}.beta`, `{id}.running_mean`, `{id}.running_var` (each shape: channels).

**Output shape:** same as input.

```
layer bn1 = BatchNorm();
layer bn2 = BatchNorm(epsilon: 1e-6);
```

---

### Dropout

Identity pass-through at inference time. Exists so that models exported from training frameworks can be represented without editing.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `rate` | Number | No | `0.5` |

The `rate` parameter is ignored during compilation.

**Weight files:** none.

**Output shape:** same as input.

```
layer drop = Dropout(rate: 0.25);
```

---

### Add

Element-wise addition of two or more inputs. Requires explicit connections.

No parameters.

**Constraint:** all inputs must have identical shapes.

**Weight files:** none.

**Output shape:** same as each input.

```
layer res = Add();
```

---

### Concat

Channel-wise concatenation of two or more inputs. Requires explicit connections.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `axis` | Integer | No | `-1` |

**Constraint:** all inputs must have identical shapes except along the concatenation axis.

**Weight files:** none.

**Output shape:** input shape with dimension along `axis` summed across inputs.

```
layer merged = Concat();
layer merged = Concat(axis: -1);
```

---

### ReLU

Standalone activation: `max(0, x)`.

No parameters. No weight files.

**Output shape:** same as input.

```
layer relu1 = ReLU();
```

---

### Sigmoid

Standalone activation: `1 / (1 + exp(-x))`.

No parameters. No weight files.

**Output shape:** same as input.

```
layer sig = Sigmoid();
```

---

### Softmax

Normalized exponential activation.

| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `axis` | Integer | No | `-1` |

No weight files.

**Output shape:** same as input.

```
layer sm = Softmax();
```

---

## Connections

### Implicit Sequential

When the `connections` block is omitted, layers are connected in declaration order — each layer receives the output of the previous layer. This is the simplest form and works for linear stacks:

```
model simple {
    config { weights: "./weights"; io: "stdio"; }

    layer input  = Input(shape: [4]);
    layer fc1    = Dense(units: 8, activation: "relu");
    layer output = Dense(units: 2);
}
// Equivalent to: input -> fc1 -> output
```

### Explicit Graph

When a `connections` block is present, it **fully** defines the data flow. Use this for skip connections, branches, and multi-input layers.

```
connections {
    input -> conv1;
    conv1 -> bn1;
    bn1   -> relu1;
    relu1 -> output;
}
```

### Multi-Input Syntax

Layers like `Add` and `Concat` accept multiple inputs using bracket syntax:

```
[input, bn2] -> res;   // feeds both 'input' and 'bn2' into 'res'
```

---

## Complete Examples

### Simple MLP

A minimal multi-layer perceptron:

```
version 0.2;

model mlp {
    config {
        weights: "./weights";
        io: "stdio";
    }

    layer input  = Input(shape: [4]);
    layer fc1    = Dense(units: 16, activation: "relu");
    layer fc2    = Dense(units: 8, activation: "relu");
    layer output = Dense(units: 3, activation: "softmax");
}
```

### CNN with Pooling

An MNIST digit classifier with convolution and pooling:

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

### ResNet Block with Skip Connections

A residual block using explicit connections and `Add`:

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
        bn1   -> relu1;
        relu1 -> conv2;
        conv2 -> bn2;
        [input, bn2] -> res;   // skip connection
        res   -> relu2;
    }
}
```
