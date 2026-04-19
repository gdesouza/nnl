# Weight Files

## Supported Formats

| Format | Description |
|--------|-------------|
| **Directory of `.npy` files** | Each file named `{layer_id}.{param}.npy` (e.g., `fc1.weight.npy`, `fc1.bias.npy`) |
| **`.npz` archive** | Keys must match `{layer_id}.{param}` (e.g., `fc1.weight`, `fc1.bias`) |

## Naming Convention

The `weights` config key points to the weight source. `nnc` resolves it relative to the `.nnl` file's directory.

```toml
[config]
weights = "weights/"       # directory of .npy files
# or
weights = "model.npz"      # single .npz archive
```

## Expected Shapes Per Layer

| Layer | Parameter | Shape |
|-------|-----------|-------|
| Dense | weight | `[input_dim, units]` |
| Dense | bias | `[units]` |
| Conv2D | weight | `[filters, in_channels, kH, kW]` |
| Conv2D | bias | `[filters]` |
| BatchNorm | gamma | `[channels]` |
| BatchNorm | beta | `[channels]` |
| BatchNorm | running_mean | `[channels]` |
| BatchNorm | running_var | `[channels]` |

## Data Types

| Precision | Weight dtype |
|-----------|-------------|
| `"float32"` | `float32` |
| `"float64"` | `float64` |

## Generating Test Weights (Python)

```python
import numpy as np

# Create weights matching a Dense layer with 784 inputs and 128 units
np.save("fc1.weight.npy", np.random.randn(784, 128).astype(np.float32))
np.save("fc1.bias.npy", np.zeros(128, dtype=np.float32))

# Or bundle into an .npz archive
np.savez("model.npz",
    **{"fc1.weight": np.random.randn(784, 128).astype(np.float32),
       "fc1.bias": np.zeros(128, dtype=np.float32)})
```

## Error Messages

| Error | Meaning | Fix |
|-------|---------|-----|
| **E003: missing weight** | A layer expects a weight file or key that was not found in the weight source. | Ensure the weight source contains an entry named `{layer_id}.{param}` for every parameterised layer. |
| **Shape mismatch** | The shape of a loaded weight does not match what the layer definition expects (e.g., expected `[784, 128]` but found `[128, 784]`). | Regenerate or transpose the weight so its shape matches the table above. |
