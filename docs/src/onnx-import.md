# ONNX Import

## Overview

`nnc import` converts ONNX models to NNL format with extracted weights.

```sh
nnc import model.onnx -o model.nnl
```

## Supported ONNX Operators

| ONNX Op | NNL Layer |
|---------|-----------|
| Gemm / MatMul | Dense |
| Conv | Conv2D |
| MaxPool | MaxPool2D |
| AveragePool | AvgPool2D |
| Flatten | Flatten |
| BatchNormalization | BatchNorm |
| Dropout | Dropout |
| Add | Add |
| Concat | Concat |
| Relu | ReLU |
| Sigmoid | Sigmoid |
| Softmax | Softmax |

## Weight Handling

- Weights are extracted from ONNX initializers and saved as individual `.npy` files.
- `Gemm` nodes with `transB=1` have their weights automatically transposed to the NNL `[in, out]` layout.
- The batch dimension is stripped from input shapes.

## Unsupported Operators

Operators without a mapping are emitted as comments in the generated `.nnl` file:

```
// UNSUPPORTED: Reshape(reshape_0)
```

These require manual resolution — replace the comment with an equivalent NNL layer or restructure the model before export.

## Round-Trip Workflow

1. **Train** your model in PyTorch, TensorFlow, or another framework.
2. **Export** to ONNX (e.g., `torch.onnx.export(model, dummy, "model.onnx")`).
3. **Import** into NNL: `nnc import model.onnx -o model.nnl`
4. **Compile** and run: `nnc compile model.nnl -o model && ./model`
5. **Test** outputs against the original framework to verify correctness.

## Limitations

- Only `float32` weights are supported.
- External data is not supported — weights must be embedded in the `.onnx` file.
- Dynamic shapes are not supported; all dimensions must be fixed at export time.
