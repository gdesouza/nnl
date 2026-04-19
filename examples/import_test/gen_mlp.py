#!/usr/bin/env python3
"""Generate a simple ONNX MLP model for round-trip import testing.

Architecture: [4] -> Dense(3, relu) -> Dense(2, none)

Uses fixed weights so we can verify the output exactly.
Also produces input.npy and expected.npy for nnc test.
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Fixed weights
w1 = np.array([
    [ 0.1,  0.2, -0.1],
    [-0.3,  0.4,  0.5],
    [ 0.2, -0.1,  0.3],
    [ 0.1,  0.3, -0.2],
], dtype=np.float32)  # [4, 3]

b1 = np.array([0.1, -0.1, 0.05], dtype=np.float32)  # [3]

w2 = np.array([
    [ 0.5, -0.3],
    [-0.2,  0.4],
    [ 0.1,  0.6],
], dtype=np.float32)  # [3, 2]

b2 = np.array([0.0, 0.1], dtype=np.float32)  # [2]

# Build ONNX graph
# Layer 1: Gemm (matmul + bias) -> relu
gemm1 = helper.make_node("Gemm", ["input", "w1", "b1"], ["gemm1_out"],
                          name="dense1", transB=1)  # transB=1: w1 is [out, in]
relu1 = helper.make_node("Relu", ["gemm1_out"], ["relu1_out"], name="relu1")

# Layer 2: Gemm
gemm2 = helper.make_node("Gemm", ["relu1_out", "w2", "b2"], ["output"],
                          name="dense2", transB=1)

# Initializers (weights embedded in graph)
# Note: for Gemm with transB=1, weight shape is [out_features, in_features]
w1_init = numpy_helper.from_array(w1.T, name="w1")  # [3, 4] for transB=1
b1_init = numpy_helper.from_array(b1, name="b1")
w2_init = numpy_helper.from_array(w2.T, name="w2")  # [2, 3] for transB=1
b2_init = numpy_helper.from_array(b2, name="b2")

# Input/output value info
input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

graph = helper.make_graph(
    [gemm1, relu1, gemm2],
    "test_mlp",
    [input_vi],
    [output_vi],
    initializer=[w1_init, b1_init, w2_init, b2_init],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 7
onnx.checker.check_model(model)
onnx.save(model, "model.onnx")
print("Saved model.onnx")

# Generate test input and compute expected output manually
x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

# Layer 1: y = x @ w1 + b1, then relu
h = x @ w1 + b1
h = np.maximum(h, 0)
print(f"After dense1+relu: {h}")

# Layer 2: y = h @ w2 + b2
y = h @ w2 + b2
print(f"Output: {y}")

np.save("input.npy", x.flatten())
np.save("expected.npy", y.flatten())
print("Saved input.npy, expected.npy")
