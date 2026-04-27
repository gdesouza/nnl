#!/usr/bin/env python3
"""Call an NNL-compiled model from Python using ctypes.

Build instructions:
    cd examples/integration
    nnc compile model.nnl --emit shared -o libsimple_mlp.so
    cd python
    python infer.py
"""
import ctypes
import os
import numpy as np

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "..", "libsimple_mlp.so")
lib = ctypes.CDLL(lib_path)

# Set up function signatures
lib.simple_mlp_infer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.simple_mlp_infer.restype = ctypes.c_int

lib.simple_mlp_input_size.argtypes = []
lib.simple_mlp_input_size.restype = ctypes.c_int

lib.simple_mlp_output_size.argtypes = []
lib.simple_mlp_output_size.restype = ctypes.c_int

# Query sizes
input_size = lib.simple_mlp_input_size()
output_size = lib.simple_mlp_output_size()
print(f"Input size:  {input_size}")
print(f"Output size: {output_size}")

# Prepare input/output arrays
input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
output_data = np.zeros(output_size, dtype=np.float32)

# Run inference
rc = lib.simple_mlp_infer(
    input_data.ctypes.data_as(ctypes.c_void_p),
    output_data.ctypes.data_as(ctypes.c_void_p),
)
assert rc == 0, f"inference failed with code {rc}"

print(f"Output: {output_data}")
print(f"Predicted class: {np.argmax(output_data)}")
