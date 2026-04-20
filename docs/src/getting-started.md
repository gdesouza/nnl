# Getting Started

This guide will help you install and run your first NNLang model.

## Installation

### From crates.io

```bash
cargo install nnlang
```

### From source

```bash
git clone https://github.com/gdesouza/nnl
cd nnl
cargo install --path .
```

### Pre-built binaries

Download the latest release from GitHub:

```bash
# Linux
curl -L https://github.com/gdesouza/nnl/releases/latest/download/nnc-*-x86_64-unknown-linux-gnu.tar.gz | tar xz
sudo mv nnc /usr/local/bin/

# macOS
curl -L https://github.com/gdesouza/nnl/releases/latest/download/nnc-*-x86_64-apple-darwin.tar.gz | tar xz
sudo mv nnc /usr/local/bin/
```

## Quick Start

### 1. Create a model file

Save this as `model.nnl`:

```
version 0.2;

model my_model {
    config {
        weights: "./weights";
        io: "stdio";
    }

    layer input  = Input(shape: [4]);
    layer fc1   = Dense(units: 3, activation: "relu");
    layer fc2   = Dense(units: 2);
}
```

### 2. Create weight files

Create a `weights/` directory with:

- `weights/fc1.weight.npy` — [4, 3] matrix
- `weights/fc1.bias.npy` — [3] vector
- `weights/fc2.weight.npy` — [3, 2] matrix
- `weights/fc2.bias.npy` — [2] vector

### 3. Compile

```bash
nnc compile model.nnl --emit exe -o model
```

### 4. Run inference

```bash
# Input: 4 floats
echo -n -e '\x00\x00\x80\x3f\x00\x00\x00@\x00\x00@@\x00\x00\x80@' > input.bin
./model < input.bin > output.bin
```

Or test with known input/output:

```bash
nnc test model.nnl --input test_input.npy --expected expected_output.npy
```

## Next Steps

- [Language Reference](language-reference.md) — full language syntax
- [CLI Reference](cli.md) — all commands
- [Examples](examples.md) — complete model examples