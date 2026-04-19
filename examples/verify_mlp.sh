#!/bin/bash
set -e

# ── Hand-verifiable MLP example ──────────────────────────────────
#
# Network:  Input [2] → Dense(2, relu) → Dense(1)
#
# Weights:
#   fc1.weight = [[1, -1], [-1, 1]]   (shape [2,2])
#   fc1.bias   = [0, 0]
#   fc2.weight = [[1], [1]]           (shape [2,1])
#   fc2.bias   = [0]
#
# Input:  [3.0, 1.0]
#
# Hand computation:
#   fc1:    [3*1 + 1*(-1), 3*(-1) + 1*1] = [2.0, -2.0]
#   relu:   [2.0, 0.0]
#   fc2:    2*1 + 0*1 + 0 = 2.0
#   output: [2.0]
#
# Expected output: 2.0 (as raw float32 bytes)
# ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMPDIR=$(mktemp -d)

echo "=== Building nnc ==="
cargo build --quiet --manifest-path "$PROJECT_DIR/Cargo.toml"
NNC="$PROJECT_DIR/target/debug/nnc"

echo "=== Creating weights ==="
# We use a small Rust program to write .npy files
cat > "$TMPDIR/gen_weights.rs" << 'RUSTEOF'
use std::io::Write;
fn write_npy(path: &str, shape: &[u64], data: &[f32]) {
    // Minimal NPY v1.0 writer
    let mut header = String::from("{'descr': '<f4', 'fortran_order': False, 'shape': (");
    for (i, s) in shape.iter().enumerate() {
        if i > 0 { header.push_str(", "); }
        header.push_str(&s.to_string());
    }
    if shape.len() == 1 { header.push(','); }
    header.push_str("), }");
    // Pad header to align to 64 bytes
    let prefix_len = 10 + header.len() + 1; // magic(6) + ver(2) + hdr_len(2) + header + \n
    let padding = (64 - (prefix_len % 64)) % 64;
    for _ in 0..padding { header.push(' '); }
    header.push('\n');
    let hdr_bytes = header.as_bytes();
    let hdr_len = hdr_bytes.len() as u16;

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&[0x93, b'N', b'U', b'M', b'P', b'Y']).unwrap(); // magic
    f.write_all(&[1, 0]).unwrap(); // version 1.0
    f.write_all(&hdr_len.to_le_bytes()).unwrap();
    f.write_all(hdr_bytes).unwrap();
    for v in data { f.write_all(&v.to_le_bytes()).unwrap(); }
}
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    //              input[0]→out[0], input[0]→out[1], input[1]→out[0], input[1]→out[1]
    write_npy(&format!("{dir}/fc1.weight.npy"), &[2, 2], &[1.0, -1.0, -1.0, 1.0]);
    write_npy(&format!("{dir}/fc1.bias.npy"),   &[2],    &[0.0, 0.0]);
    write_npy(&format!("{dir}/fc2.weight.npy"), &[2, 1], &[1.0, 1.0]);
    write_npy(&format!("{dir}/fc2.bias.npy"),   &[1],    &[0.0]);
    eprintln!("Weights written to {dir}");
}
RUSTEOF
rustc "$TMPDIR/gen_weights.rs" -o "$TMPDIR/gen_weights" 2>/dev/null
"$TMPDIR/gen_weights" "$TMPDIR/weights"

echo "=== Creating model ==="
cat > "$TMPDIR/model.nnl" << EOF
version 0.2;
model verify_mlp {
    config {
        weights: "$TMPDIR/weights";
        io: "stdio";
    }
    layer input = Input(shape: [2]);
    layer fc1   = Dense(units: 2, activation: "relu");
    layer fc2   = Dense(units: 1);
}
EOF

echo "=== Inspecting model ==="
$NNC inspect "$TMPDIR/model.nnl"

echo ""
echo "=== Compiling model ==="
$NNC compile "$TMPDIR/model.nnl" --emit exe -o "$TMPDIR/verify_mlp"

echo ""
echo "=== Running inference ==="
echo "Input: [3.0, 1.0]"

# Write input as raw float32 LE bytes: 3.0 = 0x40400000, 1.0 = 0x3f800000
printf '\x00\x00\x40\x40\x00\x00\x80\x3f' | "$TMPDIR/verify_mlp" > "$TMPDIR/output.bin"

echo "Output (hex): $(xxd -p "$TMPDIR/output.bin")"

# 2.0 as float32 LE = 0x40000000 = 00 00 00 40
EXPECTED="00000040"
ACTUAL=$(xxd -p "$TMPDIR/output.bin")

if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "✅ Output matches expected value: 2.0"
else
    echo "❌ Mismatch! Expected: $EXPECTED, Got: $ACTUAL"
    exit 1
fi

echo ""
echo "=== Cleanup ==="
rm -rf "$TMPDIR"
echo "Done."
