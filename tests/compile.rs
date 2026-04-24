use std::path::Path;
use std::process::Command;

use npyz::WriterBuilder;

/// Write a float32 .npy file with given shape and data.
fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) {
    let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape_u64)
        .writer(file)
        .begin_nd()
        .unwrap();
    writer.extend(data.iter().copied()).unwrap();
    writer.finish().unwrap();
}

fn nnc() -> assert_cmd::Command {
    assert_cmd::Command::cargo_bin("nnc").unwrap()
}

/// Create a set of weight files for a mini CNN test.
/// Model: Input [4,4,1] -> Conv2D(filters:1, kernel:3, valid) -> Flatten -> Dense(1)
/// Conv2D output shape: [2,2,1], flatten: [4], dense: [1]
fn create_mini_cnn_weights(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    // conv.weight: [filters=1, in_ch=1, kh=3, kw=3] = 9 elements, all 1.0
    write_npy_f32(&dir.join("conv.weight.npy"), &[1, 1, 3, 3], &[1.0; 9]);
    // conv.bias: [1] = 0.0
    write_npy_f32(&dir.join("conv.bias.npy"), &[1], &[0.0]);
    // fc.weight: [4, 1] = all 1.0
    write_npy_f32(&dir.join("fc.weight.npy"), &[4, 1], &[1.0; 4]);
    // fc.bias: [1] = 0.0
    write_npy_f32(&dir.join("fc.bias.npy"), &[1], &[0.0]);
}

#[test]
fn compile_and_run_mlp_exe() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // MLP: input [4] -> fc1(units: 3, relu) -> fc2(units: 2, softmax)
    // fc1.weight: all 0.5, shape [4, 3]
    write_npy_f32(&weights_dir.join("fc1.weight.npy"), &[4, 3], &[0.5_f32; 12]);
    // fc1.bias: [0.1, 0.2, 0.3]
    write_npy_f32(&weights_dir.join("fc1.bias.npy"), &[3], &[0.1, 0.2, 0.3]);
    // fc2.weight: all 0.25, shape [3, 2]
    write_npy_f32(&weights_dir.join("fc2.weight.npy"), &[3, 2], &[0.25_f32; 6]);
    // fc2.bias: [0.0, 0.0]
    write_npy_f32(&weights_dir.join("fc2.bias.npy"), &[2], &[0.0, 0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"version 0.2;
model test_mlp {{
    config {{
        weights: "{}";
        io: "stdio";
    }}
    layer input = Input(shape: [4]);
    layer fc1   = Dense(units: 3, activation: "relu");
    layer fc2   = Dense(units: 2, activation: "softmax");
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("test_mlp");

    // Compile
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(exe_path.exists(), "compiled binary should exist");

    // Run inference: input = [1.0, 2.0, 3.0, 4.0] as raw float32 bytes
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run compiled binary");

    assert!(
        output.status.success(),
        "inference binary exited with error: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout.len(),
        2 * 4,
        "expected 2 float32 values (8 bytes)"
    );

    // Parse output
    let out_f32: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();

    // Verify softmax output sums to ~1.0
    let sum: f32 = out_f32.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax output should sum to 1.0, got {sum} (values: {out_f32:?})"
    );

    // Since both fc2 outputs use the same weights, they should be equal -> softmax = [0.5, 0.5]
    // (fc1 output: 0.5*(1+2+3+4)+bias = [5.1, 5.2, 5.3] all > 0, relu passes through)
    // (fc2: 0.25*(5.1+5.2+5.3) = 3.9 for both outputs, softmax = [0.5, 0.5])
    assert!(
        (out_f32[0] - 0.5).abs() < 1e-4,
        "expected ~0.5, got {}",
        out_f32[0]
    );
    assert!(
        (out_f32[1] - 0.5).abs() < 1e-4,
        "expected ~0.5, got {}",
        out_f32[1]
    );
}

#[test]
fn compile_emit_header() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[2, 3], &[1.0; 6]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[3], &[0.0; 3]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"model simple {{ config {{ weights: "{}"; }} layer input = Input(shape: [2]); layer fc = Dense(units: 3); }}"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let hdr_path = tmp.path().join("simple.h");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "header",
            "-o",
            hdr_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    let header = std::fs::read_to_string(&hdr_path).unwrap();
    assert!(header.contains("simple_infer"));
    assert!(header.contains("simple_input_size"));
    assert!(header.contains("simple_output_size"));
}

#[test]
fn compile_emit_obj() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[2, 3], &[1.0; 6]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[3], &[0.0; 3]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"model simple {{ config {{ weights: "{}"; }} layer input = Input(shape: [2]); layer fc = Dense(units: 3); }}"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let obj_path = tmp.path().join("simple.o");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "obj",
            "-o",
            obj_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(obj_path.exists(), "object file should exist");
}

#[test]
fn compile_and_run_cnn_exe() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    create_mini_cnn_weights(&weights_dir);

    let model_path = tmp.path().join("cnn.nnl");
    let model_src = format!(
        r#"version 0.2;
model mini_cnn {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input   = Input(shape: [4, 4, 1]);
    layer conv    = Conv2D(filters: 1, kernel: 3, stride: 1, padding: "valid");
    layer flatten = Flatten();
    layer fc      = Dense(units: 1);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("mini_cnn");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input: 4x4x1 = 16 floats, all 1.0
    // Conv2D(kernel=3, valid): each output pixel = sum of 3x3 window of 1s * weight 1s = 9.0
    // Output shape [2,2,1] = 4 values, all 9.0
    // Flatten: [4] = [9, 9, 9, 9]
    // Dense(1): weight all 1.0, bias 0 => 9*4 = 36.0
    let input_bytes: Vec<u8> = [1.0_f32; 16].iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run cnn binary");

    assert!(
        output.status.success(),
        "cnn binary failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    assert!((result - 36.0).abs() < 1e-3, "expected ~36.0, got {result}");
}

#[test]
fn nnc_test_pass() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // Simple: Input [2] -> Dense(1), weight=[1,1], bias=[0]
    // input [3, 1] → output [4]
    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[2, 1], &[1.0, 1.0]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"model sum {{ config {{ weights: "{}"; }} layer input = Input(shape: [2]); layer fc = Dense(units: 1); }}"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    // Write input.npy: [3.0, 1.0]
    let input_path = tmp.path().join("input.npy");
    write_npy_f32(&input_path, &[2], &[3.0, 1.0]);

    // Write expected.npy: [4.0]
    let expected_path = tmp.path().join("expected.npy");
    write_npy_f32(&expected_path, &[1], &[4.0]);

    nnc()
        .args([
            "test",
            model_path.to_str().unwrap(),
            "--input",
            input_path.to_str().unwrap(),
            "--expected",
            expected_path.to_str().unwrap(),
            "--tolerance",
            "1e-5",
        ])
        .assert()
        .success()
        .stderr(predicates::prelude::predicate::str::contains("PASS"));
}

#[test]
fn nnc_test_fail() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[2, 1], &[1.0, 1.0]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"model sum {{ config {{ weights: "{}"; }} layer input = Input(shape: [2]); layer fc = Dense(units: 1); }}"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let input_path = tmp.path().join("input.npy");
    write_npy_f32(&input_path, &[2], &[3.0, 1.0]);

    // Wrong expected: 999.0 instead of 4.0
    let expected_path = tmp.path().join("expected.npy");
    write_npy_f32(&expected_path, &[1], &[999.0]);

    nnc()
        .args([
            "test",
            model_path.to_str().unwrap(),
            "--input",
            input_path.to_str().unwrap(),
            "--expected",
            expected_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicates::prelude::predicate::str::contains("FAIL"));
}

#[test]
fn compile_residual_block() {
    // Test that explicit graph with skip connections compiles and links
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    let ch: usize = 2;
    let h: usize = 4;
    let w: usize = 4;

    // Two Conv2D layers with same padding, both [2,2,3,3]
    write_npy_f32(
        &weights_dir.join("conv1.weight.npy"),
        &[ch, ch, 3, 3],
        &vec![0.0; ch * ch * 9],
    );
    write_npy_f32(&weights_dir.join("conv1.bias.npy"), &[ch], &vec![0.0; ch]);
    write_npy_f32(
        &weights_dir.join("conv2.weight.npy"),
        &[ch, ch, 3, 3],
        &vec![0.0; ch * ch * 9],
    );
    write_npy_f32(&weights_dir.join("conv2.bias.npy"), &[ch], &vec![0.0; ch]);
    // BatchNorm params
    write_npy_f32(&weights_dir.join("bn1.gamma.npy"), &[ch], &vec![1.0; ch]);
    write_npy_f32(&weights_dir.join("bn1.beta.npy"), &[ch], &vec![0.0; ch]);
    write_npy_f32(
        &weights_dir.join("bn1.running_mean.npy"),
        &[ch],
        &vec![0.0; ch],
    );
    write_npy_f32(
        &weights_dir.join("bn1.running_var.npy"),
        &[ch],
        &vec![1.0; ch],
    );
    write_npy_f32(&weights_dir.join("bn2.gamma.npy"), &[ch], &vec![1.0; ch]);
    write_npy_f32(&weights_dir.join("bn2.beta.npy"), &[ch], &vec![0.0; ch]);
    write_npy_f32(
        &weights_dir.join("bn2.running_mean.npy"),
        &[ch],
        &vec![0.0; ch],
    );
    write_npy_f32(
        &weights_dir.join("bn2.running_var.npy"),
        &[ch],
        &vec![1.0; ch],
    );

    let model_path = tmp.path().join("resblock.nnl");
    let model_src = format!(
        r#"version 0.2;
model resblock {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input = Input(shape: [{h}, {w}, {ch}]);
    layer conv1 = Conv2D(filters: {ch}, kernel: 3, stride: 1, padding: "same");
    layer bn1   = BatchNorm();
    layer relu1 = ReLU();
    layer conv2 = Conv2D(filters: {ch}, kernel: 3, stride: 1, padding: "same");
    layer bn2   = BatchNorm();
    layer res   = Add();
    layer relu2 = ReLU();
    connections {{
        input -> conv1;
        conv1 -> bn1;
        bn1 -> relu1;
        relu1 -> conv2;
        conv2 -> bn2;
        [input, bn2] -> res;
        res -> relu2;
    }}
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("resblock");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Run with input = all 1.0
    // Conv weights are 0 → conv output = bias = 0
    // BN(gamma=1, beta=0, mean=0, var=1): (0-0)/sqrt(1+1e-5)*1+0 ≈ 0
    // relu(0) = 0, conv2 same → 0, bn2 → 0
    // Add: input(1) + bn2(0) = 1, relu(1) = 1
    // Output should be all 1.0
    let input_size = h * w * ch;
    let input_bytes: Vec<u8> = vec![1.0_f32; input_size]
        .iter()
        .flat_map(|v| v.to_ne_bytes())
        .collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run resblock binary");

    assert!(
        output.status.success(),
        "resblock failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );

    let out_floats: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect();

    assert_eq!(out_floats.len(), input_size);
    for (i, &v) in out_floats.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-4, "output[{i}] = {v}, expected ~1.0");
    }
}

#[test]
fn compile_and_run_concat_channel_axis() {
    // Regression test: Concat along channel axis (axis=-1) on 3D HWC tensors.
    // Previously used flat memcpy which only worked for 1D concat.
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // Model: Input [4,4,1] -> two Conv2D(filters:2, kernel:1, same) -> Concat(axis:-1) -> Flatten -> Dense(1)
    // Each conv output: [4,4,2], concat -> [4,4,4], flatten -> [64], dense -> [1]
    let ic = 1;
    let f = 2;

    // conv1: weights [2,1,1,1], all 1.0; bias [2] = [0, 0]
    write_npy_f32(
        &weights_dir.join("conv1.weight.npy"),
        &[f, ic, 1, 1],
        &vec![1.0; f * ic],
    );
    write_npy_f32(&weights_dir.join("conv1.bias.npy"), &[f], &[0.0; 2]);

    // conv2: weights [2,1,1,1], all 2.0; bias [2] = [0, 0]
    write_npy_f32(
        &weights_dir.join("conv2.weight.npy"),
        &[f, ic, 1, 1],
        &vec![2.0; f * ic],
    );
    write_npy_f32(&weights_dir.join("conv2.bias.npy"), &[f], &[0.0; 2]);

    // fc: weights [64, 1], all 1.0; bias [1] = [0]
    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[64, 1], &vec![1.0; 64]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("concat_test.nnl");
    let model_src = format!(
        r#"version 0.2;
model concat_test {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input   = Input(shape: [4, 4, 1]);
    layer conv1   = Conv2D(filters: 2, kernel: 1, stride: 1, padding: "same");
    layer conv2   = Conv2D(filters: 2, kernel: 1, stride: 1, padding: "same");
    layer cat     = Concat(axis: 2);
    layer flatten = Flatten();
    layer fc      = Dense(units: 1);
    connections {{
        input -> conv1;
        input -> conv2;
        [conv1, conv2] -> cat;
        cat -> flatten;
        flatten -> fc;
    }}
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("concat_test");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input: 4x4x1 = 16 floats, all 1.0
    // conv1(w=1.0): each pixel -> [1.0, 1.0] -> [4,4,2]
    // conv2(w=2.0): each pixel -> [2.0, 2.0] -> [4,4,2]
    // Concat(axis=-1): [4,4,4] -> each pixel has [1.0, 1.0, 2.0, 2.0]
    // Flatten: [64] -> 16 pixels * 4 channels = 16 * (1+1+2+2) = 96
    // Dense(1): sum all = 96.0
    let input_bytes: Vec<u8> = [1.0_f32; 16].iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run concat_test binary");

    assert!(
        output.status.success(),
        "concat_test binary failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    assert!(
        (result - 96.0).abs() < 1e-3,
        "expected ~96.0 for channel-axis concat, got {result}"
    );
}

#[test]
fn compile_and_run_conv2d_rectangular_kernel() {
    // Regression test: rectangular kernels (kh != kw) previously produced wrong
    // weight indices because the C loop variable `kh` shadowed the constant.
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // Model: Input [6,6,1] -> Conv2D(filters:1, kernel:[3,5], stride:1, valid) -> Flatten -> Dense(1)
    // Conv2D output shape with valid padding: [(6-3)/1+1, (6-5)/1+1, 1] = [4, 2, 1]
    // Flatten: [8], Dense: [1]
    let kh = 3;
    let kw = 5;
    let ic = 1;
    let filters = 1;

    // conv weights: all 1.0, shape [filters, ic, kh, kw] = [1,1,3,5]
    write_npy_f32(
        &weights_dir.join("conv.weight.npy"),
        &[filters, ic, kh, kw],
        &vec![1.0; filters * ic * kh * kw],
    );
    write_npy_f32(&weights_dir.join("conv.bias.npy"), &[filters], &[0.0]);

    let flat_size = 4 * 2 * filters; // 8
    write_npy_f32(
        &weights_dir.join("fc.weight.npy"),
        &[flat_size, 1],
        &vec![1.0; flat_size],
    );
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("rect_cnn.nnl");
    let model_src = format!(
        r#"version 0.2;
model rect_cnn {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input   = Input(shape: [6, 6, 1]);
    layer conv    = Conv2D(filters: 1, kernel: [3, 5], stride: 1, padding: "valid");
    layer flatten = Flatten();
    layer fc      = Dense(units: 1);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("rect_cnn");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input: 6x6x1 = 36 floats, all 1.0
    // Conv2D(kernel=[3,5], valid): each output = sum of 3x5=15 ones * weight 1 = 15.0
    // Output [4,2,1] = 8 values, all 15.0
    // Flatten: [8] = [15, 15, ..., 15]
    // Dense(1): weight all 1.0, bias 0 => 15*8 = 120.0
    let input_bytes: Vec<u8> = [1.0_f32; 36].iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run rect_cnn binary");

    assert!(
        output.status.success(),
        "rect_cnn binary failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    assert!(
        (result - 120.0).abs() < 1e-3,
        "expected ~120.0 for rectangular kernel [3,5], got {result}"
    );
}

#[test]
fn compile_and_run_hardswish() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // MLP: input [4] -> hardswish -> fc(units: 1)
    // fc.weight: all 1.0, bias 0 → sums the hardswish outputs
    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[4, 1], &[1.0; 4]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"version 0.2;
model hardswish_test {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input = Input(shape: [4]);
    layer hs    = Hardswish();
    layer fc    = Dense(units: 1);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("hardswish_test");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input: [-4.0, -1.0, 0.0, 5.0]
    // Hardswish(x) = x * min(max(0, x+3), 6) / 6
    //   -4.0: -4 * min(max(0,-1),6)/6 = -4 * 0/6 = 0
    //   -1.0: -1 * min(max(0,2),6)/6 = -1 * 2/6 = -0.3333..
    //    0.0: 0 * min(max(0,3),6)/6 = 0
    //    5.0: 5 * min(max(0,8),6)/6 = 5 * 6/6 = 5
    // Sum = 0 + (-0.3333) + 0 + 5 = 4.6666..
    let input: Vec<f32> = vec![-4.0, -1.0, 0.0, 5.0];
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run hardswish_test binary");

    assert!(
        output.status.success(),
        "hardswish_test failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    let expected = 4.6666666;
    assert!(
        (result - expected).abs() < 1e-3,
        "expected ~{expected}, got {result}"
    );
}

#[test]
fn compile_and_run_upsample() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // Model: Input [2,2,1] -> Upsample(scale:2) -> Flatten -> Dense(1)
    // Upsample output: [4,4,1] = 16 elements (nearest neighbor)
    // Flatten: [16], Dense(1): sum all
    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[16, 1], &[1.0; 16]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"version 0.2;
model upsample_test {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input   = Input(shape: [2, 2, 1]);
    layer up      = Upsample(scale: 2);
    layer flatten = Flatten();
    layer fc      = Dense(units: 1);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("upsample_test");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input [2,2,1]: [[1, 2], [3, 4]] in HWC layout
    // Upsample 2x nearest: each pixel repeated 2x2:
    // [[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]]
    // Sum = 4*(1+2+3+4) = 40
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run upsample_test binary");

    assert!(
        output.status.success(),
        "upsample_test failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    assert!(
        (result - 40.0).abs() < 1e-3,
        "expected ~40.0 for 2x upsample, got {result}"
    );
}

#[test]
fn compile_and_run_conv1d() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // Model: Input [8,1] -> Conv1D(filters:1, kernel:3, valid) -> Flatten -> Dense(1)
    // Conv1D output: [(8-3)/1+1, 1] = [6, 1]
    // Flatten: [6], Dense(1): sum all

    // conv.weight: [filters=1, in_ch=1, kernel=3] = 3 elements, all 1.0
    write_npy_f32(&weights_dir.join("conv.weight.npy"), &[1, 1, 3], &[1.0; 3]);
    write_npy_f32(&weights_dir.join("conv.bias.npy"), &[1], &[0.0]);
    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[6, 1], &[1.0; 6]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"version 0.2;
model conv1d_test {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input   = Input(shape: [8, 1]);
    layer conv    = Conv1D(filters: 1, kernel: 3, stride: 1, padding: "valid");
    layer flatten = Flatten();
    layer fc      = Dense(units: 1);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("conv1d_test");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input: [1,1,1,1,1,1,1,1] (8 values, all 1.0)
    // Conv1D(kernel=3, valid): each output = sum of 3 ones = 3.0
    // Output [6,1] = [3,3,3,3,3,3]
    // Dense(1): sum = 18.0
    let input: Vec<f32> = vec![1.0; 8];
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run conv1d_test binary");

    assert!(
        output.status.success(),
        "conv1d_test failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    assert!(
        (result - 18.0).abs() < 1e-3,
        "expected ~18.0 for Conv1D, got {result}"
    );
}

#[test]
fn compile_and_run_maxpool1d() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // Model: Input [6,1] -> MaxPool1D(kernel:2) -> Flatten -> Dense(1)
    // MaxPool1D output: [(6-2)/2+1, 1] = [3, 1]
    // Flatten: [3], Dense(1): sum all
    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[3, 1], &[1.0; 3]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[1], &[0.0]);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"version 0.2;
model maxpool1d_test {{
    config {{ weights: "{}"; io: "stdio"; }}
    layer input   = Input(shape: [6, 1]);
    layer pool    = MaxPool1D(kernel: 2);
    layer flatten = Flatten();
    layer fc      = Dense(units: 1);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    let exe_path = tmp.path().join("maxpool1d_test");
    nnc()
        .args([
            "compile",
            model_path.to_str().unwrap(),
            "--emit",
            "exe",
            "-o",
            exe_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Input: [1, 4, 2, 5, 3, 6] (6 values)
    // MaxPool1D(kernel=2, stride=2): max([1,4])=4, max([2,5])=5, max([3,6])=6
    // Dense(1): sum = 15.0
    let input: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();

    let output = Command::new(&exe_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(&input_bytes)?;
            child.wait_with_output()
        })
        .expect("failed to run maxpool1d_test binary");

    assert!(
        output.status.success(),
        "maxpool1d_test failed: {:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout.len(), 4, "expected 1 float32 (4 bytes)");

    let result = f32::from_ne_bytes(output.stdout[..4].try_into().unwrap());
    assert!(
        (result - 15.0).abs() < 1e-3,
        "expected ~15.0 for MaxPool1D, got {result}"
    );
}
