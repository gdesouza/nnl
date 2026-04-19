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

#[test]
fn compile_and_run_mlp_exe() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    // MLP: input [4] -> fc1(units: 3, relu) -> fc2(units: 2, softmax)
    // fc1.weight: all 0.5, shape [4, 3]
    write_npy_f32(
        &weights_dir.join("fc1.weight.npy"),
        &[4, 3],
        &vec![0.5_f32; 12],
    );
    // fc1.bias: [0.1, 0.2, 0.3]
    write_npy_f32(
        &weights_dir.join("fc1.bias.npy"),
        &[3],
        &[0.1, 0.2, 0.3],
    );
    // fc2.weight: all 0.25, shape [3, 2]
    write_npy_f32(
        &weights_dir.join("fc2.weight.npy"),
        &[3, 2],
        &vec![0.25_f32; 6],
    );
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

    assert!(output.status.success(), "inference binary exited with error: {:?}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.stdout.len(), 2 * 4, "expected 2 float32 values (8 bytes)");

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

    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[2, 3], &vec![1.0; 6]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[3], &vec![0.0; 3]);

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

    write_npy_f32(&weights_dir.join("fc.weight.npy"), &[2, 3], &vec![1.0; 6]);
    write_npy_f32(&weights_dir.join("fc.bias.npy"), &[3], &vec![0.0; 3]);

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
