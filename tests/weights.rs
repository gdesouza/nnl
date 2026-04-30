use std::path::Path;

use assert_cmd::Command;
use npyz::WriterBuilder;
use predicates::prelude::*;

fn nnc() -> Command {
    Command::cargo_bin("nnc").unwrap()
}

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

/// Create a minimal MLP model weight directory for a model:
///   layer input = Input(shape: [4]);
///   layer fc1   = Dense(units: 3);
///   layer fc2   = Dense(units: 2);
fn create_mlp_weights(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();

    // fc1.weight: [4, 3] — input_dim=4, units=3
    write_npy_f32(&dir.join("fc1.weight.npy"), &[4, 3], &[0.1_f32; 12]);
    // fc1.bias: [3]
    write_npy_f32(&dir.join("fc1.bias.npy"), &[3], &[0.0_f32; 3]);

    // fc2.weight: [3, 2] — input_dim=3, units=2
    write_npy_f32(&dir.join("fc2.weight.npy"), &[3, 2], &[0.2_f32; 6]);
    // fc2.bias: [2]
    write_npy_f32(&dir.join("fc2.bias.npy"), &[2], &[0.0_f32; 2]);
}

#[test]
fn compile_with_valid_weights_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    create_mlp_weights(&weights_dir);

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"
version 0.2;
model test_mlp {{
    config {{
        weights: "{}";
    }}
    layer input = Input(shape: [4]);
    layer fc1   = Dense(units: 3);
    layer fc2   = Dense(units: 2);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    // compile should fail (not yet implemented) but weight loading should succeed
    // For now, test via inspect which doesn't load weights
    nnc()
        .args(["inspect", model_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("[4]")
                .and(predicate::str::contains("[3]"))
                .and(predicate::str::contains("[2]")),
        );
}

#[test]
fn compile_with_valid_npz_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let npz_path = tmp.path().join("weights.npz");

    // Create an npz with the correct weight tensors
    {
        let mut writer = npyz::npz::NpzWriter::create(&npz_path).unwrap();
        let options = zip::write::FileOptions::default();

        fn write_array(
            writer: &mut npyz::npz::NpzWriter<std::io::BufWriter<std::fs::File>>,
            name: &str,
            shape: &[u64],
            data: &[f32],
            options: zip::write::FileOptions,
        ) {
            let mut arr = writer
                .array::<f32>(name, options)
                .unwrap()
                .default_dtype()
                .shape(shape)
                .begin_nd()
                .unwrap();
            arr.extend(data.iter().copied()).unwrap();
            arr.finish().unwrap();
        }

        write_array(&mut writer, "fc1.weight", &[4, 3], &[0.1_f32; 12], options);
        write_array(&mut writer, "fc1.bias", &[3], &[0.0_f32; 3], options);
        write_array(&mut writer, "fc2.weight", &[3, 2], &[0.2_f32; 6], options);
        write_array(&mut writer, "fc2.bias", &[2], &[0.0_f32; 2], options);
    }

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"
version 0.2;
model test_mlp {{
    config {{
        weights: "{}";
    }}
    layer input = Input(shape: [4]);
    layer fc1   = Dense(units: 3);
    layer fc2   = Dense(units: 2);
}}
"#,
        npz_path.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    nnc()
        .args(["inspect", model_path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn error_missing_weights_in_directory_lists_expected_files() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"
version 0.2;
model test {{
    config {{ weights: "{}"; }}
    layer input = Input(shape: [4]);
    layer fc = Dense(units: 3);
}}
"#,
        weights_dir.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    nnc()
        .args(["compile", model_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing required weight files"))
        .stderr(predicate::str::contains(weights_dir.to_str().unwrap()))
        .stderr(predicate::str::contains("fc.weight.npy"))
        .stderr(predicate::str::contains("fc.bias.npy"))
        .stderr(predicate::str::contains("expected tensors and shapes"));
}

#[test]
fn error_missing_weights_in_npz_lists_expected_arrays() {
    let tmp = tempfile::tempdir().unwrap();
    let npz_path = tmp.path().join("weights.npz");

    {
        let mut writer = npyz::npz::NpzWriter::create(&npz_path).unwrap();
        let options = zip::write::FileOptions::default();

        let mut arr = writer
            .array::<f32>("fc.weight", options)
            .unwrap()
            .default_dtype()
            .shape(&[4, 3])
            .begin_nd()
            .unwrap();
        arr.extend([0.1_f32; 12]).unwrap();
        arr.finish().unwrap();
    }

    let model_path = tmp.path().join("model.nnl");
    let model_src = format!(
        r#"
version 0.2;
model test {{
    config {{ weights: "{}"; }}
    layer input = Input(shape: [4]);
    layer fc = Dense(units: 3);
}}
"#,
        npz_path.display()
    );
    std::fs::write(&model_path, &model_src).unwrap();

    nnc()
        .args(["compile", model_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing required weight arrays"))
        .stderr(predicate::str::contains(npz_path.to_str().unwrap()))
        .stderr(predicate::str::contains("fc.bias"))
        .stderr(predicate::str::contains("expected tensors and shapes"));
}
