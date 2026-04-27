use assert_cmd::Command;
use predicates::prelude::*;

fn nnc() -> Command {
    Command::cargo_bin("nnc").unwrap()
}

#[test]
fn inspect_mnist() {
    nnc()
        .args(["inspect", "examples/mnist/mnist.nnl"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Model: mnist_classifier")
                .and(predicate::str::contains("conv1"))
                .and(predicate::str::contains("[26, 26, 32]"))
                .and(predicate::str::contains("692,352"))
                .and(predicate::str::contains("Total params:"))
                .and(predicate::str::contains("Weight memory:"))
                .and(predicate::str::contains("Workspace:")),
        );
}

#[test]
fn inspect_resnet_block() {
    nnc()
        .args(["inspect", "examples/resnet_block/resnet_block.nnl"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Model: resnet_block")
                .and(predicate::str::contains("[32, 32, 64]"))
                .and(predicate::str::contains("Add"))
                .and(predicate::str::contains("36,928")),
        );
}

#[test]
fn error_missing_weights_config() {
    let source = r#"
version 0.2;
model test {
    config {
        precision: "float32";
    }
    layer input = Input(shape: [1]);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "missing required config key `weights`",
        ));
}

#[test]
fn error_unknown_config_key() {
    let source = r#"
model test {
    config {
        weights: "./w.npz";
        foobar: "baz";
    }
    layer input = Input(shape: [1]);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown config key `foobar`"));
}

#[test]
fn error_invalid_precision() {
    let source = r#"
model test {
    config {
        weights: "./w.npz";
        precision: "float16";
    }
    layer input = Input(shape: [1]);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid precision"));
}

#[test]
fn error_unsupported_precision_int8() {
    let source = r#"
model test {
    config {
        weights: "./w.npz";
        precision: "int8";
    }
    layer input = Input(shape: [1]);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("not yet supported by codegen"));
}

#[test]
fn error_unsupported_precision_float64() {
    let source = r#"
model test {
    config {
        weights: "./w.npz";
        precision: "float64";
    }
    layer input = Input(shape: [1]);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("not yet supported by codegen"));
}

#[test]
fn error_no_input_connection() {
    let source = r#"
model test {
    config { weights: "./w.npz"; }
    layer input = Input(shape: [10]);
    layer fc1 = Dense(units: 5);
    layer fc2 = Dense(units: 3);
    connections {
        input -> fc2;
    }
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("E001").or(predicate::str::contains("no input connection")),
        );
}

#[test]
fn error_shape_mismatch_dense_needs_1d() {
    let source = r#"
model test {
    config { weights: "./w.npz"; }
    layer input = Input(shape: [28, 28, 1]);
    layer fc = Dense(units: 10);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("E002").or(predicate::str::contains("Dense expects 1D")));
}

#[test]
fn warning_no_version() {
    let source = r#"
model test {
    config { weights: "./w.npz"; }
    layer input = Input(shape: [10]);
    layer fc = Dense(units: 5);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .success()
        .stderr(predicate::str::contains("W002"));
}

#[test]
fn error_unsupported_io_mode() {
    let source = r#"
model test {
    config {
        weights: "./w.npz";
        io: "tcp";
    }
    layer input = Input(shape: [1]);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains("unsupported io mode"));
}

#[test]
fn error_io_none_with_emit_exe() {
    let tmp = tempfile::tempdir().unwrap();
    let weights_dir = tmp.path().join("weights");
    std::fs::create_dir_all(&weights_dir).unwrap();

    let model_path = tmp.path().join("model.nnl");
    std::fs::write(
        &model_path,
        format!(
            r#"model test {{ config {{ weights: "{}"; io: "none"; }} layer input = Input(shape: [1]); }}"#,
            weights_dir.display()
        ),
    )
    .unwrap();

    nnc()
        .args(["compile", model_path.to_str().unwrap(), "--emit", "exe"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "cannot emit executable with io: \"none\"",
        ));
}

#[test]
fn error_missing_required_param() {
    let source = r#"
model test {
    config { weights: "./w.npz"; }
    layer input = Input();
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "missing required parameter `shape`",
        ));
}

#[test]
fn implicit_sequential_connections() {
    let source = r#"
model test {
    config { weights: "./w.npz"; }
    layer input = Input(shape: [10]);
    layer fc1 = Dense(units: 5);
    layer fc2 = Dense(units: 3);
}
"#;
    nnc()
        .args(["inspect", "/dev/stdin"])
        .write_stdin(source)
        .assert()
        .success()
        .stdout(
            predicate::str::contains("[10]")
                .and(predicate::str::contains("[5]"))
                .and(predicate::str::contains("[3]")),
        );
}
