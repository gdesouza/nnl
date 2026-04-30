use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;

fn nnc() -> Command {
    Command::cargo_bin("nnc").unwrap()
}

#[test]
fn create_rust_project() {
    let tmp = tempfile::tempdir().unwrap();
    let project_dir = tmp.path().join("demo_rust");

    nnc()
        .current_dir(tmp.path())
        .args(["new", "demo_rust", "--project", "rust"])
        .assert()
        .success();

    assert_exists(&project_dir.join("model.nnl"));
    assert_exists(&project_dir.join("Cargo.toml"));
    assert_exists(&project_dir.join("build.rs"));
    assert_exists(&project_dir.join("src/main.rs"));

    let build_rs = std::fs::read_to_string(project_dir.join("build.rs")).unwrap();
    assert!(build_rs.contains("nnc"));
    assert!(build_rs.contains("demo_rust_model"));
}

#[test]
fn create_go_cpp_and_python_projects() {
    let tmp = tempfile::tempdir().unwrap();

    nnc()
        .current_dir(tmp.path())
        .args(["new", "demo_go", "--project", "go"])
        .assert()
        .success();
    nnc()
        .current_dir(tmp.path())
        .args(["new", "demo_cpp", "--project", "cpp"])
        .assert()
        .success();
    nnc()
        .current_dir(tmp.path())
        .args(["new", "demo_python", "--project", "python"])
        .assert()
        .success();

    assert_exists(&tmp.path().join("demo_go/go.mod"));
    assert_exists(&tmp.path().join("demo_go/build.sh"));
    assert_exists(&tmp.path().join("demo_cpp/Makefile"));
    assert_exists(&tmp.path().join("demo_cpp/main.cpp"));
    assert_exists(&tmp.path().join("demo_python/infer.py"));
    assert_exists(&tmp.path().join("demo_python/build.sh"));

    let go_main = std::fs::read_to_string(tmp.path().join("demo_go/main.go")).unwrap();
    assert!(go_main.contains("demo_go_model"));

    let cpp_main = std::fs::read_to_string(tmp.path().join("demo_cpp/main.cpp")).unwrap();
    assert!(cpp_main.contains("demo_cpp_model"));

    let py_main = std::fs::read_to_string(tmp.path().join("demo_python/infer.py")).unwrap();
    assert!(py_main.contains("demo_python_model"));
}

#[test]
fn error_when_destination_exists() {
    let tmp = tempfile::tempdir().unwrap();
    let project_dir = tmp.path().join("demo");
    std::fs::create_dir_all(&project_dir).unwrap();

    nnc()
        .current_dir(tmp.path())
        .args(["new", "demo", "--project", "rust"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("already exists"));
}

fn assert_exists(path: &Path) {
    assert!(path.exists(), "expected {} to exist", path.display());
}
