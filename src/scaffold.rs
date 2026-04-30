use std::path::{Path, PathBuf};

use crate::cli::ProjectKind;

pub fn create_project(dir: &Path, kind: &ProjectKind) -> Result<(), String> {
    if dir.exists() {
        return Err(format!("destination `{}` already exists", dir.display()));
    }

    let project_name = dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("invalid project path `{}`", dir.display()))?;
    let ident = normalize_ident(project_name);
    let model_name = format!("{}_model", ident);

    std::fs::create_dir_all(dir)
        .map_err(|e| format!("failed to create `{}`: {e}", dir.display()))?;

    write_files(
        dir,
        &common_files(&model_name),
        "failed to write scaffold files",
    )?;

    let language_files = match kind {
        ProjectKind::Rust => rust_files(project_name, &model_name),
        ProjectKind::Go => go_files(project_name, &model_name),
        ProjectKind::Cpp => cpp_files(project_name, &model_name),
        ProjectKind::Python => python_files(project_name, &model_name),
    };
    write_files(dir, &language_files, "failed to write language scaffold")?;

    Ok(())
}

fn write_files(dir: &Path, files: &[(PathBuf, String)], error_prefix: &str) -> Result<(), String> {
    for (relative_path, content) in files {
        let full_path = dir.join(relative_path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!("{error_prefix}: cannot create `{}`: {e}", parent.display())
            })?;
        }
        std::fs::write(&full_path, content).map_err(|e| {
            format!(
                "{error_prefix}: cannot write `{}`: {e}",
                full_path.display()
            )
        })?;
    }
    Ok(())
}

fn common_files(model_name: &str) -> Vec<(PathBuf, String)> {
    vec![
        (
            PathBuf::from("model.nnl"),
            format!(
                "version 0.2;\n\nmodel {model_name} {{\n    config {{\n        io: \"none\";\n    }}\n\n    layer input = Input(shape: [4]);\n    layer output = Softmax();\n}}\n"
            ),
        ),
        (
            PathBuf::from("README.md"),
            format!(
                "# Starter Project\n\nThis project shows how to call an NNL-compiled model from a host application.\n\nThe generated model is `model.nnl` with model name `{model_name}` and `io: \"none\"`, so it can be compiled into a library artifact without a `main()` wrapper.\n"
            ),
        ),
    ]
}

fn rust_files(project_name: &str, model_name: &str) -> Vec<(PathBuf, String)> {
    vec![
        (
            PathBuf::from("Cargo.toml"),
            format!(
                "[package]\nname = \"{project_name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n"
            ),
        ),
        (
            PathBuf::from("build.rs"),
            format!(
                "use std::process::Command;\n\nfn main() {{\n    let out_dir = std::env::var(\"OUT_DIR\").unwrap();\n\n    println!(\"cargo:rerun-if-changed=model.nnl\");\n\n    let status = Command::new(\"nnc\")\n        .args([\"compile\", \"model.nnl\", \"--emit\", \"lib\", \"-o\"])\n        .arg(format!(\"{{}}/lib{model_name}.a\", out_dir))\n        .status()\n        .expect(\"failed to invoke nnc\");\n\n    assert!(status.success(), \"nnc compile failed with {{status}}\");\n\n    println!(\"cargo:rustc-link-search=native={{}}\", out_dir);\n    println!(\"cargo:rustc-link-lib=static={model_name}\");\n    println!(\"cargo:rustc-link-lib=dylib=m\");\n}}\n"
            ),
        ),
        (
            PathBuf::from("src/main.rs"),
            format!(
                "unsafe extern \"C\" {{\n    fn {model_name}_infer(input: *const f32, output: *mut f32) -> i32;\n    fn {model_name}_input_size() -> i32;\n    fn {model_name}_output_size() -> i32;\n}}\n\nfn main() {{\n    let input_size = unsafe {{ {model_name}_input_size() }} as usize;\n    let output_size = unsafe {{ {model_name}_output_size() }} as usize;\n\n    let input = vec![1.0f32, 2.0, 3.0, 4.0];\n    let mut output = vec![0.0f32; output_size];\n\n    let rc = unsafe {{ {model_name}_infer(input.as_ptr(), output.as_mut_ptr()) }};\n    assert_eq!(rc, 0, \"inference failed with code {{rc}}\");\n\n    println!(\"input size: {{input_size}}\");\n    println!(\"output size: {{output_size}}\");\n    println!(\"output: {{:?}}\", output);\n}}\n"
            ),
        ),
        (
            PathBuf::from("README.md"),
            format!(
                "# {project_name}\n\nRust starter for an NNL-compiled model.\n\n## Run\n\n```sh\ncargo run\n```\n\n`build.rs` invokes `nnc compile model.nnl --emit lib` automatically and links the generated static library into the Rust binary.\n"
            ),
        ),
    ]
}

fn go_files(project_name: &str, model_name: &str) -> Vec<(PathBuf, String)> {
    vec![
        (
            PathBuf::from("go.mod"),
            format!("module {project_name}\n\ngo 1.21\n"),
        ),
        (
            PathBuf::from("main.go"),
            format!(
                "package main\n\n/*\n#cgo CFLAGS: -I.\n#cgo LDFLAGS: -L. -l{model_name} -lm\n#include \"{model_name}.h\"\n*/\nimport \"C\"\nimport (\n\t\"fmt\"\n\t\"unsafe\"\n)\n\nfunc main() {{\n\tinputSize := int(C.{model_name}_input_size())\n\toutputSize := int(C.{model_name}_output_size())\n\tinput := [4]C.float{{1.0, 2.0, 3.0, 4.0}}\n\toutput := make([]C.float, outputSize)\n\n\trc := C.{model_name}_infer(unsafe.Pointer(&input[0]), unsafe.Pointer(&output[0]))\n\tif rc != 0 {{\n\t\tpanic(fmt.Sprintf(\"inference failed with code %d\", rc))\n\t}}\n\n\tfmt.Printf(\"input size: %d\\n\", inputSize)\n\tfmt.Printf(\"output size: %d\\n\", outputSize)\n\tfmt.Printf(\"output: %v\\n\", output)\n}}\n"
            ),
        ),
        (
            PathBuf::from("build.sh"),
            format!(
                "#!/usr/bin/env bash\nset -euo pipefail\n\nnnc compile model.nnl --emit lib -o lib{model_name}.a\nnnc compile model.nnl --emit header -o {model_name}.h\ngo run .\n"
            ),
        ),
        (
            PathBuf::from("README.md"),
            format!(
                "# {project_name}\n\nGo starter for an NNL-compiled model.\n\n## Run\n\n```sh\nbash build.sh\n```\n"
            ),
        ),
    ]
}

fn cpp_files(project_name: &str, model_name: &str) -> Vec<(PathBuf, String)> {
    vec![
        (
            PathBuf::from("main.cpp"),
            format!(
                "#include \"{model_name}.h\"\n\n#include <cstdio>\n\nint main() {{\n    float input[4] = {{1.0f, 2.0f, 3.0f, 4.0f}};\n    float output[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};\n\n    int rc = {model_name}_infer(input, output);\n    if (rc != 0) {{\n        std::printf(\"inference failed with code %d\\n\", rc);\n        return rc;\n    }}\n\n    std::printf(\"output: %f %f %f %f\\n\", output[0], output[1], output[2], output[3]);\n    return 0;\n}}\n"
            ),
        ),
        (
            PathBuf::from("Makefile"),
            format!(
                "CXX = g++\nCXXFLAGS = -std=c++17 -Wall -Wextra -O2\nLDFLAGS = -L. -l{model_name} -lm\n\nall: demo\n\nlib{model_name}.a {model_name}.h: model.nnl\n\tnnc compile model.nnl --emit lib -o lib{model_name}.a\n\tnnc compile model.nnl --emit header -o {model_name}.h\n\ndemo: main.cpp lib{model_name}.a {model_name}.h\n\t$(CXX) $(CXXFLAGS) -I. -o $@ main.cpp $(LDFLAGS)\n\nrun: demo\n\t./demo\n\nclean:\n\trm -f demo lib{model_name}.a {model_name}.h\n\n.PHONY: all run clean\n"
            ),
        ),
        (
            PathBuf::from("README.md"),
            format!(
                "# {project_name}\n\nC++ starter for an NNL-compiled model.\n\n## Run\n\n```sh\nmake run\n```\n"
            ),
        ),
    ]
}

fn python_files(project_name: &str, model_name: &str) -> Vec<(PathBuf, String)> {
    vec![
        (
            PathBuf::from("infer.py"),
            format!(
                "#!/usr/bin/env python3\nimport ctypes\nimport os\n\nlib_path = os.path.join(os.path.dirname(__file__), \"lib{model_name}.so\")\nlib = ctypes.CDLL(lib_path)\n\nlib.{model_name}_infer.argtypes = [ctypes.c_void_p, ctypes.c_void_p]\nlib.{model_name}_infer.restype = ctypes.c_int\nlib.{model_name}_output_size.argtypes = []\nlib.{model_name}_output_size.restype = ctypes.c_int\n\ninput_data = (ctypes.c_float * 4)(1.0, 2.0, 3.0, 4.0)\noutput_size = lib.{model_name}_output_size()\noutput_data = (ctypes.c_float * output_size)()\n\nrc = lib.{model_name}_infer(input_data, output_data)\nif rc != 0:\n    raise RuntimeError(f\"inference failed with code {{rc}}\")\n\nprint(\"output:\", [output_data[i] for i in range(output_size)])\n"
            ),
        ),
        (
            PathBuf::from("build.sh"),
            format!(
                "#!/usr/bin/env bash\nset -euo pipefail\n\nnnc compile model.nnl --emit shared -o lib{model_name}.so\npython3 infer.py\n"
            ),
        ),
        (
            PathBuf::from("README.md"),
            format!(
                "# {project_name}\n\nPython starter for an NNL-compiled model.\n\n## Run\n\n```sh\nbash build.sh\n```\n"
            ),
        ),
    ]
}

fn normalize_ident(input: &str) -> String {
    let mut normalized = String::new();

    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            normalized.push(ch.to_ascii_lowercase());
        } else if !normalized.ends_with('_') {
            normalized.push('_');
        }
    }

    let trimmed = normalized.trim_matches('_');
    if trimmed.is_empty() {
        "nnl_app".to_string()
    } else {
        trimmed.to_string()
    }
}
