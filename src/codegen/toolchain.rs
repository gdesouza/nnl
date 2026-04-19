use std::path::Path;
use std::process::Command;

use crate::cli::EmitFormat;

#[derive(Debug)]
pub struct ToolchainError {
    pub message: String,
}

impl std::fmt::Display for ToolchainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Compile generated C source (and header) into the requested artifact.
pub fn compile(
    c_source: &str,
    c_header: &str,
    emit: &EmitFormat,
    output: &Path,
    model_name: &str,
) -> Result<(), ToolchainError> {
    let tmp_dir = tempfile::tempdir().map_err(|e| ToolchainError {
        message: format!("failed to create temp directory: {e}"),
    })?;

    let src_path = tmp_dir.path().join(format!("{model_name}.c"));
    let hdr_path = tmp_dir.path().join(format!("{model_name}.h"));

    std::fs::write(&src_path, c_source).map_err(|e| ToolchainError {
        message: format!("failed to write C source: {e}"),
    })?;
    std::fs::write(&hdr_path, c_header).map_err(|e| ToolchainError {
        message: format!("failed to write C header: {e}"),
    })?;

    match emit {
        EmitFormat::Exe => {
            cc(&["-O2", "-o", &output.display().to_string(), &src_path.display().to_string(), "-lm"])?;
        }
        EmitFormat::Obj => {
            cc(&["-O2", "-c", "-o", &output.display().to_string(), &src_path.display().to_string()])?;
            copy_header_beside(&hdr_path, output)?;
        }
        EmitFormat::Lib => {
            let tmp_obj = tmp_dir.path().join(format!("{model_name}.o"));
            cc(&["-O2", "-c", "-o", &tmp_obj.display().to_string(), &src_path.display().to_string()])?;
            ar(&[&output.display().to_string(), &tmp_obj.display().to_string()])?;
            copy_header_beside(&hdr_path, output)?;
        }
        EmitFormat::Shared => {
            cc(&["-O2", "-shared", "-fPIC", "-o", &output.display().to_string(), &src_path.display().to_string(), "-lm"])?;
            copy_header_beside(&hdr_path, output)?;
        }
        EmitFormat::Header => {
            std::fs::write(output, c_header).map_err(|e| ToolchainError {
                message: format!("failed to write header: {e}"),
            })?;
        }
    }

    Ok(())
}

fn cc(args: &[&str]) -> Result<(), ToolchainError> {
    let output = Command::new("cc")
        .args(args)
        .output()
        .map_err(|e| ToolchainError {
            message: format!("failed to invoke cc: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ToolchainError {
            message: format!("cc failed (exit {}):\n{stderr}", output.status),
        });
    }
    Ok(())
}

fn ar(args: &[&str]) -> Result<(), ToolchainError> {
    let mut full_args = vec!["rcs"];
    full_args.extend_from_slice(args);

    let output = Command::new("ar")
        .args(&full_args)
        .output()
        .map_err(|e| ToolchainError {
            message: format!("failed to invoke ar: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ToolchainError {
            message: format!("ar failed (exit {}):\n{stderr}", output.status),
        });
    }
    Ok(())
}

fn copy_header_beside(hdr_path: &Path, output: &Path) -> Result<(), ToolchainError> {
    if let Some(parent) = output.parent() {
        let dest = parent.join(
            hdr_path
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("model.h")),
        );
        std::fs::copy(hdr_path, &dest).map_err(|e| ToolchainError {
            message: format!("failed to copy header to {}: {e}", dest.display()),
        })?;
    }
    Ok(())
}
