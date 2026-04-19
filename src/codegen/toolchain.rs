use std::path::Path;
use std::process::Command;

use crate::cli::EmitFormat;
use crate::ir::model::Target;

#[derive(Debug)]
pub struct ToolchainError {
    pub message: String,
}

impl std::fmt::Display for ToolchainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Options for the toolchain invocation.
pub struct CompileOptions<'a> {
    pub target: Target,
    pub target_triple: Option<&'a str>,
}

/// Compile generated C source (and header) into the requested artifact.
pub fn compile(
    c_source: &str,
    c_header: &str,
    emit: &EmitFormat,
    output: &Path,
    model_name: &str,
    opts: &CompileOptions<'_>,
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

    // Build base flags: -O2 + target-specific SIMD flags
    let mut flags: Vec<&str> = vec!["-O2"];
    let target_flag = target_cc_flag(opts.target);
    if let Some(f) = target_flag {
        flags.push(f);
    }

    // Determine compiler binary (cross-compilation support)
    let cc_bin = if let Some(triple) = opts.target_triple {
        format!("{triple}-gcc")
    } else {
        "cc".to_string()
    };

    let src_str = src_path.display().to_string();
    let out_str = output.display().to_string();

    match emit {
        EmitFormat::Exe => {
            let mut args = flags.clone();
            args.extend_from_slice(&["-o", &out_str, &src_str, "-lm"]);
            cc_cmd(&cc_bin, &args)?;
        }
        EmitFormat::Obj => {
            let mut args = flags.clone();
            args.extend_from_slice(&["-c", "-o", &out_str, &src_str]);
            cc_cmd(&cc_bin, &args)?;
            copy_header_beside(&hdr_path, output)?;
        }
        EmitFormat::Lib => {
            let tmp_obj = tmp_dir.path().join(format!("{model_name}.o"));
            let obj_str = tmp_obj.display().to_string();
            let mut args = flags.clone();
            args.extend_from_slice(&["-c", "-o", &obj_str, &src_str]);
            cc_cmd(&cc_bin, &args)?;
            ar(&[&out_str, &obj_str])?;
            copy_header_beside(&hdr_path, output)?;
        }
        EmitFormat::Shared => {
            let mut args = flags.clone();
            args.extend_from_slice(&["-shared", "-fPIC", "-o", &out_str, &src_str, "-lm"]);
            cc_cmd(&cc_bin, &args)?;
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

/// Return the -m flag for the given target, or None for generic.
fn target_cc_flag(target: Target) -> Option<&'static str> {
    match target {
        Target::Generic => None,
        Target::Avx2 => Some("-mavx2"),
        Target::Avx512 => Some("-mavx512f"),
        Target::ArmNeon => Some("-mfpu=neon"),
    }
}

fn cc_cmd(bin: &str, args: &[&str]) -> Result<(), ToolchainError> {
    let output = Command::new(bin)
        .args(args)
        .output()
        .map_err(|e| ToolchainError {
            message: format!("failed to invoke {bin}: {e}"),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ToolchainError {
            message: format!("{bin} failed (exit {}):\n{stderr}", output.status),
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
