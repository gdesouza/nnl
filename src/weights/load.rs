use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use npyz::NpyFile;
use npyz::{TypeChar, TypeStr};

use crate::ir::model::Precision;
use crate::weights::tensor::WeightTensor;
use crate::weights::WeightError;

/// Load weights from a path (directory of .npy files, or .npz archive).
pub fn load_weights(
    path: &str,
    precision: Precision,
) -> Result<HashMap<String, WeightTensor>, WeightError> {
    let p = Path::new(path);

    if p.is_dir() {
        load_from_directory(p, precision)
    } else if path.ends_with(".npz") {
        load_from_npz(p, precision)
    } else if path.ends_with(".npy") {
        Err(WeightError {
            code: "E003",
            message: format!(
                "bare .npy file `{path}` is not supported as weight source; use a directory or .npz archive"
            ),
        })
    } else {
        Err(WeightError {
            code: "E003",
            message: format!("unsupported weight source `{path}`; expected directory or .npz file"),
        })
    }
}

/// Load from a directory: each file is `{layer_id}.{param}.npy`.
fn load_from_directory(
    dir: &Path,
    precision: Precision,
) -> Result<HashMap<String, WeightTensor>, WeightError> {
    let mut weights = HashMap::new();

    let entries = std::fs::read_dir(dir).map_err(|e| WeightError {
        code: "E003",
        message: format!("cannot read weight directory `{}`: {e}", dir.display()),
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| WeightError {
            code: "E003",
            message: format!("error reading directory entry: {e}"),
        })?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "npy") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| WeightError {
                    code: "E003",
                    message: format!("invalid filename: {}", path.display()),
                })?
                .to_string();
            let bytes = std::fs::read(&path).map_err(|e| WeightError {
                code: "E003",
                message: format!("cannot read `{}`: {e}", path.display()),
            })?;
            let tensor = load_npy_bytes(&bytes, precision, &stem)?;
            weights.insert(stem, tensor);
        }
    }

    Ok(weights)
}

/// Load from a .npz archive.
fn load_from_npz(
    path: &Path,
    precision: Precision,
) -> Result<HashMap<String, WeightTensor>, WeightError> {
    let mut archive = npyz::npz::NpzArchive::open(path).map_err(|e| WeightError {
        code: "E003",
        message: format!("cannot open npz archive `{}`: {e}", path.display()),
    })?;

    let names: Vec<String> = archive.array_names().map(|s| s.to_string()).collect();
    let mut weights = HashMap::new();

    for name in &names {
        let npy = archive.by_name(name).map_err(|e| WeightError {
            code: "E003",
            message: format!("cannot read array `{name}` from npz: {e}"),
        })?.ok_or_else(|| WeightError {
            code: "E003",
            message: format!("array `{name}` not found in npz archive"),
        })?;

        let tensor = npy_to_tensor(npy, precision, name)?;
        weights.insert(name.clone(), tensor);
    }

    Ok(weights)
}

/// Load a single .npy from bytes already in memory.
fn load_npy_bytes(bytes: &[u8], precision: Precision, name: &str) -> Result<WeightTensor, WeightError> {
    let npy = NpyFile::new(bytes).map_err(|e| WeightError {
        code: "E003",
        message: format!("cannot parse npy data for `{name}`: {e}"),
    })?;
    npy_to_tensor(npy, precision, name)
}

/// Convert an NpyFile into a WeightTensor, casting to the target precision.
fn npy_to_tensor<R: Read>(
    npy: NpyFile<R>,
    precision: Precision,
    name: &str,
) -> Result<WeightTensor, WeightError> {
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let dtype = npy.dtype();

    // Extract type info from dtype
    let type_str = match dtype {
        npyz::DType::Plain(ts) => ts.clone(),
        _ => {
            return Err(WeightError {
                code: "E003",
                message: format!("structured dtypes are not supported for weight `{name}`"),
            });
        }
    };

    match precision {
        Precision::Float32 => {
            let data: Vec<f32> = if is_float32(&type_str) {
                npy.into_vec::<f32>().map_err(|e| WeightError {
                    code: "E003",
                    message: format!("cannot read float32 data from `{name}`: {e}"),
                })?
            } else if is_float64(&type_str) {
                let data_f64: Vec<f64> = npy.into_vec::<f64>().map_err(|e| WeightError {
                    code: "E003",
                    message: format!("cannot read float64 data from `{name}`: {e}"),
                })?;
                eprintln!("warning: casting weight `{name}` from float64 to float32");
                data_f64.iter().map(|&v| v as f32).collect()
            } else {
                return Err(WeightError {
                    code: "E003",
                    message: format!(
                        "unsupported dtype for weight `{name}`; expected float32 or float64",
                    ),
                });
            };

            let byte_data: Vec<u8> = data
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();

            Ok(WeightTensor {
                shape,
                data: byte_data,
                elem_bytes: 4,
            })
        }
        Precision::Float64 => {
            let data: Vec<f64> = npy.into_vec::<f64>().map_err(|e| WeightError {
                code: "E003",
                message: format!("cannot read float64 data from `{name}`: {e}"),
            })?;

            let byte_data: Vec<u8> = data
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();

            Ok(WeightTensor {
                shape,
                data: byte_data,
                elem_bytes: 8,
            })
        }
        Precision::Int8 => {
            Err(WeightError {
                code: "E003",
                message: format!("int8 weight loading is not yet supported for `{name}`"),
            })
        }
    }
}

fn is_float32(ts: &TypeStr) -> bool {
    ts.type_char() == TypeChar::Float && ts.size_field() == 4
}

fn is_float64(ts: &TypeStr) -> bool {
    ts.type_char() == TypeChar::Float && ts.size_field() == 8
}
