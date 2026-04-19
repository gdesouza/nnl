use crate::syntax::lexer::Span;

/// A fully typed, validated model ready for semantic analysis.
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub version: Option<f64>,
    pub config: Config,
    pub layers: Vec<Layer>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub precision: Precision,
    pub weights: String,
    pub target: Target,
    pub align: usize,
    pub batch: usize,
    pub preprocess: Preprocess,
    pub preprocess_mean: Option<Vec<f64>>,
    pub preprocess_std: Option<Vec<f64>>,
    #[allow(dead_code)]
    pub io: IoMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Float32,
    Float64,
    Int8,
}

impl Precision {
    pub fn byte_size(&self) -> usize {
        match self {
            Precision::Float32 => 4,
            Precision::Float64 => 8,
            Precision::Int8 => 1,
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::Float32 => write!(f, "float32"),
            Precision::Float64 => write!(f, "float64"),
            Precision::Int8 => write!(f, "int8"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Generic,
    Avx2,
    Avx512,
    ArmNeon,
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Target::Generic => write!(f, "generic"),
            Target::Avx2 => write!(f, "avx2"),
            Target::Avx512 => write!(f, "avx512"),
            Target::ArmNeon => write!(f, "arm_neon"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preprocess {
    None,
    Normalize01,
    Standardize,
}

impl std::fmt::Display for Preprocess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Preprocess::None => write!(f, "none"),
            Preprocess::Normalize01 => write!(f, "normalize_0_1"),
            Preprocess::Standardize => write!(f, "standardize"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    Stdio,
}

impl std::fmt::Display for IoMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoMode::Stdio => write!(f, "stdio"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub id: String,
    pub kind: LayerKind,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum LayerKind {
    Input {
        shape: Vec<usize>,
    },
    Dense {
        units: usize,
        activation: Activation,
    },
    Conv2D {
        filters: usize,
        kernel: KernelSize,
        stride: usize,
        padding: Padding,
    },
    MaxPool2D {
        kernel: KernelSize,
        stride: Option<usize>,
    },
    AvgPool2D {
        kernel: KernelSize,
        stride: Option<usize>,
    },
    Flatten,
    BatchNorm {
        epsilon: f64,
    },
    Dropout {
        #[allow(dead_code)]
        rate: f64,
    },
    Add,
    Concat {
        #[allow(dead_code)]
        axis: i64,
    },
    ReLU,
    Sigmoid,
    Softmax {
        #[allow(dead_code)]
        axis: i64,
    },
}

impl LayerKind {
    pub fn type_name(&self) -> &'static str {
        match self {
            LayerKind::Input { .. } => "Input",
            LayerKind::Dense { .. } => "Dense",
            LayerKind::Conv2D { .. } => "Conv2D",
            LayerKind::MaxPool2D { .. } => "MaxPool2D",
            LayerKind::AvgPool2D { .. } => "AvgPool2D",
            LayerKind::Flatten => "Flatten",
            LayerKind::BatchNorm { .. } => "BatchNorm",
            LayerKind::Dropout { .. } => "Dropout",
            LayerKind::Add => "Add",
            LayerKind::Concat { .. } => "Concat",
            LayerKind::ReLU => "ReLU",
            LayerKind::Sigmoid => "Sigmoid",
            LayerKind::Softmax { .. } => "Softmax",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    ReLU,
    Sigmoid,
    Softmax,
}

#[derive(Debug, Clone)]
pub enum KernelSize {
    Square(usize),
    Rect(usize, usize),
}

impl KernelSize {
    pub fn height(&self) -> usize {
        match self {
            KernelSize::Square(s) => *s,
            KernelSize::Rect(h, _) => *h,
        }
    }

    pub fn width(&self) -> usize {
        match self {
            KernelSize::Square(s) => *s,
            KernelSize::Rect(_, w) => *w,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    Valid,
    Same,
}

/// A directed edge: from source layer to target layer.
#[derive(Debug, Clone)]
pub struct Edge {
    pub source: String,
    pub target: String,
}
