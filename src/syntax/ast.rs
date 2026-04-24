use crate::syntax::lexer::Span;

/// Top-level AST node representing a complete NNL file.
#[derive(Debug, Clone)]
pub struct File {
    pub version: Option<Version>,
    pub model: ModelDecl,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Version {
    pub number: f64,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ModelDecl {
    pub name: Ident,
    pub config: ConfigBlock,
    pub layers: Vec<LayerDecl>,
    pub connections: Option<ConnectionBlock>,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConfigBlock {
    pub settings: Vec<Setting>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Setting {
    pub key: Ident,
    pub value: Value,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct LayerDecl {
    pub name: Ident,
    pub layer_type: LayerType,
    pub params: Vec<Param>,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub key: Ident,
    pub value: Value,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConnectionBlock {
    pub connections: Vec<Connection>,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Connection {
    /// Source layer(s). Single-element for `a -> b`, multiple for `[a, b] -> c`.
    pub sources: Vec<Ident>,
    pub target: Ident,
    #[allow(dead_code)]
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Value {
    String(String, Span),
    Integer(u64, Span),
    Float(f64, Span),
    Bool(#[allow(dead_code)] bool, Span),
    Shape(Vec<f64>, Span),
}

impl Value {
    pub fn span(&self) -> &Span {
        match self {
            Value::String(_, s) => s,
            Value::Integer(_, s) => s,
            Value::Float(_, s) => s,
            Value::Bool(_, s) => s,
            Value::Shape(_, s) => s,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Input,
    Dense,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    Flatten,
    BatchNorm,
    Dropout,
    Add,
    Concat,
    ReLU,
    Sigmoid,
    Softmax,
    GlobalAvgPool2D,
    ReLU6,
    LeakyReLU,
    SiLU,
    Mul,
    Hardswish,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerType::Input => write!(f, "Input"),
            LayerType::Dense => write!(f, "Dense"),
            LayerType::Conv2D => write!(f, "Conv2D"),
            LayerType::MaxPool2D => write!(f, "MaxPool2D"),
            LayerType::AvgPool2D => write!(f, "AvgPool2D"),
            LayerType::Flatten => write!(f, "Flatten"),
            LayerType::BatchNorm => write!(f, "BatchNorm"),
            LayerType::Dropout => write!(f, "Dropout"),
            LayerType::Add => write!(f, "Add"),
            LayerType::Concat => write!(f, "Concat"),
            LayerType::ReLU => write!(f, "ReLU"),
            LayerType::Sigmoid => write!(f, "Sigmoid"),
            LayerType::Softmax => write!(f, "Softmax"),
            LayerType::GlobalAvgPool2D => write!(f, "GlobalAvgPool2D"),
            LayerType::ReLU6 => write!(f, "ReLU6"),
            LayerType::LeakyReLU => write!(f, "LeakyReLU"),
            LayerType::SiLU => write!(f, "SiLU"),
            LayerType::Mul => write!(f, "Mul"),
            LayerType::Hardswish => write!(f, "Hardswish"),
        }
    }
}
