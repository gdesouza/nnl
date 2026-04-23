use logos::Logos;
use std::fmt;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/")]
pub enum Token {
    // Keywords
    #[token("version")]
    Version,
    #[token("model")]
    Model,
    #[token("config")]
    Config,
    #[token("layer")]
    Layer,
    #[token("connections")]
    Connections,
    #[token("true")]
    True,
    #[token("false")]
    False,

    // Layer type keywords
    #[token("Input")]
    Input,
    #[token("Dense")]
    Dense,
    #[token("Conv2D")]
    Conv2D,
    #[token("MaxPool2D")]
    MaxPool2D,
    #[token("AvgPool2D")]
    AvgPool2D,
    #[token("Flatten")]
    Flatten,
    #[token("BatchNorm")]
    BatchNorm,
    #[token("Dropout")]
    Dropout,
    #[token("Add")]
    Add,
    #[token("Concat")]
    Concat,
    #[token("ReLU")]
    ReLU,
    #[token("Sigmoid")]
    Sigmoid,
    #[token("Softmax")]
    Softmax,
    #[token("GlobalAvgPool2D")]
    GlobalAvgPool2D,
    #[token("ReLU6")]
    ReLU6,
    #[token("LeakyReLU")]
    LeakyReLU,
    #[token("SiLU")]
    SiLU,
    #[token("Mul")]
    Mul,

    // Literals
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<u64>().ok())]
    Integer(u64),
    #[regex(r#""[^"]*""#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    String(String),

    // Identifiers (must come after keywords so keywords take priority)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,

    // Punctuation
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token(";")]
    Semicolon,
    #[token(":")]
    Colon,
    #[token("=")]
    Equals,
    #[token("->")]
    Arrow,
    #[token(",")]
    Comma,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Version => write!(f, "version"),
            Token::Model => write!(f, "model"),
            Token::Config => write!(f, "config"),
            Token::Layer => write!(f, "layer"),
            Token::Connections => write!(f, "connections"),
            Token::True => write!(f, "true"),
            Token::False => write!(f, "false"),
            Token::Input => write!(f, "Input"),
            Token::Dense => write!(f, "Dense"),
            Token::Conv2D => write!(f, "Conv2D"),
            Token::MaxPool2D => write!(f, "MaxPool2D"),
            Token::AvgPool2D => write!(f, "AvgPool2D"),
            Token::Flatten => write!(f, "Flatten"),
            Token::BatchNorm => write!(f, "BatchNorm"),
            Token::Dropout => write!(f, "Dropout"),
            Token::Add => write!(f, "Add"),
            Token::Concat => write!(f, "Concat"),
            Token::ReLU => write!(f, "ReLU"),
            Token::Sigmoid => write!(f, "Sigmoid"),
            Token::Softmax => write!(f, "Softmax"),
            Token::GlobalAvgPool2D => write!(f, "GlobalAvgPool2D"),
            Token::ReLU6 => write!(f, "ReLU6"),
            Token::LeakyReLU => write!(f, "LeakyReLU"),
            Token::SiLU => write!(f, "SiLU"),
            Token::Mul => write!(f, "Mul"),
            Token::Float(v) => write!(f, "{v}"),
            Token::Integer(v) => write!(f, "{v}"),
            Token::String(v) => write!(f, "\"{v}\""),
            Token::Ident => write!(f, "identifier"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::Semicolon => write!(f, ";"),
            Token::Colon => write!(f, ":"),
            Token::Equals => write!(f, "="),
            Token::Arrow => write!(f, "->"),
            Token::Comma => write!(f, ","),
        }
    }
}

/// A token with its span in the source.
#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub span: Span,
}

/// Byte offset range in the source text.
pub type Span = std::ops::Range<usize>;

/// Tokenize source text into a vector of spanned tokens.
pub fn tokenize(source: &str) -> Result<Vec<Spanned>, LexError> {
    let mut lexer = Token::lexer(source);
    let mut tokens = Vec::new();
    while let Some(result) = lexer.next() {
        match result {
            Ok(token) => {
                tokens.push(Spanned {
                    token,
                    span: lexer.span(),
                });
            }
            Err(()) => {
                return Err(LexError { span: lexer.span() });
            }
        }
    }
    Ok(tokens)
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub span: Span,
}
