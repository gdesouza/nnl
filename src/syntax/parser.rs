use crate::syntax::ast::*;
use crate::syntax::lexer::{Span, Spanned, Token};

pub fn parse(tokens: &[Spanned], source: &str) -> Result<File, ParseError> {
    let mut p = Parser::new(tokens, source);
    p.parse_file()
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

struct Parser<'a> {
    tokens: &'a [Spanned],
    source: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Spanned], source: &'a str) -> Self {
        Self {
            tokens,
            source,
            pos: 0,
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|s| &s.token)
    }

    fn peek_spanned(&self) -> Option<&Spanned> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<&Spanned> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<&Spanned, ParseError> {
        match self.peek_spanned() {
            Some(spanned) if &spanned.token == expected => {
                let s = &self.tokens[self.pos];
                self.pos += 1;
                Ok(s)
            }
            Some(spanned) => Err(ParseError {
                message: format!("expected `{expected}`, got `{}`", spanned.token),
                span: spanned.span.clone(),
            }),
            None => Err(self.eof_error(&format!("expected `{expected}`"))),
        }
    }

    fn current_span(&self) -> Span {
        match self.tokens.get(self.pos) {
            Some(s) => s.span.clone(),
            None => {
                let end = self.source.len();
                end..end
            }
        }
    }

    fn eof_error(&self, msg: &str) -> ParseError {
        let end = self.source.len();
        ParseError {
            message: format!("{msg}, got end of file"),
            span: end..end,
        }
    }

    fn parse_file(&mut self) -> Result<File, ParseError> {
        let version = if self.peek() == Some(&Token::Version) {
            Some(self.parse_version()?)
        } else {
            None
        };
        let model = self.parse_model()?;
        Ok(File { version, model })
    }

    fn parse_version(&mut self) -> Result<Version, ParseError> {
        let start = self.current_span().start;
        self.expect(&Token::Version)?;
        let number = self.parse_number_value()?;
        let end = self.current_span().start;
        self.expect(&Token::Semicolon)?;
        Ok(Version {
            number,
            span: start..end,
        })
    }

    fn parse_number_value(&mut self) -> Result<f64, ParseError> {
        match self.peek() {
            Some(Token::Float(_)) => {
                let spanned = self.advance().unwrap();
                if let Token::Float(v) = &spanned.token {
                    Ok(*v)
                } else {
                    unreachable!()
                }
            }
            Some(Token::Integer(_)) => {
                let spanned = self.advance().unwrap();
                if let Token::Integer(v) = &spanned.token {
                    Ok(*v as f64)
                } else {
                    unreachable!()
                }
            }
            Some(other) => Err(ParseError {
                message: format!("expected number, got `{other}`"),
                span: self.current_span(),
            }),
            None => Err(self.eof_error("expected number")),
        }
    }

    fn parse_model(&mut self) -> Result<ModelDecl, ParseError> {
        let start = self.current_span().start;
        self.expect(&Token::Model)?;
        let name = self.parse_ident()?;
        self.expect(&Token::LBrace)?;
        let config = self.parse_config_block()?;
        let layers = self.parse_layer_list()?;
        let connections = if self.peek() == Some(&Token::Connections) {
            Some(self.parse_connection_block()?)
        } else {
            None
        };
        let end_tok = self.expect(&Token::RBrace)?;
        let end = end_tok.span.end;
        Ok(ModelDecl {
            name,
            config,
            layers,
            connections,
            span: start..end,
        })
    }

    fn parse_ident(&mut self) -> Result<Ident, ParseError> {
        match self.peek() {
            Some(Token::Ident) => {
                let span = self.tokens[self.pos].span.clone();
                let name = self.source[span.clone()].to_string();
                self.pos += 1;
                Ok(Ident { name, span })
            }
            Some(other) => Err(ParseError {
                message: format!("expected identifier, got `{other}`"),
                span: self.current_span(),
            }),
            None => Err(self.eof_error("expected identifier")),
        }
    }

    fn parse_config_block(&mut self) -> Result<ConfigBlock, ParseError> {
        let start = self.current_span().start;
        self.expect(&Token::Config)?;
        self.expect(&Token::LBrace)?;
        let mut settings = Vec::new();
        while self.peek() != Some(&Token::RBrace) {
            settings.push(self.parse_setting()?);
        }
        let end_tok = self.expect(&Token::RBrace)?;
        Ok(ConfigBlock {
            settings,
            span: start..end_tok.span.end,
        })
    }

    fn parse_setting(&mut self) -> Result<Setting, ParseError> {
        let key = self.parse_ident()?;
        let start = key.span.start;
        self.expect(&Token::Colon)?;
        let value = self.parse_value()?;
        let end_tok = self.expect(&Token::Semicolon)?;
        Ok(Setting {
            key,
            value,
            span: start..end_tok.span.end,
        })
    }

    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(Token::String(_)) => {
                let spanned = self.advance().unwrap();
                if let Token::String(s) = &spanned.token {
                    Ok(Value::String(s.clone(), spanned.span.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Token::Float(_)) => {
                let spanned = self.advance().unwrap();
                if let Token::Float(v) = &spanned.token {
                    Ok(Value::Float(*v, spanned.span.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Token::Integer(_)) => {
                let spanned = self.advance().unwrap();
                if let Token::Integer(v) = &spanned.token {
                    Ok(Value::Integer(*v, spanned.span.clone()))
                } else {
                    unreachable!()
                }
            }
            Some(Token::True) => {
                let spanned = self.advance().unwrap();
                Ok(Value::Bool(true, spanned.span.clone()))
            }
            Some(Token::False) => {
                let spanned = self.advance().unwrap();
                Ok(Value::Bool(false, spanned.span.clone()))
            }
            Some(Token::LBracket) => self.parse_shape(),
            Some(other) => Err(ParseError {
                message: format!("expected value, got `{other}`"),
                span: self.current_span(),
            }),
            None => Err(self.eof_error("expected value")),
        }
    }

    fn parse_shape(&mut self) -> Result<Value, ParseError> {
        let start_tok = self.expect(&Token::LBracket)?;
        let start = start_tok.span.start;
        let mut numbers = Vec::new();
        if self.peek() != Some(&Token::RBracket) {
            numbers.push(self.parse_number_value()?);
            while self.peek() == Some(&Token::Comma) {
                self.advance();
                numbers.push(self.parse_number_value()?);
            }
        }
        let end_tok = self.expect(&Token::RBracket)?;
        Ok(Value::Shape(numbers, start..end_tok.span.end))
    }

    fn parse_layer_list(&mut self) -> Result<Vec<LayerDecl>, ParseError> {
        let mut layers = Vec::new();
        while self.peek() == Some(&Token::Layer) {
            layers.push(self.parse_layer()?);
        }
        if layers.is_empty() {
            return Err(ParseError {
                message: "expected at least one layer declaration".to_string(),
                span: self.current_span(),
            });
        }
        Ok(layers)
    }

    fn parse_layer(&mut self) -> Result<LayerDecl, ParseError> {
        let start = self.current_span().start;
        self.expect(&Token::Layer)?;
        let name = self.parse_ident()?;
        self.expect(&Token::Equals)?;
        let layer_type = self.parse_layer_type()?;
        self.expect(&Token::LParen)?;
        let params = if self.peek() != Some(&Token::RParen) {
            self.parse_param_list()?
        } else {
            Vec::new()
        };
        self.expect(&Token::RParen)?;
        let end_tok = self.expect(&Token::Semicolon)?;
        Ok(LayerDecl {
            name,
            layer_type,
            params,
            span: start..end_tok.span.end,
        })
    }

    fn parse_layer_type(&mut self) -> Result<LayerType, ParseError> {
        let lt = match self.peek() {
            Some(Token::Input) => LayerType::Input,
            Some(Token::Dense) => LayerType::Dense,
            Some(Token::Conv2D) => LayerType::Conv2D,
            Some(Token::MaxPool2D) => LayerType::MaxPool2D,
            Some(Token::AvgPool2D) => LayerType::AvgPool2D,
            Some(Token::Flatten) => LayerType::Flatten,
            Some(Token::BatchNorm) => LayerType::BatchNorm,
            Some(Token::Dropout) => LayerType::Dropout,
            Some(Token::Add) => LayerType::Add,
            Some(Token::Concat) => LayerType::Concat,
            Some(Token::ReLU) => LayerType::ReLU,
            Some(Token::Sigmoid) => LayerType::Sigmoid,
            Some(Token::Softmax) => LayerType::Softmax,
            Some(Token::GlobalAvgPool2D) => LayerType::GlobalAvgPool2D,
            Some(Token::ReLU6) => LayerType::ReLU6,
            Some(Token::LeakyReLU) => LayerType::LeakyReLU,
            Some(Token::SiLU) => LayerType::SiLU,
            Some(Token::Mul) => LayerType::Mul,
            Some(Token::Hardswish) => LayerType::Hardswish,
            Some(Token::Upsample) => LayerType::Upsample,
            Some(Token::Conv1D) => LayerType::Conv1D,
            Some(Token::MaxPool1D) => LayerType::MaxPool1D,
            Some(other) => {
                return Err(ParseError {
                    message: format!("expected layer type, got `{other}`"),
                    span: self.current_span(),
                });
            }
            None => return Err(self.eof_error("expected layer type")),
        };
        self.advance();
        Ok(lt)
    }

    fn parse_param_list(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        params.push(self.parse_param()?);
        while self.peek() == Some(&Token::Comma) {
            self.advance();
            params.push(self.parse_param()?);
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<Param, ParseError> {
        let key = self.parse_ident()?;
        let start = key.span.start;
        self.expect(&Token::Colon)?;
        let value = self.parse_value()?;
        let end = value.span().end;
        Ok(Param {
            key,
            value,
            span: start..end,
        })
    }

    fn parse_connection_block(&mut self) -> Result<ConnectionBlock, ParseError> {
        let start = self.current_span().start;
        self.expect(&Token::Connections)?;
        self.expect(&Token::LBrace)?;
        let mut connections = Vec::new();
        while self.peek() != Some(&Token::RBrace) {
            connections.push(self.parse_connection()?);
        }
        let end_tok = self.expect(&Token::RBrace)?;
        Ok(ConnectionBlock {
            connections,
            span: start..end_tok.span.end,
        })
    }

    fn parse_connection(&mut self) -> Result<Connection, ParseError> {
        let start = self.current_span().start;
        let sources = if self.peek() == Some(&Token::LBracket) {
            self.advance();
            let mut ids = Vec::new();
            ids.push(self.parse_ident()?);
            while self.peek() == Some(&Token::Comma) {
                self.advance();
                ids.push(self.parse_ident()?);
            }
            self.expect(&Token::RBracket)?;
            ids
        } else {
            vec![self.parse_ident()?]
        };
        self.expect(&Token::Arrow)?;
        let target = self.parse_ident()?;
        let end_tok = self.expect(&Token::Semicolon)?;
        Ok(Connection {
            sources,
            target,
            span: start..end_tok.span.end,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::lexer::tokenize;

    fn parse_source(source: &str) -> Result<File, ParseError> {
        let tokens = tokenize(source).expect("lexer failed");
        parse(&tokens, source)
    }

    #[test]
    fn test_mnist_example() {
        let source = r#"
version 0.2;

model mnist_classifier {
    config {
        precision: "float32";
        weights: "./weights/mnist.npz";
        target: "avx2";
        batch: 1;
        preprocess: "normalize_0_1";
        io: "stdio";
    }

    layer input   = Input(shape: [28, 28, 1]);
    layer conv1   = Conv2D(filters: 32, kernel: 3, stride: 1, padding: "valid");
    layer pool1   = MaxPool2D(kernel: 2);
    layer flatten  = Flatten();
    layer fc1     = Dense(units: 128, activation: "relu");
    layer output  = Dense(units: 10, activation: "softmax");
}
"#;
        let file = parse_source(source).expect("parse failed");
        insta::assert_debug_snapshot!(file);
    }

    #[test]
    fn test_resnet_block() {
        let source = r#"
version 0.2;

model resnet_block {
    config {
        precision: "float32";
        weights: "./weights/resnet.npz";
        target: "generic";
        io: "stdio";
    }

    layer input  = Input(shape: [32, 32, 64]);
    layer conv1  = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer bn1    = BatchNorm();
    layer relu1  = ReLU();
    layer conv2  = Conv2D(filters: 64, kernel: 3, stride: 1, padding: "same");
    layer bn2    = BatchNorm();
    layer res    = Add();
    layer relu2  = ReLU();

    connections {
        input -> conv1;
        conv1 -> bn1;
        bn1 -> relu1;
        relu1 -> conv2;
        conv2 -> bn2;
        [input, bn2] -> res;
        res -> relu2;
    }
}
"#;
        let file = parse_source(source).expect("parse failed");
        insta::assert_debug_snapshot!(file);
    }

    #[test]
    fn test_missing_semicolon() {
        let source = r#"
version 0.2;
model test {
    config {
        precision: "float32"
        weights: "./w.npz";
    }
    layer input = Input(shape: [1]);
}
"#;
        let err = parse_source(source).unwrap_err();
        assert!(err.message.contains("expected `;`"), "got: {}", err.message);
    }

    #[test]
    fn test_missing_layer_type() {
        let source = r#"
version 0.2;
model test {
    config {
        weights: "./w.npz";
    }
    layer x = Unknown();
}
"#;
        let err = parse_source(source).unwrap_err();
        assert!(
            err.message.contains("expected layer type"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn test_no_layers() {
        let source = r#"
model test {
    config {
        weights: "./w.npz";
    }
}
"#;
        let err = parse_source(source).unwrap_err();
        assert!(
            err.message.contains("expected at least one layer"),
            "got: {}",
            err.message
        );
    }

    #[test]
    fn test_no_version() {
        let source = r#"
model test {
    config {
        weights: "./w.npz";
    }
    layer input = Input(shape: [1]);
}
"#;
        let file = parse_source(source).expect("parse failed");
        assert!(file.version.is_none());
    }

    #[test]
    fn test_bracket_connection() {
        let source = r#"
model test {
    config { weights: "./w.npz"; }
    layer a = Input(shape: [1]);
    layer b = Input(shape: [1]);
    layer c = Add();
    connections {
        [a, b] -> c;
    }
}
"#;
        let file = parse_source(source).expect("parse failed");
        let conn = &file.model.connections.unwrap().connections[0];
        assert_eq!(conn.sources.len(), 2);
        assert_eq!(conn.sources[0].name, "a");
        assert_eq!(conn.sources[1].name, "b");
        assert_eq!(conn.target.name, "c");
    }
}
