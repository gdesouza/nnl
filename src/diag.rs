use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::syntax::lexer::Span;

#[derive(Debug, Error, Diagnostic)]
pub enum NncError {
    #[error("{message}")]
    #[diagnostic()]
    Syntax {
        message: String,
        #[source_code]
        src: miette::NamedSource<String>,
        #[label("here")]
        span: SourceSpan,
    },

    #[error("{message}")]
    #[diagnostic()]
    Lex {
        message: String,
        #[source_code]
        src: miette::NamedSource<String>,
        #[label("unexpected character")]
        span: SourceSpan,
    },
}

impl NncError {
    pub fn syntax(message: String, span: Span, filename: &str, source: &str) -> Self {
        NncError::Syntax {
            message,
            src: miette::NamedSource::new(filename, source.to_string()),
            span: (span.start, span.len()).into(),
        }
    }

    pub fn lex(span: Span, filename: &str, source: &str) -> Self {
        NncError::Lex {
            message: "unexpected character".to_string(),
            src: miette::NamedSource::new(filename, source.to_string()),
            span: (span.start, span.len()).into(),
        }
    }
}
