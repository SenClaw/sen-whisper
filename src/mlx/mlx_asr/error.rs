use mlx_rs::error::Exception;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Exception(#[from] Exception),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error(transparent)]
    LoadWeights(#[from] mlx_rs::error::IoError),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),

    /// A plain message. Carried over from the tokenizer's own error type when
    /// the two merged: the Whisper tokenizer reports a missing special token
    /// this way, and there is nothing to wrap.
    #[error("{0}")]
    Custom(String),
}
