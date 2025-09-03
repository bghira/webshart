use pyo3::PyErr;
use pyo3::exceptions::PyException;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WebshartError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Invalid URL: {0}")]
    InvalidUrl(String),

    #[error("Metadata not found: {0}")]
    MetadataNotFound(String),

    #[error("Invalid shard format: {0}")]
    InvalidShardFormat(String),

    #[error("Discovery failed: {0}")]
    DiscoveryFailed(String),

    #[error("No shards found in dataset")]
    NoShardsFound,

    #[error("Rate limit exceeded")]
    RateLimited,
}

pub type Result<T> = std::result::Result<T, WebshartError>;

// Convert our Rust errors to Python exceptions
impl From<WebshartError> for PyErr {
    fn from(err: WebshartError) -> PyErr {
        PyException::new_err(err.to_string())
    }
}
