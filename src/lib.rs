use pyo3::prelude::*;
mod batch;
mod discovery;
mod error;
mod metadata;

// Re-export main types
pub use batch::{BatchOperations, BatchResult, FileReadRequest};
pub use discovery::{DatasetDiscovery, DiscoveredDataset};
pub use error::{Result, WebshartError};
pub use metadata::{FileInfo, ShardMetadata};

/// A Python module implemented in Rust for fast webdataset shard reading
#[pymodule]
fn _webshart(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Add Python classes
    m.add_class::<discovery::PyDatasetDiscovery>()?;
    m.add_class::<discovery::PyDiscoveredDataset>()?;
    m.add_class::<discovery::PyShardReader>()?;
    m.add_class::<batch::PyBatchOperations>()?;

    Ok(())
}
