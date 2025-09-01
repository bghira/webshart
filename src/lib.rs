use pyo3::prelude::*;
mod batch;
mod dataloader;
mod discovery;
mod error;
mod extract;
mod metadata;
mod streaming;
// Re-export main types
pub use batch::{BatchOperations, BatchResult, FileReadRequest};
use dataloader::{PyBatchDataLoader, PyTarDataLoader, PyTarFileEntry};
pub use discovery::{DatasetDiscovery, DiscoveredDataset};
pub use error::{Result, WebshartError};
pub use extract::MetadataExtractor;
pub use metadata::{FileInfo, ShardMetadata};
pub use streaming::{StreamConfig, TarFileEntry, TarStreamer};

/// A Python module implemented in Rust for fast webdataset shard reading
#[pymodule]
fn _webshart(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Add Python classes
    m.add_class::<discovery::PyDatasetDiscovery>()?;
    m.add_class::<discovery::PyDiscoveredDataset>()?;
    m.add_class::<discovery::PyShardReader>()?;
    m.add_class::<batch::PyBatchOperations>()?;
    m.add_class::<extract::PyMetadataExtractor>()?;
    m.add_class::<PyBatchDataLoader>()?;
    m.add_class::<PyTarDataLoader>()?;
    m.add_class::<PyTarFileEntry>()?;

    Ok(())
}
