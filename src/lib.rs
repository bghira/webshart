use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod dataloader;
mod discovery;
mod error;
mod extract;
mod metadata;
mod metadata_resolver;
// Re-export main types
pub use dataloader::{AspectBucketIterator, BatchOperations, BatchResult, FileReadRequest};
use dataloader::{PyBucketDataLoader, PyTarDataLoader, PyTarFileEntry, scale_dimensions};
pub use discovery::{DatasetDiscovery, DiscoveredDataset};
pub use error::{Result, WebshartError};
pub use extract::MetadataExtractor;
pub use metadata::{FileInfo, ShardMetadata};

/// A Python module implemented in Rust for fast webdataset shard reading
#[pymodule]
fn _webshart(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Add Python classes
    m.add_class::<discovery::PyDatasetDiscovery>()?;
    m.add_class::<discovery::PyDiscoveredDataset>()?;
    m.add_class::<discovery::PyShardReader>()?;
    m.add_class::<dataloader::PyBatchOperations>()?;
    m.add_class::<extract::PyMetadataExtractor>()?;
    m.add_class::<PyTarDataLoader>()?;
    m.add_class::<PyTarFileEntry>()?;
    m.add_class::<PyBucketDataLoader>()?;
    m.add_function(wrap_pyfunction!(scale_dimensions, m)?)?;

    Ok(())
}
