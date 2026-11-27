use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod dataloader;
mod discovery;
mod error;
mod extract;
mod field_rename;
mod jxl;
mod jxl_codec;
mod metadata;
mod metadata_resolver;
mod parquet;
mod prefetch;
mod recompress;
mod s3;
mod transform_cache;
mod url_images;
mod xet;
// Re-export main types
use dataloader::{scale_dimensions, PyTarDataLoader, PyTarFileEntry};
pub use dataloader::{AspectBucketIterator, BatchOperations, BatchResult, FileReadRequest};
pub use discovery::{DatasetDiscovery, DiscoveredDataset};
pub use error::{Result, WebshartError};
pub use extract::MetadataExtractor;
pub use metadata::{FileInfo, ShardMetadata};

fn digest_to_hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A Python module implemented in Rust for fast webdataset shard reading
#[pymodule]
fn _webshart(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add Python classes
    m.add_class::<discovery::PyDatasetDiscovery>()?;
    m.add_class::<discovery::PyDiscoveredDataset>()?;
    m.add_class::<discovery::PyShardReader>()?;
    m.add_class::<dataloader::PyBatchOperations>()?;
    m.add_class::<extract::PyMetadataExtractor>()?;
    m.add_class::<PyTarDataLoader>()?;
    m.add_class::<PyTarFileEntry>()?;
    m.add_class::<s3::PyS3Store>()?;
    m.add_class::<parquet::PyParquetCache>()?;
    m.add_class::<jxl::DecodedJxlImage>()?;
    m.add_class::<jxl::JxlDecodeError>()?;
    m.add_function(wrap_pyfunction!(parquet::_parquet_bucket_keys, m)?)?;
    m.add_class::<parquet::PyParquetFile>()?;
    m.add_class::<parquet::PyParquetRows>()?;
    m.add_class::<prefetch::PrefetchIterator>()?;
    m.add_function(wrap_pyfunction!(prefetch::_shuffled_shard_order, m)?)?;
    m.add_class::<transform_cache::TransformCache>()?;
    m.add_class::<url_images::PyURLImageClient>()?;
    m.add_function(wrap_pyfunction!(scale_dimensions, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::digest_to_hex;

    #[test]
    fn digest_to_hex_preserves_leading_zeroes() {
        assert_eq!(digest_to_hex([0x00, 0x01, 0x0f, 0x10, 0xff]), "00010f10ff");
    }
}
