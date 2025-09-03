use crate::discovery::{DatasetDiscovery, DiscoveredDataset};
use crate::error::{Result, WebshartError};
use futures::future::join_all;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Batch read request for a single file
#[derive(Debug, Clone)]
pub struct FileReadRequest {
    /// Dataset to read from
    pub dataset_idx: usize,
    /// Shard index within the dataset
    pub shard_idx: usize,
    /// File index within the shard
    pub file_idx: usize,
}

/// Result of a batch operation
#[derive(Debug)]
pub enum BatchResult<T> {
    Ok(T),
    Err(String),
}

/// Batch operations handler
pub struct BatchOperations {
    runtime: Arc<Runtime>,
}

impl BatchOperations {
    pub fn new() -> Self {
        Self {
            runtime: Arc::new(Runtime::new().expect("Failed to create Tokio runtime")),
        }
    }

    pub fn with_runtime(runtime: Arc<Runtime>) -> Self {
        Self { runtime }
    }

    /// Discover multiple datasets in parallel
    pub fn discover_datasets_batch(
        &self,
        sources: Vec<String>,
        hf_token: Option<String>,
        subfolders: Option<Vec<Option<String>>>,
    ) -> Vec<BatchResult<DiscoveredDataset>> {
        let runtime = self.runtime.clone();

        runtime.block_on(async {
            let futures = sources.into_iter().enumerate().map(|(idx, source)| {
                let token = hf_token.clone();
                let subfolder = subfolders
                    .as_ref()
                    .and_then(|subs| subs.get(idx))
                    .cloned()
                    .flatten();
                let runtime = runtime.clone();

                async move {
                    let discovery =
                        DatasetDiscovery::with_runtime(runtime).with_optional_token(token);

                    // Check if local or remote
                    if std::path::Path::new(&source).exists() {
                        match discovery.discover_local(std::path::Path::new(&source)) {
                            Ok(dataset) => BatchResult::Ok(dataset),
                            Err(e) => BatchResult::Err(e.to_string()),
                        }
                    } else {
                        match discovery
                            .discover_huggingface(&source, subfolder.as_deref())
                            .await
                        {
                            Ok(dataset) => BatchResult::Ok(dataset),
                            Err(e) => BatchResult::Err(e.to_string()),
                        }
                    }
                }
            });

            join_all(futures).await
        })
    }

    /// Load metadata for multiple shards in parallel
    pub fn load_metadata_batch(
        &self,
        dataset: &mut DiscoveredDataset,
        shard_indices: Vec<usize>,
    ) -> Vec<BatchResult<()>> {
        // Use the existing ensure_shard_metadata method which handles loading
        let mut results = Vec::new();

        for idx in shard_indices {
            match dataset.ensure_shard_metadata(idx) {
                Ok(()) => results.push(BatchResult::Ok(())),
                Err(e) => results.push(BatchResult::Err(e.to_string())),
            }
        }

        results
    }

    /// Read multiple files from potentially different shards in parallel
    pub fn read_files_batch(
        &self,
        datasets: &mut [&mut DiscoveredDataset],
        requests: Vec<FileReadRequest>,
    ) -> Vec<BatchResult<Vec<u8>>> {
        let runtime = self.runtime.clone();

        // First ensure all required metadata is loaded
        for req in &requests {
            if let Some(dataset) = datasets.get_mut(req.dataset_idx) {
                let _ = dataset.ensure_shard_metadata(req.shard_idx);
            }
        }

        // Prepare readers and file info
        let mut read_tasks = Vec::new();
        for req in requests {
            if let Some(dataset) = datasets.get(req.dataset_idx) {
                if let Some(shard) = dataset.shards.get(req.shard_idx) {
                    if let Some(metadata) = &shard.metadata {
                        if let Some((filename, file_info)) =
                            metadata.get_file_by_index(req.file_idx)
                        {
                            read_tasks.push((
                                shard.tar_path.clone(),
                                dataset.is_remote,
                                dataset.get_hf_token(),
                                filename,
                                file_info.offset,
                                file_info.length,
                            ));
                        } else {
                            read_tasks.push((String::new(), false, None, String::new(), 0, 0));
                        }
                    } else {
                        read_tasks.push((String::new(), false, None, String::new(), 0, 0));
                    }
                } else {
                    read_tasks.push((String::new(), false, None, String::new(), 0, 0));
                }
            } else {
                read_tasks.push((String::new(), false, None, String::new(), 0, 0));
            }
        }

        // Execute reads in parallel
        runtime.block_on(async {
            let futures = read_tasks.into_iter().map(
                |(tar_path, is_remote, token, filename, offset, length)| async move {
                    if tar_path.is_empty() {
                        return BatchResult::Err("Invalid request".to_string());
                    }

                    let result = if is_remote {
                        read_file_remote(&tar_path, token.as_deref(), offset, length).await
                    } else {
                        read_file_local(&tar_path, &filename, offset, length)
                    };

                    match result {
                        Ok(data) => BatchResult::Ok(data),
                        Err(e) => BatchResult::Err(e.to_string()),
                    }
                },
            );

            join_all(futures).await
        })
    }
}

// Helper functions for file reading
async fn read_file_remote(
    tar_url: &str,
    hf_token: Option<&str>,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();

    let mut request = client
        .get(tar_url)
        .header("Range", format!("bytes={}-{}", offset, offset + length - 1));

    if let Some(token) = hf_token {
        request = request.bearer_auth(token);
    }

    let response = request.send().await?;
    if !response.status().is_success() {
        return Err(WebshartError::InvalidShardFormat(format!(
            "Failed to read file content: {}",
            response.status()
        )));
    }

    Ok(response.bytes().await?.to_vec())
}

fn read_file_local(tar_path: &str, _filename: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(tar_path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut buffer = vec![0u8; length as usize];
    file.read_exact(&mut buffer)?;

    Ok(buffer)
}

/// Python wrapper for batch operations
#[pyclass(name = "BatchOperations")]
pub struct PyBatchOperations {
    inner: BatchOperations,
}

#[pymethods]
impl PyBatchOperations {
    #[new]
    fn new() -> Self {
        Self {
            inner: BatchOperations::new(),
        }
    }

    #[pyo3(signature = (sources, hf_token=None, subfolders=None))]
    fn discover_datasets_batch(
        &self,
        sources: Vec<String>,
        hf_token: Option<String>,
        subfolders: Option<Vec<Option<String>>>,
    ) -> PyResult<Py<PyList>> {
        let results = self
            .inner
            .discover_datasets_batch(sources, hf_token, subfolders);

        Python::with_gil(|py| {
            let list = PyList::empty(py);
            for result in results {
                match result {
                    BatchResult::Ok(dataset) => {
                        let py_dataset = PyDiscoveredDataset { inner: dataset };
                        list.append(py_dataset.into_py(py))?;
                    }
                    BatchResult::Err(_e) => {
                        list.append(py.None())?;
                    }
                }
            }
            Ok(list.into())
        })
    }

    fn load_metadata_batch(
        &self,
        dataset: &mut PyDiscoveredDataset,
        shard_indices: Vec<usize>,
    ) -> PyResult<Py<PyList>> {
        let results = self
            .inner
            .load_metadata_batch(&mut dataset.inner, shard_indices);

        Python::with_gil(|py| {
            let list = PyList::empty(py);
            for result in results {
                match result {
                    BatchResult::Ok(()) => list.append(true)?,
                    BatchResult::Err(e) => {
                        let dict = PyDict::new(py);
                        dict.set_item("error", e)?;
                        list.append(dict)?;
                    }
                }
            }
            Ok(list.into())
        })
    }

    fn read_files_batch(
        &self,
        py: Python,
        datasets: &PyList,
        requests: Vec<(usize, usize, usize)>, // (dataset_idx, shard_idx, file_idx)
    ) -> PyResult<Py<PyList>> {
        // Extract mutable references to datasets
        let mut dataset_refs: Vec<&mut DiscoveredDataset> = Vec::new();
        let mut py_datasets: Vec<PyRefMut<PyDiscoveredDataset>> = Vec::new();

        for i in 0..datasets.len() {
            let item = datasets.get_item(i)?;
            if let Ok(dataset) = item.extract::<PyRefMut<PyDiscoveredDataset>>() {
                py_datasets.push(dataset);
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Expected list of DiscoveredDataset objects",
                ));
            }
        }

        for dataset in &mut py_datasets {
            dataset_refs.push(&mut dataset.inner);
        }

        let requests: Vec<FileReadRequest> = requests
            .into_iter()
            .map(|(dataset_idx, shard_idx, file_idx)| FileReadRequest {
                dataset_idx,
                shard_idx,
                file_idx,
            })
            .collect();

        let results = self.inner.read_files_batch(&mut dataset_refs, requests);

        let list = PyList::empty(py);
        for result in results {
            match result {
                BatchResult::Ok(data) => {
                    list.append(PyBytes::new(py, &data))?;
                }
                BatchResult::Err(_e) => {
                    list.append(py.None())?;
                }
            }
        }
        Ok(list.into())
    }
}

pub trait BatchIterable<T> {
    fn next_item(&mut self) -> PyResult<Option<T>>;

    fn next_batch(&mut self) -> PyResult<Option<Vec<T>>> {
        let batch_size = self.get_batch_size().unwrap_or(1);
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            match self.next_item() {
                Ok(Some(entry)) => batch.push(entry),
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }

        if batch.is_empty() {
            Ok(None)
        } else {
            Ok(Some(batch))
        }
    }

    fn get_batch_size(&self) -> Option<usize>;
}

#[macro_export]
macro_rules! impl_batch_iterator {
    ($name:ident, $py_name:literal, $loader_type:ty, $item_type:ty) => {
        #[pyclass(name = $py_name)]
        pub struct $name {
            loader: Py<$loader_type>,
        }

        #[pymethods]
        impl $name {
            fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            fn __next__(&mut self, py: Python) -> PyResult<Option<Vec<$item_type>>> {
                let mut loader = self.loader.borrow_mut(py);
                loader.next_batch()
            }
        }
    };
}

// Re-export from discovery module
use crate::discovery::PyDiscoveredDataset;
