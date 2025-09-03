use crate::FileInfo;
use crate::aspect_buckets::{
    AspectBucketIterator, AspectBuckets, format_aspect, scale_dimensions_with_multiple,
};
use crate::discovery::{DatasetDiscovery, DiscoveredDataset};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rand::{rng, seq::SliceRandom};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

#[derive(Debug, Clone)]
pub enum BucketKeyType {
    Aspect,
    GeometryTuple,
    GeometryList,
}

/// Python wrapper for a file entry from a TAR
#[pyclass(name = "TarFileEntry")]
#[derive(Clone)]
pub struct PyTarFileEntry {
    pub path: String,
    pub offset: u64,
    pub size: u64,
    pub data: Vec<u8>,
}

#[pymethods]
impl PyTarFileEntry {
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }

    #[getter]
    fn size(&self) -> u64 {
        self.size
    }

    #[getter]
    fn data(&self) -> PyResult<Py<PyBytes>> {
        Python::with_gil(|py| Ok(PyBytes::new(py, &self.data).into()))
    }

    fn __repr__(&self) -> String {
        format!(
            "TarFileEntry(path='{}', offset={}, size={})",
            self.path, self.offset, self.size
        )
    }
}

/// Buffered streaming dataloader for TAR files
/// Uses a configurable buffer size to balance memory usage and performance
#[pyclass(name = "TarDataLoader")]
pub struct PyTarDataLoader {
    dataset: Arc<Mutex<DiscoveredDataset>>,
    runtime: Arc<Runtime>,
    current_shard: usize,
    entry_buffer: Vec<PyTarFileEntry>,
    buffer_position: usize,
    buffer_size: usize,
    chunk_size_bytes: usize,
    load_file_data: bool,
    max_file_size: u64,
    next_file_to_load: usize,
    source: String,
    hf_token: Option<String>,
    metadata_source: Option<String>,
    batch_size: Option<usize>,
}

#[pymethods]
impl PyTarDataLoader {
    #[new]
    #[pyo3(signature = (dataset_or_path, load_file_data=true, max_file_size=50_000_000, buffer_size=100, hf_token=None, chunk_size_mb=10, batch_size=None))]
    fn new(
        dataset_or_path: &PyAny,
        load_file_data: bool,
        max_file_size: u64,
        buffer_size: usize,
        hf_token: Option<String>,
        chunk_size_mb: usize,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().expect("Failed to create runtime"));

        // Handle either a DiscoveredDataset or a path string
        let (dataset, source) =
            if let Ok(py_dataset) = dataset_or_path.extract::<PyRef<PyDiscoveredDataset>>() {
                // Extract source from the dataset if possible
                let source = py_dataset.inner.name.clone();
                (py_dataset.inner.clone(), source)
            } else if let Ok(path) = dataset_or_path.extract::<String>() {
                // Auto-discover dataset
                let discovery = DatasetDiscovery::new().with_optional_token(hf_token.clone());

                let dataset = if Path::new(&path).exists() {
                    discovery.discover_local(Path::new(&path))?
                } else {
                    runtime.block_on(discovery.discover_huggingface(&path, None))?
                };
                (dataset, path)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected either a DiscoveredDataset or a path string",
                ));
            };
        let metadata_source = dataset.metadata_source.clone();

        Ok(Self {
            dataset: Arc::new(Mutex::new(dataset)),
            runtime,
            current_shard: 0,
            entry_buffer: Vec::with_capacity(buffer_size),
            buffer_position: 0,
            buffer_size: buffer_size.max(1),
            chunk_size_bytes: chunk_size_mb * 1024 * 1024, // Convert MB to bytes
            metadata_source,
            load_file_data,
            max_file_size,
            next_file_to_load: 0,
            source,
            hf_token,
            batch_size,
        })
    }

    fn state_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        // Calculate the actual current file index within the shard
        // This represents the next file that would be returned to the user
        let current_file_index = if self.entry_buffer.is_empty() {
            // No buffer, so we haven't loaded any files yet from current position
            self.next_file_to_load
        } else {
            // We have a buffer with files. The current file index is:
            // next_file_to_load - (total files in buffer - files already consumed)
            let total_in_buffer = self.entry_buffer.len();
            let consumed = self.buffer_position;

            // The actual current position is where we started loading the buffer
            // plus how many we've consumed
            self.next_file_to_load.saturating_sub(total_in_buffer) + consumed
        };

        // Core state
        dict.set_item("current_shard", self.current_shard)?;
        dict.set_item("current_file_index", current_file_index)?; // Local to shard
        dict.set_item("buffer_position", self.buffer_position)?;

        // Configuration
        dict.set_item("buffer_size", self.buffer_size)?;
        dict.set_item("chunk_size_mb", self.chunk_size_bytes / (1024 * 1024))?;
        dict.set_item("load_file_data", self.load_file_data)?;
        dict.set_item("max_file_size", self.max_file_size)?;
        dict.set_item("source", &self.source)?;
        dict.set_item("metadata_source", &self.metadata_source)?;
        dict.set_item("hf_token", &self.hf_token)?;
        dict.set_item("batch_size", &self.batch_size)?;

        // Dataset info
        let dataset = self.dataset.lock().unwrap();
        dict.set_item("num_shards", dataset.num_shards())?;
        dict.set_item("is_remote", dataset.is_remote)?;

        // Version for future compatibility
        dict.set_item("version", 4)?;

        Ok(dict.into())
    }

    /// Load state from a dictionary
    fn load_state_dict(&mut self, state_dict: &PyDict) -> PyResult<()> {
        // Validate version
        if let Ok(Some(version_item)) = state_dict.get_item("version") {
            if let Ok(version) = version_item.extract::<i32>() {
                if version > 4 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported state dict version: {}",
                        version
                    )));
                }
            }
        }

        // Load core state
        let mut new_shard = self.current_shard;
        let mut new_file_index = 0;

        if let Ok(Some(item)) = state_dict.get_item("current_shard") {
            if let Ok(v) = item.extract::<usize>() {
                new_shard = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("current_file_index") {
            if let Ok(v) = item.extract::<usize>() {
                new_file_index = v;
            }
        }

        // Load configuration (optional - only update if present)
        if let Ok(Some(item)) = state_dict.get_item("buffer_size") {
            if let Ok(v) = item.extract::<usize>() {
                self.buffer_size = v.max(1);
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("chunk_size_mb") {
            if let Ok(v) = item.extract::<usize>() {
                self.chunk_size_bytes = v * 1024 * 1024;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("load_file_data") {
            if let Ok(v) = item.extract::<bool>() {
                self.load_file_data = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("max_file_size") {
            if let Ok(v) = item.extract::<u64>() {
                self.max_file_size = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("batch_size") {
            if let Ok(v) = item.extract::<Option<usize>>() {
                self.batch_size = v;
            }
        }

        // Ensure metadata is loaded for the target shard
        if new_shard < self.dataset.lock().unwrap().num_shards() {
            self.dataset
                .lock()
                .unwrap()
                .ensure_shard_metadata(new_shard)?;
        }
        if let Ok(Some(item)) = state_dict.get_item("metadata_source") {
            if let Ok(v) = item.extract::<Option<String>>() {
                self.metadata_source = v;
            }
        }

        // Clear buffer since we're repositioning
        self.entry_buffer.clear();
        self.buffer_position = 0;

        // Set the new position
        self.current_shard = new_shard;
        self.next_file_to_load = new_file_index;

        Ok(())
    }

    #[staticmethod]
    #[pyo3(signature = (state_dict, dataset_or_path=None))]
    fn from_state_dict(
        py: Python,
        state_dict: &PyDict,
        dataset_or_path: Option<&PyAny>,
    ) -> PyResult<Self> {
        // Extract configuration from state dict
        let load_file_data = match state_dict.get_item("load_file_data")? {
            Some(item) => item.extract::<bool>().unwrap_or(true),
            None => true,
        };
        let max_file_size = match state_dict.get_item("max_file_size")? {
            Some(item) => item.extract::<u64>().unwrap_or(50_000_000),
            None => 50_000_000,
        };
        let buffer_size = match state_dict.get_item("buffer_size")? {
            Some(item) => item.extract::<usize>().unwrap_or(100),
            None => 100,
        };
        let chunk_size_mb = match state_dict.get_item("chunk_size_mb")? {
            Some(item) => item.extract::<usize>().unwrap_or(10),
            None => 10,
        };
        let metadata_source = match state_dict.get_item("metadata_source")? {
            Some(item) => item.extract::<Option<String>>().unwrap_or(None),
            None => None,
        };
        let hf_token = match state_dict.get_item("hf_token")? {
            Some(item) => item.extract::<Option<String>>().unwrap_or(None),
            None => None,
        };
        let batch_size = match state_dict.get_item("batch_size")? {
            Some(item) => item.extract::<Option<usize>>().unwrap_or(None),
            None => None,
        };

        // Determine the dataset source
        let source = if let Some(dataset_or_path) = dataset_or_path {
            dataset_or_path
        } else {
            // Try to get source from state dict
            let source_str = match state_dict.get_item("source")? {
                Some(item) => item.extract::<String>().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid source in state_dict")
                })?,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "No dataset_or_path provided and no source in state_dict",
                    ));
                }
            };
            let py_source = py.eval(&format!("'{}'", source_str), None, None)?;
            py_source
        };
        // If recreating from source string and we have metadata_source,
        // we need to create the discovery with metadata_source
        let source = if let Some(dataset_or_path) = dataset_or_path {
            dataset_or_path
        } else {
            let source_str = match state_dict.get_item("source")? {
                Some(item) => item.extract::<String>()?,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "No dataset_or_path provided and no source in state_dict",
                    ));
                }
            };

            // If we have metadata_source, we need to recreate the dataset properly
            if metadata_source.is_some() {
                // Create a DiscoveredDataset with the metadata source
                let discovery = DatasetDiscovery::new()
                    .with_optional_token(hf_token.clone())
                    .with_metadata_source(metadata_source.clone());

                let dataset = if Path::new(&source_str).exists() {
                    discovery.discover_local(Path::new(&source_str))?
                } else {
                    py.allow_threads(|| {
                        Runtime::new()?.block_on(discovery.discover_huggingface(&source_str, None))
                    })?
                };

                // Return the dataset as PyAny
                let py_dataset = Py::new(py, PyDiscoveredDataset { inner: dataset })?;
                return Self::new(
                    py_dataset.as_ref(py),
                    load_file_data,
                    max_file_size,
                    buffer_size,
                    hf_token,
                    chunk_size_mb,
                    batch_size,
                );
            }

            let py_source = py.eval(&format!("'{}'", source_str), None, None)?;
            py_source
        };

        // Create new dataloader
        let mut loader = Self::new(
            source,
            load_file_data,
            max_file_size,
            buffer_size,
            hf_token,
            chunk_size_mb,
            batch_size,
        )?;

        // Load the state
        loader.load_state_dict(state_dict)?;

        Ok(loader)
    }

    fn get_state_summary(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        let mut dataset = self.dataset.lock().unwrap();

        // Calculate current position more accurately
        let current_file_index_in_shard = self.current_file_index();

        // Calculate global position across all shards
        let mut files_processed = 0;
        for i in 0..self.current_shard {
            // Ensure metadata is loaded for accurate count
            dataset.ensure_shard_metadata(i)?;
            if let Some(shard) = dataset.shards.get(i) {
                if let Some(metadata) = &shard.metadata {
                    files_processed += metadata.num_files();
                }
            }
        }
        files_processed += current_file_index_in_shard;

        let total_files = dataset.total_files().unwrap_or(0);

        dict.set_item("current_shard", self.current_shard)?;
        dict.set_item("total_shards", dataset.num_shards())?;
        dict.set_item("current_file_index", current_file_index_in_shard)?;
        dict.set_item("files_processed", files_processed)?;
        dict.set_item("total_files", total_files)?;
        dict.set_item(
            "progress_percent",
            if total_files > 0 {
                files_processed as f64 / total_files as f64 * 100.0
            } else {
                0.0
            },
        )?;
        dict.set_item("batch_size", self.batch_size)?;

        Ok(dict.into())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        slf.next_entry()
    }

    /// Get the next batch of entries
    fn next_batch(&mut self) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        let batch_size = self.batch_size.unwrap_or(1);
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            match self.next_entry() {
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

    /// Create an iterator that yields batches
    fn iter_batches(slf: PyRef<'_, Self>) -> PyResult<PyBatchIterator> {
        if slf.batch_size.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be set to use iter_batches()",
            ));
        }
        Ok(PyBatchIterator { loader: slf.into() })
    }

    #[getter]
    fn num_shards(&self) -> usize {
        self.dataset.lock().unwrap().num_shards()
    }

    #[getter]
    fn current_shard_index(&self) -> usize {
        self.current_shard
    }

    #[getter]
    fn current_file_index(&self) -> usize {
        // Return the actual current position (what would be returned next)
        if self.entry_buffer.is_empty() {
            self.next_file_to_load
        } else {
            // Calculate based on buffer state
            let total_in_buffer = self.entry_buffer.len();
            let consumed = self.buffer_position;
            self.next_file_to_load.saturating_sub(total_in_buffer) + consumed
        }
    }

    #[getter]
    fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    #[getter]
    fn chunk_size_mb(&self) -> usize {
        self.chunk_size_bytes / (1024 * 1024)
    }

    #[getter]
    fn load_file_data(&self) -> bool {
        self.load_file_data
    }

    #[getter]
    fn max_file_size(&self) -> u64 {
        self.max_file_size
    }

    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    #[setter]
    fn set_buffer_size(&mut self, size: usize) {
        self.buffer_size = size.max(1);
        // If we increase buffer size, we might want to grow the capacity
        if self.entry_buffer.capacity() < self.buffer_size {
            self.entry_buffer
                .reserve(self.buffer_size - self.entry_buffer.capacity());
        }
    }

    #[setter]
    fn set_chunk_size_mb(&mut self, size_mb: usize) {
        self.chunk_size_bytes = size_mb.max(1) * 1024 * 1024;
    }

    #[setter]
    fn set_batch_size(&mut self, batch_size: Option<usize>) {
        self.batch_size = batch_size;
    }

    fn get_metadata(&self, shard_idx: usize, py: Python) -> PyResult<PyObject> {
        let mut dataset = self.dataset.lock().unwrap();
        dataset.ensure_shard_metadata(shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shard index {} out of range",
                shard_idx
            ))
        })?;

        let files = shard
            .metadata
            .as_ref()
            .map(|m| m.files())
            .unwrap_or_default();

        let dict = PyDict::new(py);

        for file_info in files.iter() {
            // Convert to JSON value, then to Python dict
            let json_value = serde_json::to_value(file_info)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            let file_dict = pythonize::pythonize(py, &json_value)?;
            dict.set_item(&file_info.path, file_dict)?;
        }

        Ok(dict.into())
    }

    fn reset(&mut self) -> PyResult<()> {
        self.current_shard = 0;
        self.entry_buffer.clear();
        self.buffer_position = 0;
        self.next_file_to_load = 0;
        Ok(())
    }

    fn skip(&mut self, idx: usize) -> PyResult<()> {
        // If it's further than the dataset can support, error out
        let total_files = self.dataset.lock().unwrap().total_files().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to get total files: {}",
                e
            ))
        })?;
        if idx > total_files {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "File index {} out of range",
                idx
            )));
        }

        // Clear the current buffer
        self.entry_buffer.clear();
        self.buffer_position = 0;

        // Find which shard and file index within that shard
        let mut remaining = idx;
        let mut target_shard = 0;
        let dataset = self.dataset.lock().unwrap();

        for (shard_idx, shard) in dataset.shards.iter().enumerate() {
            if let Some(metadata) = &shard.metadata {
                let num_files = metadata.num_files();
                if remaining < num_files {
                    target_shard = shard_idx;
                    break;
                }
                remaining -= num_files;
            }
        }

        drop(dataset);

        self.current_shard = target_shard;
        self.next_file_to_load = remaining;

        Ok(())
    }

    #[pyo3(signature = (shard_idx=None, filename=None, cursor_idx=None))]
    fn shard(
        &mut self,
        shard_idx: Option<usize>,
        filename: Option<String>,
        cursor_idx: Option<usize>,
    ) -> PyResult<()> {
        // Clear the current buffer
        self.entry_buffer.clear();
        self.buffer_position = 0;

        let dataset = self.dataset.lock().unwrap();

        // Determine target shard index
        let target_shard = if let Some(idx) = shard_idx {
            idx
        } else if let Some(fname) = filename {
            // Find shard by filename
            let fname_no_ext = fname.trim_end_matches(".tar");
            dataset
                .shards
                .iter()
                .position(|s| {
                    let shard_name = s
                        .tar_path
                        .rsplit('/')
                        .next()
                        .unwrap_or(&s.tar_path)
                        .trim_end_matches(".tar");
                    shard_name == fname_no_ext
                })
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Shard with filename '{}' not found",
                        fname
                    ))
                })?
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Either shard_idx or filename must be provided",
            ));
        };

        if target_shard >= dataset.num_shards() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shard index {} out of range (0-{})",
                target_shard,
                dataset.num_shards() - 1
            )));
        }

        drop(dataset);

        // Update current shard
        self.current_shard = target_shard;

        // Set file index if cursor provided
        self.next_file_to_load = cursor_idx.unwrap_or(0);

        Ok(())
    }

    #[pyo3(signature = (shard_indices, key="aspect", target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2)))]
    pub fn list_shard_aspect_buckets(
        &self,
        py: Python,
        shard_indices: Vec<usize>,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: u32,
        round_to: Option<usize>,
    ) -> PyResult<Vec<PyObject>> {
        let _key_type = match key {
            "aspect" => BucketKeyType::Aspect,
            "geometry-tuple" => BucketKeyType::GeometryTuple,
            "geometry-list" => BucketKeyType::GeometryList,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "key must be 'aspect', 'geometry-tuple', or 'geometry-list'",
                ));
            }
        };

        let results = if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let mut results = Vec::new();

                    for shard_idx in shard_indices {
                        match self.get_shard_aspect_buckets_internal(
                            shard_idx,
                            key,
                            target_pixel_area,
                            Some(target_resolution_multiple),
                            round_to,
                        ) {
                            Ok(buckets) => results.push(buckets),
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(results)
                })
            })?
        } else {
            self.runtime.block_on(async {
                let mut results = Vec::new();

                for shard_idx in shard_indices {
                    match self.get_shard_aspect_buckets_internal(
                        shard_idx,
                        key,
                        target_pixel_area,
                        Some(target_resolution_multiple),
                        round_to,
                    ) {
                        Ok(buckets) => results.push(buckets),
                        Err(e) => return Err(e),
                    }
                }

                Ok(results)
            })?
        };

        // Convert to Python objects
        let py_results: Vec<PyObject> = results
            .into_iter()
            .map(|bucket| {
                let dict = PyDict::new(py);
                dict.set_item("shard_idx", bucket.shard_idx).unwrap();
                dict.set_item("shard_name", bucket.shard_name).unwrap();

                let buckets_dict = PyDict::new(py);
                for (key, files) in bucket.buckets {
                    let files_list = pyo3::types::PyList::new(
                        py,
                        files
                            .into_iter()
                            .map(|(filename, file_info, original_size)| {
                                let file_dict = PyDict::new(py);
                                file_dict.set_item("filename", filename).unwrap();
                                file_dict.set_item("offset", file_info.offset).unwrap();
                                file_dict.set_item("size", file_info.length).unwrap();
                                if let Some(w) = file_info.width {
                                    file_dict.set_item("width", w).unwrap();
                                }
                                if let Some(h) = file_info.height {
                                    file_dict.set_item("height", h).unwrap();
                                }
                                if let Some(a) = file_info.aspect {
                                    file_dict.set_item("aspect", a).unwrap();
                                }
                                // Add original_size if dimensions were changed
                                if let Some((orig_w, orig_h)) = original_size {
                                    let orig_list = pyo3::types::PyList::new(py, &[orig_w, orig_h]);
                                    file_dict.set_item("original_size", orig_list).unwrap();
                                }
                                file_dict
                            }),
                    );
                    buckets_dict.set_item(key, files_list).unwrap();
                }

                dict.set_item("buckets", buckets_dict).unwrap();
                dict.into()
            })
            .collect();

        Ok(py_results)
    }

    // Generator that yields aspect buckets for all shards
    #[pyo3(signature = (key=None, target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2)))]
    fn list_all_aspect_buckets(
        slf: PyRef<'_, Self>,
        py: Python,
        key: Option<&str>,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: u32,
        round_to: Option<usize>,
    ) -> PyResult<PyObject> {
        let key_str = key.unwrap_or("aspect");
        let _key_type = match key_str {
            "aspect" => BucketKeyType::Aspect,
            "geometry-tuple" => BucketKeyType::GeometryTuple,
            "geometry-list" => BucketKeyType::GeometryList,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "key must be 'aspect', 'geometry-tuple', or 'geometry-list'",
                ));
            }
        };

        let num_shards = slf.num_shards();
        let iterator = AspectBucketIterator {
            loader: slf.into(),
            key_type: key_str.to_string(),
            target_pixel_area,
            target_resolution_multiple,
            round_to,
            current_shard: 0,
            num_shards,
        };

        Py::new(py, iterator).map(|py_iter| py_iter.to_object(py))
    }
}

// Batch iterator for PyTarDataLoader
#[pyclass(name = "TarBatchIterator")]
pub struct PyBatchIterator {
    loader: Py<PyTarDataLoader>,
}

#[pymethods]
impl PyBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        let mut loader = self.loader.borrow_mut(py);
        loader.next_batch()
    }
}

impl PyTarDataLoader {
    fn get_shard_aspect_buckets_internal(
        &self,
        shard_idx: usize,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: Option<u32>,
        round_to: Option<usize>,
    ) -> PyResult<AspectBuckets> {
        let key_type = match key {
            "aspect" => BucketKeyType::Aspect,
            "geometry-tuple" => BucketKeyType::GeometryTuple,
            "geometry-list" => BucketKeyType::GeometryList,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "key must be 'aspect', 'geometry-tuple', or 'geometry-list'",
                ));
            }
        };
        let mut dataset = self.dataset.lock().unwrap();

        // Ensure metadata is loaded with retry logic
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 5;

        loop {
            match dataset.ensure_shard_metadata(shard_idx) {
                Ok(_) => break,
                Err(e) => {
                    if attempts >= MAX_ATTEMPTS {
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            e.to_string(),
                        ));
                    }

                    // Check if it's a rate limit error (429)
                    if e.to_string().contains("429") || e.to_string().contains("rate") {
                        attempts += 1;
                        let wait_time = Duration::from_secs(2u64.pow(attempts));
                        eprintln!(
                            "[webshart] Rate limited, waiting {:?} before retry",
                            wait_time
                        );
                        drop(dataset);
                        std::thread::sleep(wait_time);
                        dataset = self.dataset.lock().unwrap();
                    } else {
                        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                            e.to_string(),
                        ));
                    }
                }
            }
        }

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shard index {} out of range",
                shard_idx
            ))
        })?;

        let shard_name = shard.tar_path.clone();
        let metadata = shard.metadata.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Metadata not loaded")
        })?;

        let mut buckets: BTreeMap<String, Vec<(String, FileInfo, Option<(u32, u32)>)>> =
            BTreeMap::new();

        // Process all files in the shard
        for file_info in metadata.files() {
            if let (Some(width), Some(height)) = (file_info.width, file_info.height) {
                let (bucket_key, original_size) = match key_type {
                    BucketKeyType::Aspect => {
                        if let Some(target_res) = target_pixel_area {
                            let target_resolution_multiple = target_resolution_multiple
                                .expect("target_pixel_area must be Some for this branch");
                            let (scaled_w, scaled_h) = scale_dimensions_with_multiple(
                                width,
                                height,
                                target_res,
                                target_resolution_multiple,
                            );
                            // Recalculate aspect from scaled dimensions
                            let new_aspect = scaled_w as f32 / scaled_h as f32;
                            let key = format_aspect(new_aspect, round_to);
                            // Only include original size if dimensions changed
                            let orig = if scaled_w != width || scaled_h != height {
                                Some((width, height))
                            } else {
                                None
                            };
                            (key, orig)
                        } else {
                            // Use original aspect or calculate it
                            let aspect = file_info.aspect.unwrap_or(width as f32 / height as f32);
                            (format_aspect(aspect, round_to), None)
                        }
                    }
                    BucketKeyType::GeometryTuple => {
                        if let Some(target_res) = target_pixel_area {
                            let (scaled_w, scaled_h) = scale_dimensions_with_multiple(
                                width,
                                height,
                                target_res,
                                Option::expect(
                                    target_resolution_multiple,
                                    "You need to supply a target resolution multiple.",
                                ),
                            );
                            let key = format!("({}, {})", scaled_w, scaled_h);
                            let orig = if scaled_w != width || scaled_h != height {
                                Some((width, height))
                            } else {
                                None
                            };
                            (key, orig)
                        } else {
                            (format!("({}, {})", width, height), None)
                        }
                    }
                    BucketKeyType::GeometryList => {
                        if let Some(target_res) = target_pixel_area {
                            let (scaled_w, scaled_h) = scale_dimensions_with_multiple(
                                width,
                                height,
                                target_res,
                                Option::expect(
                                    target_resolution_multiple,
                                    "You need to supply a target resolution multiple.",
                                ),
                            );
                            let key = format!("[{}, {}]", scaled_w, scaled_h);
                            let orig = if scaled_w != width || scaled_h != height {
                                Some((width, height))
                            } else {
                                None
                            };
                            (key, orig)
                        } else {
                            (format!("[{}, {}]", width, height), None)
                        }
                    }
                };

                let filename = file_info
                    .path
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                buckets.entry(bucket_key).or_insert_with(Vec::new).push((
                    filename,
                    file_info,
                    original_size,
                ));
            }
        }

        Ok(AspectBuckets {
            buckets,
            shard_idx,
            shard_name,
        })
    }
}

impl PyTarDataLoader {
    fn next_entry(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        // Check if we have entries in the buffer
        if self.buffer_position < self.entry_buffer.len() {
            let entry = self.entry_buffer[self.buffer_position].clone();
            self.buffer_position += 1;
            return Ok(Some(entry));
        }

        // Buffer is empty, need to refill
        self.refill_buffer()?;

        // Try again after refilling
        if self.buffer_position < self.entry_buffer.len() {
            let entry = self.entry_buffer[self.buffer_position].clone();
            self.buffer_position += 1;
            Ok(Some(entry))
        } else {
            // No more entries
            Ok(None)
        }
    }

    fn refill_buffer(&mut self) -> PyResult<()> {
        // Clear the buffer and reset position
        self.entry_buffer.clear();
        self.buffer_position = 0;

        // Keep trying shards until we get some entries or run out
        while self.entry_buffer.is_empty()
            && self.current_shard < self.dataset.lock().unwrap().num_shards()
        {
            self.load_entries_from_current_shard()?;

            if self.entry_buffer.is_empty() {
                // No entries in this shard, move to next
                self.current_shard += 1;
                self.next_file_to_load = 0;
            }
        }

        Ok(())
    }

    fn load_entries_from_current_shard(&mut self) -> PyResult<()> {
        let mut dataset = self.dataset.lock().unwrap();

        if self.current_shard >= dataset.num_shards() {
            return Ok(());
        }

        // Ensure metadata is loaded
        dataset.ensure_shard_metadata(self.current_shard)?;

        let shard = dataset.shards.get(self.current_shard).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shard index {} out of range",
                self.current_shard
            ))
        })?;

        let metadata = shard.metadata.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Metadata not loaded")
        })?;

        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };

        // Get file entries from metadata
        let total_files = metadata.num_files();

        // If we've already read all files from this shard, return empty
        if self.next_file_to_load >= total_files {
            return Ok(());
        }

        // Calculate range to load
        let start_idx = self.next_file_to_load;
        let end_idx = std::cmp::min(start_idx + self.buffer_size, total_files);

        // Get ALL files and sort them to ensure consistent ordering
        let mut all_files: Vec<(String, crate::metadata::FileInfo)> = Vec::new();
        for idx in 0..total_files {
            if let Some((filename, file_info)) = metadata.get_file_by_index(idx) {
                all_files.push((filename, file_info.clone()));
            }
        }

        // Sort files by name to ensure consistent ordering
        all_files.sort_by(|a, b| a.0.cmp(&b.0));

        // Collect the files we need
        let mut file_entries = Vec::new();
        for idx in start_idx..end_idx {
            if idx < all_files.len() {
                let (filename, file_info) = &all_files[idx];
                file_entries.push((filename.clone(), file_info.clone()));
            }
        }

        drop(dataset);

        if file_entries.is_empty() {
            return Ok(());
        }

        // Load the files
        if is_remote {
            self.load_files_remote_streaming(tar_path, token, file_entries)?;
        } else {
            self.load_files_local(tar_path, file_entries)?;
        }

        // Update next_file_to_load to point to the next file we'll load
        self.next_file_to_load = end_idx;

        Ok(())
    }

    fn load_files_local(
        &mut self,
        tar_path: String,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(&tar_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to open {}: {}",
                tar_path, e
            ))
        })?;

        for (filename, file_info) in file_entries {
            let offset = file_info.offset;
            let length = file_info.length;

            // Read file data if requested
            let data = if self.load_file_data && length <= self.max_file_size && length > 0 {
                file.seek(SeekFrom::Start(offset)).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to seek: {}", e))
                })?;

                let mut buffer = vec![0u8; length as usize];
                file.read_exact(&mut buffer).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to read file data: {}",
                        e
                    ))
                })?;

                buffer
            } else {
                Vec::new()
            };

            self.entry_buffer.push(PyTarFileEntry {
                path: filename,
                offset,
                size: length,
                data,
            });
        }

        Ok(())
    }

    fn load_files_remote_streaming(
        &mut self,
        url: String,
        token: Option<String>,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        if file_entries.is_empty() {
            return Ok(());
        }

        // Calculate the byte range we need to fetch
        let first_offset = file_entries[0].1.offset;
        let last_entry = &file_entries[file_entries.len() - 1];
        let last_end = last_entry.1.offset + last_entry.1.length;

        // Calculate total range size
        let range_size = last_end - first_offset;

        // If the range is too large, fall back to chunked approach
        if range_size > self.chunk_size_bytes as u64 {
            return self.load_files_remote_chunked(url, token, file_entries);
        }

        // Clone what we need for the async block
        let load_file_data = self.load_file_data;
        let max_file_size = self.max_file_size;

        // Perform the fetch in the async block
        let fetch_result = self.runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to build client");

            // Fetch the entire range in one request
            let mut request = client
                .get(&url)
                .header("Range", format!("bytes={}-{}", first_offset, last_end - 1));

            if let Some(ref token) = token {
                request = request.bearer_auth(token);
            }

            match request.send().await {
                Ok(response) => {
                    if response.status().is_success()
                        || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
                    {
                        response
                            .bytes()
                            .await
                            .map(|bytes| bytes.to_vec())
                            .map_err(|e| {
                                eprintln!("Failed to read response: {}", e);
                                anyhow::anyhow!(e)
                            })
                    } else {
                        eprintln!("HTTP error {}", response.status());
                        Err(anyhow::anyhow!(format!("HTTP error {}", response.status())))
                    }
                }
                Err(e) => {
                    eprintln!("Request failed: {}", e);
                    Err(anyhow::anyhow!(e))
                }
            }
        });

        // Process the result
        match fetch_result {
            Ok(data) => {
                // Extract each file from the downloaded data
                for (filename, file_info) in file_entries {
                    let relative_offset = (file_info.offset - first_offset) as usize;
                    let length = file_info.length as usize;

                    let file_data = if load_file_data
                        && file_info.length <= max_file_size
                        && file_info.length > 0
                        && relative_offset + length <= data.len()
                    {
                        data[relative_offset..relative_offset + length].to_vec()
                    } else {
                        Vec::new()
                    };

                    self.entry_buffer.push(PyTarFileEntry {
                        path: filename,
                        offset: file_info.offset,
                        size: file_info.length,
                        data: file_data,
                    });
                }
                Ok(())
            }
            Err(_) => {
                // Fall back to individual requests
                self.load_files_remote_individual(url, token, file_entries)
            }
        }
    }

    fn load_files_remote_chunked(
        &mut self,
        url: String,
        token: Option<String>,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        // For very large ranges, process in chunks
        let mut processed = 0;

        while processed < file_entries.len() {
            let mut chunk_size = 0u64;
            let mut chunk_end = processed;

            // Build a chunk up to chunk_size_bytes
            for i in processed..file_entries.len() {
                let entry_size = file_entries[i].1.length;
                if chunk_size + entry_size > self.chunk_size_bytes as u64 && chunk_end > processed {
                    break;
                }
                chunk_size += entry_size;
                chunk_end = i + 1;
            }

            // Process this chunk
            let chunk_entries = file_entries[processed..chunk_end].to_vec();
            self.load_files_remote_streaming(url.clone(), token.clone(), chunk_entries)?;

            processed = chunk_end;
        }

        Ok(())
    }

    fn load_files_remote_individual(
        &mut self,
        url: String,
        token: Option<String>,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        // Fallback to individual requests (original implementation)
        self.runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to build client");

            for (filename, file_info) in file_entries {
                let offset = file_info.offset;
                let length = file_info.length;

                let data = if self.load_file_data && length <= self.max_file_size && length > 0 {
                    let mut request = client
                        .get(&url)
                        .header("Range", format!("bytes={}-{}", offset, offset + length - 1));

                    if let Some(ref token) = token {
                        request = request.bearer_auth(token);
                    }

                    match request.send().await {
                        Ok(response) => {
                            if response.status().is_success()
                                || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
                            {
                                match response.bytes().await {
                                    Ok(bytes) => bytes.to_vec(),
                                    Err(e) => {
                                        eprintln!(
                                            "Failed to read response for {}: {}",
                                            filename, e
                                        );
                                        Vec::new()
                                    }
                                }
                            } else {
                                eprintln!("HTTP error {} for file {}", response.status(), filename);
                                Vec::new()
                            }
                        }
                        Err(e) => {
                            eprintln!("Request failed for {}: {}", filename, e);
                            Vec::new()
                        }
                    }
                } else {
                    Vec::new()
                };

                self.entry_buffer.push(PyTarFileEntry {
                    path: filename,
                    offset,
                    size: length,
                    data,
                });
            }

            Ok::<(), PyErr>(())
        })?;

        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (width, height, target_pixel_area, target_resolution_multiple=64))]
pub fn scale_dimensions(
    width: u32,
    height: u32,
    target_pixel_area: u32,
    target_resolution_multiple: u32,
) -> (u32, u32) {
    scale_dimensions_with_multiple(width, height, target_pixel_area, target_resolution_multiple)
}

#[derive(Debug, Clone)]
pub struct BucketEntry {
    pub shard_idx: usize,
    pub filename: String,
    pub file_info: crate::metadata::FileInfo,
    pub original_size: Option<(u32, u32)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BucketSamplingStrategy {
    Sequential,          // Process buckets and their contents in order
    RandomWithinBuckets, // Random sampling within each bucket, sequential bucket order
    FullyRandom,         // Random sampling across all buckets
}

#[pyclass(name = "BucketDataLoader")]
pub struct PyBucketDataLoader {
    dataset: Arc<Mutex<DiscoveredDataset>>,
    runtime: Arc<Runtime>,

    // Bucket data - now built lazily
    buckets: BTreeMap<String, Vec<BucketEntry>>,
    bucket_keys: Vec<String>,

    // Track which shards have been processed
    processed_shards: Vec<bool>,
    next_shard_to_process: usize,

    // Iteration state
    current_bucket_idx: usize,
    current_entry_idx: usize,
    sampling_strategy: BucketSamplingStrategy,

    // Randomization state
    randomized_entries: Option<Vec<(String, usize)>>, // (bucket_key, entry_idx)
    random_position: usize,

    // Loading configuration
    load_file_data: bool,
    max_file_size: u64,
    chunk_size_bytes: usize,

    // Original configuration for state persistence
    key_type: String,
    target_pixel_area: Option<u32>,
    target_resolution_multiple: u32,
    round_to: Option<usize>,
    source: String,
    hf_token: Option<String>,
    metadata_source: Option<String>,

    // Lazy loading configuration
    lazy_load: bool,
    shard_batch_size: usize, // How many shards to process at once

    // Batch loading support
    batch_size: Option<usize>,
}

#[pymethods]
impl PyBucketDataLoader {
    #[new]
    #[pyo3(signature = (
        dataset_or_path,
        key="aspect",
        target_pixel_area=None,
        target_resolution_multiple=64,
        round_to=Some(2),
        sampling_strategy="sequential",
        load_file_data=true,
        max_file_size=50_000_000,
        hf_token=None,
        chunk_size_mb=10,
        lazy_load=true,
        shard_batch_size=10,
        batch_size=None
    ))]
    fn new(
        dataset_or_path: &PyAny,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: u32,
        round_to: Option<usize>,
        sampling_strategy: &str,
        load_file_data: bool,
        max_file_size: u64,
        hf_token: Option<String>,
        chunk_size_mb: usize,
        lazy_load: bool,
        shard_batch_size: usize,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().expect("Failed to create runtime"));

        // Parse sampling strategy
        let sampling = match sampling_strategy {
            "sequential" => BucketSamplingStrategy::Sequential,
            "random_within_buckets" => BucketSamplingStrategy::RandomWithinBuckets,
            "fully_random" => BucketSamplingStrategy::FullyRandom,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "sampling_strategy must be 'sequential', 'random_within_buckets', or 'fully_random'",
                ));
            }
        };

        // Handle dataset discovery (similar to PyTarDataLoader)
        let (dataset, source) =
            if let Ok(py_dataset) = dataset_or_path.extract::<PyRef<PyDiscoveredDataset>>() {
                let source = py_dataset.inner.name.clone();
                (py_dataset.inner.clone(), source)
            } else if let Ok(path) = dataset_or_path.extract::<String>() {
                let discovery = DatasetDiscovery::new().with_optional_token(hf_token.clone());
                let dataset = if Path::new(&path).exists() {
                    discovery.discover_local(Path::new(&path))?
                } else {
                    runtime.block_on(discovery.discover_huggingface(&path, None))?
                };
                (dataset, path)
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected either a DiscoveredDataset or a path string",
                ));
            };

        let metadata_source = dataset.metadata_source.clone();
        let num_shards = dataset.num_shards();

        let mut loader = Self {
            dataset: Arc::new(Mutex::new(dataset)),
            runtime,
            buckets: BTreeMap::new(),
            bucket_keys: Vec::new(),
            processed_shards: vec![false; num_shards],
            next_shard_to_process: 0,
            current_bucket_idx: 0,
            current_entry_idx: 0,
            sampling_strategy: sampling,
            randomized_entries: None,
            random_position: 0,
            load_file_data,
            max_file_size,
            chunk_size_bytes: chunk_size_mb * 1024 * 1024,
            key_type: key.to_string(),
            target_pixel_area,
            target_resolution_multiple,
            round_to,
            source,
            hf_token,
            metadata_source,
            lazy_load,
            shard_batch_size: shard_batch_size.max(1),
            batch_size,
        };

        // For non-lazy mode or fully random sampling, build all buckets upfront
        if !lazy_load || sampling == BucketSamplingStrategy::FullyRandom {
            println!(
                "[webshart] Non-lazy mode or fully random sampling: building all buckets upfront"
            );
            loader.build_all_buckets()?;
        } else {
            println!("[webshart] Lazy mode enabled: buckets will be built on demand");
            // Just ensure we have some initial buckets
            loader.ensure_buckets_available()?;
        }

        Ok(loader)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        slf.next_entry()
    }

    /// Get the next batch of entries
    fn next_batch(&mut self) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        let batch_size = self.batch_size.unwrap_or(1);
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            match self.next_entry() {
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

    /// Create an iterator that yields batches
    fn iter_batches(slf: PyRef<'_, Self>) -> PyResult<PyBucketBatchIterator> {
        if slf.batch_size.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be set to use iter_batches()",
            ));
        }
        Ok(PyBucketBatchIterator { loader: slf.into() })
    }

    fn reset(&mut self) -> PyResult<()> {
        self.current_bucket_idx = 0;
        self.current_entry_idx = 0;
        self.random_position = 0;
        self.next_shard_to_process = 0;

        // Reset processed shards tracking
        self.processed_shards.fill(false);

        // Clear buckets for lazy rebuilding
        self.buckets.clear();
        self.bucket_keys.clear();
        self.randomized_entries = None;

        // Rebuild initial buckets
        if !self.lazy_load || self.sampling_strategy == BucketSamplingStrategy::FullyRandom {
            self.build_all_buckets()?;
        } else {
            self.ensure_buckets_available()?;
        }

        Ok(())
    }

    fn get_bucket_stats(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        dict.set_item("num_buckets", self.buckets.len())?;
        dict.set_item("sampling_strategy", format!("{:?}", self.sampling_strategy))?;
        dict.set_item("lazy_load", self.lazy_load)?;
        dict.set_item(
            "shards_processed",
            self.processed_shards.iter().filter(|&&x| x).count(),
        )?;
        dict.set_item("total_shards", self.processed_shards.len())?;
        dict.set_item("batch_size", self.batch_size)?;

        let mut total_files = 0;
        let mut min_files = usize::MAX;
        let mut max_files = 0;

        let bucket_details = PyDict::new(py);
        for (key, entries) in &self.buckets {
            let count = entries.len();
            total_files += count;
            min_files = min_files.min(count);
            max_files = max_files.max(count);

            bucket_details.set_item(key, count)?;
        }

        dict.set_item("total_files_loaded", total_files)?;
        dict.set_item(
            "min_files_per_bucket",
            if min_files == usize::MAX {
                0
            } else {
                min_files
            },
        )?;
        dict.set_item("max_files_per_bucket", max_files)?;
        dict.set_item(
            "avg_files_per_bucket",
            if self.buckets.is_empty() {
                0.0
            } else {
                total_files as f64 / self.buckets.len() as f64
            },
        )?;
        dict.set_item("bucket_details", bucket_details)?;

        Ok(dict.into())
    }

    fn get_current_bucket(&self) -> Option<String> {
        if self.current_bucket_idx < self.bucket_keys.len() {
            Some(self.bucket_keys[self.current_bucket_idx].clone())
        } else {
            None
        }
    }

    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.batch_size
    }

    #[setter]
    fn set_batch_size(&mut self, batch_size: Option<usize>) {
        self.batch_size = batch_size;
    }

    fn skip_to_bucket(&mut self, bucket_key: &str) -> PyResult<()> {
        // In lazy mode, we might need to load more shards to find this bucket
        if self.lazy_load && !self.buckets.contains_key(bucket_key) {
            // Try to find the bucket by loading more shards
            while self.next_shard_to_process < self.processed_shards.len() {
                self.process_shard_batch()?;
                if self.buckets.contains_key(bucket_key) {
                    break;
                }
            }
        }

        if let Some(idx) = self.bucket_keys.iter().position(|k| k == bucket_key) {
            self.current_bucket_idx = idx;
            self.current_entry_idx = 0;
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Bucket '{}' not found",
                bucket_key
            )))
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "BucketDataLoader(buckets={}, strategy={:?}, current_bucket={}, lazy={}, shards_processed={}/{}, batch_size={:?})",
            self.buckets.len(),
            self.sampling_strategy,
            self.current_bucket_idx,
            self.lazy_load,
            self.processed_shards.iter().filter(|&&x| x).count(),
            self.processed_shards.len(),
            self.batch_size
        )
    }
}

// Batch iterator for PyBucketDataLoader
#[pyclass(name = "BucketBatchIterator")]
pub struct PyBucketBatchIterator {
    loader: Py<PyBucketDataLoader>,
}

#[pymethods]
impl PyBucketBatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        let mut loader = self.loader.borrow_mut(py);
        loader.next_batch()
    }
}

// Internal implementation methods
impl PyBucketDataLoader {
    fn build_all_buckets(&mut self) -> PyResult<()> {
        let num_shards = self.dataset.lock().unwrap().num_shards();

        // Clear existing buckets
        self.buckets.clear();

        // Collect buckets from all shards
        for shard_idx in 0..num_shards {
            println!(
                "[webshart] Processing shard {}/{}",
                shard_idx + 1,
                num_shards
            );
            self.add_shard_to_buckets(shard_idx)?;
        }

        // Mark all shards as processed
        self.processed_shards.fill(true);
        self.next_shard_to_process = num_shards;

        // Finalize bucket structure
        self.finalize_buckets()?;

        Ok(())
    }

    fn ensure_buckets_available(&mut self) -> PyResult<()> {
        // Check if we need more buckets
        if self.current_bucket_idx >= self.bucket_keys.len()
            && self.next_shard_to_process < self.processed_shards.len()
        {
            // Process next batch of shards
            self.process_shard_batch()?;
        }

        Ok(())
    }

    fn process_shard_batch(&mut self) -> PyResult<()> {
        let start = self.next_shard_to_process;
        let end = (start + self.shard_batch_size).min(self.processed_shards.len());

        if start >= end {
            return Ok(());
        }

        println!(
            "[webshart] Processing shards {}-{} (lazy loading)",
            start + 1,
            end
        );

        for shard_idx in start..end {
            if !self.processed_shards[shard_idx] {
                self.add_shard_to_buckets(shard_idx)?;
            }
        }

        self.next_shard_to_process = end;

        // Update bucket keys after adding new shards
        self.update_bucket_keys();

        Ok(())
    }

    fn add_shard_to_buckets(&mut self, shard_idx: usize) -> PyResult<()> {
        let shard_buckets = self.get_shard_buckets(shard_idx)?;

        // Merge into main buckets
        for (bucket_key, entries) in shard_buckets.buckets {
            for (filename, file_info, original_size) in entries {
                let entry = BucketEntry {
                    shard_idx,
                    filename,
                    file_info,
                    original_size,
                };

                self.buckets
                    .entry(bucket_key.clone())
                    .or_insert_with(Vec::new)
                    .push(entry);
            }
        }

        self.processed_shards[shard_idx] = true;
        Ok(())
    }

    fn update_bucket_keys(&mut self) {
        // Update bucket keys to include any new buckets
        let mut new_keys: Vec<String> = self
            .buckets
            .keys()
            .filter(|k| !self.bucket_keys.contains(k))
            .cloned()
            .collect();

        new_keys.sort();
        self.bucket_keys.extend(new_keys);

        // Apply randomization to new entries if needed
        match self.sampling_strategy {
            BucketSamplingStrategy::RandomWithinBuckets => {
                // Randomize entries within each bucket
                let mut rng = rng();
                for entries in self.buckets.values_mut() {
                    // Only shuffle if this bucket has new entries
                    entries.shuffle(&mut rng);
                }
            }
            _ => {}
        }
    }

    fn finalize_buckets(&mut self) -> PyResult<()> {
        // Update bucket keys
        self.bucket_keys = self.buckets.keys().cloned().collect();

        // Initialize randomization if needed
        match self.sampling_strategy {
            BucketSamplingStrategy::RandomWithinBuckets => {
                // Randomize entries within each bucket
                let mut rng = rng();
                for entries in self.buckets.values_mut() {
                    entries.shuffle(&mut rng);
                }
            }
            BucketSamplingStrategy::FullyRandom => {
                // Create a flat list of all entries
                let mut all_entries = Vec::new();
                for (bucket_key, entries) in &self.buckets {
                    for (idx, _) in entries.iter().enumerate() {
                        all_entries.push((bucket_key.clone(), idx));
                    }
                }

                let mut rng = rng();
                all_entries.shuffle(&mut rng);
                self.randomized_entries = Some(all_entries);
            }
            _ => {}
        }

        Ok(())
    }

    fn get_shard_buckets(&self, shard_idx: usize) -> PyResult<AspectBuckets> {
        // Use the existing internal method with retry logic
        let loader = PyTarDataLoader {
            dataset: self.dataset.clone(),
            runtime: self.runtime.clone(),
            current_shard: shard_idx,
            entry_buffer: Vec::new(),
            buffer_position: 0,
            buffer_size: 100,
            chunk_size_bytes: self.chunk_size_bytes,
            load_file_data: false, // Don't need file data for bucket building
            max_file_size: self.max_file_size,
            next_file_to_load: 0,
            source: self.source.clone(),
            hf_token: self.hf_token.clone(),
            metadata_source: self.metadata_source.clone(),
            batch_size: None,
        };

        loader.get_shard_aspect_buckets_internal(
            shard_idx,
            &self.key_type,
            self.target_pixel_area,
            Some(self.target_resolution_multiple),
            self.round_to,
        )
    }

    fn next_entry(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        // Ensure we have buckets available for iteration
        if self.lazy_load {
            self.ensure_buckets_available()?;
        }

        match self.sampling_strategy {
            BucketSamplingStrategy::Sequential => self.next_sequential(),
            BucketSamplingStrategy::RandomWithinBuckets => self.next_random_within_buckets(),
            BucketSamplingStrategy::FullyRandom => self.next_fully_random(),
        }
    }

    fn next_sequential(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        loop {
            // Check if we've exhausted all buckets
            if self.current_bucket_idx >= self.bucket_keys.len() {
                // In lazy mode, try to load more shards
                if self.lazy_load && self.next_shard_to_process < self.processed_shards.len() {
                    self.process_shard_batch()?;
                    continue;
                }
                return Ok(None);
            }

            let bucket_key = &self.bucket_keys[self.current_bucket_idx];
            if let Some(entries) = self.buckets.get(bucket_key) {
                // Check if we've exhausted current bucket
                if self.current_entry_idx >= entries.len() {
                    self.current_bucket_idx += 1;
                    self.current_entry_idx = 0;
                    continue;
                }

                let entry = &entries[self.current_entry_idx];
                self.current_entry_idx += 1;

                return self.load_entry(entry);
            } else {
                // Bucket doesn't exist, move to next
                self.current_bucket_idx += 1;
                self.current_entry_idx = 0;
            }
        }
    }

    fn next_random_within_buckets(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        // Similar to sequential but entries within buckets are already randomized
        self.next_sequential()
    }

    fn next_fully_random(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        if let Some(randomized) = &self.randomized_entries {
            if self.random_position >= randomized.len() {
                return Ok(None);
            }

            let (bucket_key, entry_idx) = &randomized[self.random_position];
            self.random_position += 1;

            let entries = self.buckets.get(bucket_key).unwrap();
            let entry = &entries[*entry_idx];

            self.load_entry(entry)
        } else {
            Ok(None)
        }
    }

    fn load_entry(&self, entry: &BucketEntry) -> PyResult<Option<PyTarFileEntry>> {
        let dataset = self.dataset.lock().unwrap();
        let shard = dataset.shards.get(entry.shard_idx).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Shard index out of range")
        })?;

        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };

        drop(dataset);

        // Load the file data
        let data = if self.load_file_data && entry.file_info.length <= self.max_file_size {
            if is_remote {
                self.load_file_remote(tar_path, token, &entry.filename, &entry.file_info)?
            } else {
                self.load_file_local(tar_path, &entry.filename, &entry.file_info)?
            }
        } else {
            Vec::new()
        };

        Ok(Some(PyTarFileEntry {
            path: entry.filename.clone(),
            offset: entry.file_info.offset,
            size: entry.file_info.length,
            data,
        }))
    }

    fn load_file_remote(
        &self,
        url: String,
        token: Option<String>,
        _filename: &str,
        file_info: &crate::metadata::FileInfo,
    ) -> PyResult<Vec<u8>> {
        // Use tokio::task::block_in_place if we're already in a runtime
        let result = if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let client = reqwest::Client::builder()
                        .timeout(std::time::Duration::from_secs(60))
                        .build()
                        .expect("Failed to build client");

                    let mut request = client.get(&url).header(
                        "Range",
                        format!(
                            "bytes={}-{}",
                            file_info.offset,
                            file_info.offset + file_info.length - 1
                        ),
                    );

                    if let Some(ref token) = token {
                        request = request.bearer_auth(token);
                    }

                    let response = request.send().await.map_err(anyhow::Error::from)?;
                    if response.status().is_success()
                        || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
                    {
                        Ok(response.bytes().await?.to_vec())
                    } else {
                        Err(anyhow::anyhow!("HTTP error {}", response.status()))
                    }
                })
            })
        } else {
            self.runtime.block_on(async {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                    .expect("Failed to build client");

                let mut request = client.get(&url).header(
                    "Range",
                    format!(
                        "bytes={}-{}",
                        file_info.offset,
                        file_info.offset + file_info.length - 1
                    ),
                );

                if let Some(ref token) = token {
                    request = request.bearer_auth(token);
                }

                let response = request.send().await?;
                if response.status().is_success()
                    || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
                {
                    Ok(response.bytes().await?.to_vec())
                } else {
                    Err(anyhow::anyhow!("HTTP error {}", response.status()))
                }
            })
        };

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))
    }

    fn load_file_local(
        &self,
        tar_path: String,
        _filename: &str,
        file_info: &FileInfo,
    ) -> PyResult<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(&tar_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to open {}: {}",
                tar_path, e
            ))
        })?;

        file.seek(SeekFrom::Start(file_info.offset)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to seek: {}", e))
        })?;

        let mut buffer = vec![0u8; file_info.length as usize];
        file.read_exact(&mut buffer).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read file data: {}", e))
        })?;

        Ok(buffer)
    }
}

// Re-export PyDiscoveredDataset for use with dataloader
pub use crate::discovery::PyDiscoveredDataset;
