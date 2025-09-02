use crate::discovery::{DatasetDiscovery, DiscoveredDataset};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::runtime::Runtime;

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
    buffer_size: usize,      // How many entries to load at once
    chunk_size_bytes: usize, // NEW: Size of byte chunks to stream
    load_file_data: bool,
    max_file_size: u64,
    // Tracks the global file index within the current shard
    // This represents the index of the next file to be loaded from the shard
    next_file_to_load: usize,
    // Store the original source and hf_token for state persistence
    source: String,
    hf_token: Option<String>,
}

#[pymethods]
impl PyTarDataLoader {
    #[new]
    #[pyo3(signature = (dataset_or_path, load_file_data=true, max_file_size=50_000_000, buffer_size=100, hf_token=None, chunk_size_mb=10))]
    fn new(
        dataset_or_path: &PyAny,
        load_file_data: bool,
        max_file_size: u64,
        buffer_size: usize,
        hf_token: Option<String>,
        chunk_size_mb: usize, // NEW: Chunk size in MB
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

        Ok(Self {
            dataset: Arc::new(Mutex::new(dataset)),
            runtime,
            current_shard: 0,
            entry_buffer: Vec::with_capacity(buffer_size),
            buffer_position: 0,
            buffer_size: buffer_size.max(1),
            chunk_size_bytes: chunk_size_mb * 1024 * 1024, // Convert MB to bytes
            load_file_data,
            max_file_size,
            next_file_to_load: 0,
            source,
            hf_token,
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
        dict.set_item("hf_token", &self.hf_token)?;

        // Dataset info
        let dataset = self.dataset.lock().unwrap();
        dict.set_item("num_shards", dataset.num_shards())?;
        dict.set_item("is_remote", dataset.is_remote)?;

        // Version for future compatibility
        dict.set_item("version", 2)?;

        Ok(dict.into())
    }

    /// Load state from a dictionary
    fn load_state_dict(&mut self, state_dict: &PyDict) -> PyResult<()> {
        // Validate version
        if let Ok(Some(version_item)) = state_dict.get_item("version") {
            if let Ok(version) = version_item.extract::<i32>() {
                if version > 2 {
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

        // Ensure metadata is loaded for the target shard
        if new_shard < self.dataset.lock().unwrap().num_shards() {
            self.dataset
                .lock()
                .unwrap()
                .ensure_shard_metadata(new_shard)?;
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
        let hf_token = match state_dict.get_item("hf_token")? {
            Some(item) => item.extract::<Option<String>>().unwrap_or(None),
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

        // Create new dataloader
        let mut loader = Self::new(
            source,
            load_file_data,
            max_file_size,
            buffer_size,
            hf_token,
            chunk_size_mb,
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

        Ok(dict.into())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        slf.next_entry()
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
}

// Rest of the implementation remains the same...
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
                        match response.bytes().await {
                            Ok(bytes) => Ok(bytes.to_vec()),
                            Err(e) => {
                                eprintln!("Failed to read response: {}", e);
                                Err(anyhow::anyhow!(e))
                            }
                        }
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

// Simple batch dataloader that yields lists of entries
#[pyclass(name = "BatchDataLoader")]
pub struct PyBatchDataLoader {
    base_loader: PyTarDataLoader,
    batch_size: usize,
}

#[pymethods]
impl PyBatchDataLoader {
    #[new]
    #[pyo3(signature = (dataset_or_path, batch_size=32, load_file_data=true, max_file_size=50_000_000, entry_buffer_size=1000, hf_token=None, chunk_size_mb=10))]
    fn new(
        dataset_or_path: &PyAny,
        batch_size: usize,
        load_file_data: bool,
        max_file_size: u64,
        entry_buffer_size: usize,
        hf_token: Option<String>,
        chunk_size_mb: usize,
    ) -> PyResult<Self> {
        let base_loader = PyTarDataLoader::new(
            dataset_or_path,
            load_file_data,
            max_file_size,
            entry_buffer_size,
            hf_token,
            chunk_size_mb,
        )?;

        Ok(Self {
            base_loader,
            batch_size,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        let mut batch = Vec::with_capacity(slf.batch_size);

        for _ in 0..slf.batch_size {
            match slf.base_loader.next_entry() {
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

    fn reset(&mut self) -> PyResult<()> {
        self.base_loader.reset()
    }

    /// Get the current state of the batch dataloader
    fn state_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = self.base_loader.state_dict(py)?;
        if let Ok(dict) = dict.downcast::<PyDict>(py) {
            dict.set_item("batch_size", self.batch_size)?;
        }
        Ok(dict)
    }

    /// Load state from a dictionary
    fn load_state_dict(&mut self, state_dict: &PyDict) -> PyResult<()> {
        self.base_loader.load_state_dict(state_dict)?;
        if let Ok(Some(item)) = state_dict.get_item("batch_size") {
            if let Ok(batch_size) = item.extract::<usize>() {
                self.batch_size = batch_size;
            }
        }
        Ok(())
    }
}

// Re-export PyDiscoveredDataset for use with dataloader
pub use crate::discovery::PyDiscoveredDataset;
