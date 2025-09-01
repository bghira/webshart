use crate::discovery::{DatasetDiscovery, DiscoveredDataset};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tar::Archive;
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
    buffer_size: usize, // How many entries to load at once
    load_file_data: bool,
    max_file_size: u64,
    skip_to_index: Option<usize>, // where to begin reading the dataset
    current_file_index: usize,    // where we are in the current shard
}

#[pymethods]
impl PyTarDataLoader {
    #[new]
    #[pyo3(signature = (dataset_or_path, load_file_data=true, max_file_size=50_000_000, buffer_size=100, hf_token=None))]
    fn new(
        dataset_or_path: &PyAny,
        load_file_data: bool,
        max_file_size: u64,
        buffer_size: usize,
        hf_token: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(Runtime::new().expect("Failed to create runtime"));

        // Handle either a DiscoveredDataset or a path string
        let dataset =
            if let Ok(py_dataset) = dataset_or_path.extract::<PyRef<PyDiscoveredDataset>>() {
                py_dataset.inner.clone()
            } else if let Ok(path) = dataset_or_path.extract::<String>() {
                // Auto-discover dataset
                let discovery = DatasetDiscovery::new().with_optional_token(hf_token.clone());

                if Path::new(&path).exists() {
                    discovery.discover_local(Path::new(&path))?
                } else {
                    runtime.block_on(discovery.discover_huggingface(&path, None))?
                }
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
            load_file_data,
            max_file_size,
            skip_to_index: None,
            current_file_index: 0,
        })
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
    fn buffer_size(&self) -> usize {
        self.buffer_size
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
        self.skip_to_index = None;
        self.current_file_index = 0;
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

        // Store the skip index
        self.skip_to_index = Some(idx);

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

        // Set skip index if cursor provided
        self.skip_to_index = cursor_idx;

        Ok(())
    }
}

impl PyTarDataLoader {
    fn process_tar_entries(&mut self, reader: Box<dyn Read>, shard_idx: usize) -> PyResult<()> {
        let mut archive = Archive::new(reader);
        let entries = match archive.entries() {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!(
                    "Failed to read archive entries in shard {}: {}. Skipping shard.",
                    shard_idx, e
                );
                self.current_shard += 1;
                return Ok(());
            }
        };

        let mut entries_loaded = 0;
        let mut entry_errors = 0;
        let debug = std::env::var("WEBSHART_DEBUG").is_ok();

        for (entry_idx, entry) in entries.enumerate() {
            if entries_loaded >= self.buffer_size {
                break; // Buffer is full
            }

            match entry {
                Ok(mut entry) => {
                    let path = match entry.path() {
                        Ok(p) => p.to_string_lossy().to_string(),
                        Err(e) => {
                            eprintln!("Failed to read path for entry {}: {}", entry_idx, e);
                            entry_errors += 1;
                            continue;
                        }
                    };

                    let offset = entry.raw_header_position();
                    let size = entry.size();

                    // Get entry type safely
                    let entry_type = entry.header().entry_type();

                    if debug {
                        eprintln!(
                            "Reading entry {}: {} (size: {}, type: {:?})",
                            entry_idx, path, size, entry_type
                        );
                    }

                    // Only process regular files
                    if entry_type != tar::EntryType::Regular {
                        continue;
                    }

                    // Read file data if requested and within size limit
                    let data = if self.load_file_data && size <= self.max_file_size && size > 0 {
                        let mut buffer = Vec::with_capacity(size as usize);
                        match entry.read_to_end(&mut buffer) {
                            Ok(_) => buffer,
                            Err(e) => {
                                eprintln!("Failed to read data for {}: {}. Continuing...", path, e);
                                entry_errors += 1;
                                // Skip remaining content using copy
                                let _ = std::io::copy(&mut entry, &mut std::io::sink());
                                Vec::new()
                            }
                        }
                    } else {
                        // Skip the content - use the same method as the metadata extractor
                        match std::io::copy(&mut entry, &mut std::io::sink()) {
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!(
                                    "Error skipping content for {}: {}. Continuing...",
                                    path, e
                                );
                                entry_errors += 1;
                            }
                        }
                        Vec::new()
                    };

                    self.entry_buffer.push(PyTarFileEntry {
                        path,
                        offset: offset + 512, // Add header size
                        size,
                        data,
                    });

                    entries_loaded += 1;
                }
                Err(e) => {
                    eprintln!(
                        "Error reading tar entry {} in shard {}: {}",
                        entry_idx, shard_idx, e
                    );
                    entry_errors += 1;

                    // If we're getting too many errors at the beginning, skip the shard
                    if entry_idx < 10 && entry_errors > 5 {
                        eprintln!(
                            "Too many errors at beginning of shard {}. Skipping to next shard.",
                            shard_idx
                        );
                        self.current_shard += 1;
                        return Ok(());
                    }
                    // Otherwise continue trying other entries
                    continue;
                }
            }
        }

        // If we loaded some entries but not a full buffer, we're at the end of the shard
        if entries_loaded > 0 {
            if entries_loaded < self.buffer_size {
                self.current_shard += 1;
            }
            if debug {
                eprintln!(
                    "Loaded {} entries from shard {} ({} errors)",
                    entries_loaded, shard_idx, entry_errors
                );
            }
        } else {
            // No entries loaded, move to next shard
            eprintln!(
                "No entries loaded from shard {}. Moving to next shard.",
                shard_idx
            );
            self.current_shard += 1;
        }

        Ok(())
    }

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
                self.current_file_index = 0;
                self.skip_to_index = None;
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

        // Determine starting index
        let start_idx = if let Some(skip_idx) = self.skip_to_index {
            eprintln!("[DEBUG] Using skip_to_index={}", skip_idx);
            self.skip_to_index = None; // Clear skip index after use
            self.current_file_index = skip_idx;
            skip_idx
        } else {
            eprintln!(
                "[DEBUG] Using current_file_index={}",
                self.current_file_index
            );
            self.current_file_index
        };

        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };

        // Get file entries from metadata
        let mut file_entries = Vec::new();
        let total_files = metadata.num_files();

        // Debug: print what we're trying to load
        eprintln!(
            "[DEBUG] Loading entries from index {} to {} (total files: {})",
            start_idx,
            std::cmp::min(start_idx + self.buffer_size, total_files),
            total_files
        );

        // WORKAROUND: Get ALL files and sort them to ensure consistent ordering
        // This is needed because HashMap doesn't guarantee order
        let mut all_files: Vec<(String, crate::metadata::FileInfo)> = Vec::new();
        for idx in 0..total_files {
            if let Some((filename, file_info)) = metadata.get_file_by_index(idx) {
                all_files.push((filename, file_info.clone()));
            }
        }

        // Sort files by name to ensure consistent ordering
        all_files.sort_by(|a, b| a.0.cmp(&b.0));

        // Now get the files we need based on the sorted order
        let end_idx = std::cmp::min(start_idx + self.buffer_size, total_files);
        for idx in start_idx..end_idx {
            if idx < all_files.len() {
                let (filename, file_info) = &all_files[idx];
                eprintln!("[DEBUG] Index {}: {}", idx, filename);
                file_entries.push((filename.clone(), file_info.clone()));
            }
        }

        drop(dataset);

        if file_entries.is_empty() {
            return Ok(());
        }

        // Load the files
        if is_remote {
            self.load_files_remote(tar_path, token, file_entries)?;
        } else {
            self.load_files_local(tar_path, file_entries)?;
        }

        // Update current file index
        self.current_file_index = start_idx + self.entry_buffer.len();

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

    fn load_files_remote(
        &mut self,
        url: String,
        token: Option<String>,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        // For remote files, we can either:
        // 1. Make individual range requests for each file (more requests but simpler)
        // 2. Make one large range request and parse out the files (fewer requests but more complex)

        // For now, let's use approach 1 for simplicity
        self.runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("Failed to build client");

            for (filename, file_info) in file_entries {
                let offset = file_info.offset;
                let length = file_info.length;

                let data = if self.load_file_data && length <= self.max_file_size && length > 0 {
                    // Make range request for this specific file
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
    #[pyo3(signature = (dataset_or_path, batch_size=32, load_file_data=true, max_file_size=50_000_000, entry_buffer_size=1000, hf_token=None))]
    fn new(
        dataset_or_path: &PyAny,
        batch_size: usize,
        load_file_data: bool,
        max_file_size: u64,
        entry_buffer_size: usize,
        hf_token: Option<String>,
    ) -> PyResult<Self> {
        let base_loader = PyTarDataLoader::new(
            dataset_or_path,
            load_file_data,
            max_file_size,
            entry_buffer_size,
            hf_token,
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
}

// Re-export PyDiscoveredDataset for use with dataloader
pub use crate::discovery::PyDiscoveredDataset;
