use crate::FileInfo;
use crate::discovery::{DatasetDiscovery, DiscoveredDataset, PyDiscoveredDataset};
use crate::error::{Result, WebshartError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{rng, seq::SliceRandom};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

// Import from modules
mod aspect_buckets;
mod batch;
mod config;
mod entry_types;
mod file_loading;
pub mod shard_cache;
use crate::impl_batch_iterator;
use crate::metadata::ensure_shard_metadata_with_retry;
pub use aspect_buckets::{AspectBucketIterator, AspectBuckets, scale_dimensions_with_multiple};
use aspect_buckets::{BucketKeyType, BucketSamplingStrategy, calculate_bucket_key};
pub use batch::{BatchIterable, BatchOperations, BatchResult, FileReadRequest, PyBatchOperations};
use config::DataLoaderConfig;
pub use entry_types::{BucketEntry, PyTarFileEntry, create_tar_entry};
use file_loading::{FileLoader, create_file_loader};

// Re-export the entry type as it's part of the public API
pub use entry_types::PyTarFileEntry as TarFileEntry;

// Generate batch iterators using the macro
impl_batch_iterator!(
    PyBatchIterator,
    "PyBatchIterator",
    PyTarDataLoader,
    PyTarFileEntry
);

impl_batch_iterator!(
    PyBucketBatchIterator,
    "PyBucketBatchIterator",
    PyBucketDataLoader,
    PyTarFileEntry
);

// ===== TAR DATA LOADER =====

#[pyclass(name = "TarDataLoader")]
pub struct PyTarDataLoader {
    dataset: Arc<Mutex<DiscoveredDataset>>,
    runtime: Arc<Runtime>,
    config: DataLoaderConfig,
    current_shard: usize,
    entry_buffer: Vec<PyTarFileEntry>,
    buffer_position: usize,
    next_file_to_load: usize,
    source: String,
    metadata_source: Option<String>,
}

impl BatchIterable<PyTarFileEntry> for PyTarDataLoader {
    fn next_item(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        self.next_entry()
    }

    fn get_batch_size(&self) -> Option<usize> {
        self.config.batch_size
    }
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
        let runtime =
            Arc::new(Runtime::new().map_err(|e| {
                WebshartError::Io(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?);

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

        Ok(Self {
            dataset: Arc::new(Mutex::new(dataset)),
            runtime,
            config: DataLoaderConfig {
                load_file_data,
                max_file_size,
                buffer_size: buffer_size.max(1),
                chunk_size_mb,
                hf_token,
                batch_size,
            },
            current_shard: 0,
            entry_buffer: Vec::with_capacity(buffer_size),
            buffer_position: 0,
            metadata_source,
            next_file_to_load: 0,
            source,
        })
    }

    fn state_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        let current_file_index = self.calculate_current_file_index();

        dict.set_item("current_shard", self.current_shard)?;
        dict.set_item("current_file_index", current_file_index)?;
        dict.set_item("buffer_position", self.buffer_position)?;

        self.config.to_state_dict(dict)?;

        dict.set_item("source", &self.source)?;
        dict.set_item("metadata_source", &self.metadata_source)?;

        let dataset = self.dataset.lock().unwrap();
        dict.set_item("num_shards", dataset.num_shards())?;
        dict.set_item("is_remote", dataset.is_remote)?;
        dict.set_item("version", 4)?;

        Ok(dict.into())
    }

    fn load_state_dict(&mut self, state_dict: &PyDict) -> PyResult<()> {
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

        // Load configuration fields individually to preserve existing values
        if let Ok(Some(item)) = state_dict.get_item("load_file_data") {
            if let Ok(v) = item.extract::<bool>() {
                self.config.load_file_data = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("max_file_size") {
            if let Ok(v) = item.extract::<u64>() {
                self.config.max_file_size = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("buffer_size") {
            if let Ok(v) = item.extract::<usize>() {
                self.config.buffer_size = v.max(1);
                if self.entry_buffer.capacity() < self.config.buffer_size {
                    self.entry_buffer
                        .reserve(self.config.buffer_size - self.entry_buffer.capacity());
                }
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("chunk_size_mb") {
            if let Ok(v) = item.extract::<usize>() {
                self.config.chunk_size_mb = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("hf_token") {
            if let Ok(v) = item.extract::<Option<String>>() {
                self.config.hf_token = v;
            }
        }
        if let Ok(Some(item)) = state_dict.get_item("batch_size") {
            if let Ok(v) = item.extract::<Option<usize>>() {
                self.config.batch_size = v;
            }
        }

        if let Ok(Some(item)) = state_dict.get_item("metadata_source") {
            if let Ok(v) = item.extract::<Option<String>>() {
                self.metadata_source = v;
            }
        }

        if new_shard < self.dataset.lock().unwrap().num_shards() {
            ensure_shard_metadata_with_retry(&mut self.dataset.lock().unwrap(), new_shard)?;
        }

        self.entry_buffer.clear();
        self.buffer_position = 0;
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
        let config = DataLoaderConfig::from_state_dict(state_dict);

        let source_obj = Self::determine_source_from_state_dict(py, state_dict, dataset_or_path)?;
        let source = source_obj.as_ref(py);

        let mut loader = Self::new(
            source,
            config.load_file_data,
            config.max_file_size,
            config.buffer_size,
            config.hf_token,
            config.chunk_size_mb,
            config.batch_size,
        )?;

        loader.load_state_dict(state_dict)?;
        Ok(loader)
    }

    fn get_state_summary(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        let mut dataset = self.dataset.lock().unwrap();
        let current_file_index_in_shard = self.calculate_current_file_index();

        let files_processed =
            self.calculate_files_processed(&mut dataset, current_file_index_in_shard)?;
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
        dict.set_item("batch_size", self.config.batch_size)?;

        Ok(dict.into())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        slf.next_entry()
    }

    fn next_batch(&mut self) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        <Self as BatchIterable<PyTarFileEntry>>::next_batch(self)
    }

    fn iter_batches(slf: PyRef<'_, Self>) -> PyResult<PyBatchIterator> {
        if slf.config.batch_size.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be set to use iter_batches()",
            ));
        }
        Ok(PyBatchIterator { loader: slf.into() })
    }

    // Getters
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
        self.calculate_current_file_index()
    }

    #[getter]
    fn buffer_size(&self) -> usize {
        self.config.buffer_size
    }

    #[getter]
    fn chunk_size_mb(&self) -> usize {
        self.config.chunk_size_mb
    }

    #[getter]
    fn load_file_data(&self) -> bool {
        self.config.load_file_data
    }

    #[getter]
    fn max_file_size(&self) -> u64 {
        self.config.max_file_size
    }

    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.config.batch_size
    }

    // Setters
    #[setter]
    fn set_buffer_size(&mut self, size: usize) {
        self.config.buffer_size = size.max(1);
        if self.entry_buffer.capacity() < self.config.buffer_size {
            self.entry_buffer
                .reserve(self.config.buffer_size - self.entry_buffer.capacity());
        }
    }

    #[setter]
    fn set_chunk_size_mb(&mut self, size_mb: usize) {
        self.config.chunk_size_mb = size_mb.max(1);
    }

    #[setter]
    fn set_batch_size(&mut self, batch_size: Option<usize>) {
        self.config.batch_size = batch_size;
    }

    fn get_metadata(&self, shard_idx: usize, py: Python) -> PyResult<PyObject> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let files = shard
            .metadata
            .as_ref()
            .map(|m| m.files())
            .unwrap_or_default();

        let dict = PyDict::new(py);

        for file_info in files.iter() {
            let file_dict = pythonize::pythonize(py, file_info)?;
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
        let total_files = self.dataset.lock().unwrap().total_files().map_err(|e| {
            WebshartError::DiscoveryFailed(format!("Failed to get total files: {}", e))
        })?;

        if idx > total_files {
            return Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range",
                idx
            ))
            .into());
        }

        self.entry_buffer.clear();
        self.buffer_position = 0;

        let (target_shard, remaining) = self.find_shard_for_index(idx)?;

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
        self.entry_buffer.clear();
        self.buffer_position = 0;

        let dataset = self.dataset.lock().unwrap();

        let target_shard = if let Some(idx) = shard_idx {
            idx
        } else if let Some(fname) = filename {
            self.find_shard_by_filename(&dataset, &fname)?
        } else {
            return Err(WebshartError::InvalidShardFormat(
                "Either shard_idx or filename must be provided".to_string(),
            )
            .into());
        };

        if target_shard >= dataset.num_shards() {
            return Err(WebshartError::InvalidShardFormat(format!(
                "Shard index {} out of range (0-{})",
                target_shard,
                dataset.num_shards() - 1
            ))
            .into());
        }

        drop(dataset);

        self.current_shard = target_shard;
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
        let results = self.get_aspect_buckets_for_shards(
            shard_indices,
            key,
            target_pixel_area,
            Some(target_resolution_multiple),
            round_to,
        )?;

        let py_results: Vec<PyObject> = results
            .into_iter()
            .map(|bucket| self.aspect_buckets_to_py_dict(py, bucket))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py_results)
    }

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
        let _key_type = BucketKeyType::parse(key_str)?;

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

// Implementation methods for PyTarDataLoader (continued in same file for now)
impl PyTarDataLoader {
    pub(crate) fn load_file_by_info(
        &self,
        shard_idx: usize,
        file_path: &str,
        file_info: &FileInfo,
    ) -> PyResult<PyTarFileEntry> {
        let dataset = self.dataset.lock().unwrap();
        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat("Shard index out of range".to_string())
        })?;

        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };

        drop(dataset);

        let data = if self.config.load_file_data && file_info.length <= self.config.max_file_size {
            self.load_single_file_data(&tar_path, file_info, is_remote, token)?
        } else {
            Vec::new()
        };

        Ok(create_tar_entry(file_path.to_string(), file_info, data))
    }

    pub(crate) fn get_shard_aspect_buckets_internal(
        &self,
        shard_idx: usize,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: Option<u32>,
        round_to: Option<usize>,
    ) -> PyResult<AspectBuckets> {
        let key_type = BucketKeyType::parse(key)?;
        let mut dataset = self.dataset.lock().unwrap();

        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let shard_name = shard.tar_path.clone();
        let metadata = shard
            .metadata
            .as_ref()
            .ok_or_else(|| WebshartError::MetadataNotFound("Metadata not loaded".to_string()))?;

        let mut buckets: BTreeMap<String, Vec<(String, FileInfo, Option<(u32, u32)>)>> =
            BTreeMap::new();

        for file_info in metadata.files() {
            if let (Some(width), Some(height)) = (file_info.width, file_info.height) {
                let target_resolution_multiple = target_resolution_multiple.unwrap_or(64);

                let (bucket_key, original_size) = calculate_bucket_key(
                    &key_type,
                    width,
                    height,
                    file_info.aspect,
                    target_pixel_area,
                    target_resolution_multiple,
                    round_to,
                );

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

    // Private helper methods
    fn calculate_current_file_index(&self) -> usize {
        if self.entry_buffer.is_empty() {
            self.next_file_to_load
        } else {
            let total_in_buffer = self.entry_buffer.len();
            let consumed = self.buffer_position;
            self.next_file_to_load.saturating_sub(total_in_buffer) + consumed
        }
    }

    fn calculate_files_processed(
        &self,
        dataset: &mut DiscoveredDataset,
        current_file_index_in_shard: usize,
    ) -> PyResult<usize> {
        let mut files_processed = 0;
        for i in 0..self.current_shard {
            ensure_shard_metadata_with_retry(dataset, i)?;
            if let Some(shard) = dataset.shards.get(i) {
                if let Some(metadata) = &shard.metadata {
                    files_processed += metadata.num_files();
                }
            }
        }
        files_processed += current_file_index_in_shard;
        Ok(files_processed)
    }

    fn find_shard_for_index(&self, idx: usize) -> PyResult<(usize, usize)> {
        let mut remaining = idx;
        let mut dataset = self.dataset.lock().unwrap();
        let num_shards = dataset.num_shards();

        for shard_idx in 0..num_shards {
            if dataset.shards[shard_idx].metadata.is_none() {
                ensure_shard_metadata_with_retry(&mut *dataset, shard_idx)?;
            }

            if let Some(metadata) = &dataset.shards[shard_idx].metadata {
                let num_files = metadata.num_files();
                if remaining < num_files {
                    return Ok((shard_idx, remaining));
                }
                remaining -= num_files;
            } else {
                return Err(WebshartError::MetadataNotFound(format!(
                    "Metadata for shard {} could not be loaded",
                    shard_idx
                ))
                .into());
            }
        }

        Ok((num_shards.saturating_sub(1), remaining))
    }

    fn find_shard_by_filename(
        &self,
        dataset: &DiscoveredDataset,
        filename: &str,
    ) -> PyResult<usize> {
        let fname_no_ext = filename.trim_end_matches(".tar");
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
                WebshartError::InvalidShardFormat(format!(
                    "Shard with filename '{}' not found",
                    filename
                ))
                .into()
            })
    }

    fn determine_source_from_state_dict<'py>(
        py: Python<'py>,
        state_dict: &PyDict,
        dataset_or_path: Option<&'py PyAny>,
    ) -> PyResult<PyObject> {
        if let Some(dataset_or_path) = dataset_or_path {
            return Ok(dataset_or_path.to_object(py));
        }

        let source_str = match state_dict.get_item("source")? {
            Some(item) => item.extract::<String>()?,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "No dataset_or_path provided and no source in state_dict",
                ));
            }
        };

        let metadata_source = match state_dict.get_item("metadata_source")? {
            Some(item) => item.extract::<Option<String>>().unwrap_or(None),
            None => None,
        };

        let hf_token = match state_dict.get_item("hf_token")? {
            Some(item) => item.extract::<Option<String>>().unwrap_or(None),
            None => None,
        };

        if metadata_source.is_some() {
            let discovery = DatasetDiscovery::new()
                .with_optional_token(hf_token)
                .with_metadata_source(metadata_source);

            let dataset = if Path::new(&source_str).exists() {
                discovery.discover_local(Path::new(&source_str))?
            } else {
                py.allow_threads(|| {
                    Runtime::new()?.block_on(discovery.discover_huggingface(&source_str, None))
                })?
            };

            let py_dataset = Py::new(py, PyDiscoveredDataset { inner: dataset })?;
            return Ok(py_dataset.to_object(py));
        }

        let py_source = py.eval(&format!("'{}'", source_str), None, None)?;
        Ok(py_source.to_object(py))
    }

    fn get_aspect_buckets_for_shards(
        &self,
        shard_indices: Vec<usize>,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: Option<u32>,
        round_to: Option<usize>,
    ) -> PyResult<Vec<AspectBuckets>> {
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let mut results = Vec::new();

                    for shard_idx in shard_indices {
                        match self.get_shard_aspect_buckets_internal(
                            shard_idx,
                            key,
                            target_pixel_area,
                            target_resolution_multiple,
                            round_to,
                        ) {
                            Ok(buckets) => results.push(buckets),
                            Err(e) => return Err(e),
                        }
                    }

                    Ok(results)
                })
            })
        } else {
            self.runtime.block_on(async {
                let mut results = Vec::new();

                for shard_idx in shard_indices {
                    match self.get_shard_aspect_buckets_internal(
                        shard_idx,
                        key,
                        target_pixel_area,
                        target_resolution_multiple,
                        round_to,
                    ) {
                        Ok(buckets) => results.push(buckets),
                        Err(e) => return Err(e),
                    }
                }

                Ok(results)
            })
        }
    }

    fn aspect_buckets_to_py_dict(&self, py: Python, bucket: AspectBuckets) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("shard_idx", bucket.shard_idx)?;
        dict.set_item("shard_name", bucket.shard_name)?;

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
                        if let Some((orig_w, orig_h)) = original_size {
                            let orig_list = pyo3::types::PyList::new(py, &[orig_w, orig_h]);
                            file_dict.set_item("original_size", orig_list).unwrap();
                        }
                        file_dict
                    }),
            );
            buckets_dict.set_item(key, files_list)?;
        }

        dict.set_item("buckets", buckets_dict)?;
        Ok(dict.into())
    }

    fn load_single_file_data(
        &self,
        tar_path: &str,
        file_info: &FileInfo,
        is_remote: bool,
        token: Option<String>,
    ) -> Result<Vec<u8>> {
        let dataset = self.dataset.lock().unwrap();

        // If we have a shard cache and this is remote, try to get cached version
        if is_remote && dataset.shard_cache.is_some() {
            let shard_name = tar_path.rsplit('/').next().unwrap_or(tar_path);
            let cache = dataset.shard_cache.as_ref().unwrap().clone();
            drop(dataset); // Release lock before async operation

            // Try to get/download to cache
            if let Ok(cached_path) =
                self.runtime
                    .block_on(cache.cache_shard(shard_name, tar_path, token.clone()))
            {
                // Use the cached local file
                let loader = create_file_loader(
                    &cached_path.to_string_lossy(),
                    false, // it's local now
                    None,  // no token needed for local
                    self.runtime.clone(),
                );
                return loader.load_file(file_info);
            }
        } else {
            drop(dataset);
        }

        // Fallback to original behavior
        let loader = create_file_loader(tar_path, is_remote, token, self.runtime.clone());
        loader.load_file(file_info)
    }

    fn next_entry(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        if self.buffer_position < self.entry_buffer.len() {
            let entry = self.entry_buffer[self.buffer_position].clone();
            self.buffer_position += 1;
            return Ok(Some(entry));
        }

        self.refill_buffer()?;

        if self.buffer_position < self.entry_buffer.len() {
            let entry = self.entry_buffer[self.buffer_position].clone();
            self.buffer_position += 1;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }

    fn refill_buffer(&mut self) -> PyResult<()> {
        self.entry_buffer.clear();
        self.buffer_position = 0;

        while self.entry_buffer.is_empty()
            && self.current_shard < self.dataset.lock().unwrap().num_shards()
        {
            self.load_entries_from_current_shard()?;

            if self.entry_buffer.is_empty() {
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

        ensure_shard_metadata_with_retry(&mut dataset, self.current_shard)?;

        let shard = dataset.shards.get(self.current_shard).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!(
                "Shard index {} out of range",
                self.current_shard
            ))
        })?;

        let metadata = shard
            .metadata
            .as_ref()
            .ok_or_else(|| WebshartError::MetadataNotFound("Metadata not loaded".to_string()))?;

        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };

        let total_files = metadata.num_files();

        if self.next_file_to_load >= total_files {
            return Ok(());
        }

        let start_idx = self.next_file_to_load;
        let end_idx = std::cmp::min(start_idx + self.config.buffer_size, total_files);

        let mut all_files: Vec<(String, crate::metadata::FileInfo)> = Vec::new();
        for idx in 0..total_files {
            if let Some((filename, file_info)) = metadata.get_file_by_index(idx) {
                all_files.push((filename, file_info.clone()));
            }
        }

        all_files.sort_by(|a, b| a.0.cmp(&b.0));

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

        self.load_file_batch(tar_path, token, file_entries)?;
        self.next_file_to_load = end_idx;

        Ok(())
    }

    fn load_file_batch(
        &mut self,
        tar_path: String,
        token: Option<String>,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        let is_remote = tar_path.starts_with("http");

        if is_remote && file_entries.len() > 1 {
            self.load_files_remote_streaming(tar_path, token, file_entries)
        } else {
            for (filename, file_info) in file_entries {
                let data = if self.config.load_file_data
                    && file_info.length <= self.config.max_file_size
                {
                    self.load_single_file_data(&tar_path, &file_info, is_remote, token.clone())
                        .unwrap_or_else(|e| {
                            eprintln!("Failed to load {}: {}", filename, e);
                            Vec::new()
                        })
                } else {
                    Vec::new()
                };

                self.entry_buffer
                    .push(create_tar_entry(filename, &file_info, data));
            }
            Ok(())
        }
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

        // Check if we have shard cache
        let dataset = self.dataset.lock().unwrap();
        if let Some(cache) = &dataset.shard_cache {
            let shard_name = url.rsplit('/').next().unwrap_or(&url);
            let cache_clone = cache.clone();
            drop(dataset);

            // Try to cache the entire shard first
            match self
                .runtime
                .block_on(cache_clone.cache_shard(shard_name, &url, token.clone()))
            {
                Ok(cached_path) => {
                    // Load from cached file instead
                    let cached_url = cached_path.to_string_lossy().to_string();
                    drop(cache_clone);

                    // Process all files from the cached shard
                    for (filename, file_info) in file_entries {
                        let data = if self.config.load_file_data
                            && file_info.length <= self.config.max_file_size
                        {
                            self.load_single_file_data(&cached_url, &file_info, false, None)
                                .unwrap_or_else(|e| {
                                    eprintln!("Failed to load {}: {}", filename, e);
                                    Vec::new()
                                })
                        } else {
                            Vec::new()
                        };

                        self.entry_buffer
                            .push(create_tar_entry(filename, &file_info, data));
                    }
                    return Ok(());
                }
                Err(_) => {
                    // Cache failed, continue with original streaming approach
                }
            }
        } else {
            drop(dataset);
        }

        let first_offset = file_entries[0].1.offset;
        let last_entry = &file_entries[file_entries.len() - 1];
        let last_end = last_entry.1.offset + last_entry.1.length;

        let range_size = last_end - first_offset;

        let chunk_size_bytes = self.config.chunk_size_mb * 1024 * 1024;
        if range_size > chunk_size_bytes as u64 {
            return self.load_files_remote_chunked(url, token, file_entries);
        }

        let load_file_data = self.config.load_file_data;
        let max_file_size = self.config.max_file_size;

        let fetch_result = self.runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .map_err(WebshartError::from)?;

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
                            .map_err(WebshartError::from)
                    } else if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        Err(WebshartError::RateLimited)
                    } else {
                        Err(WebshartError::Http(
                            response.error_for_status().unwrap_err(),
                        ))
                    }
                }
                Err(e) => Err(WebshartError::Http(e)),
            }
        });

        match fetch_result {
            Ok(data) => {
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

                    self.entry_buffer
                        .push(create_tar_entry(filename, &file_info, file_data));
                }
                Ok(())
            }
            Err(_) => self.load_file_batch(url, token, file_entries),
        }
    }

    fn load_files_remote_chunked(
        &mut self,
        url: String,
        token: Option<String>,
        file_entries: Vec<(String, crate::metadata::FileInfo)>,
    ) -> PyResult<()> {
        let chunk_size_bytes = self.config.chunk_size_mb * 1024 * 1024;
        let mut processed = 0;

        while processed < file_entries.len() {
            let mut chunk_size = 0u64;
            let mut chunk_end = processed;

            for i in processed..file_entries.len() {
                let entry_size = file_entries[i].1.length;
                if chunk_size + entry_size > chunk_size_bytes as u64 && chunk_end > processed {
                    break;
                }
                chunk_size += entry_size;
                chunk_end = i + 1;
            }

            let chunk_entries = file_entries[processed..chunk_end].to_vec();
            self.load_files_remote_streaming(url.clone(), token.clone(), chunk_entries)?;

            processed = chunk_end;
        }

        Ok(())
    }
}

// ===== BUCKET DATA LOADER =====

#[pyclass(name = "BucketDataLoader")]
pub struct PyBucketDataLoader {
    tar_loader: Py<PyTarDataLoader>,
    buckets: BTreeMap<String, Vec<BucketEntry>>,
    bucket_keys: Vec<String>,
    processed_shards: Vec<bool>,
    next_shard_to_process: usize,
    current_bucket_idx: usize,
    current_entry_idx: usize,
    sampling_strategy: BucketSamplingStrategy,
    randomized_entries: Option<Vec<(String, usize)>>,
    random_position: usize,
    key_type: String,
    target_pixel_area: Option<u32>,
    target_resolution_multiple: u32,
    round_to: Option<usize>,
    lazy_load: bool,
    shard_batch_size: usize,
    batch_size: Option<usize>,
}

impl BatchIterable<PyTarFileEntry> for PyBucketDataLoader {
    fn next_item(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        Python::with_gil(|py| self.next_entry(py))
    }

    fn get_batch_size(&self) -> Option<usize> {
        self.batch_size
    }
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
        py: Python,
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
        let sampling = BucketSamplingStrategy::parse(sampling_strategy)?;

        let tar_loader = PyTarDataLoader::new(
            dataset_or_path,
            load_file_data,
            max_file_size,
            100, // Default buffer size
            hf_token.clone(),
            chunk_size_mb,
            batch_size,
        )?;

        let num_shards = tar_loader.num_shards();
        let tar_loader_py = Py::new(py, tar_loader)?;

        let mut loader = Self {
            tar_loader: tar_loader_py,
            buckets: BTreeMap::new(),
            bucket_keys: Vec::new(),
            processed_shards: vec![false; num_shards],
            next_shard_to_process: 0,
            current_bucket_idx: 0,
            current_entry_idx: 0,
            sampling_strategy: sampling,
            randomized_entries: None,
            random_position: 0,
            key_type: key.to_string(),
            target_pixel_area,
            target_resolution_multiple,
            round_to,
            lazy_load,
            shard_batch_size: shard_batch_size.max(1),
            batch_size,
        };

        if !lazy_load || sampling == BucketSamplingStrategy::FullyRandom {
            println!("[webshart] Building all buckets upfront");
            loader.build_all_buckets(py)?;
        } else {
            println!("[webshart] Lazy mode enabled: buckets will be built on demand");
            loader.ensure_buckets_available(py)?;
        }

        Ok(loader)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        Python::with_gil(|py| slf.next_entry(py))
    }

    fn next_batch(&mut self) -> PyResult<Option<Vec<PyTarFileEntry>>> {
        Python::with_gil(|py| <Self as BatchIterable<PyTarFileEntry>>::next_batch(self))
    }

    fn iter_batches(slf: PyRef<'_, Self>) -> PyResult<PyBucketBatchIterator> {
        if slf.batch_size.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "batch_size must be set to use iter_batches()",
            ));
        }
        Ok(PyBucketBatchIterator { loader: slf.into() })
    }

    fn reset(&mut self, py: Python) -> PyResult<()> {
        self.current_bucket_idx = 0;
        self.current_entry_idx = 0;
        self.random_position = 0;
        self.next_shard_to_process = 0;

        self.processed_shards.fill(false);
        self.buckets.clear();
        self.bucket_keys.clear();
        self.randomized_entries = None;

        self.tar_loader.borrow_mut(py).reset()?;

        if !self.lazy_load || self.sampling_strategy == BucketSamplingStrategy::FullyRandom {
            self.build_all_buckets(py)?;
        } else {
            self.ensure_buckets_available(py)?;
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

    fn skip_to_bucket(&mut self, py: Python, bucket_key: &str) -> PyResult<()> {
        if self.lazy_load && !self.buckets.contains_key(bucket_key) {
            while self.next_shard_to_process < self.processed_shards.len() {
                self.process_shard_batch(py)?;
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
            Err(
                WebshartError::InvalidShardFormat(format!("Bucket '{}' not found", bucket_key))
                    .into(),
            )
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

// Implementation methods for PyBucketDataLoader
impl PyBucketDataLoader {
    fn build_all_buckets(&mut self, py: Python) -> PyResult<()> {
        let num_shards = self.tar_loader.borrow(py).num_shards();

        self.buckets.clear();

        for shard_idx in 0..num_shards {
            println!(
                "[webshart] Processing shard {}/{}",
                shard_idx + 1,
                num_shards
            );
            self.add_shard_to_buckets(py, shard_idx)?;
        }

        self.processed_shards.fill(true);
        self.next_shard_to_process = num_shards;
        self.finalize_buckets()?;

        Ok(())
    }

    fn ensure_buckets_available(&mut self, py: Python) -> PyResult<()> {
        if self.current_bucket_idx >= self.bucket_keys.len()
            && self.next_shard_to_process < self.processed_shards.len()
        {
            self.process_shard_batch(py)?;
        }
        Ok(())
    }

    fn process_shard_batch(&mut self, py: Python) -> PyResult<()> {
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
                self.add_shard_to_buckets(py, shard_idx)?;
            }
        }

        self.next_shard_to_process = end;
        self.update_bucket_keys();

        Ok(())
    }

    fn add_shard_to_buckets(&mut self, py: Python, shard_idx: usize) -> PyResult<()> {
        let tar_loader = self.tar_loader.borrow(py);
        let shard_buckets = tar_loader.get_shard_aspect_buckets_internal(
            shard_idx,
            &self.key_type,
            self.target_pixel_area,
            Some(self.target_resolution_multiple),
            self.round_to,
        )?;

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
        let mut new_keys: Vec<String> = self
            .buckets
            .keys()
            .filter(|k| !self.bucket_keys.contains(k))
            .cloned()
            .collect();

        new_keys.sort();
        self.bucket_keys.extend(new_keys);

        if self.sampling_strategy == BucketSamplingStrategy::RandomWithinBuckets {
            let mut rng = rng();
            for entries in self.buckets.values_mut() {
                entries.shuffle(&mut rng);
            }
        }
    }

    fn finalize_buckets(&mut self) -> PyResult<()> {
        self.bucket_keys = self.buckets.keys().cloned().collect();

        match self.sampling_strategy {
            BucketSamplingStrategy::RandomWithinBuckets => {
                let mut rng = rng();
                for entries in self.buckets.values_mut() {
                    entries.shuffle(&mut rng);
                }
            }
            BucketSamplingStrategy::FullyRandom => {
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

    fn next_entry(&mut self, py: Python) -> PyResult<Option<PyTarFileEntry>> {
        if self.lazy_load {
            self.ensure_buckets_available(py)?;
        }

        match self.sampling_strategy {
            BucketSamplingStrategy::Sequential => self.next_sequential(py),
            BucketSamplingStrategy::RandomWithinBuckets => self.next_random_within_buckets(py),
            BucketSamplingStrategy::FullyRandom => self.next_fully_random(py),
        }
    }

    fn next_sequential(&mut self, py: Python) -> PyResult<Option<PyTarFileEntry>> {
        loop {
            if self.current_bucket_idx >= self.bucket_keys.len() {
                if self.lazy_load && self.next_shard_to_process < self.processed_shards.len() {
                    self.process_shard_batch(py)?;
                    continue;
                }
                return Ok(None);
            }

            let bucket_key = &self.bucket_keys[self.current_bucket_idx];
            if let Some(entries) = self.buckets.get(bucket_key) {
                if self.current_entry_idx >= entries.len() {
                    self.current_bucket_idx += 1;
                    self.current_entry_idx = 0;
                    continue;
                }

                let entry = &entries[self.current_entry_idx];
                self.current_entry_idx += 1;

                return self.load_entry(py, entry);
            } else {
                self.current_bucket_idx += 1;
                self.current_entry_idx = 0;
            }
        }
    }

    fn next_random_within_buckets(&mut self, py: Python) -> PyResult<Option<PyTarFileEntry>> {
        self.next_sequential(py)
    }

    fn next_fully_random(&mut self, py: Python) -> PyResult<Option<PyTarFileEntry>> {
        if let Some(randomized) = &self.randomized_entries {
            if self.random_position >= randomized.len() {
                return Ok(None);
            }

            let (bucket_key, entry_idx) = &randomized[self.random_position];
            self.random_position += 1;

            let entries = self.buckets.get(bucket_key).unwrap();
            let entry = &entries[*entry_idx];

            self.load_entry(py, entry)
        } else {
            Ok(None)
        }
    }

    fn load_entry(&self, py: Python, entry: &BucketEntry) -> PyResult<Option<PyTarFileEntry>> {
        let tar_loader = self.tar_loader.borrow(py);
        let loaded_entry =
            tar_loader.load_file_by_info(entry.shard_idx, &entry.filename, &entry.file_info)?;

        Ok(Some(loaded_entry))
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
