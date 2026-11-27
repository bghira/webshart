use crate::discovery::{DatasetDiscovery, DiscoveredDataset, PyDiscoveredDataset};
use crate::error::{Result, WebshartError};
use crate::metadata::{CaptionValue, ShardMetadata};
use crate::prefetch::{PrefetchIterator, Sample, ShardOrder};
use crate::FileInfo;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use regex::Regex;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

type IndexedFileEntry = (usize, String, FileInfo);

pub(crate) enum PathMatcher {
    Contains(String),
    Exact(String),
    Glob(String),
    Regex(Regex),
}

impl PathMatcher {
    fn new(pattern: String, mode: &str) -> Result<Self> {
        match mode {
            "contains" => Ok(Self::Contains(pattern)),
            "exact" => Ok(Self::Exact(pattern)),
            "glob" => Ok(Self::Glob(pattern)),
            "regex" => {
                Ok(Self::Regex(Regex::new(&pattern).map_err(|err| {
                    WebshartError::InvalidShardFormat(err.to_string())
                })?))
            }
            "auto" => {
                if let Some(regex_pattern) = pattern.strip_prefix("re:") {
                    Ok(Self::Regex(Regex::new(regex_pattern).map_err(|err| {
                        WebshartError::InvalidShardFormat(err.to_string())
                    })?))
                } else if pattern.contains('*') || pattern.contains('?') {
                    Ok(Self::Glob(pattern))
                } else {
                    Ok(Self::Contains(pattern))
                }
            }
            _ => Err(WebshartError::InvalidShardFormat(
                "path_filter_mode must be 'auto', 'contains', 'glob', 'regex', or 'exact'"
                    .to_string(),
            )),
        }
    }

    fn is_match(&self, value: &str) -> bool {
        match self {
            Self::Contains(pattern) => value.contains(pattern),
            Self::Exact(pattern) => value == pattern,
            Self::Glob(pattern) => glob_match(pattern, value),
            Self::Regex(pattern) => pattern.is_match(value),
        }
    }
}

fn build_path_matchers(patterns: Option<Vec<String>>, mode: &str) -> Result<Vec<PathMatcher>> {
    patterns
        .unwrap_or_default()
        .into_iter()
        .filter(|pattern| !pattern.is_empty())
        .map(|pattern| PathMatcher::new(pattern, mode))
        .collect()
}

fn filename_matches_path_filter(
    filename: &str,
    include: &[PathMatcher],
    exclude: &[PathMatcher],
) -> bool {
    let basename = Path::new(filename)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(filename);
    let matches_any =
        |matcher: &PathMatcher| matcher.is_match(filename) || matcher.is_match(basename);

    if !include.is_empty() && !include.iter().any(matches_any) {
        return false;
    }
    if exclude.iter().any(matches_any) {
        return false;
    }
    true
}

fn glob_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.as_bytes();
    let value = value.as_bytes();
    let (mut pattern_idx, mut value_idx) = (0usize, 0usize);
    let mut star_idx: Option<usize> = None;
    let mut star_value_idx = 0usize;

    while value_idx < value.len() {
        if pattern_idx < pattern.len()
            && (pattern[pattern_idx] == b'?' || pattern[pattern_idx] == value[value_idx])
        {
            pattern_idx += 1;
            value_idx += 1;
        } else if pattern_idx < pattern.len() && pattern[pattern_idx] == b'*' {
            star_idx = Some(pattern_idx);
            star_value_idx = value_idx;
            pattern_idx += 1;
        } else if let Some(star) = star_idx {
            pattern_idx = star + 1;
            star_value_idx += 1;
            value_idx = star_value_idx;
        } else {
            return false;
        }
    }

    while pattern_idx < pattern.len() && pattern[pattern_idx] == b'*' {
        pattern_idx += 1;
    }
    pattern_idx == pattern.len()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RemoteByteRange {
    start: u64,
    end: u64,
}

fn plan_remote_byte_ranges(
    file_entries: &[IndexedFileEntry],
    max_gap: u64,
    max_span: u64,
) -> Vec<RemoteByteRange> {
    let mut entries: Vec<&IndexedFileEntry> = file_entries
        .iter()
        .filter(|(_, _, file_info)| file_info.length > 0)
        .collect();
    entries.sort_by_key(|(file_idx, _, file_info)| (file_info.offset, *file_idx));

    let mut ranges: Vec<RemoteByteRange> = Vec::new();
    for (_, _, file_info) in entries {
        let entry_end = file_info.offset.saturating_add(file_info.length);
        let Some(current) = ranges.last_mut() else {
            ranges.push(RemoteByteRange {
                start: file_info.offset,
                end: entry_end,
            });
            continue;
        };

        let gap = file_info.offset.saturating_sub(current.end);
        let combined_end = current.end.max(entry_end);
        let combined_span = combined_end.saturating_sub(current.start);
        if gap > max_gap || combined_span > max_span {
            ranges.push(RemoteByteRange {
                start: file_info.offset,
                end: entry_end,
            });
        } else {
            current.end = combined_end;
        }
    }
    ranges
}

fn handle_file_read_error(filename: &str, error: WebshartError) -> Result<Vec<u8>> {
    match error {
        WebshartError::Xet(_) | WebshartError::XetHttp(_) => Err(error),
        error => {
            eprintln!("Failed to load {filename}: {error}");
            Ok(Vec::new())
        }
    }
}

#[cfg(test)]
mod remote_range_tests {
    use super::{
        handle_file_read_error, plan_remote_byte_ranges, FileInfo, IndexedFileEntry,
        RemoteByteRange, WebshartError,
    };

    #[test]
    fn file_read_errors_preserve_native_failures_and_legacy_fallback() {
        assert!(matches!(
            handle_file_read_error("sample.jpg", WebshartError::Xet("invalid chunk".into())),
            Err(WebshartError::Xet(_))
        ));
        assert!(matches!(
            handle_file_read_error("sample.jpg", WebshartError::XetHttp(403)),
            Err(WebshartError::XetHttp(403))
        ));
        assert!(
            handle_file_read_error("sample.jpg", WebshartError::RateLimited)
                .unwrap()
                .is_empty()
        );
    }

    fn entry(file_idx: usize, offset: u64, length: u64) -> IndexedFileEntry {
        (
            file_idx,
            format!("{file_idx}.bin"),
            FileInfo {
                path: None,
                offset,
                length,
                sha256: None,
                width: None,
                height: None,
                aspect: None,
                json_path: None,
                json_offset: None,
                json_length: None,
                captions: None,
                json_metadata: None,
                extra: Default::default(),
            },
        )
    }

    #[test]
    fn remote_ranges_split_large_physical_gaps_and_sort_by_offset() {
        let entries = vec![entry(0, 1_000, 100), entry(1, 0, 100), entry(2, 150, 100)];

        assert_eq!(
            plan_remote_byte_ranges(&entries, 100, 10_000),
            vec![
                RemoteByteRange { start: 0, end: 250 },
                RemoteByteRange {
                    start: 1_000,
                    end: 1_100,
                },
            ]
        );
    }

    #[test]
    fn remote_ranges_respect_maximum_combined_span() {
        let entries = vec![entry(0, 0, 100), entry(1, 150, 100)];

        assert_eq!(
            plan_remote_byte_ranges(&entries, 100, 200),
            vec![
                RemoteByteRange { start: 0, end: 100 },
                RemoteByteRange {
                    start: 150,
                    end: 250,
                },
            ]
        );
    }
}

// Import from modules
pub(crate) mod aspect_buckets;
mod batch;
mod config;
mod entry_types;
pub(crate) mod file_loading;
pub mod shard_cache;
use crate::impl_batch_iterator;
use crate::metadata::ensure_shard_metadata_with_retry;
use aspect_buckets::{calculate_bucket_key, BucketKeyType};
pub use aspect_buckets::{
    scale_dimensions_with_multiple, AspectBucketEntry, AspectBucketIterator, AspectBuckets,
};
pub use batch::{BatchIterable, BatchOperations, BatchResult, FileReadRequest, PyBatchOperations};
use config::DataLoaderConfig;
pub use entry_types::{create_tar_entry, PyTarFileEntry};
use file_loading::{create_file_loader, file_http_client};

// Generate batch iterators using the macro
impl_batch_iterator!(
    PyBatchIterator,
    "PyBatchIterator",
    PyTarDataLoader,
    PyTarFileEntry
);

// ===== TAR DATA LOADER =====
#[pyclass]
pub struct PyRangeIterator {
    loader: Py<PyTarDataLoader>,
    end: usize,
    current: usize,
}

#[pymethods]
impl PyRangeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        if slf.current >= slf.end {
            return Ok(None);
        }

        slf.current += 1;

        Python::attach(|py| {
            let mut loader = slf.loader.borrow_mut(py);
            loader.next_entry()
        })
    }
}

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
    ranges: Option<Vec<(usize, usize)>>,
    current_range_idx: usize,
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
        dataset_or_path: &Bound<'_, PyAny>,
        load_file_data: bool,
        max_file_size: u64,
        buffer_size: usize,
        hf_token: Option<String>,
        chunk_size_mb: usize,
        batch_size: Option<usize>,
    ) -> PyResult<Self> {
        let runtime =
            Arc::new(Runtime::new().map_err(|e| WebshartError::Io(std::io::Error::other(e)))?);

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
            current_range_idx: 0,
            ranges: None,
            source,
        })
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);

        let current_file_index = self.calculate_current_file_index();

        dict.set_item("current_shard", self.current_shard)?;
        dict.set_item("current_file_index", current_file_index)?;
        dict.set_item("buffer_position", self.buffer_position)?;

        self.config.to_state_dict(&dict)?;

        dict.set_item("source", &self.source)?;
        dict.set_item("metadata_source", &self.metadata_source)?;

        let dataset = self.dataset.lock().unwrap();
        dict.set_item("num_shards", dataset.num_shards())?;
        dict.set_item("is_remote", dataset.is_remote)?;
        dict.set_item("version", 4)?;

        Ok(dict.into_any().unbind())
    }

    fn load_state_dict(&mut self, state_dict: &Bound<'_, PyDict>) -> PyResult<()> {
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
        state_dict: &Bound<'_, PyDict>,
        dataset_or_path: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let config = DataLoaderConfig::from_state_dict(state_dict);

        let source_obj = Self::determine_source_from_state_dict(py, state_dict, dataset_or_path)?;
        let source = source_obj.bind(py);

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

    fn get_state_summary(&self, py: Python) -> PyResult<Py<PyAny>> {
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

        Ok(dict.into_any().unbind())
    }

    /// Create an iterator for a specific range
    pub fn iter_range(
        mut slf: PyRefMut<'_, Self>,
        start: usize,
        end: usize,
    ) -> PyResult<PyRangeIterator> {
        if start >= end {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "start must be less than end",
            ));
        }

        // Skip to start position
        slf.skip(start)?;

        Ok(PyRangeIterator {
            loader: slf.into(),
            end,
            current: start,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyTarFileEntry>> {
        slf.next_entry()
    }

    /// Read files on a background thread from the current cursor or a seeded shard order.
    #[pyo3(signature = (max_bytes, *, max_items=128, shuffle_seed=None, start_shard_position=0))]
    fn iter_prefetched(
        &self,
        py: Python<'_>,
        max_bytes: usize,
        max_items: usize,
        shuffle_seed: Option<u64>,
        start_shard_position: usize,
    ) -> PyResult<PrefetchIterator> {
        let num_shards = self.dataset.lock().unwrap().num_shards();
        if start_shard_position > num_shards
            || (shuffle_seed.is_none() && start_shard_position != 0)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "start_shard_position requires a shuffle seed and a position within the dataset",
            ));
        }
        let mut reader = Self {
            dataset: self.dataset.clone(),
            runtime: self.runtime.clone(),
            config: self.config.clone(),
            current_shard: self.current_shard,
            entry_buffer: Vec::new(),
            buffer_position: 0,
            next_file_to_load: self.calculate_current_file_index(),
            source: self.source.clone(),
            metadata_source: self.metadata_source.clone(),
            ranges: self.ranges.clone(),
            current_range_idx: self.current_range_idx,
        };
        let mut shard_order = shuffle_seed.map(|seed| {
            py.detach(|| ShardOrder::shuffled(num_shards, seed))
                .into_indices()
        });
        if let Some(order) = &mut shard_order {
            reader.current_shard = order.nth(start_shard_position).unwrap_or(num_shards);
            reader.next_file_to_load = 0;
            reader.ranges = None;
        }
        PrefetchIterator::start(max_bytes, max_items, move |bytes, items| {
            let entry = reader.next_entry_with_limits(bytes as u64, items, shard_order.as_mut())?;
            Ok(entry.map(|entry| {
                let bytes = entry.data.len();
                (Sample::Tar(entry), bytes)
            }))
        })
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

    fn will_block(&self) -> PyResult<bool> {
        // If we have entries in buffer, won't block
        if self.buffer_position < self.entry_buffer.len() {
            return Ok(false);
        }
        let dataset = self.dataset.lock().unwrap();
        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(false);
        }
        if self.current_shard >= dataset.num_shards() {
            return Ok(false);
        }
        let shard_name = dataset.shards[self.current_shard]
            .tar_path
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string();
        let cache = dataset.shard_cache.as_ref().unwrap().clone();

        drop(dataset);
        let is_cached = self.runtime.block_on(cache.is_cached(&shard_name));
        Ok(!is_cached)
    }

    fn is_shard_locked(&self, shard_name: &str) -> bool {
        let dataset = self.dataset.lock().unwrap();

        if let Some(cache) = &dataset.shard_cache {
            cache.is_shard_locked(shard_name)
        } else {
            false
        }
    }

    fn prepare_shard_by_name(&self, filename: &str) -> PyResult<bool> {
        let dataset = self.dataset.lock().unwrap();

        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(false);
        }

        let shard_idx = self.find_shard_by_filename(&dataset, filename)?;
        let shard = &dataset.shards[shard_idx];
        let tar_path = shard.tar_path.clone();
        let shard_name = tar_path.rsplit('/').next().unwrap_or(&tar_path).to_string();
        let token = dataset.get_hf_token();
        let cache = dataset.shard_cache.as_ref().unwrap().clone();

        // Drop the dataset lock before any async/await or spawn
        drop(dataset);

        // Check cache status before spawning
        let is_cached = self.runtime.block_on({
            let cache = cache.clone();
            let shard_name = shard_name.clone();
            async move { cache.is_cached(&shard_name).await }
        });
        if is_cached {
            return Ok(false);
        }

        // Clone all data needed for the async block before spawning
        let runtime = self.runtime.clone();
        let shard_name_cloned = shard_name.clone();
        let tar_path_cloned = tar_path.clone();
        let token_cloned = token.clone();
        let cache_cloned = cache.clone();

        // All clones above are Send, so nothing non-Send is captured
        runtime.spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                match cache_cloned
                    .cache_shard(&shard_name_cloned, &tar_path_cloned, token_cloned)
                    .await
                {
                    Ok(_) => {
                        println!("[webshart] Pre-cached shard: {}", shard_name_cloned);
                    }
                    Err(e) => {
                        eprintln!(
                            "[webshart] Failed to pre-cache shard {}: {}",
                            shard_name_cloned, e
                        );
                    }
                }
            })
        });

        Ok(true)
    }

    fn prepare_next_shard(&self) -> PyResult<bool> {
        let dataset = self.dataset.lock().unwrap();

        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(false);
        }

        if self.current_shard >= dataset.num_shards() {
            return Ok(false);
        }

        let shard = &dataset.shards[self.current_shard];
        let tar_path = shard.tar_path.clone();
        let shard_name = tar_path.rsplit('/').next().unwrap_or(&tar_path).to_string();
        let token = dataset.get_hf_token();
        let cache = dataset.shard_cache.as_ref().unwrap().clone();

        // Drop the dataset lock before any async/await or spawn
        drop(dataset);

        // Check cache status before spawning
        let is_cached = self.runtime.block_on({
            let cache = cache.clone();
            let shard_name = shard_name.clone();
            async move { cache.is_cached(&shard_name).await }
        });
        if is_cached {
            return Ok(false);
        }

        // Clone all data needed for the async block before spawning
        let runtime = self.runtime.clone();
        let shard_name_cloned = shard_name.clone();
        let tar_path_cloned = tar_path.clone();
        let token_cloned = token.clone();
        let cache_cloned = cache.clone();

        // All clones above are Send, so nothing non-Send is captured
        runtime.spawn_blocking(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                match cache_cloned
                    .cache_shard(&shard_name_cloned, &tar_path_cloned, token_cloned)
                    .await
                {
                    Ok(_) => {
                        println!("[webshart] Pre-cached shard: {}", shard_name_cloned);
                    }
                    Err(e) => {
                        eprintln!(
                            "[webshart] Failed to pre-cache shard {}: {}",
                            shard_name_cloned, e
                        );
                    }
                }
            })
        });

        Ok(true)
    }

    /// Get information about which shard will be loaded next
    fn get_next_shard_info(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.buffer_position < self.entry_buffer.len() {
            return Ok(None); // Still have buffered entries
        }

        let dataset = self.dataset.lock().unwrap();

        if self.current_shard >= dataset.num_shards() {
            return Ok(None); // No more shards
        }

        let shard = &dataset.shards[self.current_shard];
        let dict = PyDict::new(py);

        dict.set_item("index", self.current_shard)?;
        dict.set_item("name", &shard.name)?;
        dict.set_item("tar_path", &shard.tar_path)?;

        if let Some(cache) = dataset.shard_cache.as_ref().filter(|_| dataset.is_remote) {
            let shard_name = shard
                .tar_path
                .rsplit('/')
                .next()
                .unwrap_or(&shard.tar_path)
                .to_string();
            let cache = cache.clone();
            drop(dataset);

            let is_cached = self.runtime.block_on(cache.is_cached(&shard_name));
            dict.set_item("is_cached", is_cached)?;
        } else {
            dict.set_item("is_cached", true)?; // Local files are always "cached"
        }

        Ok(Some(dict.into_any().unbind()))
    }

    /// Get information about a specific shard's cache status
    fn get_shard_cache_status(&self, py: Python, filename: &str) -> PyResult<Py<PyAny>> {
        let dataset = self.dataset.lock().unwrap();

        let shard_idx = self.find_shard_by_filename(&dataset, filename)?;
        let shard = &dataset.shards[shard_idx];

        let dict = PyDict::new(py);
        dict.set_item("index", shard_idx)?;
        dict.set_item("name", &shard.name)?;
        dict.set_item("tar_path", &shard.tar_path)?;

        if let Some(cache) = dataset.shard_cache.as_ref().filter(|_| dataset.is_remote) {
            let shard_filename = shard
                .tar_path
                .rsplit('/')
                .next()
                .unwrap_or(&shard.tar_path)
                .to_string();
            let cache = cache.clone();
            drop(dataset);

            let (is_cached, size) = self.runtime.block_on(async {
                let is_cached = cache.is_cached(&shard_filename).await;
                let size = cache
                    .get_cached_file_size(&shard_filename)
                    .await
                    .unwrap_or(0);
                (is_cached, size)
            });
            dict.set_item("is_cached", is_cached)?;
            dict.set_item("cur_filesize", size)?;
        } else {
            drop(dataset);
            dict.set_item("is_cached", true)?; // Local files are always "cached"
        }

        Ok(dict.into_any().unbind())
    }

    #[pyo3(signature = (lookahead=5))]
    fn get_lookahead_cache_status(&self, py: Python, lookahead: usize) -> PyResult<Py<PyAny>> {
        let dataset = self.dataset.lock().unwrap();
        let list = PyList::empty(py);

        let start_shard = self.current_shard;
        let end_shard = (start_shard + lookahead).min(dataset.num_shards());

        for shard_idx in start_shard..end_shard {
            let shard = &dataset.shards[shard_idx];
            let dict = PyDict::new(py);

            dict.set_item("index", shard_idx)?;
            dict.set_item("name", &shard.name)?;

            if let Some(cache) = dataset.shard_cache.as_ref().filter(|_| dataset.is_remote) {
                let shard_name = shard.tar_path.rsplit('/').next().unwrap_or(&shard.tar_path);
                let cache = cache.clone();
                let is_cached = self.runtime.block_on(cache.is_cached(shard_name));
                dict.set_item("is_cached", is_cached)?;
            } else {
                dict.set_item("is_cached", true)?;
            }

            list.append(dict)?;
        }

        Ok(list.into_any().unbind())
    }

    #[pyo3(signature = (num_shards=1))]
    fn prepare_shards_ahead(&self, num_shards: usize) -> PyResult<Vec<String>> {
        let dataset = self.dataset.lock().unwrap();
        let mut started_caching = Vec::new();

        if !dataset.is_remote || dataset.shard_cache.is_none() {
            return Ok(started_caching);
        }

        let start_shard = self.current_shard;
        let end_shard = (start_shard + num_shards).min(dataset.num_shards());

        for shard_idx in start_shard..end_shard {
            let shard = &dataset.shards[shard_idx];
            let tar_path = shard.tar_path.clone();
            let shard_name = tar_path.rsplit('/').next().unwrap_or(&tar_path).to_string();
            let token = dataset.get_hf_token();
            let cache = dataset.shard_cache.as_ref().unwrap().clone();

            // Check if already cached
            let is_cached = self.runtime.block_on(cache.is_cached(&shard_name));
            if !is_cached {
                // Spawn async task to cache the shard
                let runtime = self.runtime.clone();
                let shard_name_clone = shard_name.clone();
                let tar_path_clone = tar_path.clone();
                let token_clone = token.clone();
                let cache_clone = cache.clone();
                runtime.spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    rt.block_on(async {
                        match cache_clone
                            .cache_shard(&shard_name_clone, &tar_path_clone, token_clone)
                            .await
                        {
                            Ok(_) => {
                                println!("[webshart] Pre-cached shard: {}", shard_name_clone);
                            }
                            Err(e) => {
                                eprintln!(
                                    "[webshart] Failed to pre-cache shard {}: {}",
                                    shard_name_clone, e
                                );
                            }
                        }
                    })
                });

                started_caching.push(shard_name);
            }
        }

        drop(dataset);
        Ok(started_caching)
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
    fn current_shard_filename(&self) -> String {
        let dataset = self.dataset.lock().unwrap();
        if self.current_shard < dataset.shards.len() {
            let shard = &dataset.shards[self.current_shard];
            shard
                .tar_path
                .rsplit('/')
                .next()
                .unwrap_or(&shard.tar_path)
                .to_string()
        } else {
            String::new()
        }
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

    fn set_ranges(&mut self, ranges: Vec<(usize, usize)>) -> PyResult<()> {
        // Validate ranges
        for (start, end) in &ranges {
            if start >= end {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid range: start {} must be less than end {}",
                    start, end
                )));
            }
        }

        self.ranges = Some(ranges);
        self.current_range_idx = 0;

        // Jump to first range start
        if let Some(ranges) = &self.ranges {
            if !ranges.is_empty() {
                self.skip(ranges[0].0)?;
            }
        }

        Ok(())
    }

    fn get_metadata(&self, shard_idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let dict = PyDict::new(py);

        if let Some(metadata) = &shard.metadata {
            for (filename, file_info) in metadata
                .iter_files()
                .filter(|(_, file_info)| file_info.length <= self.config.max_file_size)
            {
                let file_dict = pythonize::pythonize(py, &file_info)?;
                dict.set_item(filename, file_dict)?;
            }
        }

        Ok(dict.into_any().unbind())
    }

    /// List visible samples with their stable, unfiltered sample indices.
    fn list_samples_in_shard(&self, shard_idx: usize, py: Python) -> PyResult<Py<PyAny>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let list = PyList::empty(py);
        if let Some(metadata) = &shard.metadata {
            for (sample_idx, (filename, _)) in metadata
                .sample_range(0, metadata.num_samples())
                .into_iter()
                .enumerate()
                .filter(|(_, (_, file_info))| file_info.length <= self.config.max_file_size)
            {
                let sample = PyDict::new(py);
                sample.set_item("sample_idx", sample_idx)?;
                sample.set_item("filename", filename)?;
                list.append(sample)?;
            }
        }

        Ok(list.into_any().unbind())
    }

    #[pyo3(signature = (shard_idx, path_include=None, path_exclude=None, path_filter_mode="auto"))]
    fn list_samples_in_shard_filtered(
        &self,
        shard_idx: usize,
        py: Python,
        path_include: Option<Vec<String>>,
        path_exclude: Option<Vec<String>>,
        path_filter_mode: &str,
    ) -> PyResult<Py<PyAny>> {
        let include = build_path_matchers(path_include, path_filter_mode)?;
        let exclude = build_path_matchers(path_exclude, path_filter_mode)?;
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;

        let list = PyList::empty(py);
        if let Some(metadata) = &shard.metadata {
            for (sample_idx, (filename, _file_info)) in metadata
                .sample_range(0, metadata.num_samples())
                .into_iter()
                .enumerate()
                .filter(|(_, (filename, file_info))| {
                    file_info.length <= self.config.max_file_size
                        && filename_matches_path_filter(filename, &include, &exclude)
                })
            {
                let sample = PyDict::new(py);
                sample.set_item("sample_idx", sample_idx)?;
                sample.set_item("filename", filename)?;
                list.append(sample)?;
            }
        }

        Ok(list.into_any().unbind())
    }

    /// Load a logical sample, or return None when it exceeds max_file_size.
    #[pyo3(signature = (shard_idx, sample_idx, *, load_sidecars=true))]
    fn load_sample(
        &self,
        py: Python,
        shard_idx: usize,
        sample_idx: usize,
        load_sidecars: bool,
    ) -> PyResult<Option<PyTarFileEntry>> {
        py.detach(|| {
            let mut dataset = self.dataset.lock().unwrap();
            ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

            let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
                WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
            })?;

            let metadata = shard.metadata.as_ref().ok_or_else(|| {
                WebshartError::MetadataNotFound("Metadata not loaded".to_string())
            })?;

            let (filename, file_info) =
                metadata.get_sample_by_index(sample_idx).ok_or_else(|| {
                    WebshartError::InvalidShardFormat(format!(
                        "Sample index {} out of range for shard {}",
                        sample_idx, shard_idx
                    ))
                })?;
            let txt_sidecar = metadata
                .get_txt_sidecar_by_sample_index(sample_idx)
                .map(|(_, file_info)| file_info);

            let tar_path = shard.tar_path.clone();
            let is_remote = dataset.is_remote;
            let token = if is_remote {
                dataset.get_hf_token()
            } else {
                None
            };
            drop(dataset);

            if file_info.length > self.config.max_file_size {
                return Ok(None);
            }

            let data = if self.config.load_file_data {
                self.load_single_file_data(&tar_path, &file_info, is_remote, token.clone())?
            } else {
                Vec::new()
            };

            let mut entry = create_tar_entry(
                filename,
                &file_info,
                data,
                Some(shard_idx),
                Some(sample_idx),
            );
            if !load_sidecars {
                return Ok(Some(entry));
            }
            entry.json_data =
                self.load_json_sidecar_bytes(&tar_path, &file_info, is_remote, token.clone())?;
            entry.captions = self.load_caption_value(
                &tar_path,
                &file_info,
                txt_sidecar.as_ref(),
                entry.json_data.as_deref(),
                is_remote,
                token,
            )?;
            Ok(Some(entry))
        })
    }

    /// Load the first caption for a logical sample from metadata or a paired sidecar.
    fn load_caption(
        &self,
        py: Python,
        shard_idx: usize,
        sample_idx: usize,
    ) -> PyResult<Option<Py<PyAny>>> {
        let mut dataset = self.dataset.lock().unwrap();
        ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;

        let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!("Shard index {} out of range", shard_idx))
        })?;
        let metadata = shard
            .metadata
            .as_ref()
            .ok_or_else(|| WebshartError::MetadataNotFound("Metadata not loaded".to_string()))?;
        let (_filename, file_info) = metadata.get_sample_by_index(sample_idx).ok_or_else(|| {
            WebshartError::InvalidShardFormat(format!(
                "Sample index {} out of range for shard {}",
                sample_idx, shard_idx
            ))
        })?;
        let txt_sidecar = metadata
            .get_txt_sidecar_by_sample_index(sample_idx)
            .map(|(_, file_info)| file_info);
        let tar_path = shard.tar_path.clone();
        let is_remote = dataset.is_remote;
        let token = if is_remote {
            dataset.get_hf_token()
        } else {
            None
        };
        drop(dataset);

        if file_info.length > self.config.max_file_size {
            return Ok(None);
        }

        let captions = self.load_caption_value(
            &tar_path,
            &file_info,
            txt_sidecar.as_ref(),
            None,
            is_remote,
            token,
        )?;
        captions
            .as_ref()
            .and_then(CaptionValue::first)
            .map(|caption| pythonize::pythonize(py, caption).map(Bound::unbind))
            .transpose()
            .map_err(Into::into)
    }

    /// Fold sidecar captions into normal webshart metadata JSON files.
    #[pyo3(signature = (destination=None, shard_indices=None))]
    fn coalesce_caption_metadata(
        &self,
        py: Python,
        destination: Option<String>,
        shard_indices: Option<Vec<usize>>,
    ) -> PyResult<Py<PyAny>> {
        let (num_shards, cache_dir) = {
            let dataset = self.dataset.lock().unwrap();
            (dataset.num_shards(), dataset.metadata_cache_dir())
        };
        let persist_to_cache = destination.is_none();
        let destination = match destination {
            Some(path) => Path::new(&path).to_path_buf(),
            None => cache_dir.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "enable_metadata_cache() first or provide destination",
                )
            })?,
        };
        let shard_indices = shard_indices.unwrap_or_else(|| (0..num_shards).collect());
        let mut output_paths = Vec::with_capacity(shard_indices.len());
        let mut captioned_samples = 0usize;
        let mut coalesced_samples = 0usize;

        for shard_idx in shard_indices {
            let (shard_name, tar_path, is_remote, token, mut metadata) = {
                let mut dataset = self.dataset.lock().unwrap();
                ensure_shard_metadata_with_retry(&mut dataset, shard_idx)?;
                let shard = dataset.shards.get(shard_idx).ok_or_else(|| {
                    WebshartError::InvalidShardFormat(format!(
                        "Shard index {} out of range",
                        shard_idx
                    ))
                })?;
                let metadata = shard.metadata.as_deref().cloned().ok_or_else(|| {
                    WebshartError::MetadataNotFound("Metadata not loaded".to_string())
                })?;
                (
                    shard.name.clone(),
                    shard.tar_path.clone(),
                    dataset.is_remote,
                    dataset.get_hf_token(),
                    metadata,
                )
            };

            for sample_idx in 0..metadata.num_samples() {
                let Some((_filename, file_info)) = metadata.get_sample_by_index(sample_idx) else {
                    continue;
                };
                let had_caption = file_info.captions.is_some();
                let txt_sidecar = metadata
                    .get_txt_sidecar_by_sample_index(sample_idx)
                    .map(|(_, file_info)| file_info);
                if let Some(captions) = self.load_caption_value(
                    &tar_path,
                    &file_info,
                    txt_sidecar.as_ref(),
                    None,
                    is_remote,
                    token.clone(),
                )? {
                    captioned_samples += 1;
                    coalesced_samples += usize::from(!had_caption);
                    metadata.set_sample_captions(sample_idx, captions);
                }
            }

            let output_path = Self::metadata_output_path(&destination, &shard_name)?;
            if !persist_to_cache {
                Self::write_metadata_file(&output_path, &metadata)?;
            }
            self.dataset.lock().unwrap().replace_shard_metadata(
                shard_idx,
                metadata,
                (!persist_to_cache).then(|| output_path.clone()),
            )?;
            output_paths.push(output_path.to_string_lossy().to_string());
        }

        let result = PyDict::new(py);
        result.set_item("shards", output_paths.len())?;
        result.set_item("captioned_samples", captioned_samples)?;
        result.set_item("coalesced_samples", coalesced_samples)?;
        result.set_item("files", output_paths)?;
        Ok(result.into_any().unbind())
    }

    fn load_sample_json(
        &self,
        py: Python,
        shard_idx: usize,
        sample_idx: usize,
    ) -> PyResult<Option<Py<PyBytes>>> {
        let Some(entry) = self.load_sample(py, shard_idx, sample_idx, true)? else {
            return Ok(None);
        };
        Ok(entry
            .json_data
            .as_ref()
            .map(|data| PyBytes::new(py, data).unbind()))
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
    ) -> PyResult<Vec<Py<PyAny>>> {
        let results = self.get_aspect_buckets_for_shards(
            shard_indices,
            key,
            target_pixel_area,
            Some(target_resolution_multiple),
            round_to,
        )?;

        let py_results: Vec<Py<PyAny>> = results
            .into_iter()
            .map(|bucket| self.aspect_buckets_to_py_dict(py, bucket))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py_results)
    }

    #[pyo3(signature = (shard_indices, key="aspect", target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2)))]
    pub fn list_shard_sample_aspect_buckets(
        &self,
        py: Python,
        shard_indices: Vec<usize>,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: u32,
        round_to: Option<usize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let results = self.get_sample_aspect_buckets_for_shards(
            shard_indices,
            key,
            target_pixel_area,
            Some(target_resolution_multiple),
            round_to,
            None,
        )?;

        let py_results: Vec<Py<PyAny>> = results
            .into_iter()
            .map(|bucket| self.aspect_buckets_to_py_dict(py, bucket))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(py_results)
    }

    #[pyo3(signature = (shard_indices, key="aspect", target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2), path_include=None, path_exclude=None, path_filter_mode="auto"))]
    #[allow(clippy::too_many_arguments)] // Preserve the Python keyword argument interface.
    pub fn list_shard_sample_aspect_buckets_filtered(
        &self,
        py: Python,
        shard_indices: Vec<usize>,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: u32,
        round_to: Option<usize>,
        path_include: Option<Vec<String>>,
        path_exclude: Option<Vec<String>>,
        path_filter_mode: &str,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let include = build_path_matchers(path_include, path_filter_mode)?;
        let exclude = build_path_matchers(path_exclude, path_filter_mode)?;
        let results = self.get_sample_aspect_buckets_for_shards(
            shard_indices,
            key,
            target_pixel_area,
            Some(target_resolution_multiple),
            round_to,
            Some((&include, &exclude)),
        )?;

        let py_results: Vec<Py<PyAny>> = results
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
    ) -> PyResult<Py<PyAny>> {
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

        Py::new(py, iterator).map(Py::into_any)
    }
}

impl PyTarDataLoader {
    fn metadata_output_path(destination: &Path, shard_name: &str) -> Result<PathBuf> {
        let relative = Path::new(shard_name);
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(WebshartError::InvalidShardFormat(format!(
                "Unsafe shard name for metadata export: {shard_name}"
            )));
        }
        Ok(destination.join(relative).with_extension("json"))
    }

    fn write_metadata_file(path: &Path, metadata: &ShardMetadata) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temp_path = path.with_extension("json.tmp");
        fs::write(&temp_path, serde_json::to_vec_pretty(metadata)?)?;
        fs::rename(temp_path, path)?;
        Ok(())
    }

    fn caption_from_txt_bytes(data: Vec<u8>) -> PyResult<Option<CaptionValue>> {
        let caption = String::from_utf8(data).map_err(|error| {
            WebshartError::InvalidShardFormat(format!(
                "Caption sidecar is not valid UTF-8: {error}"
            ))
        })?;
        Ok(CaptionValue::from_text(&caption))
    }

    fn load_caption_value(
        &self,
        tar_path: &str,
        file_info: &FileInfo,
        txt_sidecar: Option<&FileInfo>,
        json_data: Option<&[u8]>,
        is_remote: bool,
        token: Option<String>,
    ) -> PyResult<Option<CaptionValue>> {
        if let Some(captions) = &file_info.captions {
            return Ok(Some(captions.clone()));
        }

        if let Some(txt_info) = txt_sidecar {
            if txt_info.length > self.config.max_file_size {
                return Ok(None);
            }
            let data = self.load_single_file_data(tar_path, txt_info, is_remote, token)?;
            return Self::caption_from_txt_bytes(data);
        }

        let owned_json_data;
        let json_data = if let Some(data) = json_data {
            Some(data)
        } else {
            owned_json_data =
                self.load_json_sidecar_bytes(tar_path, file_info, is_remote, token)?;
            owned_json_data.as_deref()
        };

        Ok(json_data
            .and_then(|data| serde_json::from_slice(data).ok())
            .as_ref()
            .and_then(ShardMetadata::extract_caption_value))
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

        let mut buckets: BTreeMap<String, Vec<AspectBucketEntry>> = BTreeMap::new();

        for (filename, file_info) in metadata.iter_files() {
            if file_info.length > self.config.max_file_size {
                continue;
            }
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

                buckets
                    .entry(bucket_key)
                    .or_default()
                    .push(AspectBucketEntry {
                        filename: filename.to_string(),
                        file_info,
                        original_size,
                        sample_idx: None,
                    });
            }
        }

        Ok(AspectBuckets {
            buckets,
            shard_idx,
            shard_name,
        })
    }

    pub(crate) fn get_shard_sample_aspect_buckets_internal(
        &self,
        shard_idx: usize,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: Option<u32>,
        round_to: Option<usize>,
        path_filter: Option<(&[PathMatcher], &[PathMatcher])>,
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

        let mut buckets: BTreeMap<String, Vec<AspectBucketEntry>> = BTreeMap::new();

        for (sample_idx, (filename, file_info)) in metadata
            .sample_range(0, metadata.num_samples())
            .into_iter()
            .enumerate()
        {
            if file_info.length > self.config.max_file_size {
                continue;
            }
            if let Some((include, exclude)) = path_filter {
                if !filename_matches_path_filter(&filename, include, exclude) {
                    continue;
                }
            }
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

                buckets
                    .entry(bucket_key)
                    .or_default()
                    .push(AspectBucketEntry {
                        filename,
                        file_info,
                        original_size,
                        sample_idx: Some(sample_idx),
                    });
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
        self.entry_buffer
            .get(self.buffer_position)
            .and_then(|entry| entry.file_idx)
            .unwrap_or(self.next_file_to_load)
    }

    fn calculate_files_processed(
        &self,
        dataset: &mut DiscoveredDataset,
        current_file_index_in_shard: usize,
    ) -> PyResult<usize> {
        let mut files_processed = 0;
        for i in 0..self.current_shard {
            files_processed += Self::shard_file_count(dataset, i)?;
        }
        files_processed += current_file_index_in_shard;
        Ok(files_processed)
    }

    fn calculate_global_file_index(&self) -> PyResult<usize> {
        let mut dataset = self.dataset.lock().unwrap();
        let current_file_index_in_shard = self.calculate_current_file_index();
        self.calculate_files_processed(&mut dataset, current_file_index_in_shard)
    }

    fn shard_file_count(dataset: &mut DiscoveredDataset, index: usize) -> Result<usize> {
        if let Some(summary) = dataset
            .shards
            .get(index)
            .and_then(|shard| shard.metadata_summary)
        {
            return Ok(summary.num_files);
        }
        ensure_shard_metadata_with_retry(dataset, index)?;
        Ok(dataset.shards[index].metadata_summary.unwrap().num_files)
    }

    fn find_shard_for_index(&self, idx: usize) -> PyResult<(usize, usize)> {
        let mut remaining = idx;
        let mut dataset = self.dataset.lock().unwrap();
        let num_shards = dataset.num_shards();

        for shard_idx in 0..num_shards {
            let num_files = Self::shard_file_count(&mut dataset, shard_idx)?;
            if remaining < num_files {
                return Ok((shard_idx, remaining));
            }
            remaining -= num_files;
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
        state_dict: &Bound<'py, PyDict>,
        dataset_or_path: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(dataset_or_path) = dataset_or_path {
            return Ok(dataset_or_path.clone().unbind());
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
                py.detach(|| {
                    Runtime::new()?.block_on(discovery.discover_huggingface(&source_str, None))
                })?
            };

            let py_dataset = Py::new(py, PyDiscoveredDataset { inner: dataset })?;
            return Ok(py_dataset.into_any());
        }

        Ok(PyString::new(py, &source_str).into_any().unbind())
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

    fn get_sample_aspect_buckets_for_shards(
        &self,
        shard_indices: Vec<usize>,
        key: &str,
        target_pixel_area: Option<u32>,
        target_resolution_multiple: Option<u32>,
        round_to: Option<usize>,
        path_filter: Option<(&[PathMatcher], &[PathMatcher])>,
    ) -> PyResult<Vec<AspectBuckets>> {
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| {
                let rt = tokio::runtime::Handle::current();
                rt.block_on(async {
                    let mut results = Vec::new();

                    for shard_idx in shard_indices {
                        match self.get_shard_sample_aspect_buckets_internal(
                            shard_idx,
                            key,
                            target_pixel_area,
                            target_resolution_multiple,
                            round_to,
                            path_filter,
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
                    match self.get_shard_sample_aspect_buckets_internal(
                        shard_idx,
                        key,
                        target_pixel_area,
                        target_resolution_multiple,
                        round_to,
                        path_filter,
                    ) {
                        Ok(buckets) => results.push(buckets),
                        Err(e) => return Err(e),
                    }
                }

                Ok(results)
            })
        }
    }

    fn aspect_buckets_to_py_dict(&self, py: Python, bucket: AspectBuckets) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("shard_idx", bucket.shard_idx)?;
        dict.set_item("shard_name", bucket.shard_name)?;

        let buckets_dict = PyDict::new(py);
        for (key, files) in bucket.buckets {
            let files_list = PyList::new(
                py,
                files.into_iter().map(|entry| {
                    let file_dict = PyDict::new(py);
                    file_dict.set_item("filename", entry.filename).unwrap();
                    file_dict
                        .set_item("offset", entry.file_info.offset)
                        .unwrap();
                    file_dict.set_item("size", entry.file_info.length).unwrap();
                    if let Some(sample_idx) = entry.sample_idx {
                        file_dict.set_item("sample_idx", sample_idx).unwrap();
                    }
                    if let Some(w) = entry.file_info.width {
                        file_dict.set_item("width", w).unwrap();
                    }
                    if let Some(h) = entry.file_info.height {
                        file_dict.set_item("height", h).unwrap();
                    }
                    if let Some(a) = entry.file_info.aspect {
                        file_dict.set_item("aspect", a).unwrap();
                    }
                    if let Some((orig_w, orig_h)) = entry.original_size {
                        let orig_list = PyList::new(py, [orig_w, orig_h]).unwrap();
                        file_dict.set_item("original_size", orig_list).unwrap();
                    }
                    file_dict
                }),
            )?;
            buckets_dict.set_item(key, files_list)?;
        }

        dict.set_item("buckets", buckets_dict)?;
        Ok(dict.into_any().unbind())
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
        if let Some(cache) = dataset.shard_cache.as_ref().filter(|_| is_remote) {
            let shard_name = tar_path.rsplit('/').next().unwrap_or(tar_path);
            let cache = cache.clone();
            drop(dataset); // Release lock before async operation

            // Cache and acquire the read lock without an eviction race between
            // those operations.
            let cached = self.runtime.block_on(cache.cache_shard_for_reading(
                shard_name,
                tar_path,
                token.clone(),
            ));
            if let Err(WebshartError::Xet(_) | WebshartError::XetHttp(_)) = &cached {
                return Err(cached.err().unwrap());
            }
            if let Ok((cached_path, _lock)) = cached {
                let loader = create_file_loader(
                    &cached_path.to_string_lossy(),
                    false,
                    None,
                    self.runtime.clone(),
                );

                return loader.load_file(file_info);
            }
        } else {
            drop(dataset);
        }

        // Fallback to original behavior (no locking needed for non-cached files)
        let loader = create_file_loader(tar_path, is_remote, token, self.runtime.clone());
        loader.load_file(file_info)
    }

    fn load_json_sidecar_bytes(
        &self,
        tar_path: &str,
        file_info: &FileInfo,
        is_remote: bool,
        token: Option<String>,
    ) -> PyResult<Option<Vec<u8>>> {
        if let (Some(offset), Some(length)) = (file_info.json_offset, file_info.json_length) {
            if length > self.config.max_file_size {
                return Ok(None);
            }
            let json_info = FileInfo {
                path: file_info.json_path.clone(),
                offset,
                length,
                sha256: None,
                width: None,
                height: None,
                aspect: None,
                json_path: None,
                json_offset: None,
                json_length: None,
                captions: None,
                json_metadata: None,
                extra: Default::default(),
            };

            let data = self.load_single_file_data(tar_path, &json_info, is_remote, token)?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    fn next_entry(&mut self) -> PyResult<Option<PyTarFileEntry>> {
        self.next_entry_with_limits(u64::MAX, usize::MAX, None)
    }

    fn next_entry_with_limits(
        &mut self,
        max_bytes: u64,
        max_entries: usize,
        shard_order: Option<&mut std::vec::IntoIter<usize>>,
    ) -> PyResult<Option<PyTarFileEntry>> {
        if let Some(ranges) = &self.ranges {
            // Check if we're within current range
            let current_idx = self.calculate_global_file_index()?;

            if self.current_range_idx >= ranges.len() {
                return Ok(None);
            }

            let (_, range_end) = ranges[self.current_range_idx];

            // If we've exceeded current range, move to next
            if current_idx >= range_end {
                self.current_range_idx += 1;

                // Skip to next range start if available
                if self.current_range_idx < ranges.len() {
                    let (next_start, _) = ranges[self.current_range_idx];
                    self.skip(next_start)?;
                } else {
                    return Ok(None);
                }
            }
        }
        if self.buffer_position < self.entry_buffer.len() {
            return Ok(self.take_buffered_entry());
        }

        self.refill_buffer(max_bytes, max_entries, shard_order)?;

        if self.buffer_position < self.entry_buffer.len() {
            Ok(self.take_buffered_entry())
        } else {
            Ok(None)
        }
    }

    fn take_buffered_entry(&mut self) -> Option<PyTarFileEntry> {
        if self.buffer_position >= self.entry_buffer.len() {
            return None;
        }

        let entry = &mut self.entry_buffer[self.buffer_position];
        let result = PyTarFileEntry {
            path: std::mem::take(&mut entry.path),
            offset: entry.offset,
            size: entry.size,
            data: std::mem::take(&mut entry.data),
            width: entry.width,
            height: entry.height,
            aspect: entry.aspect,
            json_path: entry.json_path.clone(),
            json_data: entry.json_data.clone(),
            captions: entry.captions.clone(),
            json_metadata: entry.json_metadata.clone(),
            extra: entry.extra.clone(),
            shard_idx: entry.shard_idx,
            file_idx: entry.file_idx,
        };
        self.buffer_position += 1;
        Some(result)
    }

    fn refill_buffer(
        &mut self,
        max_bytes: u64,
        max_entries: usize,
        mut shard_order: Option<&mut std::vec::IntoIter<usize>>,
    ) -> PyResult<()> {
        self.entry_buffer.clear();
        self.buffer_position = 0;
        let num_shards = self.dataset.lock().unwrap().num_shards();

        while self.entry_buffer.is_empty() && self.current_shard < num_shards {
            let previous_next_file = self.next_file_to_load;
            self.load_entries_from_current_shard(max_bytes, max_entries)?;

            if self.entry_buffer.is_empty() && self.next_file_to_load == previous_next_file {
                self.current_shard = match &mut shard_order {
                    Some(order) => order.next().unwrap_or(num_shards),
                    None => self.current_shard + 1,
                };
                self.next_file_to_load = 0;
            }
        }

        Ok(())
    }

    fn load_entries_from_current_shard(
        &mut self,
        max_bytes: u64,
        max_entries: usize,
    ) -> PyResult<()> {
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
        let scan_end = start_idx
            .saturating_add(self.config.buffer_size.min(max_entries))
            .min(total_files);
        let mut file_entries = Vec::new();
        let mut end_idx = start_idx;
        let mut bytes = 0u64;
        for (filename, file_info) in metadata.file_range(start_idx, scan_end) {
            if file_info.length <= self.config.max_file_size {
                let length = if self.config.load_file_data {
                    file_info.length
                } else {
                    0
                };
                if !file_entries.is_empty() && (bytes > max_bytes || length > max_bytes - bytes) {
                    break;
                }
                bytes += length;
                file_entries.push((end_idx, filename, file_info));
            }
            end_idx += 1;
        }

        drop(dataset);

        self.next_file_to_load = end_idx;

        if file_entries.is_empty() {
            return Ok(());
        }

        self.load_file_batch(tar_path, token, file_entries)?;

        Ok(())
    }

    fn load_file_batch(
        &mut self,
        tar_path: String,
        token: Option<String>,
        file_entries: Vec<IndexedFileEntry>,
    ) -> PyResult<()> {
        let is_remote = tar_path.starts_with("http");

        if !self.config.load_file_data {
            self.push_entries_without_data(file_entries);
            return Ok(());
        }

        if is_remote && file_entries.len() > 1 {
            self.load_files_remote_streaming(tar_path, token, file_entries)
        } else {
            for (file_idx, filename, file_info) in file_entries {
                let data = self
                    .load_single_file_data(&tar_path, &file_info, is_remote, token.clone())
                    .or_else(|error| handle_file_read_error(&filename, error))?;

                self.entry_buffer.push(create_tar_entry(
                    filename,
                    &file_info,
                    data,
                    Some(self.current_shard),
                    Some(file_idx),
                ));
            }
            Ok(())
        }
    }

    fn push_entries_without_data(&mut self, file_entries: Vec<IndexedFileEntry>) {
        for (file_idx, filename, file_info) in file_entries {
            self.entry_buffer.push(create_tar_entry(
                filename,
                &file_info,
                Vec::new(),
                Some(self.current_shard),
                Some(file_idx),
            ));
        }
    }

    fn try_load_file_batch_from_cache(
        &mut self,
        tar_path: &str,
        token: Option<String>,
        file_entries: &[IndexedFileEntry],
    ) -> PyResult<bool> {
        let dataset = self.dataset.lock().unwrap();

        if let Some(cache) = &dataset.shard_cache {
            let shard_name = tar_path.rsplit('/').next().unwrap_or(tar_path);
            let cache_clone = cache.clone();
            drop(dataset);

            // Cache and acquire the read lock without an eviction race between
            // those operations.
            let cached = self.runtime.block_on(cache_clone.cache_shard_for_reading(
                shard_name,
                tar_path,
                token.clone(),
            ));
            if let Err(WebshartError::Xet(_) | WebshartError::XetHttp(_)) = &cached {
                return Err(cached.err().unwrap().into());
            }
            if let Ok((cached_path, _lock)) = cached {
                use std::io::{Read, Seek, SeekFrom};

                let mut file = std::fs::File::open(&cached_path).map_err(WebshartError::Io)?;
                for (file_idx, filename, file_info) in file_entries {
                    let data = if self.config.load_file_data {
                        let mut buffer = vec![0u8; file_info.length as usize];
                        file.seek(SeekFrom::Start(file_info.offset))
                            .and_then(|_| file.read_exact(&mut buffer))
                            .map(|_| buffer)
                            .unwrap_or_else(|e| {
                                eprintln!("Failed to load {}: {}", filename, e);
                                Vec::new()
                            })
                    } else {
                        Vec::new()
                    };

                    self.entry_buffer.push(create_tar_entry(
                        filename.clone(),
                        file_info,
                        data,
                        Some(self.current_shard),
                        Some(*file_idx),
                    ));
                }

                return Ok(true);
            }
        } else {
            drop(dataset);
        }

        Ok(false)
    }

    fn load_files_remote_streaming(
        &mut self,
        url: String,
        token: Option<String>,
        file_entries: Vec<IndexedFileEntry>,
    ) -> PyResult<()> {
        if file_entries.is_empty() {
            return Ok(());
        }

        if self.try_load_file_batch_from_cache(&url, token.clone(), &file_entries)? {
            return Ok(());
        }

        let chunk_size_bytes = (self.config.chunk_size_mb * 1024 * 1024) as u64;
        let ranges =
            plan_remote_byte_ranges(&file_entries, self.config.max_file_size, chunk_size_bytes);
        let fetch_result: Result<Vec<(RemoteByteRange, Vec<u8>)>> = self.runtime.block_on(async {
            if let Some(file) = crate::xet::resolve(&url, token.as_deref()).await? {
                use futures::{stream, StreamExt, TryStreamExt};
                let client = crate::xet::client()?;
                return stream::iter(ranges)
                    .map(|range| {
                        let file = file.clone();
                        async move {
                            let data = client
                                .read_range(&file, range.start, range.end - range.start)
                                .await?;
                            Ok((range, data))
                        }
                    })
                    .buffered(8)
                    .try_collect()
                    .await;
            }
            let client = file_http_client()?;
            let mut fetched = Vec::with_capacity(ranges.len());

            for range in ranges {
                let mut request = client
                    .get(&url)
                    .header("Range", format!("bytes={}-{}", range.start, range.end - 1))
                    .timeout(Duration::from_secs(60));

                if let Some(ref token) = token {
                    request = request.bearer_auth(token);
                }

                let response = request.send().await?;
                let data = if response.status().is_success()
                    || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
                {
                    response.bytes().await?.to_vec()
                } else if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    return Err(WebshartError::RateLimited);
                } else {
                    return Err(WebshartError::Http(
                        response.error_for_status().unwrap_err(),
                    ));
                };
                fetched.push((range, data));
            }

            Ok(fetched)
        });

        match fetch_result {
            Ok(fetched_ranges) => {
                for (file_idx, filename, file_info) in file_entries {
                    let file_end = file_info.offset.saturating_add(file_info.length);
                    let file_data = if file_info.length == 0 {
                        Vec::new()
                    } else {
                        fetched_ranges
                            .iter()
                            .find(|(range, _)| {
                                file_info.offset >= range.start && file_end <= range.end
                            })
                            .and_then(|(range, data)| {
                                let start = (file_info.offset - range.start) as usize;
                                let end = start.saturating_add(file_info.length as usize);
                                data.get(start..end).map(<[u8]>::to_vec)
                            })
                            .unwrap_or_default()
                    };

                    self.entry_buffer.push(create_tar_entry(
                        filename,
                        &file_info,
                        file_data,
                        Some(self.current_shard),
                        Some(file_idx),
                    ));
                }
                Ok(())
            }
            Err(error @ (WebshartError::Xet(_) | WebshartError::XetHttp(_))) => Err(error.into()),
            Err(_) => {
                for (file_idx, filename, file_info) in file_entries {
                    let data = self
                        .load_single_file_data(&url, &file_info, true, token.clone())
                        .or_else(|error| handle_file_read_error(&filename, error))?;
                    self.entry_buffer.push(create_tar_entry(
                        filename,
                        &file_info,
                        data,
                        Some(self.current_shard),
                        Some(file_idx),
                    ));
                }
                Ok(())
            }
        }
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
