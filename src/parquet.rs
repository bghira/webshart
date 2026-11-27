use bytes::Bytes;
use futures::{stream, StreamExt, TryStreamExt};
use parquet::basic::Type as PhysicalType;
use parquet::column::reader::{ColumnReader, ColumnReaderImpl};
use parquet::data_type::{ByteArray, DataType};
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::{ReaderProperties, ReaderPropertiesPtr};
use parquet::file::reader::{ChunkReader, FileReader, Length, RowGroupReader};
use parquet::file::serialized_reader::SerializedRowGroupReader;
use parquet::record::reader::RowIter;
use parquet::record::{Field, Row};
use parquet::schema::types::{SchemaDescriptor, Type};
use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::runtime::Runtime;

use crate::dataloader::file_loading::read_remote_range;
use crate::dataloader::shard_cache::{ShardCache, ShardLockGuard};
use crate::error::WebshartError;
use crate::jxl::{decoded_to_python, is_jxl, JxlOptions};
use crate::jxl_codec::Pixels;
use crate::s3::{PyS3Store, Store};

const RANGE_CHUNK: usize = 8 * 1024 * 1024;

fn s3_cache_name(store: &Store, key: &str, etag: &str) -> String {
    let identity = format!("{}:{key}:{:?}", store.identity, Some(etag));
    format!(
        "{}.parquet",
        crate::digest_to_hex(Sha256::digest(identity.as_bytes()))
    )
}

fn parquet_error(message: impl ToString) -> ParquetError {
    ParquetError::General(message.to_string())
}

fn count_present_values<T: DataType>(
    mut reader: ColumnReaderImpl<T>,
    expected_rows: usize,
) -> ParquetResult<usize> {
    let mut levels = Vec::new();
    let mut values = Vec::new();
    let mut rows = 0;
    let mut present = 0;
    while rows < expected_rows {
        levels.clear();
        values.clear();
        let (read, non_null, _) = reader.read_records(
            (expected_rows - rows).min(256),
            Some(&mut levels),
            None,
            &mut values,
        )?;
        if read == 0 {
            return Err(parquet_error("Truncated Parquet image column"));
        }
        rows += read;
        present += non_null;
    }
    Ok(present)
}

fn python_error(error: ParquetError) -> PyErr {
    PyOSError::new_err(error.to_string())
}

pub(crate) fn runtime() -> ParquetResult<&'static Runtime> {
    static RUNTIME: OnceLock<std::result::Result<Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(parquet_error)
}

struct Source {
    path: String,
    size: u64,
    token: Option<String>,
    remote: bool,
    s3: Option<(Arc<Store>, String, Option<String>)>,
    bytes_read: AtomicU64,
    max_page_bytes: usize,
    cache: Mutex<Option<Bytes>>,
    cached_ranges: Mutex<BTreeMap<u64, Bytes>>,
}

impl Source {
    fn cached_range(&self, start: u64) -> ParquetResult<Option<Bytes>> {
        let ranges = self
            .cached_ranges
            .lock()
            .map_err(|_| parquet_error("Parquet range cache lock poisoned"))?;
        Ok(ranges
            .range(..=start)
            .next_back()
            .and_then(|(&offset, data)| {
                let relative = (start - offset) as usize;
                (relative < data.len()).then(|| data.slice(relative..))
            }))
    }

    fn read(&self, start: u64, length: usize) -> ParquetResult<Vec<u8>> {
        if start
            .checked_add(length as u64)
            .is_none_or(|end| end > self.size)
        {
            return Err(parquet_error("Parquet range exceeds the source size"));
        }
        if let Some(data) = self
            .cache
            .lock()
            .map_err(|_| parquet_error("Parquet cache lock poisoned"))?
            .as_ref()
        {
            return Ok(data.slice(start as usize..start as usize + length).to_vec());
        }
        if let Some(data) = self.cached_range(start)? {
            if length <= data.len() {
                return Ok(data.slice(..length).to_vec());
            }
        }
        let data = if let Some((store, key, etag)) = &self.s3 {
            let mut data = Vec::with_capacity(length);
            while data.len() < length {
                let count = RANGE_CHUNK.min(length - data.len());
                let chunk = runtime()?
                    .block_on(store.range(key, start + data.len() as u64, count, etag.clone()))
                    .map_err(|error| match error {
                        object_store::Error::Precondition { .. } => {
                            parquet_error("S3 object changed during reading")
                        }
                        _ => parquet_error("S3 Parquet range request failed"),
                    })?;
                if chunk.len() != count {
                    return Err(parquet_error("S3 Parquet range length mismatch"));
                }
                data.extend_from_slice(&chunk);
            }
            data
        } else if self.remote {
            let mut data = Vec::with_capacity(length);
            while data.len() < length {
                let count = RANGE_CHUNK.min(length - data.len());
                let chunk = runtime()?
                    .block_on(read_remote_range(
                        &self.path,
                        self.token.as_deref(),
                        start + data.len() as u64,
                        count as u64,
                    ))
                    .map_err(|_| parquet_error("Parquet remote range request failed"))?;
                if chunk.len() != count {
                    return Err(parquet_error("Parquet remote range length mismatch"));
                }
                data.extend_from_slice(&chunk);
            }
            data
        } else {
            let mut file = File::open(&self.path)?;
            file.seek(SeekFrom::Start(start))?;
            let mut data = vec![0; length];
            file.read_exact(&mut data)?;
            data
        };
        self.bytes_read.fetch_add(length as u64, Ordering::Relaxed);
        Ok(data)
    }
}

struct LocalShard {
    path: PathBuf,
    _guard: ShardLockGuard,
}

#[pyclass(name = "_ParquetShardLease")]
struct PyParquetShardLease {
    shard: Arc<LocalShard>,
}

#[pymethods]
impl PyParquetShardLease {
    #[getter]
    fn path(&self) -> PathBuf {
        self.shard.path.clone()
    }
}

async fn cached_shard(cache: &ShardCache, name: &str) -> crate::Result<Option<LocalShard>> {
    if !cache.is_cached(name).await {
        return Ok(None);
    }
    match cache.lock_shard_for_reading(name).await {
        Ok(guard) => {
            cache.touch_shard(name).await;
            Ok(Some(LocalShard {
                path: cache.get_cached_shard_path(name),
                _guard: guard,
            }))
        }
        Err(WebshartError::CacheMiss(_)) => Ok(None),
        Err(error) => Err(error),
    }
}

struct CacheRead {
    cache: ShardCache,
    populate: bool,
    // Distinguish a cache miss already checked during this read from an unread source.
    local: Mutex<Option<Option<Arc<LocalShard>>>>,
}

impl CacheRead {
    fn open(&self, source: &Source) -> ParquetResult<Option<Arc<LocalShard>>> {
        let mut local = self.local.lock().unwrap();
        if let Some(shard) = local.as_ref() {
            return Ok(shard.clone());
        }
        let shard = runtime()?
            .block_on(self.acquire(source))
            .map_err(parquet_error)?;
        if let Some(shard) = &shard {
            if std::fs::metadata(&shard.path)?.len() != source.size {
                return Err(parquet_error(
                    "Cached Parquet file size does not match its source",
                ));
            }
        }
        let shard = shard.map(Arc::new);
        *local = Some(shard.clone());
        Ok(shard)
    }

    async fn acquire(&self, source: &Source) -> crate::Result<Option<LocalShard>> {
        let s3 = source.s3.as_ref();
        let name = match s3 {
            Some((store, key, Some(etag))) => s3_cache_name(store, key, etag),
            None if source.remote => format!(
                "{}.parquet",
                crate::digest_to_hex(Sha256::digest(source.path.as_bytes()))
            ),
            _ => return Ok(None),
        };
        if let Some(shard) = cached_shard(&self.cache, &name).await? {
            return Ok(Some(shard));
        }
        if !self.populate {
            return Ok(None);
        }
        let (path, guard) = if let Some((store, key, etag)) = s3 {
            let (path, guard) = self
                .cache
                .ensure_cached_with(&name, true, |target| async move {
                    store
                        .download(key, source.size, etag.as_deref(), &target)
                        .await
                })
                .await?;
            (path, guard.expect("read lock requested"))
        } else {
            self.cache
                .cache_shard_for_reading(&name, &source.path, source.token.clone())
                .await?
        };
        Ok(Some(LocalShard {
            path,
            _guard: guard,
        }))
    }
}

// A streaming decoder keeps its page state between calls. Its cached file must
// be evictable while the caller is handling a row, including after an error.
struct StreamRead(Option<Arc<CacheRead>>);

impl Drop for StreamRead {
    fn drop(&mut self) {
        if let Some(cache) = &self.0 {
            cache.local.lock().unwrap().take();
        }
    }
}

#[derive(Clone)]
struct RangeSource {
    source: Arc<Source>,
    cache: Option<Arc<CacheRead>>,
}

impl RangeSource {
    fn new(source: Arc<Source>, cache: Option<ShardCache>, populate: bool) -> Self {
        let cacheable = source.remote
            || source
                .s3
                .as_ref()
                .is_some_and(|(_, _, etag)| etag.is_some());
        let cache = cache.filter(|_| cacheable).map(|cache| {
            Arc::new(CacheRead {
                cache,
                populate,
                local: Mutex::new(None),
            })
        });
        Self { source, cache }
    }

    fn read(&self, start: u64, length: usize) -> ParquetResult<Vec<u8>> {
        if length == 0 {
            return Ok(Vec::new());
        }
        if let Some(cache) = &self.cache {
            if let Some(shard) = cache.open(&self.source)? {
                if start
                    .checked_add(length as u64)
                    .is_none_or(|end| end > self.source.size)
                {
                    return Err(parquet_error("Parquet range exceeds the source size"));
                }
                let mut file = File::open(&shard.path)?;
                file.seek(SeekFrom::Start(start))?;
                let mut data = vec![0; length];
                file.read_exact(&mut data)?;
                self.source
                    .bytes_read
                    .fetch_add(length as u64, Ordering::Relaxed);
                return Ok(data);
            }
        }
        self.source.read(start, length)
    }
}

struct RangeStream {
    range: RangeSource,
    offset: u64,
}

impl Read for RangeStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let mut length = buffer.len().min((self.range.len() - self.offset) as usize);
        if let Some(cached) = self
            .range
            .source
            .cached_range(self.offset)
            .map_err(std::io::Error::other)?
        {
            length = length.min(cached.len());
        }
        let data = self
            .range
            .read(self.offset, length)
            .map_err(std::io::Error::other)?;
        buffer[..length].copy_from_slice(&data);
        self.offset += length as u64;
        Ok(length)
    }
}

impl Length for RangeSource {
    fn len(&self) -> u64 {
        self.source.size
    }
}

impl ChunkReader for RangeSource {
    type T = BufReader<RangeStream>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        if start > self.source.size {
            return Err(parquet_error("Parquet stream offset exceeds source size"));
        }
        Ok(BufReader::with_capacity(
            1024,
            RangeStream {
                range: self.clone(),
                offset: start,
            },
        ))
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        if length > self.source.max_page_bytes {
            return Err(parquet_error("Parquet page exceeds max_page_bytes"));
        }
        Ok(self.read(start, length)?.into())
    }
}

fn binary_field(field: &Type) -> bool {
    if field.is_group() {
        field.get_fields().iter().any(|child| binary_field(child))
    } else {
        matches!(
            field.get_physical_type(),
            parquet::basic::Type::BYTE_ARRAY | parquet::basic::Type::FIXED_LEN_BYTE_ARRAY
        ) && field.get_basic_info().logical_type_ref().is_none()
            && field.get_basic_info().converted_type() == parquet::basic::ConvertedType::NONE
    }
}

fn sparse_binary_columns(
    schema: &SchemaDescriptor,
    projection: &Type,
) -> Option<Vec<(usize, String)>> {
    projection
        .get_fields()
        .iter()
        .map(|field| {
            if field.is_group()
                || !binary_field(field)
                || field.get_physical_type() != PhysicalType::BYTE_ARRAY
            {
                return None;
            }
            let index = schema.columns().iter().position(|col| {
                col.path().parts().len() == 1
                    && col.name() == field.name()
                    && col.max_rep_level() == 0
            })?;
            Some((index, field.name().to_owned()))
        })
        .collect()
}

// Flat binary image columns can skip whole pages and discard unrequested
// records without constructing image values for intervening rows.
fn sparse_binary_rows(
    group: &impl RowGroupReader,
    columns: &[(usize, String)],
    selected: &[(usize, usize)],
) -> ParquetResult<Vec<(usize, Row)>> {
    let mut rows = vec![Vec::new(); selected.len()];
    for (index, name) in columns {
        let ColumnReader::ByteArrayColumnReader(mut column) = group.get_column_reader(*index)?
        else {
            return Err(parquet_error("Expected a binary image column"));
        };
        let mut cursor = 0;
        let mut values = Vec::with_capacity(1);
        let mut defs = Vec::with_capacity(1);
        let mut output = rows.iter_mut();
        for occurrences in selected.chunk_by(|a, b| a.0 == b.0) {
            let wanted = occurrences[0].0;
            if column.skip_records(wanted - cursor)? != wanted - cursor {
                return Err(parquet_error("Parquet ended before requested row"));
            }
            values.clear();
            defs.clear();
            let (records, _, _) = column.read_records(1, Some(&mut defs), None, &mut values)?;
            if records != 1 {
                return Err(parquet_error("Parquet ended before requested row"));
            }
            let value = values.pop().map_or(Field::Null, Field::Bytes);
            for fields in output.by_ref().take(occurrences.len()) {
                fields.push((name.clone(), value.clone()));
            }
            cursor = wanted + 1;
        }
    }
    Ok(selected
        .iter()
        .zip(rows)
        .map(|(&(_, position), fields)| (position, Row::new(fields)))
        .collect())
}

fn row_to_python(py: Python<'_>, row: &Row) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    for (name, field) in row.get_column_iter() {
        output.set_item(name, field_to_python(py, field)?)?;
    }
    Ok(output.into_any().unbind())
}

fn row_field<'a>(row: &'a Row, name: &str) -> Option<&'a Field> {
    row.get_column_iter()
        .find_map(|(column, field)| (column == name).then_some(field))
}

fn image_field<'a>(row: &'a Row, column: &str) -> Option<&'a ByteArray> {
    let field = match row_field(row, column)? {
        Field::Group(image) => row_field(image, "bytes")?,
        field => field,
    };
    match field {
        Field::Bytes(payload) => Some(payload),
        _ => None,
    }
}

fn decode_jxl_rows(
    rows: &[(usize, Row)],
    indices: &[usize],
    column: &str,
    options: JxlOptions,
    parallel_reads: usize,
) -> PyResult<BTreeMap<usize, Result<Pixels, String>>> {
    let mut payloads = BTreeMap::new();
    for (position, row) in rows {
        if let Some(payload) = image_field(row, column).filter(|payload| is_jxl(payload.data())) {
            payloads
                .entry(indices[*position])
                .or_insert_with(|| payload.clone());
        }
    }
    runtime()
        .map_err(python_error)?
        .block_on(async {
            stream::iter(payloads.into_iter().map(|(index, payload)| async move {
                tokio::task::spawn_blocking(move || (index, options.decode(payload.data())))
                    .await
                    .map_err(|_| parquet_error("JPEG XL decode worker failed"))
            }))
            .buffer_unordered(parallel_reads)
            .try_collect()
            .await
        })
        .map_err(python_error)
}

fn field_to_python(py: Python<'_>, field: &Field) -> PyResult<Py<PyAny>> {
    macro_rules! scalar {
        ($value:expr) => {
            Ok($value.into_pyobject(py)?.into_any().unbind())
        };
    }
    match field {
        Field::Null => Ok(py.None()),
        Field::Bool(value) => Ok(value.into_pyobject(py)?.to_owned().into_any().unbind()),
        Field::Byte(value) => scalar!(value),
        Field::Short(value) => scalar!(value),
        Field::Int(value) | Field::Date(value) | Field::TimeMillis(value) => scalar!(value),
        Field::Long(value)
        | Field::TimeMicros(value)
        | Field::TimestampMillis(value)
        | Field::TimestampMicros(value) => scalar!(value),
        Field::UByte(value) => scalar!(value),
        Field::UShort(value) => scalar!(value),
        Field::UInt(value) => scalar!(value),
        Field::ULong(value) => scalar!(value),
        Field::Float16(value) => scalar!(value.to_f32()),
        Field::Float(value) => scalar!(value),
        Field::Double(value) => scalar!(value),
        Field::Str(value) => scalar!(value),
        Field::Decimal(_) => Ok(py
            .import("decimal")?
            .getattr("Decimal")?
            .call1((field.to_string(),))?
            .unbind()),
        Field::Bytes(value) => Ok(PyBytes::new(py, value.data()).into_any().unbind()),
        Field::Group(row) => row_to_python(py, row),
        Field::ListInternal(values) => {
            let output = PyList::empty(py);
            for value in values.elements() {
                output.append(field_to_python(py, value)?)?;
            }
            Ok(output.into_any().unbind())
        }
        Field::MapInternal(values) => {
            let output = PyDict::new(py);
            for (key, value) in values.entries() {
                output.set_item(field_to_python(py, key)?, field_to_python(py, value)?)?;
            }
            Ok(output.into_any().unbind())
        }
    }
}

fn project_columns(schema: &Type, columns: Option<Vec<String>>) -> PyResult<Option<Type>> {
    let Some(columns) = columns else {
        return Ok(None);
    };
    if columns.is_empty() {
        return Err(PyValueError::new_err("Select at least one Parquet column"));
    }
    let mut fields = Vec::new();
    for name in columns {
        let field = schema
            .get_fields()
            .iter()
            .find(|field| field.name() == name)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown Parquet column: {name}")))?;
        if fields.contains(field) {
            return Err(PyValueError::new_err("Duplicate Parquet projection column"));
        }
        fields.push(field.clone());
    }
    Ok(Some(
        Type::group_type_builder(schema.name())
            .with_fields(fields)
            .build()
            .map_err(python_error)?,
    ))
}

#[pyfunction]
#[pyo3(signature=(dimensions, key="aspect", target_pixel_area=None, target_resolution_multiple=64, round_to=Some(2)))]
pub fn _parquet_bucket_keys(
    py: Python<'_>,
    dimensions: Vec<(u32, u32)>,
    key: &str,
    target_pixel_area: Option<u32>,
    target_resolution_multiple: u32,
    round_to: Option<usize>,
) -> PyResult<Vec<String>> {
    use crate::dataloader::aspect_buckets::{calculate_bucket_key, BucketKeyType};
    if target_resolution_multiple == 0 || target_pixel_area == Some(0) {
        return Err(PyValueError::new_err(
            "Target size and multiple must be positive",
        ));
    }
    if dimensions.iter().any(|&(w, h)| w == 0 || h == 0) {
        return Err(PyValueError::new_err("Image dimensions must be positive"));
    }
    let key = BucketKeyType::parse(key)?;
    Ok(py.detach(move || {
        dimensions
            .into_iter()
            .map(|(w, h)| {
                calculate_bucket_key(
                    &key,
                    w,
                    h,
                    None,
                    target_pixel_area,
                    target_resolution_multiple,
                    round_to,
                )
                .0
            })
            .collect()
    }))
}

#[pyclass(name = "_ParquetCache")]
pub struct PyParquetCache {
    cache: ShardCache,
}

#[pymethods]
impl PyParquetCache {
    #[new]
    #[pyo3(signature=(location, cache_limit_gb=25.0, parallel_downloads=4))]
    fn new(
        py: Python<'_>,
        location: String,
        cache_limit_gb: f64,
        parallel_downloads: usize,
    ) -> PyResult<Self> {
        if !cache_limit_gb.is_finite() || cache_limit_gb <= 0.0 || parallel_downloads == 0 {
            return Err(PyValueError::new_err(
                "Cache size and parallel_downloads must be positive",
            ));
        }
        py.detach(move || {
            let mut cache = ShardCache::new(location.into(), cache_limit_gb, parallel_downloads);
            runtime()
                .map_err(python_error)?
                .block_on(async {
                    cache.ensure_cache_dir().await?;
                    cache.initialize_from_disk().await
                })
                .map_err(PyErr::from)?;
            Ok(Self { cache })
        })
    }
    #[pyo3(signature = (identity, build=None))]
    fn repacked_shard(
        &self,
        py: Python<'_>,
        identity: &str,
        build: Option<Py<PyAny>>,
    ) -> PyResult<Option<PyParquetShardLease>> {
        let name = format!(
            "{}.parquet",
            crate::digest_to_hex(Sha256::digest(identity.as_bytes()))
        );
        py.detach(|| {
            runtime().map_err(python_error)?.block_on(async {
                let shard = if let Some(build) = build {
                    let (path, guard) = self
                        .cache
                        .ensure_cached_with(&name, true, |dest| async move {
                            tokio::task::spawn_blocking(move || {
                                Python::attach(|py| build.call1(py, (&dest,)))
                                    .map_err(|e| std::io::Error::other(e.to_string()))?;
                                Ok(std::fs::metadata(dest)?.len())
                            })
                            .await
                            .map_err(std::io::Error::other)?
                        })
                        .await?;
                    Some(LocalShard {
                        path,
                        _guard: guard.expect("read lease requested"),
                    })
                } else {
                    cached_shard(&self.cache, &name).await?
                };
                Ok(shard.map(|file| PyParquetShardLease {
                    shard: Arc::new(file),
                }))
            })
        })
    }
}

#[derive(Clone)]
struct ParquetReader {
    ranges: Arc<RangeSource>,
    metadata: Arc<ParquetMetaData>,
    properties: ReaderPropertiesPtr,
}

impl ParquetReader {
    fn with_cache(&self, cache: Option<ShardCache>, populate: bool) -> Self {
        match cache {
            Some(cache) => Self {
                ranges: Arc::new(RangeSource::new(
                    self.ranges.source.clone(),
                    Some(cache),
                    populate,
                )),
                metadata: self.metadata.clone(),
                properties: self.properties.clone(),
            },
            None => self.clone(),
        }
    }

    fn metadata(&self) -> &ParquetMetaData {
        &self.metadata
    }

    fn get_row_group(&self, index: usize) -> ParquetResult<impl RowGroupReader + '_> {
        SerializedRowGroupReader::new(
            self.ranges.clone(),
            self.metadata.row_group(index),
            self.metadata.page_index_for_row_group(index),
            self.properties.clone(),
        )
    }
}

struct RowGroupWindow {
    reader: ParquetReader,
    start: usize,
    stop: usize,
}

impl FileReader for RowGroupWindow {
    fn metadata(&self) -> &ParquetMetaData {
        self.reader.metadata()
    }

    fn num_row_groups(&self) -> usize {
        self.stop - self.start
    }

    fn get_row_group(&self, index: usize) -> ParquetResult<Box<dyn RowGroupReader + '_>> {
        Ok(Box::new(self.reader.get_row_group(self.start + index)?))
    }

    fn get_row_iter(&self, projection: Option<Type>) -> ParquetResult<RowIter<'_>> {
        RowIter::from_file(projection, self)
    }
}

#[pyclass(name = "_ParquetFile")]
pub struct PyParquetFile {
    reader: ParquetReader,
    source: Arc<Source>,
    info_json: String,
    num_rows: usize,
    row_group_starts: Vec<usize>,
}

#[pymethods]
impl PyParquetFile {
    #[new]
    #[pyo3(signature = (path, size=None, hf_token=None, max_page_bytes=536870912, cache_bytes=0, s3=None, cache=None, expected_sha256=None, *, verified_s3_etag=None, cache_payload=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: String,
        size: Option<u64>,
        hf_token: Option<String>,
        max_page_bytes: usize,
        cache_bytes: usize,
        s3: Option<PyRef<'_, PyS3Store>>,
        cache: Option<PyRef<'_, PyParquetCache>>,
        expected_sha256: Option<String>,
        verified_s3_etag: Option<String>,
        cache_payload: bool,
    ) -> PyResult<Self> {
        if max_page_bytes == 0 {
            return Err(PyValueError::new_err("max_page_bytes must be positive"));
        }
        let store = s3.map(|s| s.inner.clone());
        let cache = cache.map(|c| c.cache.clone());
        py.detach(move || {
            let remote = path.starts_with("https://") || path.starts_with("http://");
            let mut s3_source = None;
            let size = if let Some(uri) = path.strip_prefix("s3://") {
                let (bucket, key) = uri.split_once('/').ok_or_else(|| {
                    PyValueError::new_err("S3 Parquet URI requires an object key")
                })?;
                let store = store.ok_or_else(|| {
                    PyValueError::new_err("S3 Parquet reads require an S3 store")
                })?;
                if bucket != store.bucket {
                    return Err(PyValueError::new_err("S3 store belongs to a different bucket"));
                }
                let (size, etag) = if let (Some(size), Some(etag)) = (size, verified_s3_etag) {
                    (size, Some(etag))
                } else {
                    let (meta, attributes) = runtime()
                        .map_err(python_error)?
                        .block_on(store.head_with_attributes(key))
                        .map_err(crate::s3::python_error)?;
                    let digest = attributes.get(&object_store::Attribute::Metadata("sha256".into()));
                    if let (Some(expected), Some(actual)) = (&expected_sha256, digest) {
                        if expected != actual.as_ref() {
                            return Err(PyValueError::new_err(
                                "S3 object SHA-256 metadata does not match its capture manifest",
                            ));
                        }
                    }
                    if size.is_some_and(|expected| expected != meta.size) {
                        return Err(PyValueError::new_err("S3 object size does not match its manifest"));
                    }
                    (meta.size, meta.e_tag)
                };
                s3_source = Some((store, key.to_string(), etag));
                size
            } else if remote {
                size.ok_or_else(|| PyValueError::new_err("Remote Parquet files require their exact size"))?
            } else {
                let actual = std::fs::metadata(&path)?.len();
                if size.is_some_and(|expected| expected != actual) { return Err(PyValueError::new_err("Parquet source size does not match its manifest")); }
                actual
            };
            let source = Arc::new(Source {
                path,
                size,
                token: hf_token,
                remote,
                s3: s3_source,
                bytes_read: AtomicU64::new(0),
                max_page_bytes,
                cache: Mutex::new(None),
                cached_ranges: Mutex::new(BTreeMap::new()),
            });
            let ranges = RangeSource::new(source.clone(), cache, cache_payload);
            if size <= cache_bytes as u64 {
                let data = ranges.read(0, size as usize).map_err(python_error)?;
                *source
                    .cache
                    .lock()
                    .map_err(|_| PyOSError::new_err("Parquet cache lock poisoned"))? = Some(data.into());
            }
            let metadata = ParquetMetaDataReader::new().parse_and_finish(&ranges).map_err(python_error)?;
            drop(ranges);
            let fields = metadata.file_metadata().schema_descr().root_schema().get_fields();
            let columns: Vec<_> = fields.iter().map(|field| field.name()).collect();
            let binary_columns: Vec<_> = fields.iter().filter(|field| binary_field(field)).map(|field| field.name()).collect();
            let row_groups: Vec<_> = metadata.row_groups().iter().map(|group| group.num_rows()).collect();
            let num_rows = metadata.file_metadata().num_rows() as usize;
            let etag = source.s3.as_ref().and_then(|(_, _, etag)| etag.as_deref());
            let info_json = serde_json::json!({"num_rows": num_rows, "columns": columns, "binary_columns": binary_columns, "row_groups": row_groups, "size": size, "etag": etag}).to_string();
            let row_group_starts = std::iter::once(0)
                .chain(row_groups.iter().scan(0, |start, &length| {
                    *start += length as usize;
                    Some(*start)
                }))
                .collect();
            let reader = ParquetReader {
                ranges: Arc::new(RangeSource::new(source.clone(), None, false)),
                metadata: Arc::new(metadata),
                properties: Arc::new(ReaderProperties::builder().build()),
            };
            Ok(Self { reader, source, info_json, num_rows, row_group_starts })
        })
    }

    fn cache_shard(&self, py: Python<'_>, cache: &PyParquetCache) -> PyResult<PyParquetShardLease> {
        py.detach(|| {
            let ranges = RangeSource::new(self.source.clone(), Some(cache.cache.clone()), true);
            let shard = ranges
                .cache
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("Shard is not cacheable"))?
                .open(&self.source)
                .map_err(python_error)?
                .ok_or_else(|| PyValueError::new_err("Shard has no cache identity"))?;
            Ok(PyParquetShardLease { shard })
        })
    }

    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value: serde_json::Value = serde_json::from_str(&self.info_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(pythonize::pythonize(py, &value)?.unbind())
    }

    /// Count non-null values in a scalar image column or an image struct's bytes field.
    fn count_present_images(&self, py: Python<'_>, column: &str) -> PyResult<usize> {
        py.detach(|| {
            let metadata = self.reader.metadata();
            let descriptor = metadata.file_metadata().schema_descr();
            let (index, leaf) = descriptor
                .columns()
                .iter()
                .enumerate()
                .find(|(_, field)| {
                    matches!(field.path().parts(), [name] if name == column)
                        || matches!(field.path().parts(), [name, bytes] if name == column && bytes == "bytes")
                })
                .filter(|(_, field)| {
                    field.max_rep_level() == 0
                        && matches!(
                            field.physical_type(),
                            PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY
                        )
                })
                .ok_or_else(|| PyValueError::new_err("Expected a scalar binary image column or an image struct with binary bytes"))?;

            if leaf.max_def_level() == 0 {
                return Ok(self.num_rows);
            }

            let mut present = 0;
            for (group_index, group) in metadata.row_groups().iter().enumerate() {
                let rows = group.num_rows() as usize;
                let nulls = group.column(index).statistics().and_then(|stats| stats.null_count_opt());
                if let Some(nulls) = nulls {
                    let count = (rows as u64)
                        .checked_sub(nulls)
                        .ok_or_else(|| PyValueError::new_err("Parquet image null count exceeds row count"))?;
                    present += count as usize;
                } else {
                    let group = self.reader.get_row_group(group_index).map_err(python_error)?;
                    let reader = group.get_column_reader(index).map_err(python_error)?;
                    present += match reader {
                        ColumnReader::ByteArrayColumnReader(reader) => count_present_values(reader, rows),
                        ColumnReader::FixedLenByteArrayColumnReader(reader) => count_present_values(reader, rows),
                        _ => Err(parquet_error("Unexpected Parquet image column type")),
                    }
                    .map_err(python_error)?;
                }
            }
            Ok(present)
        })
    }

    #[getter]
    fn bytes_read(&self) -> u64 {
        self.source.bytes_read.load(Ordering::Relaxed)
    }

    fn cache_source(&self, py: Python<'_>, max_bytes: usize) -> PyResult<bool> {
        py.detach(|| {
            if self.source.size > max_bytes as u64 {
                return Ok(false);
            }
            if self
                .source
                .cache
                .lock()
                .map_err(|_| PyOSError::new_err("Parquet cache lock poisoned"))?
                .is_some()
            {
                return Ok(true);
            }
            let data = self
                .source
                .read(0, self.source.size as usize)
                .map_err(python_error)?;
            *self
                .source
                .cache
                .lock()
                .map_err(|_| PyOSError::new_err("Parquet cache lock poisoned"))? =
                Some(data.into());
            Ok(true)
        })
    }

    fn verify_sha256(&self, py: Python<'_>, expected: String) -> PyResult<()> {
        py.detach(|| {
            let mut digest = Sha256::new();
            let mut offset = 0;
            while offset < self.source.size {
                let length = RANGE_CHUNK.min((self.source.size - offset) as usize);
                digest.update(self.source.read(offset, length).map_err(python_error)?);
                offset += length as u64;
            }
            if crate::digest_to_hex(digest.finalize()) != expected {
                return Err(PyValueError::new_err("Metadata SHA-256 mismatch"));
            }
            Ok(())
        })
    }

    /// Coalesce only selected column chunks into a bounded metadata cache.
    #[pyo3(signature=(columns, max_bytes=67108864, parallel_reads=4, start_group=0, stop_group=None, *, shard_cache=None))]
    #[allow(clippy::too_many_arguments)]
    fn prefetch_columns(
        &self,
        py: Python<'_>,
        columns: Vec<String>,
        max_bytes: usize,
        parallel_reads: usize,
        start_group: usize,
        stop_group: Option<usize>,
        shard_cache: Option<PyRef<'_, PyParquetCache>>,
    ) -> PyResult<bool> {
        if parallel_reads == 0 {
            return Err(PyValueError::new_err("parallel_reads must be positive"));
        }
        let shard_cache = shard_cache.map(|cache| cache.cache.clone());
        py.detach(|| {
            project_columns(
                self.reader
                    .metadata()
                    .file_metadata()
                    .schema_descr()
                    .root_schema(),
                Some(columns.clone()),
            )?;
            let groups = self.reader.metadata().row_groups();
            let stop_group = stop_group.unwrap_or(groups.len());
            let groups = groups
                .get(start_group..stop_group)
                .ok_or_else(|| PyIndexError::new_err("Parquet row group range is out of bounds"))?;
            let mut spans: Vec<_> = groups
                .iter()
                .flat_map(|group| group.columns())
                .filter(|column| columns.contains(&column.column_path().parts()[0]))
                .map(|column| {
                    let (start, length) = column.byte_range();
                    start
                        .checked_add(length)
                        .map(|end| start..end)
                        .ok_or_else(|| PyValueError::new_err("Parquet column range overflow"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            spans.sort_unstable_by_key(|span| span.start);
            let mut ranges: Vec<std::ops::Range<u64>> = Vec::new();
            for span in spans {
                if let Some(previous) = ranges.last_mut() {
                    if span.start <= previous.end {
                        previous.end = previous.end.max(span.end);
                        continue;
                    }
                }
                ranges.push(span);
            }
            let total = ranges
                .iter()
                .try_fold(0u64, |sum, span| sum.checked_add(span.end - span.start))
                .ok_or_else(|| PyValueError::new_err("Parquet metadata size overflow"))?;
            if total > max_bytes as u64 {
                return Ok(false);
            }
            let ranges_source = self.reader.with_cache(shard_cache, false).ranges;
            let data = runtime()
                .map_err(python_error)?
                .block_on(async {
                    stream::iter(ranges.into_iter().map(|span| {
                        let source = ranges_source.clone();
                        async move {
                            tokio::task::spawn_blocking(move || {
                                source
                                    .read(span.start, (span.end - span.start) as usize)
                                    .map(|data| (span.start, Bytes::from(data)))
                            })
                            .await
                            .map_err(|_| parquet_error("Parquet metadata prefetch failed"))?
                        }
                    }))
                    .buffer_unordered(parallel_reads)
                    .try_collect::<BTreeMap<_, _>>()
                    .await
                })
                .map_err(python_error)?;
            *self
                .source
                .cached_ranges
                .lock()
                .map_err(|_| PyOSError::new_err("Parquet range cache lock poisoned"))? = data;
            Ok(true)
        })
    }

    /// Read each selected row group once, preserving request order and duplicates.
    #[pyo3(signature=(indices, columns=None, parallel_reads=4, shard_cache=None, *, cache_payload=Some(false), decode_jxl=None))]
    #[allow(clippy::too_many_arguments)]
    fn rows_at(
        &self,
        py: Python<'_>,
        indices: Vec<usize>,
        columns: Option<Vec<String>>,
        parallel_reads: usize,
        shard_cache: Option<PyRef<'_, PyParquetCache>>,
        cache_payload: Option<bool>,
        decode_jxl: Option<(String, usize)>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        if parallel_reads == 0 {
            return Err(PyValueError::new_err("parallel_reads must be positive"));
        }
        if indices.iter().any(|&i| i >= self.num_rows) {
            return Err(PyIndexError::new_err("Parquet row index out of bounds"));
        }
        let shard_cache = shard_cache.map(|cache| cache.cache.clone());
        let jxl = decode_jxl
            .map(|(column, max_pixels)| Ok::<_, PyErr>((column, JxlOptions::new(max_pixels)?)))
            .transpose()?;
        let (rows, decoded) = py.detach(|| {
            let schema = self.reader.metadata().file_metadata().schema_descr();
            let projection = project_columns(schema.root_schema(), columns)?;
            let sparse_columns = projection
                .as_ref()
                .and_then(|fields| sparse_binary_columns(schema, fields));
            let populate = cache_payload.unwrap_or_else(|| {
                sparse_columns.is_none()
                    || (indices.len() >= self.num_rows
                        && indices.iter().collect::<HashSet<_>>().len() == self.num_rows)
            });
            let reader = self.reader.with_cache(shard_cache, populate);
            let starts = &self.row_group_starts;
            let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
            for (position, index) in indices.iter().copied().enumerate() {
                let group = starts.partition_point(|&start| start <= index) - 1;
                groups
                    .entry(group)
                    .or_default()
                    .push((index - starts[group], position));
            }
            if let Some(cache) = reader.ranges.cache.as_ref().filter(|_| !groups.is_empty()) {
                cache.open(&self.source).map_err(python_error)?;
            }
            let output = runtime()
                .map_err(python_error)?
                .block_on(async {
                    stream::iter(groups.into_iter().map(|(group, mut selected)| {
                        let reader = reader.clone();
                        let projection = projection.clone();
                        let sparse_columns = sparse_columns.clone();
                        async move {
                            tokio::task::spawn_blocking(
                                move || -> ParquetResult<Vec<(usize, Row)>> {
                                    selected.sort_unstable();
                                    let group_reader = reader.get_row_group(group)?;
                                    if let Some(columns) = sparse_columns {
                                        return sparse_binary_rows(
                                            &group_reader,
                                            &columns,
                                            &selected,
                                        );
                                    }
                                    let mut rows =
                                        RowIter::from_row_group(projection, &group_reader)?
                                            .with_batch_size(1);
                                    let mut current = 0;
                                    let mut output = Vec::new();
                                    let mut selected = selected.into_iter().peekable();
                                    while let Some(&(wanted, _)) = selected.peek() {
                                        while current < wanted {
                                            rows.next().ok_or_else(|| {
                                                parquet_error("Parquet ended before requested row")
                                            })??;
                                            current += 1;
                                        }
                                        let row = rows.next().ok_or_else(|| {
                                            parquet_error("Parquet ended before requested row")
                                        })??;
                                        current += 1;
                                        while selected.peek().is_some_and(|&(i, _)| i == wanted) {
                                            output.push((selected.next().unwrap().1, row.clone()));
                                        }
                                    }
                                    Ok(output)
                                },
                            )
                            .await
                            .map_err(|_| parquet_error("Parquet batch reader failed"))?
                        }
                    }))
                    .buffer_unordered(parallel_reads)
                    .try_collect::<Vec<_>>()
                    .await
                })
                .map_err(python_error)?;
            // The row data is owned. Release the shard before CPU-only image decoding.
            drop(reader);
            let mut rows: Vec<_> = output.into_iter().flatten().collect();
            rows.sort_unstable_by_key(|(position, _)| *position);
            let decoded = jxl
                .map(|(column, options)| {
                    decode_jxl_rows(&rows, &indices, &column, options, parallel_reads)
                })
                .transpose()?;
            Ok::<_, PyErr>((rows, decoded))
        })?;
        let Some(decoded) = decoded else {
            return rows.iter().map(|(_, row)| row_to_python(py, row)).collect();
        };
        let decoded = decoded
            .into_iter()
            .map(|(index, result)| Ok((index, decoded_to_python(py, result)?)))
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        rows.iter()
            .map(|(position, row)| {
                let image = decoded
                    .get(&indices[*position])
                    .map_or_else(|| py.None(), |image| image.clone_ref(py));
                Ok(PyTuple::new(py, [row_to_python(py, row)?, image])?
                    .into_any()
                    .unbind())
            })
            .collect()
    }

    #[pyo3(signature = (columns=None, start=0, stop=None, *, shard_cache=None, cache_payload=true))]
    fn rows(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        start: usize,
        stop: Option<usize>,
        shard_cache: Option<PyRef<'_, PyParquetCache>>,
        cache_payload: bool,
    ) -> PyResult<PyParquetRows> {
        let stop = stop.unwrap_or(self.num_rows);
        if start > stop || stop > self.num_rows {
            return Err(PyIndexError::new_err("Parquet row range is out of bounds"));
        }
        let shard_cache = shard_cache.map(|cache| cache.cache.clone());
        py.detach(|| {
            let group_start = self
                .row_group_starts
                .partition_point(|&index| index <= start)
                - 1;
            let group_stop = self.row_group_starts.partition_point(|&index| index < stop);
            let reader = self.reader.with_cache(shard_cache, cache_payload);
            let cache = reader.ranges.cache.clone();
            let _read = StreamRead(cache.clone());
            let reader = RowGroupWindow {
                reader,
                start: group_start,
                stop: group_stop.max(group_start),
            };
            let schema = reader
                .metadata()
                .file_metadata()
                .schema_descr()
                .root_schema();
            let projection = project_columns(schema, columns)?;
            let rows = RowIter::from_file_into(Box::new(reader))
                .project(projection)
                .map_err(python_error)?
                .with_batch_size(1);
            Ok(PyParquetRows {
                state: Mutex::new(RowState {
                    rows,
                    cache,
                    skip: start - self.row_group_starts[group_start],
                    remaining: stop - start,
                }),
            })
        })
    }
}

struct RowState {
    rows: RowIter<'static>,
    cache: Option<Arc<CacheRead>>,
    skip: usize,
    remaining: usize,
}

#[pyclass(name = "_ParquetRows")]
pub struct PyParquetRows {
    state: Mutex<RowState>,
}

#[pymethods]
impl PyParquetRows {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let row = py.detach(|| {
            let mut state = self
                .state
                .lock()
                .map_err(|_| PyOSError::new_err("Parquet reader lock poisoned"))?;
            if state.remaining == 0 {
                return Ok(None);
            }
            let _read = StreamRead(state.cache.clone());
            let result = (|| {
                while state.skip > 0 {
                    state
                        .rows
                        .next()
                        .ok_or_else(|| parquet_error("Parquet ended before requested row"))??;
                    state.skip -= 1;
                }
                state
                    .rows
                    .next()
                    .ok_or_else(|| parquet_error("Parquet ended before requested row"))?
            })();
            match result {
                Ok(row) => {
                    state.remaining -= 1;
                    Ok(Some(row))
                }
                Err(error) => {
                    state.remaining = 0;
                    Err(python_error(error))
                }
            }
        })?;
        row.as_ref().map(|row| row_to_python(py, row)).transpose()
    }
}
