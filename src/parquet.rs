use bytes::Bytes;
use parquet::errors::{ParquetError, Result as ParquetResult};
use parquet::file::reader::{ChunkReader, FileReader, Length};
use parquet::file::serialized_reader::{ReadOptionsBuilder, SerializedFileReader};
use parquet::record::reader::RowIter;
use parquet::record::{Field, Row};
use parquet::schema::types::Type;
use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tokio::runtime::Runtime;

use crate::dataloader::file_loading::read_remote_range;

const RANGE_CHUNK: usize = 8 * 1024 * 1024;

fn parquet_error(message: impl ToString) -> ParquetError {
    ParquetError::General(message.to_string())
}

fn python_error(error: ParquetError) -> PyErr {
    PyOSError::new_err(error.to_string())
}

fn runtime() -> ParquetResult<&'static Runtime> {
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
    bytes_read: AtomicU64,
    max_page_bytes: usize,
    cache: Mutex<Option<Bytes>>,
}

impl Source {
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
        let data = if self.remote {
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

#[derive(Clone)]
struct RangeSource(Arc<Source>);

struct RangeStream {
    source: Arc<Source>,
    offset: u64,
}

impl Read for RangeStream {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let length = buffer.len().min((self.source.size - self.offset) as usize);
        let data = self
            .source
            .read(self.offset, length)
            .map_err(std::io::Error::other)?;
        buffer[..length].copy_from_slice(&data);
        self.offset += length as u64;
        Ok(length)
    }
}

impl Length for RangeSource {
    fn len(&self) -> u64 {
        self.0.size
    }
}

impl ChunkReader for RangeSource {
    type T = BufReader<RangeStream>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        if start > self.0.size {
            return Err(parquet_error("Parquet stream offset exceeds source size"));
        }
        Ok(BufReader::with_capacity(
            1024,
            RangeStream {
                source: self.0.clone(),
                offset: start,
            },
        ))
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        if length > self.0.max_page_bytes {
            return Err(parquet_error("Parquet page exceeds max_page_bytes"));
        }
        Ok(self.0.read(start, length)?.into())
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

fn row_to_python(py: Python<'_>, row: &Row) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    for (name, field) in row.get_column_iter() {
        output.set_item(name, field_to_python(py, field)?)?;
    }
    Ok(output.into_any().unbind())
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

#[pyclass(name = "_ParquetFile")]
pub struct PyParquetFile {
    source: Arc<Source>,
    info_json: String,
    num_rows: usize,
}

#[pymethods]
impl PyParquetFile {
    #[new]
    #[pyo3(signature = (path, size=None, hf_token=None, max_page_bytes=536870912, cache_bytes=0))]
    fn new(
        py: Python<'_>,
        path: String,
        size: Option<u64>,
        hf_token: Option<String>,
        max_page_bytes: usize,
        cache_bytes: usize,
    ) -> PyResult<Self> {
        if max_page_bytes == 0 {
            return Err(PyValueError::new_err("max_page_bytes must be positive"));
        }
        py.detach(move || {
            let remote = path.starts_with("https://") || path.starts_with("http://");
            let size = if remote {
                size.ok_or_else(|| PyValueError::new_err("Remote Parquet files require their exact size"))?
            } else {
                let actual = std::fs::metadata(&path)?.len();
                if size.is_some_and(|expected| expected != actual) {
                    return Err(PyValueError::new_err("Parquet source size does not match its manifest"));
                }
                actual
            };
            let source = Arc::new(Source { path, size, token: hf_token, remote, bytes_read: AtomicU64::new(0), max_page_bytes, cache: Mutex::new(None) });
            if size <= cache_bytes as u64 {
                let data = source.read(0, size as usize).map_err(python_error)?;
                *source.cache.lock().map_err(|_| PyOSError::new_err("Parquet cache lock poisoned"))? = Some(data.into());
            }
            let reader = SerializedFileReader::new(RangeSource(source.clone())).map_err(python_error)?;
            let metadata = reader.metadata();
            let fields = metadata.file_metadata().schema_descr().root_schema().get_fields();
            let columns: Vec<_> = fields.iter().map(|field| field.name()).collect();
            let binary_columns: Vec<_> = fields.iter().filter(|field| binary_field(field)).map(|field| field.name()).collect();
            let row_groups: Vec<_> = metadata.row_groups().iter().map(|group| group.num_rows()).collect();
            let num_rows = metadata.file_metadata().num_rows() as usize;
            let info_json = serde_json::json!({"num_rows": num_rows, "columns": columns, "binary_columns": binary_columns, "row_groups": row_groups, "size": size}).to_string();
            Ok(Self { source, info_json, num_rows })
        })
    }

    fn info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value: serde_json::Value = serde_json::from_str(&self.info_json)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(pythonize::pythonize(py, &value)?.unbind())
    }

    #[getter]
    fn bytes_read(&self) -> u64 {
        self.source.bytes_read.load(Ordering::Relaxed)
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

    #[pyo3(signature = (columns=None, start=0, stop=None))]
    fn rows(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        start: usize,
        stop: Option<usize>,
    ) -> PyResult<PyParquetRows> {
        let stop = stop.unwrap_or(self.num_rows);
        if start > stop || stop > self.num_rows {
            return Err(PyIndexError::new_err("Parquet row range is out of bounds"));
        }
        let source = self.source.clone();
        py.detach(move || {
            let mut row_start = 0usize;
            let skipped = Arc::new(AtomicU64::new(0));
            let skipped_by_predicate = skipped.clone();
            let options = ReadOptionsBuilder::new()
                .with_predicate(Box::new(move |group, _| {
                    let row_end = row_start + group.num_rows() as usize;
                    let selected = row_end > start && row_start < stop;
                    if row_end <= start {
                        skipped_by_predicate.store(row_end as u64, Ordering::Relaxed);
                    }
                    row_start = row_end;
                    selected
                }))
                .build();
            let reader = SerializedFileReader::new_with_options(RangeSource(source), options)
                .map_err(python_error)?;
            let schema = reader
                .metadata()
                .file_metadata()
                .schema_descr()
                .root_schema();
            let projection = if let Some(columns) = columns {
                if columns.is_empty() {
                    return Err(PyValueError::new_err("Select at least one Parquet column"));
                }
                let mut fields = Vec::new();
                for name in columns {
                    let field = schema
                        .get_fields()
                        .iter()
                        .find(|field| field.name() == name)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!("Unknown Parquet column: {name}"))
                        })?;
                    if fields.contains(field) {
                        return Err(PyValueError::new_err("Duplicate Parquet projection column"));
                    }
                    fields.push(field.clone());
                }
                Some(
                    Type::group_type_builder(schema.name())
                        .with_fields(fields)
                        .build()
                        .map_err(python_error)?,
                )
            } else {
                None
            };
            let rows = RowIter::from_file_into(Box::new(reader))
                .project(projection)
                .map_err(python_error)?
                .with_batch_size(1);
            Ok(PyParquetRows {
                state: Mutex::new(RowState {
                    rows,
                    skip: start - skipped.load(Ordering::Relaxed) as usize,
                    remaining: stop - start,
                }),
            })
        })
    }
}

struct RowState {
    rows: RowIter<'static>,
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
