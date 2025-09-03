use crate::FileInfo;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

#[derive(Debug, Clone)]
pub struct BucketEntry {
    pub shard_idx: usize,
    pub filename: String,
    pub file_info: crate::metadata::FileInfo,
    pub original_size: Option<(u32, u32)>,
}

pub fn create_tar_entry(path: String, file_info: &FileInfo, data: Vec<u8>) -> PyTarFileEntry {
    PyTarFileEntry {
        path,
        offset: file_info.offset,
        size: file_info.length,
        data,
    }
}
