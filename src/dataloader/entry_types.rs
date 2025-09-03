use crate::FileInfo;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

#[pyclass(name = "TarFileEntry")]
#[derive(Clone)]
pub struct PyTarFileEntry {
    pub path: String,
    pub offset: u64,
    pub size: u64,
    pub data: Vec<u8>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub aspect: Option<f32>,
    pub shard_idx: Option<usize>,
    pub file_idx: Option<usize>,
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

    #[getter]
    fn width(&self) -> Option<u32> {
        self.width
    }

    #[getter]
    fn height(&self) -> Option<u32> {
        self.height
    }

    #[getter]
    fn aspect(&self) -> Option<f32> {
        self.aspect
    }

    /// Get all metadata as a dictionary
    #[getter]
    fn metadata(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("path", &self.path)?;
        dict.set_item("offset", self.offset)?;
        dict.set_item("size", self.size)?;

        if let Some(w) = self.width {
            dict.set_item("width", w)?;
        }
        if let Some(h) = self.height {
            dict.set_item("height", h)?;
        }
        if let Some(a) = self.aspect {
            dict.set_item("aspect", a)?;
        }

        Ok(dict.into())
    }

    /// Get formatted job ID for tracking
    #[getter]
    fn job_id(&self) -> String {
        if let (Some(shard), Some(file)) = (self.shard_idx, self.file_idx) {
            format!("shard{:04}_file{:06}", shard, file)
        } else {
            // Fallback to path-based ID
            let path_hash = self
                .path
                .chars()
                .fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
            format!("file_{:08x}", path_hash)
        }
    }

    fn __repr__(&self) -> String {
        let mut parts = vec![
            format!("path='{}'", self.path),
            format!("offset={}", self.offset),
            format!("size={}", self.size),
        ];

        if let Some(w) = self.width {
            parts.push(format!("width={}", w));
        }
        if let Some(h) = self.height {
            parts.push(format!("height={}", h));
        }

        format!("TarFileEntry({})", parts.join(", "))
    }
}

#[derive(Debug, Clone)]
pub struct BucketEntry {
    pub shard_idx: usize,
    pub filename: String,
    pub file_info: crate::metadata::FileInfo,
    pub original_size: Option<(u32, u32)>,
}

pub fn create_tar_entry(
    path: String,
    file_info: &FileInfo,
    data: Vec<u8>,
    shard_idx: Option<usize>,
    file_idx: Option<usize>,
) -> PyTarFileEntry {
    PyTarFileEntry {
        path,
        offset: file_info.offset,
        size: file_info.length,
        data,
        width: file_info.width,
        height: file_info.height,
        aspect: file_info.aspect,
        shard_idx,
        file_idx,
    }
}
