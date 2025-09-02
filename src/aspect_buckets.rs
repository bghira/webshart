use crate::FileInfo;
use crate::dataloader::PyTarDataLoader;
use pyo3::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct AspectBuckets {
    pub buckets: BTreeMap<String, Vec<(String, FileInfo)>>,
    pub shard_idx: usize,
    pub shard_name: String,
}

#[pyclass]
pub struct AspectBucketIterator {
    pub loader: Py<PyTarDataLoader>,
    pub key_type: String,
    pub target_resolution: Option<u32>,
    pub current_shard: usize,
    pub num_shards: usize,
}

#[pymethods]
impl AspectBucketIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<PyObject>> {
        if slf.current_shard >= slf.num_shards {
            return Ok(None);
        }

        Python::with_gil(|py| {
            slf.current_shard += 1;
            let loader = slf.loader.borrow(py);
            let result = loader.list_shard_aspect_buckets(
                py,
                vec![slf.current_shard],
                &slf.key_type,
                slf.target_resolution,
            )?;

            if let Some(first) = result.first() {
                Ok(Some(first.clone()))
            } else {
                Ok(None)
            }
        })
    }
}
