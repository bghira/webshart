use pyo3::prelude::*;
use pyo3::types::PyDict;

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub load_file_data: bool,
    pub max_file_size: u64,
    pub buffer_size: usize,
    pub chunk_size_mb: usize,
    pub hf_token: Option<String>,
    pub batch_size: Option<usize>,
}

impl DataLoaderConfig {
    pub fn from_state_dict(state_dict: &PyDict) -> Self {
        Self {
            load_file_data: state_dict
                .get_item("load_file_data")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(true),
            max_file_size: state_dict
                .get_item("max_file_size")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(50_000_000),
            buffer_size: state_dict
                .get_item("buffer_size")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(100),
            chunk_size_mb: state_dict
                .get_item("chunk_size_mb")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(10),
            hf_token: state_dict
                .get_item("hf_token")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok()),
            batch_size: state_dict
                .get_item("batch_size")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok()),
        }
    }

    pub fn to_state_dict(&self, dict: &PyDict) -> PyResult<()> {
        dict.set_item("load_file_data", self.load_file_data)?;
        dict.set_item("max_file_size", self.max_file_size)?;
        dict.set_item("buffer_size", self.buffer_size)?;
        dict.set_item("chunk_size_mb", self.chunk_size_mb)?;
        dict.set_item("hf_token", &self.hf_token)?;
        dict.set_item("batch_size", self.batch_size)?;
        Ok(())
    }
}
