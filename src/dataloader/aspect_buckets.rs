use crate::FileInfo;
use crate::dataloader::PyTarDataLoader;
use crate::error::{Result, WebshartError};
use pyo3::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct AspectBuckets {
    pub buckets: BTreeMap<String, Vec<(String, FileInfo, Option<(u32, u32)>)>>,
    pub shard_idx: usize,
    pub shard_name: String,
}

#[pyclass]
pub struct AspectBucketIterator {
    pub loader: Py<PyTarDataLoader>,
    pub key_type: String,
    pub target_pixel_area: Option<u32>,
    pub target_resolution_multiple: u32,
    pub round_to: Option<usize>,
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
            let shard_idx = slf.current_shard;
            slf.current_shard += 1;
            let loader = slf.loader.borrow(py);
            let result = loader.list_shard_aspect_buckets(
                py,
                vec![shard_idx],
                &slf.key_type,
                slf.target_pixel_area,
                slf.target_resolution_multiple,
                slf.round_to,
            )?;

            if let Some(first) = result.first() {
                Ok(Some(first.clone()))
            } else {
                Ok(None)
            }
        })
    }
}

pub fn scale_dimensions_with_multiple(
    width: u32,
    height: u32,
    target_pixel_area: u32,
    multiple: u32,
) -> (u32, u32) {
    // Target resolution is the desired total area (width * height)
    // let aspect_ratio = width as f64 / height as f64;
    let current_area = (width as f64) * (height as f64);
    let scale_factor = (target_pixel_area as f64 / current_area).sqrt();

    // Calculate new dimensions maintaining aspect ratio
    let new_width = (width as f64 * scale_factor).round() as u32;
    let new_height = (height as f64 * scale_factor).round() as u32;

    // Round to nearest multiple
    let rounded_width = ((new_width as f32 / multiple as f32).round() * multiple as f32) as u32;
    let rounded_height = ((new_height as f32 / multiple as f32).round() * multiple as f32) as u32;

    // Ensure minimum size is at least the multiple
    (rounded_width.max(multiple), rounded_height.max(multiple))
}

/// Formats an aspect ratio with optional rounding
pub fn format_aspect(aspect: f32, round_to: Option<usize>) -> String {
    match round_to {
        Some(decimals) => format!("{:.prec$}", aspect, prec = decimals),
        None => aspect.to_string(),
    }
}

#[derive(Debug, Clone)]
pub enum BucketKeyType {
    Aspect,
    GeometryTuple,
    GeometryList,
}

impl BucketKeyType {
    pub fn parse(key: &str) -> Result<Self> {
        match key {
            "aspect" => Ok(Self::Aspect),
            "geometry-tuple" => Ok(Self::GeometryTuple),
            "geometry-list" => Ok(Self::GeometryList),
            _ => Err(WebshartError::InvalidShardFormat(
                "key must be 'aspect', 'geometry-tuple', or 'geometry-list'".to_string(),
            )),
        }
    }

    pub fn format_dimensions(&self, width: u32, height: u32) -> String {
        match self {
            Self::Aspect => {
                let aspect = width as f32 / height as f32;
                format_aspect(aspect, Some(2))
            }
            Self::GeometryTuple => format!("({}, {})", width, height),
            Self::GeometryList => format!("[{}, {}]", width, height),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BucketSamplingStrategy {
    Sequential,
    RandomWithinBuckets,
    FullyRandom,
}

impl BucketSamplingStrategy {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "sequential" => Ok(Self::Sequential),
            "random_within_buckets" => Ok(Self::RandomWithinBuckets),
            "fully_random" => Ok(Self::FullyRandom),
            _ => Err(WebshartError::InvalidShardFormat(
                "sampling_strategy must be 'sequential', 'random_within_buckets', or 'fully_random'".to_string()
            )),
        }
    }
}

pub fn calculate_bucket_key(
    key_type: &BucketKeyType,
    width: u32,
    height: u32,
    aspect: Option<f32>,
    target_pixel_area: Option<u32>,
    target_resolution_multiple: u32,
    round_to: Option<usize>,
) -> (String, Option<(u32, u32)>) {
    // Extract common scaling logic
    let (final_width, final_height, original_size) = if let Some(target_res) = target_pixel_area {
        let (scaled_w, scaled_h) =
            scale_dimensions_with_multiple(width, height, target_res, target_resolution_multiple);
        let orig = if scaled_w != width || scaled_h != height {
            Some((width, height))
        } else {
            None
        };
        (scaled_w, scaled_h, orig)
    } else {
        (width, height, None)
    };

    // Format based on key type
    let key = match key_type {
        BucketKeyType::Aspect => {
            let final_aspect = if target_pixel_area.is_some() {
                final_width as f32 / final_height as f32
            } else {
                aspect.unwrap_or(width as f32 / height as f32)
            };
            format_aspect(final_aspect, round_to)
        }
        _ => key_type.format_dimensions(final_width, final_height),
    };

    (key, original_size)
}
