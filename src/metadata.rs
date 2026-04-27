use crate::discovery::DiscoveredDataset;
use crate::error::{Result, WebshartError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Caption metadata extracted from paired JSON sidecars.
///
/// This serializes as either a single string or a list of strings so Python
/// callers see `captions: str | list[str] | None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CaptionValue {
    Single(String),
    Multiple(Vec<String>),
}

impl CaptionValue {
    pub fn first(&self) -> Option<&str> {
        match self {
            Self::Single(text) => Some(text.as_str()),
            Self::Multiple(captions) => captions.first().map(String::as_str),
        }
    }
}

/// Information about a single file within a tar shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// Name/path of the file (optional in JSON, as it may be the HashMap key)
    #[serde(skip_serializing_if = "Option::is_none", default)]
    #[serde(alias = "fname", alias = "filename")]
    pub path: Option<String>,

    /// Offset within the tar file
    pub offset: u64,

    /// Length of the file in bytes
    #[serde(alias = "size")]
    pub length: u64,

    /// SHA256 hash of the file (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,

    /// Image width in pixels (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,

    /// Image height in pixels (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,

    /// Image aspect ratio (width / height) (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect: Option<f32>,

    /// Path of a paired JSON metadata file inside the same tar shard.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_path: Option<String>,

    /// Offset of the paired JSON metadata file inside the tar shard.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_offset: Option<u64>,

    /// Length of the paired JSON metadata file in bytes.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_length: Option<u64>,

    /// Caption field extracted from paired JSON metadata.
    #[serde(skip_serializing_if = "Option::is_none", default, alias = "caption")]
    pub captions: Option<CaptionValue>,

    /// Parsed paired JSON metadata when it is stored directly in the index.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub json_metadata: Option<Value>,
}

/// Metadata for a single shard - supports both HashMap and Vec formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShardMetadataFormat {
    /// Standard format with HashMap
    HashMap {
        #[serde(default)]
        path: Option<String>,
        filesize: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        hash: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        hash_lfs: Option<String>,
        files: HashMap<String, FileInfo>,
        /// Whether image geometry was extracted
        #[serde(default)]
        includes_image_geometry: bool,
    },
    /// Alternative format with Vec (common in some webdatasets)
    Vec {
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        filesize: u64,
        files: Vec<FileInfo>,
        /// Whether image geometry was extracted
        #[serde(default)]
        includes_image_geometry: bool,
    },
}

/// Unified metadata interface
#[derive(Debug, Clone)]
pub struct ShardMetadata {
    pub path: String,
    pub filesize: u64,
    pub hash: Option<String>,
    pub hash_lfs: Option<String>,
    pub includes_image_geometry: bool,
    files: Vec<FileInfoInternal>, // Internal storage with guaranteed path
    sample_indices: Vec<usize>,   // Logical samples, excluding paired JSON sidecars
}

impl ShardMetadata {
    /// Get files as a vector of FileInfo
    pub fn files(&self) -> Vec<FileInfo> {
        self.files.iter().map(FileInfo::from).collect()
    }

    /// Iterate over files without materializing the entire metadata table.
    pub fn iter_files(&self) -> impl Iterator<Item = (&str, FileInfo)> + '_ {
        self.files
            .iter()
            .map(|info| (info.path.as_str(), FileInfo::from(info)))
    }

    /// Return a bounded range of file entries without cloning unrelated files.
    pub fn file_range(&self, start: usize, end: usize) -> Vec<(String, FileInfo)> {
        self.files[start..end]
            .iter()
            .map(|info| (info.path.clone(), FileInfo::from(info)))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize)]
struct FileInfoInternal {
    pub path: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub aspect: Option<f32>,
    pub json_path: Option<String>,
    pub json_offset: Option<u64>,
    pub json_length: Option<u64>,
    pub captions: Option<CaptionValue>,
    pub json_metadata: Option<Value>,
}

impl From<FileInfo> for FileInfoInternal {
    fn from(info: FileInfo) -> Self {
        Self {
            path: info.path.unwrap_or_else(|| String::from("unknown")),
            offset: info.offset,
            length: info.length,
            sha256: info.sha256,
            width: info.width,
            height: info.height,
            aspect: info.aspect,
            json_path: info.json_path,
            json_offset: info.json_offset,
            json_length: info.json_length,
            captions: info.captions,
            json_metadata: info.json_metadata,
        }
    }
}

impl From<&FileInfoInternal> for FileInfo {
    fn from(info: &FileInfoInternal) -> Self {
        Self {
            path: Some(info.path.clone()),
            offset: info.offset,
            length: info.length,
            sha256: info.sha256.clone(),
            width: info.width,
            height: info.height,
            aspect: info.aspect,
            json_path: info.json_path.clone(),
            json_offset: info.json_offset,
            json_length: info.json_length,
            captions: info.captions.clone(),
            json_metadata: info.json_metadata.clone(),
        }
    }
}

impl ShardMetadata {
    fn sample_key(path: &str) -> String {
        match path.rsplit_once('.') {
            Some((stem, _)) => stem.to_string(),
            None => path.to_string(),
        }
    }

    fn is_json_path(path: &str) -> bool {
        path.rsplit_once('.')
            .map(|(_, ext)| ext.eq_ignore_ascii_case("json"))
            .unwrap_or(false)
    }

    fn infer_json_sidecars(&mut self) {
        let mut json_by_key: HashMap<String, (String, u64, u64)> = HashMap::new();
        let mut non_json_keys: HashSet<String> = HashSet::new();

        for file in &self.files {
            let key = Self::sample_key(&file.path);
            if Self::is_json_path(&file.path) {
                json_by_key.insert(key, (file.path.clone(), file.offset, file.length));
            } else {
                non_json_keys.insert(key);
            }
        }

        for file in &mut self.files {
            if Self::is_json_path(&file.path) {
                continue;
            }

            let key = Self::sample_key(&file.path);
            if let Some((json_path, json_offset, json_length)) = json_by_key.get(&key) {
                if file.json_path.is_none() {
                    file.json_path = Some(json_path.clone());
                }
                if file.json_offset.is_none() {
                    file.json_offset = Some(*json_offset);
                }
                if file.json_length.is_none() {
                    file.json_length = Some(*json_length);
                }
            }
        }

        for file in &mut self.files {
            if Self::is_json_path(&file.path) {
                let key = Self::sample_key(&file.path);
                if non_json_keys.contains(&key) {
                    file.json_path = None;
                    file.json_offset = None;
                    file.json_length = None;
                }
            }
        }
    }

    fn rebuild_sample_index(&mut self) {
        let non_json_keys: HashSet<String> = self
            .files
            .iter()
            .filter(|info| !Self::is_json_path(&info.path))
            .map(|info| Self::sample_key(&info.path))
            .collect();

        self.sample_indices = self
            .files
            .iter()
            .enumerate()
            .filter_map(|(idx, info)| {
                let is_paired_json = Self::is_json_path(&info.path)
                    && non_json_keys.contains(&Self::sample_key(&info.path));
                if is_paired_json {
                    None
                } else {
                    Some(idx)
                }
            })
            .collect();
    }

    fn extract_caption_value(value: &Value) -> Option<CaptionValue> {
        let keys = [
            "caption",
            "captions",
            "text",
            "txt",
            "description",
            "descriptions",
            "prompt",
            "alt_text",
        ];

        let mut captions = Vec::new();
        let mut seen = HashSet::new();
        if let Some(obj) = value.as_object() {
            for key in keys {
                if let Some(field) = obj.get(key) {
                    match field {
                        Value::String(text) if !text.is_empty() => {
                            if seen.insert(text.clone()) {
                                captions.push(text.clone());
                            }
                        }
                        Value::Array(items) => {
                            for item in items {
                                if let Some(text) = item.as_str() {
                                    if !text.is_empty() && seen.insert(text.to_string()) {
                                        captions.push(text.to_string());
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        match captions.len() {
            0 => None,
            1 => Some(CaptionValue::Single(captions.remove(0))),
            _ => Some(CaptionValue::Multiple(captions)),
        }
    }

    pub fn attach_json_metadata(&mut self, json_by_path: &HashMap<String, Value>) {
        for file in &mut self.files {
            if let Some(json_path) = &file.json_path {
                if let Some(value) = json_by_path.get(json_path) {
                    let captions = Self::extract_caption_value(value);
                    if file.captions.is_none() {
                        file.captions = captions;
                    }
                    if file.json_metadata.is_none() {
                        file.json_metadata = Some(value.clone());
                    }
                }
            }
        }
    }

    /// Create from either format
    pub fn from_format(format: ShardMetadataFormat) -> Self {
        let mut metadata = match format {
            ShardMetadataFormat::HashMap {
                path,
                filesize,
                hash,
                hash_lfs,
                files,
                includes_image_geometry,
            } => {
                // Convert HashMap to Vec, setting the path from the HashMap key
                let mut file_vec: Vec<FileInfoInternal> = files
                    .into_iter()
                    .map(|(filename, mut file_info)| {
                        // Set the path from the HashMap key if not already set
                        if file_info.path.is_none() {
                            file_info.path = Some(filename);
                        }
                        FileInfoInternal::from(file_info)
                    })
                    .collect();
                file_vec.sort_by(|a, b| a.path.cmp(&b.path));

                Self {
                    path: path.unwrap_or_else(|| String::from("unknown")),
                    filesize,
                    hash,
                    hash_lfs,
                    includes_image_geometry,
                    files: file_vec,
                    sample_indices: Vec::new(),
                }
            }
            ShardMetadataFormat::Vec {
                path,
                filesize,
                files,
                includes_image_geometry,
            } => {
                let mut file_vec: Vec<FileInfoInternal> =
                    files.into_iter().map(FileInfoInternal::from).collect();
                file_vec.sort_by(|a, b| a.path.cmp(&b.path));

                Self {
                    path: path.unwrap_or_else(|| String::from("unknown")),
                    filesize,
                    hash: None,
                    hash_lfs: None,
                    includes_image_geometry,
                    files: file_vec,
                    sample_indices: Vec::new(),
                }
            }
        };

        metadata.infer_json_sidecars();
        metadata.rebuild_sample_index();
        metadata
    }

    /// Get the number of files in this shard
    pub fn num_files(&self) -> usize {
        self.files.len()
    }

    /// Get file info by name
    pub fn get_file(&self, name: &str) -> Option<FileInfo> {
        self.files
            .iter()
            .find(|f| f.path == name)
            .map(FileInfo::from)
    }

    /// Get all filenames in order
    pub fn filenames(&self) -> Vec<String> {
        self.files.iter().map(|f| f.path.clone()).collect()
    }

    /// Get the number of logical samples, excluding paired JSON sidecars.
    pub fn num_samples(&self) -> usize {
        self.sample_indices.len()
    }

    /// Get sample filenames in order, excluding paired JSON sidecars.
    pub fn sample_filenames(&self) -> Vec<String> {
        self.sample_indices
            .iter()
            .filter_map(|idx| self.files.get(*idx))
            .map(|info| info.path.clone())
            .collect()
    }

    /// Get sample by logical sample index, excluding paired JSON sidecars.
    pub fn get_sample_by_index(&self, index: usize) -> Option<(String, FileInfo)> {
        self.sample_indices
            .get(index)
            .and_then(|idx| self.files.get(*idx))
            .map(|info| (info.path.clone(), FileInfo::from(info)))
    }

    /// Return a bounded range of logical samples without cloning unrelated files.
    pub fn sample_range(&self, start: usize, end: usize) -> Vec<(String, FileInfo)> {
        self.sample_indices
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .filter_map(|idx| self.files.get(*idx))
            .map(|info| (info.path.clone(), FileInfo::from(info)))
            .collect()
    }

    /// Get file by index
    pub fn get_file_by_index(&self, index: usize) -> Option<(String, FileInfo)> {
        self.files
            .get(index)
            .map(|info| (info.path.clone(), FileInfo::from(info)))
    }
}

// Custom deserializer that tries both formats
impl<'de> Deserialize<'de> for ShardMetadata {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let format = ShardMetadataFormat::deserialize(deserializer)?;
        Ok(ShardMetadata::from_format(format))
    }
}

impl Serialize for ShardMetadata {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize back to HashMap format for compatibility
        let mut files_map = HashMap::new();
        for file in &self.files {
            files_map.insert(
                file.path.clone(),
                FileInfo {
                    path: Some(file.path.clone()),
                    offset: file.offset,
                    length: file.length,
                    sha256: file.sha256.clone(),
                    width: file.width,
                    height: file.height,
                    aspect: file.aspect,
                    json_path: file.json_path.clone(),
                    json_offset: file.json_offset,
                    json_length: file.json_length,
                    captions: file.captions.clone(),
                    json_metadata: file.json_metadata.clone(),
                },
            );
        }

        #[derive(Serialize)]
        struct Helper<'a> {
            path: &'a str,
            filesize: u64,
            #[serde(skip_serializing_if = "Option::is_none")]
            hash: &'a Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            hash_lfs: &'a Option<String>,
            files: HashMap<String, FileInfo>,
            includes_image_geometry: bool,
        }

        let helper = Helper {
            path: &self.path,
            filesize: self.filesize,
            hash: &self.hash,
            hash_lfs: &self.hash_lfs,
            files: files_map,
            includes_image_geometry: self.includes_image_geometry,
        };

        helper.serialize(serializer)
    }
}

pub fn ensure_shard_metadata_with_retry(
    dataset: &mut DiscoveredDataset,
    shard_idx: usize,
) -> Result<()> {
    let mut attempts = 0;
    const MAX_ATTEMPTS: u32 = 5;

    loop {
        match dataset.ensure_shard_metadata(shard_idx) {
            Ok(_) => return Ok(()),
            Err(e) => {
                if attempts >= MAX_ATTEMPTS {
                    return Err(e.into());
                }

                if matches!(e, WebshartError::RateLimited) || e.to_string().contains("429") {
                    attempts += 1;
                    let wait_time = Duration::from_secs(2u64.pow(attempts));
                    eprintln!(
                        "[webshart] Rate limited, waiting {:?} before retry",
                        wait_time
                    );
                    std::thread::sleep(wait_time);
                } else {
                    return Err(e.into());
                }
            }
        }
    }
}
