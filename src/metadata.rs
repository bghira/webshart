use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    },
    /// Alternative format with Vec (common in some webdatasets)
    Vec {
        #[serde(default)]
        path: Option<String>,
        #[serde(default)]
        filesize: u64,
        files: Vec<FileInfo>,
    },
}

/// Unified metadata interface
#[derive(Debug, Clone)]
pub struct ShardMetadata {
    pub path: String,
    pub filesize: u64,
    pub hash: Option<String>,
    pub hash_lfs: Option<String>,
    files: Vec<FileInfoInternal>, // Internal storage with guaranteed path
}

/// Internal file info with guaranteed path
#[derive(Debug, Clone)]
struct FileInfoInternal {
    pub path: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: Option<String>,
}

impl From<FileInfo> for FileInfoInternal {
    fn from(info: FileInfo) -> Self {
        Self {
            path: info.path.unwrap_or_else(|| String::from("unknown")),
            offset: info.offset,
            length: info.length,
            sha256: info.sha256,
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
        }
    }
}

impl ShardMetadata {
    /// Create from either format
    pub fn from_format(format: ShardMetadataFormat) -> Self {
        match format {
            ShardMetadataFormat::HashMap {
                path,
                filesize,
                hash,
                hash_lfs,
                files,
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
                    files: file_vec,
                }
            }
            ShardMetadataFormat::Vec {
                path,
                filesize,
                files,
            } => {
                let file_vec: Vec<FileInfoInternal> =
                    files.into_iter().map(FileInfoInternal::from).collect();

                Self {
                    path: path.unwrap_or_else(|| String::from("unknown")),
                    filesize,
                    hash: None,
                    hash_lfs: None,
                    files: file_vec,
                }
            }
        }
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

    /// Get file by index
    pub fn get_file_by_index(&self, index: usize) -> Option<(String, FileInfo)> {
        self.files
            .get(index)
            .map(|info| (info.path.clone(), FileInfo::from(info)))
    }
}

// Custom deserializer that tries both formats
impl<'de> Deserialize<'de> for ShardMetadata {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let format = ShardMetadataFormat::deserialize(deserializer)?;
        Ok(ShardMetadata::from_format(format))
    }
}

impl Serialize for ShardMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
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
        }

        let helper = Helper {
            path: &self.path,
            filesize: self.filesize,
            hash: &self.hash,
            hash_lfs: &self.hash_lfs,
            files: files_map,
        };

        helper.serialize(serializer)
    }
}
