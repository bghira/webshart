use crate::error::{Result, WebshartError};
use crate::metadata::ShardMetadata;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Represents a discovered shard pair (tar + json)
#[derive(Debug, Clone)]
pub struct ShardPair {
    /// Base name without extension (e.g., "data-0000")
    pub name: String,

    /// Path/URL to the tar file
    pub tar_path: String,

    /// Path/URL to the json file
    pub json_path: String,

    /// Loaded metadata (lazy loaded)
    pub metadata: Option<ShardMetadata>,
}

/// Represents a discovered dataset with all its shards
#[derive(Debug, Clone)]
pub struct DiscoveredDataset {
    /// Dataset name/path
    pub name: String,

    /// Whether this is a remote dataset
    pub is_remote: bool,

    /// All discovered shard pairs
    pub shards: Vec<ShardPair>,

    /// Discovery service for lazy loading
    discovery_token: Option<String>,

    /// Cached total size from HF API (for remote datasets)
    cached_total_size: Option<u64>,

    /// Cached total files from HF API (for remote datasets)
    cached_total_files: Option<usize>,

    /// Runtime for async operations
    runtime: Arc<Runtime>,
}

impl DiscoveredDataset {
    /// Get total number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Get the HuggingFace token if available
    pub fn get_hf_token(&self) -> Option<String> {
        self.discovery_token.clone()
    }

    /// Get total number of files (requires loading all metadata)
    pub fn total_files(&mut self) -> Result<usize> {
        self.ensure_all_metadata_loaded()?;
        Ok(self
            .shards
            .iter()
            .filter_map(|s| s.metadata.as_ref())
            .map(|m| m.num_files())
            .sum())
    }

    /// Get total size (requires loading all metadata)
    pub fn total_size(&mut self) -> Result<u64> {
        self.ensure_all_metadata_loaded()?;
        Ok(self
            .shards
            .iter()
            .filter_map(|s| s.metadata.as_ref())
            .map(|m| m.filesize)
            .sum())
    }

    /// Get quick stats from cached values (instant, no metadata loading)
    pub fn quick_stats(&self) -> (Option<u64>, Option<usize>) {
        (self.cached_total_size, self.cached_total_files)
    }

    /// Ensure metadata is loaded for a specific shard
    pub fn ensure_shard_metadata(&mut self, shard_index: usize) -> Result<()> {
        if let Some(shard) = self.shards.get_mut(shard_index) {
            if shard.metadata.is_none() {
                let discovery = DatasetDiscovery::with_runtime(self.runtime.clone())
                    .with_optional_token(self.discovery_token.clone());

                let metadata = if self.is_remote {
                    self.runtime
                        .block_on(discovery.load_remote_metadata(&shard.json_path))?
                } else {
                    discovery.load_local_metadata(&shard.json_path)?
                };
                shard.metadata = Some(metadata);
            }
            Ok(())
        } else {
            Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range for shard {}",
                shard_index, self.name
            )))
        }
    }

    /// Ensure all metadata is loaded
    fn ensure_all_metadata_loaded(&mut self) -> Result<()> {
        // Use index-based loop to avoid borrow checker issues
        let num_shards = self.shards.len();
        for i in 0..num_shards {
            self.ensure_shard_metadata(i)?;
        }
        Ok(())
    }

    /// Find which shard contains a file by global index (loads metadata as needed)
    pub fn find_shard_for_file(&mut self, file_index: usize) -> Result<Option<(usize, usize)>> {
        let mut current_offset = 0;

        // Use index-based loop to avoid borrow checker issues
        for shard_idx in 0..self.shards.len() {
            self.ensure_shard_metadata(shard_idx)?;

            if let Some(metadata) = &self.shards[shard_idx].metadata {
                let num_files = metadata.num_files();
                if file_index < current_offset + num_files {
                    return Ok(Some((shard_idx, file_index - current_offset)));
                }
                current_offset += num_files;
            }
        }

        Ok(None)
    }

    /// Open a shard for reading
    pub fn open_shard(&mut self, shard_index: usize) -> Result<ShardReader> {
        // Ensure metadata is loaded
        self.ensure_shard_metadata(shard_index)?;

        if let Some(shard) = self.shards.get(shard_index) {
            if let Some(metadata) = &shard.metadata {
                ShardReader::new(
                    &shard.tar_path,
                    self.is_remote,
                    metadata.clone(),
                    self.discovery_token.clone(),
                    self.runtime.clone(),
                )
            } else {
                Err(WebshartError::InvalidShardFormat(
                    "Metadata not loaded".to_string(),
                ))
            }
        } else {
            Err(WebshartError::InvalidShardFormat(format!(
                "Shard index {} out of range",
                shard_index
            )))
        }
    }
}

/// Reader for accessing files within a shard
pub struct ShardReader {
    /// Path or URL to the tar file
    tar_location: String,

    /// Whether this is a remote shard
    is_remote: bool,

    /// Shard metadata
    metadata: ShardMetadata,

    /// Optional HuggingFace token for remote access
    hf_token: Option<String>,

    /// Runtime for async operations
    runtime: Arc<Runtime>,
}

impl ShardReader {
    /// Create a new shard reader
    pub fn new(
        tar_location: &str,
        is_remote: bool,
        metadata: ShardMetadata,
        hf_token: Option<String>,
        runtime: Arc<Runtime>,
    ) -> Result<Self> {
        Ok(Self {
            tar_location: tar_location.to_string(),
            is_remote,
            metadata,
            hf_token,
            runtime,
        })
    }

    /// Read a file by index within this shard
    pub fn read_file(&self, file_index: usize) -> Result<Vec<u8>> {
        if file_index >= self.metadata.num_files() {
            return Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range for shard",
                file_index
            )));
        }

        // Use get_file_by_index to access files by numeric index
        let (filename, file_info) =
            self.metadata.get_file_by_index(file_index).ok_or_else(|| {
                WebshartError::InvalidShardFormat(format!(
                    "File index {} not found in metadata",
                    file_index
                ))
            })?;

        if self.is_remote {
            self.runtime.block_on(self.read_file_remote(
                &filename,
                file_info.offset,
                file_info.length,
            ))
        } else {
            self.read_file_local(&filename, file_info.offset, file_info.length)
        }
    }

    /// Read a file from remote tar archive using HTTP range requests
    async fn read_file_remote(&self, _filename: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        let client = reqwest::Client::new();

        // For this dataset, offsets point directly to file content
        // Just read the bytes from offset to offset+length
        let mut request = client
            .get(&self.tar_location)
            .header("Range", format!("bytes={}-{}", offset, offset + length - 1));

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;
        if !response.status().is_success() {
            return Err(WebshartError::InvalidShardFormat(format!(
                "Failed to read file content: {}",
                response.status()
            )));
        }

        Ok(response.bytes().await?.to_vec())
    }

    /// Read a file from local tar archive
    fn read_file_local(&self, filename: &str, offset: u64, length: u64) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = fs::File::open(&self.tar_location)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer)?;

        // Debug: verify WEBP files
        if filename.ends_with(".webp") && buffer.len() >= 12 {
            let riff = &buffer[0..4];
            let webp = &buffer[8..12];
            if riff != b"RIFF" || webp != b"WEBP" {
                eprintln!(
                    "[webshart] Warning: {} doesn't look like a valid WEBP file",
                    filename
                );
            }
        }

        Ok(buffer)
    }

    /// Get list of filenames in this shard
    pub fn filenames(&self) -> Vec<String> {
        self.metadata.filenames()
    }

    /// Get number of files in this shard
    pub fn num_files(&self) -> usize {
        self.metadata.num_files()
    }
}

/// Discovery service for finding dataset shards
#[derive(Clone)]
pub struct DatasetDiscovery {
    hf_token: Option<String>,
    shard_pattern: Regex,
    client: reqwest::Client,
    runtime: Arc<Runtime>,
}

impl DatasetDiscovery {
    /// Create a new discovery service
    pub fn new() -> Self {
        Self {
            hf_token: None,
            // Match patterns like: data-0000.tar, shard_001.tar, etc.
            shard_pattern: Regex::new(r"^(.+?)\.tar$").unwrap(),
            client: reqwest::Client::new(),
            runtime: Arc::new(Runtime::new().expect("Failed to create Tokio runtime")),
        }
    }

    /// Create with existing runtime
    pub fn with_runtime(runtime: Arc<Runtime>) -> Self {
        Self {
            hf_token: None,
            shard_pattern: Regex::new(r"^(.+?)\.tar$").unwrap(),
            client: reqwest::Client::new(),
            runtime,
        }
    }

    /// Set HuggingFace token
    pub fn with_hf_token(mut self, token: String) -> Self {
        self.hf_token = Some(token);
        self
    }

    /// Set optional token
    pub fn with_optional_token(mut self, token: Option<String>) -> Self {
        self.hf_token = token;
        self
    }

    /// Set custom shard pattern
    pub fn with_pattern(mut self, pattern: &str) -> Result<Self> {
        self.shard_pattern =
            Regex::new(pattern).map_err(|e| WebshartError::InvalidShardFormat(e.to_string()))?;
        Ok(self)
    }

    /// Discover shards in a local directory
    pub fn discover_local(&self, path: &Path) -> Result<DiscoveredDataset> {
        let mut shards = Vec::new();

        // Find all tar files
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_name = entry.file_name().to_string_lossy().to_string();

            if let Some(captures) = self.shard_pattern.captures(&file_name) {
                let base_name = captures.get(1).unwrap().as_str();
                let json_name = format!("{}.json", base_name);
                let json_path = path.join(&json_name);

                // Check if corresponding JSON exists
                if json_path.exists() {
                    shards.push(ShardPair {
                        name: base_name.to_string(),
                        tar_path: entry.path().to_string_lossy().to_string(),
                        json_path: json_path.to_string_lossy().to_string(),
                        metadata: None,
                    });
                }
            }
        }

        if shards.is_empty() {
            return Err(WebshartError::NoShardsFound);
        }

        // Sort shards by name
        shards.sort_by(|a, b| a.name.cmp(&b.name));

        // Create dataset - no cached values for local datasets
        Ok(DiscoveredDataset {
            name: path.to_string_lossy().to_string(),
            is_remote: false,
            shards,
            discovery_token: self.hf_token.clone(),
            cached_total_size: None,
            cached_total_files: None,
            runtime: self.runtime.clone(),
        })
    }

    /// Discover shards from HuggingFace Hub with pagination support
    pub async fn discover_huggingface(
        &self,
        repo_id: &str,
        subfolder: Option<&str>,
    ) -> Result<DiscoveredDataset> {
        let mut all_shards = Vec::new();

        // Try to use the dataset info API first (no pagination needed)
        if let Some(folder) = subfolder {
            // Specific subfolder requested
            println!("[webshart] Discovering shards in {}/{}", repo_id, folder);
            match self
                .discover_shards_from_dataset_info(repo_id, subfolder)
                .await
            {
                Ok(shards) => {
                    println!(
                        "[webshart] Successfully discovered {} shards from dataset info",
                        shards.len()
                    );
                    all_shards = shards;
                }
                Err(_) => {
                    // Fall back to tree API
                    println!("[webshart] Falling back to tree API for subfolder discovery");
                    all_shards = self.discover_shards_in_folder(repo_id, subfolder).await?;
                }
            }
        } else {
            // No subfolder specified - discover all
            println!("[webshart] Discovering all shards in {}", repo_id);
            match self.discover_shards_from_dataset_info(repo_id, None).await {
                Ok(shards) => {
                    // Group by directory to show what was found
                    let mut dirs: HashMap<String, usize> = HashMap::new();
                    for shard in &shards {
                        // Extract directory from path
                        let path_parts: Vec<&str> = shard.tar_path.split('/').collect();
                        let dir = if let Some(pos) = path_parts
                            .iter()
                            .position(|&p| p == "resolve" || p == "main")
                        {
                            if pos + 2 < path_parts.len() && path_parts[pos + 2].ends_with(".tar") {
                                "root"
                            } else if pos + 2 < path_parts.len() {
                                path_parts[pos + 2]
                            } else {
                                "root"
                            }
                        } else {
                            "unknown"
                        };
                        *dirs.entry(dir.to_string()).or_insert(0) += 1;
                    }

                    println!("[webshart] Found {} total shards:", shards.len());
                    for (dir, count) in dirs.iter() {
                        println!("  - {}: {} shards", dir, count);
                    }

                    all_shards = shards;
                }
                Err(e) => {
                    println!(
                        "[webshart] Dataset info API failed: {}, falling back to tree API",
                        e
                    );

                    // Fall back to tree API with subdirectory discovery
                    // First, get the root directory listing
                    let root_url =
                        format!("https://huggingface.co/api/datasets/{}/tree/main", repo_id);
                    let mut request = self.client.get(&root_url);

                    if let Some(token) = &self.hf_token {
                        request = request.bearer_auth(token);
                    }

                    let response = request.send().await?;
                    if !response.status().is_success() {
                        return Err(WebshartError::DiscoveryFailed(format!(
                            "Failed to list root directory: {}",
                            response.status()
                        )));
                    }

                    #[derive(Deserialize)]
                    struct FileInfo {
                        path: String,
                        #[serde(rename = "type")]
                        file_type: String,
                    }

                    let items: Vec<FileInfo> = response.json().await?;
                    let mut subdirs = Vec::new();
                    let mut has_root_tars = false;

                    // Check for subdirectories and root tar files
                    for item in items {
                        if item.file_type == "directory" {
                            subdirs.push(item.path);
                        } else if item.file_type == "file" && item.path.ends_with(".tar") {
                            has_root_tars = true;
                        }
                    }

                    // If we have subdirectories, search each one
                    if !subdirs.is_empty() {
                        println!(
                            "[webshart] Found {} subdirectories to search",
                            subdirs.len()
                        );

                        for subdir in subdirs {
                            println!("[webshart] Searching in {}/{}...", repo_id, subdir);
                            if let Ok(shards) =
                                self.discover_shards_in_folder(repo_id, Some(&subdir)).await
                            {
                                all_shards.extend(shards);
                            }
                        }
                    }

                    // Also check root if it has tar files
                    if has_root_tars {
                        println!("[webshart] Searching in root directory...");
                        if let Ok(shards) = self.discover_shards_in_folder(repo_id, None).await {
                            all_shards.extend(shards);
                        }
                    }
                }
            }
        }

        if all_shards.is_empty() {
            return Err(WebshartError::NoShardsFound);
        }

        // Sort shards by name
        all_shards.sort_by(|a, b| a.name.cmp(&b.name));

        println!("[webshart] Total discovered shards: {}", all_shards.len());

        // Fetch dataset size information from HF API
        let (cached_size, cached_files) = self
            .fetch_dataset_size(repo_id)
            .await
            .unwrap_or((None, None));

        // If we used the dataset info API and got a lot of shards, that's likely the real count
        let final_cached_files = if !all_shards.is_empty() {
            // We got actual shard count
            Some(all_shards.len())
        } else {
            cached_files
        };

        // Create dataset without loading metadata
        Ok(DiscoveredDataset {
            name: repo_id.to_string(),
            is_remote: true,
            shards: all_shards,
            discovery_token: self.hf_token.clone(),
            cached_total_size: cached_size,
            cached_total_files: final_cached_files,
            runtime: self.runtime.clone(),
        })
    }

    /// Discover shards in a specific folder
    async fn discover_shards_in_folder(
        &self,
        repo_id: &str,
        subfolder: Option<&str>,
    ) -> Result<Vec<ShardPair>> {
        let mut all_files = Vec::new();
        let mut cursor: Option<String> = None;
        let mut page_count = 0;

        // Paginate through all files in this folder
        loop {
            let api_url = match (subfolder, &cursor) {
                (Some(folder), Some(cur)) => {
                    // URL encode the cursor to handle special characters
                    let encoded_cursor = cur
                        .replace("+", "%2B")
                        .replace("/", "%2F")
                        .replace("=", "%3D");
                    format!(
                        "https://huggingface.co/api/datasets/{}/tree/main/{}?cursor={}",
                        repo_id, folder, encoded_cursor
                    )
                }
                (Some(folder), None) => format!(
                    "https://huggingface.co/api/datasets/{}/tree/main/{}",
                    repo_id, folder
                ),
                (None, Some(cur)) => {
                    // URL encode the cursor to handle special characters
                    let encoded_cursor = cur
                        .replace("+", "%2B")
                        .replace("/", "%2F")
                        .replace("=", "%3D");
                    format!(
                        "https://huggingface.co/api/datasets/{}/tree/main?cursor={}",
                        repo_id, encoded_cursor
                    )
                }
                (None, None) => {
                    format!("https://huggingface.co/api/datasets/{}/tree/main", repo_id)
                }
            };

            let mut request = self.client.get(&api_url);

            if let Some(token) = &self.hf_token {
                request = request.bearer_auth(token);
            }

            let response = request.send().await?;

            if !response.status().is_success() {
                return Err(WebshartError::DiscoveryFailed(format!(
                    "Failed to list files: {}",
                    response.status()
                )));
            }

            // Check for pagination headers BEFORE consuming the response body
            let headers = response.headers();

            let has_more = headers
                .get("x-has-more")
                .and_then(|v| v.to_str().ok())
                .map(|v| v == "true")
                .unwrap_or(false);

            let next_cursor = headers
                .get("x-cursor")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            #[derive(Deserialize)]
            struct FileInfo {
                path: String,
                #[serde(rename = "type")]
                file_type: String,
            }

            let files: Vec<FileInfo> = response.json().await?;
            let files_in_page = files.len();
            all_files.extend(files);

            page_count += 1;

            // Debug pagination info
            if has_more && next_cursor.is_some() {
                println!(
                    "[webshart] Page {}: fetched {} files, continuing...",
                    page_count, files_in_page
                );
            } else {
                println!(
                    "[webshart] Page {}: fetched {} files (total: {})",
                    page_count,
                    files_in_page,
                    all_files.len()
                );
            }

            if !has_more || next_cursor.is_none() {
                if page_count > 1 {
                    println!("[webshart] Pagination complete after {} pages", page_count);
                }
                break;
            }

            cursor = next_cursor;
        }

        // Process files to find tar/json pairs
        let mut tar_files = HashMap::new();
        let mut json_files = HashMap::new();

        for file in all_files {
            if file.file_type != "file" {
                continue;
            }

            let file_name = Path::new(&file.path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            if let Some(captures) = self.shard_pattern.captures(&file_name) {
                let base_name = captures.get(1).unwrap().as_str();
                tar_files.insert(base_name.to_string(), file.path);
            } else if file_name.ends_with(".json") {
                let base_name = file_name.trim_end_matches(".json");
                json_files.insert(base_name.to_string(), file.path);
            }
        }

        // Match pairs
        let mut shards = Vec::new();
        for (base_name, tar_path) in tar_files {
            if let Some(json_path) = json_files.get(&base_name) {
                let base_url = format!("https://huggingface.co/datasets/{}/resolve/main", repo_id);

                shards.push(ShardPair {
                    name: base_name,
                    tar_path: format!("{}/{}", base_url, tar_path),
                    json_path: format!("{}/{}", base_url, json_path),
                    metadata: None,
                });
            }
        }

        println!(
            "[webshart] Found {} shards in {}",
            shards.len(),
            subfolder.unwrap_or("root")
        );

        Ok(shards)
    }

    /// Discover shards using the dataset info API (avoids pagination)
    async fn discover_shards_from_dataset_info(
        &self,
        repo_id: &str,
        subfolder: Option<&str>,
    ) -> Result<Vec<ShardPair>> {
        let api_url = format!("https://huggingface.co/api/datasets/{}", repo_id);

        let mut request = self.client.get(&api_url);

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(WebshartError::DiscoveryFailed(format!(
                "Failed to get dataset info: {}",
                response.status()
            )));
        }

        let json: Value = response.json().await?;

        // Extract siblings array
        let siblings = json["siblings"].as_array().ok_or_else(|| {
            WebshartError::DiscoveryFailed("No siblings array in dataset info".to_string())
        })?;

        println!("[webshart] Found {} total files in dataset", siblings.len());

        // Process siblings to find tar/json pairs
        let mut tar_files = HashMap::new();
        let mut json_files = HashMap::new();

        for sibling in siblings {
            let path = sibling["rfilename"].as_str().ok_or_else(|| {
                WebshartError::DiscoveryFailed("Invalid sibling entry".to_string())
            })?;

            // Filter by subfolder if specified
            let in_target_folder = if let Some(folder) = subfolder {
                path.starts_with(&format!("{}/", folder))
            } else {
                // If no subfolder specified, skip non-tar/json files to reduce noise
                path.ends_with(".tar") || path.ends_with(".json")
            };

            if !in_target_folder {
                continue;
            }

            let file_name = Path::new(path)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            if let Some(captures) = self.shard_pattern.captures(&file_name) {
                let base_name = captures.get(1).unwrap().as_str();
                tar_files.insert(base_name.to_string(), path.to_string());
            } else if file_name.ends_with(".json") {
                let base_name = file_name.trim_end_matches(".json");
                json_files.insert(base_name.to_string(), path.to_string());
            }
        }

        // Match pairs
        let mut shards = Vec::new();
        for (base_name, tar_path) in tar_files {
            if let Some(json_path) = json_files.get(&base_name) {
                let base_url = format!("https://huggingface.co/datasets/{}/resolve/main", repo_id);

                shards.push(ShardPair {
                    name: base_name,
                    tar_path: format!("{}/{}", base_url, tar_path),
                    json_path: format!("{}/{}", base_url, json_path),
                    metadata: None,
                });
            }
        }

        println!("[webshart] Matched {} shard pairs", shards.len());

        Ok(shards)
    }

    /// Fetch dataset size information from HuggingFace API
    async fn fetch_dataset_size(&self, repo_id: &str) -> Result<(Option<u64>, Option<usize>)> {
        // Try the dataset info endpoint first
        let api_url = format!("https://huggingface.co/api/datasets/{}", repo_id);

        let mut request = self.client.get(&api_url);

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            println!(
                "[webshart] Could not fetch dataset info: {}",
                response.status()
            );
            return Ok((None, None));
        }

        let json: Value = response.json().await?;

        // Try different places where size info might be stored
        // First check top-level size field
        if let Some(size) = json["size"].as_u64() {
            return Ok((Some(size), None));
        }

        // Check siblings for total size calculation
        if let Some(siblings) = json["siblings"].as_array() {
            let mut total_size = 0u64;
            let mut file_count = 0usize;

            for sibling in siblings {
                if let Some(size) = sibling["size"].as_u64() {
                    // Only count .tar files
                    if let Some(filename) = sibling["rfilename"].as_str() {
                        if filename.ends_with(".tar") {
                            total_size += size;
                            file_count += 1;
                        }
                    }
                }
            }

            if total_size > 0 {
                return Ok((Some(total_size), Some(file_count)));
            }
        }

        Ok((None, None))
    }

    /// Load metadata from a local JSON file
    pub fn load_local_metadata(&self, path: &str) -> Result<ShardMetadata> {
        let content = fs::read_to_string(path)?;
        let metadata: ShardMetadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }

    /// Load metadata from a remote JSON file
    pub async fn load_remote_metadata(&self, url: &str) -> Result<ShardMetadata> {
        let mut request = self.client.get(url);

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(WebshartError::MetadataNotFound(format!(
                "Failed to fetch metadata from {}: HTTP {}",
                url,
                response.status()
            )));
        }

        // Get the response text first to help with debugging
        let response_text = response.text().await?;

        // Try to parse as JSON
        match serde_json::from_str::<ShardMetadata>(&response_text) {
            Ok(metadata) => Ok(metadata),
            Err(e) => {
                // Log the first 500 chars of the response for debugging
                let preview = if response_text.len() > 500 {
                    &response_text[..500]
                } else {
                    &response_text
                };

                eprintln!("[webshart] Failed to parse JSON metadata from {}", url);
                eprintln!("[webshart] Parse error: {}", e);
                eprintln!("[webshart] Response preview: {}", preview);

                // Check if it's HTML (common for 404 pages)
                if response_text.trim_start().starts_with("<") {
                    Err(WebshartError::MetadataNotFound(format!(
                        "Expected JSON but got HTML response from {} (likely a 404 or error page)",
                        url
                    )))
                } else {
                    Err(WebshartError::MetadataNotFound(format!(
                        "Invalid JSON in metadata file: {}",
                        e
                    )))
                }
            }
        }
    }
}

/// Python wrapper for DatasetDiscovery
#[pyclass(name = "DatasetDiscovery")]
pub struct PyDatasetDiscovery {
    inner: DatasetDiscovery,
}

#[pymethods]
impl PyDatasetDiscovery {
    #[new]
    #[pyo3(signature = (hf_token=None))]
    fn new(hf_token: Option<String>) -> Self {
        let mut discovery = DatasetDiscovery::new();
        if let Some(token) = hf_token {
            discovery = discovery.with_hf_token(token);
        }
        Self { inner: discovery }
    }

    fn discover_local(&self, path: &str) -> PyResult<PyDiscoveredDataset> {
        let dataset = self.inner.discover_local(Path::new(path))?;
        Ok(PyDiscoveredDataset { inner: dataset })
    }

    #[pyo3(signature = (repo_id, subfolder=None))]
    fn discover_huggingface(
        &self,
        repo_id: &str,
        subfolder: Option<&str>,
    ) -> PyResult<PyDiscoveredDataset> {
        let dataset = self
            .inner
            .runtime
            .block_on(self.inner.discover_huggingface(repo_id, subfolder))?;
        Ok(PyDiscoveredDataset { inner: dataset })
    }

    // Async variants for Python asyncio users
    #[pyo3(signature = (repo_id, subfolder=None))]
    fn discover_huggingface_async<'py>(
        &self,
        py: Python<'py>,
        repo_id: &str,
        subfolder: Option<&str>,
    ) -> PyResult<&'py PyAny> {
        let inner = self.inner.clone();
        let repo_id = repo_id.to_string();
        let subfolder = subfolder.map(|s| s.to_string());

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let dataset = inner
                .discover_huggingface(&repo_id, subfolder.as_deref())
                .await?;
            Ok(PyDiscoveredDataset { inner: dataset })
        })
    }
}

/// Python wrapper for DiscoveredDataset
#[pyclass(name = "DiscoveredDataset")]
pub struct PyDiscoveredDataset {
    pub inner: DiscoveredDataset,
}

#[pymethods]
impl PyDiscoveredDataset {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn is_remote(&self) -> bool {
        self.inner.is_remote
    }

    #[getter]
    fn num_shards(&self) -> usize {
        self.inner.num_shards()
    }

    #[getter]
    fn total_files(&mut self) -> PyResult<usize> {
        Ok(self.inner.total_files()?)
    }

    #[getter]
    fn total_size(&mut self) -> PyResult<u64> {
        Ok(self.inner.total_size()?)
    }

    fn quick_stats(&self) -> (Option<u64>, Option<usize>) {
        self.inner.quick_stats()
    }

    fn get_shard_info(&mut self, index: usize) -> PyResult<Py<PyDict>> {
        // Ensure metadata is loaded for this shard
        self.inner.ensure_shard_metadata(index)?;

        Python::with_gil(|py| {
            if let Some(shard) = self.inner.shards.get(index) {
                let dict = PyDict::new(py);
                dict.set_item("name", &shard.name)?;
                dict.set_item("tar_path", &shard.tar_path)?;
                dict.set_item("json_path", &shard.json_path)?;

                if let Some(metadata) = &shard.metadata {
                    dict.set_item("num_files", metadata.num_files())?;
                    dict.set_item("size", metadata.filesize)?;
                }

                Ok(dict.into())
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Shard index {} out of range",
                    index
                )))
            }
        })
    }

    fn list_files_in_shard(&mut self, shard_index: usize) -> PyResult<Py<PyList>> {
        // Ensure metadata is loaded for this shard
        self.inner.ensure_shard_metadata(shard_index)?;

        Python::with_gil(|py| {
            if let Some(shard) = self.inner.shards.get(shard_index) {
                if let Some(metadata) = &shard.metadata {
                    let filenames = metadata.filenames();
                    let list = PyList::new(py, filenames);
                    Ok(list.into())
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "Failed to load shard metadata",
                    ))
                }
            } else {
                Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Shard index {} out of range",
                    shard_index
                )))
            }
        })
    }

    fn find_file_location(&mut self, file_index: usize) -> PyResult<(usize, usize)> {
        match self.inner.find_shard_for_file(file_index)? {
            Some(location) => Ok(location),
            None => Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "File index {} out of range",
                file_index
            ))),
        }
    }

    fn open_shard(&mut self, shard_index: usize) -> PyResult<PyShardReader> {
        let reader = self.inner.open_shard(shard_index)?;
        Ok(PyShardReader { inner: reader })
    }

    fn get_shard_file_count(&mut self, shard_index: usize) -> PyResult<usize> {
        if shard_index >= self.inner.shards.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Shard index {} out of range. Dataset has {} shards.",
                shard_index,
                self.inner.shards.len()
            )));
        }
        self.inner.ensure_shard_metadata(shard_index)?;
        if let Some(shard) = self.inner.shards.get(shard_index) {
            if let Some(metadata) = &shard.metadata {
                Ok(metadata.num_files())
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Failed to load shard metadata",
                ))
            }
        } else {
            Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Shard index {} out of range",
                shard_index
            )))
        }
    }

    fn get_stats(&mut self) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("total_shards", self.inner.num_shards())?;
            let (cached_size, cached_files) = self.inner.quick_stats();
            if let Some(size) = cached_size {
                dict.set_item("total_size", size)?;
                dict.set_item("total_size_gb", size as f64 / (1024.0_f64).powi(3))?;
            }
            if let Some(files) = cached_files {
                dict.set_item("total_files", files)?;
                dict.set_item(
                    "average_files_per_shard",
                    files as f64 / self.inner.num_shards() as f64,
                )?;
            }
            // Add a flag indicating if these are cached or computed values
            dict.set_item(
                "from_cache",
                cached_size.is_some() || cached_files.is_some(),
            )?;
            let shard_details = PyList::empty(py);
            for (i, shard) in self.inner.shards.iter().enumerate() {
                let shard_dict = PyDict::new(py);
                shard_dict.set_item("index", i)?;
                shard_dict.set_item("name", &shard.name)?;
                if let Some(metadata) = &shard.metadata {
                    shard_dict.set_item("num_files", metadata.num_files())?;
                    shard_dict.set_item("size", metadata.filesize)?;
                    shard_dict.set_item("metadata_loaded", true)?;
                } else {
                    shard_dict.set_item("metadata_loaded", false)?;
                }
                shard_details.append(shard_dict)?;
            }
            dict.set_item("shard_details", shard_details)?;

            Ok(dict.into())
        })
    }

    fn get_detailed_stats(&mut self) -> PyResult<Py<PyDict>> {
        self.inner.ensure_all_metadata_loaded()?;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            let mut total_files = 0usize;
            let mut total_size = 0u64;
            let mut min_files = usize::MAX;
            let mut max_files = 0usize;
            let shard_details = PyList::empty(py);
            for (i, shard) in self.inner.shards.iter().enumerate() {
                if let Some(metadata) = &shard.metadata {
                    let num_files = metadata.num_files();
                    let size = metadata.filesize;
                    total_files += num_files;
                    total_size += size;
                    min_files = min_files.min(num_files);
                    max_files = max_files.max(num_files);
                    // Create detailed shard info
                    let shard_dict = PyDict::new(py);
                    shard_dict.set_item("index", i)?;
                    shard_dict.set_item("name", &shard.name)?;
                    shard_dict.set_item("num_files", num_files)?;
                    shard_dict.set_item("size", size)?;
                    shard_dict.set_item("size_mb", size as f64 / (1024.0_f64).powi(2))?;
                    shard_details.append(shard_dict)?;
                }
            }
            let num_shards = self.inner.num_shards();
            let avg_files = if num_shards > 0 {
                total_files as f64 / num_shards as f64
            } else {
                0.0
            };
            let avg_size = if num_shards > 0 {
                total_size as f64 / num_shards as f64
            } else {
                0.0
            };
            // Set all statistics
            dict.set_item("total_shards", num_shards)?;
            dict.set_item("total_files", total_files)?;
            dict.set_item("total_size", total_size)?;
            dict.set_item("total_size_gb", total_size as f64 / (1024.0_f64).powi(3))?;
            dict.set_item("average_files_per_shard", avg_files)?;
            dict.set_item("average_size_per_shard", avg_size)?;
            dict.set_item("average_size_per_shard_mb", avg_size / (1024.0_f64).powi(2))?;
            dict.set_item(
                "min_files_in_shard",
                if min_files == usize::MAX {
                    0
                } else {
                    min_files
                },
            )?;
            dict.set_item("max_files_in_shard", max_files)?;
            dict.set_item("shard_details", shard_details)?;
            dict.set_item("from_cache", false)?;
            Ok(dict.into())
        })
    }

    fn get_shard_by_name(&mut self, shard_name: &str) -> PyResult<Py<PyDict>> {
        for (i, shard) in self.inner.shards.iter().enumerate() {
            if shard.name == shard_name {
                return self.get_shard_info(i);
            }
        }

        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Shard '{}' not found in dataset",
            shard_name
        )))
    }

    fn print_summary(&mut self, detailed: Option<bool>) -> PyResult<()> {
        let detailed = detailed.unwrap_or(false);

        println!("\nDataset Summary: {}", self.inner.name);
        println!("{}", "=".repeat(50));
        println!("Total shards: {}", self.inner.num_shards());

        // Try to use cached values first
        let (cached_size, cached_files) = self.inner.quick_stats();

        if let Some(files) = cached_files {
            println!("Total files: {} (estimated)", files);
        }
        if let Some(size) = cached_size {
            let size_gb = size as f64 / (1024.0_f64).powi(3);
            println!("Total size: {:.2} GB", size_gb);
        }
        if detailed {
            println!("\nShard Details:");
            println!("{}", "-".repeat(50));
            println!("{:<30} {:<12} {:<10}", "Shard Name", "Size (MB)", "Files");
            println!("{}", "-".repeat(50));

            let num_shards = self.inner.shards.len();
            let max_display = if detailed {
                num_shards
            } else {
                10.min(num_shards)
            };

            for i in 0..max_display {
                let shard_name = self.inner.shards[i].name.clone();
                let has_metadata = self.inner.shards[i].metadata.is_some();

                if has_metadata {
                    let metadata = self.inner.shards[i].metadata.as_ref().unwrap();
                    let size_mb = metadata.filesize as f64 / (1024.0_f64).powi(2);
                    println!(
                        "{:<30} {:<12.2} {:<10}",
                        shard_name,
                        size_mb,
                        metadata.num_files()
                    );
                } else if detailed {
                    if self.inner.ensure_shard_metadata(i).is_ok() {
                        if let Some(metadata) = &self.inner.shards[i].metadata {
                            let size_mb = metadata.filesize as f64 / (1024.0_f64).powi(2);
                            println!(
                                "{:<30} {:<12.2} {:<10}",
                                shard_name,
                                size_mb,
                                metadata.num_files()
                            );
                        } else {
                            println!("{:<30} {:<12} {:<10}", shard_name, "error", "error");
                        }
                    } else {
                        println!("{:<30} {:<12} {:<10}", shard_name, "error", "error");
                    }
                } else {
                    println!("{:<30} {:<12} {:<10}", shard_name, "?", "?");
                }
            }

            if num_shards > max_display {
                println!("... and {} more shards", num_shards - max_display);
            }
        }

        Ok(())
    }

    fn __repr__(&self) -> String {
        // Don't call total_files/total_size to avoid loading all metadata
        format!(
            "DiscoveredDataset(name='{}', shards={}, remote={})",
            self.inner.name,
            self.inner.num_shards(),
            self.inner.is_remote
        )
    }
}

#[pyclass(name = "ShardReader")]
pub struct PyShardReader {
    inner: ShardReader,
}

#[pymethods]
impl PyShardReader {
    #[getter]
    fn num_files(&self) -> usize {
        self.inner.num_files()
    }

    fn filenames(&self) -> Vec<String> {
        self.inner.filenames()
    }

    fn read_file(&self, file_index: usize) -> PyResult<Py<PyBytes>> {
        let data = self.inner.read_file(file_index)?;
        Python::with_gil(|py| Ok(PyBytes::new(py, &data).into()))
    }

    fn __repr__(&self) -> String {
        format!("ShardReader(num_files={})", self.inner.num_files())
    }
}
