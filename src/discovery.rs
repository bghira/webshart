use crate::dataloader::shard_cache::ShardCache;
use crate::error::{Result, WebshartError};
use crate::metadata::ShardMetadata;
use crate::metadata_resolver::MetadataResolver;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
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
    pub name: String,
    pub subfolder: Option<String>,
    pub is_remote: bool,
    pub shards: Vec<ShardPair>,
    discovery_token: Option<String>,
    cached_total_size: Option<u64>,
    cached_total_files: Option<usize>,
    pub metadata_source: Option<String>,
    runtime: Arc<Runtime>,
    /// Optional cache directory for metadata
    cache_dir: Option<PathBuf>,
    /// Track if we've hit rate limits
    rate_limit_delay: Option<Duration>,
    pub shard_cache: Option<Arc<ShardCache>>,
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

    pub async fn enable_shard_cache(
        &mut self,
        location: PathBuf,
        cache_limit_gb: f64,
        parallel_downloads: usize,
    ) -> Result<()> {
        let mut cache = ShardCache::new(location, cache_limit_gb, parallel_downloads);
        cache.ensure_cache_dir().await?;
        cache.initialize_from_disk().await?;
        self.shard_cache = Some(Arc::new(cache));
        Ok(())
    }

    /// Enable metadata caching and optionally pre-load some shards
    pub fn enable_metadata_cache(
        &mut self,
        cache_location: &str,
        init_shard_count: usize,
    ) -> Result<()> {
        let cache_path = PathBuf::from(cache_location);

        // Create cache directory if it doesn't exist
        if !cache_path.exists() {
            fs::create_dir_all(&cache_path)?;
        }

        // Create a subdirectory for this dataset
        let dataset_cache = cache_path.join(self.get_cache_key());
        if !dataset_cache.exists() {
            fs::create_dir_all(&dataset_cache)?;
        }

        self.cache_dir = Some(dataset_cache);

        // Pre-load initial shards
        if init_shard_count > 0 {
            println!(
                "[webshart] Pre-loading {} shard metadata...",
                init_shard_count.min(self.shards.len())
            );
            let count = init_shard_count.min(self.shards.len());

            for i in 0..count {
                match self.ensure_shard_metadata(i) {
                    Ok(_) => {
                        println!("[webshart] Loaded metadata for shard {}/{}", i + 1, count);
                    }
                    Err(e) => {
                        eprintln!(
                            "[webshart] Warning: Failed to load metadata for shard {}: {}",
                            i, e
                        );
                        // Check if it's a rate limit error
                        if let WebshartError::RateLimited = e {
                            // Stop pre-loading if we hit rate limit
                            println!("[webshart] Rate limited, stopping pre-load");
                            break;
                        }
                    }
                }

                // Add small delay between requests to be nice to the server
                if i < count - 1 && self.is_remote {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }

        Ok(())
    }

    /// Generate a cache key for this dataset
    fn get_cache_key(&self) -> String {
        let mut key = self.name.replace('/', "_");
        if let Some(subfolder) = &self.subfolder {
            key.push_str("__");
            key.push_str(&subfolder.replace('/', "_"));
        }
        key
    }

    /// Get cached metadata path for a shard
    fn get_cached_metadata_path(&self, shard_name: &str) -> Option<PathBuf> {
        self.cache_dir
            .as_ref()
            .map(|dir| dir.join(format!("{}.json", shard_name)))
    }

    /// Load metadata from cache if available
    fn load_cached_metadata(&self, shard_name: &str) -> Option<ShardMetadata> {
        let cache_path = self.get_cached_metadata_path(shard_name)?;

        if cache_path.exists() {
            match fs::read_to_string(&cache_path) {
                Ok(content) => {
                    match serde_json::from_str(&content) {
                        Ok(metadata) => {
                            println!("[webshart] Loaded metadata for {} from cache", shard_name);
                            Some(metadata)
                        }
                        Err(e) => {
                            eprintln!(
                                "[webshart] Failed to parse cached metadata for {}: {}",
                                shard_name, e
                            );
                            // Remove corrupted cache file
                            let _ = fs::remove_file(&cache_path);
                            None
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[webshart] Failed to read cached metadata for {}: {}",
                        shard_name, e
                    );
                    None
                }
            }
        } else {
            None
        }
    }

    /// Save metadata to cache
    fn save_metadata_to_cache(&self, shard_name: &str, metadata: &ShardMetadata) -> Result<()> {
        if let Some(cache_path) = self.get_cached_metadata_path(shard_name) {
            let json = serde_json::to_string_pretty(metadata)?;
            fs::write(&cache_path, json)?;
            println!("[webshart] Cached metadata for {}", shard_name);
        }
        Ok(())
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
        // Scope the first borrow to check if metadata is already loaded
        if let Some(shard) = self.shards.get(shard_index) {
            if shard.metadata.is_some() {
                return Ok(());
            }
        } else {
            return Err(WebshartError::InvalidShardFormat(format!(
                "File index {} out of range for shard {}",
                shard_index, self.name
            )));
        }

        // Clone shard info to release the borrow on `self`
        let (shard_name, json_path) = {
            let shard = &self.shards[shard_index];
            (shard.name.clone(), shard.json_path.clone())
        };

        // First, check cache
        if let Some(cached) = self.load_cached_metadata(&shard_name) {
            if let Some(shard) = self.shards.get_mut(shard_index) {
                shard.metadata = Some(cached);
            }
            return Ok(());
        }

        // Apply rate limit delay if needed
        if let Some(delay) = self.rate_limit_delay {
            println!("[webshart] Rate limit delay: {:?}", delay);
            std::thread::sleep(delay);
            // Reset delay after using it
            self.rate_limit_delay = None;
        }

        let discovery = DatasetDiscovery::with_runtime(self.runtime.clone())
            .with_optional_token(self.discovery_token.clone());

        // Use block_in_place to avoid blocking the async runtime
        let result = tokio::task::block_in_place(|| {
            if self.is_remote {
                self.runtime
                    .block_on(discovery.load_remote_metadata(&json_path))
            } else {
                self.runtime
                    .block_on(discovery.load_local_metadata(&json_path))
            }
        });

        match result {
            Ok(metadata) => {
                // Save to cache if caching is enabled
                let _ = self.save_metadata_to_cache(&shard_name, &metadata);
                // Re-borrow to update the shard
                if let Some(shard) = self.shards.get_mut(shard_index) {
                    shard.metadata = Some(metadata);
                }
                Ok(())
            }
            Err(e) => {
                // Check if it's a rate limit error
                if let WebshartError::RateLimited = e {
                    // Set exponential backoff delay
                    let current_delay = self.rate_limit_delay.unwrap_or(Duration::from_secs(1));
                    self.rate_limit_delay = Some(current_delay * 2);
                    println!(
                        "[webshart] Rate limited, next delay will be {:?}",
                        self.rate_limit_delay
                    );
                }
                Err(e)
            }
        }
    }

    /// Clear the metadata cache for this dataset
    pub fn clear_cache(&self) -> Result<()> {
        if let Some(cache_dir) = &self.cache_dir {
            if cache_dir.exists() {
                fs::remove_dir_all(cache_dir)?;
                println!("[webshart] Cleared metadata cache for {}", self.name);
            }
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<(usize, u64)> {
        if let Some(cache_dir) = &self.cache_dir {
            let mut count = 0;
            let mut total_size = 0u64;

            if cache_dir.exists() {
                for entry in fs::read_dir(cache_dir)? {
                    let entry = entry?;
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                        count += 1;
                        total_size += entry.metadata()?.len();
                    }
                }
            }

            Ok((count, total_size))
        } else {
            Ok((0, 0))
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
    metadata_resolver: MetadataResolver,
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
            metadata_resolver: MetadataResolver::new(
                None,
                None,
                Arc::new(Runtime::new().expect("Failed to create Tokio runtime")),
            ),
        }
    }

    /// Create with existing runtime
    pub fn with_runtime(runtime: Arc<Runtime>) -> Self {
        Self {
            hf_token: None,
            shard_pattern: Regex::new(r"^(.+?)\.tar$").unwrap(),
            client: reqwest::Client::new(),
            runtime: runtime.clone(),
            metadata_resolver: MetadataResolver::new(None, None, runtime.clone()),
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

    pub fn with_metadata_source(mut self, metadata_source: Option<String>) -> Self {
        if let Some(source) = metadata_source {
            self.metadata_resolver =
                MetadataResolver::new(Some(source), self.hf_token.clone(), self.runtime.clone());
        }
        self
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
                let tar_path = entry.path().to_string_lossy().to_string();

                // Use resolver to find metadata path
                let json_path = self
                    .metadata_resolver
                    .resolve_metadata_path(&tar_path, base_name, false);

                // Check if metadata exists
                if self.metadata_resolver.metadata_exists(&json_path, false) {
                    shards.push(ShardPair {
                        name: base_name.to_string(),
                        tar_path,
                        json_path,
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
            subfolder: None,
            cache_dir: None,
            rate_limit_delay: None,
            is_remote: false,
            shards,
            shard_cache: None,
            discovery_token: self.hf_token.clone(),
            cached_total_size: None,
            cached_total_files: None,
            metadata_source: self.metadata_resolver.get_source(),
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

        Ok(DiscoveredDataset {
            name: repo_id.to_string(),
            subfolder: subfolder.map(|s| s.to_string()),
            cache_dir: None,
            rate_limit_delay: None,
            is_remote: true,
            shards: all_shards,
            shard_cache: None,
            discovery_token: self.hf_token.clone(),
            cached_total_size: cached_size,
            cached_total_files: final_cached_files,
            metadata_source: self.metadata_resolver.get_source(),
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

        let mut shards = Vec::new();
        for (base_name, tar_path) in tar_files {
            let base_url = format!("https://huggingface.co/datasets/{}/resolve/main", repo_id);
            let full_tar_path = format!("{}/{}", base_url, tar_path);

            // Use resolver to get metadata path
            let json_path =
                self.metadata_resolver
                    .resolve_metadata_path(&full_tar_path, &base_name, true);

            // Try to check if metadata exists
            if self.metadata_resolver.metadata_exists(&json_path, true) {
                shards.push(ShardPair {
                    name: base_name,
                    tar_path: full_tar_path,
                    json_path,
                    metadata: None,
                });
            } else {
                // Fallback to co-located metadata if custom location doesn't have it
                let default_json_path =
                    format!("{}/{}", base_url, tar_path.replace(".tar", ".json"));
                if json_files.contains_key(&base_name) {
                    shards.push(ShardPair {
                        name: base_name,
                        tar_path: full_tar_path,
                        json_path: default_json_path,
                        metadata: None,
                    });
                }
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
        let siblings = json["siblings"].as_array().ok_or_else(|| {
            WebshartError::DiscoveryFailed("No siblings array in dataset info".to_string())
        })?;

        // Process siblings to find tar files
        let mut tar_files = HashMap::new();

        for sibling in siblings {
            let path = sibling["rfilename"].as_str().ok_or_else(|| {
                WebshartError::DiscoveryFailed("Invalid sibling entry".to_string())
            })?;

            // Filter by subfolder if specified
            let in_target_folder = if let Some(folder) = subfolder {
                path.starts_with(&format!("{}/", folder))
            } else {
                path.ends_with(".tar") // Only look for tar files if no subfolder
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
            }
        }

        // Build shards with metadata resolver
        let mut shards = Vec::new();
        let base_url = format!("https://huggingface.co/datasets/{}/resolve/main", repo_id);

        for (base_name, tar_path) in tar_files {
            let full_tar_path = format!("{}/{}", base_url, tar_path);

            // Use resolver to get metadata path
            let json_path =
                self.metadata_resolver
                    .resolve_metadata_path(&full_tar_path, &base_name, true);

            shards.push(ShardPair {
                name: base_name,
                tar_path: full_tar_path,
                json_path,
                metadata: None,
            });
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

    pub async fn load_local_metadata(&self, path: &str) -> Result<ShardMetadata> {
        self.metadata_resolver.load_metadata(path, false).await
    }

    pub async fn load_remote_metadata(&self, url: &str) -> Result<ShardMetadata> {
        match self.metadata_resolver.load_metadata(url, true).await {
            Ok(metadata) => Ok(metadata),
            Err(e) => {
                // Check if it's a 429 error and convert to RateLimited
                if let WebshartError::Http(ref err) = e {
                    if let Some(status) = err.status() {
                        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                            return Err(WebshartError::RateLimited);
                        }
                    }
                }
                Err(e)
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
    #[pyo3(signature = (hf_token=None, metadata_source=None))]
    fn new(hf_token: Option<String>, metadata_source: Option<String>) -> Self {
        let mut discovery = DatasetDiscovery::new();
        if let Some(token) = hf_token {
            discovery = discovery.with_hf_token(token);
        }
        if let Some(metadata) = metadata_source {
            discovery = discovery.with_metadata_source(Some(metadata));
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

    #[pyo3(signature = (location, cache_limit_gb=25.0, parallel_downloads=4))]
    fn enable_shard_cache(
        &mut self,
        location: &str,
        cache_limit_gb: f64,
        parallel_downloads: usize,
    ) -> PyResult<()> {
        let runtime = Runtime::new()?;
        runtime.block_on(async {
            self.inner
                .enable_shard_cache(PathBuf::from(location), cache_limit_gb, parallel_downloads)
                .await
        })?;
        Ok(())
    }

    #[pyo3(signature = (location, init_shard_count=4))]
    fn enable_metadata_cache(
        &mut self,
        location: &str,
        init_shard_count: Option<usize>,
    ) -> PyResult<()> {
        let init_count = init_shard_count.unwrap_or(4);
        self.inner.enable_metadata_cache(location, init_count)?;
        Ok(())
    }

    fn clear_metadata_cache(&self) -> PyResult<()> {
        self.inner.clear_cache()?;
        Ok(())
    }

    fn get_cache_stats(&self) -> PyResult<Py<PyDict>> {
        let (count, size) = self.inner.cache_stats()?;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("cached_shards", count)?;
            dict.set_item("cache_size_bytes", size)?;
            dict.set_item("cache_size_mb", size as f64 / (1024.0 * 1024.0))?;
            dict.set_item("cache_enabled", self.inner.cache_dir.is_some())?;
            if let Some(ref cache_dir) = self.inner.cache_dir {
                dict.set_item("cache_location", cache_dir.to_string_lossy().to_string())?;
            }
            Ok(dict.into())
        })
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

    fn get_hf_token(&self) -> Option<String> {
        self.inner.discovery_token.clone()
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
