use crate::error::{Result, WebshartError};
use crate::metadata::{FileInfo, ShardMetadata, ShardMetadataFormat};
use futures::future::join_all;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn is_image_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();
    path_lower.ends_with(".png")
        || path_lower.ends_with(".jpg")
        || path_lower.ends_with(".jpeg")
        || path_lower.ends_with(".webp")
        || path_lower.ends_with(".tiff")
        || path_lower.ends_with(".tif")
        || path_lower.ends_with(".bmp")
        || path_lower.ends_with(".gif")
        || path_lower.ends_with(".ico")
}

fn extract_image_dimensions(data: &[u8]) -> Option<(u32, u32, f32)> {
    match imagesize::blob_size(data) {
        Ok(size) => {
            let width = size.width as u32;
            let height = size.height as u32;
            let aspect = width as f32 / height as f32;
            Some((width, height, aspect))
        }
        Err(_) => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCheckpoint {
    pub shard_name: String,
    pub status: CheckpointStatus,
    pub offset: u64,
    pub files_processed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckpointStatus {
    Pending,
    InProgress,
    Complete,
    Failed(String),
}

#[derive(Debug, Clone)]
struct UnindexedShard {
    name: String,
    path: String,
    size: u64,
    is_remote: bool,
}

pub struct MetadataExtractor {
    runtime: Arc<Runtime>,
    hf_token: Option<String>,
    client: reqwest::Client,
    shard_pattern: Regex,
    compute_sha256: bool,
    include_image_geometry: bool,
}

impl MetadataExtractor {
    pub fn new(hf_token: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // Increase timeout for large files
            .connect_timeout(std::time::Duration::from_secs(30))
            .pool_idle_timeout(std::time::Duration::from_secs(30))
            .pool_max_idle_per_host(1) // Reduce connection pool to save memory
            .build()
            .expect("Failed to build HTTP client");

        // Create runtime without complex signal handling in thread start
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("webshart-worker")
            .build()
            .expect("Failed to create runtime");

        Self {
            runtime: Arc::new(runtime),
            hf_token,
            client,
            shard_pattern: Regex::new(r"^(.+?)\.tar$").unwrap(),
            compute_sha256: false,
            include_image_geometry: false,
        }
    }

    pub fn with_sha256(mut self, compute: bool) -> Self {
        self.compute_sha256 = compute;
        self
    }

    pub fn with_image_geometry(mut self, include: bool) -> Self {
        self.include_image_geometry = include;
        self
    }

    pub fn extract_metadata(
        &self,
        source: &str,
        destination: &str,
        checkpoint_dir: Option<&str>,
        max_workers: usize,
        shard_range: Option<(usize, usize)>, // NEW: Add range parameter
    ) -> Result<()> {
        self.runtime.block_on(async {
            // Set up Ctrl+C handler
            let ctrl_c = tokio::signal::ctrl_c();

            // Create the main extraction future
            let extraction = async {
                // Discover unindexed shards
                let mut shards = self
                    .discover_unindexed_shards(source, checkpoint_dir)
                    .await?;

                // Apply range filter if provided
                if let Some((start, end)) = shard_range {
                    println!("[webshart] Filtering shards to range [{}, {})", start, end);

                    // Sort shards by name first to ensure consistent ordering
                    shards.sort_by(|a, b| a.name.cmp(&b.name));

                    // Filter to the specified range
                    let total_shards = shards.len();
                    shards = shards
                        .into_iter()
                        .enumerate()
                        .filter_map(|(idx, shard)| {
                            if idx >= start && idx < end {
                                Some(shard)
                            } else {
                                None
                            }
                        })
                        .collect();

                    println!(
                        "[webshart] Processing {} shards out of {} total (indices {}-{})",
                        shards.len(),
                        total_shards,
                        start,
                        std::cmp::min(end, total_shards) - 1
                    );
                }

                if shards.is_empty() {
                    println!("[webshart] No unindexed shards found in specified range");
                    return Ok(());
                }

                println!(
                    "[webshart] Found {} unindexed shards to process",
                    shards.len()
                );

                // Load checkpoints
                let checkpoints = if let Some(dir) = checkpoint_dir {
                    self.load_checkpoints(dir)?
                } else {
                    HashMap::new()
                };

                // Create multi-progress for managing multiple progress bars
                let multi_progress = Arc::new(MultiProgress::new());

                // Process shards in parallel
                let semaphore = Arc::new(tokio::sync::Semaphore::new(max_workers));
                let futures = shards.into_iter().map(|shard| {
                    let sem = semaphore.clone();
                    let checkpoint = checkpoints.get(&shard.name).cloned();
                    let token = self.hf_token.clone();
                    let dest = destination.to_string();
                    let checkpoint_dir = checkpoint_dir.map(|s| s.to_string());
                    let extractor = self.clone();
                    let mp = multi_progress.clone();

                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = extractor
                            .process_shard(
                                shard.clone(),
                                checkpoint,
                                &dest,
                                checkpoint_dir.as_deref(),
                                token,
                                mp,
                            )
                            .await;
                        result
                    }
                });

                let results = join_all(futures).await;

                // Check for failures
                let mut failed = 0;
                let mut errors = Vec::new();
                for (i, result) in results.iter().enumerate() {
                    if let Err(e) = result {
                        eprintln!("[webshart] Failed to process shard {}: {}", i, e);
                        errors.push(e.to_string());
                        failed += 1;
                    }
                }

                if failed > 0 {
                    return Err(WebshartError::DiscoveryFailed(format!(
                        "Failed to process {} shards. First error: {}",
                        failed,
                        errors.first().unwrap_or(&"Unknown error".to_string())
                    )));
                }

                println!("[webshart] Successfully extracted metadata for all shards");
                Ok(())
            };

            // Run with interrupt handling
            tokio::select! {
                result = extraction => result,
                _ = ctrl_c => {
                    println!("\n[webshart] Received interrupt signal, stopping...");
                    Err(WebshartError::DiscoveryFailed("Cancelled by user".to_string()))
                }
            }
        })
    }

    pub fn extract_metadata_internal(
        &self,
        source: &str,
        destination: &str,
        checkpoint_dir: Option<&str>,
        max_workers: usize,
    ) -> Result<()> {
        self.runtime.block_on(async {
            // Discover unindexed shards
            let shards = self
                .discover_unindexed_shards(source, checkpoint_dir)
                .await?;

            if shards.is_empty() {
                println!("[webshart] No unindexed shards found");
                return Ok(());
            }

            println!(
                "[webshart] Found {} unindexed shards to process",
                shards.len()
            );

            // Load checkpoints
            let checkpoints = if let Some(dir) = checkpoint_dir {
                self.load_checkpoints(dir)?
            } else {
                HashMap::new()
            };

            // Create multi-progress for managing multiple progress bars
            let multi_progress = Arc::new(MultiProgress::new());

            // Process shards in parallel
            let semaphore = Arc::new(tokio::sync::Semaphore::new(max_workers));
            let futures = shards.into_iter().map(|shard| {
                let sem = semaphore.clone();
                let checkpoint = checkpoints.get(&shard.name).cloned();
                let token = self.hf_token.clone();
                let dest = destination.to_string();
                let checkpoint_dir = checkpoint_dir.map(|s| s.to_string());
                let extractor = self.clone();
                let mp = multi_progress.clone();

                async move {
                    let _permit = sem.acquire().await.unwrap();
                    let result = extractor
                        .process_shard(
                            shard.clone(),
                            checkpoint,
                            &dest,
                            checkpoint_dir.as_deref(),
                            token,
                            mp,
                        )
                        .await;
                    result
                }
            });

            let results = join_all(futures).await;

            // Check for failures and stop immediately
            let mut failed = 0;
            let mut errors = Vec::new();
            for (i, result) in results.iter().enumerate() {
                if let Err(e) = result {
                    eprintln!("[webshart] Failed to process shard {}: {}", i, e);
                    errors.push(e.to_string());
                    failed += 1;
                }
            }

            if failed > 0 {
                return Err(WebshartError::DiscoveryFailed(format!(
                    "Failed to process {} shards. First error: {}",
                    failed,
                    errors.first().unwrap_or(&"Unknown error".to_string())
                )));
            }

            println!("[webshart] Successfully extracted metadata for all shards");
            Ok(())
        })
    }

    async fn discover_unindexed_shards(
        &self,
        source: &str,
        checkpoint_dir: Option<&str>,
    ) -> Result<Vec<UnindexedShard>> {
        let mut shards = Vec::new();

        // Load existing checkpoints to filter out completed shards
        let checkpoints = if let Some(dir) = checkpoint_dir {
            self.load_checkpoints(dir)?
        } else {
            HashMap::new()
        };

        if Path::new(source).exists() {
            // Local discovery
            self.discover_local_unindexed(Path::new(source), &mut shards)?;
        } else {
            // HuggingFace discovery
            shards = self.discover_hf_unindexed(source).await?;
        }

        // Filter based on checkpoints and existing JSON files
        let mut filtered_shards = Vec::new();
        for shard in shards {
            // let base_name = shard.name.trim_end_matches(".tar");
            let json_exists = if shard.is_remote {
                false // Can't easily check remote JSON existence
            } else {
                Path::new(&shard.path).with_extension("json").exists()
            };

            if let Some(checkpoint) = checkpoints.get(&shard.name) {
                match &checkpoint.status {
                    CheckpointStatus::Complete => {
                        println!(
                            "[webshart] Skipping {} (marked complete in checkpoint)",
                            shard.name
                        );
                        continue;
                    }
                    CheckpointStatus::Failed(err) => {
                        println!(
                            "[webshart] Retrying {} (previously failed: {})",
                            shard.name, err
                        );
                    }
                    CheckpointStatus::InProgress => {
                        println!(
                            "[webshart] Resuming {} from offset {}",
                            shard.name, checkpoint.offset
                        );
                    }
                    CheckpointStatus::Pending => {
                        println!("[webshart] Processing {} (pending)", shard.name);
                    }
                }
            } else if json_exists {
                // Has JSON but no checkpoint - might be from a previous incomplete run
                println!(
                    "[webshart] Found {} with existing JSON but no checkpoint, will verify",
                    shard.name
                );
            } else {
                println!(
                    "[webshart] Processing {} (no JSON, no checkpoint)",
                    shard.name
                );
            }

            filtered_shards.push(shard);
        }

        // Sort by name for consistent ordering
        filtered_shards.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(filtered_shards)
    }

    fn discover_local_unindexed(
        &self,
        path: &Path,
        shards: &mut Vec<UnindexedShard>,
    ) -> Result<()> {
        // Find ALL tar files, not just ones without JSON
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();

                if entry_path.is_dir() {
                    // Recurse into subdirectories
                    self.discover_local_unindexed(&entry_path, shards)?;
                } else if let Some(file_name) = entry_path.file_name() {
                    let file_name_str = file_name.to_string_lossy();

                    if let Some(_captures) = self.shard_pattern.captures(&file_name_str) {
                        // Found a tar file - add it regardless of JSON existence
                        shards.push(UnindexedShard {
                            name: file_name_str.to_string(),
                            path: entry_path.to_string_lossy().to_string(),
                            size: entry.metadata()?.len(),
                            is_remote: false,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    async fn discover_hf_unindexed(&self, repo_id: &str) -> Result<Vec<UnindexedShard>> {
        // Use HF dataset info API
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

        #[derive(Deserialize)]
        struct Sibling {
            rfilename: String,
            size: Option<u64>,
        }

        #[derive(Deserialize)]
        struct DatasetInfo {
            siblings: Vec<Sibling>,
        }

        let info: DatasetInfo = response.json().await?;

        // Find ALL tar files
        let mut shards = Vec::new();

        for sibling in info.siblings {
            let path = Path::new(&sibling.rfilename);
            if let Some(file_name) = path.file_name() {
                let file_name_str = file_name.to_string_lossy();

                if let Some(_captures) = self.shard_pattern.captures(&file_name_str) {
                    let size = sibling.size.unwrap_or(0);
                    let size_str = if size == 0 {
                        "unknown".to_string()
                    } else {
                        format!("{} bytes", size)
                    };

                    println!(
                        "[webshart] Found tar file: {} (size: {})",
                        file_name_str, size_str
                    );

                    shards.push(UnindexedShard {
                        name: file_name_str.to_string(),
                        path: format!(
                            "https://huggingface.co/datasets/{}/resolve/main/{}",
                            repo_id, sibling.rfilename
                        ),
                        size,
                        is_remote: true,
                    });
                }
            }
        }

        Ok(shards)
    }

    async fn process_shard(
        &self,
        shard: UnindexedShard,
        checkpoint: Option<ShardCheckpoint>,
        destination: &str,
        checkpoint_dir: Option<&str>,
        hf_token: Option<String>,
        multi_progress: Arc<MultiProgress>,
    ) -> Result<()> {
        let start_offset = checkpoint.as_ref().map(|c| c.offset).unwrap_or(0);

        // Update checkpoint to in-progress
        if let Some(dir) = checkpoint_dir {
            let checkpoint = ShardCheckpoint {
                shard_name: shard.name.clone(),
                status: CheckpointStatus::InProgress,
                offset: start_offset,
                files_processed: 0,
            };
            self.save_checkpoint(dir, &checkpoint)?;
        }

        let metadata = match if shard.is_remote {
            self.extract_remote_metadata(&shard, start_offset, hf_token.clone(), multi_progress)
                .await
        } else {
            self.extract_local_metadata(&shard, start_offset, multi_progress)
        } {
            Ok(m) => m,
            Err(e) => {
                // Save failed checkpoint
                if let Some(dir) = checkpoint_dir {
                    let checkpoint = ShardCheckpoint {
                        shard_name: shard.name.clone(),
                        status: CheckpointStatus::Failed(e.to_string()),
                        offset: start_offset,
                        files_processed: 0,
                    };
                    self.save_checkpoint(dir, &checkpoint)?;
                }
                return Err(e);
            }
        };

        // Save metadata
        self.save_metadata(&shard, metadata, destination).await?;

        // Update checkpoint to complete
        if let Some(dir) = checkpoint_dir {
            let checkpoint = ShardCheckpoint {
                shard_name: shard.name.clone(),
                status: CheckpointStatus::Complete,
                offset: shard.size,
                files_processed: 0,
            };
            self.save_checkpoint(dir, &checkpoint)?;
        }

        Ok(())
    }

    async fn extract_remote_metadata(
        &self,
        shard: &UnindexedShard,
        start_offset: u64,
        hf_token: Option<String>,
        multi_progress: Arc<MultiProgress>,
    ) -> Result<ShardMetadata> {
        // Create progress bar for download
        let download_pb = multi_progress.add(ProgressBar::new(shard.size));
        download_pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "[{elapsed_precise}] {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );
        download_pb.set_message(format!("↓ {}", shard.name));

        // Create request
        let mut request = self
            .client
            .get(&shard.path)
            .header("Accept-Encoding", "identity")
            .timeout(std::time::Duration::from_secs(600));

        if let Some(token) = &hf_token {
            request = request.bearer_auth(token);
        }

        if start_offset > 0 {
            request = request.header("Range", format!("bytes={}-", start_offset));
        }

        let response = request.send().await.map_err(|e| {
            WebshartError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to connect: {}", e),
            ))
        })?;

        if !response.status().is_success() {
            download_pb.finish_and_clear();
            return Err(WebshartError::InvalidShardFormat(format!(
                "Failed to download tar file: {}",
                response.status()
            )));
        }

        // Get actual file size
        let actual_size = response
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(shard.size);

        download_pb.set_length(actual_size);

        // Stream to memory using channels
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<std::result::Result<Vec<u8>, std::io::Error>>(100);

        // Clone progress bar for the stream task
        let download_pb_clone = download_pb.clone();

        // Spawn task to stream chunks into channel
        let shard_name_clone = shard.name.clone();
        let stream_task = tokio::spawn(async move {
            use futures::StreamExt;

            let mut stream = response.bytes_stream();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_len = chunk.len() as u64;
                        download_pb_clone.inc(chunk_len);

                        // Send chunk through channel
                        if tx.send(std::result::Result::Ok(chunk.to_vec())).is_err() {
                            break; // Receiver dropped
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(std::result::Result::Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.to_string(),
                        )));
                        break;
                    }
                }
            }
            download_pb_clone.finish_with_message(format!("✓ Downloaded {}", shard_name_clone));
        });

        // Create progress bar for processing
        let process_pb = multi_progress.add(ProgressBar::new_spinner());
        process_pb.set_style(
            ProgressStyle::default_spinner()
                .template("[{elapsed_precise}] {msg} {spinner} [{pos} files]")
                .unwrap(),
        );
        process_pb.set_message(format!("⚙ Processing {}", shard.name));

        // Process tar in blocking task using channel reader
        let shard_name = shard.name.clone();
        let compute_sha256 = self.compute_sha256;
        let include_image_geometry = self.include_image_geometry;

        let result = tokio::task::spawn_blocking(move || {
            use tar::Archive;
            // Create a custom reader that reads from the channel
            struct ChannelReader {
                rx: std::sync::mpsc::Receiver<std::result::Result<Vec<u8>, std::io::Error>>,
                buffer: Vec<u8>,
                pos: usize,
            }
            impl Read for ChannelReader {
                fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                    // If buffer is empty, get next chunk
                    if self.pos >= self.buffer.len() {
                        match self.rx.recv() {
                            Ok(Ok(chunk)) => {
                                self.buffer = chunk;
                                self.pos = 0;
                            }
                            Ok(Err(e)) => return Err(e),
                            Err(_) => return Ok(0), // Channel closed, EOF
                        }
                    }
                    // Copy from buffer to output
                    let available = self.buffer.len() - self.pos;
                    if available == 0 {
                        return Ok(0);
                    }
                    let to_copy = std::cmp::min(buf.len(), available);
                    buf[..to_copy].copy_from_slice(&self.buffer[self.pos..self.pos + to_copy]);
                    self.pos += to_copy;
                    Ok(to_copy)
                }
            }

            let reader = ChannelReader {
                rx,
                buffer: Vec::new(),
                pos: 0,
            };

            let mut files = HashMap::new();
            let mut file_count = 0;
            let mut archive = Archive::new(reader);
            let entries = archive.entries()?;

            for entry in entries {
                match entry {
                    Ok(mut entry) => {
                        let path = entry.path()?.to_string_lossy().to_string();
                        let offset = entry.raw_header_position();
                        let size = entry.size();

                        if entry.header().entry_type() == tar::EntryType::Regular {
                            let mut file_data = Vec::new();
                            let mut hasher = if compute_sha256 && size > 0 && size < 10_000_000 {
                                Some(Sha256::new())
                            } else {
                                None
                            };
                            // Determine if we need to read the file data
                            let need_hash = hasher.is_some();
                            let need_dimensions = include_image_geometry && is_image_file(&path) && size > 0 && size < 50_000_000;
                            let should_read = need_hash || need_dimensions;
                            if should_read {
                                if need_dimensions {
                                    file_data.reserve(size as usize);
                                }
                                let mut buffer = [0; 8192];
                                let mut total_read = 0u64;
                                while total_read < size {
                                    let to_read = std::cmp::min(buffer.len(), (size - total_read) as usize);
                                    match entry.read_exact(&mut buffer[..to_read]) {
                                        Ok(_) => {
                                            if let Some(ref mut h) = hasher {
                                                h.update(&buffer[..to_read]);
                                            }
                                            if need_dimensions {
                                                file_data.extend_from_slice(&buffer[..to_read]);
                                            }
                                            total_read += to_read as u64;
                                        }
                                        Err(e) => {
                                            eprintln!("[webshart] Error reading file {} for processing: {}", path, e);
                                            break;
                                        }
                                    }
                                }
                            } else {
                                // Skip the file if we don't need to process it
                                std::io::copy(&mut entry, &mut std::io::sink())?;
                            }
                            let file_hash = if compute_sha256 {
                                if size == 0 {
                                    Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string())
                                } else if let Some(h) = hasher {
                                    Some(format!("{:x}", h.finalize()))
                                } else {
                                    None
                                }
                            } else {
                                None
                            };
                            // Extract image dimensions if needed
                            let (width, height, aspect) = if need_dimensions && !file_data.is_empty() {
                                extract_image_dimensions(&file_data).unwrap_or((0, 0, 0.0))
                            } else {
                                (0, 0, 0.0)
                            };
                            files.insert(path.clone(), FileInfo {
                                path: Some(path.clone()),
                                offset: offset + 512,
                                length: size,
                                sha256: file_hash,
                                width: if width > 0 { Some(width) } else { None },
                                height: if height > 0 { Some(height) } else { None },
                                aspect: if aspect > 0.0 { Some(aspect) } else { None },
                            });
                            file_count += 1;
                            process_pb.inc(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("[webshart] Error reading tar entry: {}", e);
                    }
                }
            }

            process_pb.finish_with_message(format!("✓ Processed {} ({} files)", shard_name, file_count));
            Ok::<HashMap<String, FileInfo>, WebshartError>(files)
        }).await.map_err(|e| WebshartError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Task join error: {}", e)
        )))??;

        // Wait for stream task to complete
        let _ = stream_task.await;

        Ok(ShardMetadata::from_format(ShardMetadataFormat::HashMap {
            path: Some(shard.name.clone()),
            filesize: actual_size,
            hash: None,
            hash_lfs: None,
            files: result,
            includes_image_geometry: self.include_image_geometry,
        }))
    }

    fn extract_local_metadata(
        &self,
        shard: &UnindexedShard,
        start_offset: u64,
        multi_progress: Arc<MultiProgress>,
    ) -> Result<ShardMetadata> {
        use std::fs::File;
        use tar::Archive;

        let mut files = HashMap::new();
        let mut file_count = 0;

        // Create progress bar
        let pb = multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("[{elapsed_precise}] {msg} {spinner} [{pos} files]")
                .unwrap(),
        );
        pb.set_message(format!("⚙ Processing {}", shard.name));

        let file = File::open(&shard.path)?;
        let mut archive = Archive::new(file);

        // Process entries
        let entries = archive.entries()?;
        for entry in entries {
            match entry {
                Ok(mut entry) => {
                    let path = entry.path()?.to_string_lossy().to_string();
                    let offset = entry.raw_header_position();
                    let size = entry.size();

                    // Skip if before start_offset (for resuming)
                    if offset < start_offset {
                        continue;
                    }

                    // Only process regular files
                    if entry.header().entry_type() == tar::EntryType::Regular {
                        let mut file_data = Vec::new();
                        let mut hasher = if self.compute_sha256 && size > 0 && size < 10_000_000 {
                            Some(Sha256::new())
                        } else {
                            None
                        };

                        // Determine if we need to read the file
                        let need_hash = hasher.is_some();
                        let need_dimensions = self.include_image_geometry
                            && is_image_file(&path)
                            && size > 0
                            && size < 50_000_000;
                        let should_read = need_hash || need_dimensions;

                        if should_read {
                            if need_dimensions {
                                file_data.reserve(size as usize);
                            }

                            let mut buffer = [0; 8192];
                            loop {
                                match std::io::Read::read(&mut entry, &mut buffer) {
                                    Ok(0) => break,
                                    Ok(n) => {
                                        if let Some(ref mut h) = hasher {
                                            h.update(&buffer[..n]);
                                        }
                                        if need_dimensions {
                                            file_data.extend_from_slice(&buffer[..n]);
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[webshart] Error reading file {}: {}", path, e);
                                        break;
                                    }
                                }
                            }
                        } else {
                            // Skip the file if we don't need to process it
                            std::io::copy(&mut entry, &mut std::io::sink())?;
                        }

                        let file_hash = if self.compute_sha256 {
                            if size == 0 {
                                Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string())
                            } else if let Some(h) = hasher {
                                Some(format!("{:x}", h.finalize()))
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        // Extract dimensions if needed
                        let (width, height, aspect) = if need_dimensions && !file_data.is_empty() {
                            extract_image_dimensions(&file_data).unwrap_or((0, 0, 0.0))
                        } else {
                            (0, 0, 0.0)
                        };

                        files.insert(
                            path.clone(),
                            FileInfo {
                                path: Some(path.clone()),
                                offset: offset + 512, // Add header size to get file content offset
                                length: size,
                                sha256: file_hash,
                                width: if width > 0 { Some(width) } else { None },
                                height: if height > 0 { Some(height) } else { None },
                                aspect: if aspect > 0.0 { Some(aspect) } else { None },
                            },
                        );

                        file_count += 1;
                        pb.inc(1);
                    }
                }
                Err(e) => {
                    eprintln!("[webshart] Error reading tar entry: {}", e);
                    // Continue processing other entries
                }
            }
        }

        pb.finish_with_message(format!("✓ Processed {} ({} files)", shard.name, file_count));

        // Calculate file hash if requested
        let tar_hash = if self.compute_sha256 && shard.size < 100_000_000 {
            let hash_pb = multi_progress.add(ProgressBar::new(shard.size));
            hash_pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "[{elapsed_precise}] {msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes}",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );
            hash_pb.set_message(format!("# Computing hash for {}", shard.name));

            let mut file = File::open(&shard.path)?;
            let mut hasher = Sha256::new();
            let mut buffer = [0; 8192];
            let mut total_read = 0u64;
            loop {
                match std::io::Read::read(&mut file, &mut buffer) {
                    Ok(0) => break,
                    Ok(n) => {
                        hasher.update(&buffer[..n]);
                        total_read += n as u64;
                        hash_pb.set_position(total_read);
                    }
                    Err(e) => {
                        hash_pb.finish_and_clear();
                        return Err(WebshartError::Io(e));
                    }
                }
            }
            hash_pb.finish_with_message(format!("✓ Hash computed for {}", shard.name));
            Some(format!("{:x}", hasher.finalize()))
        } else {
            None
        };

        Ok(ShardMetadata::from_format(ShardMetadataFormat::HashMap {
            path: Some(shard.name.clone()),
            filesize: shard.size,
            hash: tar_hash.clone(),
            hash_lfs: tar_hash,
            files,
            includes_image_geometry: self.include_image_geometry,
        }))
    }

    fn load_checkpoints(&self, dir: &str) -> Result<HashMap<String, ShardCheckpoint>> {
        let mut checkpoints = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if let Ok(checkpoint) =
                                serde_json::from_str::<ShardCheckpoint>(&content)
                            {
                                checkpoints.insert(checkpoint.shard_name.clone(), checkpoint);
                            }
                        }
                    }
                }
            }
        }
        Ok(checkpoints)
    }

    fn save_checkpoint(&self, dir: &str, checkpoint: &ShardCheckpoint) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = Path::new(dir).join(format!("{}.json", checkpoint.shard_name));
        let content = serde_json::to_string_pretty(checkpoint)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    async fn save_metadata(
        &self,
        shard: &UnindexedShard,
        metadata: ShardMetadata,
        destination: &str,
    ) -> Result<()> {
        let json_name = shard.name.replace(".tar", ".json");

        if destination.contains('/') && !destination.starts_with("http") {
            // Local destination
            let path = Path::new(destination).join(&json_name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let content = serde_json::to_string(&metadata)?;
            std::fs::write(path, content)?;
        } else {
            // HF Hub destination - would need upload logic
            return Err(WebshartError::DiscoveryFailed(
                "HF Hub upload not implemented yet".to_string(),
            ));
        }

        Ok(())
    }
}

impl Clone for MetadataExtractor {
    fn clone(&self) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            runtime: self.runtime.clone(),
            hf_token: self.hf_token.clone(),
            client,
            shard_pattern: Regex::new(r"^(.+?)\.tar$").unwrap(),
            compute_sha256: self.compute_sha256,
            include_image_geometry: self.include_image_geometry,
        }
    }
}

// Python bindings
#[pyclass(name = "MetadataExtractor")]
pub struct PyMetadataExtractor {
    inner: MetadataExtractor,
}

#[pymethods]
impl PyMetadataExtractor {
    #[new]
    #[pyo3(signature = (hf_token=None))]
    fn new(hf_token: Option<String>) -> Self {
        Self {
            inner: MetadataExtractor::new(hf_token),
        }
    }

    #[pyo3(signature = (source, destination, checkpoint_dir=None, max_workers=2, shard_range=None, include_image_geometry=false))]
    fn extract_metadata(
        &self,
        source: &str,
        destination: &str,
        checkpoint_dir: Option<&str>,
        max_workers: usize,
        shard_range: Option<(usize, usize)>,
        include_image_geometry: bool,
    ) -> PyResult<()> {
        // Create a new extractor with the image geometry setting
        let extractor = self
            .inner
            .clone()
            .with_image_geometry(include_image_geometry);

        extractor
            .extract_metadata(
                source,
                destination,
                checkpoint_dir,
                max_workers,
                shard_range,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}
