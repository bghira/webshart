// dataloader/shard_cache.rs
use crate::dataloader::file_loading::FileLoader;
use crate::error::{Result, WebshartError};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

#[derive(Debug, Clone)]
pub struct ShardCache {
    cache_dir: PathBuf,
    cache_limit_bytes: u64,
    current_size_bytes: Arc<Mutex<u64>>,
    lru_queue: Arc<Mutex<VecDeque<String>>>,
    shard_sizes: Arc<Mutex<HashMap<String, u64>>>,
    download_semaphore: Arc<Semaphore>,
    active_downloads: Arc<Mutex<HashMap<String, PathBuf>>>,
}
impl ShardCache {
    pub fn new(cache_dir: PathBuf, cache_limit_gb: f64, parallel_downloads: usize) -> Self {
        Self {
            cache_dir,
            cache_limit_bytes: (cache_limit_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            current_size_bytes: Arc::new(Mutex::new(0)),
            lru_queue: Arc::new(Mutex::new(VecDeque::new())),
            shard_sizes: Arc::new(Mutex::new(HashMap::new())),
            download_semaphore: Arc::new(Semaphore::new(parallel_downloads)),
            active_downloads: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn ensure_cache_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.cache_dir)
            .await
            .map_err(|e| WebshartError::Io(e))
    }

    pub fn get_cached_shard_path(&self, shard_name: &str) -> PathBuf {
        self.cache_dir.join(shard_name)
    }

    pub async fn is_cached(&self, shard_name: &str) -> bool {
        self.get_cached_shard_path(shard_name).exists()
    }

    pub async fn cache_shard(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
    ) -> Result<PathBuf> {
        let cached_path = self.get_cached_shard_path(shard_name);

        // Check if already cached
        if cached_path.exists() {
            self.touch_shard(shard_name).await;
            return Ok(cached_path);
        }

        // Acquire download permit
        let _permit = self.download_semaphore.acquire().await.map_err(|e| {
            WebshartError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to acquire download permit: {}", e),
            ))
        })?;

        // Download the shard directly to disk
        let shard_size = self
            .download_shard_to_disk(remote_url, token, shard_name)
            .await?;

        // Update cache metadata
        {
            let mut sizes = self.shard_sizes.lock().unwrap();
            let mut queue = self.lru_queue.lock().unwrap();
            let mut current_size = self.current_size_bytes.lock().unwrap();

            sizes.insert(shard_name.to_string(), shard_size);
            queue.push_back(shard_name.to_string());
            *current_size += shard_size;
        }

        Ok(cached_path)
    }

    async fn download_shard_to_disk(
        &self,
        url: &str,
        token: Option<String>,
        shard_name: &str,
    ) -> Result<u64> {
        use futures::StreamExt;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(WebshartError::from)?;

        let mut request = client.get(url);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.map_err(WebshartError::from)?;

        // Get total size if available
        let total_size = response.content_length();

        // Create temporary file path
        let temp_path = self.cache_dir.join(format!("{}.download", shard_name));
        let final_path = self.get_cached_shard_path(shard_name);

        // Make room if we know the size
        if let Some(size) = total_size {
            self.evict_if_needed(size).await?;
        }

        // Start tracking this download
        {
            let mut active = self.active_downloads.lock().unwrap();
            active.insert(shard_name.to_string(), temp_path.clone());
        }

        // Create the file
        let mut file = File::create(&temp_path)
            .await
            .map_err(|e| WebshartError::Io(e))?;

        // Stream the response directly to disk
        let mut stream = response.bytes_stream();
        let mut bytes_written = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(WebshartError::from)?;
            bytes_written += chunk.len() as u64;

            file.write_all(&chunk)
                .await
                .map_err(|e| WebshartError::Io(e))?;

            if bytes_written % (1024 * 1024) == 0 {
                // Use sync_data() instead of flush() to update file metadata
                file.sync_data().await.map_err(|e| WebshartError::Io(e))?;
            }
        }

        // Final flush and sync
        file.flush().await.map_err(|e| WebshartError::Io(e))?;
        file.sync_all().await.map_err(|e| WebshartError::Io(e))?;
        drop(file);

        // If we didn't know the size before, make room now
        if total_size.is_none() {
            self.evict_if_needed(bytes_written).await?;
        }

        // Move temp file to final location
        fs::rename(&temp_path, &final_path)
            .await
            .map_err(|e| WebshartError::Io(e))?;

        // Remove from active downloads
        {
            let mut active = self.active_downloads.lock().unwrap();
            active.remove(shard_name);
        }

        Ok(bytes_written)
    }

    async fn evict_if_needed(&self, needed_bytes: u64) -> Result<()> {
        let mut sizes = self.shard_sizes.lock().unwrap();
        let mut queue = self.lru_queue.lock().unwrap();
        let mut current_size = self.current_size_bytes.lock().unwrap();

        while *current_size + needed_bytes > self.cache_limit_bytes && !queue.is_empty() {
            if let Some(shard_to_evict) = queue.pop_front() {
                if let Some(size) = sizes.remove(&shard_to_evict) {
                    *current_size -= size;

                    // Delete the file
                    let path = self.get_cached_shard_path(&shard_to_evict);
                    drop(sizes);
                    drop(queue);
                    drop(current_size);

                    let _ = fs::remove_file(path).await;

                    sizes = self.shard_sizes.lock().unwrap();
                    queue = self.lru_queue.lock().unwrap();
                    current_size = self.current_size_bytes.lock().unwrap();
                }
            }
        }

        Ok(())
    }

    pub async fn get_cached_file_size(&self, shard_name: &str) -> Result<u64> {
        // First, check if it's actively downloading.
        if let Some(temp_path) = self.get_active_download_path(shard_name) {
            if let Ok(metadata) = fs::metadata(&temp_path).await {
                let size = metadata.len();
                return Ok(size);
            }
        }

        // Second, check the in-memory cache of shard sizes for fully downloaded files.
        {
            let sizes = self.shard_sizes.lock().unwrap();
            if let Some(&size) = sizes.get(shard_name) {
                return Ok(size);
            }
        }

        // Finally, check the disk directly for a fully cached file that might not be in memory map.
        let path = self.get_cached_shard_path(shard_name);
        if path.exists() {
            let metadata = fs::metadata(&path)
                .await
                .map_err(|e| WebshartError::Io(e))?;
            let size = metadata.len();
            return Ok(size);
        }

        Err(WebshartError::CacheMiss(shard_name.to_string()))
    }

    pub fn get_active_download_path(&self, shard_name: &str) -> Option<PathBuf> {
        let active = self.active_downloads.lock().unwrap();
        active.get(shard_name).cloned()
    }

    pub async fn get_download_progress(&self, shard_name: &str) -> Option<u64> {
        if let Some(path) = self.get_active_download_path(shard_name) {
            println!("Checking download progress for {}: {:?}", shard_name, path);
            match fs::metadata(&path).await {
                Ok(metadata) => {
                    let size = metadata.len();
                    println!("File size: {}", size);
                    return Some(size);
                }
                Err(e) => {
                    println!("Error getting metadata: {}", e);
                }
            }
        } else {
            println!("No active download found for {}", shard_name);
        }
        None
    }

    async fn touch_shard(&self, shard_name: &str) {
        let mut queue = self.lru_queue.lock().unwrap();

        // Remove from current position
        if let Some(pos) = queue.iter().position(|x| x == shard_name) {
            queue.remove(pos);
        }

        // Add to end (most recently used)
        queue.push_back(shard_name.to_string());
    }

    pub async fn initialize_from_disk(&mut self) -> Result<()> {
        // Clean up any incomplete downloads
        let mut entries = fs::read_dir(&self.cache_dir)
            .await
            .map_err(|e| WebshartError::Io(e))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| WebshartError::Io(e))?
        {
            if let Some(filename) = entry.file_name().to_str() {
                if filename.ends_with(".download") {
                    // Remove incomplete downloads
                    let _ = fs::remove_file(entry.path()).await;
                }
            }
        }

        // Scan cache directory and rebuild metadata
        let mut entries = fs::read_dir(&self.cache_dir)
            .await
            .map_err(|e| WebshartError::Io(e))?;

        let mut total_size = 0u64;
        let mut cached_shards = Vec::new();

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| WebshartError::Io(e))?
        {
            if let Ok(metadata) = entry.metadata().await {
                if metadata.is_file() {
                    if let Some(filename) = entry.file_name().to_str() {
                        if !filename.ends_with(".download") {
                            let size = metadata.len();
                            cached_shards.push((filename.to_string(), size));
                            total_size += size;
                        }
                    }
                }
            }
        }

        // Update internal state
        {
            let mut sizes = self.shard_sizes.lock().unwrap();
            let mut queue = self.lru_queue.lock().unwrap();
            let mut current_size = self.current_size_bytes.lock().unwrap();

            for (shard_name, size) in cached_shards {
                sizes.insert(shard_name.clone(), size);
                queue.push_back(shard_name);
            }

            *current_size = total_size;
        }

        Ok(())
    }
}
