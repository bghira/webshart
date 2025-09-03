// dataloader/shard_cache.rs
use crate::dataloader::file_loading::FileLoader;
use crate::error::{Result, WebshartError};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::fs;
use tokio::sync::Semaphore;

#[derive(Debug, Clone)]
pub struct ShardCache {
    cache_dir: PathBuf,
    cache_limit_bytes: u64,
    current_size_bytes: Arc<Mutex<u64>>,
    lru_queue: Arc<Mutex<VecDeque<String>>>, // shard names in LRU order
    shard_sizes: Arc<Mutex<HashMap<String, u64>>>,
    download_semaphore: Arc<Semaphore>,
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

        // Download the shard
        let shard_data = self.download_shard(remote_url, token).await?;
        let shard_size = shard_data.len() as u64;

        // Make room if needed
        self.evict_if_needed(shard_size).await?;

        // Write to cache
        fs::write(&cached_path, shard_data)
            .await
            .map_err(|e| WebshartError::Io(e))?;

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

    async fn download_shard(&self, url: &str, token: Option<String>) -> Result<Vec<u8>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(WebshartError::from)?;

        let mut request = client.get(url);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await.map_err(WebshartError::from)?;
        let bytes = response.bytes().await.map_err(WebshartError::from)?;

        Ok(bytes.to_vec())
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
                        let size = metadata.len();
                        cached_shards.push((filename.to_string(), size));
                        total_size += size;
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
