// dataloader/shard_cache.rs
use crate::digest_to_hex;
use crate::error::{Result, WebshartError};
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;

const LOCK_DIR_NAME: &str = ".webshart-locks";
const DOWNLOAD_DIR_NAME: &str = ".webshart-downloads";
const ACCESS_DIR_NAME: &str = ".webshart-access";
const CACHE_LOCK_NAME: &str = "cache.lock";

static DOWNLOAD_ID: AtomicU64 = AtomicU64::new(0);

/// An RAII guard that holds a shared lock on a shard.
/// The lock is released when this struct is dropped.
#[derive(Debug)]
pub struct ShardLockGuard {
    #[allow(dead_code)]
    file: std::fs::File,
}

#[derive(Debug)]
struct ExclusiveLockGuard {
    #[allow(dead_code)]
    file: std::fs::File,
}

#[derive(Debug)]
struct CachedShard {
    name: String,
    size: u64,
    last_used: SystemTime,
}

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
            .map_err(WebshartError::Io)?;
        fs::create_dir_all(self.lock_dir())
            .await
            .map_err(WebshartError::Io)?;
        fs::create_dir_all(self.download_dir())
            .await
            .map_err(WebshartError::Io)?;
        fs::create_dir_all(self.access_dir())
            .await
            .map_err(WebshartError::Io)
    }

    pub fn get_cached_shard_path(&self, shard_name: &str) -> PathBuf {
        self.cache_dir.join(shard_name)
    }

    pub async fn is_cached(&self, shard_name: &str) -> bool {
        self.get_cached_shard_path(shard_name).is_file()
    }

    pub async fn lock_shard_for_reading(&self, shard_name: &str) -> Result<ShardLockGuard> {
        let lock_path = self.shard_lock_path(shard_name);
        let cached_path = self.get_cached_shard_path(shard_name);
        let file = tokio::task::spawn_blocking(move || -> std::io::Result<std::fs::File> {
            let file = Self::open_lock_file(&lock_path)?;
            file.lock_shared()?;
            Ok(file)
        })
        .await
        .map_err(Self::join_error)??;

        // Check after taking the lock. An evictor cannot remove the shard between
        // this check and the caller finishing its read.
        if !cached_path.is_file() {
            return Err(WebshartError::CacheMiss(shard_name.to_string()));
        }

        Ok(ShardLockGuard { file })
    }

    pub fn is_shard_locked(&self, shard_name: &str) -> bool {
        let Ok(file) = Self::open_lock_file(&self.shard_lock_path(shard_name)) else {
            return false;
        };
        file.try_lock_exclusive().is_err()
    }

    pub async fn cache_shard(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
    ) -> Result<PathBuf> {
        let (path, _) = self
            .ensure_cached(shard_name, remote_url, token, false)
            .await?;
        Ok(path)
    }

    pub async fn cache_shard_for_reading(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
    ) -> Result<(PathBuf, ShardLockGuard)> {
        let (path, read_lock) = self
            .ensure_cached(shard_name, remote_url, token, true)
            .await?;
        Ok((path, read_lock.expect("read lock requested")))
    }

    async fn ensure_cached(
        &self,
        shard_name: &str,
        remote_url: &str,
        token: Option<String>,
        lock_for_reading: bool,
    ) -> Result<(PathBuf, Option<ShardLockGuard>)> {
        let cached_path = self.get_cached_shard_path(shard_name);

        // The semaphore limits this process. The file lock prevents a second
        // process using the same cache directory from downloading the same shard.
        let _permit = self.download_semaphore.acquire().await.map_err(|e| {
            WebshartError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to acquire download permit: {}", e),
            ))
        })?;
        let _download_lock = self
            .lock_exclusive(self.download_lock_path(shard_name))
            .await?;

        // Recheck after taking the cross-process download lock.
        if let Ok(read_lock) = self.lock_shard_for_reading(shard_name).await {
            self.touch_shard(shard_name).await;
            return Ok((
                cached_path,
                if lock_for_reading {
                    Some(read_lock)
                } else {
                    None
                },
            ));
        }

        let temp_path = self.temp_download_path(shard_name);
        self.active_downloads
            .lock()
            .unwrap()
            .insert(shard_name.to_string(), temp_path.clone());

        let result = self
            .download_shard_to_disk(remote_url, token, shard_name, &temp_path)
            .await;

        self.active_downloads.lock().unwrap().remove(shard_name);

        match result {
            Ok(shard_size) => {
                self.record_cached_shard(shard_name, shard_size);
                // The download lock is still held, and evictors take that lock
                // before the shard lock. This closes the commit-to-read gap.
                let read_lock = if lock_for_reading {
                    Some(self.lock_shard_for_reading(shard_name).await?)
                } else {
                    None
                };
                Ok((cached_path, read_lock))
            }
            Err(error) => {
                let _ = fs::remove_file(&temp_path).await;
                Err(error)
            }
        }
    }

    async fn download_shard_to_disk(
        &self,
        url: &str,
        token: Option<String>,
        shard_name: &str,
        temp_path: &Path,
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

        let response = request
            .send()
            .await
            .map_err(WebshartError::from)?
            .error_for_status()
            .map_err(WebshartError::from)?;

        let mut file = File::create(temp_path).await.map_err(WebshartError::Io)?;
        let mut stream = response.bytes_stream();
        let mut bytes_written = 0u64;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(WebshartError::from)?;
            bytes_written += chunk.len() as u64;
            file.write_all(&chunk).await.map_err(WebshartError::Io)?;

            if bytes_written % (1024 * 1024) == 0 {
                file.sync_data().await.map_err(WebshartError::Io)?;
            }
        }

        file.flush().await.map_err(WebshartError::Io)?;
        file.sync_all().await.map_err(WebshartError::Io)?;
        drop(file);

        self.commit_download(temp_path, shard_name, bytes_written)
            .await?;

        Ok(bytes_written)
    }

    /// Serializes the disk-space decision and final rename across processes.
    async fn commit_download(
        &self,
        temp_path: &Path,
        shard_name: &str,
        shard_size: u64,
    ) -> Result<()> {
        let _cache_lock = self.lock_exclusive(self.cache_lock_path()).await?;
        let _shard_lock = self
            .lock_exclusive(self.shard_lock_path(shard_name))
            .await?;

        self.evict_if_needed_locked(shard_size, Some(shard_name))
            .await?;

        fs::rename(temp_path, self.get_cached_shard_path(shard_name))
            .await
            .map_err(WebshartError::Io)?;
        self.touch_shard(shard_name).await;
        Ok(())
    }

    /// The caller must hold the cache-wide exclusive lock.
    async fn evict_if_needed_locked(
        &self,
        needed_bytes: u64,
        shard_to_keep: Option<&str>,
    ) -> Result<()> {
        let mut cached_shards = self.scan_cached_shards().await?;
        let mut current_size = cached_shards.iter().map(|shard| shard.size).sum::<u64>();
        cached_shards.sort_by_key(|shard| shard.last_used);

        for shard in &cached_shards {
            if current_size.saturating_add(needed_bytes) <= self.cache_limit_bytes {
                break;
            }
            if shard_to_keep == Some(shard.name.as_str()) {
                continue;
            }

            let download_lock = match Self::open_lock_file(&self.download_lock_path(&shard.name)) {
                Ok(file) => file,
                Err(_) => continue,
            };
            if download_lock.try_lock_exclusive().is_err() {
                continue;
            }

            let lock_path = self.shard_lock_path(&shard.name);
            let lock_file = match Self::open_lock_file(&lock_path) {
                Ok(file) => file,
                Err(_) => continue,
            };

            // A shared reader lock means another process is using this shard.
            // Keep the exclusive lock held until deletion completes, closing the
            // old check/unlock/delete race.
            if lock_file.try_lock_exclusive().is_err() {
                continue;
            }

            let path = self.get_cached_shard_path(&shard.name);
            match fs::remove_file(&path).await {
                Ok(()) => {
                    current_size = current_size.saturating_sub(shard.size);
                    let _ = fs::remove_file(self.access_path(&shard.name)).await;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    current_size = current_size.saturating_sub(shard.size);
                }
                Err(_) => {}
            }
            // Both lock files are deliberately dropped only after remove_file
            // returns.
        }

        let remaining = self.scan_cached_shards().await?;
        self.replace_metadata(&remaining);
        Ok(())
    }

    pub async fn get_cached_file_size(&self, shard_name: &str) -> Result<u64> {
        if let Some(temp_path) = self.get_active_download_path(shard_name) {
            if let Ok(metadata) = fs::metadata(&temp_path).await {
                return Ok(metadata.len());
            }
        }

        let path = self.get_cached_shard_path(shard_name);
        if path.is_file() {
            return fs::metadata(&path)
                .await
                .map(|metadata| metadata.len())
                .map_err(WebshartError::Io);
        }

        Err(WebshartError::CacheMiss(shard_name.to_string()))
    }

    pub fn get_active_download_path(&self, shard_name: &str) -> Option<PathBuf> {
        self.active_downloads
            .lock()
            .unwrap()
            .get(shard_name)
            .cloned()
    }

    pub async fn get_download_progress(&self, shard_name: &str) -> Option<u64> {
        let path = self.get_active_download_path(shard_name)?;
        fs::metadata(path).await.ok().map(|metadata| metadata.len())
    }

    async fn touch_shard(&self, shard_name: &str) {
        // The marker's mtime is a process-shared LRU signal.
        let _ = fs::write(self.access_path(shard_name), b"").await;

        let mut queue = self.lru_queue.lock().unwrap();
        if let Some(position) = queue.iter().position(|name| name == shard_name) {
            queue.remove(position);
        }
        queue.push_back(shard_name.to_string());
    }

    pub async fn initialize_from_disk(&mut self) -> Result<()> {
        self.cleanup_stale_downloads().await?;
        let mut cached_shards = self.scan_cached_shards().await?;
        cached_shards.sort_by_key(|shard| shard.last_used);
        self.replace_metadata(&cached_shards);
        Ok(())
    }

    async fn cleanup_stale_downloads(&self) -> Result<()> {
        let mut entries = fs::read_dir(self.download_dir())
            .await
            .map_err(WebshartError::Io)?;

        while let Some(entry) = entries.next_entry().await.map_err(WebshartError::Io)? {
            let filename = entry.file_name();
            let Some(filename) = filename.to_str() else {
                continue;
            };
            let Some(shard_key) = filename.split('.').next() else {
                continue;
            };
            if shard_key.len() != 64 {
                continue;
            }

            let lock_file = match Self::open_lock_file(&self.download_lock_path_for_key(shard_key))
            {
                Ok(file) => file,
                Err(_) => continue,
            };
            if lock_file.try_lock_exclusive().is_ok() {
                let _ = fs::remove_file(entry.path()).await;
            }
        }

        // Clean up temporary files left by webshart versions that downloaded
        // directly in the cache root. Current downloads use the protected
        // download directory above.
        let mut entries = fs::read_dir(&self.cache_dir)
            .await
            .map_err(WebshartError::Io)?;
        while let Some(entry) = entries.next_entry().await.map_err(WebshartError::Io)? {
            if entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".download"))
            {
                let _ = fs::remove_file(entry.path()).await;
            }
        }

        Ok(())
    }

    async fn scan_cached_shards(&self) -> Result<Vec<CachedShard>> {
        let mut entries = fs::read_dir(&self.cache_dir)
            .await
            .map_err(WebshartError::Io)?;
        let mut cached_shards = Vec::new();

        while let Some(entry) = entries.next_entry().await.map_err(WebshartError::Io)? {
            let metadata = match entry.metadata().await {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => continue,
            };
            let Some(filename) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if filename.ends_with(".download") {
                continue;
            }

            let last_used = match fs::metadata(self.access_path(&filename)).await {
                Ok(access_metadata) => access_metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                Err(_) => metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            };
            cached_shards.push(CachedShard {
                name: filename,
                size: metadata.len(),
                last_used,
            });
        }

        Ok(cached_shards)
    }

    fn replace_metadata(&self, cached_shards: &[CachedShard]) {
        let mut sizes = self.shard_sizes.lock().unwrap();
        let mut queue = self.lru_queue.lock().unwrap();
        let mut current_size = self.current_size_bytes.lock().unwrap();
        sizes.clear();
        queue.clear();
        *current_size = 0;

        for shard in cached_shards {
            sizes.insert(shard.name.clone(), shard.size);
            queue.push_back(shard.name.clone());
            *current_size += shard.size;
        }
    }

    fn record_cached_shard(&self, shard_name: &str, shard_size: u64) {
        let mut sizes = self.shard_sizes.lock().unwrap();
        let mut queue = self.lru_queue.lock().unwrap();
        let mut current_size = self.current_size_bytes.lock().unwrap();

        if let Some(previous_size) = sizes.insert(shard_name.to_string(), shard_size) {
            *current_size = current_size.saturating_sub(previous_size);
        }
        if let Some(position) = queue.iter().position(|name| name == shard_name) {
            queue.remove(position);
        }
        queue.push_back(shard_name.to_string());
        *current_size = current_size.saturating_add(shard_size);
    }

    async fn lock_exclusive(&self, path: PathBuf) -> Result<ExclusiveLockGuard> {
        let file = tokio::task::spawn_blocking(move || -> std::io::Result<std::fs::File> {
            let file = Self::open_lock_file(&path)?;
            file.lock_exclusive()?;
            Ok(file)
        })
        .await
        .map_err(Self::join_error)??;
        Ok(ExclusiveLockGuard { file })
    }

    fn open_lock_file(path: &Path) -> std::io::Result<std::fs::File> {
        OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
    }

    fn join_error(error: tokio::task::JoinError) -> WebshartError {
        WebshartError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Cache lock task failed: {}", error),
        ))
    }

    fn shard_key(shard_name: &str) -> String {
        let digest = Sha256::digest(shard_name.as_bytes());
        digest_to_hex(digest)
    }

    fn lock_dir(&self) -> PathBuf {
        self.cache_dir.join(LOCK_DIR_NAME)
    }

    fn download_dir(&self) -> PathBuf {
        self.cache_dir.join(DOWNLOAD_DIR_NAME)
    }

    fn access_dir(&self) -> PathBuf {
        self.cache_dir.join(ACCESS_DIR_NAME)
    }

    fn cache_lock_path(&self) -> PathBuf {
        self.lock_dir().join(CACHE_LOCK_NAME)
    }

    fn shard_lock_path(&self, shard_name: &str) -> PathBuf {
        self.lock_dir()
            .join(format!("{}.shard.lock", Self::shard_key(shard_name)))
    }

    fn download_lock_path(&self, shard_name: &str) -> PathBuf {
        self.download_lock_path_for_key(&Self::shard_key(shard_name))
    }

    fn download_lock_path_for_key(&self, shard_key: &str) -> PathBuf {
        self.lock_dir().join(format!("{}.download.lock", shard_key))
    }

    fn access_path(&self, shard_name: &str) -> PathBuf {
        self.access_dir().join(Self::shard_key(shard_name))
    }

    fn temp_download_path(&self, shard_name: &str) -> PathBuf {
        let id = DOWNLOAD_ID.fetch_add(1, Ordering::Relaxed);
        self.download_dir().join(format!(
            "{}.{}.{}.download",
            Self::shard_key(shard_name),
            std::process::id(),
            id
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    const CHILD_CACHE_DIR: &str = "WEBSHART_TEST_CHILD_CACHE_DIR";
    const CHILD_READY_PATH: &str = "WEBSHART_TEST_CHILD_READY_PATH";
    const CHILD_RELEASE_PATH: &str = "WEBSHART_TEST_CHILD_RELEASE_PATH";
    const CHILD_TEST_NAME: &str = "dataloader::shard_cache::tests::shard_lock_child_process";

    fn cache_with_byte_limit(path: &Path, bytes: u64) -> ShardCache {
        ShardCache::new(
            path.to_path_buf(),
            bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            2,
        )
    }

    #[tokio::test]
    async fn eviction_does_not_remove_a_shard_locked_by_another_cache() {
        let temp_dir = tempdir().unwrap();
        fs::write(temp_dir.path().join("locked.tar"), vec![0; 10])
            .await
            .unwrap();

        let mut reader_cache = cache_with_byte_limit(temp_dir.path(), 10);
        let mut evictor_cache = cache_with_byte_limit(temp_dir.path(), 10);
        reader_cache.ensure_cache_dir().await.unwrap();
        reader_cache.initialize_from_disk().await.unwrap();
        evictor_cache.initialize_from_disk().await.unwrap();

        let read_lock = reader_cache
            .lock_shard_for_reading("locked.tar")
            .await
            .unwrap();
        let cache_lock = evictor_cache
            .lock_exclusive(evictor_cache.cache_lock_path())
            .await
            .unwrap();
        evictor_cache.evict_if_needed_locked(1, None).await.unwrap();
        drop(cache_lock);
        assert!(temp_dir.path().join("locked.tar").exists());

        drop(read_lock);
        let cache_lock = evictor_cache
            .lock_exclusive(evictor_cache.cache_lock_path())
            .await
            .unwrap();
        evictor_cache.evict_if_needed_locked(1, None).await.unwrap();
        drop(cache_lock);
        assert!(!temp_dir.path().join("locked.tar").exists());
    }

    // This ignored test is invoked directly by the multi-process regression
    // test below. Keeping the lock holder in the test binary avoids requiring a
    // separate test executable while still crossing a real OS process boundary.
    #[test]
    #[ignore]
    fn shard_lock_child_process() {
        let cache_dir = PathBuf::from(std::env::var_os(CHILD_CACHE_DIR).unwrap());
        let ready_path = PathBuf::from(std::env::var_os(CHILD_READY_PATH).unwrap());
        let release_path = PathBuf::from(std::env::var_os(CHILD_RELEASE_PATH).unwrap());
        let runtime = tokio::runtime::Runtime::new().unwrap();

        runtime.block_on(async {
            let mut cache = cache_with_byte_limit(&cache_dir, 10);
            cache.ensure_cache_dir().await.unwrap();
            cache.initialize_from_disk().await.unwrap();
            let _read_lock = cache.lock_shard_for_reading("locked.tar").await.unwrap();
            fs::write(&ready_path, b"ready").await.unwrap();

            for _ in 0..500 {
                if release_path.exists() {
                    return;
                }
                thread::sleep(Duration::from_millis(10));
            }
            panic!("parent process did not release child shard lock");
        });
    }

    #[tokio::test]
    async fn eviction_respects_a_lock_held_by_another_process() {
        let temp_dir = tempdir().unwrap();
        let ready_path = temp_dir.path().join("child.ready");
        let release_path = temp_dir.path().join("child.release");
        let shard_path = temp_dir.path().join("locked.tar");
        let cache = cache_with_byte_limit(temp_dir.path(), 10);
        cache.ensure_cache_dir().await.unwrap();
        fs::write(&shard_path, vec![0; 10]).await.unwrap();

        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", CHILD_TEST_NAME, "--nocapture"])
            .env(CHILD_CACHE_DIR, temp_dir.path())
            .env(CHILD_READY_PATH, &ready_path)
            .env(CHILD_RELEASE_PATH, &release_path)
            .stdout(Stdio::null())
            .spawn()
            .unwrap();

        for _ in 0..500 {
            if ready_path.exists() {
                break;
            }
            if let Some(status) = child.try_wait().unwrap() {
                panic!("lock-holder child exited early with {status}");
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(ready_path.exists(), "lock-holder child never became ready");

        let mut evictor_cache = cache_with_byte_limit(temp_dir.path(), 10);
        evictor_cache.initialize_from_disk().await.unwrap();
        let cache_lock = evictor_cache
            .lock_exclusive(evictor_cache.cache_lock_path())
            .await
            .unwrap();
        evictor_cache.evict_if_needed_locked(1, None).await.unwrap();
        drop(cache_lock);
        assert!(shard_path.exists());

        fs::write(&release_path, b"release").await.unwrap();
        assert!(child.wait().unwrap().success());

        let cache_lock = evictor_cache
            .lock_exclusive(evictor_cache.cache_lock_path())
            .await
            .unwrap();
        evictor_cache.evict_if_needed_locked(1, None).await.unwrap();
        drop(cache_lock);
        assert!(!shard_path.exists());
    }

    #[tokio::test]
    async fn initialization_preserves_another_process_active_download() {
        let temp_dir = tempdir().unwrap();
        let cache = cache_with_byte_limit(temp_dir.path(), 10);
        cache.ensure_cache_dir().await.unwrap();

        let download_lock = cache
            .lock_exclusive(cache.download_lock_path("shard.tar"))
            .await
            .unwrap();
        let temp_path = cache.temp_download_path("shard.tar");
        fs::write(&temp_path, b"partial").await.unwrap();

        let mut second_cache = cache_with_byte_limit(temp_dir.path(), 10);
        second_cache.initialize_from_disk().await.unwrap();
        assert!(temp_path.exists());

        drop(download_lock);
        second_cache.initialize_from_disk().await.unwrap();
        assert!(!temp_path.exists());
    }

    #[tokio::test]
    async fn eviction_refreshes_cache_contents_from_disk() {
        let temp_dir = tempdir().unwrap();
        let mut cache = cache_with_byte_limit(temp_dir.path(), 10);
        cache.ensure_cache_dir().await.unwrap();
        cache.initialize_from_disk().await.unwrap();

        // Simulate a shard committed by another process after initialization.
        fs::write(temp_dir.path().join("external.tar"), vec![0; 10])
            .await
            .unwrap();

        let cache_lock = cache.lock_exclusive(cache.cache_lock_path()).await.unwrap();
        cache.evict_if_needed_locked(1, None).await.unwrap();
        drop(cache_lock);
        assert!(!temp_dir.path().join("external.tar").exists());
    }

    #[tokio::test]
    async fn cache_instances_download_a_shared_shard_only_once() {
        let temp_dir = tempdir().unwrap();
        let cache_one = cache_with_byte_limit(temp_dir.path(), 100);
        let cache_two = cache_with_byte_limit(temp_dir.path(), 100);
        cache_one.ensure_cache_dir().await.unwrap();

        let mut server = mockito::Server::new_async().await;
        let request = server
            .mock("GET", "/shared.tar")
            .with_status(200)
            .with_body("shard contents")
            .expect(1)
            .create_async()
            .await;
        let url = format!("{}/shared.tar", server.url());

        let (first, second) = tokio::join!(
            cache_one.cache_shard("shared.tar", &url, None),
            cache_two.cache_shard("shared.tar", &url, None),
        );

        assert_eq!(first.unwrap(), temp_dir.path().join("shared.tar"));
        assert_eq!(second.unwrap(), temp_dir.path().join("shared.tar"));
        assert_eq!(
            fs::read(temp_dir.path().join("shared.tar")).await.unwrap(),
            b"shard contents"
        );
        request.assert_async().await;
    }
}
