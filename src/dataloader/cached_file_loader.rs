// dataloader/cached_file_loader.rs
use crate::dataloader::shard_cache::ShardCache;
use crate::error::Result;
use crate::file_loading::{FileLoader, create_file_loader};
use crate::metadata::FileInfo;
use std::sync::Arc;
use tokio::runtime::Runtime;

pub struct CachedFileLoader {
    cache: Arc<ShardCache>,
    runtime: Arc<Runtime>,
    token: Option<String>,
}

impl CachedFileLoader {
    pub fn new(cache: Arc<ShardCache>, runtime: Arc<Runtime>, token: Option<String>) -> Self {
        Self {
            cache,
            runtime,
            token,
        }
    }
}

impl FileLoader for CachedFileLoader {
    fn load_file(&self, tar_path: &str, file_info: &FileInfo) -> Result<Vec<u8>> {
        // Extract shard name from path
        let shard_name = tar_path.rsplit('/').next().unwrap_or(tar_path);

        // Check if we have a cached version
        let cached_path = self.runtime.block_on(async {
            if tar_path.starts_with("http") {
                // It's a remote shard, try to cache it
                match self
                    .cache
                    .cache_shard(shard_name, tar_path, self.token.clone())
                    .await
                {
                    Ok(path) => Some(path),
                    Err(_) => None,
                }
            } else if self.cache.is_cached(shard_name).await {
                Some(self.cache.get_cached_shard_path(shard_name))
            } else {
                None
            }
        });

        // Create appropriate loader
        let actual_path = if let Some(local_path) = cached_path {
            local_path.to_string_lossy().to_string()
        } else {
            tar_path.to_string()
        };

        let loader = create_file_loader(
            &actual_path,
            !cached_path.is_some(), // is_remote only if we didn't cache
            self.token.clone(),
            self.runtime.clone(),
        );

        loader.load_file(file_info)
    }
}
