use crate::error::{Result, WebshartError};
use crate::metadata::ShardMetadata;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::runtime::Runtime;

#[derive(Debug, Clone)]
pub struct MetadataResolver {
    /// Optional separate metadata location (local path or HF repo)
    metadata_source: Option<String>,
    /// HF token for accessing metadata
    hf_token: Option<String>,
    /// Client for HTTP requests
    client: reqwest::Client,
    /// Runtime for async operations
    runtime: Arc<Runtime>,
}

impl MetadataResolver {
    pub fn new(
        metadata_source: Option<String>,
        hf_token: Option<String>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            metadata_source,
            hf_token,
            client: reqwest::Client::new(),
            runtime,
        }
    }

    /// Resolve metadata location for a given tar file
    pub fn resolve_metadata_path(
        &self,
        tar_path: &str,
        base_name: &str,
        is_remote: bool,
    ) -> String {
        if let Some(metadata_source) = &self.metadata_source {
            // Extract subfolder path from tar_path
            let subfolder = self.extract_subfolder(tar_path, is_remote);

            if is_remote {
                // If metadata source is a HF repo
                if !metadata_source.starts_with("http") && metadata_source.contains('/') {
                    let base_url = format!(
                        "https://huggingface.co/datasets/{}/resolve/main",
                        metadata_source
                    );
                    if let Some(sub) = subfolder {
                        format!("{}/{}/{}.json", base_url, sub, base_name)
                    } else {
                        format!("{}/{}.json", base_url, base_name)
                    }
                } else if metadata_source.starts_with("http") {
                    let base_url = metadata_source.trim_end_matches('/');
                    if let Some(sub) = subfolder {
                        format!("{}/{}/{}.json", base_url, sub, base_name)
                    } else {
                        format!("{}/{}.json", base_url, base_name)
                    }
                } else {
                    // Local path for remote dataset metadata
                    let mut path = Path::new(metadata_source).to_path_buf();
                    if let Some(sub) = subfolder {
                        path = path.join(sub);
                    }
                    path.join(format!("{}.json", base_name))
                        .to_string_lossy()
                        .to_string()
                }
            } else {
                // Local metadata path
                let mut path = Path::new(metadata_source).to_path_buf();
                if let Some(sub) = subfolder {
                    path = path.join(sub);
                }
                path.join(format!("{}.json", base_name))
                    .to_string_lossy()
                    .to_string()
            }
        } else {
            // Default: co-located with tar
            tar_path.replace(".tar", ".json")
        }
    }

    pub fn get_source(&self) -> Option<String> {
        self.metadata_source.clone()
    }

    /// Extract subfolder from tar path
    fn extract_subfolder(&self, tar_path: &str, is_remote: bool) -> Option<String> {
        if is_remote {
            // For URLs like: https://huggingface.co/datasets/repo/resolve/main/subfolder/file.tar
            // Extract "subfolder" part
            if let Some(pos) = tar_path.find("/resolve/main/") {
                let after_main = &tar_path[pos + 14..]; // Skip "/resolve/main/"
                if let Some(last_slash) = after_main.rfind('/') {
                    if last_slash > 0 {
                        return Some(after_main[..last_slash].to_string());
                    }
                }
            }
        } else {
            // For local paths, this is trickier - we'd need to know the base dataset path
            // For now, return None and handle this in the discovery module
        }
        None
    }

    /// Load metadata from resolved path
    pub async fn load_metadata(
        &self,
        metadata_path: &str,
        is_remote: bool,
    ) -> Result<ShardMetadata> {
        if is_remote && metadata_path.starts_with("http") {
            self.load_remote_metadata(metadata_path).await
        } else {
            self.load_local_metadata(metadata_path)
        }
    }

    /// Check if metadata exists at the resolved location
    pub fn metadata_exists(&self, metadata_path: &str, is_remote: bool) -> bool {
        if is_remote && metadata_path.starts_with("http") {
            self.runtime
                .block_on(self.check_remote_metadata(metadata_path))
        } else {
            Path::new(metadata_path).exists()
        }
    }

    fn load_local_metadata(&self, path: &str) -> Result<ShardMetadata> {
        let content = fs::read_to_string(path)?;
        let metadata: ShardMetadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }

    async fn load_remote_metadata(&self, url: &str) -> Result<ShardMetadata> {
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

        let response_text = response.text().await?;
        serde_json::from_str::<ShardMetadata>(&response_text).map_err(|e| {
            WebshartError::MetadataNotFound(format!("Invalid JSON in metadata file: {}", e))
        })
    }

    async fn check_remote_metadata(&self, url: &str) -> bool {
        let mut request = self.client.head(url);

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        match request.send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        }
    }
}
