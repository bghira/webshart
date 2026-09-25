use crate::{
    error::{Result, WebshartError},
    FileInfo,
};
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Duration;
use tokio::runtime::Runtime;

pub trait FileLoader: Send + Sync {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>>;
}

pub(crate) fn file_http_client() -> Result<reqwest::Client> {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();

    if let Some(client) = CLIENT.get() {
        return Ok(client.clone());
    }

    let client = reqwest::Client::builder()
        .pool_idle_timeout(Duration::from_secs(30))
        .pool_max_idle_per_host(8)
        .build()
        .map_err(WebshartError::from)?;

    let _ = CLIENT.set(client);
    Ok(CLIENT.get().expect("HTTP client initialized").clone())
}

pub struct LocalFileLoader {
    tar_path: String,
}

impl LocalFileLoader {
    pub fn new(tar_path: String) -> Self {
        Self { tar_path }
    }
}

impl FileLoader for LocalFileLoader {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};

        let mut file = std::fs::File::open(&self.tar_path)?;
        file.seek(SeekFrom::Start(file_info.offset))?;

        let mut buffer = vec![0u8; file_info.length as usize];
        file.read_exact(&mut buffer)?;

        Ok(buffer)
    }
}

pub struct RemoteFileLoader {
    url: String,
    token: Option<String>,
    runtime: Arc<Runtime>,
}

impl RemoteFileLoader {
    pub fn new(url: String, token: Option<String>, runtime: Arc<Runtime>) -> Self {
        Self {
            url,
            token,
            runtime,
        }
    }
}

impl FileLoader for RemoteFileLoader {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>> {
        self.runtime.block_on(read_remote_range(
            &self.url,
            self.token.as_deref(),
            file_info.offset,
            file_info.length,
        ))
    }
}

pub(crate) async fn read_remote_range(
    url: &str,
    token: Option<&str>,
    offset: u64,
    length: u64,
) -> Result<Vec<u8>> {
    if length == 0 {
        return Ok(Vec::new());
    }
    if let Some(file) = crate::xet::resolve(url, token).await? {
        return crate::xet::client()?
            .read_range(&file, offset, length)
            .await;
    }
    let end = offset
        .checked_add(length - 1)
        .ok_or_else(|| WebshartError::InvalidShardFormat("Byte range overflow".into()))?;
    let mut request = file_http_client()?
        .get(url)
        .header("Range", format!("bytes={offset}-{end}"))
        .timeout(Duration::from_secs(60));
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let response = request.send().await?;
    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Err(WebshartError::RateLimited);
    }
    Ok(response.error_for_status()?.bytes().await?.to_vec())
}

pub fn create_file_loader(
    tar_path: &str,
    is_remote: bool,
    token: Option<String>,
    runtime: Arc<Runtime>,
) -> Box<dyn FileLoader> {
    if is_remote {
        Box::new(RemoteFileLoader::new(tar_path.to_string(), token, runtime))
    } else {
        Box::new(LocalFileLoader::new(tar_path.to_string()))
    }
}
