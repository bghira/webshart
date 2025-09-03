use crate::{
    FileInfo,
    error::{Result, WebshartError},
};
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

pub trait FileLoader: Send + Sync {
    fn load_file(&self, file_info: &FileInfo) -> Result<Vec<u8>>;
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
        self.runtime.block_on(async {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()?;

            let mut request = client.get(&self.url).header(
                "Range",
                format!(
                    "bytes={}-{}",
                    file_info.offset,
                    file_info.offset + file_info.length - 1
                ),
            );

            if let Some(token) = &self.token {
                request = request.bearer_auth(token);
            }

            let response = request.send().await?;

            if response.status().is_success()
                || response.status() == reqwest::StatusCode::PARTIAL_CONTENT
            {
                Ok(response.bytes().await?.to_vec())
            } else if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                Err(WebshartError::RateLimited)
            } else {
                Err(WebshartError::Http(reqwest::Error::from(
                    response.error_for_status().unwrap_err(),
                )))
            }
        })
    }
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
