use crate::error::{Result, WebshartError};
use futures::Stream;
use std::io::Read;
use std::pin::Pin;
use std::sync::Arc;
use tar::Archive;
use tokio::runtime::Runtime;

/// A file entry from a TAR archive
#[derive(Debug, Clone)]
pub struct TarFileEntry {
    pub path: String,
    pub offset: u64,
    pub size: u64,
    pub data: Vec<u8>,
}

/// Configuration for streaming TAR files
#[derive(Clone)]
pub struct StreamConfig {
    pub compute_sha256: bool,
    pub include_image_geometry: bool,
    pub load_file_data: bool,
    pub max_file_size: u64, // Max size to load into memory
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            compute_sha256: false,
            include_image_geometry: false,
            load_file_data: true,
            max_file_size: 50_000_000, // 50MB default
        }
    }
}

/// Trait for streaming TAR entries
pub trait TarStreamer {
    fn stream_entries(&self) -> Result<Box<dyn Iterator<Item = Result<TarFileEntry>>>>;
}

/// Local TAR file streamer
pub struct LocalTarStreamer {
    path: String,
    config: StreamConfig,
}

impl LocalTarStreamer {
    pub fn new(path: String, config: StreamConfig) -> Self {
        Self { path, config }
    }
}

impl TarStreamer for LocalTarStreamer {
    fn stream_entries(&self) -> Result<Box<dyn Iterator<Item = Result<TarFileEntry>>>> {
        let file = std::fs::File::open(&self.path)?;
        let config = self.config.clone();
        let mut archive = Archive::new(file);

        // Get all entries at once and collect them into a Vec
        // This is necessary because we can't store the entries iterator directly
        // due to lifetime constraints
        let entries_result = archive.entries();

        match entries_result {
            Ok(entries) => {
                // Process all entries and collect results
                let collected_entries: Vec<Result<TarFileEntry>> = entries
                    .filter_map(|entry_result| {
                        match entry_result {
                            Ok(mut entry) => {
                                // Get path
                                let path = match entry.path() {
                                    Ok(p) => p.to_string_lossy().to_string(),
                                    Err(e) => return Some(Err(e.into())),
                                };

                                let offset = entry.raw_header_position();
                                let size = entry.size();

                                // Only process regular files
                                if entry.header().entry_type() != tar::EntryType::Regular {
                                    return None;
                                }

                                // Load data if requested and within size limit
                                let data = if config.load_file_data
                                    && size <= config.max_file_size
                                    && size > 0
                                {
                                    let mut buffer = Vec::with_capacity(size as usize);
                                    match entry.read_to_end(&mut buffer) {
                                        Ok(_) => buffer,
                                        Err(e) => {
                                            eprintln!("Failed to read data for {}: {}", path, e);
                                            return Some(Err(e.into()));
                                        }
                                    }
                                } else {
                                    // Skip the entry data
                                    match std::io::copy(&mut entry, &mut std::io::sink()) {
                                        Ok(_) => Vec::new(),
                                        Err(e) => {
                                            eprintln!("Failed to skip data for {}: {}", path, e);
                                            return Some(Err(e.into()));
                                        }
                                    }
                                };

                                Some(Ok(TarFileEntry {
                                    path,
                                    offset: offset + 512, // Add header size
                                    size,
                                    data,
                                }))
                            }
                            Err(e) => {
                                eprintln!("Error reading tar entry: {}", e);
                                // Continue processing other entries instead of failing entirely
                                None
                            }
                        }
                    })
                    .collect();

                Ok(Box::new(collected_entries.into_iter()))
            }
            Err(e) => {
                eprintln!("Failed to read archive entries: {}", e);
                Err(e.into())
            }
        }
    }
}

/// Remote TAR streamer using HTTP
pub struct RemoteTarStreamer {
    url: String,
    hf_token: Option<String>,
    config: StreamConfig,
    runtime: Arc<Runtime>,
    client: reqwest::Client,
}

impl RemoteTarStreamer {
    pub fn new(
        url: String,
        hf_token: Option<String>,
        config: StreamConfig,
        runtime: Arc<Runtime>,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            url,
            hf_token,
            config,
            runtime,
            client,
        }
    }

    /// Stream entries from remote TAR file
    pub async fn stream_entries_async(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<TarFileEntry>> + Send>>> {
        let mut request = self.client.get(&self.url);

        if let Some(token) = &self.hf_token {
            request = request.bearer_auth(token);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(WebshartError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to download: {}", response.status()),
            )));
        }

        // Download the entire file to memory
        // In a production version, you'd want to stream chunks
        let bytes = response.bytes().await?;
        let mut archive = Archive::new(std::io::Cursor::new(bytes));
        let config = self.config.clone();

        // Process all entries
        let entries_result = archive.entries();

        let entries: Vec<Result<TarFileEntry>> = match entries_result {
            Ok(entries) => entries
                .filter_map(|entry_result| match entry_result {
                    Ok(mut entry) => {
                        let path = match entry.path() {
                            Ok(p) => p.to_string_lossy().to_string(),
                            Err(e) => return Some(Err(e.into())),
                        };

                        let offset = entry.raw_header_position();
                        let size = entry.size();

                        if entry.header().entry_type() != tar::EntryType::Regular {
                            return None;
                        }

                        let data =
                            if config.load_file_data && size <= config.max_file_size && size > 0 {
                                let mut buffer = Vec::with_capacity(size as usize);
                                match entry.read_to_end(&mut buffer) {
                                    Ok(_) => buffer,
                                    Err(e) => {
                                        eprintln!("Failed to read data for {}: {}", path, e);
                                        return Some(Err(e.into()));
                                    }
                                }
                            } else {
                                match std::io::copy(&mut entry, &mut std::io::sink()) {
                                    Ok(_) => Vec::new(),
                                    Err(e) => {
                                        eprintln!("Failed to skip data for {}: {}", path, e);
                                        return Some(Err(e.into()));
                                    }
                                }
                            };

                        Some(Ok(TarFileEntry {
                            path,
                            offset: offset + 512,
                            size,
                            data,
                        }))
                    }
                    Err(e) => {
                        eprintln!("Error reading tar entry: {}", e);
                        None
                    }
                })
                .collect(),
            Err(e) => {
                return Err(e.into());
            }
        };

        use futures::stream;
        Ok(Box::pin(stream::iter(entries)))
    }

    /// Synchronous version that downloads and processes the TAR file
    pub fn stream_entries(&self) -> Result<Box<dyn Iterator<Item = Result<TarFileEntry>>>> {
        self.runtime
            .block_on(async {
                let mut request = self.client.get(&self.url);

                if let Some(token) = &self.hf_token {
                    request = request.bearer_auth(token);
                }

                let response = request.send().await?;

                if !response.status().is_success() {
                    return Err(WebshartError::Io(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to download: {}", response.status()),
                    )));
                }

                let bytes = response.bytes().await?;
                Ok(bytes)
            })
            .and_then(|bytes| {
                let config = self.config.clone();
                let mut archive = Archive::new(std::io::Cursor::new(bytes));

                let entries_result = archive.entries();

                match entries_result {
                    Ok(entries) => {
                        let collected_entries: Vec<Result<TarFileEntry>> = entries
                            .filter_map(|entry_result| match entry_result {
                                Ok(mut entry) => {
                                    let path = match entry.path() {
                                        Ok(p) => p.to_string_lossy().to_string(),
                                        Err(e) => return Some(Err(e.into())),
                                    };

                                    let offset = entry.raw_header_position();
                                    let size = entry.size();

                                    if entry.header().entry_type() != tar::EntryType::Regular {
                                        return None;
                                    }

                                    let data = if config.load_file_data
                                        && size <= config.max_file_size
                                        && size > 0
                                    {
                                        let mut buffer = Vec::with_capacity(size as usize);
                                        match entry.read_to_end(&mut buffer) {
                                            Ok(_) => buffer,
                                            Err(e) => {
                                                eprintln!(
                                                    "Failed to read data for {}: {}",
                                                    path, e
                                                );
                                                return Some(Err(e.into()));
                                            }
                                        }
                                    } else {
                                        match std::io::copy(&mut entry, &mut std::io::sink()) {
                                            Ok(_) => Vec::new(),
                                            Err(e) => {
                                                eprintln!(
                                                    "Failed to skip data for {}: {}",
                                                    path, e
                                                );
                                                return Some(Err(e.into()));
                                            }
                                        }
                                    };

                                    Some(Ok(TarFileEntry {
                                        path,
                                        offset: offset + 512,
                                        size,
                                        data,
                                    }))
                                }
                                Err(e) => {
                                    eprintln!("Error reading tar entry: {}", e);
                                    None
                                }
                            })
                            .collect();

                        Ok(Box::new(collected_entries.into_iter())
                            as Box<dyn Iterator<Item = Result<TarFileEntry>>>)
                    }
                    Err(e) => {
                        eprintln!("Failed to read archive entries: {}", e);
                        Err(e.into())
                    }
                }
            })
    }
}

impl TarStreamer for RemoteTarStreamer {
    fn stream_entries(&self) -> Result<Box<dyn Iterator<Item = Result<TarFileEntry>>>> {
        self.stream_entries()
    }
}

/// Batch TAR processor for efficient parallel processing
pub struct BatchTarProcessor {
    runtime: Arc<Runtime>,
}

impl BatchTarProcessor {
    pub fn new(runtime: Arc<Runtime>) -> Self {
        Self { runtime }
    }

    /// Process TAR files in batches
    pub fn process_batch<F>(
        &self,
        tar_paths: Vec<String>,
        config: StreamConfig,
        processor: F,
    ) -> Vec<Result<Vec<TarFileEntry>>>
    where
        F: Fn(TarFileEntry) -> TarFileEntry + Send + Sync + 'static + Copy,
    {
        use rayon::prelude::*;

        tar_paths
            .par_iter()
            .map(|path| {
                let streamer = LocalTarStreamer::new(path.clone(), config.clone());
                match streamer.stream_entries() {
                    Ok(entries) => {
                        let processed: Result<Vec<_>> =
                            entries.map(|entry| entry.map(|e| processor(e))).collect();
                        processed
                    }
                    Err(e) => Err(e),
                }
            })
            .collect()
    }
}

// Helper functions for image processing
pub fn is_image_file(path: &str) -> bool {
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

pub fn extract_image_dimensions(data: &[u8]) -> Option<(u32, u32, f32)> {
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
