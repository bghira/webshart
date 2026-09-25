use crate::{digest_to_hex, Result, WebshartError};
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::{header::HeaderMap, Client, Method, StatusCode, Url};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};

mod format;
#[cfg(test)]
mod tests;

const WINDOW_BYTES: u64 = 16 * 1024 * 1024;
const MAX_XORB_BYTES: usize = 64 * 1024 * 1024;
const MAX_DECODED_BYTES: u64 = 64 * 1024 * 1024;
const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 256;
const CACHE_TTL: Duration = Duration::from_secs(60);
const ATTEMPTS: usize = 3;

type FileCache = HashMap<(String, String), (Instant, Option<Arc<XetFile>>)>;

pub(crate) struct XetClient {
    http: Client,
    hub: Url,
    concurrency: usize,
    transfers: Semaphore,
    files: Mutex<FileCache>,
}

pub(crate) struct XetFile {
    hash: String,
    size: u64,
    sha256: Option<String>,
    auth_url: Url,
    hub_token: Option<String>,
    credentials: AsyncMutex<Option<Credentials>>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Credentials {
    access_token: String,
    exp: u64,
    cas_url: Url,
}

#[derive(Clone, Debug, Deserialize)]
struct Interval {
    start: u64,
    end: u64,
}

impl Interval {
    fn byte_length(&self) -> Result<u64> {
        self.end
            .checked_sub(self.start)
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| invalid("Invalid Xet byte range"))
    }
}

#[derive(Deserialize)]
struct Term {
    hash: String,
    unpacked_length: u64,
    range: Interval,
}

#[derive(Clone, Deserialize)]
struct FetchRange {
    chunks: Interval,
    bytes: Interval,
}

#[derive(Deserialize)]
struct Fetch {
    url: Url,
    ranges: Vec<FetchRange>,
}

#[derive(Deserialize)]
struct LegacyFetch {
    url: Url,
    range: Interval,
    url_range: Interval,
}

#[derive(Deserialize)]
struct Reconstruction {
    offset_into_first_range: u64,
    terms: Vec<Term>,
    #[serde(default)]
    xorbs: HashMap<String, Vec<Fetch>>,
    #[serde(default)]
    fetch_info: HashMap<String, Vec<LegacyFetch>>,
}

struct Reply {
    status: StatusCode,
    headers: HeaderMap,
    body: Vec<u8>,
}

fn invalid(message: impl Into<String>) -> WebshartError {
    WebshartError::Xet(message.into())
}

fn native_error(error: WebshartError) -> WebshartError {
    match error {
        WebshartError::Xet(_) | WebshartError::XetHttp(_) => error,
        _ => invalid(error.to_string()),
    }
}

fn check_status(reply: &Reply) -> Result<()> {
    if reply.status.is_success() {
        Ok(())
    } else {
        Err(WebshartError::XetHttp(reply.status.as_u16()))
    }
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn header<'headers>(headers: &'headers HeaderMap, name: &str) -> Option<&'headers str> {
    headers.get(name)?.to_str().ok()
}

pub(crate) fn client() -> Result<&'static XetClient> {
    static CLIENT: OnceLock<XetClient> = OnceLock::new();
    if let Some(client) = CLIENT.get() {
        return Ok(client);
    }
    let concurrency = match std::env::var("WEBSHART_XET_CONCURRENCY") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .filter(|value| (1..=64).contains(value))
            .ok_or_else(|| invalid("WEBSHART_XET_CONCURRENCY must be between 1 and 64"))?,
        Err(_) => 8,
    };
    let hub = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let instance = XetClient::new(&hub, concurrency)?;
    let _ = CLIENT.set(instance);
    Ok(CLIENT.get().expect("Xet client initialized"))
}

pub(crate) async fn resolve(url: &str, token: Option<&str>) -> Result<Option<Arc<XetFile>>> {
    if std::env::var("WEBSHART_DISABLE_XET").as_deref() == Ok("1") {
        return Ok(None);
    }
    client()?.resolve(url, token).await.map_err(native_error)
}

impl XetClient {
    fn new(hub: &str, concurrency: usize) -> Result<Self> {
        let hub = Url::parse(hub).map_err(|_| invalid("Invalid Hugging Face endpoint"))?;
        let http = Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(30))
            .timeout(Duration::from_secs(120))
            .pool_max_idle_per_host(concurrency)
            .build()?;
        Ok(Self {
            http,
            hub,
            concurrency,
            transfers: Semaphore::new(concurrency),
            files: Mutex::new(HashMap::new()),
        })
    }

    fn auth_url(&self, url: &Url, revision: Option<&str>) -> Option<Url> {
        if url.origin() != self.hub.origin() {
            return None;
        }
        let parts: Vec<_> = url.path().trim_start_matches('/').split('/').collect();
        let (kind, parts) = match parts.first().copied()? {
            "datasets" | "spaces" => (parts[0], &parts[1..]),
            _ => ("models", parts.as_slice()),
        };
        if parts.len() < 5 || parts[2] != "resolve" || parts.iter().any(|part| part.is_empty()) {
            return None;
        }
        self.hub
            .join(&format!(
                "/api/{kind}/{}/{}/xet-read-token/{}",
                parts[0],
                parts[1],
                revision.unwrap_or(parts[3])
            ))
            .ok()
    }

    async fn resolve(&self, url: &str, token: Option<&str>) -> Result<Option<Arc<XetFile>>> {
        let parsed = Url::parse(url).map_err(|_| invalid("Invalid remote file URL"))?;
        if self.auth_url(&parsed, None).is_none() {
            return Ok(None);
        }
        let key = (
            url.to_owned(),
            digest_to_hex(Sha256::digest(token.unwrap_or_default())),
        );
        if let Some((created, entry)) = self.files.lock().unwrap().get(&key) {
            if created.elapsed() < CACHE_TTL {
                return Ok(entry.clone());
            }
        }
        let reply = self.request(Method::HEAD, &parsed, token, None, 0).await?;
        if reply.status.is_client_error() || reply.status.is_server_error() {
            check_status(&reply)?;
        }
        let entry = if let Some(hash) = header(&reply.headers, "x-xet-hash") {
            if !valid_hash(hash) {
                return Err(invalid("Invalid Xet file ID"));
            }
            let size = header(&reply.headers, "x-linked-size")
                .and_then(|value| value.parse().ok())
                .ok_or_else(|| invalid("Missing or invalid Xet file size"))?;
            let sha256 = header(&reply.headers, "x-linked-etag")
                .map(|value| value.trim_matches('"'))
                .filter(|value| valid_hash(value))
                .map(str::to_ascii_lowercase);
            let revision = header(&reply.headers, "x-repo-commit");
            Some(Arc::new(XetFile {
                hash: hash.to_owned(),
                size,
                sha256,
                auth_url: self
                    .auth_url(&parsed, revision)
                    .ok_or_else(|| invalid("Invalid Xet authentication route"))?,
                hub_token: token.map(str::to_owned),
                credentials: AsyncMutex::new(None),
            }))
        } else {
            None
        };
        let mut files = self.files.lock().unwrap();
        if files.len() >= MAX_CACHE_ENTRIES {
            if let Some(oldest) = files
                .iter()
                .min_by_key(|(_, (created, _))| *created)
                .map(|(key, _)| key.clone())
            {
                files.remove(&oldest);
            }
        }
        files.insert(key, (Instant::now(), entry.clone()));
        Ok(entry)
    }

    async fn request(
        &self,
        method: Method,
        url: &Url,
        token: Option<&str>,
        range: Option<&str>,
        limit: usize,
    ) -> Result<Reply> {
        for attempt in 0..ATTEMPTS {
            let result = async {
                let mut request = self
                    .http
                    .request(method.clone(), url.clone())
                    .header("Accept-Encoding", "identity")
                    .header("Cache-Control", "no-cache");
                if let Some(token) = token {
                    request = request.bearer_auth(token);
                }
                if let Some(range) = range {
                    request = request.header("Range", range);
                }
                let response = request
                    .send()
                    .await
                    .map_err(|error| WebshartError::Http(error.without_url()))?;
                let status = response.status();
                let headers = response.headers().clone();
                let mut body = Vec::new();
                if method != Method::HEAD && status.is_success() {
                    if response
                        .content_length()
                        .is_some_and(|size| size > limit as u64)
                    {
                        return Err(invalid("Xet response exceeded its size limit"));
                    }
                    let mut chunks = response.bytes_stream();
                    while let Some(chunk) = chunks.next().await {
                        let chunk =
                            chunk.map_err(|error| WebshartError::Http(error.without_url()))?;
                        if chunk.len() > limit.saturating_sub(body.len()) {
                            return Err(invalid("Xet response exceeded its size limit"));
                        }
                        body.extend_from_slice(&chunk);
                    }
                }
                Ok(Reply {
                    status,
                    headers,
                    body,
                })
            }
            .await;
            let retry = match &result {
                Ok(reply) => {
                    reply.status == StatusCode::TOO_MANY_REQUESTS
                        || (reply.status.is_server_error()
                            && reply.status != StatusCode::NOT_IMPLEMENTED)
                }
                Err(WebshartError::Http(error)) => {
                    error.is_timeout()
                        || error.is_connect()
                        || error.is_body()
                        || error.is_decode()
                        || error.is_request()
                }
                _ => false,
            };
            if !retry || attempt + 1 == ATTEMPTS {
                return result;
            }
            let retry_after = result
                .as_ref()
                .ok()
                .and_then(|reply| header(&reply.headers, "retry-after"))
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(1 << attempt)
                .min(30);
            tokio::time::sleep(Duration::from_secs(retry_after)).await;
        }
        unreachable!()
    }

    async fn credentials(&self, file: &XetFile, refresh: bool) -> Result<Credentials> {
        let mut cached = file.credentials.lock().await;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        if let Some(credentials) = cached.as_ref() {
            if !refresh && credentials.exp > now + 30 {
                return Ok(credentials.clone());
            }
        }
        let reply = self
            .request(
                Method::GET,
                &file.auth_url,
                file.hub_token.as_deref(),
                None,
                MAX_JSON_BYTES,
            )
            .await?;
        check_status(&reply)?;
        let credentials: Credentials = serde_json::from_slice(&reply.body)?;
        if credentials.cas_url.scheme() != "https"
            && credentials.cas_url.origin() != self.hub.origin()
        {
            return Err(invalid("Xet CAS endpoint must use HTTPS"));
        }
        if credentials.exp <= now {
            return Err(invalid("Hub returned an expired Xet token"));
        }
        *cached = Some(credentials.clone());
        Ok(credentials)
    }

    async fn reconstruction(
        &self,
        file: &XetFile,
        start: u64,
        length: u64,
        refresh: bool,
    ) -> Result<Reconstruction> {
        let credentials = self.credentials(file, refresh).await?;
        let range = format!("bytes={}-{}", start, start + length - 1);
        let mut reply = self
            .request(
                Method::GET,
                &credentials
                    .cas_url
                    .join(&format!("/v2/reconstructions/{}", file.hash))
                    .map_err(|_| invalid("Invalid CAS endpoint"))?,
                Some(&credentials.access_token),
                Some(&range),
                MAX_JSON_BYTES,
            )
            .await?;
        if matches!(
            reply.status,
            StatusCode::NOT_FOUND | StatusCode::NOT_IMPLEMENTED
        ) {
            reply = self
                .request(
                    Method::GET,
                    &credentials
                        .cas_url
                        .join(&format!("/v1/reconstructions/{}", file.hash))
                        .map_err(|_| invalid("Invalid CAS endpoint"))?,
                    Some(&credentials.access_token),
                    Some(&range),
                    MAX_JSON_BYTES,
                )
                .await?;
        }
        check_status(&reply)?;
        let mut reconstruction: Reconstruction = serde_json::from_slice(&reply.body)?;
        for (hash, entries) in reconstruction.fetch_info.drain() {
            reconstruction
                .xorbs
                .entry(hash)
                .or_default()
                .extend(entries.into_iter().map(|entry| Fetch {
                    url: entry.url,
                    ranges: vec![FetchRange {
                        chunks: entry.range,
                        bytes: entry.url_range,
                    }],
                }));
        }
        Ok(reconstruction)
    }

    async fn fetch(&self, fetch: &Fetch, needed: &HashSet<u64>) -> Result<HashMap<u64, Vec<u8>>> {
        if fetch.url.scheme() != "https" && fetch.url.origin() != self.hub.origin() {
            return Err(invalid("Xet transfer URL must use HTTPS"));
        }
        if fetch.ranges.is_empty() || fetch.ranges.len() > 8192 {
            return Err(invalid("Invalid Xet fetch ranges"));
        }
        let mut encoded_length = 0u64;
        for range in &fetch.ranges {
            if range.bytes.start > range.bytes.end
                || range.chunks.start >= range.chunks.end
                || range.chunks.end - range.chunks.start > (MAX_XORB_BYTES / 8) as u64
            {
                return Err(invalid("Invalid Xet chunk or byte range"));
            }
            encoded_length = encoded_length
                .checked_add(range.bytes.byte_length()?)
                .ok_or_else(|| invalid("Xet range size overflow"))?;
        }
        if encoded_length > MAX_XORB_BYTES as u64 {
            return Err(invalid("Xet fetch exceeds maximum xorb size"));
        }
        let range = format!(
            "bytes={}",
            fetch
                .ranges
                .iter()
                .map(|range| format!("{}-{}", range.bytes.start, range.bytes.end))
                .collect::<Vec<_>>()
                .join(",")
        );
        let _permit = self
            .transfers
            .acquire()
            .await
            .map_err(|_| invalid("Xet downloader closed"))?;
        let reply = self
            .request(
                Method::GET,
                &fetch.url,
                None,
                Some(&range),
                MAX_XORB_BYTES + fetch.ranges.len() * 16384,
            )
            .await?;
        check_status(&reply)?;
        if reply.status != StatusCode::PARTIAL_CONTENT {
            return Err(invalid(
                "Xet transfer did not honor the requested byte ranges",
            ));
        }
        let parts = format::parts(&reply.body, &reply.headers, &fetch.ranges)?;
        let mut chunks = HashMap::new();
        let mut decoded_bytes = 0usize;
        for (range, part) in parts {
            for (index, chunk) in format::decode_chunks(part, &range.chunks, needed)? {
                decoded_bytes += chunk.len();
                if decoded_bytes > MAX_DECODED_BYTES as usize {
                    return Err(invalid("Xet decoded data exceeded its size limit"));
                }
                if chunks.insert(index, chunk).is_some() {
                    return Err(invalid("Overlapping Xet fetch ranges"));
                }
            }
        }
        Ok(chunks)
    }

    async fn read_window(&self, file: &XetFile, start: u64, length: u64) -> Result<Vec<u8>> {
        for attempt in 0..2 {
            let result = async {
                let reconstruction = self
                    .reconstruction(file, start, length, attempt != 0)
                    .await?;
                self.assemble(reconstruction, length).await
            }
            .await;
            match result {
                Err(WebshartError::XetHttp(401 | 403)) if attempt == 0 => continue,
                result => return result,
            }
        }
        unreachable!()
    }

    async fn assemble(&self, reconstruction: Reconstruction, length: u64) -> Result<Vec<u8>> {
        let mut needed: HashMap<&str, HashSet<u64>> = HashMap::new();
        let mut unpacked = 0u64;
        let mut chunk_references = 0u64;
        for term in &reconstruction.terms {
            if !valid_hash(&term.hash)
                || term.range.start >= term.range.end
                || term.range.end - term.range.start > 65536
            {
                return Err(invalid("Invalid Xet reconstruction term"));
            }
            let chunk_count = term.range.end - term.range.start;
            chunk_references += chunk_count;
            if chunk_count > term.unpacked_length || chunk_references > 65536 {
                return Err(invalid("Invalid Xet reconstruction chunk count"));
            }
            unpacked = unpacked
                .checked_add(term.unpacked_length)
                .ok_or_else(|| invalid("Xet size overflow"))?;
            if unpacked > MAX_DECODED_BYTES {
                return Err(invalid("Xet reconstruction exceeded its size limit"));
            }
            needed
                .entry(&term.hash)
                .or_default()
                .extend(term.range.start..term.range.end);
        }
        if reconstruction.offset_into_first_range
            > reconstruction
                .terms
                .first()
                .map_or(0, |term| term.unpacked_length)
            || unpacked.saturating_sub(reconstruction.offset_into_first_range) < length
        {
            return Err(invalid(
                "Xet reconstruction does not cover the requested bytes",
            ));
        }
        let mut jobs = Vec::new();
        for (hash, indices) in &needed {
            let fetches = reconstruction
                .xorbs
                .get(*hash)
                .ok_or_else(|| invalid("Missing Xet fetch information"))?;
            for fetch in fetches {
                jobs.push((*hash, indices, fetch));
            }
        }
        let mut downloads = stream::iter(jobs)
            .map(|(hash, indices, fetch)| async move {
                Ok::<_, WebshartError>((hash, self.fetch(fetch, indices).await?))
            })
            .buffer_unordered(self.concurrency);
        let mut chunks: HashMap<(&str, u64), Vec<u8>> = HashMap::new();
        let mut stored_bytes = 0usize;
        while let Some((hash, fetched)) = downloads.try_next().await? {
            for (index, chunk) in fetched {
                if let Some(previous) = chunks.get(&(hash, index)) {
                    if *previous != chunk {
                        return Err(invalid("Conflicting Xet chunks"));
                    }
                } else {
                    stored_bytes += chunk.len();
                    if stored_bytes > MAX_DECODED_BYTES as usize {
                        return Err(invalid("Xet decoded data exceeded its size limit"));
                    }
                    chunks.insert((hash, index), chunk);
                }
            }
        }
        let mut skip = reconstruction.offset_into_first_range;
        let mut output = Vec::with_capacity(length as usize);
        for term in &reconstruction.terms {
            let mut term_size = 0u64;
            for index in term.range.start..term.range.end {
                let chunk = chunks
                    .get(&(term.hash.as_str(), index))
                    .ok_or_else(|| invalid("Missing Xet chunk"))?;
                term_size += chunk.len() as u64;
                let offset = skip.min(chunk.len() as u64) as usize;
                skip -= offset as u64;
                let count = (chunk.len() - offset).min(length as usize - output.len());
                output.extend_from_slice(&chunk[offset..offset + count]);
            }
            if term_size != term.unpacked_length {
                return Err(invalid("Xet term decompressed length mismatch"));
            }
        }
        if output.len() as u64 != length || skip != 0 {
            return Err(invalid("Xet reconstruction length mismatch"));
        }
        Ok(output)
    }

    pub(crate) async fn read_range(
        &self,
        file: &XetFile,
        start: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        let end = start
            .checked_add(length)
            .filter(|end| *end <= file.size)
            .ok_or_else(|| invalid("Requested byte range exceeds Xet file size"))?;
        let capacity =
            usize::try_from(length).map_err(|_| invalid("Requested Xet range is too large"))?;
        let mut output = Vec::with_capacity(capacity);
        let mut windows = stream::iter((start..end).step_by(WINDOW_BYTES as usize))
            .map(|offset| self.read_window(file, offset, (end - offset).min(WINDOW_BYTES)))
            .buffered(self.concurrency);
        while let Some(window) = windows.try_next().await.map_err(native_error)? {
            output.extend_from_slice(&window);
        }
        Ok(output)
    }

    pub(crate) async fn download(&self, file: &XetFile, path: &Path) -> Result<u64> {
        let mut output = tokio::fs::File::create(path).await?;
        let mut digest = Sha256::new();
        let mut written = 0u64;
        let mut windows = stream::iter((0..file.size).step_by(WINDOW_BYTES as usize))
            .map(|offset| self.read_window(file, offset, (file.size - offset).min(WINDOW_BYTES)))
            .buffered(self.concurrency);
        while let Some(window) = windows.try_next().await.map_err(native_error)? {
            digest.update(&window);
            output.write_all(&window).await?;
            written += window.len() as u64;
        }
        if written != file.size {
            return Err(invalid("Downloaded Xet file size mismatch"));
        }
        if file
            .sha256
            .as_ref()
            .is_some_and(|expected| *expected != digest_to_hex(digest.finalize()))
        {
            return Err(invalid("Downloaded Xet file SHA-256 mismatch"));
        }
        output.flush().await?;
        output.sync_all().await?;
        Ok(written)
    }
}
