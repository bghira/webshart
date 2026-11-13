//! On-demand HTTP image payloads and concurrent geometry probes.

use futures::{stream, StreamExt};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use reqwest::{Client, Url};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use std::time::Duration;
use tokio::runtime::Runtime;

fn runtime() -> PyResult<&'static Runtime> {
    static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .enable_all()
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| PyOSError::new_err(e.clone()))
}

#[derive(Debug)]
struct Failure {
    code: &'static str,
    status: Option<u16>,
    retry: bool,
}

fn failure(code: &'static str) -> Failure {
    Failure {
        code,
        status: None,
        retry: false,
    }
}

fn transport(error: reqwest::Error) -> Failure {
    Failure {
        code: if error.is_timeout() {
            "timeout"
        } else {
            "network_error"
        },
        status: None,
        retry: !error.is_redirect() && !error.is_builder(),
    }
}

fn valid_url(url: &Url) -> bool {
    matches!(url.scheme(), "http" | "https")
        && url.host_str().is_some()
        && url.username().is_empty()
        && url.password().is_none()
}

#[derive(Serialize)]
struct Probe {
    width: Option<usize>,
    height: Option<usize>,
    url_status: &'static str,
    url_http_status: Option<u16>,
    download_bytes: Option<usize>,
    download_sha256: Option<String>,
}

#[pyclass(name = "_URLImageClient")]
pub struct PyURLImageClient {
    client: Client,
    retries: usize,
    max_bytes: usize,
}

impl PyURLImageClient {
    fn create(timeout: f64, retries: usize, max_bytes: usize) -> Result<Self, &'static str> {
        if !timeout.is_finite() || timeout <= 0.0 || timeout > 86400.0 || max_bytes == 0 {
            return Err("timeout must be in (0, 86400] and max_bytes positive");
        }
        // This client is separate from Hub transport: no tokens or cookie store.
        let client = Client::builder()
            .user_agent("webshart-url-images/1")
            .timeout(Duration::from_secs_f64(timeout))
            .connect_timeout(Duration::from_secs_f64(timeout))
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() >= 5 || !valid_url(attempt.url()) {
                    attempt.error("invalid image redirect")
                } else {
                    attempt.follow()
                }
            }))
            .build()
            .map_err(|_| "Cannot create image HTTP client")?;
        Ok(Self {
            client,
            retries,
            max_bytes,
        })
    }

    async fn attempt(&self, url: &Url) -> Result<Vec<u8>, Failure> {
        let mut response = self
            .client
            .get(url.clone())
            .send()
            .await
            .map_err(transport)?;
        let status = response.status();
        if !status.is_success() {
            return Err(Failure {
                code: "http_error",
                status: Some(status.as_u16()),
                retry: status.is_server_error() || matches!(status.as_u16(), 408 | 429),
            });
        }
        let length = response.content_length();
        if length.is_some_and(|n| n > self.max_bytes as u64) {
            return Err(failure("too_large"));
        }
        let mut data = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(transport)? {
            if chunk.len() > self.max_bytes - data.len() {
                return Err(failure("too_large"));
            }
            data.extend_from_slice(&chunk);
        }
        if data.is_empty() {
            return Err(failure("empty_response"));
        }
        if length.is_some_and(|n| n != data.len() as u64) {
            return Err(failure("incomplete_response"));
        }
        Ok(data)
    }

    async fn download(&self, value: Option<&str>) -> Result<Vec<u8>, Failure> {
        let url = value
            .and_then(|s| Url::parse(s).ok())
            .filter(valid_url)
            .ok_or_else(|| failure("invalid_url"))?;
        for attempt in 0..=self.retries {
            match self.attempt(&url).await {
                Ok(data) => return Ok(data),
                Err(error) if error.retry && attempt < self.retries => {
                    tokio::time::sleep(Duration::from_millis(250 * (1 << attempt.min(3)))).await;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!()
    }

    async fn probe(&self, url: Option<&str>, max_pixels: usize) -> Probe {
        let mut result = Probe {
            width: None,
            height: None,
            url_status: "ok",
            url_http_status: None,
            download_bytes: None,
            download_sha256: None,
        };
        match self.download(url).await {
            Ok(data) => {
                result.download_bytes = Some(data.len());
                result.download_sha256 = Some(crate::digest_to_hex(Sha256::digest(&data)));
                match imagesize::blob_size(&data) {
                    Ok(size) if size.width == 0 || size.height == 0 => {
                        result.url_status = "invalid_image"
                    }
                    Ok(size)
                        if size
                            .width
                            .checked_mul(size.height)
                            .is_none_or(|n| n > max_pixels) =>
                    {
                        result.url_status = "too_many_pixels"
                    }
                    Ok(size) => {
                        result.width = Some(size.width);
                        result.height = Some(size.height);
                    }
                    Err(_) => result.url_status = "invalid_image",
                }
            }
            Err(error) => {
                result.url_status = error.code;
                result.url_http_status = error.status;
            }
        }
        result
    }
}

#[pymethods]
impl PyURLImageClient {
    #[new]
    #[pyo3(signature = (timeout=15.0, retries=1, max_bytes=33554432))]
    fn new(py: Python<'_>, timeout: f64, retries: usize, max_bytes: usize) -> PyResult<Self> {
        py.detach(|| Self::create(timeout, retries, max_bytes).map_err(PyValueError::new_err))
    }

    fn fetch(
        &self,
        py: Python<'_>,
        url: Option<String>,
    ) -> PyResult<(Option<Py<PyBytes>>, Option<String>, Option<u16>)> {
        let result =
            py.detach(|| Ok::<_, PyErr>(runtime()?.block_on(self.download(url.as_deref()))))?;
        Ok(match result {
            Ok(data) => (Some(PyBytes::new(py, &data).unbind()), None, None),
            Err(error) => (None, Some(error.code.to_string()), error.status),
        })
    }

    #[pyo3(signature = (urls, workers=32, max_pixels=64000000))]
    fn scan(
        &self,
        py: Python<'_>,
        urls: Vec<Option<String>>,
        workers: usize,
        max_pixels: usize,
    ) -> PyResult<Py<PyAny>> {
        if workers == 0 || max_pixels == 0 {
            return Err(PyValueError::new_err(
                "workers and max_pixels must be positive",
            ));
        }
        let result = py.detach(|| {
            runtime().map(|rt| {
                rt.block_on(async {
                    let mut results: Vec<_> =
                        stream::iter(urls.iter().enumerate().map(|(index, url)| async move {
                            (index, self.probe(url.as_deref(), max_pixels).await)
                        }))
                        .buffer_unordered(workers)
                        .collect()
                        .await;
                    results.sort_unstable_by_key(|(index, _)| *index);
                    results
                        .into_iter()
                        .map(|(_, probe)| probe)
                        .collect::<Vec<_>>()
                })
            })
        })?;
        Ok(pythonize::pythonize(py, &result)?.unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounds_errors_and_geometry() {
        let mut server = mockito::Server::new_async().await;
        let ok = server
            .mock("GET", "/image")
            .with_status(200)
            .with_body(include_bytes!("../tests/fixtures/url-image.png"))
            .expect(2)
            .create_async()
            .await;
        let missing = server
            .mock("GET", "/missing")
            .with_status(404)
            .expect(1)
            .create_async()
            .await;
        let client = PyURLImageClient::create(1.0, 2, 1024).unwrap();
        let probe = client
            .probe(Some(&format!("{}/image", server.url())), 1000)
            .await;
        assert_eq!((probe.width, probe.height), (Some(17), Some(9)));
        assert_eq!(probe.url_status, "ok");
        let error = client
            .download(Some(&format!("{}/missing", server.url())))
            .await
            .unwrap_err();
        assert_eq!(error.status, Some(404));
        assert_eq!(
            client.download(Some("file:secret")).await.unwrap_err().code,
            "invalid_url"
        );
        let small = PyURLImageClient::create(1.0, 0, 4).unwrap();
        assert_eq!(
            small
                .download(Some(&format!("{}/image", server.url())))
                .await
                .unwrap_err()
                .code,
            "too_large"
        );
        ok.assert_async().await;
        missing.assert_async().await;
    }
}
