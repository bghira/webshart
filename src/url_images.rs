//! On-demand HTTP image payloads and concurrent geometry probes.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use reqwest::{Client, Url};
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bounds_errors_and_payload() {
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
        let data = client
            .download(Some(&format!("{}/image", server.url())))
            .await
            .unwrap();
        assert_eq!(data, include_bytes!("../tests/fixtures/url-image.png"));
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
