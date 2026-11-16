//! Authenticated object reads shared by Parquet readers and their shard cache.
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use object_store::{
    aws::{AmazonS3Builder, AwsCredential},
    path::Path as ObjectPath,
    GetOptions, GetRange, ObjectMeta, ObjectStore, ObjectStoreExt, StaticCredentialProvider,
};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyOSError, PyPermissionError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use sha2::{Digest, Sha256};
use std::{path::Path, sync::Arc};
use tokio::{io::AsyncWriteExt, sync::Semaphore};

pub(crate) struct Store {
    pub bucket: String,
    pub identity: String,
    store: Arc<dyn ObjectStore>,
    permits: Semaphore,
}

pub(crate) fn python_error(error: object_store::Error) -> PyErr {
    match error {
        object_store::Error::NotFound { .. } => PyFileNotFoundError::new_err("S3 object not found"),
        object_store::Error::PermissionDenied { .. }
        | object_store::Error::Unauthenticated { .. } => {
            PyPermissionError::new_err("S3 authentication or access denied")
        }
        object_store::Error::Precondition { .. } => {
            PyOSError::new_err("S3 object changed during reading")
        }
        _ => PyOSError::new_err("S3 request failed; check endpoint, credentials, and network"),
    }
}

fn path(key: &str) -> object_store::Result<ObjectPath> {
    Ok(ObjectPath::parse(key)?)
}

impl Store {
    pub async fn head(&self, key: &str) -> object_store::Result<ObjectMeta> {
        let _permit = self
            .permits
            .acquire()
            .await
            .expect("S3 semaphore stays open");
        self.store.head(&path(key)?).await
    }

    pub async fn range(
        &self,
        key: &str,
        start: u64,
        length: usize,
        etag: Option<String>,
    ) -> object_store::Result<Bytes> {
        let _permit = self
            .permits
            .acquire()
            .await
            .expect("S3 semaphore stays open");
        let options = GetOptions {
            range: Some(GetRange::Bounded(start..start + length as u64)),
            if_match: etag,
            ..Default::default()
        };
        self.store
            .get_opts(&path(key)?, options)
            .await?
            .bytes()
            .await
    }

    pub async fn read(&self, key: &str, limit: usize) -> PyResult<Vec<u8>> {
        let meta = self.head(key).await.map_err(python_error)?;
        if meta.size > limit as u64 {
            return Err(PyValueError::new_err("S3 object exceeds max_bytes"));
        }
        let data = self
            .range(key, 0, meta.size as usize, meta.e_tag)
            .await
            .map_err(python_error)?;
        if data.len() as u64 != meta.size {
            return Err(PyOSError::new_err("S3 object length mismatch"));
        }
        Ok(data.to_vec())
    }

    pub async fn download(
        &self,
        key: &str,
        meta: &ObjectMeta,
        target: &Path,
    ) -> crate::Result<u64> {
        let _permit = self
            .permits
            .acquire()
            .await
            .expect("S3 semaphore stays open");
        let error = || crate::WebshartError::Io(std::io::Error::other("S3 shard download failed"));
        let options = GetOptions {
            if_match: meta.e_tag.clone(),
            ..Default::default()
        };
        let mut stream = self
            .store
            .get_opts(&path(key).map_err(|_| error())?, options)
            .await
            .map_err(|_| error())?
            .into_stream();
        let mut file = tokio::fs::File::create(target).await?;
        let mut written = 0u64;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|_| error())?;
            written += chunk.len() as u64;
            if written > meta.size {
                return Err(error());
            }
            file.write_all(&chunk).await?;
        }
        if written != meta.size {
            return Err(error());
        }
        file.flush().await?;
        file.sync_all().await?;
        Ok(written)
    }
}

#[pyclass(name = "_S3Store")]
pub struct PyS3Store {
    pub(crate) inner: Arc<Store>,
}

#[pymethods]
impl PyS3Store {
    #[new]
    #[pyo3(signature=(bucket, *, region_name=None, endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None, allow_http=false, max_connections=32))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        bucket: String,
        region_name: Option<String>,
        endpoint_url: Option<String>,
        aws_access_key_id: Option<String>,
        aws_secret_access_key: Option<String>,
        aws_session_token: Option<String>,
        allow_http: bool,
        max_connections: usize,
    ) -> PyResult<Self> {
        if max_connections == 0 {
            return Err(PyValueError::new_err("max_connections must be positive"));
        }
        if aws_access_key_id.is_some() != aws_secret_access_key.is_some() {
            return Err(PyValueError::new_err(
                "Provide both AWS access key and secret key",
            ));
        }
        if bucket.is_empty() || bucket.contains('/') || bucket.contains('@') {
            return Err(PyValueError::new_err("Invalid S3 bucket name"));
        }
        py.detach(move || {
            let mut builder = AmazonS3Builder::from_env()
                .with_bucket_name(&bucket)
                .with_allow_http(allow_http)
                .with_virtual_hosted_style_request(false);
            if let Some(value) = region_name {
                builder = builder.with_region(value);
            }
            if let Some(value) = endpoint_url.clone() {
                builder = builder.with_endpoint(value);
            }
            if let (Some(key_id), Some(secret_key)) = (aws_access_key_id, aws_secret_access_key) {
                builder = builder.with_credentials(Arc::new(StaticCredentialProvider::new(
                    AwsCredential {
                        key_id,
                        secret_key,
                        token: aws_session_token,
                    },
                )));
            } else if let Some(token) = aws_session_token {
                builder = builder.with_token(token);
            }
            if builder
                .get_config_value(&object_store::aws::AmazonS3ConfigKey::Region)
                .is_none()
            {
                builder = builder.with_region("us-east-1");
            }
            let identity = format!(
                "{:?}:{:?}:{bucket}",
                builder.get_config_value(&object_store::aws::AmazonS3ConfigKey::Endpoint),
                builder.get_config_value(&object_store::aws::AmazonS3ConfigKey::Region)
            );
            let store = builder
                .build()
                .map_err(|_| PyValueError::new_err("Invalid S3 configuration"))?;
            Ok(Self {
                inner: Arc::new(Store {
                    bucket,
                    identity,
                    store: Arc::new(store),
                    permits: Semaphore::new(max_connections),
                }),
            })
        })
    }

    fn head(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let meta = py.detach(|| {
            crate::parquet::runtime()
                .map_err(|_| PyOSError::new_err("I/O runtime unavailable"))?
                .block_on(self.inner.head(&key))
                .map_err(python_error)
        })?;
        let value = serde_json::json!({"key":meta.location.as_ref(),"size":meta.size,"etag":meta.e_tag,"version":meta.version});
        Ok(pythonize::pythonize(py, &value)?.unbind())
    }

    #[pyo3(signature=(key, max_bytes=16777216))]
    fn read(&self, py: Python<'_>, key: String, max_bytes: usize) -> PyResult<Py<PyBytes>> {
        let data = py.detach(|| {
            crate::parquet::runtime()
                .map_err(|_| PyOSError::new_err("I/O runtime unavailable"))?
                .block_on(self.inner.read(&key, max_bytes))
        })?;
        Ok(PyBytes::new(py, &data).unbind())
    }

    #[pyo3(signature=(key, expected_sha256=None, max_bytes=536870912))]
    fn read_jsonl(
        &self,
        py: Python<'_>,
        key: String,
        expected_sha256: Option<String>,
        max_bytes: usize,
    ) -> PyResult<Py<PyAny>> {
        let records = py.detach(|| {
            crate::parquet::runtime()
                .map_err(|_| PyOSError::new_err("I/O runtime unavailable"))?
                .block_on(async {
                    let meta = self.inner.head(&key).await.map_err(python_error)?;
                    if meta.size > max_bytes as u64 {
                        return Err(PyValueError::new_err("Manifest exceeds max_bytes"));
                    }
                    let options = GetOptions {
                        if_match: meta.e_tag,
                        ..Default::default()
                    };
                    let _permit = self
                        .inner
                        .permits
                        .acquire()
                        .await
                        .expect("S3 semaphore stays open");
                    let mut stream = self
                        .inner
                        .store
                        .get_opts(&path(&key).map_err(python_error)?, options)
                        .await
                        .map_err(python_error)?
                        .into_stream();
                    let mut buffer = Vec::new();
                    let mut records = Vec::new();
                    let mut size = 0;
                    let mut hash = Sha256::new();
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk.map_err(python_error)?;
                        size += chunk.len();
                        if size > max_bytes {
                            return Err(PyValueError::new_err("Manifest exceeds max_bytes"));
                        }
                        hash.update(&chunk);
                        buffer.extend_from_slice(&chunk);
                        let mut start = 0;
                        for end in 0..buffer.len() {
                            if buffer[end] == b'\n' {
                                if end > start {
                                    records.push(
                                        serde_json::from_slice::<serde_json::Value>(
                                            &buffer[start..end],
                                        )
                                        .map_err(|_| {
                                            PyValueError::new_err("Invalid JSONL manifest")
                                        })?,
                                    );
                                }
                                start = end + 1;
                            }
                        }
                        buffer.drain(..start);
                        if buffer.len() > 1024 * 1024 {
                            return Err(PyValueError::new_err("Manifest line exceeds 1 MiB"));
                        }
                    }
                    if size as u64 != meta.size {
                        return Err(PyOSError::new_err("Manifest length mismatch"));
                    }
                    if !buffer.is_empty() {
                        records.push(
                            serde_json::from_slice(&buffer)
                                .map_err(|_| PyValueError::new_err("Invalid JSONL manifest"))?,
                        );
                    }
                    if expected_sha256
                        .is_some_and(|expected| expected != crate::digest_to_hex(hash.finalize()))
                    {
                        return Err(PyValueError::new_err("Manifest SHA-256 mismatch"));
                    }
                    Ok(records)
                })
        })?;
        Ok(pythonize::pythonize(py, &records)?.unbind())
    }

    #[pyo3(signature=(prefix=""))]
    fn list(&self, py: Python<'_>, prefix: &str) -> PyResult<Py<PyAny>> {
        let data = py.detach(|| {
            let prefix = path(prefix).map_err(python_error)?;
            crate::parquet::runtime().map_err(|_| PyOSError::new_err("I/O runtime unavailable"))?.block_on(async {
                self.inner.store.list(Some(&prefix)).map_ok(|m| serde_json::json!({"key":m.location.as_ref(),"size":m.size,"etag":m.e_tag})).try_collect::<Vec<_>>().await.map_err(python_error)
            })
        })?;
        Ok(pythonize::pythonize(py, &data)?.unbind())
    }
}
