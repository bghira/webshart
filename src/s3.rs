//! Authenticated object reads shared by Parquet readers and their shard cache.
use bytes::Bytes;
use futures::{StreamExt, TryStreamExt};
use object_store::{
    aws::{AmazonS3Builder, AwsCredential},
    path::Path as ObjectPath,
    Attributes, GetOptions, GetRange, ObjectMeta, ObjectStore, StaticCredentialProvider,
};
use pyo3::{
    exceptions::{PyFileNotFoundError, PyOSError, PyPermissionError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{io::AsyncWriteExt, sync::Semaphore};

#[derive(Default)]
struct TransportCounts {
    heads: AtomicU64,
    ranges: AtomicU64,
    range_bytes: AtomicU64,
    downloads: AtomicU64,
    download_bytes: AtomicU64,
    lists: AtomicU64,
}

pub(crate) struct Store {
    pub bucket: String,
    pub identity: String,
    store: Arc<dyn ObjectStore>,
    permits: Semaphore,
    counts: TransportCounts,
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
        Ok(self.head_with_attributes(key).await?.0)
    }

    pub async fn head_with_attributes(
        &self,
        key: &str,
    ) -> object_store::Result<(ObjectMeta, Attributes)> {
        let _permit = self
            .permits
            .acquire()
            .await
            .expect("S3 semaphore stays open");
        self.counts.heads.fetch_add(1, Ordering::Relaxed);
        let result = self
            .store
            .get_opts(&path(key)?, GetOptions::new().with_head(true))
            .await?;
        Ok((result.meta, result.attributes))
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
        self.counts.ranges.fetch_add(1, Ordering::Relaxed);
        let data = self
            .store
            .get_opts(&path(key)?, options)
            .await?
            .bytes()
            .await?;
        self.counts
            .range_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(data)
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
        self.counts.downloads.fetch_add(1, Ordering::Relaxed);
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
            self.counts
                .download_bytes
                .fetch_add(chunk.len() as u64, Ordering::Relaxed);
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
                    counts: TransportCounts::default(),
                }),
            })
        })
    }

    fn io_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let counts = &self.inner.counts;
        let result = serde_json::json!({
            "head_requests": counts.heads.load(Ordering::Relaxed),
            "range_requests": counts.ranges.load(Ordering::Relaxed),
            "range_bytes": counts.range_bytes.load(Ordering::Relaxed),
            "downloads": counts.downloads.load(Ordering::Relaxed),
            "download_bytes": counts.download_bytes.load(Ordering::Relaxed),
            "list_operations": counts.lists.load(Ordering::Relaxed)
        });
        Ok(pythonize::pythonize(py, &result)?.unbind())
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

    #[pyo3(signature=(key, destination, max_bytes=536870912))]
    fn download_manifest(
        &self,
        py: Python<'_>,
        key: String,
        destination: std::path::PathBuf,
        max_bytes: u64,
    ) -> PyResult<()> {
        py.detach(|| {
            crate::parquet::runtime()
                .map_err(|_| PyOSError::new_err("I/O runtime unavailable"))?
                .block_on(async {
                    let meta = self.inner.head(&key).await.map_err(python_error)?;
                    if meta.size > max_bytes {
                        return Err(PyValueError::new_err("Manifest exceeds max_bytes"));
                    }
                    self.inner
                        .download(&key, &meta, &destination)
                        .await
                        .map_err(PyErr::from)?;
                    Ok(())
                })
        })
    }

    #[pyo3(signature=(prefix=""))]
    fn list(&self, py: Python<'_>, prefix: &str) -> PyResult<Py<PyAny>> {
        let data = py.detach(|| {
            let prefix = path(prefix).map_err(python_error)?;
            self.inner.counts.lists.fetch_add(1, Ordering::Relaxed);
            crate::parquet::runtime().map_err(|_| PyOSError::new_err("I/O runtime unavailable"))?.block_on(async {
                self.inner.store.list(Some(&prefix)).map_ok(|m| serde_json::json!({"key":m.location.as_ref(),"size":m.size,"etag":m.e_tag})).try_collect::<Vec<_>>().await.map_err(python_error)
            })
        })?;
        Ok(pythonize::pythonize(py, &data)?.unbind())
    }
}
