use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

type Records = Vec<(i64, Vec<u8>)>;
type CacheResult<T> = Result<T, CacheError>;

#[derive(Debug, thiserror::Error)]
enum CacheError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

struct Location {
    directory: PathBuf,
    recipe: String,
}

impl Location {
    fn shard_path(&self, shard: &str) -> PathBuf {
        let mut digest = Sha256::new();
        digest.update(b"webshart-transform-cache-v1");
        digest.update((self.recipe.len() as u64).to_le_bytes());
        digest.update(self.recipe.as_bytes());
        digest.update(shard.as_bytes());
        let id = crate::digest_to_hex(digest.finalize());
        self.directory.join(&id[..2]).join(format!("{id}.sqlite3"))
    }

    fn read(&self, requests: Vec<(String, i64)>) -> CacheResult<Vec<Option<Vec<u8>>>> {
        let mut result = vec![None; requests.len()];
        let mut groups = BTreeMap::<_, Vec<_>>::new();
        for (position, (shard, row)) in requests.into_iter().enumerate() {
            groups.entry(shard).or_default().push((position, row));
        }
        for (shard, rows) in groups {
            let path = self.shard_path(&shard);
            if !path.try_exists()? {
                continue;
            }
            let mut db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
            db.busy_timeout(Duration::from_secs(5))?;
            let transaction = db.transaction()?;
            // The file can be visible before its first writer has installed the schema.
            let ready: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name = 'transforms')",
                [],
                |row| row.get(0),
            )?;
            if ready {
                let mut select =
                    transaction.prepare("SELECT value FROM transforms WHERE row = ?1")?;
                for (position, row) in rows {
                    result[position] = select.query_row([row], |row| row.get(0)).optional()?;
                }
            }
        }
        Ok(result)
    }

    fn open_writer(&self, shard: &str) -> CacheResult<Connection> {
        let path = self.shard_path(shard);
        std::fs::create_dir_all(path.parent().expect("cache shard has a parent"))?;
        let db = Connection::open(path)?;
        db.busy_timeout(Duration::from_secs(5))?;
        db.execute(
            "CREATE TABLE IF NOT EXISTS transforms (row INTEGER PRIMARY KEY, value BLOB NOT NULL)",
            [],
        )?;
        Ok(db)
    }
}

enum Write {
    Batch(String, Records),
    Flush(mpsc::Sender<()>),
}

fn write_batches(location: &Location, receiver: &mpsc::Receiver<Write>) -> CacheResult<()> {
    let mut writer = None::<(String, Connection)>;
    for command in receiver {
        match command {
            Write::Batch(shard, records) => {
                if writer.as_ref().is_none_or(|(current, _)| current != &shard) {
                    let db = location.open_writer(&shard)?;
                    writer = Some((shard, db));
                }
                let (_, db) = writer.as_mut().expect("cache writer is open");
                let transaction = db.transaction()?;
                {
                    let mut insert = transaction.prepare_cached(
                        "INSERT INTO transforms VALUES (?1, ?2) ON CONFLICT(row) DO NOTHING",
                    )?;
                    for (row, value) in records {
                        insert.execute(params![row, value])?;
                    }
                }
                transaction.commit()?;
            }
            Write::Flush(done) => {
                let _ = done.send(());
            }
        }
    }
    Ok(())
}

struct Writer {
    sender: Option<mpsc::SyncSender<Write>>,
    thread: Option<JoinHandle<()>>,
}

/// A local cache of model-specific transform values, partitioned by pinned shard.
#[pyclass(module = "webshart._webshart")]
pub struct TransformCache {
    location: Arc<Location>,
    writer: Mutex<Writer>,
    failure: Arc<Mutex<Option<String>>>,
    closed: AtomicBool,
    max_batch_bytes: usize,
}

impl TransformCache {
    fn write_error(&self) -> PyErr {
        let error = self.failure.lock().expect("cache error mutex");
        match error.as_ref() {
            Some(error) => PyOSError::new_err(error.clone()),
            None => PyOSError::new_err("transform cache writer stopped"),
        }
    }

    fn send(&self, command: Write) -> PyResult<()> {
        let writer = self.writer.lock().expect("cache writer mutex");
        writer
            .sender
            .as_ref()
            .ok_or_else(|| PyOSError::new_err("transform cache is closed"))?
            .send(command)
            .map_err(|_| self.write_error())
    }

    fn finish(&self) -> PyResult<()> {
        let mut writer = self.writer.lock().expect("cache writer mutex");
        self.closed.store(true, Ordering::Release);
        writer.sender.take();
        if let Some(thread) = writer.thread.take() {
            thread.join().map_err(|_| {
                let mut error = self.failure.lock().expect("cache error mutex");
                *error = Some("transform cache writer panicked".to_owned());
                PyOSError::new_err("transform cache writer panicked")
            })?;
        }
        let failure = self.failure.lock().expect("cache error mutex");
        match failure.as_ref() {
            Some(error) => Err(PyOSError::new_err(error.clone())),
            None => Ok(()),
        }
    }
}

#[pymethods]
impl TransformCache {
    /// Open a cache for the given transform recipe fingerprint.
    #[new]
    #[pyo3(signature = (directory, recipe, *, queue_batches=16, max_batch_bytes=1048576))]
    fn new(
        py: Python<'_>,
        directory: PathBuf,
        recipe: String,
        queue_batches: usize,
        max_batch_bytes: usize,
    ) -> PyResult<Self> {
        if recipe.is_empty() || queue_batches == 0 || max_batch_bytes == 0 {
            return Err(PyValueError::new_err(
                "recipe must be nonempty and queue limits must be positive",
            ));
        }
        py.detach(move || {
            std::fs::create_dir_all(&directory).map_err(PyOSError::new_err)?;
            let location = Arc::new(Location { directory, recipe });
            let failure = Arc::new(Mutex::new(None));
            let (sender, receiver) = mpsc::sync_channel(queue_batches);
            let writer_location = Arc::clone(&location);
            let writer_failure = Arc::clone(&failure);
            let thread = thread::Builder::new()
                .name("webshart-transform-cache".to_owned())
                .spawn(move || {
                    if let Err(error) = write_batches(&writer_location, &receiver) {
                        *writer_failure.lock().expect("cache error mutex") =
                            Some(format!("transform cache write failed: {error}"));
                    }
                })
                .map_err(PyOSError::new_err)?;
            Ok(Self {
                location,
                writer: Mutex::new(Writer {
                    sender: Some(sender),
                    thread: Some(thread),
                }),
                failure,
                closed: AtomicBool::new(false),
                max_batch_bytes,
            })
        })
    }

    /// Read (shard identity, physical row) pairs in input order, with None for misses.
    fn read_batch(
        &self,
        py: Python<'_>,
        requests: Vec<(String, i64)>,
    ) -> PyResult<Vec<Option<Py<PyBytes>>>> {
        if requests
            .iter()
            .any(|(shard, row)| shard.is_empty() || *row < 0)
        {
            return Err(PyValueError::new_err(
                "shard identities must be nonempty and row numbers must be nonnegative",
            ));
        }
        let output = py.detach(|| {
            if self.closed.load(Ordering::Acquire) {
                return Err(PyOSError::new_err("transform cache is closed"));
            }
            self.location.read(requests).map_err(|error| {
                PyOSError::new_err(format!("transform cache read failed: {error}"))
            })
        })?;
        Ok(output
            .into_iter()
            .map(|value| value.map(|value| PyBytes::new(py, &value).unbind()))
            .collect())
    }

    /// Queue one shard's (physical row, bytes) records, waiting for queue space.
    fn enqueue(&self, py: Python<'_>, shard: String, records: Records) -> PyResult<()> {
        if shard.is_empty() || records.iter().any(|(row, _)| *row < 0) {
            return Err(PyValueError::new_err(
                "shard identity must be nonempty and row numbers must be nonnegative",
            ));
        }
        let bytes = shard.len()
            + records.len() * size_of::<(i64, Vec<u8>)>()
            + records.iter().map(|(_, value)| value.len()).sum::<usize>();
        if bytes > self.max_batch_bytes {
            return Err(PyValueError::new_err(
                "transform batch exceeds max_batch_bytes",
            ));
        }
        py.detach(|| self.send(Write::Batch(shard, records)))
    }

    /// Wait for all previously accepted batches to commit.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let (done, wait) = mpsc::channel();
            self.send(Write::Flush(done))?;
            wait.recv().map_err(|_| self.write_error())
        })
    }

    /// Commit pending batches and stop the writer.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.finish())
    }
}

impl Drop for TransformCache {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}
