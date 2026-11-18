use fs2::FileExt;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::SWEEP_INTERVAL;

const SIDECARS: [&str; 3] = ["-journal", "-wal", "-shm"];

pub(super) struct Disk {
    directory: PathBuf,
}

impl Disk {
    pub(super) fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    pub(super) fn directory(&self) -> &Path {
        &self.directory
    }

    pub(super) fn entry_lock(&self, path: &Path) -> io::Result<File> {
        self.lock_file(lock_stripe(path))
    }

    pub(super) fn creation_lock(&self, path: &Path) -> io::Result<File> {
        // Serialize cleanup of old sidecars and creation without blocking other
        // SQLite connections in the stripe.
        let file = self.lock_file(&format!("create-{}", lock_stripe(path)))?;
        FileExt::lock_exclusive(&file)?;
        Ok(file)
    }

    fn lock_file(&self, name: &str) -> io::Result<File> {
        let directory = self.directory.join(".webshart-transform-locks");
        let path = directory.join(name);
        let mut options = OpenOptions::new();
        options.create(true).truncate(false).write(true).read(true);
        match options.open(&path) {
            Err(error) if error.kind() == ErrorKind::NotFound => {
                fs::create_dir_all(directory)?;
                options.open(path)
            }
            result => result,
        }
    }

    pub(super) fn touch(path: &Path) -> io::Result<()> {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.with_extension("access"))?
            .set_modified(SystemTime::now())
    }

    pub(super) fn clear_sidecars(path: &Path) -> io::Result<()> {
        for suffix in SIDECARS {
            remove_if_exists(&sidecar_path(path, suffix))?;
        }
        Ok(())
    }

    /// Remove idle databases until the directory is within its budget.
    /// Returns false when a periodic pass is deferred to another sweep.
    pub(super) fn trim(&self, limit: u64, wait: bool) -> io::Result<bool> {
        let sweep = self.lock_file("sweep")?;
        if wait {
            FileExt::lock_exclusive(&sweep)?;
        } else {
            if !try_exclusive(&sweep)? {
                return Ok(false);
            }
            if sweep
                .metadata()?
                .modified()?
                .elapsed()
                .is_ok_and(|elapsed| elapsed < SWEEP_INTERVAL)
            {
                return Ok(false);
            }
        }

        let mut entries = self.scan()?;
        let mut retained = entries
            .iter()
            .filter(|entry| entry.size <= limit)
            .map(|entry| entry.size)
            .sum::<u64>();
        if retained <= limit
            && entries
                .iter()
                .all(|entry| entry.has_db && entry.size <= limit)
        {
            sweep.set_modified(SystemTime::now())?;
            return Ok(true);
        }
        // Drop oversized entries first, so a single shard larger than the limit
        // cannot cause the eviction of every smaller shard.
        entries.sort_unstable_by(|a, b| {
            (a.size <= limit, a.last_used(), &a.path).cmp(&(
                b.size <= limit,
                b.last_used(),
                &b.path,
            ))
        });
        let mut retry = false;

        for entry in entries {
            if retained <= limit && entry.has_db && entry.size <= limit {
                continue;
            }

            let guard = self.entry_lock(&entry.path)?;
            if !try_exclusive(&guard)? {
                retry = true;
                continue;
            }
            let current = Entry::read(entry.path.clone())?;
            if entry.size <= limit {
                retained = retained.saturating_sub(entry.size);
            }
            if current.last_used() > entry.last_used()
                || (current.has_db
                    && current.size <= limit
                    && retained.saturating_add(current.size) <= limit)
            {
                if current.size <= limit {
                    retained = retained.saturating_add(current.size);
                }
                retry |= current.last_used() > entry.last_used();
                continue;
            }
            // Unlink the database before its sidecars. A crash during eviction
            // then leaves only orphan sidecars, which a new writer clears.
            remove_if_exists(&entry.path)?;
            Self::clear_sidecars(&entry.path)?;
            remove_if_exists(&entry.path.with_extension("access"))?;
        }
        sweep.set_modified(SystemTime::now())?;
        Ok(!retry)
    }

    fn scan(&self) -> io::Result<Vec<Entry>> {
        let mut result = Vec::new();
        for directory in fs::read_dir(&self.directory)? {
            let directory = directory?;
            let name = directory.file_name();
            let Some(prefix) = name.to_str().filter(|s| s.len() == 2 && hex(s)) else {
                continue;
            };
            if !directory.file_type()?.is_dir() {
                continue;
            }
            let path = directory.path();
            let mut entries = HashMap::new();
            for file in fs::read_dir(&path)? {
                let file = file?;
                let name = file.file_name();
                let Some((id, kind)) = name.to_str().and_then(cache_file) else {
                    continue;
                };
                if !id.starts_with(prefix) {
                    continue;
                }
                let Some(metadata) =
                    metadata_if_exists(&file.path())?.filter(fs::Metadata::is_file)
                else {
                    continue;
                };
                let entry = entries
                    .entry(id.to_owned())
                    .or_insert_with(|| Entry::new(path.join(format!("{id}.sqlite3"))));
                entry.add(&metadata, kind);
            }
            result.extend(entries.into_values());
        }
        Ok(result)
    }
}

struct Entry {
    path: PathBuf,
    size: u64,
    has_db: bool,
    modified: SystemTime,
    accessed: Option<SystemTime>,
}

impl Entry {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            size: 0,
            has_db: false,
            modified: SystemTime::UNIX_EPOCH,
            accessed: None,
        }
    }

    fn last_used(&self) -> SystemTime {
        self.accessed
            .map_or(self.modified, |accessed| accessed.max(self.modified))
    }

    fn add(&mut self, metadata: &fs::Metadata, suffix: &str) {
        let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if suffix == ".access" {
            self.accessed = Some(modified);
        } else {
            self.has_db |= suffix == ".sqlite3";
            self.size += metadata.len();
            self.modified = self.modified.max(modified);
        }
    }

    fn read(path: PathBuf) -> io::Result<Self> {
        let mut entry = Self::new(path);
        for (path, suffix) in std::iter::once((entry.path.clone(), ".sqlite3"))
            .chain(SIDECARS.map(|suffix| (sidecar_path(&entry.path, suffix), suffix)))
            .chain(std::iter::once((
                entry.path.with_extension("access"),
                ".access",
            )))
        {
            if let Some(metadata) = metadata_if_exists(&path)?.filter(fs::Metadata::is_file) {
                entry.add(&metadata, suffix);
            }
        }
        Ok(entry)
    }
}

fn cache_file(name: &str) -> Option<(&str, &str)> {
    let (id, suffix) = name.split_at_checked(64)?;
    (hex(id)
        && (matches!(suffix, ".sqlite3" | ".access")
            || suffix
                .strip_prefix(".sqlite3")
                .is_some_and(|sidecar| SIDECARS.contains(&sidecar))))
    .then_some((id, suffix))
}

fn lock_stripe(path: &Path) -> &str {
    // Use the first three digest digits to bound the number of lock files.
    let name = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("transform cache entry has a hexadecimal name");
    &name[..3]
}

fn hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    path.with_extension(format!("sqlite3{suffix}"))
}

fn metadata_if_exists(path: &Path) -> io::Result<Option<fs::Metadata>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn try_exclusive(file: &File) -> io::Result<bool> {
    match FileExt::try_lock_exclusive(file) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(false),
        Err(error) => Err(error),
    }
}
