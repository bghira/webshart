"""Page-streamed Parquet datasets with separately hosted, row-keyed metadata."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from itertools import accumulate
import hashlib
import json
import sqlite3
import os
import pickle
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory
from threading import RLock
from typing import Any, Iterator
from weakref import finalize
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from filelock import FileLock

from ._webshart import _ParquetFile, _ParquetCache, _S3Store, _parquet_bucket_keys
from .url_images import URLImageError, URLImageOptions, URLImageClient

MANIFEST = "webshart-parquet.json"
PROJECTED_SCAN_CACHE_BYTES = 8 * 1024 * 1024


def _synchronized(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Manifest paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value or str(path) != value:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


class _SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        redirected = super().redirect_request(
            request, response, code, message, headers, new_url
        )
        if (
            redirected is not None
            and urlparse(request.full_url).netloc != urlparse(new_url).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


class _Repository:
    def __init__(
        self,
        source: str,
        token: str | None,
        revision: str | None = None,
        *,
        endpoint: str | None = None,
        repo_type: str = "dataset",
        s3_options: dict | None = None,
    ):
        self.source = (
            str(source).removeprefix("https://huggingface.co/datasets/").rstrip("/")
        )
        self.token = token
        self.endpoint = endpoint
        self.repo_type = repo_type
        self.local = Path(source).is_dir()
        self.revision = revision
        self.files: dict[str, int] = {}
        self.s3 = None
        self.prefix = ""
        if self.source.startswith("s3://"):
            uri = urlparse(self.source)
            if not uri.netloc or uri.query or uri.fragment or "@" in uri.netloc:
                raise ValueError("S3 source must be s3://bucket or s3://bucket/prefix")
            self.s3 = _S3Store(uri.netloc, **(s3_options or {}))
            self.bucket = uri.netloc
            self.prefix = (
                _relative_path(uri.path.strip("/")) + "/" if uri.path.strip("/") else ""
            )
        elif self.local:
            self.root = Path(source).resolve()
        else:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise ImportError(
                    "Hub Parquet discovery requires webshart[hub]"
                ) from error
            api = HfApi(token=token, **({"endpoint": endpoint} if endpoint else {}))
            if repo_type == "dataset":
                info = api.dataset_info(
                    self.source, revision=revision, files_metadata=True
                )
            else:
                info = api.repo_info(
                    self.source,
                    repo_type=repo_type,
                    revision=revision,
                    files_metadata=True,
                )
            self.revision = info.sha
            self.files = {entry.rfilename: entry.size for entry in info.siblings}

    def exists(self, path: str) -> bool:
        if self.s3 is not None:
            if path in self.files:
                return True
            try:
                self.files[path] = self.s3.head(self.prefix + _relative_path(path))[
                    "size"
                ]
                return True
            except FileNotFoundError:
                return False
        return Path(self.location(path)).is_file() if self.local else path in self.files

    def location(self, path: str) -> str:
        path = _relative_path(path)
        if self.s3 is not None:
            return f"s3://{self.bucket}/{self.prefix}{path}"
        if self.local:
            target = (self.root / path).resolve()
            if not target.is_relative_to(self.root):
                raise ValueError("Metadata path escapes its root")
            return str(target)
        from huggingface_hub import hf_hub_url

        return hf_hub_url(
            self.source,
            path,
            repo_type=self.repo_type,
            revision=self.revision,
            endpoint=self.endpoint,
        )

    def size(self, path: str) -> int:
        if self.s3 is not None and path not in self.files:
            self.files[path] = self.s3.head(self.prefix + _relative_path(path))["size"]
        return (
            Path(self.location(path)).stat().st_size if self.local else self.files[path]
        )

    def _json_bytes(self, path: str, expected_hash: str | None = None) -> bytes:
        if self.size(path) > 16 * 1024 * 1024:
            raise ValueError(
                "JSON metadata exceeds 16 MiB; use Parquet metadata instead"
            )
        if self.s3 is not None:
            content = self.s3.read(self.prefix + _relative_path(path))
        elif self.local:
            content = Path(self.location(path)).read_bytes()
        else:
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            with build_opener(_SafeRedirect()).open(
                Request(self.location(path), headers=headers), timeout=60
            ) as response:
                content = response.read(16 * 1024 * 1024 + 1)
            if len(content) > 16 * 1024 * 1024:
                raise ValueError("JSON metadata exceeds 16 MiB")
        if expected_hash and hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Metadata SHA-256 mismatch")
        return content

    def read_json(self, path: str, expected_hash: str | None = None) -> Any:
        return json.loads(self._json_bytes(path, expected_hash))

    def iter_json_rows(self, path, expected_hash):
        text = self._json_bytes(path, expected_hash).decode("utf-8-sig")
        whitespace = re.compile(r"[ \t\n\r]*")
        pos = whitespace.match(text).end()
        if text[pos : pos + 1] != "[":
            raise ValueError("JSON metadata must be a list of row objects")
        decoder = json.JSONDecoder()
        pos = whitespace.match(text, pos + 1).end()
        if text[pos : pos + 1] != "]":
            while True:
                row, pos = decoder.raw_decode(text, pos)
                yield row
                pos = whitespace.match(text, pos).end()
                if text[pos : pos + 1] == "]":
                    break
                if text[pos : pos + 1] != ",":
                    raise ValueError("Invalid JSON metadata array separator")
                pos = whitespace.match(text, pos + 1).end()
        if whitespace.match(text, pos + 1).end() != len(text):
            raise ValueError("Unexpected data after JSON metadata array")

    def capture_manifest_path(self, manifest: str):
        descriptor = self.read_json(manifest)
        if descriptor.get("format") != "parquet-image-capture" or descriptor.get(
            "version"
        ) not in (1, 2):
            raise ValueError("Unsupported image capture manifest")
        path = _relative_path(descriptor["files_manifest"])
        self.capture_digest = descriptor.get("manifest_sha256")
        if self.capture_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.capture_digest
        ):
            raise ValueError("Invalid manifest SHA-256")
        return path

    def capture_manifest_version(self, path):
        if self.s3 is not None:
            head = self.s3.head(self.prefix + path)
            etag = head.get("etag")
            if etag is None:
                raise ValueError("An S3 ETag is required to cache a capture manifest.")
            return (etag, head["size"])
        stat = Path(self.location(path)).stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def iter_capture_records(self, path):
        with TemporaryDirectory() as directory:
            if self.s3 is not None:
                local = Path(directory) / "manifest.jsonl"
                self.s3.download_manifest(self.prefix + path, local)
            elif self.local:
                local = Path(self.location(path))
            else:
                raise ValueError(
                    "Capture manifests currently require a local directory or S3 source"
                )
            digest = hashlib.sha256()
            with local.open("rb") as stream:
                while line := stream.readline(1024 * 1024 + 1):
                    if len(line) > 1024 * 1024:
                        raise ValueError("Manifest line exceeds 1 MiB")
                    digest.update(line)
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    checksum = entry.get("sha256")
                    if checksum is not None and (
                        not isinstance(checksum, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", checksum)
                    ):
                        raise ValueError("Invalid capture shard SHA-256")
                    yield {
                        **({"sha256": checksum} if checksum is not None else {}),
                        "path": _relative_path(entry["key"]),
                        "size": _integer(entry["size_bytes"], "size_bytes", 1),
                        "num_rows": _integer(entry["num_rows"], "num_rows"),
                    }
            if self.capture_digest and digest.hexdigest() != self.capture_digest:
                raise ValueError("Manifest SHA-256 mismatch")

    def parquet_paths(self) -> list[str]:
        if self.local:
            return [
                path.relative_to(self.root).as_posix()
                for path in sorted(self.root.rglob("*.parquet"))
            ]
        if self.s3 is not None:
            self.files = {
                entry["key"][len(self.prefix) :]: entry["size"]
                for entry in self.s3.list(self.prefix)
                if entry["key"].endswith(".parquet")
            }
        return sorted(path for path in self.files if path.endswith(".parquet"))


class _CaptureCatalogue:
    """Immutable manifest rows with a private index of ETags observed by this reader."""

    def __init__(self, path=None, *, readonly=False):
        location = (
            path.resolve().as_uri() + "?mode=ro&immutable=1"
            if readonly
            else str(path or "")
        )
        self.connection = sqlite3.connect(
            location, uri=readonly, check_same_thread=False
        )
        finalize(self, self.connection.close)
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute(
            "CREATE TEMP TABLE observed_etags(position INTEGER PRIMARY KEY, etag TEXT NOT NULL)"
        )
        if readonly:
            self.count, self.rows, self.fingerprint = self.connection.execute(
                "SELECT shards, rows, fingerprint FROM catalogue"
            ).fetchone()
        else:
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute(
                "CREATE TABLE shards (path TEXT PRIMARY KEY, position INTEGER UNIQUE, size INTEGER, num_rows INTEGER, sha256 TEXT)"
            )
            self.count = self.rows = 0

    def finish(self, fingerprint):
        self.connection.execute(
            "CREATE TABLE catalogue(shards INTEGER, rows INTEGER, fingerprint TEXT)"
        )
        self.connection.execute(
            "INSERT INTO catalogue VALUES (?, ?, ?)",
            (self.count, self.rows, fingerprint),
        )
        self.connection.commit()
        self.fingerprint = fingerprint

    def add(self, record, included):
        position = self.count if included else None
        try:
            self.connection.execute(
                "INSERT INTO shards(path, position, size, num_rows, sha256) VALUES (?, ?, ?, ?, ?)",
                (
                    record["path"],
                    position,
                    record["size"],
                    record["num_rows"],
                    record.get("sha256"),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate source shard: {record['path']}") from exc
        if included:
            self.count += 1
            self.rows += record["num_rows"]

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        path, size, rows, checksum, etag = self.connection.execute(
            "SELECT path, size, num_rows, sha256, etag FROM shards LEFT JOIN observed_etags USING(position) WHERE position = ?",
            (index,),
        ).fetchone()
        shard = {"path": path, "size": size, "num_rows": rows}
        if checksum is not None:
            shard["sha256"] = checksum
        if etag is not None:
            shard["etag"] = etag
        return shard

    def set_etag(self, index, etag):
        self.connection.execute(
            "INSERT INTO observed_etags VALUES (?, ?)", (index, etag)
        )

    def position(self, path):
        row = self.connection.execute(
            "SELECT position FROM shards WHERE path = ?", (path,)
        ).fetchone()
        return row[0] if row is not None else None


class ParquetDataset:
    """Discover local/Hub Parquet shards, optionally paired with a metadata manifest.

    Custom metadata is keyed by physical ``source_row``, never by a potentially
    duplicated dataset ID. Manifest source revisions are pinned before discovery.
    Image bytes and external image URLs are not read during metadata discovery.
    """

    supports_parallel_metadata_reads = True
    supports_catalogue_cache = True

    def __init__(
        self,
        source: str,
        *,
        metadata: str | None = None,
        manifest: str | None = None,
        s3_options: dict | None = None,
        endpoint: str | None = None,
        repo_type: str = "dataset",
        hf_token: str | None = None,
        revision: str | None = None,
        metadata_revision: str | None = None,
        subfolder: str | None = None,
        image_column: str | None = None,
        image_url_column: str | None = None,
        caption_column: str | None = None,
        width_column: str | None = None,
        height_column: str | None = None,
        json_columns: list[str] | None = None,
        metadata_cache_size: int = 1,
        url_metadata_cache_bytes: int = 128 * 1024 * 1024,
        max_page_bytes: int = 512 * 1024 * 1024,
        catalogue_cache_dir: str | Path | None = None,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if self.hf_token is None:
            try:
                from huggingface_hub import get_token
            except ImportError:
                pass
            else:
                self.hf_token = get_token()
        self._lock = RLock()
        self.metadata_cache_size = _integer(
            metadata_cache_size, "metadata_cache_size", 1
        )
        self.max_page_bytes = _integer(max_page_bytes, "max_page_bytes", 1)
        self.parallel_reads = _integer(
            (s3_options or {}).get("max_connections", 4), "max_connections", 1
        )
        self.url_metadata_cache_bytes = _integer(
            url_metadata_cache_bytes, "url_metadata_cache_bytes"
        )
        self._metadata_cache: OrderedDict[int, dict[int, dict]] = OrderedDict()
        self._files: OrderedDict[int, _ParquetFile] = OrderedDict()
        self._payload_files: OrderedDict[int, _ParquetFile] = OrderedDict()
        self._metadata_repo = None
        self._shard_cache = None
        self._expected_source_revision = None
        self._local_digests = OrderedDict()
        self._overlays = OrderedDict()
        self._metadata_cache_dir = None
        capture_manifest = manifest
        manifest: dict = {}
        if metadata is not None:
            self._metadata_repo = _Repository(
                metadata, self.hf_token, metadata_revision, s3_options=s3_options
            )
            manifest = self._metadata_repo.read_json(MANIFEST)
            if (
                manifest.get("format") != "webshart-parquet"
                or manifest.get("version") != 1
            ):
                raise ValueError("Unsupported webshart Parquet metadata manifest")
            pinned = manifest.get("source", {})
            if str(source).startswith("s3://"):
                if pinned.get("uri") != str(source).rstrip("/"):
                    raise ValueError(
                        "Metadata manifest belongs to a different S3 source"
                    )
            elif not Path(source).is_dir():
                normalized = (
                    str(source)
                    .removeprefix("https://huggingface.co/datasets/")
                    .rstrip("/")
                )
                if pinned.get("repo_id") != normalized:
                    raise ValueError(
                        "Metadata manifest belongs to a different source repository"
                    )
                if not re.fullmatch(r"[0-9a-f]{40}", pinned.get("revision", "")):
                    raise ValueError("Remote metadata must pin a source commit SHA")
                if revision is not None and revision != pinned["revision"]:
                    raise ValueError("Source revision conflicts with metadata manifest")
                revision = pinned["revision"]
        self._source_repo = _Repository(
            source,
            self.hf_token,
            revision,
            endpoint=endpoint,
            repo_type=repo_type,
            s3_options=s3_options,
        )
        self.name = str(source)
        self.source_revision = self._source_repo.revision
        self.metadata_revision = (
            self._metadata_repo.revision if self._metadata_repo else None
        )
        columns = manifest.get("columns", {})
        self.image_column = image_column or columns.get("image", "image")
        self.image_url_column = image_url_column or columns.get("image_url")
        if self.image_url_column is not None and (
            not isinstance(self.image_url_column, str) or not self.image_url_column
        ):
            raise ValueError("image_url_column must be a nonempty column name")
        self.caption_column = caption_column or columns.get("captions", "captions")
        self.width_column = width_column or columns.get("width", "width")
        self.height_column = height_column or columns.get("height", "height")
        self.json_columns = list(
            json_columns
            if json_columns is not None
            else manifest.get("json_columns", [])
        )
        self.metadata_mode = manifest.get("metadata_mode", "overlay")
        if self.metadata_mode not in ("overlay", "replace"):
            raise ValueError("metadata_mode must be 'overlay' or 'replace'")
        prefix = _relative_path(subfolder.rstrip("/")) + "/" if subfolder else ""
        capture_path = None
        if capture_manifest is not None:
            if metadata is not None:
                raise ValueError("Select either a capture manifest or custom metadata")
            capture_path = self._source_repo.capture_manifest_path(
                _relative_path(capture_manifest)
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "source": self._source_repo.source,
                    "revision": self.source_revision,
                    "endpoint": (s3_options or {}).get("endpoint_url")
                    or os.environ.get("AWS_ENDPOINT_URL_S3")
                    or os.environ.get("AWS_ENDPOINT_URL")
                    or os.environ.get("AWS_ENDPOINT"),
                    "capture": getattr(self._source_repo, "capture_digest", None),
                    "metadata": metadata,
                    "metadata_revision": self.metadata_revision,
                    "columns": [
                        self.image_column,
                        self.image_url_column,
                        self.caption_column,
                        self.width_column,
                        self.height_column,
                        self.json_columns,
                    ],
                },
                sort_keys=True,
            ).encode()
        )
        if capture_path is not None:
            self._shards = self._open_capture_catalogue(
                capture_path, prefix, fingerprint, catalogue_cache_dir
            )
            self.fingerprint = self._shards.fingerprint
        else:
            if metadata is not None:
                records = manifest.get("shards")
                if not isinstance(records, list):
                    raise ValueError("Manifest shards must be a list")
            else:
                records = ({"path": path} for path in self._source_repo.parquet_paths())
            self._shards = []
            self._scan_shards(records, prefix, fingerprint, self._shards)
            self.fingerprint = fingerprint.hexdigest()
        if not self._shards:
            raise ValueError("No Parquet shards found")

    def _open_capture_catalogue(self, path, prefix, fingerprint, cache_dir):
        repository = self._source_repo

        def build(catalogue):
            records = repository.iter_capture_records(path)
            self._scan_shards(records, prefix, fingerprint, catalogue)
            if not catalogue:
                raise ValueError("No Parquet shards found")
            catalogue.finish(fingerprint.hexdigest())

        if cache_dir is None:
            catalogue = _CaptureCatalogue()
            build(catalogue)
            return catalogue

        directory = Path(cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        version = repository.capture_manifest_version(path)
        identity = [fingerprint.hexdigest(), path, prefix, version]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        publication = directory / f"capture-catalogue-v1-{key}.sqlite3"
        if not publication.is_file():
            with FileLock(str(publication) + ".lock"):
                if not publication.is_file():
                    with TemporaryDirectory(dir=directory) as staging:
                        temporary = Path(staging) / "catalogue.sqlite3"
                        catalogue = _CaptureCatalogue(temporary)
                        try:
                            build(catalogue)
                            if repository.capture_manifest_version(path) != version:
                                raise ValueError(
                                    "Capture manifest changed while indexing it."
                                )
                        finally:
                            catalogue.connection.close()
                        temporary.replace(publication)
        return _CaptureCatalogue(publication, readonly=True)

    def _scan_shards(self, records, prefix, fingerprint, shards):
        capture = isinstance(shards, _CaptureCatalogue)
        paths = set()
        for record in records:
            record = dict(record)
            path = _relative_path(record["path"])
            included = path.startswith(prefix)
            if capture:
                shards.add(record, included)
            else:
                if path in paths:
                    raise ValueError(f"Duplicate source shard: {path}")
                paths.add(path)
            if not path.endswith(".parquet"):
                raise ValueError("Source shards must be Parquet files")
            if not included:
                continue
            if capture and self._source_repo.s3 is not None:
                actual_size = record["size"]
            else:
                if not self._source_repo.exists(path):
                    raise ValueError(f"Missing source shard: {path}")
                actual_size = self._source_repo.size(path)
            if (
                "size" in record
                and _integer(record["size"], "shard size", 1) != actual_size
            ):
                raise ValueError(f"Source size mismatch: {path}")
            record["size"] = actual_size
            if "num_rows" in record:
                _integer(record["num_rows"], "num_rows")
            if "metadata_path" in record:
                metadata_path = _relative_path(record["metadata_path"])
                if not self._metadata_repo or not self._metadata_repo.exists(
                    metadata_path
                ):
                    raise ValueError(f"Missing metadata file: {metadata_path}")
                if not metadata_path.endswith((".parquet", ".json")):
                    raise ValueError("Custom metadata must be JSON or Parquet")
            if "metadata_sha256" in record and not re.fullmatch(
                r"[0-9a-f]{64}", record["metadata_sha256"]
            ):
                raise ValueError("Invalid metadata SHA-256")
            if not capture:
                shards.append(record)
            # The manifest digest already covers the per-object hashes. Keep the existing catalogue ID.
            if capture and self._source_repo.capture_digest:
                record.pop("sha256", None)
            fingerprint.update(json.dumps(record, sort_keys=True).encode())

    def enable_shard_cache(
        self, location: str, cache_limit_gb: float = 25.0, parallel_downloads: int = 4
    ):
        self._shard_cache = _ParquetCache(
            str(location), cache_limit_gb, parallel_downloads
        )
        self._payload_files.clear()
        self._payload_budget = int(cache_limit_gb * 1024**3)
        self._payload_sizes = {}

    def io_stats(self):
        """Logical S3 transport operations and transferred bytes. SDK retries are internal."""
        return (
            self._source_repo.s3.io_stats() if self._source_repo.s3 is not None else {}
        )

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def physical_row_count(self):
        """A published row count, available without opening any shard."""
        return (
            self._shards.rows if isinstance(self._shards, _CaptureCatalogue) else None
        )

    @_synchronized
    def _shard(self, index: int) -> dict:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < self.num_shards
        ):
            raise IndexError("Parquet shard index out of bounds")
        return self._shards[index]

    def _reader_revision(self, shard, reader):
        repository = self._source_repo
        if repository.local:
            with Path(repository.location(shard["path"])).open("rb") as stream:
                stat = os.fstat(stream.fileno())
                stamp = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
                if stamp not in self._local_digests:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if os.fstat(stream.fileno()).st_ctime_ns != stat.st_ctime_ns:
                        raise ValueError(
                            "Parquet object changed while computing its content revision."
                        )
                    self._local_digests[stamp] = digest
                    if len(self._local_digests) > self.metadata_cache_size:
                        self._local_digests.popitem(last=False)
                self._local_digests.move_to_end(stamp)
                return f"sha256:{self._local_digests[stamp]}"
        if repository.s3 is None:
            return f"hub:{repository.revision}"
        etag = reader.info().get("etag")
        if etag is None:
            raise ValueError("An S3 ETag is required to pin a prepared Parquet source.")
        return f"s3:{etag}"

    @_synchronized
    def bind_source_revisions(self, lookup):
        """Look up prepared content pins by shard index on subsequent reader opens."""
        self._expected_source_revision = lookup
        self._files.clear()
        self._payload_files.clear()
        if self._shard_cache is not None:
            self._payload_sizes.clear()
        self._metadata_cache.clear()
        for db in self._overlays.values():
            db.close()
        self._overlays.clear()

    def source_revision_for_shard(self, index):
        reader = self._file(index)
        with self._lock:
            return self._view_revision(index, self._shard(index), reader)

    def _view_revision(self, index, shard, reader):
        physical = self._reader_revision(shard, reader)
        if "metadata_path" in shard and "metadata_sha256" not in shard:
            overlay = self._overlay_index(index, shard, reader.info())
            (digest,) = overlay.execute("SELECT digest FROM revision").fetchone()
            return json.dumps([physical, digest], separators=(",", ":"))
        return physical

    def _file(self, index: int, *, payload: bool = False) -> _ParquetFile:
        with self._lock:
            shard = self._shard(index)
            cache = self._shard_cache if payload else None
            files = self._payload_files if cache is not None else self._files
            if index in files:
                files.move_to_end(index)
                return files[index]

        reader = _ParquetFile(
            self._source_repo.location(shard["path"]),
            size=shard["size"],
            hf_token=self.hf_token,
            max_page_bytes=self.max_page_bytes,
            s3=self._source_repo.s3,
            cache=cache,
            expected_sha256=shard.get("sha256"),
        )
        info = reader.info()
        with self._lock:
            shard = self._shard(index)
            if self._expected_source_revision is not None:
                expected = self._expected_source_revision(index)
                if (
                    expected is not None
                    and self._view_revision(index, shard, reader) != expected
                ):
                    raise ValueError(
                        "Parquet object changed since the accepted dataset view was prepared."
                    )
            if info.get("etag") is not None:
                if "etag" in shard and shard["etag"] != info["etag"]:
                    raise OSError(
                        "S3 object changed since its first read; reopen the dataset"
                    )
                if "etag" not in shard and isinstance(self._shards, _CaptureCatalogue):
                    self._shards.set_etag(index, info["etag"])
                shard["etag"] = info["etag"]
            count = info["num_rows"]
            if "num_rows" in shard and count != shard["num_rows"]:
                raise ValueError("Source row count does not match metadata manifest")
            shard["num_rows"] = count
            if index in files:
                files.move_to_end(index)
                return files[index]
            if files is self._payload_files:
                while files and (
                    len(files) >= self.metadata_cache_size
                    or sum(self._payload_sizes.values()) + shard["size"]
                    > self._payload_budget
                ):
                    evicted = next(iter(files))
                    del files[evicted], self._payload_sizes[evicted]
            files[index] = reader
            if files is self._payload_files:
                self._payload_sizes[index] = shard["size"]
            if len(files) > self.metadata_cache_size:
                files.popitem(last=False)
            return reader

    def get_shard_info(self, index: int) -> dict:
        shard = self._shard(index)
        if "num_rows" not in shard:
            self._file(index)
        return {**shard, "num_samples": shard["num_rows"], "format": "parquet"}

    @_synchronized
    def get_shard_position(self, path: str) -> int | None:
        """Resolve a repository-relative shard path within this dataset selection."""
        path = _relative_path(path)
        if isinstance(self._shards, _CaptureCatalogue):
            return self._shards.position(path)
        return next(
            (
                index
                for index, shard in enumerate(self._shards)
                if shard["path"] == path
            ),
            None,
        )

    def __len__(self) -> int:
        return self.count_rows()

    def count_rows(self, *, exclude_null_images=False, shard_index=None) -> int:
        """Count physical rows, optionally requiring embedded non-null image bytes.

        Null counts use Parquet footer statistics when available. Missing
        statistics require reading the affected image column. An empty byte
        string counts as present. Physical row indices are unchanged.
        """
        indices = range(self.num_shards) if shard_index is None else (shard_index,)
        if not exclude_null_images:
            published = self.physical_row_count if shard_index is None else None
            if published is not None:
                return published
            return sum(self.get_shard_info(index)["num_rows"] for index in indices)
        if self.image_url_column:
            raise ValueError("Excluding null images requires an embedded image column")
        return sum(
            self._file(index).count_present_images(self.image_column)
            for index in indices
        )

    def list_samples_in_shard(self, index: int) -> list[str]:
        shard = self.get_shard_info(index)
        return [f"{shard['path']}/{row:012d}" for row in range(shard["num_rows"])]

    def _decode(self, row: dict) -> dict:
        for column in self.json_columns:
            if column in row and isinstance(row[column], str):
                row[column] = json.loads(row[column])
        return row

    @_synchronized
    def _metadata(self, index: int) -> dict[int, dict]:
        if index in self._metadata_cache:
            self._metadata_cache.move_to_end(index)
            return self._metadata_cache[index]
        reader = self._file(index)
        if self.image_url_column and self.metadata_mode == "overlay":
            reader.cache_source(self.url_metadata_cache_bytes)
        shard = self._shard(index)
        info = reader.info()
        binary_columns = set(info["binary_columns"])
        columns = [
            column
            for column in info["columns"]
            if column not in binary_columns and column != self.image_column
        ]
        records = {}
        if self.metadata_mode == "overlay" and columns:
            reader.prefetch_columns(columns, parallel_reads=self.parallel_reads)
            records = {
                row_index: self._decode(row)
                for row_index, row in enumerate(
                    reader.rows_at(
                        list(range(info["num_rows"])),
                        columns=columns,
                        parallel_reads=self.parallel_reads,
                    )
                )
            }
        if "metadata_path" in shard or self.metadata_mode == "replace":
            db = self._overlay_index(index, shard, info)
            for row_index, value in db.execute("SELECT position, value FROM overlay"):
                records.setdefault(row_index, {}).update(pickle.loads(value))
        self._metadata_cache[index] = records
        if len(self._metadata_cache) > self.metadata_cache_size:
            self._metadata_cache.popitem(last=False)
        return records

    def _overlay_index(self, index, shard, info):
        if index in self._overlays:
            self._overlays.move_to_end(index)
            return self._overlays[index]
        if "metadata_path" not in shard:
            raise ValueError(
                "Replacement metadata requires a metadata file for every shard"
            )
        path = shard["metadata_path"]
        expected_hash = shard.get("metadata_sha256")
        repository = self._metadata_repo
        if path.endswith(".json"):
            rows = repository.iter_json_rows(path, expected_hash)
        else:
            reader = _ParquetFile(
                repository.location(path),
                size=repository.size(path),
                hf_token=self.hf_token,
                max_page_bytes=self.max_page_bytes,
                s3=repository.s3,
            )
            if expected_hash:
                reader.verify_sha256(expected_hash)
            if reader.info()["binary_columns"]:
                raise ValueError(
                    "Custom metadata must not contain binary payload columns"
                )
            rows = reader.rows()
        forbidden = set(info["binary_columns"]) | {
            self.image_column,
            self.image_url_column,
        }
        db = sqlite3.connect("", check_same_thread=False)
        try:
            db.execute(
                "CREATE TABLE overlay(position INTEGER PRIMARY KEY, value BLOB NOT NULL)"
            )
            digest = hashlib.sha256() if expected_hash is None else None
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Custom metadata rows must be objects")
                i = _integer(row.get("source_row"), "source_row")
                if i >= info["num_rows"]:
                    raise ValueError("Invalid out-of-bounds metadata source_row")
                if "source_path" in row and row["source_path"] != shard["path"]:
                    raise ValueError("Metadata source_path does not match its shard")
                if forbidden.intersection(row):
                    raise ValueError("Custom metadata cannot override payload columns")
                try:
                    encoded = pickle.dumps(self._decode(row), protocol=5)
                    db.execute("INSERT INTO overlay VALUES (?,?)", (i, encoded))
                    if digest is not None:
                        digest.update(encoded)
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate metadata source_row") from exc
            if self.metadata_mode == "replace" and db.total_changes != info["num_rows"]:
                raise ValueError("Replacement metadata must cover every source row")
            if digest is not None:
                db.execute("CREATE TABLE revision(digest TEXT)")
                db.execute("INSERT INTO revision VALUES (?)", (digest.hexdigest(),))
            db.commit()
        except BaseException:
            db.close()
            raise
        self._overlays[index] = db
        if len(self._overlays) > self.metadata_cache_size:
            self._overlays.popitem(last=False)[1].close()
        return db

    @staticmethod
    def _check_row(shard, index):
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < shard["num_rows"]
        ):
            raise IndexError("Parquet sample index out of bounds")

    def _row_metadata(self, shard, index, record, fields=None):
        metadata = deepcopy(
            record
            if fields is None
            else {name: record[name] for name in fields if name in record}
        )
        for output, column in (
            ("captions", self.caption_column),
            ("width", self.width_column),
            ("height", self.height_column),
        ):
            if (
                (fields is None or output in fields)
                and column in record
                and output not in metadata
            ):
                metadata[output] = deepcopy(record[column])
        metadata["source_path"] = shard["path"]
        metadata["source_row"] = index
        return metadata

    def get_metadata(self, shard_index: int, sample_index: int) -> dict:
        shard = self.get_shard_info(shard_index)
        self._check_row(shard, sample_index)
        record = self._metadata(shard_index).get(sample_index, {})
        return self._row_metadata(shard, sample_index, record)

    def read_metadata_rows(
        self, shard_index, indices, *, fields=None, parallel_reads=4
    ):
        """Read selected physical rows in order. Identity is always returned.

        Source columns are projected at the reader, using a cached S3 shard when
        available. A disk-backed index validates custom metadata coverage and
        uniqueness independently of the requested rows.
        """
        shard = self.get_shard_info(shard_index)
        indices = list(indices)
        for index in indices:
            self._check_row(shard, index)
        if isinstance(fields, str):
            raise TypeError("fields must be a collection of column names")
        fields = None if fields is None else frozenset(fields)
        if not indices or fields == frozenset():
            return [
                {"source_path": shard["path"], "source_row": index} for index in indices
            ]
        info = self._file(shard_index).info()
        if self.metadata_mode == "replace":
            rows = ({} for _ in indices)
        else:
            reader = self._file(shard_index)
            columns = self._metadata_columns(info, fields)
            rows = (
                (
                    self._decode(row)
                    for row in reader.rows_at(
                        indices,
                        columns=columns,
                        parallel_reads=parallel_reads,
                        shard_cache=self._shard_cache,
                    )
                )
                if columns
                else ({} for _ in indices)
            )
        if "metadata_path" in shard or self.metadata_mode == "replace":
            with self._lock:
                overlay = self._overlay_index(shard_index, shard, info)
                output = []
                for index, row in zip(indices, rows, strict=True):
                    value = overlay.execute(
                        "SELECT value FROM overlay WHERE position=?", (index,)
                    ).fetchone()
                    if value is not None:
                        row.update(pickle.loads(value[0]))
                    output.append(self._row_metadata(shard, index, row, fields))
                return output
        return [
            self._row_metadata(shard, index, row, fields)
            for index, row in zip(indices, rows, strict=True)
        ]

    def _metadata_columns(self, info, fields):
        excluded = set(info["binary_columns"]) | {self.image_column}
        columns = [name for name in info["columns"] if name not in excluded]
        if fields is None:
            return columns
        aliases = {
            "captions": self.caption_column,
            "width": self.width_column,
            "height": self.height_column,
        }
        requested = fields.union(aliases[name] for name in fields if name in aliases)
        return [name for name in columns if name in requested]

    def _iter_projected_rows(self, reader, info, columns):
        if self._source_repo.local:
            yield from reader.rows(columns=columns)
            return

        starts = tuple(accumulate(info["row_groups"], initial=0))

        def read_groups(start, stop):
            cached = reader.prefetch_columns(
                columns,
                max_bytes=PROJECTED_SCAN_CACHE_BYTES,
                parallel_reads=self.parallel_reads,
                start_group=start,
                stop_group=stop,
            )
            if cached or stop - start <= 1:
                yield from reader.rows(
                    columns=columns, start=starts[start], stop=starts[stop]
                )
            else:
                mid = (start + stop) // 2
                yield from read_groups(start, mid)
                yield from read_groups(mid, stop)

        if len(starts) > 1:
            yield from read_groups(0, len(starts) - 1)

    def _iter_source_metadata(self, shard_index, shard, fields):
        reader = self._file(shard_index)
        info = reader.info()
        columns = self._metadata_columns(info, fields)
        rows = (
            self._iter_projected_rows(reader, info, columns)
            if columns
            else ({} for _ in range(shard["num_rows"]))
        )
        for index, row in enumerate(rows):
            yield self._row_metadata(shard, index, self._decode(row), fields)


@dataclass
class ParquetSample:
    """Decoded bytes and native metadata for a physical Parquet row.

    ``data`` is None for missing images or disabled/failed URL reads. ``payloads`` retains every
    requested source value. URL fetching requires explicit loader opt-in.
    """

    path: str
    data: bytes | None
    metadata: dict[str, Any]
    payloads: dict[str, Any]
    error: str | None = None

    @property
    def size(self) -> int:
        return len(self.data) if self.data is not None else 0

    @property
    def captions(self):
        return self.metadata.get("captions")

    @property
    def caption(self):
        value = self.captions
        return (value[0] if value else None) if isinstance(value, list) else value

    @property
    def width(self):
        return self.metadata.get("width")

    @property
    def height(self):
        return self.metadata.get("height")

    @property
    def aspect_ratio(self):
        return self.width / self.height if self.width and self.height else None


class ParquetDataLoader:
    """Stream selected payload columns page by page, not whole column chunks.

    Random access scans preceding rows within the selected row group. Prefer
    ``iter_shard`` for large row groups to reuse decoded dictionaries/pages.
    """

    def __init__(
        self,
        dataset: ParquetDataset,
        *,
        image_columns: list[str] | None = None,
        fetch_external_urls: bool = False,
        url_options: URLImageOptions | None = None,
        on_url_error: str = "raise",
    ):
        if not isinstance(dataset, ParquetDataset):
            raise TypeError("ParquetDataLoader requires a ParquetDataset")
        self.dataset = dataset
        if on_url_error not in {"raise", "return"}:
            raise ValueError("on_url_error must be 'raise' or 'return'")
        if fetch_external_urls and not dataset.image_url_column:
            raise ValueError("URL fetching requires image_url_column")
        self.fetch_external_urls = fetch_external_urls
        self.url_options = url_options or URLImageOptions()
        self.on_url_error = on_url_error
        self._url_client = (
            URLImageClient(self.url_options) if fetch_external_urls else None
        )
        self.image_columns = list(
            dict.fromkeys(
                [
                    dataset.image_url_column or dataset.image_column,
                    *(image_columns or []),
                ]
            )
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def get_metadata(self, shard_index: int, sample_index: int | None = None) -> dict:
        if sample_index is not None:
            return self.dataset.get_metadata(shard_index, sample_index)
        return {
            entry["filename"]: entry
            for entry in self.list_samples_in_shard(shard_index)
        }

    def iter_samples_in_shard(
        self, shard_index: int, *, metadata_fields=None, batch_size=256
    ):
        """Yield valid physical rows in capture order, with bounded batches for custom overlays."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        shard = self.dataset.get_shard_info(shard_index)
        if metadata_fields is None:
            records = self.dataset._metadata(shard_index)
            rows = (
                self.dataset._row_metadata(shard, index, records.get(index, {}))
                for index in range(shard["num_rows"])
            )
        else:
            fields = {*metadata_fields, "url_status", "stored_bytes", "stored_format"}
            if "metadata_path" not in shard and self.dataset.metadata_mode != "replace":
                rows = self.dataset._iter_source_metadata(shard_index, shard, fields)
            else:
                rows = (
                    row
                    for start in range(0, shard["num_rows"], batch_size)
                    for row in self.dataset.read_metadata_rows(
                        shard_index,
                        range(start, min(start + batch_size, shard["num_rows"])),
                        fields=fields,
                        parallel_reads=self.dataset.parallel_reads,
                    )
                )
        for metadata in rows:
            row_index = metadata["source_row"]
            if metadata.get("url_status", "ok") != "ok" or (
                "stored_bytes" in metadata and not metadata["stored_bytes"]
            ):
                continue
            extension = "jxl" if metadata.get("stored_format") == "jxl" else "jpg"
            yield {
                **metadata,
                "sample_idx": row_index,
                "shard_idx": shard_index,
                "filename": f"{metadata['source_path']}/{row_index:012d}.{extension}",
            }

    def list_samples_in_shard(
        self, shard_index: int, *, metadata_fields=None
    ) -> list[dict]:
        return list(
            self.iter_samples_in_shard(shard_index, metadata_fields=metadata_fields)
        )

    def list_shard_sample_aspect_buckets(
        self,
        shard_indices: list[int],
        *,
        key="aspect",
        target_pixel_area=None,
        target_resolution_multiple=64,
        round_to=2,
        metadata_fields=None,
    ) -> list[dict]:
        if metadata_fields is not None:
            metadata_fields = {*metadata_fields, "width", "height"}
        output = []
        for shard in shard_indices:
            entries = [
                entry
                for entry in self.list_samples_in_shard(
                    shard, metadata_fields=metadata_fields
                )
                if entry.get("width", 0)
                and entry.get("height", 0)
                and entry["width"] > 0
                and entry["height"] > 0
            ]
            keys = _parquet_bucket_keys(
                [(entry["width"], entry["height"]) for entry in entries],
                key,
                target_pixel_area,
                target_resolution_multiple,
                round_to,
            )
            buckets = {}
            for bucket, entry in zip(keys, entries):
                buckets.setdefault(bucket, []).append(entry)
            output.append(
                {
                    "shard_idx": shard,
                    "shard_name": self.dataset._shard(shard)["path"],
                    "buckets": buckets,
                }
            )
        return output

    def _sample(
        self, shard_index: int, sample_index: int, payloads: dict, metadata=None
    ) -> ParquetSample:
        error = None
        if self.dataset.image_url_column:
            data = None
            if self.fetch_external_urls:
                try:
                    data = self._url_client.fetch(
                        payloads[self.dataset.image_url_column]
                    )
                except URLImageError as failure:
                    if self.on_url_error == "raise":
                        raise
                    error = str(failure)
        else:
            value = payloads[self.dataset.image_column]
            data = value.get("bytes") if isinstance(value, dict) else value
        if data is not None and not isinstance(data, bytes):
            raise TypeError(
                "Image column must contain binary bytes or an image struct with bytes/path"
            )
        if metadata is None:
            metadata = self.get_metadata(shard_index, sample_index)
        return ParquetSample(
            path=f"{metadata['source_path']}/{sample_index:012d}",
            data=data,
            metadata=metadata,
            payloads=payloads,
            error=error,
        )

    def load_sample(self, shard_index: int, sample_index: int) -> ParquetSample:
        self.get_metadata(shard_index, sample_index)
        rows = self.dataset._file(shard_index, payload=True).rows(
            columns=self.image_columns, start=sample_index, stop=sample_index + 1
        )
        return self._sample(shard_index, sample_index, next(rows))

    def load_samples(
        self,
        requests: list[tuple[int, int]],
        *,
        parallel_reads: int = 4,
        metadata_fields=None,
    ) -> list[ParquetSample]:
        """Batch physical rows, sharing row-group reads and preserving duplicates/order."""
        groups = {}
        for position, (shard, row) in enumerate(requests):
            groups.setdefault(shard, []).append((position, row))
        output = [None] * len(requests)
        for shard, selected in groups.items():
            indices = [row for _, row in selected]
            metadata = self.dataset.read_metadata_rows(
                shard, indices, fields=metadata_fields, parallel_reads=parallel_reads
            )
            rows = self.dataset._file(shard, payload=True).rows_at(
                indices,
                columns=self.image_columns,
                parallel_reads=parallel_reads,
            )
            for (position, row), payloads, sample_metadata in zip(
                selected, rows, metadata, strict=True
            ):
                output[position] = self._sample(shard, row, payloads, sample_metadata)
        return output

    def iter_shard(
        self, shard_index: int, start: int = 0, stop: int | None = None
    ) -> Iterator[ParquetSample]:
        self.dataset._metadata(shard_index)
        rows = self.dataset._file(shard_index, payload=True).rows(
            columns=self.image_columns, start=start, stop=stop
        )
        for sample_index, payloads in enumerate(rows, start=start):
            yield self._sample(shard_index, sample_index, payloads)

    def __iter__(self) -> Iterator[ParquetSample]:
        for shard_index in range(self.dataset.num_shards):
            yield from self.iter_shard(shard_index)


def discover_parquet_dataset(source: str, **kwargs) -> ParquetDataset:
    return ParquetDataset(source, **kwargs)
