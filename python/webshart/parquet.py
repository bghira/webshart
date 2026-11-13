"""Page-streamed Parquet datasets with separately hosted, row-keyed metadata."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from threading import RLock
from typing import Any, Iterator
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ._webshart import _ParquetFile


MANIFEST = "webshart-parquet.json"


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
    def __init__(self, source: str, token: str | None, revision: str | None = None):
        self.source = (
            str(source).removeprefix("https://huggingface.co/datasets/").rstrip("/")
        )
        self.token = token
        self.local = Path(source).is_dir()
        self.revision = revision
        self.files: dict[str, int] = {}
        if self.local:
            self.root = Path(source).resolve()
        else:
            try:
                from huggingface_hub import HfApi
            except ImportError as error:
                raise ImportError(
                    "Hub Parquet discovery requires webshart[hub]"
                ) from error
            info = HfApi(token=token).dataset_info(
                self.source, revision=revision, files_metadata=True
            )
            self.revision = info.sha
            self.files = {entry.rfilename: entry.size for entry in info.siblings}

    def exists(self, path: str) -> bool:
        return Path(self.location(path)).is_file() if self.local else path in self.files

    def location(self, path: str) -> str:
        path = _relative_path(path)
        if self.local:
            target = (self.root / path).resolve()
            if not target.is_relative_to(self.root):
                raise ValueError("Metadata path escapes its root")
            return str(target)
        from huggingface_hub import hf_hub_url

        return hf_hub_url(
            self.source, path, repo_type="dataset", revision=self.revision
        )

    def size(self, path: str) -> int:
        return (
            Path(self.location(path)).stat().st_size if self.local else self.files[path]
        )

    def read_json(self, path: str, expected_hash: str | None = None) -> Any:
        if self.size(path) > 16 * 1024 * 1024:
            raise ValueError(
                "JSON metadata exceeds 16 MiB; use Parquet metadata instead"
            )
        if self.local:
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
        return json.loads(content)


class ParquetDataset:
    """Discover local/Hub Parquet shards, optionally paired with a metadata manifest.

    Custom metadata is keyed by physical ``source_row``, never by a potentially
    duplicated dataset ID. Manifest source revisions are pinned before discovery.
    Image bytes and external image URLs are not read during metadata discovery.
    """

    def __init__(
        self,
        source: str,
        *,
        metadata: str | None = None,
        hf_token: str | None = None,
        revision: str | None = None,
        metadata_revision: str | None = None,
        subfolder: str | None = None,
        image_column: str | None = None,
        caption_column: str | None = None,
        width_column: str | None = None,
        height_column: str | None = None,
        json_columns: list[str] | None = None,
        metadata_cache_size: int = 1,
        max_page_bytes: int = 512 * 1024 * 1024,
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
        self._metadata_cache: OrderedDict[int, dict[int, dict]] = OrderedDict()
        self._files: OrderedDict[int, _ParquetFile] = OrderedDict()
        self._metadata_repo = None
        manifest: dict = {}
        if metadata is not None:
            self._metadata_repo = _Repository(
                metadata, self.hf_token, metadata_revision
            )
            manifest = self._metadata_repo.read_json(MANIFEST)
            if (
                manifest.get("format") != "webshart-parquet"
                or manifest.get("version") != 1
            ):
                raise ValueError("Unsupported webshart Parquet metadata manifest")
            pinned = manifest.get("source", {})
            if not Path(source).is_dir():
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
        self._source_repo = _Repository(source, self.hf_token, revision)
        self.name = str(source)
        self.source_revision = self._source_repo.revision
        self.metadata_revision = (
            self._metadata_repo.revision if self._metadata_repo else None
        )
        columns = manifest.get("columns", {})
        self.image_column = image_column or columns.get("image", "image")
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
        if metadata is not None:
            records = manifest.get("shards")
            if not isinstance(records, list):
                raise ValueError("Manifest shards must be a list")
        elif self._source_repo.local:
            records = [
                {"path": path.relative_to(self._source_repo.root).as_posix()}
                for path in sorted(self._source_repo.root.rglob("*.parquet"))
            ]
        else:
            records = [
                {"path": path}
                for path in sorted(self._source_repo.files)
                if path.endswith(".parquet")
            ]
        self._shards: list[dict] = []
        paths = set()
        for record in records:
            record = dict(record)
            path = _relative_path(record["path"])
            if path in paths:
                raise ValueError(f"Duplicate source shard: {path}")
            paths.add(path)
            if not path.endswith(".parquet"):
                raise ValueError("Source shards must be Parquet files")
            if not path.startswith(prefix):
                continue
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
            self._shards.append(record)
        if not self._shards:
            raise ValueError("No Parquet shards found")

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    def _shard(self, index: int) -> dict:
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < self.num_shards
        ):
            raise IndexError("Parquet shard index out of bounds")
        return self._shards[index]

    @_synchronized
    def _file(self, index: int) -> _ParquetFile:
        shard = self._shard(index)
        if index not in self._files:
            reader = _ParquetFile(
                self._source_repo.location(shard["path"]),
                size=shard["size"],
                hf_token=self.hf_token,
                max_page_bytes=self.max_page_bytes,
            )
            count = reader.info()["num_rows"]
            if "num_rows" in shard and count != shard["num_rows"]:
                raise ValueError("Source row count does not match metadata manifest")
            shard["num_rows"] = count
            self._files[index] = reader
            if len(self._files) > self.metadata_cache_size:
                self._files.popitem(last=False)
        self._files.move_to_end(index)
        return self._files[index]

    def get_shard_info(self, index: int) -> dict:
        shard = self._shard(index)
        if "num_rows" not in shard:
            self._file(index)
        return {**shard, "num_samples": shard["num_rows"], "format": "parquet"}

    def __len__(self) -> int:
        return sum(
            self.get_shard_info(index)["num_rows"] for index in range(self.num_shards)
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
            records = {
                row_index: self._decode(row)
                for row_index, row in enumerate(reader.rows(columns=columns))
            }
        if "metadata_path" in shard:
            path = shard["metadata_path"]
            expected_hash = shard.get("metadata_sha256")
            repository = self._metadata_repo
            assert repository is not None
            if path.endswith(".json"):
                overlay = repository.read_json(path, expected_hash)
                if not isinstance(overlay, list):
                    raise ValueError("JSON metadata must be a list of row objects")
            else:
                metadata_reader = _ParquetFile(
                    repository.location(path),
                    size=repository.size(path),
                    hf_token=self.hf_token,
                    max_page_bytes=self.max_page_bytes,
                    cache_bytes=64 * 1024 * 1024,
                )
                if expected_hash:
                    metadata_reader.verify_sha256(expected_hash)
                if metadata_reader.info()["binary_columns"]:
                    raise ValueError(
                        "Custom metadata must not contain binary payload columns"
                    )
                overlay = metadata_reader.rows()
            seen = set()
            for row in overlay:
                if not isinstance(row, dict):
                    raise ValueError("Custom metadata rows must be objects")
                row_index = _integer(row.get("source_row"), "source_row")
                if row_index >= info["num_rows"] or row_index in seen:
                    raise ValueError("Duplicate or out-of-bounds metadata source_row")
                if "source_path" in row and row["source_path"] != shard["path"]:
                    raise ValueError("Metadata source_path does not match its shard")
                if binary_columns.intersection(row) or self.image_column in row:
                    raise ValueError("Custom metadata cannot override payload columns")
                seen.add(row_index)
                records.setdefault(row_index, {}).update(self._decode(row))
            if self.metadata_mode == "replace" and len(seen) != info["num_rows"]:
                raise ValueError("Replacement metadata must cover every source row")
        elif self.metadata_mode == "replace":
            raise ValueError(
                "Replacement metadata requires a metadata file for every shard"
            )
        self._metadata_cache[index] = records
        if len(self._metadata_cache) > self.metadata_cache_size:
            self._metadata_cache.popitem(last=False)
        return records

    def get_metadata(self, shard_index: int, sample_index: int) -> dict:
        shard = self.get_shard_info(shard_index)
        if (
            not isinstance(sample_index, int)
            or isinstance(sample_index, bool)
            or not 0 <= sample_index < shard["num_rows"]
        ):
            raise IndexError("Parquet sample index out of bounds")
        metadata = deepcopy(self._metadata(shard_index).get(sample_index, {}))
        metadata["source_path"] = shard["path"]
        metadata["source_row"] = sample_index
        for output, column in (
            ("captions", self.caption_column),
            ("width", self.width_column),
            ("height", self.height_column),
        ):
            if column in metadata and output not in metadata:
                metadata[output] = metadata[column]
        return metadata


@dataclass
class ParquetSample:
    """Decoded bytes and native metadata for a physical Parquet row.

    ``data`` is None for missing/external images. ``payloads`` retains every
    requested binary or Hugging Face image-struct value; URLs are never fetched.
    """

    path: str
    data: bytes | None
    metadata: dict[str, Any]
    payloads: dict[str, Any]

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
        self, dataset: ParquetDataset, *, image_columns: list[str] | None = None
    ):
        if not isinstance(dataset, ParquetDataset):
            raise TypeError("ParquetDataLoader requires a ParquetDataset")
        self.dataset = dataset
        self.image_columns = list(
            dict.fromkeys([dataset.image_column, *(image_columns or [])])
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def get_metadata(self, shard_index: int, sample_index: int) -> dict:
        return self.dataset.get_metadata(shard_index, sample_index)

    def _sample(
        self, shard_index: int, sample_index: int, payloads: dict
    ) -> ParquetSample:
        value = payloads[self.dataset.image_column]
        data = value.get("bytes") if isinstance(value, dict) else value
        if data is not None and not isinstance(data, bytes):
            raise TypeError(
                "Image column must contain binary bytes or an image struct with bytes/path"
            )
        metadata = self.get_metadata(shard_index, sample_index)
        return ParquetSample(
            path=f"{metadata['source_path']}/{sample_index:012d}",
            data=data,
            metadata=metadata,
            payloads=payloads,
        )

    def load_sample(self, shard_index: int, sample_index: int) -> ParquetSample:
        self.get_metadata(shard_index, sample_index)
        rows = self.dataset._file(shard_index).rows(
            columns=self.image_columns, start=sample_index, stop=sample_index + 1
        )
        return self._sample(shard_index, sample_index, next(rows))

    def iter_shard(
        self, shard_index: int, start: int = 0, stop: int | None = None
    ) -> Iterator[ParquetSample]:
        self.dataset._metadata(shard_index)
        rows = self.dataset._file(shard_index).rows(
            columns=self.image_columns, start=start, stop=stop
        )
        for sample_index, payloads in enumerate(rows, start=start):
            yield self._sample(shard_index, sample_index, payloads)

    def __iter__(self) -> Iterator[ParquetSample]:
        for shard_index in range(self.dataset.num_shards):
            yield from self.iter_shard(shard_index)


def discover_parquet_dataset(source: str, **kwargs) -> ParquetDataset:
    return ParquetDataset(source, **kwargs)
