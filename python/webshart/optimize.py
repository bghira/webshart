"""Rolling conversion of loose payload/sidecar pairs into indexed webshart shards."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Iterator, Optional, Sequence, Union
import json
import os
import shutil
import tarfile
import urllib.request

from tqdm import tqdm
from webshart._webshart import MetadataExtractor


CaptionValue = Union[str, list[str]]

DEFAULT_PAYLOAD_EXTENSIONS = (
    ".avif",
    ".bmp",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".png",
    ".tif",
    ".tiff",
    ".wav",
    ".webm",
    ".webp",
)
CAPTION_KEYS = (
    "caption",
    "captions",
    "text",
    "txt",
    "description",
    "descriptions",
    "prompt",
    "alt_text",
)
STATE_FILENAME = ".webshart-optimize-state.json"
STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    local_path: Optional[Path] = None


@dataclass(frozen=True)
class LooseSample:
    path: str
    size: int
    payload: SourceFile
    sidecar: Optional[SourceFile]


@dataclass
class OptimizationState:
    schema_version: int
    status: str
    source: dict[str, Any]
    manifest_sha256: str
    output_prefix: str
    max_shard_size_bytes: int
    payload_extensions: list[str]
    total_samples: int
    next_sample_index: int = 0
    next_shard_index: int = 0
    captioned_samples: int = 0
    uncaptioned_samples: int = 0
    bytes_sharded: int = 0


def _require_hub():
    try:
        from huggingface_hub import (
            CommitOperationAdd,
            HfApi,
            get_hf_file_metadata,
            hf_hub_download,
            hf_hub_url,
        )
    except ImportError as exc:
        raise ImportError(
            "Hub optimization requires huggingface-hub; install webshart[hub]"
        ) from exc
    return (
        HfApi,
        CommitOperationAdd,
        hf_hub_download,
        hf_hub_url,
        get_hf_file_metadata,
    )


def _normalize_prefix(value: str) -> str:
    value = value.strip("/")
    if not value:
        return ""
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid repository path prefix: {value!r}")
    return str(path)


def _repo_path(prefix: str, filename: str) -> str:
    return str(PurePosixPath(prefix, filename)) if prefix else filename


def _normalize_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for extension in extensions:
        extension = extension.strip().lower()
        if not extension:
            continue
        normalized.append(extension if extension.startswith(".") else f".{extension}")
    if not normalized:
        raise ValueError("at least one payload extension is required")
    return tuple(sorted(set(normalized)))


def _relative_source_path(path: str, subfolder: str) -> Optional[str]:
    source_path = PurePosixPath(path)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError(f"unsafe source path: {path!r}")
    if not subfolder:
        return str(source_path)
    prefix = PurePosixPath(subfolder)
    try:
        return str(source_path.relative_to(prefix))
    except ValueError:
        return None


def _list_local_files(source: Path, subfolder: str) -> list[SourceFile]:
    root = source / Path(subfolder) if subfolder else source
    if not root.is_dir():
        raise ValueError(f"local source folder does not exist: {root}")
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files.append(SourceFile(relative, path.stat().st_size, path))
    return files


def _list_hub_files(
    repo_id: str,
    subfolder: str,
    revision: str,
    token: Optional[str],
) -> list[SourceFile]:
    HfApi, _, _, _, _ = _require_hub()
    api = HfApi(token=token)
    info = api.dataset_info(
        repo_id,
        revision=revision,
        token=token,
        files_metadata=False,
    )
    files = []
    for entry in info.siblings or ():
        relative = _relative_source_path(entry.rfilename, subfolder)
        if relative is not None:
            files.append(SourceFile(relative, -1))
    return files


def _build_samples(
    files: Sequence[SourceFile], payload_extensions: Sequence[str]
) -> list[LooseSample]:
    sidecars: dict[str, SourceFile] = {}
    for file in files:
        path = PurePosixPath(file.path)
        extension = path.suffix.lower()
        if extension not in {".txt", ".json"}:
            continue
        stem = str(path.with_suffix(""))
        existing = sidecars.get(stem)
        if existing is None or extension == ".txt":
            sidecars[stem] = file
    payloads = [
        file
        for file in files
        if PurePosixPath(file.path).suffix.lower() in payload_extensions
    ]
    samples = []
    for payload in sorted(payloads, key=lambda file: file.path):
        stem = str(PurePosixPath(payload.path).with_suffix(""))
        sidecar = sidecars.get(stem)
        samples.append(
            LooseSample(
                path=payload.path,
                size=payload.size,
                payload=payload,
                sidecar=sidecar,
            )
        )
    return samples


def _manifest_sha256(samples: Sequence[LooseSample]) -> str:
    digest = sha256()
    for sample in samples:
        digest.update(sample.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sample.size).encode("ascii"))
        digest.update(b"\0")
        if sample.sidecar is not None:
            digest.update(sample.sidecar.path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(sample.sidecar.size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@contextmanager
def _open_source_file(
    file: SourceFile,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> Iterator[BinaryIO]:
    if file.local_path is not None:
        with file.local_path.open("rb") as handle:
            yield handle
        return

    if source_repo is None:
        raise ValueError("remote source repository is missing")
    _, _, _, hf_hub_url, _ = _require_hub()
    remote_path = _repo_path(source_subfolder, file.path)
    url = hf_hub_url(
        source_repo,
        remote_path,
        repo_type="dataset",
        revision=source_revision,
    )
    headers = {"Accept-Encoding": "identity", "User-Agent": "webshart/optimize-dataset"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        yield response


def _source_file_size(
    file: SourceFile,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> int:
    if file.size >= 0:
        return file.size
    if source_repo is None:
        raise ValueError(f"source size is unavailable: {file.path}")
    _, _, _, hf_hub_url, get_hf_file_metadata = _require_hub()
    remote_path = _repo_path(source_subfolder, file.path)
    url = hf_hub_url(
        source_repo,
        remote_path,
        repo_type="dataset",
        revision=source_revision,
    )
    metadata = get_hf_file_metadata(url, token=token)
    if metadata.size is None:
        raise ValueError(f"Hub did not report a size for {remote_path}")
    return int(metadata.size)


def _read_small_file(
    file: SourceFile,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
    max_bytes: int = 4 * 1024 * 1024,
) -> bytes:
    if file.size > max_bytes:
        raise ValueError(f"caption sidecar exceeds {max_bytes} bytes: {file.path}")
    with _open_source_file(
        file,
        source_repo=source_repo,
        source_subfolder=source_subfolder,
        source_revision=source_revision,
        token=token,
    ) as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"caption sidecar exceeds {max_bytes} bytes: {file.path}")
    return data


def _normalize_caption(value: Any) -> Optional[CaptionValue]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        captions = []
        seen = set()
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    captions.append(item)
        if len(captions) == 1:
            return captions[0]
        return captions or None
    return None


def _metadata_from_sidecar(
    sidecar: Optional[SourceFile],
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> tuple[Optional[CaptionValue], Optional[dict[str, Any]]]:
    if sidecar is None:
        return None, None
    data = _read_small_file(
        sidecar,
        source_repo=source_repo,
        source_subfolder=source_subfolder,
        source_revision=source_revision,
        token=token,
    )
    if PurePosixPath(sidecar.path).suffix.lower() == ".txt":
        return _normalize_caption(data.decode("utf-8")), None

    value = json.loads(data)
    if not isinstance(value, dict):
        return None, None
    captions = []
    for key in CAPTION_KEYS:
        caption = _normalize_caption(value.get(key))
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, list):
            captions.extend(caption)
    return _normalize_caption(captions), value


def _tar_member_size(payload_size: int) -> int:
    return 512 + ((payload_size + 511) // 512) * 512


def _add_to_tar(
    archive: tarfile.TarFile,
    sample: LooseSample,
    payload_size: int,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> None:
    info = tarfile.TarInfo(sample.path)
    info.size = payload_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with _open_source_file(
        sample.payload,
        source_repo=source_repo,
        source_subfolder=source_subfolder,
        source_revision=source_revision,
        token=token,
    ) as handle:
        archive.addfile(info, handle)


def _apply_sidecar_metadata(
    metadata_path: Path,
    captions: dict[str, CaptionValue],
    json_metadata: dict[str, dict[str, Any]],
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"invalid webshart metadata generated at {metadata_path}")
    for path, caption in captions.items():
        entry = files.get(path)
        if isinstance(entry, dict):
            entry.pop("caption", None)
            entry["captions"] = caption
    for path, value in json_metadata.items():
        entry = files.get(path)
        if isinstance(entry, dict):
            entry["json_metadata"] = value
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_state(path: Path, state: OptimizationState) -> None:
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_local_state(path: Path) -> Optional[OptimizationState]:
    if not path.is_file():
        return None
    return OptimizationState(**json.loads(path.read_text(encoding="utf-8")))


def _load_hub_state(
    api: Any,
    repo_id: str,
    state_repo_path: str,
    revision: str,
    token: Optional[str],
) -> Optional[OptimizationState]:
    _, _, hf_hub_download, _, _ = _require_hub()
    if not api.file_exists(
        repo_id,
        state_repo_path,
        repo_type="dataset",
        revision=revision,
        token=token,
    ):
        return None
    state_path = hf_hub_download(
        repo_id,
        state_repo_path,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return _load_local_state(Path(state_path))


def _validate_resume_state(
    state: OptimizationState,
    expected: OptimizationState,
) -> None:
    immutable_fields = (
        "schema_version",
        "source",
        "manifest_sha256",
        "output_prefix",
        "max_shard_size_bytes",
        "payload_extensions",
        "total_samples",
    )
    changed = [
        field
        for field in immutable_fields
        if getattr(state, field) != getattr(expected, field)
    ]
    if changed:
        raise ValueError(
            "optimization state does not match this source/configuration; "
            f"changed fields: {', '.join(changed)}"
        )
    if state.status not in {"running", "complete"}:
        raise ValueError(f"invalid optimization state status: {state.status!r}")
    if not 0 <= state.next_sample_index <= state.total_samples:
        raise ValueError("optimization state sample position is out of range")
    if state.next_shard_index < 0 or state.bytes_sharded < 0:
        raise ValueError("optimization state contains negative counters")
    if state.captioned_samples + state.uncaptioned_samples != state.next_sample_index:
        raise ValueError(
            "optimization state caption counters do not match its position"
        )


def optimize_dataset(
    source: Union[str, Path],
    *,
    destination: Optional[Union[str, Path]] = None,
    push_to_hub: Optional[str] = None,
    source_subfolder: str = "",
    output_prefix: str = "webshart",
    source_revision: str = "main",
    target_revision: str = "main",
    hf_token: Optional[str] = None,
    max_shard_size_bytes: int = 1024**3,
    payload_extensions: Sequence[str] = DEFAULT_PAYLOAD_EXTENSIONS,
    include_image_geometry: bool = True,
    max_shards: Optional[int] = None,
    private: Optional[bool] = None,
) -> dict[str, Any]:
    """Convert loose payload/sidecar pairs into rolling, resumable webshart shards.

    When ``push_to_hub`` is set, each completed tar/index pair and the resume
    state are committed atomically to that dataset repository. Without a local
    destination, only one shard is retained on disk at a time.
    """
    if destination is None and push_to_hub is None:
        raise ValueError("destination or push_to_hub is required")
    if max_shard_size_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be greater than zero")
    if max_shards is not None and max_shards <= 0:
        raise ValueError("max_shards must be greater than zero")

    token = hf_token or os.environ.get("HF_TOKEN")
    extensions = _normalize_extensions(payload_extensions)
    source_subfolder = _normalize_prefix(source_subfolder)
    output_prefix = _normalize_prefix(output_prefix)
    source_path = Path(source).expanduser()
    is_local = source_path.is_dir()
    source_repo = None if is_local else str(source)

    if is_local:
        files = _list_local_files(source_path, source_subfolder)
        source_identity = {"kind": "local", "subfolder": source_subfolder}
    else:
        files = _list_hub_files(
            source_repo,
            source_subfolder,
            source_revision,
            token,
        )
        source_identity = {
            "kind": "hub",
            "repo_id": source_repo,
            "revision": source_revision,
            "subfolder": source_subfolder,
        }

    samples = _build_samples(files, extensions)
    if not samples:
        tar_count = sum(
            PurePosixPath(file.path).suffix.lower() == ".tar" for file in files
        )
        if tar_count:
            raise ValueError(
                "source already contains tar shards; use optimize-captions instead"
            )
        raise ValueError(
            "no loose payload files matched the configured extensions: "
            + ", ".join(extensions)
        )

    expected_state = OptimizationState(
        schema_version=STATE_SCHEMA_VERSION,
        status="running",
        source=source_identity,
        manifest_sha256=_manifest_sha256(samples),
        output_prefix=output_prefix,
        max_shard_size_bytes=max_shard_size_bytes,
        payload_extensions=list(extensions),
        total_samples=len(samples),
    )

    destination_path = (
        Path(destination).expanduser() if destination is not None else None
    )
    local_state_path = (
        destination_path / output_prefix / STATE_FILENAME
        if destination_path is not None
        else None
    )
    state_repo_path = _repo_path(output_prefix, STATE_FILENAME)
    api = None
    CommitOperationAdd = None
    if push_to_hub:
        HfApi, CommitOperationAdd, _, _, _ = _require_hub()
        api = HfApi(token=token)
        api.create_repo(
            push_to_hub,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token,
        )
        state = _load_hub_state(
            api,
            push_to_hub,
            state_repo_path,
            target_revision,
            token,
        )
        if state is None and (
            api.file_exists(
                push_to_hub,
                _repo_path(output_prefix, "shard-00000.tar"),
                repo_type="dataset",
                revision=target_revision,
                token=token,
            )
            or api.file_exists(
                push_to_hub,
                _repo_path(output_prefix, "shard-00000.json"),
                repo_type="dataset",
                revision=target_revision,
                token=token,
            )
        ):
            raise ValueError(
                "optimized shards exist at the target but the resume state is missing; "
                "restore the state or choose a different output_prefix"
            )
    else:
        state = _load_local_state(local_state_path) if local_state_path else None
        output_dir = destination_path / output_prefix
        if state is None and (
            (output_dir / "shard-00000.tar").exists()
            or (output_dir / "shard-00000.json").exists()
        ):
            raise ValueError(
                "optimized shards exist at the destination but the resume state is missing; "
                "restore the state or choose a different destination/output_prefix"
            )

    if state is not None:
        _validate_resume_state(state, expected_state)
    else:
        state = expected_state

    if state.status == "complete":
        return {**asdict(state), "shards_created": 0, "resumed": True}

    if destination_path is not None:
        (destination_path / output_prefix).mkdir(parents=True, exist_ok=True)

    shards_created = 0
    resumed = state.next_sample_index > 0
    with TemporaryDirectory(prefix="webshart-optimize-") as temporary:
        staging = Path(temporary)
        extractor = MetadataExtractor(hf_token=token)
        progress = tqdm(
            total=len(samples),
            initial=state.next_sample_index,
            unit="sample",
            desc="Optimizing dataset",
        )

        try:
            while state.next_sample_index < len(samples):
                if max_shards is not None and shards_created >= max_shards:
                    break

                shard_name = f"shard-{state.next_shard_index:05d}"
                tar_path = staging / f"{shard_name}.tar"
                metadata_path = staging / f"{shard_name}.json"
                state_path = staging / STATE_FILENAME
                captions: dict[str, CaptionValue] = {}
                json_metadata: dict[str, dict[str, Any]] = {}
                shard_size = 1024
                shard_samples = 0
                start_index = state.next_sample_index

                with tarfile.open(
                    tar_path, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    while state.next_sample_index < len(samples):
                        sample = samples[state.next_sample_index]
                        payload_size = _source_file_size(
                            sample.payload,
                            source_repo=source_repo,
                            source_subfolder=source_subfolder,
                            source_revision=source_revision,
                            token=token,
                        )
                        member_size = _tar_member_size(payload_size)
                        if (
                            shard_samples
                            and shard_size + member_size > max_shard_size_bytes
                        ):
                            break

                        caption, sidecar_json = _metadata_from_sidecar(
                            sample.sidecar,
                            source_repo=source_repo,
                            source_subfolder=source_subfolder,
                            source_revision=source_revision,
                            token=token,
                        )
                        _add_to_tar(
                            archive,
                            sample,
                            payload_size,
                            source_repo=source_repo,
                            source_subfolder=source_subfolder,
                            source_revision=source_revision,
                            token=token,
                        )
                        if caption is not None:
                            captions[sample.path] = caption
                            state.captioned_samples += 1
                        else:
                            state.uncaptioned_samples += 1
                        if sidecar_json is not None:
                            json_metadata[sample.path] = sidecar_json
                        state.next_sample_index += 1
                        state.bytes_sharded += payload_size
                        shard_samples += 1
                        shard_size += member_size
                        progress.update(1)

                if shard_samples == 0:
                    raise RuntimeError(
                        f"failed to add sample at position {start_index}"
                    )

                extractor.extract_metadata(
                    source=str(staging),
                    destination=str(staging),
                    max_workers=1,
                    include_image_geometry=include_image_geometry,
                )
                _apply_sidecar_metadata(metadata_path, captions, json_metadata)
                state.next_shard_index += 1
                state.status = (
                    "complete" if state.next_sample_index == len(samples) else "running"
                )
                _write_state(state_path, state)

                tar_repo_path = _repo_path(output_prefix, tar_path.name)
                metadata_repo_path = _repo_path(output_prefix, metadata_path.name)
                if push_to_hub:
                    assert api is not None and CommitOperationAdd is not None
                    api.create_commit(
                        push_to_hub,
                        operations=[
                            CommitOperationAdd(tar_repo_path, str(tar_path)),
                            CommitOperationAdd(metadata_repo_path, str(metadata_path)),
                            CommitOperationAdd(state_repo_path, str(state_path)),
                        ],
                        commit_message=(
                            f"Add optimized webshart shard {state.next_shard_index - 1:05d}"
                        ),
                        repo_type="dataset",
                        revision=target_revision,
                        token=token,
                    )

                if destination_path is not None:
                    output_dir = destination_path / output_prefix
                    shutil.move(str(tar_path), output_dir / tar_path.name)
                    shutil.move(str(metadata_path), output_dir / metadata_path.name)
                    shutil.copy2(state_path, output_dir / STATE_FILENAME)

                for path in (tar_path, metadata_path, state_path):
                    path.unlink(missing_ok=True)
                shards_created += 1
        finally:
            progress.close()

    return {**asdict(state), "shards_created": shards_created, "resumed": resumed}
