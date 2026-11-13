"""Upload completed capture scans in byte-triggered, bounded commit groups."""
from pathlib import Path
import hashlib
import json
import sqlite3

from .parquet import _integer, _relative_path
from .url_scan import _atomic_json


def upload_capture_queue(destination, staging_directory, *, upload_repo,
                         upload_repo_type="dataset", upload_endpoint=None,
                         upload_token=None, min_bytes=1024 ** 4, flush=False,
                         max_files_per_commit=64, num_threads=4):
    """Drain a byte-triggered batch from completed child scan directories.

    Producers use matching child names under destination and staging_directory,
    with defer_uploads=True. Only complete scans are consumed. Capture completion
    does not mean upload completion: flush the final remainder after producers exit.
    Receipts survive interruption between a remote commit and checkpoint updates.
    """
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

    _integer(min_bytes, "min_bytes")
    _integer(max_files_per_commit, "max_files_per_commit", 1)
    _integer(num_threads, "num_threads", 1)
    root, staging = Path(destination), Path(staging_directory)
    root.mkdir(parents=True, exist_ok=True)
    state_root = root / ".upload-queue"
    state_root.mkdir(exist_ok=True)
    api = HfApi(endpoint=upload_endpoint, token=upload_token)
    identity = dict(repo=upload_repo, repo_type=upload_repo_type, endpoint=api.endpoint)
    lock = sqlite3.connect(state_root / "lock.sqlite", timeout=0)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        config = state_root / "destination.json"
        if config.exists() and json.loads(config.read_text()) != identity:
            raise ValueError("Upload queue destination changed")
        _atomic_json(config, identity)

        def checkpoint(entry):
            return root / _relative_path(entry["checkpoint"])

        def image_path(entry):
            return staging / _relative_path(entry["scan"]) / _relative_path(entry["image_path"])

        def acknowledge(receipt):
            groups = {}
            for entry in receipt["entries"]:
                groups.setdefault(entry["checkpoint"], []).append(entry)
            for name, entries in groups.items():
                path = root / _relative_path(name)
                progress = json.loads(path.read_text())
                by_image = {p["image_path"]: p for p in progress["parts"]}
                for entry in entries:
                    part = by_image[entry["image_path"]]
                    if part["image_sha256"] != entry["image_sha256"]:
                        raise ValueError("Upload receipt does not match capture checkpoint")
                    part["upload_commit"] = receipt["commit"]
                _atomic_json(path, progress)
                for entry in entries:
                    image_path(entry).unlink(missing_ok=True)

        receipt_path = state_root / "receipt.json"
        if receipt_path.exists():
            acknowledge(json.loads(receipt_path.read_text()))
            receipt_path.unlink()
        active_path = state_root / "active-batch.json"
        if active_path.exists():
            entries = json.loads(active_path.read_text())
        else:
            entries = []
            for summary in sorted(root.glob("*/scan-summary.json")):
                if not json.loads(summary.read_text()).get("complete"):
                    continue
                for cp in sorted((summary.parent / "checkpoints").glob("*.json")):
                    for part in json.loads(cp.read_text())["parts"]:
                        if part.get("upload_commit"):
                            continue
                        entries.append(dict(scan=summary.parent.name,
                                            checkpoint=cp.relative_to(root).as_posix(),
                                            image_path=part["image_path"],
                                            image_size=part["image_size"],
                                            image_sha256=part["image_sha256"]))
            pending_bytes = sum(e["image_size"] for e in entries)
            if not entries or (pending_bytes < min_bytes and not flush):
                return dict(uploaded_bytes=0, uploaded_files=0, pending_bytes=pending_bytes)
            selected, size = [], 0
            for entry in entries:
                selected.append(entry)
                size += entry["image_size"]
                if not flush and size >= min_bytes:
                    break
            entries = selected
            _atomic_json(active_path, entries)
        remaining = []
        cached = {}
        for entry in entries:
            cp = entry["checkpoint"]
            if cp not in cached:
                cached[cp] = {p["image_path"]: p for p in json.loads(checkpoint(entry).read_text())["parts"]}
            if not cached[cp][entry["image_path"]].get("upload_commit"):
                remaining.append(entry)
        remote_paths = {entry.rfilename for entry in api.repo_info(
            upload_repo, repo_type=upload_repo_type).siblings} if remaining else set()
        uploaded_bytes = uploaded_files = 0
        for offset in range(0, len(remaining), max_files_per_commit):
            group = remaining[offset:offset + max_files_per_commit]
            operations = []
            for entry in group:
                path = image_path(entry)
                if not path.is_file() or path.stat().st_size != entry["image_size"]:
                    raise ValueError("Missing or changed staged capture; restore staging before upload")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != entry["image_sha256"]:
                        raise ValueError("Changed staged capture hash")
                operations.append(CommitOperationAdd(path_in_repo=entry["image_path"], path_or_fileobj=str(path)))
            marker_data = json.dumps({e["image_path"]: e["image_sha256"] for e in group}, sort_keys=True).encode()
            marker = ".webshart-upload-receipts/" + hashlib.sha256(marker_data).hexdigest() + ".json"
            if marker not in remote_paths:
                operations.append(CommitOperationAdd(path_in_repo=marker, path_or_fileobj=marker_data))
                commit = api.create_commit(repo_id=upload_repo, repo_type=upload_repo_type,
                                           operations=operations, num_threads=num_threads,
                                           commit_message=f"Upload {len(group)} captured image shards")
                commit_id = commit.oid
            else:
                existing = hf_hub_download(upload_repo, marker, repo_type=upload_repo_type,
                                           endpoint=api.endpoint, token=upload_token)
                if Path(existing).read_bytes() != marker_data:
                    raise ValueError("Remote upload receipt does not match staged batch")
                # The marker and data are written in the same atomic Hub commit.
                commit_id = "receipt:" + marker
            receipt = dict(commit=commit_id, entries=group)
            _atomic_json(receipt_path, receipt)
            acknowledge(receipt)
            receipt_path.unlink()
            uploaded_bytes += sum(e["image_size"] for e in group)
            uploaded_files += len(group)
        active_path.unlink()
        pending_bytes = 0
        for summary in root.glob("*/scan-summary.json"):
            if not json.loads(summary.read_text()).get("complete"):
                continue
            for cp in (summary.parent / "checkpoints").glob("*.json"):
                pending_bytes += sum(p["image_size"] for p in json.loads(cp.read_text())["parts"]
                                     if not p.get("upload_commit"))
        return dict(uploaded_bytes=uploaded_bytes, uploaded_files=uploaded_files,
                    pending_bytes=pending_bytes)
    finally:
        lock.close()
