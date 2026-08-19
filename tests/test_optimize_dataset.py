"""Tests for rolling loose-dataset optimization."""

from pathlib import Path
import json
import tarfile

import webshart
import webshart.optimize as optimize_module


def _write_loose_pairs(root: Path, count: int = 5) -> None:
    for index in range(count):
        folder = root / "nested" / f"group-{index % 2}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"sample-{index}.jpg").write_bytes(f"payload-{index}".encode("utf-8"))
        (folder / f"sample-{index}.txt").write_text(
            f" caption {index}\n", encoding="utf-8"
        )


def _write_legacy_tar(root: Path, count: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "train_0000.tar", "w") as archive:
        for index in range(count):
            payload = root / f"caption_{index}.jpg"
            payload.write_bytes(f"legacy-{index}".encode())
            archive.add(payload, arcname=f"./caption_{index}.jpg")
            payload.unlink()


def test_optimize_dataset_repackages_legacy_tars_and_resumes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_legacy_tar(source)

    first = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
        max_shards=1,
    )
    assert first["input_layout"] == "legacy_tar"
    assert first["status"] == "running"
    assert first["next_sample_index"] == 1
    assert first["next_source_archive_index"] == 0
    assert first["next_source_member_offset"] > 0

    second = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
    )
    assert second["status"] == "complete"
    assert second["resumed"] is True
    assert second["next_sample_index"] == 3
    assert second["next_source_archive_index"] == 1
    assert second["captioned_samples"] == 3

    output = destination / "webshart"
    with tarfile.open(output / "shard-00000.tar") as archive:
        assert archive.getnames() == ["train_0000/caption_0.jpg"]
    metadata = json.loads((output / "shard-00000.json").read_text())
    entry = metadata["files"]["train_0000/caption_0.jpg"]
    assert entry["captions"] == "caption 0"

    state_text = (output / ".webshart-optimize-state.json").read_text()
    assert str(source.resolve()) not in state_text
    assert str(destination.resolve()) not in state_text


def test_optimize_dataset_skips_truncated_legacy_tar_tail(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_legacy_tar(source, count=2)
    archive_path = source / "train_0000.tar"
    with tarfile.open(archive_path) as archive:
        last = archive.getmembers()[-1]
        truncated_size = last.offset_data + last.size // 2
    with archive_path.open("r+b") as handle:
        handle.truncate(truncated_size)

    result = webshart.optimize_dataset(
        source,
        destination=destination,
        include_image_geometry=False,
    )

    assert result["status"] == "complete"
    assert result["next_sample_index"] == 1
    with tarfile.open(destination / "webshart" / "shard-00000.tar") as archive:
        assert archive.getnames() == ["train_0000/caption_0.jpg"]


def test_optimize_dataset_shards_embeds_captions_and_resumes_locally(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_loose_pairs(source)

    first = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
        max_shards=2,
    )

    assert first["status"] == "running"
    assert first["next_sample_index"] == 2
    assert first["next_shard_index"] == 2
    assert first["shards_created"] == 2

    second = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
    )

    assert second["resumed"] is True
    assert second["status"] == "complete"
    assert second["next_sample_index"] == 5
    assert second["captioned_samples"] == 5
    assert second["uncaptioned_samples"] == 0

    output = destination / "webshart"
    assert len(list(output.glob("shard-*.tar"))) == 5
    assert len(list(output.glob("shard-*.json"))) == 5
    with tarfile.open(output / "shard-00000.tar") as archive:
        assert archive.getnames() == ["nested/group-0/sample-0.jpg"]
    metadata = json.loads((output / "shard-00000.json").read_text())
    entry = metadata["files"]["nested/group-0/sample-0.jpg"]
    assert entry["captions"] == "caption 0"
    assert not any(path.endswith(".txt") for path in metadata["files"])

    state_text = (output / ".webshart-optimize-state.json").read_text()
    state = json.loads(state_text)
    assert state["source"] == {"kind": "local", "subfolder": ""}
    assert str(source.resolve()) not in state_text
    assert str(destination.resolve()) not in state_text

    completed = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
    )
    assert completed["status"] == "complete"
    assert completed["shards_created"] == 0


def test_optimize_dataset_uploads_each_shard_with_resume_state(tmp_path, monkeypatch):
    source = tmp_path / "source"
    _write_loose_pairs(source, count=2)
    uploaded_state = tmp_path / "uploaded-state.json"
    commits = []

    class FakeOperation:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = Path(path_or_fileobj)

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, *args, **kwargs):
            return None

        def file_exists(self, *args, **kwargs):
            return uploaded_state.is_file()

        def create_commit(self, repo_id, operations, **kwargs):
            snapshot = {}
            for operation in operations:
                snapshot[operation.path_in_repo] = (
                    operation.path_or_fileobj.read_bytes()
                )
            uploaded_state.write_bytes(
                snapshot["converted/.webshart-optimize-state.json"]
            )
            commits.append((repo_id, snapshot, kwargs))
            return "commit"

    def fake_download(*args, **kwargs):
        return str(uploaded_state)

    def unused_url(*args, **kwargs):
        raise AssertionError("local source should not construct a Hub download URL")

    monkeypatch.setattr(
        optimize_module,
        "_require_hub",
        lambda: (FakeApi, FakeOperation, fake_download, unused_url, None),
    )

    first = webshart.optimize_dataset(
        source,
        push_to_hub="owner/target",
        output_prefix="converted",
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
        max_shards=1,
        hf_token="token",
    )
    second = webshart.optimize_dataset(
        source,
        push_to_hub="owner/target",
        output_prefix="converted",
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
        hf_token="token",
    )

    assert first["status"] == "running"
    assert second["status"] == "complete"
    assert second["resumed"] is True
    assert len(commits) == 2
    for index, (repo_id, files, kwargs) in enumerate(commits):
        assert repo_id == "owner/target"
        assert set(files) == {
            f"converted/shard-{index:05d}.tar",
            f"converted/shard-{index:05d}.json",
            "converted/.webshart-optimize-state.json",
        }
        assert kwargs["repo_type"] == "dataset"

    state_text = uploaded_state.read_text()
    assert str(source.resolve()) not in state_text
    assert json.loads(state_text)["next_sample_index"] == 2


def test_optimize_dataset_rejects_changed_manifest_on_resume(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_loose_pairs(source, count=2)
    webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
        max_shards=1,
    )
    (source / "nested" / "group-0" / "new.jpg").write_bytes(b"new")

    try:
        webshart.optimize_dataset(
            source,
            destination=destination,
            max_shard_size_bytes=1_500,
            include_image_geometry=False,
        )
    except ValueError as error:
        assert "manifest_sha256" in str(error)
    else:
        raise AssertionError("changed source manifest should reject stale resume state")


def test_optimize_dataset_coalesces_json_sidecar_metadata(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    source.mkdir()
    (source / "sample.webp").write_bytes(b"payload")
    sidecar = {"prompt": "json caption", "score": 0.9, "tags": ["one", "two"]}
    (source / "sample.json").write_text(json.dumps(sidecar), encoding="utf-8")

    webshart.optimize_dataset(
        source,
        destination=destination,
        include_image_geometry=False,
    )

    metadata = json.loads((destination / "webshart" / "shard-00000.json").read_text())
    entry = metadata["files"]["sample.webp"]
    assert entry["captions"] == "json caption"
    assert entry["json_metadata"] == sidecar
