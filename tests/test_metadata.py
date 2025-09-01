"""Tests for metadata handling in webshart."""

import json
import tempfile
import pytest
from pathlib import Path

import webshart


def create_hashmap_metadata():
    """Create metadata in HashMap format."""
    return {
        "path": "test-shard.tar",
        "filesize": 1048576,
        "hash": "abcdef123456",
        "hash_lfs": "fedcba654321",
        "files": {
            "image_001.webp": {
                "offset": 512,
                "length": 102400,
                "sha256": "deadbeef" * 8,
            },
            "image_002.webp": {
                "offset": 102912,
                "length": 98304,
                "sha256": "cafebabe" * 8,
            },
            "image_003.webp": {
                "offset": 201216,
                "length": 110592,
                "sha256": "badc0de5" * 8,
            },
        },
    }


def create_vec_metadata():
    """Create metadata in Vec/Array format (alternative format)."""
    return {
        "path": "test-shard.tar",
        "filesize": 1048576,
        "files": [
            {
                "path": "image_001.webp",
                "offset": 512,
                "length": 102400,
                "sha256": "deadbeef" * 8,
            },
            {
                "path": "image_002.webp",
                "offset": 102912,
                "length": 98304,
                "sha256": "cafebabe" * 8,
            },
            {
                "path": "image_003.webp",
                "offset": 201216,
                "length": 110592,
                "sha256": "badc0de5" * 8,
            },
        ],
    }


def test_metadata_hashmap_format():
    """Test loading metadata in HashMap format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with HashMap metadata
        metadata = create_hashmap_metadata()
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Discover and load metadata
        dataset = webshart.discover_dataset(tmpdir)

        # Force metadata loading
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 3
        assert shard_info["size"] == 1048576

        # Check file listing
        files = dataset.list_files_in_shard(0)
        assert len(files) == 3
        assert "image_001.webp" in files
        assert "image_002.webp" in files
        assert "image_003.webp" in files


def test_metadata_vec_format():
    """Test loading metadata in Vec/Array format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with Vec metadata
        metadata = create_vec_metadata()
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Discover and load metadata
        dataset = webshart.discover_dataset(tmpdir)

        # Force metadata loading
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 3
        assert shard_info["size"] == 1048576

        # Check file listing
        files = dataset.list_files_in_shard(0)
        assert len(files) == 3
        # Files should be accessible even in Vec format
        assert any("image_001" in f for f in files)


def test_lazy_metadata_loading():
    """Test that metadata is loaded lazily."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple shards
        for i in range(3):
            metadata = {
                "path": f"data-{i:04d}.tar",
                "filesize": 1048576 * (i + 1),
                "files": {
                    f"file_{j}.webp": {"offset": j * 1024, "length": 1024}
                    for j in range(10)
                },
            }
            json_path = Path(tmpdir) / f"data-{i:04d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)
            tar_path = Path(tmpdir) / f"data-{i:04d}.tar"
            tar_path.touch()

        # Discover dataset
        dataset = webshart.discover_dataset(tmpdir)

        # Quick stats should not load metadata
        size, files = dataset.quick_stats()
        assert size is None  # Local datasets don't have cached stats

        # Accessing specific shard should load only that metadata
        shard_info = dataset.get_shard_info(1)
        assert shard_info["num_files"] == 10

        # total_files should load all metadata
        total = dataset.total_files
        assert total == 30  # 3 shards * 10 files each


def test_metadata_with_missing_fields():
    """Test handling metadata with optional fields missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal metadata
        metadata = {
            "filesize": 1024,
            "files": {"test.webp": {"offset": 0, "length": 512}},
        }
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)
        shard_info = dataset.get_shard_info(0)

        assert shard_info["num_files"] == 1
        # hash/hash_lfs are optional and may not be in the info


def test_invalid_metadata_format():
    """Test error handling for invalid metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Invalid metadata (missing required fields)
        metadata = {"some_field": "value"}
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)

        # Should fail when trying to access the shard
        with pytest.raises(Exception):
            dataset.get_shard_info(0)


def test_file_info_aliases():
    """Test that metadata handles field aliases (size vs length)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use 'size' instead of 'length' (should be handled by alias)
        metadata = {
            "path": "test.tar",
            "filesize": 2048,
            "files": {
                "file1.webp": {
                    "offset": 0,
                    "size": 1024,  # Using 'size' instead of 'length'
                },
                "file2.webp": {"offset": 1024, "length": 1024},  # Using 'length'
            },
        }
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)
        files = dataset.list_files_in_shard(0)
        assert len(files) == 2


def test_batch_metadata_loading():
    """Test batch loading of metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 shards
        for i in range(5):
            metadata = create_hashmap_metadata()
            metadata["path"] = f"data-{i:04d}.tar"
            json_path = Path(tmpdir) / f"data-{i:04d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)
            tar_path = Path(tmpdir) / f"data-{i:04d}.tar"
            tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)

        # Use batch operations to load metadata
        batch_ops = webshart.BatchOperations()
        results = batch_ops.load_metadata_batch(dataset, [0, 2, 4])

        # Check all requested metadata was loaded
        assert len(results) == 3
        for result in results:
            assert result is True  # All should succeed

        # Verify metadata is actually loaded
        for idx in [0, 2, 4]:
            info = dataset.get_shard_info(idx)
            assert info["num_files"] == 3
