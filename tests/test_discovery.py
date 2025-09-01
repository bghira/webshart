"""Tests for dataset discovery functionality."""

import json
import tempfile
import pytest
from pathlib import Path
import os

import webshart


def create_test_shard_metadata(shard_num, num_files=100):
    """Create test shard metadata in the expected format."""
    files = {}
    offset = 512  # tar header

    for i in range(num_files):
        filename = f"{shard_num * 1000 + i}.webp"
        size = 1024 * (i % 10 + 1)  # Vary sizes
        files[filename] = {
            "offset": offset,
            "length": size,  # webshart expects 'length' not 'size'
            "sha256": f"deadbeef{i:08x}" * 8,
        }
        offset += size + 512  # Add tar padding

    return {
        "path": f"data-{shard_num:04d}.tar",
        "filesize": offset,
        "hash": f"hash{shard_num:04d}",
        "hash_lfs": f"lfs_hash{shard_num:04d}",
        "files": files,
    }


def test_discover_local_dataset():
    """Test discovering a local dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset structure
        for i in range(3):
            # Create JSON metadata
            metadata = create_test_shard_metadata(i, num_files=50)
            json_path = Path(tmpdir) / f"data-{i:04d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)

            # Create empty tar file (just for discovery)
            tar_path = Path(tmpdir) / f"data-{i:04d}.tar"
            tar_path.touch()

        # Discover dataset
        dataset = webshart.discover_dataset(tmpdir)

        assert dataset.name == tmpdir
        assert dataset.num_shards == 3
        assert not dataset.is_remote

        # Check shard info
        shard_info = dataset.get_shard_info(0)
        assert shard_info["name"] == "data-0000"
        assert shard_info["num_files"] == 50

        # List files in first shard
        files = dataset.list_files_in_shard(0)
        assert len(files) == 50
        assert "0.webp" in files

        # Find file location
        shard_idx, local_idx = dataset.find_file_location(75)  # File in second shard
        assert shard_idx == 1
        assert local_idx == 25


def test_discovery_no_shards():
    """Test discovery when no shards are found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Empty directory
        with pytest.raises(Exception) as exc_info:
            webshart.discover_dataset(tmpdir)
        assert "No shards found" in str(exc_info.value)


def test_discovery_missing_json():
    """Test discovery when JSON files are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tar without JSON
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Should not find any complete pairs
        with pytest.raises(Exception) as exc_info:
            webshart.discover_dataset(tmpdir)
        assert "No shards found" in str(exc_info.value)


def test_dataset_repr():
    """Test string representation of discovered dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal dataset
        metadata = create_test_shard_metadata(0, num_files=10)
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)
        repr_str = repr(dataset)

        assert "DiscoveredDataset" in repr_str
        assert "shards=1" in repr_str


def test_file_location_out_of_range():
    """Test error handling for out of range file index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal dataset
        metadata = create_test_shard_metadata(0, num_files=10)
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)

        with pytest.raises(IndexError):
            dataset.find_file_location(100)

        with pytest.raises(Exception):
            dataset.get_shard_info(10)


def test_custom_discovery_pattern():
    """Test discovery with custom shard naming pattern."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create shards with different naming pattern
        for i in range(2):
            metadata = create_test_shard_metadata(i, num_files=25)
            # Update path in metadata to match actual filename
            metadata["path"] = f"shard_{i:03d}.tar"
            json_path = Path(tmpdir) / f"shard_{i:03d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)
            tar_path = Path(tmpdir) / f"shard_{i:03d}.tar"
            tar_path.touch()

        # Default pattern should still work with shard_ prefix
        dataset = webshart.discover_dataset(tmpdir)

        assert dataset.num_shards == 2
        # Note: total_files is a property that loads metadata
        # so we check it triggers loading
        total = dataset.total_files
        assert total == 50


def test_quick_stats():
    """Test quick_stats method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal dataset
        metadata = create_test_shard_metadata(0, num_files=10)
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)

        # Local datasets don't have cached stats
        size, files = dataset.quick_stats()
        assert size is None
        assert files is None


def test_shard_reader():
    """Test opening and using a shard reader."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with proper tar file
        metadata = create_test_shard_metadata(0, num_files=5)
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)

        # Create a minimal tar file with actual data
        tar_path = Path(tmpdir) / "data-0000.tar"
        # For testing, just create a file with enough bytes
        with open(tar_path, "wb") as f:
            # Write enough data to cover the offsets in metadata
            f.write(b"\0" * 10000)

        dataset = webshart.discover_dataset(tmpdir)
        reader = dataset.open_shard(0)

        assert reader.num_files == 5
        filenames = reader.filenames()
        assert len(filenames) == 5
        assert "0.webp" in filenames
