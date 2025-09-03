# test_range_iteration.py

import pytest
from webshart import TarDataLoader, DiscoveredDataset, DatasetDiscovery
import tempfile
import json
import os
from pathlib import Path


@pytest.fixture
def test_dataset_dir():
    """Create a temporary dataset directory with tar files and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock tar files and metadata
        for shard_idx in range(3):
            # Create empty tar file
            tar_path = Path(tmpdir) / f"shard-{shard_idx:04d}.tar"
            tar_path.write_bytes(b"mock tar data")

            # Create metadata JSON
            files = []
            for file_idx in range(100):
                files.append(
                    {
                        "path": f"shard{shard_idx}_file{file_idx}.jpg",
                        "offset": file_idx * 1000,
                        "length": 1000,
                        "width": 512,
                        "height": 512,
                        "aspect": 1.0,
                    }
                )

            metadata = {"version": 2, "filesize": 100000, "files": files}

            json_path = Path(tmpdir) / f"shard-{shard_idx:04d}.json"
            json_path.write_text(json.dumps(metadata))

        yield str(tmpdir)


@pytest.fixture
def discovered_dataset(test_dataset_dir):
    """Create a discovered dataset from test directory."""
    discovery = DatasetDiscovery()
    return discovery.discover_local(test_dataset_dir)


@pytest.fixture
def mock_tar_entry():
    """Create a mock TarFileEntry for testing."""
    from types import SimpleNamespace

    entry = SimpleNamespace()
    entry.path = "test.jpg"
    entry.offset = 0
    entry.size = 1000
    entry.data = b"test data"
    entry.width = 512
    entry.height = 512
    entry.aspect = 1.0
    entry.job_id = "shard0000_file000000"
    entry.metadata = {
        "path": "test.jpg",
        "offset": 0,
        "size": 1000,
        "width": 512,
        "height": 512,
        "aspect": 1.0,
    }
    return entry


class TestRangeBasedIteration:
    """Test range-based iteration functionality."""

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "set_ranges"),
        reason="set_ranges not implemented yet",
    )
    def test_set_ranges_single(self, discovered_dataset):
        """Test setting a single range."""
        loader = TarDataLoader(discovered_dataset)
        loader.set_ranges([(10, 20)])

        # Collect all entries
        entries = list(loader)

        assert len(entries) == 10
        # Check that entries are from the expected range

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "set_ranges"),
        reason="set_ranges not implemented yet",
    )
    def test_set_ranges_multiple(self, discovered_dataset):
        """Test setting multiple non-overlapping ranges."""
        loader = TarDataLoader(discovered_dataset)
        loader.set_ranges([(10, 20), (50, 60), (100, 110)])

        entries = list(loader)

        assert len(entries) == 30  # 10 + 10 + 10

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "set_ranges"),
        reason="set_ranges not implemented yet",
    )
    def test_set_ranges_validation(self, discovered_dataset):
        """Test range validation."""
        loader = TarDataLoader(discovered_dataset)

        # Invalid range (start >= end)
        with pytest.raises(ValueError, match="Invalid range"):
            loader.set_ranges([(20, 10)])

        # Empty range
        with pytest.raises(ValueError, match="Invalid range"):
            loader.set_ranges([(10, 10)])

    def test_iter_range(self, discovered_dataset):
        """Test iter_range method."""
        loader = TarDataLoader(discovered_dataset)

        # Test normal range
        entries = list(loader.iter_range(start=10, end=20))
        assert len(entries) == 10

        # Test invalid range
        with pytest.raises(ValueError, match="start must be less than end"):
            list(loader.iter_range(start=20, end=10))

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "iter_range"),
        reason="iter_range not implemented yet",
    )
    def test_range_boundaries_across_shards(self, discovered_dataset):
        """Test ranges that cross shard boundaries."""
        loader = TarDataLoader(discovered_dataset)

        # Range spans from shard 0 (file 90) to shard 1 (file 10)
        entries = list(loader.iter_range(start=90, end=110))

        assert len(entries) == 20


class TestRichMetadata:
    """Test enriched metadata in TarFileEntry."""

    def test_current_entry_structure(self, discovered_dataset):
        """Test the current structure of entries returned by the loader."""
        loader = TarDataLoader(discovered_dataset, load_file_data=False)

        # Get first entry
        entry = next(loader)

        # Test current attributes
        assert hasattr(entry, "path")
        assert hasattr(entry, "offset")
        assert hasattr(entry, "size")
        assert hasattr(entry, "data")

        # Document what new attributes we need
        new_attrs_needed = ["width", "height", "aspect", "metadata", "job_id"]

        for attr in new_attrs_needed:
            if not hasattr(entry, attr):
                print(f"Need to add: {attr}")

    def test_metadata_fields(self, discovered_dataset):
        """Test that all metadata fields are accessible."""
        loader = TarDataLoader(discovered_dataset)
        entry = next(loader)

        # Check new fields
        assert hasattr(entry, "width")
        assert hasattr(entry, "height")
        assert hasattr(entry, "aspect")
        assert hasattr(entry, "metadata")
        assert hasattr(entry, "job_id")

    def test_metadata_dict(self, discovered_dataset):
        """Test metadata dictionary property."""
        loader = TarDataLoader(discovered_dataset)
        entry = next(loader)

        metadata = entry.metadata
        assert isinstance(metadata, dict)
        assert "path" in metadata
        assert "offset" in metadata
        assert "size" in metadata

    def test_job_id_format(self, discovered_dataset):
        """Test job_id formatting."""
        loader = TarDataLoader(discovered_dataset)
        entry = next(loader)

        # Should have format like "shard0000_file000000"
        assert entry.job_id.startswith("shard")
        assert "_file" in entry.job_id


class TestIntegration:
    """Integration tests combining range iteration and metadata."""

    def test_basic_iteration(self, discovered_dataset):
        """Test that basic iteration still works."""
        loader = TarDataLoader(discovered_dataset, load_file_data=False)

        # Should be able to iterate
        entries = []
        for i, entry in enumerate(loader):
            entries.append(entry)
            if i >= 10:  # Just get first 10
                break

        assert len(entries) == 11
        assert all(e.path for e in entries)

    def test_skip_functionality(self, discovered_dataset):
        """Test skip functionality works."""
        loader = TarDataLoader(discovered_dataset, load_file_data=False)

        # Skip to file 150 (should be in shard 1)
        loader.skip(150)

        entry = next(loader)
        assert entry is not None


@pytest.mark.parametrize(
    "batch_size,expected_batches",
    [
        (10, 30),  # 300 files / 10 per batch
        (50, 6),  # 300 files / 50 per batch
        (100, 3),  # 300 files / 100 per batch
    ],
)
def test_batch_iteration_still_works(discovered_dataset, batch_size, expected_batches):
    """Test that existing batch functionality isn't broken."""
    loader = TarDataLoader(
        discovered_dataset, load_file_data=False, batch_size=batch_size
    )

    batches = list(loader.iter_batches())
    assert len(batches) == expected_batches


# Alternative test approach using mocks at a different level
class TestWithMocking:
    """Test using mocking at the Rust binding level."""

    @pytest.fixture
    def mock_loader(self, monkeypatch):
        """Create a loader with mocked internals."""
        from unittest.mock import Mock, MagicMock

        # Create a mock loader that behaves like TarDataLoader
        mock = MagicMock()
        mock.num_shards = 3
        mock.current_shard_index = 0
        mock.current_file_index = 0

        # Mock the iteration
        mock.__iter__ = Mock(return_value=mock)
        mock.__next__ = Mock(side_effect=StopIteration)

        return mock

    def test_mock_approach(self, mock_loader):
        """Example of testing with mocks."""
        assert mock_loader.num_shards == 3
