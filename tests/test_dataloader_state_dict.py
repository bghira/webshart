import pytest
import webshart
import tempfile
import json
import os
import tarfile
from pathlib import Path
import io


@pytest.fixture
def mock_dataset_dir():
    """Create a temporary directory with proper webshart-compatible dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock tar files with proper structure
        for i in range(3):
            shard_name = f"shard_{i:04d}"
            tar_path = Path(tmpdir) / f"{shard_name}.tar"

            # Create a proper tar file
            with tarfile.open(tar_path, "w") as tar:
                # Add some dummy files
                for j in range(10):
                    filename = f"{shard_name}_{j:06d}.jpg"
                    # Create dummy image data
                    data = f"FAKE_JPEG_DATA_SHARD{i}_FILE{j}".encode() * 100

                    # Create tarinfo
                    info = tarfile.TarInfo(name=filename)
                    info.size = len(data)

                    # Add to tar
                    tar.addfile(info, io.BytesIO(data))

            # Create metadata JSON that matches the tar structure
            metadata = {"filesize": os.path.getsize(tar_path), "files": {}}

            # Read the tar to get actual offsets
            with tarfile.open(tar_path, "r") as tar:
                offset = 0
                for member in tar:
                    if member.isfile():
                        # In tar files, the header is 512 bytes, followed by data rounded up to 512 bytes
                        header_size = 512
                        data_blocks = (member.size + 511) // 512
                        data_size = data_blocks * 512

                        metadata["files"][member.name] = {
                            "offset": offset + header_size,  # Offset to actual data
                            "length": member.size,  # Actual file size
                        }
                        offset += header_size + data_size

            json_path = Path(tmpdir) / f"{shard_name}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)

        yield tmpdir


@pytest.fixture
def discovered_dataset(mock_dataset_dir):
    """Create a discovered dataset from mock directory."""
    return webshart.discover_dataset(mock_dataset_dir)


class TestTarDataLoaderStateDict:
    """Test state dict functionality for resumable pipelines."""

    def test_state_dict_basic(self, discovered_dataset):
        """Test basic state dict saving."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Read a few files
        for _ in range(3):
            next(loader)

        # Get state
        state = loader.state_dict()

        # Check state contents
        assert isinstance(state, dict)
        assert state["current_shard"] == 0
        assert state["current_file_index"] == 3
        assert state["buffer_size"] == 5
        assert state["version"] == 4
        assert "source" in state
        assert "num_shards" in state

    def test_load_state_dict(self, discovered_dataset):
        """Test loading state dict."""
        loader1 = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Read some files and switch shards
        for _ in range(15):  # Read past first shard
            next(loader1)

        # Save state
        state = loader1.state_dict()

        # Create new loader and load state
        loader2 = webshart.TarDataLoader(discovered_dataset)
        loader2.load_state_dict(state)

        # Should continue from same position
        entry = next(loader2)
        assert "shard_0001" in entry.path
        assert "000005" in entry.path  # Should be 6th file in shard 1

    def test_from_state_dict(self, discovered_dataset):
        """Test creating loader from state dict."""
        loader1 = webshart.TarDataLoader(
            discovered_dataset, buffer_size=10, load_file_data=False, max_file_size=1000
        )

        # Read to middle of dataset
        for _ in range(17):
            next(loader1)

        state = loader1.state_dict()

        # Create new loader from state
        loader2 = webshart.TarDataLoader.from_state_dict(state, discovered_dataset)

        # Check configuration was restored
        assert loader2.buffer_size == 10
        assert not loader2.load_file_data
        assert loader2.max_file_size == 1000

        # Check position was restored
        entry = next(loader2)
        assert "shard_0001" in entry.path
        assert "000007" in entry.path

    def test_from_state_dict_with_source(self, mock_dataset_dir):
        """Test creating loader from state dict with source in state."""
        loader1 = webshart.TarDataLoader(mock_dataset_dir, buffer_size=8)

        # Read some files
        for _ in range(5):
            next(loader1)

        state = loader1.state_dict()

        # Create new loader without providing dataset
        loader2 = webshart.TarDataLoader.from_state_dict(state)

        # Should work and continue from same position
        entry = next(loader2)
        assert "000005" in entry.path

    def test_state_summary(self, discovered_dataset):
        """Test getting state summary."""
        loader = webshart.TarDataLoader(discovered_dataset)

        # Initial state
        summary = loader.get_state_summary()
        assert summary["current_shard"] == 0
        assert summary["total_shards"] == 3
        assert summary["files_processed"] == 0
        assert summary["total_files"] == 30
        assert summary["progress_percent"] == 0.0

        # Read some files
        for _ in range(15):
            next(loader)

        # Updated state
        summary = loader.get_state_summary()
        assert summary["current_shard"] == 1
        assert summary["files_processed"] == 15
        assert summary["progress_percent"] == 50.0

    def test_state_persistence_with_skip(self, discovered_dataset):
        """Test state persistence with skip operations."""
        loader1 = webshart.TarDataLoader(discovered_dataset)

        # Skip to middle of first shard
        loader1.skip(5)
        next(loader1)  # Read one file

        state = loader1.state_dict()

        # Create new loader from state
        loader2 = webshart.TarDataLoader.from_state_dict(state, discovered_dataset)

        # Should continue from correct position
        entry = next(loader2)
        assert "000006" in entry.path

    def test_state_persistence_with_shard_switch(self, discovered_dataset):
        """Test state persistence after shard switching."""
        loader1 = webshart.TarDataLoader(discovered_dataset)

        # Switch to shard 2 with cursor
        loader1.shard(shard_idx=2, cursor_idx=7)
        next(loader1)

        state = loader1.state_dict()

        # Create new loader
        loader2 = webshart.TarDataLoader.from_state_dict(state, discovered_dataset)

        # Should be at correct position
        entry = next(loader2)
        assert "shard_0002" in entry.path
        assert "000008" in entry.path

    def test_state_dict_version_check(self, discovered_dataset):
        """Test version checking in state dict."""
        loader = webshart.TarDataLoader(discovered_dataset)
        state = loader.state_dict()

        # Modify version
        state["version"] = 999

        # Should raise error
        with pytest.raises(Exception) as exc_info:
            loader.load_state_dict(state)

        assert "Unsupported state dict version" in str(exc_info.value)

    def test_state_dict_with_buffer_position(self, discovered_dataset):
        """Test that buffer position is saved/restored correctly."""
        loader1 = webshart.TarDataLoader(discovered_dataset, buffer_size=10)

        # Read exactly 5 files (half a buffer)
        for _ in range(5):
            next(loader1)

        state = loader1.state_dict()
        assert state["buffer_position"] == 5

        # Create new loader
        loader2 = webshart.TarDataLoader.from_state_dict(state, discovered_dataset)

        # Should continue without re-reading buffer
        entry = next(loader2)
        assert "000005" in entry.path

    def test_state_dict_partial_load(self, discovered_dataset):
        """Test partial state dict loading (only some fields)."""
        loader = webshart.TarDataLoader(
            discovered_dataset, buffer_size=20, load_file_data=False
        )

        # Create partial state dict
        partial_state = {
            "current_shard": 1,
            "current_file_index": 5,
            "version": 1,
            # Missing other fields
        }

        # Should load without error
        loader.load_state_dict(partial_state)

        # Position should be updated
        assert loader.current_shard_index == 1
        # But config should remain unchanged
        assert loader.buffer_size == 20
        assert not loader.load_file_data

    def test_from_state_dict_no_source_error(self, discovered_dataset):
        """Test error when creating from state dict without source."""
        loader = webshart.TarDataLoader(discovered_dataset)
        state = loader.state_dict()

        # Remove source from state
        del state["source"]

        # Should raise error without dataset_or_path
        with pytest.raises(Exception) as exc_info:
            webshart.TarDataLoader.from_state_dict(state)

        assert "No dataset_or_path provided" in str(exc_info.value)

    def test_resumable_iteration_pattern(self, discovered_dataset):
        """Test a typical resumable iteration pattern."""
        import pickle

        # Simulate checkpoint file
        checkpoint_data = None
        files_processed = []

        # First run - process some files
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        for i, entry in enumerate(loader):
            files_processed.append(entry.path)

            # Simulate checkpoint every 7 files
            if (i + 1) % 7 == 0:
                checkpoint_data = pickle.dumps(loader.state_dict())
                if i >= 13:  # Stop after 2 checkpoints
                    break

        assert len(files_processed) == 14

        # Resume from checkpoint
        state = pickle.loads(checkpoint_data)
        resumed_loader = webshart.TarDataLoader.from_state_dict(
            state, discovered_dataset
        )

        # Continue processing
        resumed_files = []
        for i, entry in enumerate(resumed_loader):
            resumed_files.append(entry.path)
            if i >= 5:  # Read 6 more files
                break

        # Should not have duplicates
        assert resumed_files[0] not in files_processed

        # The 15th file (index 14) should be the 5th file in shard 1
        assert "shard_0001_000004" in resumed_files[0]
        # Alternative check - just check it's the 5th file in its shard
        # assert "000004" in resumed_files[0]
