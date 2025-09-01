# test_tar_dataloader.py
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


class TestTarDataLoader:
    """Test the TarDataLoader functionality."""

    def test_basic_iteration(self, discovered_dataset):
        """Test basic iteration through files."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        files_read = []
        count = 0
        for entry in loader:
            files_read.append(entry.path)
            count += 1
            if count >= 10:  # Read first 10 files
                break

        assert len(files_read) == 10
        # Files should start with shard_0000
        assert all(f.startswith("shard_0000") for f in files_read)

    def test_skip_within_shard(self, discovered_dataset):
        """Test skipping files within a shard."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Skip first 5 files
        loader.skip(5)

        # Next file should be the 6th file (index 5)
        entry = next(loader)
        assert "000005" in entry.path  # Should be file 5 (0-indexed)

    def test_skip_to_end_of_shard(self, discovered_dataset):
        """Test skipping to near end of shard."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Skip to file 8 (9th file)
        loader.skip(8)

        # Should be able to read files 8 and 9
        entries = []
        for entry in loader:
            entries.append(entry)
            if len(entries) >= 2:
                break

        assert len(entries) == 2
        assert "000008" in entries[0].path
        assert "000009" in entries[1].path

    def test_shard_switch_by_index(self, discovered_dataset):
        """Test switching shards by index."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Switch to shard 1
        loader.shard(shard_idx=1)

        # Next file should be from shard 1
        entry = next(loader)
        assert "shard_0001" in entry.path
        assert "000000" in entry.path  # First file in shard

    def test_shard_switch_by_filename(self, discovered_dataset):
        """Test switching shards by filename."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Switch to shard 2 by filename
        loader.shard(filename="shard_0002")

        # Next file should be from shard 2
        entry = next(loader)
        assert "shard_0002" in entry.path

    def test_shard_switch_with_cursor(self, discovered_dataset):
        """Test switching shards with cursor position."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Switch to shard 1, starting at file 3
        loader.shard(shard_idx=1, cursor_idx=3)

        # Next file should be file 3 from shard 1
        entry = next(loader)
        assert "shard_0001" in entry.path
        assert "000003" in entry.path

    def test_reset(self, discovered_dataset):
        """Test resetting the loader."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Read a few files
        for _ in range(5):
            next(loader)

        # Reset
        loader.reset()

        # Should start from beginning again
        entry = next(loader)
        assert "shard_0000" in entry.path
        assert "000000" in entry.path

    def test_buffer_size_property(self, discovered_dataset):
        """Test buffer size property."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=10)

        assert loader.buffer_size == 10

        # Change buffer size
        loader.buffer_size = 20
        assert loader.buffer_size == 20

    def test_num_shards_property(self, discovered_dataset):
        """Test num_shards property."""
        loader = webshart.TarDataLoader(discovered_dataset)

        assert loader.num_shards == 3  # We created 3 shards

    def test_current_shard_index_property(self, discovered_dataset):
        """Test current_shard_index property."""
        loader = webshart.TarDataLoader(discovered_dataset)

        # Initially should be 0
        assert loader.current_shard_index == 0

        # After switching shards
        loader.shard(shard_idx=2)
        assert loader.current_shard_index == 2

    def test_get_metadata(self, discovered_dataset):
        """Test getting metadata for a shard."""
        loader = webshart.TarDataLoader(discovered_dataset)

        metadata = loader.get_metadata(0)

        # Should be a dict with file information
        assert isinstance(metadata, dict)
        assert len(metadata) == 10  # 10 files in shard

        # Check a specific file's metadata
        first_file = next(iter(metadata.keys()))
        assert "offset" in metadata[first_file]
        assert "length" in metadata[first_file]

    def test_file_data_loading(self, discovered_dataset):
        """Test that file data is loaded correctly."""
        loader = webshart.TarDataLoader(discovered_dataset, load_file_data=True)

        entry = next(loader)

        # Check that data was loaded
        assert len(entry.data) > 0
        assert b"FAKE_JPEG_DATA_SHARD0_FILE0" in entry.data

    def test_file_data_not_loaded(self, discovered_dataset):
        """Test skipping file data loading."""
        loader = webshart.TarDataLoader(discovered_dataset, load_file_data=False)

        entry = next(loader)

        # Data should be empty
        assert len(entry.data) == 0

    def test_iteration_across_shards(self, discovered_dataset):
        """Test that iteration automatically moves across shards."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Collect all files
        all_files = []
        for entry in loader:
            all_files.append(entry.path)

        # Should have all 30 files (3 shards × 10 files)
        assert len(all_files) == 30

        # Check that we got files from all shards
        shard_0_files = [f for f in all_files if "shard_0000" in f]
        shard_1_files = [f for f in all_files if "shard_0001" in f]
        shard_2_files = [f for f in all_files if "shard_0002" in f]

        assert len(shard_0_files) == 10
        assert len(shard_1_files) == 10
        assert len(shard_2_files) == 10

    def test_entry_properties(self, discovered_dataset):
        """Test TarFileEntry properties."""
        loader = webshart.TarDataLoader(discovered_dataset)

        entry = next(loader)

        # Check properties
        assert isinstance(entry.path, str)
        assert isinstance(entry.offset, int)
        assert isinstance(entry.size, int)
        assert isinstance(entry.data, bytes)

        # Size should match data length (when loaded)
        assert entry.size == len(entry.data)


class TestTarDataLoaderErrors:
    """Test error handling in TarDataLoader."""

    def test_skip_out_of_bounds(self, discovered_dataset):
        """Test skipping beyond shard bounds."""
        loader = webshart.TarDataLoader(discovered_dataset)

        # Try to skip beyond the 10 files in shard 0
        with pytest.raises(Exception) as exc_info:
            loader.skip(100)

        assert "out of range" in str(exc_info.value).lower()

    def test_invalid_shard_index(self, discovered_dataset):
        """Test switching to invalid shard."""
        loader = webshart.TarDataLoader(discovered_dataset)

        with pytest.raises(Exception) as exc_info:
            loader.shard(shard_idx=999)

        assert "out of range" in str(exc_info.value).lower()

    def test_invalid_shard_filename(self, discovered_dataset):
        """Test switching to non-existent shard by filename."""
        loader = webshart.TarDataLoader(discovered_dataset)

        with pytest.raises(Exception) as exc_info:
            loader.shard(filename="nonexistent_shard")

        assert "not found" in str(exc_info.value).lower()

    def test_no_shard_specified(self, discovered_dataset):
        """Test shard() without arguments."""
        loader = webshart.TarDataLoader(discovered_dataset)

        with pytest.raises(Exception) as exc_info:
            loader.shard()

        assert "must be provided" in str(exc_info.value).lower()


class TestDataLoaderIntegration:
    """Integration tests with other webshart features."""

    def test_loader_with_string_path(self, mock_dataset_dir):
        """Test creating loader with string path."""
        loader = webshart.TarDataLoader(mock_dataset_dir)

        # Should work the same as with discovered dataset
        entry = next(loader)
        assert entry.path.startswith("shard_0000")

    def test_loader_preserves_hf_token(self, discovered_dataset):
        """Test that HF token is preserved when using discovered dataset."""
        # Create dataset with token
        dataset = webshart.DatasetDiscovery(hf_token="test_token").discover_local(
            discovered_dataset.name
        )

        loader = webshart.TarDataLoader(dataset)

        # The loader should have access to the token internally
        # (This is used for remote datasets)
        assert dataset.get_hf_token() == "test_token"

    def test_max_file_size_limit(self, mock_dataset_dir):
        """Test max_file_size parameter."""
        # Create a loader with very small max file size
        loader = webshart.TarDataLoader(mock_dataset_dir, max_file_size=100)

        entry = next(loader)

        # File should not be loaded due to size limit
        assert len(entry.data) == 0

    def test_entry_repr(self, discovered_dataset):
        """Test TarFileEntry __repr__ method."""
        loader = webshart.TarDataLoader(discovered_dataset)
        entry = next(loader)

        repr_str = repr(entry)
        assert "TarFileEntry" in repr_str
        assert entry.path in repr_str
        assert str(entry.offset) in repr_str
        assert str(entry.size) in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
