"""
Tests for webshart dataset statistics methods
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from webshart import discover_dataset, DatasetDiscovery


class TestDatasetStats:
    """Test dataset statistics and info methods."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory with tar and json files."""
        # Create mock tar files
        for i in range(3):
            tar_path = tmp_path / f"shard-{i:04d}.tar"
            tar_path.write_bytes(b"mock tar content")

            # Create corresponding JSON metadata
            metadata = {
                "path": f"shard-{i:04d}.tar",
                "filesize": 1000000 * (i + 1),  # 1MB, 2MB, 3MB
                "files": {
                    f"image_{j:03d}.webp": {
                        "offset": j * 1000,
                        "length": 950,
                        "width": 1920,
                        "height": 1080,
                        "aspect": 1.7777778,
                    }
                    for j in range(10 * (i + 1))  # 10, 20, 30 files
                },
                "includes_image_geometry": True,
            }
            json_path = tmp_path / f"shard-{i:04d}.json"
            json_path.write_text(json.dumps(metadata))

        return tmp_path

    @pytest.fixture
    def local_dataset(self, mock_dataset_dir):
        """Create a discovered dataset from mock directory."""
        return discover_dataset(str(mock_dataset_dir))

    def test_get_shard_file_count(self, local_dataset):
        """Test getting file count for specific shard."""
        # Test valid indices
        assert local_dataset.get_shard_file_count(0) == 10
        assert local_dataset.get_shard_file_count(1) == 20
        assert local_dataset.get_shard_file_count(2) == 30

        # Test out of range
        with pytest.raises(IndexError) as exc_info:
            local_dataset.get_shard_file_count(3)
        assert "out of range" in str(exc_info.value)

    def test_get_total_files(self, local_dataset):
        """Test getting total file count."""
        # This loads all metadata
        total = local_dataset.total_files
        assert total == 60  # 10 + 20 + 30

    def test_stats_property(self, local_dataset):
        """Test the stats property getter."""
        stats = local_dataset.get_stats()

        assert stats["total_shards"] == 3
        assert "shard_details" in stats
        assert len(stats["shard_details"]) == 3

        # First access won't have loaded metadata
        assert stats["shard_details"][0]["metadata_loaded"] == False

        # Load a shard
        local_dataset.get_shard_file_count(0)

        # Check again
        stats = local_dataset.get_stats()
        assert stats["shard_details"][0]["metadata_loaded"] == True
        assert stats["shard_details"][0]["num_files"] == 10

    def test_detailed_stats_property(self, local_dataset):
        """Test the detailed_stats property getter."""
        stats = local_dataset.get_detailed_stats()

        # Should load all metadata
        assert stats["total_shards"] == 3
        assert stats["total_files"] == 60
        assert stats["total_size"] == 6000000  # 1MB + 2MB + 3MB
        assert stats["total_size_gb"] == pytest.approx(0.00559, rel=0.01)
        assert stats["average_files_per_shard"] == 20.0
        assert stats["min_files_in_shard"] == 10
        assert stats["max_files_in_shard"] == 30
        assert stats["from_cache"] == False

        # Check shard details
        assert len(stats["shard_details"]) == 3
        for i, shard in enumerate(stats["shard_details"]):
            assert shard["index"] == i
            assert shard["name"] == f"shard-{i:04d}"
            assert shard["num_files"] == 10 * (i + 1)

    def test_get_shard_by_name(self, local_dataset):
        """Test finding shard by name."""
        # Test valid names
        shard_info = local_dataset.get_shard_by_name("shard-0001")
        assert shard_info["name"] == "shard-0001"
        assert shard_info["num_files"] == 20

        # Test not found
        with pytest.raises(ValueError) as exc_info:
            local_dataset.get_shard_by_name("shard-9999")
        assert "not found" in str(exc_info.value)

    @patch("webshart.discover_dataset")
    def test_remote_dataset_cached_stats(self, mock_discover):
        """Test that remote datasets use cached values."""
        # Create a mock remote dataset with cached values
        mock_dataset = MagicMock()
        mock_dataset.num_shards = 100
        mock_dataset.stats = {
            "total_shards": 100,
            "total_size": 1073741824000,  # 1TB
            "total_size_gb": 1000.0,
            "total_files": 50000,
            "average_files_per_shard": 500.0,
            "from_cache": True,
            "shard_details": [],
        }

        mock_discover.return_value = mock_dataset

        dataset = discover_dataset(
            source="laion/conceptual-captions-12m-webdataset",
            metadata="webshart/conceptual-captions-12m-webdataset-metadata",
        )
        stats = dataset.get_stats()

        # Should use cached values
        assert stats["from_cache"] == True
        assert stats["total_files"] == 1100

    def test_lazy_metadata_loading(self, local_dataset):
        """Test that metadata is loaded lazily."""
        # Initially, no metadata should be loaded
        stats = local_dataset.get_stats()
        for shard in stats["shard_details"]:
            assert shard["metadata_loaded"] == False

        # Access one shard
        local_dataset.get_shard_file_count(1)

        # Only that shard should have metadata loaded
        stats = local_dataset.get_stats()
        assert stats["shard_details"][0]["metadata_loaded"] == False
        assert stats["shard_details"][1]["metadata_loaded"] == True
        assert stats["shard_details"][2]["metadata_loaded"] == False

    def test_stats_with_empty_dataset(self):
        """Test stats methods with empty dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty dataset (no shards)
            with pytest.raises(Exception) as exc_info:
                dataset = discover_dataset(tmpdir)
            assert "NoShardsFound" in str(
                exc_info.typename
            ) or "No shards found" in str(exc_info.value)

    def test_stats_consistency(self, local_dataset):
        """Test that different stats methods return consistent data."""
        # Get stats without loading metadata
        quick_stats = local_dataset.get_stats()

        # Get detailed stats (loads all metadata)
        detailed = local_dataset.get_detailed_stats()

        # Basic counts should match
        assert quick_stats["total_shards"] == detailed["total_shards"]

        updated_stats = local_dataset.get_stats()
        assert updated_stats["shard_details"][0]["metadata_loaded"] == True
        assert updated_stats["shard_details"][0]["num_files"] == 10


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_corrupted_metadata(self, tmp_path):
        """Test handling of corrupted metadata files."""
        # Create tar file
        tar_path = tmp_path / "shard-0000.tar"
        tar_path.write_bytes(b"mock tar content")

        # Create corrupted JSON
        json_path = tmp_path / "shard-0000.json"
        json_path.write_text("{ invalid json")

        # Should handle gracefully
        dataset = discover_dataset(str(tmp_path))

    def test_missing_json_files(self, tmp_path):
        """Test handling when JSON files are missing."""
        # Create only tar files
        for i in range(3):
            tar_path = tmp_path / f"shard-{i:04d}.tar"
            tar_path.write_bytes(b"mock tar content")

        # Should not discover any shards
        with pytest.raises(Exception) as exc_info:
            dataset = discover_dataset(str(tmp_path))
        assert "NoShardsFound" in str(exc_info.typename) or "No shards found" in str(
            exc_info.value
        )

    def test_large_dataset_performance(self, tmp_path):
        """Test performance with many shards."""
        # Create a dataset with many shards
        num_shards = 100

        for i in range(num_shards):
            tar_path = tmp_path / f"shard-{i:04d}.tar"
            tar_path.write_bytes(b"mock")

            metadata = {
                "path": f"shard-{i:04d}.tar",
                "filesize": 1000000,
                "files": {
                    f"file_{j}.dat": {"offset": j * 100, "length": 90}
                    for j in range(10)
                },
            }
            json_path = tmp_path / f"shard-{i:04d}.json"
            json_path.write_text(json.dumps(metadata))

        dataset = discover_dataset(str(tmp_path))

        # Quick stats should be instant
        import time

        start = time.time()
        stats = dataset.get_stats()
        quick_time = time.time() - start

        assert stats["total_shards"] == num_shards
        assert quick_time < 0.1  # Should be very fast

        # Detailed stats will be slower (loads all metadata)
        start = time.time()
        detailed = dataset.get_detailed_stats()
        detailed_time = time.time() - start

        assert detailed["total_files"] == num_shards * 10
        assert detailed_time > quick_time  # Should take longer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
