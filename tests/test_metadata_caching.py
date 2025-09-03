import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch, MagicMock

# These would be the actual Python imports from webshart
# from webshart import DatasetDiscovery, DiscoveredDataset


class TestMetadataCachingPythonAPI:
    """Test suite for metadata caching through Python API"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory with mock dataset files"""
        temp_dir = tempfile.mkdtemp()

        # Create mock shard files
        for i in range(5):
            # Create tar file
            tar_path = os.path.join(temp_dir, f"shard-{i:04d}.tar")
            with open(tar_path, "wb") as f:
                f.write(b"mock tar content")

            # Create metadata json
            metadata = {
                "filesize": 1024 * (i + 1),
                "files": {
                    f"image_{i}_001.webp": {"offset": 0, "length": 512},
                    f"image_{i}_002.webp": {"offset": 512, "length": 512},
                },
            }
            json_path = os.path.join(temp_dir, f"shard-{i:04d}.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_enable_metadata_cache_basic(self, temp_cache_dir, temp_dataset_dir):
        """Test basic cache enabling functionality"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Should work without cache
        info = dataset.get_shard_info(0)
        assert info["name"] == "shard-0000"

        # Enable cache
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=2)

        # Cache should be enabled
        stats = dataset.get_cache_stats()
        assert stats["cache_enabled"] is True
        assert "cache_location" in stats
        assert stats["cached_shards"] >= 0  # May have pre-loaded some

    def test_cache_stats_accuracy(self, temp_cache_dir, temp_dataset_dir):
        """Test that cache statistics are accurate"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Enable cache with no pre-loading
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=0)

        # Initially no cached shards
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 0
        assert stats["cache_size_bytes"] == 0

        # Access some shards
        dataset.get_shard_info(0)
        dataset.get_shard_info(1)

        # Check stats updated
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 2
        assert stats["cache_size_bytes"] > 0
        assert stats["cache_size_mb"] > 0

    def test_preload_on_enable(self, temp_cache_dir, temp_dataset_dir):
        """Test that init_shard_count pre-loads metadata"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Enable cache with pre-loading
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=3)

        # Should have cached 3 shards
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 3

    def test_cache_persistence(self, temp_cache_dir, temp_dataset_dir):
        """Test that cache persists across dataset instances"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()

        # First instance - populate cache
        dataset1 = discovery.discover_local(temp_dataset_dir)
        dataset1.enable_metadata_cache(temp_cache_dir, init_shard_count=2)

        # Access additional shard
        dataset1.get_shard_info(2)

        stats1 = dataset1.get_cache_stats()
        cached_count = stats1["cached_shards"]

        # Second instance - should find existing cache
        dataset2 = discovery.discover_local(temp_dataset_dir)
        dataset2.enable_metadata_cache(temp_cache_dir, init_shard_count=0)

        stats2 = dataset2.get_cache_stats()
        assert stats2["cached_shards"] == cached_count

    def test_clear_cache(self, temp_cache_dir, temp_dataset_dir):
        """Test clearing the cache"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Enable and populate cache
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=3)

        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 3

        # Clear cache
        dataset.clear_metadata_cache()

        # Cache should be empty
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 0

    def test_cache_improves_performance(self, temp_cache_dir, temp_dataset_dir):
        """Test that cache improves access time"""
        from webshart import DatasetDiscovery
        import time

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=0)

        # First access - no cache
        start = time.time()
        for i in range(5):
            dataset.get_shard_info(0)
        uncached_time = time.time() - start

        # Ensure it's cached
        dataset.get_shard_info(0)

        # Second access - with cache
        start = time.time()
        for i in range(5):
            dataset.get_shard_info(0)
        cached_time = time.time() - start

        # Cached should be faster (though this might be flaky in CI)
        # Just check it doesn't get slower
        assert cached_time <= uncached_time * 1.5

    def test_all_metadata_methods_work_with_cache(
        self, temp_cache_dir, temp_dataset_dir
    ):
        """Test that all metadata-dependent methods work with caching"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=2)

        # Test various methods that depend on metadata
        assert dataset.num_shards == 5
        assert dataset.total_files == 10  # 2 files per shard
        assert dataset.total_size > 0

        # Test shard info access
        info = dataset.get_shard_info(3)
        assert "name" in info
        assert "num_files" in info

        # Test file listing
        files = dataset.list_files_in_shard(0)
        assert len(files) == 2

        # Test file location finding
        shard_idx, file_idx = dataset.find_file_location(5)  # 6th file overall
        assert shard_idx == 2
        assert file_idx == 1

        # Test opening shard
        reader = dataset.open_shard(0)
        assert reader.num_files == 2

    def test_stats_methods_with_cache(self, temp_cache_dir, temp_dataset_dir):
        """Test get_stats and get_detailed_stats with caching"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=1)

        # Quick stats
        size, files = dataset.quick_stats()
        # These might be None for local datasets

        # Regular stats
        stats = dataset.get_stats()
        assert stats["total_shards"] == 5
        assert "shard_details" in stats

        # Detailed stats (forces loading all metadata)
        detailed = dataset.get_detailed_stats()
        assert detailed["total_shards"] == 5
        assert detailed["total_files"] == 10
        assert detailed["average_files_per_shard"] == 2.0
        assert detailed["from_cache"] is False  # Always False for detailed

        # After detailed stats, all should be cached
        cache_stats = dataset.get_cache_stats()
        assert cache_stats["cached_shards"] == 5

    @pytest.mark.parametrize(
        "init_count,expected",
        [
            (0, 0),  # No pre-loading
            (3, 3),  # Pre-load 3
            (10, 5),  # More than available - should load all 5
        ],
    )
    def test_init_shard_count_variations(
        self, temp_cache_dir, temp_dataset_dir, init_count, expected
    ):
        """Test different init_shard_count values"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=init_count)

        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == expected

    def test_remote_dataset_caching_mock(self, temp_cache_dir):
        """Test caching with mocked remote dataset"""
        from webshart import DatasetDiscovery

        # This would need proper mocking of the HuggingFace API
        # Just showing the structure of how it would be tested
        with patch("webshart.DatasetDiscovery.discover_huggingface") as mock_discover:
            # Mock a dataset with shards
            mock_dataset = Mock()
            mock_dataset.num_shards.return_value = 10
            mock_dataset.is_remote = True
            mock_dataset.name = "test/dataset"
            mock_discover.return_value = mock_dataset

            discovery = DatasetDiscovery(hf_token="fake-token")
            dataset = discovery.discover_huggingface("test/dataset")

            # Enable caching
            dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=5)

    def test_cache_with_subfolder_datasets(self, temp_cache_dir):
        """Test that cache keys handle subfolders correctly"""
        from webshart import DatasetDiscovery

        # Create dataset structure with subfolders
        temp_dir = tempfile.mkdtemp()
        try:
            subfolder = os.path.join(temp_dir, "train")
            os.makedirs(subfolder)

            # Add mock files to subfolder
            for i in range(2):
                tar_path = os.path.join(subfolder, f"shard-{i:04d}.tar")
                json_path = os.path.join(subfolder, f"shard-{i:04d}.json")
                with open(tar_path, "wb") as f:
                    f.write(b"mock")
                with open(json_path, "w") as f:
                    json.dump({"filesize": 1024, "files": {}}, f)

            discovery = DatasetDiscovery()
            dataset = discovery.discover_local(subfolder)

            # Enable cache
            dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=1)

            stats = dataset.get_cache_stats()
            assert stats["cache_enabled"] is True

            # Cache location should include the dataset name
            assert (
                "train" in stats["cache_location"]
                or "subfolder" in stats["cache_location"]
            )

        finally:
            shutil.rmtree(temp_dir)

    def test_print_summary_with_cache(self, temp_cache_dir, temp_dataset_dir, capsys):
        """Test print_summary works with caching enabled"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=2)

        # Print summary. It'll just hang if the cache isn't working.
        dataset.print_summary(detailed=True)
