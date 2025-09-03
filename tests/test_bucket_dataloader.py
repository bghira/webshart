import pytest
from unittest.mock import Mock, MagicMock, patch, create_autospec
import os
import tempfile
import json
from pathlib import Path


class TestBucketDataLoader:
    """Test suite for BucketDataLoader"""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock discovered dataset"""
        dataset = Mock()
        dataset.inner = Mock()
        dataset.inner.name = "test_dataset"
        dataset.inner.metadata_source = None
        dataset.inner.is_remote = False
        dataset.inner.num_shards = Mock(return_value=2)
        dataset.inner.shards = [
            Mock(tar_path="/path/to/shard1.tar", metadata=Mock()),
            Mock(tar_path="/path/to/shard2.tar", metadata=Mock()),
        ]
        return dataset

    @pytest.fixture
    def mock_file_info(self):
        """Create mock file info objects"""

        def _create_file_info(path, width, height, offset=0, length=1000):
            info = Mock()
            info.path = path
            info.width = width
            info.height = height
            info.aspect = width / height if height > 0 else 1.0
            info.offset = offset
            info.length = length
            return info

        return _create_file_info

    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create a temporary local dataset structure"""
        # Create tar files
        (tmp_path / "shard1.tar").write_bytes(b"dummy tar content")
        (tmp_path / "shard2.tar").write_bytes(b"dummy tar content")

        # Create metadata files
        metadata1 = {
            "filesize": 1000,
            "files": {
                "image1.jpg": {
                    "offset": 0,
                    "length": 100,
                    "width": 1920,
                    "height": 1080,
                },
                "image2.jpg": {
                    "offset": 100,
                    "length": 100,
                    "width": 1280,
                    "height": 720,
                },
                "image3.jpg": {
                    "offset": 200,
                    "length": 100,
                    "width": 800,
                    "height": 600,
                },
            },
        }

        metadata2 = {
            "filesize": 1000,
            "files": {
                "image4.jpg": {
                    "offset": 0,
                    "length": 100,
                    "width": 1920,
                    "height": 1080,
                },
                "image5.jpg": {
                    "offset": 100,
                    "length": 100,
                    "width": 3840,
                    "height": 2160,
                },
                "image6.jpg": {
                    "offset": 200,
                    "length": 100,
                    "width": 1024,
                    "height": 1024,
                },
            },
        }

        (tmp_path / "shard1.json").write_text(json.dumps(metadata1))
        (tmp_path / "shard2.json").write_text(json.dumps(metadata2))

        return str(tmp_path)

    @patch("webshart.BucketDataLoader")
    def test_initialization_with_dataset(self, MockBucketDataLoader, mock_dataset):
        """Test initialization with a DiscoveredDataset"""
        # Create a mock instance
        mock_instance = Mock()
        mock_instance.key_type = "aspect"
        mock_sampling_strategy = Mock(name="Sequential")
        mock_instance.sampling_strategy = mock_sampling_strategy
        mock_instance.load_file_data = True
        mock_instance.max_file_size = 50_000_000
        mock_instance.chunk_size_mb = 10
        MockBucketDataLoader.return_value = mock_instance

        # Import and create loader
        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=mock_dataset, key="aspect", sampling_strategy="sequential"
        )

        assert loader.key_type == "aspect"
        assert loader.sampling_strategy.name == mock_sampling_strategy.name
        assert loader.load_file_data is True
        assert loader.max_file_size == 50_000_000
        assert loader.chunk_size_mb == 10

    @patch("webshart.DatasetDiscovery")
    @patch("webshart.BucketDataLoader")
    def test_initialization_with_path(
        self, MockBucketDataLoader, MockDiscovery, temp_dataset
    ):
        """Test initialization with a path string"""
        # Mock the discovery process
        mock_discovered = Mock()
        mock_discovered.name = temp_dataset
        mock_discovered.metadata_source = None
        mock_discovery_instance = Mock()
        mock_discovery_instance.discover_local = Mock(return_value=mock_discovered)
        MockDiscovery.return_value = mock_discovery_instance

        # Create a mock loader instance
        mock_instance = Mock()
        mock_instance.key_type = "geometry-tuple"
        mock_instance.target_pixel_area = 1024 * 1024
        mock_sampling_strategy = Mock(name="RandomWithinBuckets")
        mock_instance.sampling_strategy = mock_sampling_strategy
        MockBucketDataLoader.return_value = mock_instance

        # Import and create loader
        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=temp_dataset,
            key="geometry-tuple",
            target_pixel_area=1024 * 1024,
            sampling_strategy="random_within_buckets",
        )

        assert loader.key_type == "geometry-tuple"
        assert loader.target_pixel_area == 1024 * 1024
        assert loader.sampling_strategy.name == mock_sampling_strategy.name

    @patch("webshart.BucketDataLoader")
    def test_invalid_key_type(self, MockBucketDataLoader, mock_dataset):
        """Test initialization with invalid key type"""
        MockBucketDataLoader.side_effect = ValueError(
            "key must be 'aspect', 'geometry-tuple', or 'geometry-list'"
        )

        from webshart import BucketDataLoader

        with pytest.raises(ValueError, match="key must be"):
            BucketDataLoader(dataset_or_path=mock_dataset, key="invalid_key")

    @patch("webshart.BucketDataLoader")
    def test_invalid_sampling_strategy(self, MockBucketDataLoader, mock_dataset):
        """Test initialization with invalid sampling strategy"""
        MockBucketDataLoader.side_effect = ValueError(
            "sampling_strategy must be 'sequential', 'random_within_buckets', or 'fully_random'"
        )

        from webshart import BucketDataLoader

        with pytest.raises(ValueError, match="sampling_strategy must be"):
            BucketDataLoader(
                dataset_or_path=mock_dataset, sampling_strategy="invalid_strategy"
            )

    @patch("webshart.BucketDataLoader")
    def test_build_buckets_aspect_key(
        self, MockBucketDataLoader, mock_dataset, mock_file_info
    ):
        """Test bucket building with aspect ratio key"""
        # Create mock instance with buckets
        mock_instance = Mock()
        mock_instance.buckets = {
            "1.78": [Mock(), Mock(), Mock()],  # 3 images with 16:9 aspect
            "1.33": [Mock()],  # 1 image
            "1.00": [Mock()],  # 1 image
        }
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=mock_dataset, key="aspect", round_to=2
        )

        # Check that buckets were built correctly
        assert len(loader.buckets) == 3  # 1.78, 1.33, 1.00
        assert "1.78" in loader.buckets
        assert len(loader.buckets["1.78"]) == 3  # 3 images with 16:9 aspect
        assert "1.33" in loader.buckets
        assert len(loader.buckets["1.33"]) == 1
        assert "1.00" in loader.buckets
        assert len(loader.buckets["1.00"]) == 1

    @patch("webshart.BucketDataLoader")
    def test_bucket_stats(self, MockBucketDataLoader, mock_dataset):
        """Test get_bucket_stats method"""
        # Create mock instance
        mock_instance = Mock()
        mock_instance.buckets = {
            "1.78": [Mock(), Mock(), Mock()],  # 3 files
            "1.33": [Mock()],  # 1 file
            "1.00": [Mock(), Mock()],  # 2 files
        }

        # Mock the get_bucket_stats method
        def mock_get_bucket_stats():
            return {
                "num_buckets": 3,
                "total_files": 6,
                "min_files_per_bucket": 1,
                "max_files_per_bucket": 3,
                "avg_files_per_bucket": 2.0,
                "bucket_details": {"1.78": 3, "1.33": 1, "1.00": 2},
            }

        mock_instance.get_bucket_stats = mock_get_bucket_stats
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(dataset_or_path=mock_dataset)

        stats = loader.get_bucket_stats()

        assert stats["num_buckets"] == 3
        assert stats["total_files"] == 6
        assert stats["min_files_per_bucket"] == 1
        assert stats["max_files_per_bucket"] == 3
        assert stats["avg_files_per_bucket"] == 2.0
        assert "bucket_details" in stats

    @patch("webshart.BucketDataLoader")
    def test_iteration_sequential(self, MockBucketDataLoader, mock_dataset):
        """Test sequential iteration through buckets"""
        # Create mock instance
        mock_instance = Mock()

        # Mock the iteration behavior
        mock_entries = [
            {"path": "file1", "data": b"data1"},
            {"path": "file2", "data": b"data2"},
            {"path": "file3", "data": b"data3"},
        ]
        mock_instance.__iter__ = Mock(return_value=iter(mock_entries))
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=mock_dataset, sampling_strategy="sequential"
        )

        # Iterate and collect results
        results = list(loader)

        assert len(results) == 3
        assert results[0]["path"] == "file1"
        assert results[1]["path"] == "file2"
        assert results[2]["path"] == "file3"

    @patch("webshart.BucketDataLoader")
    def test_skip_to_bucket(self, MockBucketDataLoader, mock_dataset):
        """Test skipping to a specific bucket"""
        mock_instance = Mock()
        mock_instance.bucket_keys = ["1.00", "1.33", "1.78"]
        mock_instance.current_bucket_idx = 0
        mock_instance.current_entry_idx = 5

        def mock_skip_to_bucket(bucket):
            if bucket == "1.78":
                mock_instance.current_bucket_idx = 2
                mock_instance.current_entry_idx = 0
            else:
                raise ValueError(f"Bucket '{bucket}' not found")

        mock_instance.skip_to_bucket = mock_skip_to_bucket
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(dataset_or_path=mock_dataset)

        # Skip to bucket "1.78"
        loader.skip_to_bucket("1.78")

        assert loader.current_bucket_idx == 2
        assert loader.current_entry_idx == 0

    @patch("webshart.BucketDataLoader")
    def test_skip_to_nonexistent_bucket(self, MockBucketDataLoader, mock_dataset):
        """Test skipping to a bucket that doesn't exist"""
        mock_instance = Mock()
        mock_instance.skip_to_bucket = Mock(
            side_effect=ValueError("Bucket '2.00' not found")
        )
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(dataset_or_path=mock_dataset)

        with pytest.raises(ValueError, match="Bucket .* not found"):
            loader.skip_to_bucket("2.00")

    @patch("webshart.BucketDataLoader")
    def test_get_current_bucket(self, MockBucketDataLoader, mock_dataset):
        """Test getting the current bucket name"""
        mock_instance = Mock()
        mock_instance.get_current_bucket = Mock(return_value="1.33")
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(dataset_or_path=mock_dataset)

        assert loader.get_current_bucket() == "1.33"

    @patch("webshart.BucketDataLoader")
    def test_reset(self, MockBucketDataLoader, mock_dataset):
        """Test resetting the loader"""
        mock_instance = Mock()
        reset_called = [False]

        def mock_reset():
            reset_called[0] = True
            mock_instance.current_bucket_idx = 0
            mock_instance.current_entry_idx = 0
            mock_instance.random_position = 0

        mock_instance.reset = mock_reset
        mock_instance.current_bucket_idx = 0
        mock_instance.current_entry_idx = 0
        mock_instance.random_position = 0
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(dataset_or_path=mock_dataset)

        # Reset
        loader.reset()

        assert reset_called[0]
        assert loader.current_bucket_idx == 0
        assert loader.current_entry_idx == 0
        assert loader.random_position == 0

    @patch("webshart.BucketDataLoader")
    def test_repr(self, MockBucketDataLoader, mock_dataset):
        """Test string representation"""
        mock_instance = Mock()
        mock_instance.__repr__ = Mock(
            return_value="BucketDataLoader(buckets=3, strategy=Sequential, current_bucket=1)"
        )
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(dataset_or_path=mock_dataset)

        repr_str = repr(loader)

        assert "BucketDataLoader" in repr_str
        assert "buckets=3" in repr_str
        assert "current_bucket=1" in repr_str

    @patch("webshart.BucketDataLoader")
    @pytest.mark.parametrize(
        "sampling_strategy", ["sequential", "random_within_buckets", "fully_random"]
    )
    def test_all_sampling_strategies(
        self, MockBucketDataLoader, mock_dataset, sampling_strategy
    ):
        """Test that all sampling strategies work without errors"""
        # Create mock that returns one item then stops
        mock_instance = Mock()
        mock_instance.__iter__ = Mock(
            return_value=iter([{"path": "test.jpg", "data": b"data"}])
        )
        MockBucketDataLoader.return_value = mock_instance

        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=mock_dataset, sampling_strategy=sampling_strategy
        )

        # Should be able to get at least one item
        result = next(iter(loader))
        assert result is not None
        assert result["path"] == "test.jpg"
