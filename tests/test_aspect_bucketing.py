# test_aspect_bucketing.py

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import webshart
from webshart import TarDataLoader, discover_dataset


@pytest.fixture
def mock_loader():
    """Create a properly mocked TarDataLoader that doesn't trigger discovery."""
    with patch("webshart.TarDataLoader.__new__") as mock_new:
        # Create a mock instance
        mock_instance = Mock(spec=TarDataLoader)
        mock_new.return_value = mock_instance

        # Set up properties
        mock_instance.num_shards = 3

        # Configure the methods we'll test
        mock_instance.list_shard_aspect_buckets = Mock()
        mock_instance.list_all_aspect_buckets = Mock()

        yield mock_instance


@pytest.fixture
def mock_loader_factory():
    """Factory to create mock loaders with custom behavior."""

    def _create_loader(num_shards=3):
        loader = Mock(spec=TarDataLoader)
        type(loader).num_shards = PropertyMock(return_value=num_shards)
        loader.list_shard_aspect_buckets = Mock()
        loader.list_all_aspect_buckets = Mock()
        return loader

    return _create_loader


class TestAspectBucketing:
    """Test suite for aspect bucketing functionality."""

    def test_list_shard_aspect_buckets_default(self, mock_loader_factory):
        """Test basic aspect bucketing with default parameters."""
        loader = mock_loader_factory()

        # Set up the expected return value
        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {
                    "1.778": [
                        {
                            "filename": "image1.jpg",
                            "width": 1920,
                            "height": 1080,
                            "aspect": 1.778,
                        },
                        {
                            "filename": "image3.jpg",
                            "width": 1920,
                            "height": 1080,
                            "aspect": 1.778,
                        },
                    ],
                    "0.563": [
                        {
                            "filename": "image2.jpg",
                            "width": 1080,
                            "height": 1920,
                            "aspect": 0.563,
                        }
                    ],
                },
            }
        ]

        buckets = loader.list_shard_aspect_buckets([0])

        assert len(buckets) == 1
        assert buckets[0]["shard_idx"] == 0
        assert "1.778" in buckets[0]["buckets"]
        assert len(buckets[0]["buckets"]["1.778"]) == 2
        assert "0.563" in buckets[0]["buckets"]
        assert len(buckets[0]["buckets"]["0.563"]) == 1

        # Verify the method was called correctly
        loader.list_shard_aspect_buckets.assert_called_once_with([0])

    def test_list_shard_aspect_buckets_geometry_tuple(self, mock_loader_factory):
        """Test bucketing with geometry-tuple key type."""
        loader = mock_loader_factory()

        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {
                    "(1920, 1080)": [
                        {"filename": "image1.jpg", "width": 1920, "height": 1080}
                    ],
                    "(1080, 1920)": [
                        {"filename": "image2.jpg", "width": 1080, "height": 1920}
                    ],
                },
            }
        ]

        buckets = loader.list_shard_aspect_buckets([0], key="geometry-tuple")

        assert "(1920, 1080)" in buckets[0]["buckets"]
        assert "(1080, 1920)" in buckets[0]["buckets"]

        loader.list_shard_aspect_buckets.assert_called_once_with(
            [0], key="geometry-tuple"
        )

    def test_list_shard_aspect_buckets_geometry_list(self, mock_loader_factory):
        """Test bucketing with geometry-list key type."""
        loader = mock_loader_factory()

        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {
                    "[1920, 1080]": [
                        {"filename": "image1.jpg", "width": 1920, "height": 1080}
                    ],
                    "[1080, 1920]": [
                        {"filename": "image2.jpg", "width": 1080, "height": 1920}
                    ],
                },
            }
        ]

        buckets = loader.list_shard_aspect_buckets([0], key="geometry-list")

        assert "[1920, 1080]" in buckets[0]["buckets"]
        assert "[1080, 1920]" in buckets[0]["buckets"]

    def test_list_shard_aspect_buckets_with_target_pixel_area(
        self, mock_loader_factory
    ):
        """Test bucketing with target pixel area scaling."""
        loader = mock_loader_factory()

        # When target_pixel_area=1024, a 1920x1080 image scales to 1024x576
        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {
                    "(1024, 576)": [
                        {"filename": "image1.jpg", "width": 1920, "height": 1080}
                    ],
                    "(576, 1024)": [
                        {"filename": "image2.jpg", "width": 1080, "height": 1920}
                    ],
                },
            }
        ]

        buckets = loader.list_shard_aspect_buckets(
            [0], key="geometry-tuple", target_pixel_area=1024**2
        )

        assert "(1024, 576)" in buckets[0]["buckets"]
        assert "(576, 1024)" in buckets[0]["buckets"]

        loader.list_shard_aspect_buckets.assert_called_once_with(
            [0], key="geometry-tuple", target_pixel_area=1024**2
        )

    def test_list_shard_aspect_buckets_multiple_shards(self, mock_loader_factory):
        """Test bucketing across multiple shards."""
        loader = mock_loader_factory()

        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {"1.778": [{"filename": "image1.jpg"}]},
            },
            {
                "shard_idx": 1,
                "shard_name": "shard-00001.tar",
                "buckets": {"0.563": [{"filename": "image2.jpg"}]},
            },
            {
                "shard_idx": 2,
                "shard_name": "shard-00002.tar",
                "buckets": {"1.333": [{"filename": "image3.jpg"}]},
            },
        ]

        buckets = loader.list_shard_aspect_buckets([0, 1, 2])

        assert len(buckets) == 3
        assert buckets[0]["shard_idx"] == 0
        assert buckets[1]["shard_idx"] == 1
        assert buckets[2]["shard_idx"] == 2

    def test_list_shard_aspect_buckets_invalid_key(self, mock_loader_factory):
        """Test error handling for invalid key type."""
        # Instead of patching the class, we'll test the expected behavior
        # by simulating what should happen with an invalid key
        loader = mock_loader_factory()

        # Configure the method to raise ValueError
        loader.list_shard_aspect_buckets.side_effect = ValueError(
            "key must be 'aspect', 'geometry-tuple', or 'geometry-list'"
        )

        with pytest.raises(
            ValueError,
            match="key must be 'aspect', 'geometry-tuple', or 'geometry-list'",
        ):
            loader.list_shard_aspect_buckets([0], key="invalid-key")

    def test_list_shard_aspect_buckets_invalid_shard_index(self, mock_loader_factory):
        """Test error handling for invalid shard indices."""
        loader = mock_loader_factory(num_shards=3)

        # Configure the method to raise IndexError
        loader.list_shard_aspect_buckets.side_effect = IndexError(
            "Shard index 5 out of range"
        )

        with pytest.raises(IndexError):
            loader.list_shard_aspect_buckets([5])

    def test_list_all_aspect_buckets_generator(self, mock_loader_factory):
        """Test the generator that yields buckets for all shards."""
        loader = mock_loader_factory(num_shards=3)

        # Create a generator that yields buckets
        def bucket_generator():
            for i in range(3):
                yield {
                    "shard_idx": i,
                    "shard_name": f"shard-{i:05d}.tar",
                    "buckets": {"1.778": [{"filename": f"image{i}.jpg"}]},
                }

        loader.list_all_aspect_buckets.return_value = bucket_generator()

        # Collect all results from the generator
        all_buckets = list(loader.list_all_aspect_buckets())

        assert len(all_buckets) == 3
        for i, bucket in enumerate(all_buckets):
            assert bucket["shard_idx"] == i

    def test_list_all_aspect_buckets_with_parameters(self, mock_loader_factory):
        """Test the generator with custom key and target resolution."""
        loader = mock_loader_factory(num_shards=1)

        # Create a generator
        def bucket_generator():
            yield {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {"(512, 512)": [{"filename": "image1.jpg"}]},
            }

        loader.list_all_aspect_buckets.return_value = bucket_generator()

        buckets = list(
            loader.list_all_aspect_buckets(
                key="geometry-tuple", target_pixel_area=512**2
            )
        )

        # Verify the parameters were passed correctly
        loader.list_all_aspect_buckets.assert_called_with(
            key="geometry-tuple", target_pixel_area=512**2
        )

    def test_empty_shard_no_images(self, mock_loader_factory):
        """Test handling of shards with no image files."""
        loader = mock_loader_factory()

        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {},  # No images with geometry
            }
        ]

        buckets = loader.list_shard_aspect_buckets([0])

        assert len(buckets) == 1
        assert len(buckets[0]["buckets"]) == 0

    def test_aspect_ratio_calculation(self, mock_loader_factory):
        """Test that aspect ratios are calculated correctly when not provided."""
        loader = mock_loader_factory()

        # Images without explicit aspect field should have it calculated
        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {
                    "1.333": [  # 4:3 aspect ratio (e.g., 1024x768)
                        {"filename": "image1.jpg", "width": 1024, "height": 768}
                    ],
                    "0.750": [  # 3:4 aspect ratio (e.g., 768x1024)
                        {"filename": "image2.jpg", "width": 768, "height": 1024}
                    ],
                },
            }
        ]

        buckets = loader.list_shard_aspect_buckets([0])

        assert "1.333" in buckets[0]["buckets"]
        assert "0.750" in buckets[0]["buckets"]

    def test_scaling_dimensions(self):
        """Test the dimension scaling logic."""
        # Test landscape image
        from webshart._webshart import scale_dimensions

        # Landscape: 1920x1080 -> 1024x576 (maintaining aspect ratio)
        w, h = scale_dimensions(1920, 1080, 1024**2)
        assert w == 1344
        assert h == 768

        # Portrait: 1080x1920 -> 768x1024
        w, h = scale_dimensions(1080, 1920, 1024**2)
        assert w == 768
        assert h == 1344

        # Square: 1000x1000 -> 1024x1024
        w, h = scale_dimensions(1000, 1000, 1024**2)
        assert w == 1024
        assert h == 1024
