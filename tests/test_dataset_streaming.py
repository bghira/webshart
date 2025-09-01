# test_streaming_fixed.py
import pytest
import webshart
import tempfile
import json
import os
import tarfile
from pathlib import Path


@pytest.fixture
def mock_dataset_dir():
    """Create a temporary directory with proper webshart-compatible dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock tar files with proper structure
        for i in range(3):
            shard_name = f"shard_{i:04d}"
            tar_path = Path(tmpdir) / f"{shard_name}.tar"
            
            # Create a proper tar file
            with tarfile.open(tar_path, 'w') as tar:
                # Add some dummy files
                for j in range(10):
                    filename = f"{shard_name}_{j:06d}.jpg"
                    # Create dummy image data
                    data = b'FAKE_JPEG_DATA' * 100
                    
                    # Create tarinfo
                    info = tarfile.TarInfo(name=filename)
                    info.size = len(data)
                    
                    # Add to tar
                    import io
                    tar.addfile(info, io.BytesIO(data))
            
            # Create metadata JSON that matches the tar structure
            metadata = {
                "filesize": os.path.getsize(tar_path),
                "files": {}
            }
            
            # Read the tar to get actual offsets
            with tarfile.open(tar_path, 'r') as tar:
                offset = 0
                for member in tar:
                    if member.isfile():
                        # In tar files, the header is 512 bytes, followed by data rounded up to 512 bytes
                        header_size = 512
                        data_blocks = (member.size + 511) // 512
                        data_size = data_blocks * 512
                        
                        metadata["files"][member.name] = {
                            "offset": offset + header_size,  # Offset to actual data
                            "length": member.size  # Actual file size
                        }
                        offset += header_size + data_size
            
            json_path = Path(tmpdir) / f"{shard_name}.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f)
        
        yield tmpdir


@pytest.fixture
def discovered_dataset(mock_dataset_dir):
    """Create a discovered dataset from mock directory."""
    return webshart.discover_dataset(mock_dataset_dir)


class TestFileStream:
    """Test the FileStream functionality."""
    
    def test_basic_streaming(self, discovered_dataset):
        """Test basic file streaming."""
        # Open first shard
        shard = discovered_dataset.open_shard(0)
        
        # Stream first 5 files
        files_read = []
        with shard.stream_files(0, 5) as stream:
            for filename, data, metadata in stream:
                files_read.append({
                    'filename': filename,
                    'data_len': len(data),
                    'metadata': metadata
                })
        
        assert len(files_read) == 5
        assert all('filename' in f for f in files_read)
        assert all(f['data_len'] > 0 for f in files_read)
        assert all('metadata' in f for f in files_read)
        
        # Check metadata contains expected fields
        for f in files_read:
            assert 'index' in f['metadata']
            assert 'offset' in f['metadata']
            assert 'length' in f['metadata']
            assert f['metadata']['length'] == f['data_len']
    
    def test_stream_entire_shard(self, discovered_dataset):
        """Test streaming an entire shard."""
        shard = discovered_dataset.open_shard(0)
        num_files = shard.num_files  # It's a property, not a method!
        
        files_read = 0
        for filename, data, metadata in shard.stream_files(0, num_files):
            files_read += 1
            assert isinstance(filename, str)
            assert isinstance(data, bytes)
            assert isinstance(metadata, dict)
        
        assert files_read == num_files
    
    def test_stream_middle_range(self, discovered_dataset):
        """Test streaming files from the middle of a shard."""
        shard = discovered_dataset.open_shard(0)
        
        # Stream files 3-7
        files_read = []
        for filename, data, metadata in shard.stream_files(3, 7):
            files_read.append(metadata['index'])
        
        assert files_read == [3, 4, 5, 6]
    
    def test_stream_single_file(self, discovered_dataset):
        """Test streaming a single file."""
        shard = discovered_dataset.open_shard(0)
        
        files = list(shard.stream_files(5, 6))
        assert len(files) == 1
        
        filename, data, metadata = files[0]
        assert metadata['index'] == 5
    
    def test_empty_range(self, discovered_dataset):
        """Test streaming with empty range."""
        shard = discovered_dataset.open_shard(0)
        
        # Same start and end should yield nothing
        files = list(shard.stream_files(5, 5))
        assert len(files) == 0
    
    def test_out_of_bounds_start(self, discovered_dataset):
        """Test streaming with out of bounds start index."""
        shard = discovered_dataset.open_shard(0)
        
        with pytest.raises(Exception) as exc_info:
            list(shard.stream_files(100, 101))
        
        assert "out of range" in str(exc_info.value).lower()
    
    def test_out_of_bounds_end(self, discovered_dataset):
        """Test streaming with out of bounds end index."""
        shard = discovered_dataset.open_shard(0)
        num_files = shard.num_files  # Property, not method
        
        with pytest.raises(Exception) as exc_info:
            list(shard.stream_files(0, num_files + 10))
        
        assert "out of range" in str(exc_info.value).lower()
    
    def test_invalid_range(self, discovered_dataset):
        """Test streaming with invalid range (start > end)."""
        shard = discovered_dataset.open_shard(0)
        num_files = shard.num_files
        
        # Use indices that are within bounds but reversed
        if num_files >= 6:
            with pytest.raises(Exception) as exc_info:
                list(shard.stream_files(5, 3))
            assert "invalid range" in str(exc_info.value).lower()
    
    def test_file_content_integrity(self, discovered_dataset):
        """Test that streamed file content matches direct read."""
        shard = discovered_dataset.open_shard(0)
        
        # Read file 3 via streaming
        stream_files = list(shard.stream_files(3, 4))
        assert len(stream_files) == 1
        stream_filename, stream_data, stream_metadata = stream_files[0]
        
        # Read same file directly
        direct_data = shard.read_file(3)
        
        # Should be identical
        assert stream_data == direct_data
        assert stream_metadata['index'] == 3
    
    def test_metadata_types(self, discovered_dataset):
        """Test that metadata values have correct Python types."""
        shard = discovered_dataset.open_shard(0)
        
        for filename, data, metadata in shard.stream_files(0, 1):
            # Check types
            assert isinstance(metadata['filename'], str)
            assert isinstance(metadata['index'], int)
            assert isinstance(metadata['offset'], int)
            assert isinstance(metadata['length'], int)
            
            # Check values
            assert metadata['filename'] == filename
            assert metadata['index'] == 0
            assert metadata['offset'] >= 0
            assert metadata['length'] > 0


class TestStreamPerformance:
    """Test performance characteristics of streaming."""
    
    def test_lazy_evaluation(self, discovered_dataset):
        """Test that streaming is lazy and doesn't load all files at once."""
        shard = discovered_dataset.open_shard(0)
        num_files = shard.num_files
        
        # Create stream but don't consume it
        stream = shard.stream_files(0, min(10, num_files))
        
        # Stream object should be created immediately
        assert stream is not None
        
        # No files should be read yet (lazy evaluation)
        # We can't directly test memory usage, but we can verify
        # the stream works incrementally
        first_file = next(iter(stream))
        assert first_file is not None
    
    def test_early_termination(self, discovered_dataset):
        """Test that we can stop streaming early without reading all files."""
        shard = discovered_dataset.open_shard(0)
        num_files = shard.num_files
        
        files_read = 0
        for filename, data, metadata in shard.stream_files(0, min(10, num_files)):
            files_read += 1
            if files_read >= 5:
                break  # Stop early
        
        assert files_read == 5
        # Stream should be properly cleaned up even if not fully consumed


class TestErrorHandling:
    """Test error handling in streaming."""
    
    def test_invalid_shard_index(self, discovered_dataset):
        """Test streaming from invalid shard."""
        with pytest.raises(Exception) as exc_info:
            list(discovered_dataset.stream_shard_files(999, 0, 5))
        
        assert "out of range" in str(exc_info.value).lower()
    
    def test_stream_after_context_exit(self, discovered_dataset):
        """Test that stream cannot be used after context exit."""
        shard = discovered_dataset.open_shard(0)
        
        stream = None
        with shard.stream_files(0, 5) as s:
            stream = s
            # Consume one item
            next(stream)
        
        # Stream should raise error after context exit
        with pytest.raises(RuntimeError) as exc_info:
            next(stream)
        assert "consumed" in str(exc_info.value).lower()


class TestIntegration:
    """Integration tests with other webshart features."""
    
    def test_stream_with_metadata_loading(self, discovered_dataset):
        """Test streaming triggers metadata loading if needed."""
        # Get a fresh dataset discovery to ensure metadata isn't loaded
        dataset = webshart.discover_dataset(discovered_dataset.name)
        
        # Stream should work even if metadata wasn't pre-loaded
        files = list(dataset.stream_shard_files(0, 0, 3))
        assert len(files) == 3
    
    def test_stream_across_chunk_boundaries(self, discovered_dataset):
        """Test streaming that might cross internal chunk boundaries."""
        shard = discovered_dataset.open_shard(0)
        num_files = shard.num_files  # Property!
        
        # Stream a large range that would typically cross boundaries
        large_range_files = list(shard.stream_files(0, min(10, num_files)))
        
        # Verify continuity
        indices = [f[2]['index'] for f in large_range_files]
        assert indices == list(range(len(indices)))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])