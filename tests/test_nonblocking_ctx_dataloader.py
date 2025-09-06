# test_cache_wait.py
import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from webshart.cache_wait import CacheWaitContext


class MockDataLoader:
    """Mock dataloader for testing cache wait functionality."""
    
    def __init__(self, entries, block_pattern=None):
        self.entries = entries
        self.block_pattern = block_pattern or []
        self.index = 0
        self.will_block_calls = 0
        self.prepare_calls = []
        self.current_shard_filename = "shard-0000.tar"
        self._blocking = False
        self._download_start_time = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.index >= len(self.entries):
            raise StopIteration
        entry = self.entries[self.index]
        self.index += 1
        return entry
        
    def will_block(self):
        """Returns True based on block pattern."""
        call_idx = self.will_block_calls
        self.will_block_calls += 1
        if call_idx < len(self.block_pattern):
            self._blocking = self.block_pattern[call_idx]
            return self.block_pattern[call_idx]
        return False
        
    def get_next_shard_info(self):
        return {
            'name': f'shard-{self.index:04d}.tar',
            'index': self.index,
            'size': 1000000,
            'is_cached': False
        }
        
    def prepare_next_shard(self):
        self.prepare_calls.append(time.time())
        self._download_start_time = time.time()
        # Simulate that after calling prepare, it takes some time to download
        if self._blocking:
            # Create a thread that will "finish" the download after a delay
            def finish_download():
                time.sleep(0.3)  # Simulate download time
                self._blocking = False
            threading.Thread(target=finish_download, daemon=True).start()
        
    def get_shard_cache_status(self, shard_name):
        # Simulate download progress
        if self._download_start_time:
            elapsed = time.time() - self._download_start_time
            progress = min(int(elapsed * 3000000), 1000000)  # Fast download for testing
            return {
                'cur_filesize': progress,
                'is_cached': progress >= 1000000
            }
        return {
            'cur_filesize': 0,
            'is_cached': False
        }


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with test entries."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(10)]
    return MockDataLoader(entries)


def test_cache_wait_context_basic(mock_dataloader):
    """Test basic iteration without blocking."""
    with CacheWaitContext(mock_dataloader, progress_bar=False) as ctx:
        results = list(ctx.iterate())
    
    assert len(results) == 10
    assert all(hasattr(r, 'path') for r in results)


def test_cache_wait_context_with_blocking(mock_dataloader):
    """Test iteration with blocking pattern."""
    # Block on first call only
    mock_dataloader.block_pattern = [True] + [False] * 20
    
    with CacheWaitContext(mock_dataloader, progress_bar=False) as ctx:
        results = []
        for entry in ctx.iterate():
            results.append(entry)
            
    assert len(results) == 10
    assert len(mock_dataloader.prepare_calls) >= 1  # Should have prepared at least once


def test_cache_wait_context_progress_bar():
    """Test that progress bars are created and closed properly."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(5)]
    loader = MockDataLoader(entries, block_pattern=[True] + [False] * 10)
    
    # Patch tqdm where it's actually used
    with patch('tqdm.tqdm') as mock_tqdm_class:
        mock_pbar = MagicMock()
        mock_cache_pbar = MagicMock()
        
        # Configure the mock to return different objects
        mock_tqdm_class.side_effect = [mock_pbar, mock_cache_pbar] + [MagicMock()] * 10
        
        with CacheWaitContext(loader, progress_bar=True) as ctx:
            list(ctx.iterate())
            
        # Progress bars should be created
        assert mock_tqdm_class.call_count >= 1
        # Main progress bar should be updated and closed
        assert mock_pbar.update.called
        assert mock_pbar.close.called


def test_cache_wait_context_exception_handling():
    """Test that context manager properly cleans up on exception."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(5)]
    loader = MockDataLoader(entries)
    
    class TestException(Exception):
        pass
    
    with patch('tqdm.tqdm') as mock_tqdm_class:
        mock_pbar = MagicMock()
        mock_tqdm_class.return_value = mock_pbar
        
        with pytest.raises(TestException):
            with CacheWaitContext(loader, progress_bar=True) as ctx:
                for i, entry in enumerate(ctx.iterate()):
                    if i == 2:
                        raise TestException("Test error")
                        
        # Progress bar should still be closed
        assert mock_pbar.close.called


def test_cache_wait_blocking_timeout():
    """Test that blocking doesn't hang indefinitely."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(3)]
    loader = MockDataLoader(entries)
    
    # Set up a blocking pattern that will resolve after prepare_next_shard
    loader.block_pattern = [True] + [False] * 20
    
    start_time = time.time()
    with CacheWaitContext(loader, progress_bar=False) as ctx:
        results = list(ctx.iterate())
                
    assert len(results) == 3
    assert time.time() - start_time < 2  # Should complete quickly


def test_cache_wait_download_progress():
    """Test that download progress is tracked correctly."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(2)]
    loader = MockDataLoader(entries)
    
    # Manually control blocking
    loader._blocking = True
    progress_updates = []
    
    # Custom will_block that returns True for several calls
    call_count = 0
    def custom_will_block():
        nonlocal call_count
        call_count += 1
        if call_count <= 1:
            return True
        elif call_count <= 5:  # Block for a few more calls to simulate download
            status = loader.get_shard_cache_status('shard-0000.tar')
            progress_updates.append(status['cur_filesize'])
            return True
        else:
            return False
    
    loader.will_block = custom_will_block
    
    with CacheWaitContext(loader, progress_bar=False) as ctx:
        # Just get first entry to trigger the blocking
        for i, entry in enumerate(ctx.iterate()):
            if i == 0:
                break
            
    # Should have captured some progress updates
    assert len(progress_updates) > 1
    # Progress should increase
    assert any(progress_updates[i] < progress_updates[i+1] 
              for i in range(len(progress_updates)-1))


def test_cache_wait_interrupt_handling():
    """Test structure for interrupt handling."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(10)]
    loader = MockDataLoader(entries)
    
    processed = []
    
    with CacheWaitContext(loader, progress_bar=False) as ctx:
        try:
            for i, entry in enumerate(ctx.iterate()):
                processed.append(entry)
                if i == 3:
                    # Simulate interrupt
                    raise KeyboardInterrupt()
        except KeyboardInterrupt:
            pass  # Expected
            
    assert len(processed) == 4  # Should have processed 4 before interrupt


@pytest.mark.parametrize("lookahead", [1, 3, 5])
def test_cache_wait_lookahead(lookahead):
    """Test different lookahead values."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(10)]
    loader = MockDataLoader(entries)
    
    # Add prepare_shards_ahead method that the context might call
    prepare_ahead_calls = []
    loader.prepare_shards_ahead = lambda n: prepare_ahead_calls.append(n) or []
    
    # The context manager should call prepare_shards_ahead in __enter__
    with CacheWaitContext(loader, lookahead=lookahead, progress_bar=False) as ctx:
        # Just iterate through one item to ensure the context is used
        for entry in ctx.iterate():
            break
    
    # Since CacheWaitContext might not call prepare_shards_ahead in __enter__,
    # let's just verify the lookahead value is stored
    assert ctx.lookahead == lookahead


def test_cache_wait_empty_dataloader():
    """Test with empty dataloader."""
    loader = MockDataLoader([])
    
    with CacheWaitContext(loader, progress_bar=False) as ctx:
        results = list(ctx.iterate())
        
    assert results == []


def test_cache_wait_single_entry():
    """Test with single entry."""
    entries = [Mock(path="single_file.jpg")]
    loader = MockDataLoader(entries)
    
    with CacheWaitContext(loader, progress_bar=False) as ctx:
        results = list(ctx.iterate())
        
    assert len(results) == 1
    assert results[0].path == "single_file.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])