import time


def next_with_cache_wait(loader):
    """Drop-in replacement for next() that waits for cache if needed."""
    import time

    # Wait for cache if needed
    while loader.will_block():
        shard_info = loader.get_next_shard_info()
        if shard_info:
            loader.prepare_next_shard()
            # Simple progress indication
            print(f"Waiting for {shard_info['name']}...", end="", flush=True)
            while loader.will_block():
                time.sleep(0.2)
                print(".", end="", flush=True)
            print(" ready!")

    return next(loader)


def iter_with_cache_wait(loader, start_idx, end_idx):
    """Iterator that waits for cache when needed."""
    import time

    for idx in range(start_idx, end_idx + 1):
        # Wait for cache if needed
        while loader.will_block():
            shard_info = loader.get_next_shard_info()
            if (
                shard_info
                and not loader.get_shard_cache_status(shard_info["name"])["is_cached"]
            ):
                loader.prepare_next_shard()
                time.sleep(0.2)
            else:
                break

        try:
            yield next(loader)
        except StopIteration:
            break


class ShardCacheMonitor:
    """Helper for monitoring shard cache downloads with async support."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    async def wait_for_shard_async(self, filename, update_interval=0.1):
        """
        Async generator that yields download progress updates.

        Usage:
            async for progress in monitor.wait_for_shard_async('data-0000.tar'):
                print(f"Downloaded: {progress['downloaded']} / {progress['total']}")
        """
        import asyncio

        # Start the download
        self.dataloader.prepare_shard_by_name(filename)

        while True:
            status = self.dataloader.get_shard_cache_status(filename)

            if status["is_cached"]:
                yield {
                    "downloaded": status["cur_filesize"],
                    "total": status["cur_filesize"],
                    "done": True,
                }
                break

            yield {
                "downloaded": status["cur_filesize"],
                "total": status.get("total_size", 0),
                "done": False,
            }

            await asyncio.sleep(update_interval)

    def wait_for_next_shard(self, callback=None):
        """
        Synchronous helper that waits for the next shard with optional progress callback.
        """
        if not self.dataloader.will_block():
            return

        shard_info = self.dataloader.get_next_shard_info()
        if not shard_info:
            return

        self.dataloader.prepare_next_shard()

        last_size = 0
        while self.dataloader.will_block():
            status = self.dataloader.get_shard_cache_status(shard_info["name"])
            current_size = status["cur_filesize"]

            if current_size > last_size and callback:
                callback(current_size - last_size, current_size)
                last_size = current_size

            time.sleep(0.1)


import signal
import sys


class CacheWaitContext:
    """Context manager that ensures shards are cached before iteration."""

    def __init__(self, dataloader, lookahead=1, progress_bar=True):
        self.dataloader = dataloader
        self.lookahead = lookahead
        self.progress_bar = progress_bar
        self.pbar = None
        self.cache_pbar = None
        self._interrupted = False
        self._original_sigint = None

    def _signal_handler(self, signum, frame):
        self._interrupted = True
        # Clean up progress bars
        if self.cache_pbar is not None:
            self.cache_pbar.close()
        if self.pbar is not None:
            self.pbar.close()
        print("\nInterrupted! Cleaning up...")
        sys.exit(0)

    def __enter__(self):
        # Install signal handler
        self._original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        return self

    def __exit__(self, *args):
        # Restore original signal handler
        signal.signal(signal.SIGINT, self._original_sigint)
        if self.pbar is not None:
            self.pbar.close()
        if self.cache_pbar is not None:
            self.cache_pbar.close()

    def iterate(self):
        """Iterate with automatic cache waiting and progress display."""
        from tqdm import tqdm
        import time

        if self.progress_bar:
            self.pbar = tqdm(desc="Processing files")

        # Create manual iterator
        data_iterator = iter(self.dataloader)

        while not self._interrupted:
            # Check BEFORE trying to get next item
            if self.dataloader.will_block():
                shard_info = self.dataloader.get_next_shard_info()
                if shard_info:
                    # Start the download
                    self.dataloader.prepare_next_shard()

                    if self.progress_bar:
                        self.cache_pbar = tqdm(
                            desc=f"Caching {shard_info['name']}",
                            unit="B",
                            unit_scale=True,
                            total=shard_info.get("size", 0),
                        )

                    last_downloaded = 0
                    while self.dataloader.will_block() and not self._interrupted:
                        # Update progress bar with download status
                        status = self.dataloader.get_shard_cache_status(
                            shard_info["name"]
                        )
                        if status and status["cur_filesize"] > last_downloaded:
                            update_amount = status["cur_filesize"] - last_downloaded
                            if self.cache_pbar is not None:
                                self.cache_pbar.update(update_amount)
                            last_downloaded = status["cur_filesize"]

                    if self.cache_pbar is not None:
                        self.cache_pbar.close()
                        self.cache_pbar = None

                    if not self._interrupted:
                        # Prepare the next one
                        self.dataloader.prepare_next_shard()

            if self._interrupted:
                break

            try:
                # Now get the next item
                entry = next(data_iterator)
                if self.pbar is not None:
                    self.pbar.update(1)
                yield entry
            except StopIteration:
                break
