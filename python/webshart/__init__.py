"""Fast and memory-efficient webdataset shard reader with sync and async support."""

import asyncio
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any
import argparse
import sys


from webshart._webshart import (
    __version__,
    DatasetDiscovery,
    DiscoveredDataset,
    BatchOperations,
    MetadataExtractor,
)

__all__ = [
    "__version__",
    "DatasetDiscovery",
    "DiscoveredDataset",
    "MetadataExtractor",
    "discover_dataset",
    "discover_dataset_async",
    "AsyncDatasetDiscovery",
    "BatchOperations",
    "discover_datasets_batch",
    "read_files_batch",
]


# Synchronous convenience function
def discover_dataset(
    source: str, hf_token: Optional[str] = None, subfolder: Optional[str] = None
) -> DiscoveredDataset:
    """
    Discover dataset shards from various sources (synchronous).

    Args:
        source: Can be:
            - Local directory path (e.g., '/path/to/dataset/')
            - HuggingFace dataset repo (e.g., 'username/dataset-name')
        hf_token: Optional HuggingFace token for private datasets
        subfolder: Optional subfolder within HuggingFace repo

    Returns:
        DiscoveredDataset object with all shards discovered

    Example:
        >>> # Local dataset
        >>> dataset = discover_dataset('/path/to/dataset/')
        >>>
        >>> # HuggingFace dataset
        >>> dataset = discover_dataset('NebulaeWis/e621-2024-webp-4Mpixel')
        >>> print(f"Found {dataset.num_shards} shards")
    """
    discovery = DatasetDiscovery(hf_token=hf_token)

    # Check if it's a local path
    if Path(source).exists() and Path(source).is_dir():
        return discovery.discover_local(source)
    else:
        # Assume it's a HuggingFace repo
        return discovery.discover_huggingface(source, subfolder=subfolder)


# Asynchronous convenience function
async def discover_dataset_async(
    source: str, hf_token: Optional[str] = None, subfolder: Optional[str] = None
) -> DiscoveredDataset:
    """
    Discover dataset shards from various sources (asynchronous).

    This is the async version of discover_dataset, useful when you're already
    in an async context and want to avoid blocking the event loop.

    Args:
        source: Can be:
            - Local directory path (e.g., '/path/to/dataset/')
            - HuggingFace dataset repo (e.g., 'username/dataset-name')
        hf_token: Optional HuggingFace token for private datasets
        subfolder: Optional subfolder within HuggingFace repo

    Returns:
        DiscoveredDataset object with all shards discovered

    Example:
        >>> # In an async function
        >>> dataset = await discover_dataset_async('NebulaeWis/e621-2024-webp-4Mpixel')
        >>> print(f"Found {dataset.num_shards} shards")
    """
    discovery = DatasetDiscovery(hf_token=hf_token)

    # Check if it's a local path
    if Path(source).exists() and Path(source).is_dir():
        # Local discovery is synchronous
        return discovery.discover_local(source)
    else:
        # Use async HuggingFace discovery
        return await discovery.discover_huggingface_async(source, subfolder=subfolder)


# Async wrapper class for convenience
class AsyncDatasetDiscovery:
    """
    Async-first wrapper around DatasetDiscovery.

    This class provides a more ergonomic async API for dataset discovery,
    especially useful when working in async contexts like web servers.

    Example:
        >>> async def main():
        ...     discovery = AsyncDatasetDiscovery()
        ...     dataset = await discovery.discover('username/dataset')
        ...     print(f"Found {dataset.num_shards} shards")
    """

    def __init__(self, hf_token: Optional[str] = None):
        self._inner = DatasetDiscovery(hf_token=hf_token)

    async def discover(
        self, source: str, subfolder: Optional[str] = None
    ) -> DiscoveredDataset:
        """
        Discover dataset from any source.

        Automatically detects whether the source is local or remote.
        """
        if Path(source).exists() and Path(source).is_dir():
            return self._inner.discover_local(source)
        else:
            return await self._inner.discover_huggingface_async(
                source, subfolder=subfolder
            )

    async def discover_huggingface(
        self, repo_id: str, subfolder: Optional[str] = None
    ) -> DiscoveredDataset:
        """Discover dataset from HuggingFace Hub."""
        return await self._inner.discover_huggingface_async(
            repo_id, subfolder=subfolder
        )

    def discover_local(self, path: str) -> DiscoveredDataset:
        """Discover dataset from local directory (synchronous)."""
        return self._inner.discover_local(path)


# Batch convenience functions
def discover_datasets_batch(
    sources: List[str],
    hf_token: Optional[str] = None,
    subfolders: Optional[List[Optional[str]]] = None,
) -> List[Optional[DiscoveredDataset]]:
    """
    Discover multiple datasets in parallel.

    Args:
        sources: List of dataset sources (local paths or HF repos)
        hf_token: Optional HuggingFace token for private datasets
        subfolders: Optional list of subfolders (one per source, or None)

    Returns:
        List of DiscoveredDataset objects (None for failed discoveries)

    Example:
        >>> datasets = discover_datasets_batch([
        ...     '/path/to/local/dataset',
        ...     'username/hf-dataset-1',
        ...     'username/hf-dataset-2'
        ... ])
        >>> for ds in datasets:
        ...     if ds:
        ...         print(f"Found {ds.num_shards} shards in {ds.name}")
    """
    batch_ops = BatchOperations()
    return batch_ops.discover_datasets_batch(
        sources, hf_token=hf_token, subfolders=subfolders
    )


def read_files_batch(
    dataset_or_datasets: Union[DiscoveredDataset, List[DiscoveredDataset]],
    file_requests: List[Union[Tuple[int, int], Tuple[int, int, int]]],
) -> List[Optional[bytes]]:
    """
    Read multiple files from datasets in parallel.

    Args:
        dataset_or_datasets: Single dataset or list of datasets
        file_requests: List of file requests as tuples:
            - If single dataset: (shard_idx, file_idx)
            - If multiple datasets: (dataset_idx, shard_idx, file_idx)

    Returns:
        List of file contents as bytes (None for failed reads)

    Example:
        >>> # Single dataset
        >>> dataset = discover_dataset('username/dataset')
        >>> files = read_files_batch(dataset, [
        ...     (0, 0),  # First file in first shard
        ...     (0, 1),  # Second file in first shard
        ...     (1, 0),  # First file in second shard
        ... ])

        >>> # Multiple datasets
        >>> datasets = discover_datasets_batch(['dataset1', 'dataset2'])
        >>> files = read_files_batch(datasets, [
        ...     (0, 0, 0),  # Dataset 0, shard 0, file 0
        ...     (1, 0, 0),  # Dataset 1, shard 0, file 0
        ... ])
    """
    batch_ops = BatchOperations()

    # Normalize to list of datasets
    if isinstance(dataset_or_datasets, DiscoveredDataset):
        datasets = [dataset_or_datasets]
        # Convert (shard, file) to (0, shard, file)
        requests = [(0, s, f) for s, f in file_requests]
    else:
        datasets = dataset_or_datasets
        requests = file_requests

    return batch_ops.read_files_batch(datasets, requests)


class BatchProcessor:
    """
    Helper class for processing webdataset files in batches.

    Example:
        >>> processor = BatchProcessor()
        >>> results = processor.process_dataset(
        ...     'username/dataset',
        ...     batch_size=100,
        ...     max_workers=10
        ... )
    """

    def __init__(self):
        self.batch_ops = BatchOperations()

    def process_dataset(
        self,
        source: str,
        batch_size: int = 50,
        max_files: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> List[Any]:
        """
        Process all files in a dataset in batches.

        Args:
            source: Dataset source (local path or HF repo)
            batch_size: Number of files to process in each batch
            max_files: Maximum number of files to process (None for all)
            callback: Optional function to process each file's data

        Returns:
            List of processed results (or raw bytes if no callback)
        """
        # Discover dataset
        dataset = discover_dataset(source)
        if not dataset:
            return []

        # Build list of all file requests
        all_requests = []
        for shard_idx in range(dataset.num_shards):
            shard_info = dataset.get_shard_info(shard_idx)
            num_files = shard_info.get("num_files", 0)

            for file_idx in range(num_files):
                all_requests.append((shard_idx, file_idx))

                if max_files and len(all_requests) >= max_files:
                    break

            if max_files and len(all_requests) >= max_files:
                break

        # Process in batches
        results = []
        for i in range(0, len(all_requests), batch_size):
            batch_requests = all_requests[i : i + batch_size]
            batch_data = read_files_batch(dataset, batch_requests)

            # Apply callback if provided
            if callback:
                for data in batch_data:
                    if data:
                        results.append(callback(data))
                    else:
                        results.append(None)
            else:
                results.extend(batch_data)

        return results


def extract_metadata(args):
    """Extract metadata from unindexed webdataset shards."""
    extractor = MetadataExtractor(hf_token=args.hf_token)

    # Parse range if provided
    shard_range = None
    if args.range:
        try:
            parts = args.range.split(",")
            if len(parts) != 2:
                raise ValueError("Range must be in format 'start,end'")
            start = int(parts[0])
            end = int(parts[1])
            if start < 0 or end < start:
                raise ValueError(
                    "Invalid range: start must be >= 0 and end must be >= start"
                )
            shard_range = (start, end)
            print(f"Processing shards in range [{start}, {end})")
        except Exception as e:
            print(f"✗ Error parsing range: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        # Note: This assumes the Rust extract_metadata method will be updated to accept shard_range
        if shard_range:
            extractor.extract_metadata(
                source=args.source,
                destination=args.destination,
                checkpoint_dir=args.checkpoint_dir,
                max_workers=args.max_workers,
                shard_range=shard_range,  # Pass the range tuple
            )
        else:
            extractor.extract_metadata(
                source=args.source,
                destination=args.destination,
                checkpoint_dir=args.checkpoint_dir,
                max_workers=args.max_workers,
            )
        print(f"✓ Metadata extraction complete for {args.source}")
    except Exception as e:
        print(f"✗ Error extracting metadata: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="webshart - Fast webdataset shard utilities"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract-metadata subcommand
    extract_parser = subparsers.add_parser(
        "extract-metadata", help="Extract metadata from unindexed webdataset shards"
    )
    extract_parser.add_argument(
        "--source",
        required=True,
        help="Source dataset (local path or HF repo like 'laion/conceptual-captions-12m-webdataset')",
    )
    extract_parser.add_argument(
        "--destination",
        required=True,
        help="Destination for metadata (local path or HF repo like 'username/dataset-name')",
    )
    extract_parser.add_argument(
        "--checkpoint-dir",
        help="Directory for checkpoint files to enable resumable extraction",
    )
    extract_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers (default: 4)",
    )
    extract_parser.add_argument(
        "--hf-token", help="HuggingFace token for private datasets"
    )
    extract_parser.add_argument(
        "--range",
        help="Range of tar file indices to process (e.g., '0,1000' for indices 0-999). "
        "Useful for distributing work across multiple machines.",
    )

    args = parser.parse_args()

    if args.command == "extract-metadata":
        extract_metadata(args)
    else:
        parser.print_help()
        sys.exit(1)
