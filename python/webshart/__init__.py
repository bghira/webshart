"""Fast and memory-efficient webdataset shard reader with synchronous and batch support."""

from pathlib import Path
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Callable,
    Iterator,
)
import argparse
import os
import sys
import json
from .cache_wait import CacheWaitContext, iter_with_cache_wait, next_with_cache_wait


from webshart._webshart import (
    __version__,
    DatasetDiscovery,
    DiscoveredDataset,
    BatchOperations,
    MetadataExtractor,
    TarDataLoader,
    BucketDataLoader,
)
from .optimize import DEFAULT_PAYLOAD_EXTENSIONS, optimize_dataset

__all__ = [
    "__version__",
    "DatasetDiscovery",
    "DiscoveredDataset",
    "MetadataExtractor",
    "TarDataLoader",
    "BucketDataLoader",
    "discover_dataset",
    "discover_paired_dataset",
    "PairedDataset",
    "PairedTarDataLoader",
    "SampleLocation",
    "SamplePair",
    "LoadedSamplePair",
    "BatchOperations",
    "discover_datasets_batch",
    "read_files_batch",
    "CacheWaitContext",
    "iter_with_cache_wait",
    "next_with_cache_wait",
    "apply_captions_to_metadata",
    "write_captions_to_metadata",
    "upload_caption_metadata",
    "optimize_dataset",
]


CaptionValue = Union[str, List[str]]
OptionalCaptionValue = Optional[CaptionValue]


@dataclass(frozen=True)
class SampleLocation:
    """Stable location of one logical sample in a discovered dataset."""

    shard_index: int
    sample_index: int
    filename: str


@dataclass(frozen=True)
class SamplePair:
    """Two logical samples joined by a shared key."""

    key: str
    left: SampleLocation
    right: SampleLocation


@dataclass(frozen=True)
class LoadedSamplePair:
    """Loaded values for a :class:`SamplePair`."""

    key: str
    left: Any
    right: Any


def _default_pair_key(filename: str) -> str:
    return str(PurePosixPath(filename).with_suffix(""))


class PairedDataset:
    """Opt-in key join over two independently discovered datasets.

    The underlying datasets and their normal loader behavior are unchanged.
    Pair locations are indexed lazily the first time they are requested.
    """

    def __init__(
        self,
        left: DiscoveredDataset,
        right: DiscoveredDataset,
        *,
        strict: bool = True,
        pair_key: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.left = left
        self.right = right
        self.strict = strict
        self.pair_key = pair_key or _default_pair_key
        self._pairs: Optional[List[SamplePair]] = None
        self._unmatched_left: Optional[List[str]] = None
        self._unmatched_right: Optional[List[str]] = None

    def _sample_locations(
        self, dataset: DiscoveredDataset, side: str
    ) -> Dict[str, SampleLocation]:
        locations: Dict[str, SampleLocation] = {}
        for shard_index in range(dataset.num_shards):
            for sample_index, filename in enumerate(
                dataset.list_samples_in_shard(shard_index)
            ):
                key = self.pair_key(str(filename))
                if key in locations:
                    raise ValueError(f"duplicate pair key on {side}: {key!r}")
                locations[key] = SampleLocation(
                    shard_index=shard_index,
                    sample_index=sample_index,
                    filename=str(filename),
                )
        return locations

    def _ensure_index(self) -> None:
        if self._pairs is not None:
            return

        left = self._sample_locations(self.left, "left")
        right = self._sample_locations(self.right, "right")
        self._unmatched_left = [key for key in left if key not in right]
        self._unmatched_right = [key for key in right if key not in left]

        if self.strict and (self._unmatched_left or self._unmatched_right):
            left_example = self._unmatched_left[:3]
            right_example = self._unmatched_right[:3]
            raise ValueError(
                "paired datasets do not have identical keys: "
                f"left_only={len(self._unmatched_left)} {left_example!r}, "
                f"right_only={len(self._unmatched_right)} {right_example!r}"
            )

        self._pairs = [
            SamplePair(key=key, left=location, right=right[key])
            for key, location in left.items()
            if key in right
        ]

    @property
    def num_pairs(self) -> int:
        self._ensure_index()
        return len(self._pairs or ())

    @property
    def unmatched_left(self) -> List[str]:
        self._ensure_index()
        return list(self._unmatched_left or ())

    @property
    def unmatched_right(self) -> List[str]:
        self._ensure_index()
        return list(self._unmatched_right or ())

    def __len__(self) -> int:
        return self.num_pairs

    def get_pair(self, index: int) -> SamplePair:
        self._ensure_index()
        assert self._pairs is not None
        return self._pairs[index]

    def list_pairs(self, start: int = 0, end: Optional[int] = None) -> List[SamplePair]:
        """Return a stable slice of pairs in left-dataset order."""
        self._ensure_index()
        assert self._pairs is not None
        return self._pairs[start:end]


class PairedTarDataLoader:
    """Load joined samples without changing :class:`TarDataLoader` semantics."""

    def __init__(self, dataset: PairedDataset, **loader_kwargs: Any) -> None:
        self.dataset = dataset
        self.left_loader = TarDataLoader(dataset.left, **loader_kwargs)
        self.right_loader = TarDataLoader(dataset.right, **loader_kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def load_pair(self, index: int) -> LoadedSamplePair:
        pair = self.dataset.get_pair(index)
        return LoadedSamplePair(
            key=pair.key,
            left=self.left_loader.load_sample(
                pair.left.shard_index, pair.left.sample_index
            ),
            right=self.right_loader.load_sample(
                pair.right.shard_index, pair.right.sample_index
            ),
        )

    def iter_pairs(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterator[LoadedSamplePair]:
        stop = len(self) if end is None else min(end, len(self))
        for index in range(start, stop):
            yield self.load_pair(index)


def _is_json_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".json"


def _sample_lookup_keys(path: str) -> List[str]:
    path_obj = Path(path)
    stem_path = str(path_obj.with_suffix(""))
    keys = [path, path_obj.name, stem_path, path_obj.stem]
    return list(dict.fromkeys(str(key) for key in keys if key))


def _normalize_captions(value: Any) -> OptionalCaptionValue:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        captions = [str(item) for item in value if item is not None and str(item)]
        return captions or None
    return str(value)


def apply_captions_to_metadata(
    metadata: MutableMapping[str, Any],
    captions_by_sample: Mapping[str, OptionalCaptionValue],
) -> int:
    """Attach captions to a webshart metadata mapping in-place.

    Captions are stored under the canonical plural ``captions`` key and may be a
    single string or a list of strings. Existing singular ``caption`` keys are
    removed from updated sample entries.
    """
    files = metadata.get("files")
    if not isinstance(files, (dict, list)):
        raise ValueError("webshart metadata must contain a 'files' dict or list")

    normalized: Dict[str, CaptionValue] = {}
    for sample, value in captions_by_sample.items():
        captions = _normalize_captions(value)
        if captions is None:
            continue
        for key in _sample_lookup_keys(str(sample)):
            normalized[key] = captions

    updated = 0

    if isinstance(files, dict):
        iterator = files.items()
    else:
        iterator = (
            (entry.get("path") or entry.get("filename") or entry.get("fname"), entry)
            for entry in files
            if isinstance(entry, dict)
        )

    for path, entry in iterator:
        if not path or not isinstance(entry, dict) or _is_json_path(str(path)):
            continue

        captions = next(
            (
                normalized[key]
                for key in _sample_lookup_keys(str(path))
                if key in normalized
            ),
            None,
        )
        if captions is None:
            continue

        entry.pop("caption", None)
        entry["captions"] = captions
        updated += 1

    return updated


def write_captions_to_metadata(
    metadata_path: Union[str, Path],
    captions_by_sample: Mapping[str, OptionalCaptionValue],
    output_path: Optional[Union[str, Path]] = None,
) -> int:
    """Write captions into a webshart shard metadata JSON file.

    Args:
        metadata_path: Existing webshart metadata JSON file to read.
        captions_by_sample: Mapping from sample path/stem to caption string or list.
        output_path: Optional destination JSON file. Defaults to updating
            ``metadata_path`` in place.

    Returns:
        Number of sample entries updated.
    """
    metadata_path = Path(metadata_path)
    destination = Path(output_path) if output_path is not None else metadata_path

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    updated = apply_captions_to_metadata(metadata, captions_by_sample)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return updated


def upload_caption_metadata(
    metadata_dir: Union[str, Path],
    repo_id: str,
    *,
    path_in_repo: str = "",
    revision: str = "main",
    hf_token: Optional[str] = None,
    commit_message: str = "Add coalesced webshart caption metadata",
):
    """Upload exported caption metadata to a Hugging Face dataset repository.

    This is deliberately separate from coalescing: callers can inspect the
    generated JSON before performing the external write.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Hub uploads require huggingface-hub; install webshart[hub]"
        ) from exc

    return HfApi(token=hf_token).upload_folder(
        folder_path=str(metadata_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        revision=revision,
        commit_message=commit_message,
        allow_patterns=["*.json", "**/*.json"],
    )


def discover_dataset(
    source: str,
    hf_token: Optional[str] = None,
    subfolder: Optional[str] = None,
    metadata: Optional[str] = None,
) -> DiscoveredDataset:
    """
    Discover dataset shards from various sources (synchronous).

    Args:
        source: Can be:
            - Local directory path (e.g., '/path/to/dataset/')
            - HuggingFace dataset repo (e.g., 'username/dataset-name')
        hf_token: Optional HuggingFace token for private datasets
        subfolder: Optional subfolder within HuggingFace repo
        metadata: Optional separate location for metadata:
            - Local directory path for metadata files
            - HuggingFace repo (e.g., 'username/dataset-index')
            - Full URL prefix

    Returns:
        DiscoveredDataset object with all shards discovered
    """
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    discovery = DatasetDiscovery(hf_token=hf_token, metadata_source=metadata)

    # Check if it's a local path
    if Path(source).exists() and Path(source).is_dir():
        return discovery.discover_local(source)
    else:
        # Assume it's a HuggingFace repo
        return discovery.discover_huggingface(source, subfolder=subfolder)


def discover_paired_dataset(
    left_source: str,
    right_source: Optional[str] = None,
    *,
    left_subfolder: Optional[str] = None,
    right_subfolder: Optional[str] = None,
    left_metadata: Optional[str] = None,
    right_metadata: Optional[str] = None,
    hf_token: Optional[str] = None,
    strict: bool = True,
    pair_key: Optional[Callable[[str], str]] = None,
) -> PairedDataset:
    """Discover two datasets and join logical samples by filename stem.

    ``right_source`` defaults to ``left_source`` for unified repositories whose
    two independently usable datasets live in separate subfolders.
    """
    right_source = right_source or left_source
    left = discover_dataset(
        left_source,
        hf_token=hf_token,
        subfolder=left_subfolder,
        metadata=left_metadata,
    )
    right = discover_dataset(
        right_source,
        hf_token=hf_token,
        subfolder=right_subfolder,
        metadata=right_metadata,
    )
    return PairedDataset(left, right, strict=strict, pair_key=pair_key)


def extract_metadata(args):
    """Extract metadata from unindexed webdataset shards."""
    extractor = MetadataExtractor(hf_token=args.hf_token or os.environ.get("HF_TOKEN"))

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
        if shard_range:
            extractor.extract_metadata(
                source=args.source,
                destination=args.destination,
                checkpoint_dir=args.checkpoint_dir,
                max_workers=args.max_workers,
                shard_range=shard_range,
                include_image_geometry=args.include_image_geometry,
            )
        else:
            extractor.extract_metadata(
                source=args.source,
                destination=args.destination,
                checkpoint_dir=args.checkpoint_dir,
                max_workers=args.max_workers,
                include_image_geometry=args.include_image_geometry,
            )
        print(f"✓ Metadata extraction complete for {args.source}")
    except Exception as e:
        print(f"✗ Error extracting metadata: {e}", file=sys.stderr)
        sys.exit(1)


def optimize_captions(args):
    """Coalesce sidecar captions into exportable per-shard metadata."""
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    dataset = discover_dataset(
        args.source,
        hf_token=hf_token,
        subfolder=args.subfolder,
        metadata=args.metadata,
    )
    if args.shard_cache_dir:
        dataset.enable_shard_cache(
            args.shard_cache_dir,
            cache_limit_gb=args.shard_cache_gb,
            parallel_downloads=args.parallel_downloads,
        )

    shard_indices = None
    if args.range:
        start, end = (int(part) for part in args.range.split(",", 1))
        if start < 0 or end < start:
            raise ValueError("range must be start,end with 0 <= start <= end")
        shard_indices = list(range(start, min(end, dataset.num_shards)))

    loader = TarDataLoader(dataset, load_file_data=False)
    result = loader.coalesce_caption_metadata(
        destination=args.destination,
        shard_indices=shard_indices,
    )
    print(
        f"Coalesced {result['coalesced_samples']} captions across "
        f"{result['shards']} shards into {args.destination}"
    )

    if args.push_to_hub:
        commit = upload_caption_metadata(
            args.destination,
            args.push_to_hub,
            path_in_repo=args.path_in_repo,
            revision=args.revision,
            hf_token=hf_token,
        )
        print(f"Uploaded caption metadata: {commit}")


def run_optimize_dataset(args):
    """Run rolling loose-file conversion from the CLI."""
    extensions = (
        tuple(part.strip() for part in args.payload_extensions.split(","))
        if args.payload_extensions
        else DEFAULT_PAYLOAD_EXTENSIONS
    )
    result = optimize_dataset(
        args.source,
        destination=args.destination,
        push_to_hub=args.push_to_hub,
        source_subfolder=args.source_subfolder,
        output_prefix=args.output_prefix,
        source_revision=args.source_revision,
        target_revision=args.target_revision,
        hf_token=args.hf_token,
        max_shard_size_bytes=int(args.max_shard_size_gb * 1024**3),
        payload_extensions=extensions,
        include_image_geometry=not args.no_image_geometry,
        max_shards=args.max_shards,
        private=True if args.private else None,
    )
    print(
        f"Optimized {result['next_sample_index']}/{result['total_samples']} samples "
        f"into {result['next_shard_index']} shards "
        f"(created {result['shards_created']} this run; status={result['status']})"
    )


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
    extract_parser.add_argument(
        "--include-image-geometry",
        action="store_true",
        help="Include image geometry (width, height, aspect ratio) in metadata extraction",
    )

    optimize_parser = subparsers.add_parser(
        "optimize-captions",
        help="Fold .txt/.json sidecar captions into webshart metadata indexes",
    )
    optimize_parser.add_argument("--source", required=True)
    optimize_parser.add_argument("--destination", required=True)
    optimize_parser.add_argument(
        "--metadata", help="Separate local or Hub metadata source"
    )
    optimize_parser.add_argument("--subfolder")
    optimize_parser.add_argument("--hf-token")
    optimize_parser.add_argument("--range", help="Half-open shard range: start,end")
    optimize_parser.add_argument("--shard-cache-dir")
    optimize_parser.add_argument("--shard-cache-gb", type=float, default=25.0)
    optimize_parser.add_argument("--parallel-downloads", type=int, default=4)
    optimize_parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Upload the export to a dataset repository",
    )
    optimize_parser.add_argument("--path-in-repo", default="")
    optimize_parser.add_argument("--revision", default="main")

    dataset_parser = subparsers.add_parser(
        "optimize-dataset",
        help="Stream loose payload/sidecar pairs into resumable webshart shards",
    )
    dataset_parser.add_argument("--source", required=True)
    dataset_parser.add_argument(
        "--destination",
        help="Optional local output directory; omit for rolling Hub-only conversion",
    )
    dataset_parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        help="Target Hugging Face dataset repo (may be the source repo)",
    )
    dataset_parser.add_argument("--source-subfolder", "--subfolder", default="")
    dataset_parser.add_argument(
        "--output-prefix",
        default="webshart",
        help="Target subfolder for shards, indexes, and resume state",
    )
    dataset_parser.add_argument("--source-revision", default="main")
    dataset_parser.add_argument("--target-revision", default="main")
    dataset_parser.add_argument("--hf-token")
    dataset_parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=1.0,
        help="Approximate maximum tar shard size (default: 1 GiB)",
    )
    dataset_parser.add_argument(
        "--payload-extensions",
        help="Comma-separated payload extensions; defaults to common media types",
    )
    dataset_parser.add_argument(
        "--no-image-geometry",
        action="store_true",
        help="Skip image width, height, and aspect extraction",
    )
    dataset_parser.add_argument(
        "--max-shards",
        type=int,
        help="Stop after this many new shards; rerun to resume",
    )
    dataset_parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private target repo when it does not already exist",
    )

    args = parser.parse_args()

    if args.command == "extract-metadata":
        extract_metadata(args)
    elif args.command == "optimize-captions":
        try:
            optimize_captions(args)
        except Exception as exc:
            print(f"✗ Error optimizing captions: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "optimize-dataset":
        try:
            run_optimize_dataset(args)
        except Exception as exc:
            print(f"✗ Error optimizing dataset: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
