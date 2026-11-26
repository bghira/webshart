"""Annotate Webshart TAR indexes with crop geometry and pixel coverage."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops, ImageOps
from tqdm import tqdm

from .optimize import _list_hub_files, _list_local_files, _normalize_prefix, _repo_path

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def measure(image, *, white_threshold, black_threshold, tight=False):
    image = ImageOps.exif_transpose(image)
    if image.has_transparency_data:
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, "white")
        image.paste(rgba, mask=rgba.getchannel("A"))
    else:
        image = image.convert("RGB")

    red, green, blue = image.split()
    darkest = ImageChops.darker(ImageChops.darker(red, green), blue)
    brightest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    w, h = image.size
    white = sum(darkest.histogram()[white_threshold:])
    stats = {
        "white_fraction": white / (w * h),
        "black_fraction": sum(brightest.histogram()[: black_threshold + 1]) / (w * h),
    }
    if 0 < white < w * h:
        # Nonwhite pixels anywhere in a strip protect that strip, including text and fine lines.
        left, top, right, bottom = darkest.point(
            [255] * white_threshold + [0] * (256 - white_threshold)
        ).getbbox()
        if tight:
            if (left, top, right, bottom) != (0, 0, w, h):
                stats["whitespace_crop"] = [left, top, right, bottom]
            return stats
        pad = max(2, round(0.04 * min(w, h)))
        margins = [
            min(int(0.15 * edge), max(0, margin - pad))
            for margin, edge in ((left, w), (top, h), (w - right, w), (h - bottom, h))
        ]
        left, top, right, bottom = margins
        if max(left / w, right / w, top / h, bottom / h) >= 0.02:
            stats["whitespace_crop"] = [left, top, w - right, h - bottom]
    return stats


def _measure_payload(data, args):
    with Image.open(BytesIO(data)) as image:
        if image.width * image.height > args.max_pixels:
            raise ValueError(f"image exceeds {args.max_pixels} pixels")
        return measure(
            image,
            white_threshold=args.white_threshold,
            black_threshold=args.black_threshold,
            tight=args.tight,
        )


def _scan_shard(archive, index, executor, args):
    files = index["files"]
    if isinstance(files, list):
        entries = (
            (row.get("path") or row.get("fname") or row["filename"], row)
            for row in files
        )
    else:
        entries = files.items()
    images = {}
    for name, row in entries:
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
            key = Path(name).as_posix()
            if key in images:
                raise ValueError(f"duplicate image in index: {name!r}")
            images[key] = row
    if (
        any(name.lower().endswith(".jxl") for name in images)
        and ".jxl" not in Image.registered_extensions()
    ):
        try:
            import pillow_jxl  # Registers the Pillow decoder.
        except ImportError as exc:
            raise ImportError("JPEG XL shards require pillow-jxl-plugin") from exc
    pending = deque()
    counts = {"scanned": 0, "cropped": 0, "unreadable": 0}

    def collect():
        name, row, future = pending.popleft()
        try:
            stats = future.result()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            logger.warning("Skipping %s in %s: %s", name, archive.name, exc)
            counts["unreadable"] += 1
        else:
            row.update(stats)
            counts["scanned"] += 1
            counts["cropped"] += int("whitespace_crop" in stats)

    with tarfile.open(archive, mode="r|", bufsize=1024 * 1024) as tar:
        for member in tar:
            name = Path(member.name).as_posix()
            row = images.pop(name, None)
            if row is None:
                continue
            if (
                not member.isfile()
                or row["offset"] != member.offset_data
                or row.get("length", row.get("size")) != member.size
            ):
                raise ValueError(f"index does not match TAR member {name!r}")
            for key in ("whitespace_crop", "white_fraction", "black_fraction"):
                row.pop(key, None)
            if member.size > args.max_bytes:
                logger.warning(
                    "Skipping %s in %s: exceeds %s bytes",
                    name,
                    archive.name,
                    args.max_bytes,
                )
                counts["unreadable"] += 1
                continue
            if len(pending) == args.workers:
                collect()
            data = tar.extractfile(member).read()
            pending.append((name, row, executor.submit(_measure_payload, data, args)))
        while pending:
            collect()
    if images:
        raise ValueError(f"indexed image missing from archive: {next(iter(images))!r}")
    return counts


def _materialize(repo, filename, revision, token, cache_dir):
    if Path(repo).is_dir():
        return Path(repo) / filename
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
        )
    )


def scan_whitespace(args):
    if not 0 <= args.black_threshold < args.white_threshold <= 255:
        raise ValueError("thresholds must satisfy 0 <= black < white <= 255")
    if min(args.workers, args.max_bytes, args.max_pixels) < 1:
        raise ValueError("workers, max-bytes and max-pixels must be positive")
    token = args.hf_token or os.getenv("HF_TOKEN")
    if args.push_to_hub:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=token)
        parent_commit = api.dataset_info(args.push_to_hub, revision=args.revision).sha
        target_prefix = _normalize_prefix(args.path_in_repo)
    folder = _normalize_prefix(args.subfolder)
    source_rev = args.source_revision
    if Path(args.source).is_dir():
        files = _list_local_files(Path(args.source), folder)
    else:
        files, source_rev = _list_hub_files(args.source, folder, source_rev, token)
    shards = sorted(f.path for f in files if f.path.endswith(".tar"))
    if not shards:
        raise ValueError("no TAR shards found")
    if args.range:
        start, stop = map(int, args.range.split(","))
        if not 0 <= start < stop <= len(shards):
            raise ValueError(f"range must satisfy 0 <= start < stop <= {len(shards)}")
        shards = shards[start:stop]
    metadata = args.metadata or args.source
    metadata_rev = args.metadata_revision or (
        source_rev if metadata == args.source else "main"
    )
    if not Path(metadata).is_dir() and (
        metadata != args.source or metadata_rev != source_rev
    ):
        from huggingface_hub import HfApi

        metadata_rev = (
            HfApi(token=token).dataset_info(metadata, revision=metadata_rev).sha
        )

    destination = Path(args.destination)
    totals = {"scanned": 0, "cropped": 0, "unreadable": 0}
    outputs = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for shard in tqdm(shards, desc="Whitespace scan", unit="shard"):
            relative = str(Path(shard).with_suffix(".json"))
            output = destination / relative
            with TemporaryDirectory() as tmp:
                cache = args.cache_dir or tmp
                index_path = _materialize(
                    metadata,
                    _repo_path(folder if not args.metadata else "", relative),
                    metadata_rev,
                    token,
                    cache,
                )
                index = json.loads(index_path.read_bytes())
                archive = _materialize(
                    args.source, _repo_path(folder, shard), source_rev, token, cache
                )
                counts = _scan_shard(archive, index, executor, args)
            output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=output.parent) as staging:
                temp = Path(staging) / output.name
                with temp.open("w", encoding="utf-8") as stream:
                    json.dump(index, stream, ensure_ascii=False, separators=(",", ":"))
                temp.replace(output)
            outputs.append(relative)
            for key, count in counts.items():
                totals[key] += count

    if args.push_to_hub:
        api.create_commit(
            repo_id=args.push_to_hub,
            repo_type="dataset",
            revision=args.revision,
            parent_commit=parent_commit,
            commit_message="Add whitespace measurements to Webshart indexes",
            operations=[
                CommitOperationAdd(
                    _repo_path(target_prefix, name),
                    destination / name,
                )
                for name in outputs
            ],
        )
    return {"shards": len(outputs), **totals}
