"""Rewrite a Parquet shard for sparse image reads and verify it before publication."""

from pathlib import Path
import os
from tempfile import TemporaryDirectory


def repack_parquet(source, destination, image_columns=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        rewritten = Path(temp) / "shard.parquet"
        count = _write_verified_shard(source, rewritten, image_columns)
        os.link(rewritten, destination)
    return {"rows": count, "bytes": destination.stat().st_size}


def _write_verified_shard(source, rewritten, image_columns):
    import pyarrow as pa
    import pyarrow.parquet as pq

    with pq.ParquetFile(source) as original:
        schema = original.schema_arrow
        binary = {f.name for f in schema if pa.types.is_binary(f.type) or pa.types.is_large_binary(f.type)}
        images = binary if image_columns is None else set(image_columns)
        if not images or not images.issubset(binary):
            raise ValueError("Select one or more flat binary image columns in the source schema")
        dictionaries = [name for name in schema.names if name not in images]
        with pq.ParquetWriter(
            rewritten, schema, use_dictionary=dictionaries, data_page_size=1024**2,
            write_batch_size=1, write_page_index=True,
        ) as writer:
            for batch in original.iter_batches(batch_size=128):
                writer.write_batch(batch)
        with pq.ParquetFile(rewritten) as result:
            if not result.schema_arrow.equals(schema, check_metadata=True):
                raise ValueError("Repacked shard schema differs from the source")
            count = 0
            for group, batch in enumerate(original.iter_batches(batch_size=128)):
                actual = result.read_row_group(group)
                for expected, column in zip(batch.columns, actual.columns, strict=True):
                    if pa.types.is_floating(expected.type):
                        # Arrow equality treats NaN as unequal. Compare the stored IEEE bits.
                        bits = pa.binary(expected.type.bit_width // 8)
                        equal = expected.view(bits).equals(column.combine_chunks().view(bits))
                    else:
                        equal = column.equals(pa.chunked_array([expected]))
                    if not equal:
                        raise ValueError(f"Repacked shard differs from the source at row {count}")
                count += len(batch)
            if count != original.metadata.num_rows or count != result.metadata.num_rows:
                raise ValueError("Repacked shard row count differs from the source")
    return count
