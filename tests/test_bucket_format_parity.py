import io
import json
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import webshart


@pytest.fixture
def datasets(tmp_path):
    tar, parquet = tmp_path / "tar", tmp_path / "parquet"
    tar.mkdir()
    parquet.mkdir()
    for shard in range(7):
        rows = []
        sizes = {}
        stem = f"{shard:03d}"
        with tarfile.open(tar / f"{stem}.tar", "w") as archive:
            for i, width in enumerate((16, 16, 32)):
                data = f"{shard}-{i}".encode()
                rows.append(dict(image=data, width=width, height=16, captions="sample"))
                name = f"{i}.jpg"
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
                sizes[name] = width
        with tarfile.open(tar / f"{stem}.tar") as archive:
            entries = {
                e.name: dict(
                    width=sizes[e.name],
                    height=16,
                    aspect=sizes[e.name] / 16,
                    offset=e.offset_data,
                    length=e.size,
                )
                for e in archive
            }
        (tar / f"{stem}.json").write_text(
            json.dumps(dict(filesize=(tar / f"{stem}.tar").stat().st_size, files=entries))
        )
        pq.write_table(pa.Table.from_pylist(rows), parquet / f"{stem}.parquet")
    return webshart.discover_dataset(str(tar)), webshart.discover_dataset(
        str(parquet), format="parquet"
    )


@pytest.mark.parametrize("strategy", ["sequential", "random_within_buckets", "fully_random"])
@pytest.mark.parametrize("lazy", [False, True])
def test_bucket_formats_share_order_and_carry(datasets, strategy, lazy):
    def read(dataset, seed):
        loader = webshart.BucketDataLoader(
            dataset,
            batch_size=4,
            lazy_load=lazy,
            shard_batch_size=1,
            sampling_strategy=strategy,
            shuffle_seed=seed,
        )

        def consume():
            result = []
            for batch in loader.iter_batches():
                assert len({(s.width, s.height) for s in batch}) == 1
                result.append([bytes(s.data) for s in batch])
            return result

        batches = consume()
        loader.reset()
        assert consume() == batches
        loader.reset()
        for i, batch in enumerate(batches):
            assert [bytes(s.data) for s in loader.next_batch()] == batch
            state = json.loads(json.dumps(loader.state_dict()))
            loader.load_state_dict(state)
        assert loader.next_batch() is None
        return batches

    expected = read(datasets[0], 17)
    assert sorted(len(b) for b in expected) == [2, 3, 4, 4, 4, 4]
    assert len(set(x for b in expected for x in b)) == 21
    assert read(datasets[1], 17) == expected
    assert read(datasets[0], 18) != expected


def test_prefill_follows_consumed_groups_and_resume(datasets, monkeypatch):
    from unittest.mock import Mock

    order = list(webshart._webshart._shuffled_shard_order(7, 23))
    for ds in datasets:
        cursors = []

        def prefill(self, *, active_shards, shuffle_seed):
            assert active_shards == 1 and shuffle_seed == 23
            cursors.append(Mock())
            return cursors[-1]

        monkeypatch.setattr(type(ds), "sequential_prefill", prefill)
        with webshart.BucketDataLoader(
            ds, batch_size=1, shard_batch_size=1, shuffle_seed=23
        ) as loader:
            for _ in range(8):
                item = next(loader)
                shard = int(bytes(item.data).split(b"-")[0])
                assert cursors[-1].advance.call_args.args == (order.index(shard),)
            checkpoint = json.loads(json.dumps(loader.state_dict()))
            expected = [bytes(row.data) for row in loader]
        with webshart.BucketDataLoader(
            ds, batch_size=1, shard_batch_size=1, shuffle_seed=23
        ) as resumed:
            resumed.load_state_dict(checkpoint)
            cursor = cursors[-1]
            assert [bytes(row.data) for row in resumed] == expected
            slots = [call.args[0] for call in cursor.advance.call_args_list]
            assert slots == list(range(2, 7))
            cursor.close.assert_called_once()
