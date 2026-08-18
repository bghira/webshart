import pytest

import webshart


class FakeDataset:
    def __init__(self, shards):
        self.shards = shards
        self.num_shards = len(shards)

    def list_samples_in_shard(self, shard_index):
        return self.shards[shard_index]


def test_pair_index_joins_by_stem_in_left_order():
    left = FakeDataset([["b.mp3", "a.mp3"], ["nested/c.mp3"]])
    right = FakeDataset([["a.flac"], ["nested/c.wav", "b.wav"]])

    paired = webshart.PairedDataset(left, right)

    assert [pair.key for pair in paired.list_pairs()] == ["b", "a", "nested/c"]
    assert paired.get_pair(0).left.sample_index == 0
    assert paired.get_pair(0).right == webshart.SampleLocation(1, 1, "b.wav")


def test_pair_index_reports_mismatches_in_strict_mode():
    paired = webshart.PairedDataset(
        FakeDataset([["shared.mp3", "left-only.mp3"]]),
        FakeDataset([["shared.mp3", "right-only.mp3"]]),
    )

    with pytest.raises(ValueError, match="left_only=1.*right_only=1"):
        len(paired)


def test_pair_index_can_use_intersection_and_report_unmatched_keys():
    paired = webshart.PairedDataset(
        FakeDataset([["shared.mp3", "left-only.mp3"]]),
        FakeDataset([["shared.mp3", "right-only.mp3"]]),
        strict=False,
    )

    assert len(paired) == 1
    assert paired.unmatched_left == ["left-only"]
    assert paired.unmatched_right == ["right-only"]


def test_pair_index_rejects_duplicate_keys():
    paired = webshart.PairedDataset(
        FakeDataset([["same.mp3", "same.wav"]]),
        FakeDataset([["same.mp3"]]),
    )

    with pytest.raises(ValueError, match="duplicate pair key on left"):
        len(paired)


def test_discover_paired_dataset_uses_same_repo_for_two_subfolders(monkeypatch):
    calls = []

    def fake_discover(source, **kwargs):
        calls.append((source, kwargs))
        return FakeDataset([[f"{kwargs['subfolder']}.mp3"]])

    monkeypatch.setattr(webshart, "discover_dataset", fake_discover)
    paired = webshart.discover_paired_dataset(
        "owner/dataset",
        left_subfolder="original",
        right_subfolder="covers",
        strict=False,
    )

    assert paired.left.list_samples_in_shard(0) == ["original.mp3"]
    assert paired.right.list_samples_in_shard(0) == ["covers.mp3"]
    assert [call[0] for call in calls] == ["owner/dataset", "owner/dataset"]


def test_paired_loader_loads_both_locations(monkeypatch):
    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

        def load_sample(self, shard_index, sample_index):
            return self.dataset.list_samples_in_shard(shard_index)[sample_index]

    monkeypatch.setattr(webshart, "TarDataLoader", FakeLoader)
    paired = webshart.PairedDataset(
        FakeDataset([["sample.mp3"]]),
        FakeDataset([["sample.wav"]]),
    )
    loader = webshart.PairedTarDataLoader(paired, load_file_data=False)

    loaded = loader.load_pair(0)
    assert loaded == webshart.LoadedSamplePair(
        key="sample", left="sample.mp3", right="sample.wav"
    )
