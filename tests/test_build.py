"""One-call dataset construction.

Downloads are stubbed — these tests exercise the part that decides whether a
notebook can run in a fresh session, not Roboflow's client.
"""

from __future__ import annotations

import pytest
from PIL import Image

from alpr.build import (
    SOURCES,
    BuildError,
    attribution,
    build_dataset,
    dataset_ready,
    ensure_dataset,
    ingest_sources,
)
from alpr.data.schema import Region


class TestSources:
    def test_both_regions_are_represented(self):
        assert {s.region for s in SOURCES} == {Region.EUROPE, Region.INDIA}

    def test_every_source_records_its_licence(self):
        # The licence decides what may be redistributed; a source without one
        # cannot be published safely.
        assert all(s.licence and s.url for s in SOURCES)

    def test_current_sources_are_redistributable(self):
        assert all(s.redistributable for s in SOURCES)

    @pytest.mark.parametrize(
        ("licence", "allowed"),
        [
            ("CC BY 4.0", True),
            ("CC BY-SA 4.0", True),
            ("ODbL-1.0", True),
            ("CC BY-NC-ND 4.0", False),  # the case a naive split got wrong
            ("CC BY-ND 4.0", False),
            ("CC BY-NC 4.0", False),
            ("cc by-nc-nd 4.0", False),  # case-insensitive
        ],
    )
    def test_licence_gates_redistribution(self, licence, allowed):
        # Regression: splitting "CC BY-NC-ND 4.0" on hyphens yields "ND 4.0",
        # so a membership test for "ND" silently marked NoDerivatives data as
        # publishable — a licence violation, not a cosmetic bug.
        from alpr.build import RoboflowSource

        source = RoboflowSource(
            workspace="w",
            project="p",
            version=1,
            directory="d",
            region=Region.INDIA,
            source_tag="t",
            id_prefix="x-",
            licence=licence,
            url="http://example.com",
        )
        assert source.redistributable is allowed

    def test_attribution_lists_redistributable_sources(self):
        text = attribution()
        for source in SOURCES:
            assert source.url in text


def _fake_download(raw_dir, n=4):
    """Mimic what Roboflow writes: one directory per source, per split."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        for split in ("train", "valid", "test"):
            images = raw_dir / source.directory / split / "images"
            labels = raw_dir / source.directory / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for i in range(n):
                stem = f"{split}_{i}"
                Image.new("RGB", (640, 480)).save(images / f"{stem}.jpg")
                (labels / f"{stem}.txt").write_text("0 0.5 0.6 0.2 0.08\n")
    return raw_dir


class TestIngest:
    def test_ingests_every_source(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        records = ingest_sources(raw)
        assert len(records) == len(SOURCES) * 3 * 4
        assert {r.source for r in records} == {s.source_tag for s in SOURCES}

    def test_regions_are_tagged(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        regions = {b.region for r in ingest_sources(raw) for b in r.boxes}
        assert regions == {s.region for s in SOURCES}

    def test_missing_source_is_an_error(self, tmp_path):
        with pytest.raises(BuildError, match="missing"):
            ingest_sources(tmp_path / "empty")


class TestBuildDataset:
    def test_produces_a_usable_export(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        data_yaml, stats = build_dataset(raw, tmp_path / "yolo", tmp_path / "manifest.jsonl")
        assert data_yaml.exists()
        assert stats.images == len(SOURCES) * 3 * 8
        assert dataset_ready(data_yaml)

    def test_is_deterministic(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        _, first = build_dataset(raw, tmp_path / "a", tmp_path / "m1.jsonl", seed=0)
        _, second = build_dataset(raw, tmp_path / "b", tmp_path / "m2.jsonl", seed=0)
        assert first.by_split == second.by_split


class TestDatasetReady:
    def test_false_when_absent(self, tmp_path):
        assert dataset_ready(tmp_path / "nope.yaml") is False

    def test_true_after_a_build(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        data_yaml, _ = build_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl")
        assert dataset_ready(data_yaml) is True


class TestEnsureDataset:
    def test_builds_when_missing(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        data_yaml = ensure_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl")
        assert data_yaml.exists()

    def test_is_a_no_op_when_present(self, tmp_path, monkeypatch):
        # The point of the function: any notebook can call it first, and it
        # costs nothing when the data is already there.
        raw = _fake_download(tmp_path / "raw")
        out = tmp_path / "yolo"
        ensure_dataset(raw, out, tmp_path / "m.jsonl")

        def _explode(*args, **kwargs):
            raise AssertionError("should not rebuild an existing dataset")

        monkeypatch.setattr("alpr.build.build_dataset", _explode)
        assert ensure_dataset(raw, out, tmp_path / "m.jsonl").exists()

    def test_force_rebuilds(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        out = tmp_path / "yolo"
        ensure_dataset(raw, out, tmp_path / "m.jsonl")
        (out / "data.yaml").unlink()
        assert ensure_dataset(raw, out, tmp_path / "m.jsonl", force=True).exists()

    def test_does_not_ask_for_a_key_when_sources_are_present(self, tmp_path, monkeypatch):
        # Re-running in a session that already downloaded should not demand a
        # credential it does not need.
        raw = _fake_download(tmp_path / "raw")

        def _explode(*args, **kwargs):
            raise AssertionError("should not request a credential")

        monkeypatch.setattr("alpr.env.get_credential", _explode)
        assert ensure_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl").exists()
