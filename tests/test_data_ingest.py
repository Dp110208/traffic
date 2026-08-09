"""Ingest adapters."""

from __future__ import annotations

import pytest
from PIL import Image

from alpr.data import DatasetError, Region
from alpr.data.ingest import from_yolo_dir, image_size, parse_yolo_label


class TestParseYoloLabel:
    def test_single_box(self):
        boxes, dropped = parse_yolo_label("0 0.5 0.5 0.2 0.1\n", keep_classes=None)
        assert len(boxes) == 1
        assert boxes[0].cx == 0.5
        assert dropped == 0

    def test_blank_lines_ignored(self):
        boxes, _ = parse_yolo_label("\n0 0.5 0.5 0.2 0.1\n\n", keep_classes=None)
        assert len(boxes) == 1

    def test_extra_fields_tolerated(self):
        # Some exporters append a confidence column.
        boxes, _ = parse_yolo_label("0 0.5 0.5 0.2 0.1 0.98\n", keep_classes=None)
        assert len(boxes) == 1

    def test_drops_unwanted_classes(self):
        text = "0 0.5 0.5 0.2 0.1\n1 0.5 0.5 0.6 0.6\n"
        boxes, dropped = parse_yolo_label(text, keep_classes=frozenset({0}))
        assert len(boxes) == 1
        assert dropped == 1

    def test_too_few_fields_raises_with_line_number(self):
        with pytest.raises(DatasetError, match="line 2: expected 5 fields"):
            parse_yolo_label("0 0.5 0.5 0.2 0.1\n0 0.5 0.5\n", keep_classes=None)

    def test_non_numeric_raises_with_line_number(self):
        with pytest.raises(DatasetError, match="line 1"):
            parse_yolo_label("0 abc 0.5 0.2 0.1\n", keep_classes=None)

    def test_out_of_range_box_raises(self):
        with pytest.raises(DatasetError, match="line 1"):
            parse_yolo_label("0 0.5 0.5 0.0 0.1\n", keep_classes=None)


def _make_yolo_dataset(tmp_path, n=4, label_body="0 0.5 0.5 0.2 0.1\n", size=(640, 480)):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(n):
        Image.new("RGB", size).save(images / f"img{i}.jpg")
        (labels / f"img{i}.txt").write_text(label_body)
    return images, labels


class TestImageSize:
    def test_reads_dimensions(self, tmp_path):
        path = tmp_path / "a.png"
        Image.new("RGB", (321, 123)).save(path)
        assert image_size(path) == (321, 123)


class TestFromYoloDir:
    def test_ingests_images_and_labels(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="test")
        assert len(report.records) == 4
        assert all(r.width == 640 and r.height == 480 for r in report.records)
        assert all(r.source == "test" for r in report.records)

    def test_defaults_to_sibling_labels_dir(self, tmp_path):
        images, _ = _make_yolo_dataset(tmp_path)
        assert len(from_yolo_dir(images, source="test").records) == 4

    def test_applies_region(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t", region=Region.GERMANY)
        assert all(b.region is Region.GERMANY for r in report.records for b in r.boxes)

    def test_id_prefix_prevents_cross_dataset_collisions(self, tmp_path):
        # Two sources both containing img0.jpg must not collapse to one record.
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t", id_prefix="uc3m-")
        assert all(r.image_id.startswith("uc3m-") for r in report.records)

    def test_missing_labels_are_reported_not_silently_dropped(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        (labels / "img0.txt").unlink()
        report = from_yolo_dir(images, labels, source="t")
        assert report.skipped_no_label == ["img0"]
        assert len(report.records) == 3

    def test_strict_mode_raises_on_bad_label(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        (labels / "img1.txt").write_text("0 0.5 0.5\n")
        with pytest.raises(DatasetError, match="expected 5 fields"):
            from_yolo_dir(images, labels, source="t")

    def test_non_strict_records_bad_labels(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        (labels / "img1.txt").write_text("0 0.5 0.5\n")
        report = from_yolo_dir(images, labels, source="t", strict=False)
        assert len(report.records) == 3
        assert report.skipped_bad_label[0][0] == "img1"
        assert "bad label" in report.summary()

    def test_drops_vehicle_class(self, tmp_path):
        images, labels = _make_yolo_dataset(
            tmp_path, label_body="0 0.5 0.5 0.2 0.1\n1 0.5 0.5 0.8 0.8\n"
        )
        report = from_yolo_dir(images, labels, source="t", keep_classes=frozenset({0}))
        assert report.dropped_boxes == 4
        assert all(len(r.boxes) == 1 for r in report.records)
        assert "dropped boxes" in report.summary()

    def test_each_still_is_its_own_group_by_default(self, tmp_path):
        # Unrelated stills must not be lumped into one group, or the split
        # would put the whole dataset in a single bucket.
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t")
        assert len({r.group_key for r in report.records}) == 4

    def test_frame_grouping_when_requested(self, tmp_path):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        images.mkdir()
        labels.mkdir()
        for clip in range(2):
            for frame in range(3):
                name = f"clip{clip}_frame_{frame:03d}"
                Image.new("RGB", (100, 100)).save(images / f"{name}.jpg")
                (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        report = from_yolo_dir(images, labels, source="t", group_by_stem=True)
        assert len({r.group_key for r in report.records}) == 2

    def test_background_image_with_empty_label(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path, label_body="")
        report = from_yolo_dir(images, labels, source="t")
        assert len(report.records) == 4
        assert all(r.boxes == () for r in report.records)

    def test_file_name_is_relative_to_images_dir_by_default(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t")
        assert sorted(r.file_name for r in report.records)[0] == "img0.jpg"

    def test_path_root_makes_paths_composable_across_sources(self, tmp_path):
        # Regression: merging two sources into one manifest and exporting
        # against their shared parent failed, because each file_name was
        # relative to its own images_dir and so resolved to the wrong file.
        raw = tmp_path / "raw"
        for source in ("srcA", "srcB"):
            (raw / source).mkdir(parents=True)
            _make_yolo_dataset(raw / source, n=2)

        records = []
        for source in ("srcA", "srcB"):
            records += from_yolo_dir(
                raw / source / "images",
                source=source,
                id_prefix=f"{source}-",
                path_root=raw,
            ).records

        # Every recorded path resolves to a real file under the shared root.
        assert len(records) == 4
        for record in records:
            assert (raw / record.file_name).exists(), record.file_name
        assert len({r.file_name for r in records}) == 4

    def test_path_root_must_contain_images_dir(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        with pytest.raises(DatasetError, match="not a parent of"):
            from_yolo_dir(images, labels, source="t", path_root=tmp_path / "elsewhere")

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="images_dir does not exist"):
            from_yolo_dir(tmp_path / "nope", source="t")

    def test_missing_labels_directory_raises(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        with pytest.raises(DatasetError, match="labels_dir does not exist"):
            from_yolo_dir(images, source="t")
