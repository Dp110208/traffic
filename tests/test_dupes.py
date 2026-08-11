"""Near-duplicate detection across splits."""

from __future__ import annotations

from PIL import Image, ImageEnhance

from alpr.data.schema import ImageRecord, Split
from alpr.data.split import SplitAssignment
from alpr.dupes import DuplicatePair, dhash, find_duplicates, hamming


def photo(path, seed=0, size=(200, 150)):
    """A structured image — flat colour would hash identically for everything."""
    import random

    rng = random.Random(seed)
    img = Image.new("RGB", size, (30, 30, 40))
    pixels = img.load()
    for _ in range(400):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        for dx in range(rng.randrange(5, 25)):
            for dy in range(rng.randrange(5, 25)):
                if x + dx < size[0] and y + dy < size[1]:
                    pixels[x + dx, y + dy] = (rng.randrange(256), rng.randrange(256), 200)
    img.save(path)
    return path


class TestDhash:
    def test_identical_images_hash_identically(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=1)
        Image.open(a).save(b)
        assert dhash(a) == dhash(b)

    def test_different_images_hash_differently(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=1)
        photo(b, seed=99)
        assert hamming(dhash(a), dhash(b)) > 5

    def test_survives_rescaling(self, tmp_path):
        # The point of a perceptual hash: an augmented copy is still the same
        # picture, where a cryptographic hash would see a different file.
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=7, size=(400, 300))
        Image.open(a).resize((200, 150)).save(b)
        assert hamming(dhash(a), dhash(b)) <= 5

    def test_survives_mild_brightness_change(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=3)
        ImageEnhance.Brightness(Image.open(a)).enhance(1.15).save(b)
        assert hamming(dhash(a), dhash(b)) <= 5

    def test_hash_is_64_bits(self, tmp_path):
        assert dhash(photo(tmp_path / "a.png")).bit_length() <= 64


class TestHamming:
    def test_identical(self):
        assert hamming(0b1010, 0b1010) == 0

    def test_counts_differing_bits(self):
        assert hamming(0b1111, 0b1010) == 2


def _setup(tmp_path, layout):
    """Build images and an assignment from {image_id: (seed, split)}."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    records, by_group = [], {}
    for image_id, (seed, split) in layout.items():
        photo(images / f"{image_id}.png", seed=seed)
        records.append(
            ImageRecord(image_id=image_id, width=200, height=150, file_name=f"{image_id}.png")
        )
        by_group[image_id] = split
    assignment = SplitAssignment(by_group=by_group, seed=0, ratios={Split.TRAIN: 1.0})
    return records, assignment, images


class TestFindDuplicates:
    def test_clean_dataset_has_no_pairs(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path,
            {"a": (1, Split.TRAIN), "b": (2, Split.VAL), "c": (3, Split.TEST)},
        )
        report = find_duplicates(records, assignment, images)
        assert report.pairs == []
        assert "no train/held-out contamination" in report.report()

    def test_detects_a_twin_across_splits(self, tmp_path):
        # The case that inflates a score: the same picture in train and test.
        records, assignment, images = _setup(
            tmp_path, {"train_img": (5, Split.TRAIN), "test_img": (5, Split.TEST)}
        )
        report = find_duplicates(records, assignment, images)
        assert len(report.contaminating) == 1
        assert report.contaminated_images(Split.TEST) == {"test_img"}
        assert "measure memorization" in report.report()

    def test_duplicates_within_one_split_are_not_contamination(self, tmp_path):
        # Wasteful, but it does not inflate a held-out score.
        records, assignment, images = _setup(
            tmp_path, {"a": (5, Split.TRAIN), "b": (5, Split.TRAIN)}
        )
        report = find_duplicates(records, assignment, images)
        assert len(report.pairs) == 1
        assert report.contaminating == []

    def test_val_test_overlap_is_flagged_but_not_contaminating(self, tmp_path):
        # Neither was trained on, so no memorization — still worth knowing.
        records, assignment, images = _setup(tmp_path, {"v": (9, Split.VAL), "t": (9, Split.TEST)})
        report = find_duplicates(records, assignment, images)
        assert len(report.cross_split) == 1
        assert report.contaminating == []

    def test_counts_hashed_images(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"a": (1, Split.TRAIN), "b": (2, Split.TEST)}
        )
        assert find_duplicates(records, assignment, images).images_hashed == 2

    def test_missing_file_is_reported_not_fatal(self, tmp_path):
        records, assignment, images = _setup(tmp_path, {"a": (1, Split.TRAIN)})
        (images / "a.png").unlink()
        report = find_duplicates(records, assignment, images)
        assert report.unreadable == ["a"]
        assert report.images_hashed == 0

    def test_threshold_zero_finds_only_exact_matches(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"a": (4, Split.TRAIN), "b": (4, Split.TEST)}
        )
        # Perturb b slightly so it is near- but not exactly-identical.
        ImageEnhance.Brightness(Image.open(images / "b.png")).enhance(1.1).save(images / "b.png")
        loose = find_duplicates(records, assignment, images, threshold=5)
        strict = find_duplicates(records, assignment, images, threshold=0)
        assert len(loose.pairs) >= len(strict.pairs)

    def test_report_includes_split_percentages(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"tr": (6, Split.TRAIN), "te": (6, Split.TEST)}
        )
        report = find_duplicates(records, assignment, images)
        text = report.report(split_totals={Split.TEST: 4})
        assert "25.0%" in text


class TestDuplicatePair:
    def test_within_split_is_not_cross_split(self):
        pair = DuplicatePair("a", "b", 0, Split.TRAIN, Split.TRAIN)
        assert pair.crosses_splits is False
        assert pair.contaminates_evaluation is False

    def test_train_to_test_contaminates(self):
        pair = DuplicatePair("a", "b", 0, Split.TRAIN, Split.TEST)
        assert pair.contaminates_evaluation is True

    def test_val_to_test_does_not_contaminate(self):
        pair = DuplicatePair("a", "b", 0, Split.VAL, Split.TEST)
        assert pair.crosses_splits is True
        assert pair.contaminates_evaluation is False
