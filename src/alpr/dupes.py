"""Near-duplicate detection across splits.

A test score is only meaningful if the test images are genuinely unseen. Two
things can break that without any bug in the splitting code:

1. Roboflow and similar platforms often export **augmented copies** of the
   same photograph — rotated, brightness-shifted, re-cropped — as separate
   images with unrelated filenames.
2. Public datasets get assembled from overlapping sources, so the same
   photograph can appear twice under different names.

`alpr.data.split` groups by `group_key` to stop video frames leaking, but it
cannot see either of these: with unrelated stills, every image is its own
group, so grouping is a no-op and near-duplicates split freely.

Detection is by **dHash** — a perceptual hash comparing adjacent pixel
brightness on a 9x8 greyscale thumbnail. It survives rescaling, compression
and mild brightness changes, which is exactly the family of transformations an
augmentation pipeline applies, while cryptographic hashing would see every
augmented copy as a completely different file.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from alpr.data.schema import ImageRecord, Split
from alpr.data.split import SplitAssignment

# 8x8 comparisons -> a 64-bit hash. Distances are in bits.
HASH_SIZE = 8

# Below this Hamming distance two images are treated as the same picture.
# Zero means byte-identical after thumbnailing; a handful of bits absorbs
# JPEG artefacts and small brightness shifts without matching genuinely
# different photographs of similar scenes.
DEFAULT_THRESHOLD = 5


def dhash(path: str | Path, hash_size: int = HASH_SIZE) -> int:
    """Perceptual hash of an image, as an integer.

    Compares each pixel with its right-hand neighbour on a greyscale
    thumbnail, so the result depends on structure rather than on exact pixel
    values or file bytes.
    """
    from PIL import Image

    with Image.open(path) as img:
        thumb = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        # tobytes() over getdata(): one byte per pixel in mode "L", and a
        # stable API — getdata() is deprecated for removal in Pillow 14.
        pixels = thumb.tobytes()

    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            left = pixels[offset + col]
            right = pixels[offset + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class DuplicatePair:
    """Two images that appear to be the same picture."""

    left: str
    right: str
    distance: int
    left_split: Split
    right_split: Split

    @property
    def crosses_splits(self) -> bool:
        return self.left_split is not self.right_split

    @property
    def contaminates_evaluation(self) -> bool:
        """True when a held-out image has a twin the model trained on.

        This is the case that inflates a score: val and test are supposed to
        be unseen, and a twin in train means they are not.
        """
        return self.crosses_splits and Split.TRAIN in (self.left_split, self.right_split)


@dataclass
class DuplicateReport:
    images_hashed: int
    unreadable: list[str] = field(default_factory=list)
    pairs: list[DuplicatePair] = field(default_factory=list)
    threshold: int = DEFAULT_THRESHOLD

    @property
    def cross_split(self) -> list[DuplicatePair]:
        return [p for p in self.pairs if p.crosses_splits]

    @property
    def contaminating(self) -> list[DuplicatePair]:
        return [p for p in self.pairs if p.contaminates_evaluation]

    def contaminated_images(self, split: Split) -> set[str]:
        """Held-out images in `split` that have a twin in train."""
        out: set[str] = set()
        for pair in self.contaminating:
            if pair.left_split is split:
                out.add(pair.left)
            if pair.right_split is split:
                out.add(pair.right)
        return out

    def report(self, split_totals: dict[Split, int] | None = None) -> str:
        lines = [
            f"images hashed        {self.images_hashed}",
            f"threshold            {self.threshold} bits (of 64)",
            f"duplicate pairs      {len(self.pairs)}",
            f"  within one split   {len(self.pairs) - len(self.cross_split)}",
            f"  across splits      {len(self.cross_split)}",
            f"  train<->held-out   {len(self.contaminating)}",
        ]
        if self.unreadable:
            lines.append(f"unreadable           {len(self.unreadable)}")

        for split in (Split.VAL, Split.TEST):
            affected = self.contaminated_images(split)
            line = f"{split.value} images with a twin in train: {len(affected)}"
            if split_totals and split_totals.get(split):
                line += f"  ({len(affected) / split_totals[split]:.1%} of {split_totals[split]})"
            lines.append(line)

        lines.append("")
        if not self.contaminating:
            lines.append("VERDICT: no train/held-out contamination — the scores stand.")
        else:
            lines.append(
                "VERDICT: held-out images have near-duplicates in train. Scores on "
                "those images measure memorization, not generalization."
            )
        return "\n".join(lines)


def find_duplicates(
    records: Sequence[ImageRecord],
    assignment: SplitAssignment,
    image_root: str | Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
) -> DuplicateReport:
    """Hash every image and report pairs that look like the same picture.

    Exact matches are found by grouping identical hashes, which is linear.
    Near-matches need pairwise comparison, but only between *distinct* hashes,
    so the quadratic part shrinks by however many exact duplicates exist.
    """
    image_root = Path(image_root)

    by_hash: dict[int, list[str]] = defaultdict(list)
    split_of: dict[str, Split] = {}
    unreadable: list[str] = []

    for record in records:
        if not record.file_name:
            unreadable.append(record.image_id)
            continue
        path = image_root / record.file_name
        try:
            value = dhash(path)
        except Exception:
            unreadable.append(record.image_id)
            continue
        by_hash[value].append(record.image_id)
        split_of[record.image_id] = assignment.of(record)

    pairs: list[DuplicatePair] = []

    def add(left: str, right: str, distance: int) -> None:
        pairs.append(
            DuplicatePair(
                left=left,
                right=right,
                distance=distance,
                left_split=split_of[left],
                right_split=split_of[right],
            )
        )

    # Identical hashes: every pair within the bucket is a duplicate.
    for ids in by_hash.values():
        for left, right in combinations(sorted(ids), 2):
            add(left, right, 0)

    # Near matches between distinct hashes.
    if threshold > 0:
        values = sorted(by_hash)
        for i, left_hash in enumerate(values):
            for right_hash in values[i + 1 :]:
                distance = hamming(left_hash, right_hash)
                if distance <= threshold:
                    for left in by_hash[left_hash]:
                        for right in by_hash[right_hash]:
                            add(left, right, distance)

    return DuplicateReport(
        images_hashed=len(split_of),
        unreadable=unreadable,
        pairs=pairs,
        threshold=threshold,
    )
