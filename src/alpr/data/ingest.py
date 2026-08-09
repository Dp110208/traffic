"""Adapters turning third-party datasets into manifest records.

Every source we use exports, or can be converted to, a YOLO directory —
Roboflow exports natively, UC3M-LP ships `labels2yolo.py`. So one adapter
covers the sources, and each gets its own `source` tag so licence provenance
survives into the manifest.

Ingest is deliberately lossy in one direction: source class ids are collapsed
to a single `license_plate` class. Sources that also label vehicles would
otherwise spend detector capacity on a class Phase 6 never uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from alpr.data.schema import DatasetError, ImageRecord, PlateBox, Region

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class IngestReport:
    """What happened during an ingest, so problems are visible not silent."""

    records: list[ImageRecord]
    skipped_no_label: list[str]
    skipped_bad_label: list[tuple[str, str]]
    dropped_boxes: int

    def summary(self) -> str:
        lines = [
            f"ingested       {len(self.records)} image(s)",
            f"plates         {sum(len(r.boxes) for r in self.records)}",
        ]
        if self.skipped_no_label:
            lines.append(f"no label file  {len(self.skipped_no_label)}")
        if self.skipped_bad_label:
            example = self.skipped_bad_label[0]
            lines.append(
                f"bad label      {len(self.skipped_bad_label)} (e.g. {example[0]}: {example[1]})"
            )
        if self.dropped_boxes:
            lines.append(f"dropped boxes  {self.dropped_boxes} (non-plate class)")
        return "\n".join(lines)


def image_size(path: Path) -> tuple[int, int]:
    """Read (width, height) from the header without decoding pixels.

    Decoding a full 4K JPEG to learn its dimensions costs ~50 ms; across
    20k images that is 15 minutes of a Colab session for two integers.
    """
    with Image.open(path) as img:
        return img.width, img.height


def _iter_images(images_dir: Path) -> Iterator[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            yield path


def parse_yolo_label(
    text: str, *, keep_classes: frozenset[int] | None
) -> tuple[list[PlateBox], int]:
    """Parse a YOLO label file body into boxes.

    Returns:
        The boxes kept, and how many were dropped for having a class id that
        is not a plate.

    Raises:
        DatasetError: on a malformed line — a silently skipped bad label is a
            silently smaller training set.
    """
    boxes: list[PlateBox] = []
    dropped = 0

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise DatasetError(f"line {lineno}: expected 5 fields, got {len(parts)}")
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError as exc:
            raise DatasetError(f"line {lineno}: {exc}") from exc

        if keep_classes is not None and class_id not in keep_classes:
            dropped += 1
            continue

        try:
            boxes.append(PlateBox(cx=cx, cy=cy, w=w, h=h))
        except DatasetError as exc:
            raise DatasetError(f"line {lineno}: {exc}") from exc

    return boxes, dropped


def from_yolo_dir(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    *,
    source: str,
    region: Region = Region.UNKNOWN,
    keep_classes: frozenset[int] | None = None,
    id_prefix: str | None = None,
    group_by_stem: bool = False,
    path_root: str | Path | None = None,
    strict: bool = True,
) -> IngestReport:
    """Ingest a YOLO-format directory.

    Args:
        images_dir: directory of images, searched recursively.
        labels_dir: matching `.txt` labels. Defaults to a sibling `labels/`.
        source: provenance tag — also the licence audit trail. Required.
        region: plate region for these images, when the source is single-country.
        keep_classes: source class ids that mean "plate". None keeps everything,
            which is right for single-class plate datasets.
        id_prefix: prefixed to every `image_id`. Two datasets both containing
            `001.jpg` would otherwise collide into one id.
        group_by_stem: derive the split group from the filename's frame suffix.
            Leave False for datasets of unrelated stills, where every image is
            already its own group.
        path_root: root that each record's `file_name` is stored relative to.
            Defaults to `images_dir`. Set it to the shared parent when merging
            several sources into one manifest — otherwise two sources both
            yield `img001.jpg` and the export cannot tell which file is meant.
        strict: raise on a malformed label instead of recording and skipping it.

    Returns:
        An `IngestReport`; the records still need splitting and exporting.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise DatasetError(f"images_dir does not exist: {images_dir}")

    labels_path = Path(labels_dir) if labels_dir else images_dir.parent / "labels"
    if not labels_path.is_dir():
        raise DatasetError(f"labels_dir does not exist: {labels_path}")

    root = Path(path_root) if path_root is not None else images_dir
    try:
        images_dir.resolve().relative_to(root.resolve())
    except ValueError:
        raise DatasetError(f"path_root {root} is not a parent of images_dir {images_dir}") from None

    records: list[ImageRecord] = []
    no_label: list[str] = []
    bad_label: list[tuple[str, str]] = []
    dropped_total = 0

    for image_path in _iter_images(images_dir):
        stem = image_path.stem
        label_file = labels_path / f"{stem}.txt"

        if not label_file.exists():
            # Not necessarily an error: some datasets omit labels for
            # background images. Recorded so the count is visible.
            no_label.append(stem)
            continue

        try:
            boxes, dropped = parse_yolo_label(
                label_file.read_text(encoding="utf-8"), keep_classes=keep_classes
            )
        except DatasetError as exc:
            if strict:
                raise DatasetError(f"{label_file}: {exc}") from exc
            bad_label.append((stem, str(exc)))
            continue

        dropped_total += dropped
        width, height = image_size(image_path)
        image_id = f"{id_prefix}{stem}" if id_prefix else stem

        records.append(
            ImageRecord(
                image_id=image_id,
                width=width,
                height=height,
                boxes=tuple(
                    PlateBox(b.cx, b.cy, b.w, b.h, text=b.text, region=region) for b in boxes
                ),
                file_name=str(image_path.resolve().relative_to(root.resolve())),
                group=None if group_by_stem else image_id,
                source=source,
            )
        )

    return IngestReport(
        records=records,
        skipped_no_label=no_label,
        skipped_bad_label=bad_label,
        dropped_boxes=dropped_total,
    )
