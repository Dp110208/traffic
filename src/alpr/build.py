"""One-call dataset construction.

Colab's free tier gives one GPU session, and everything under `/content` dies
with it. Splitting the work across notebooks assumed sessions persist — they
do not, so a notebook that depends on a previous notebook's output is a
notebook that usually cannot run.

`ensure_dataset()` fixes that: it rebuilds the dataset if it is missing and
returns immediately if it is not. Every notebook can call it, so any of them
can be the first thing run in a fresh session.

The source list lives here rather than in a notebook because it *is*
configuration — the licence of each dataset determines what may be
redistributed, and that belongs in reviewed, tested code.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from alpr.data import (
    ImageRecord,
    Region,
    export_yolo,
    from_roboflow_export,
    read_manifest,
    split_records,
    verify_split,
    write_manifest,
)
from alpr.data.stats import DatasetStats, compute_stats


class BuildError(RuntimeError):
    """Raised when the dataset cannot be built."""


@dataclass(frozen=True)
class RoboflowSource:
    """A Roboflow Universe dataset, with the licence that governs its reuse."""

    workspace: str
    project: str
    version: int
    directory: str
    region: Region
    source_tag: str
    id_prefix: str
    licence: str
    url: str

    @property
    def redistributable(self) -> bool:
        """Whether a derived dataset built from this may be published.

        CC BY permits it with attribution. **NoDerivatives does not** — and a
        re-split is exactly the derivative it forbids. NonCommercial is also
        treated as blocking, since a public portfolio repo cannot promise a
        downstream user will not use it commercially.

        Matched on word boundaries: splitting "CC BY-NC-ND 4.0" on hyphens
        yields "ND 4.0", not "ND", so a naive membership test silently marked
        NoDerivatives data as publishable.
        """
        return not re.search(r"\b(ND|NC)\b", self.licence.upper())


SOURCES: tuple[RoboflowSource, ...] = (
    RoboflowSource(
        workspace="e-hh49k",
        project="european-license-plates-tjviy",
        version=1,
        directory="roboflow-eu",
        region=Region.EUROPE,
        source_tag="roboflow-eu-plates",
        id_prefix="eu-",
        licence="CC BY 4.0",
        url="https://universe.roboflow.com/e-hh49k/european-license-plates-tjviy",
    ),
    RoboflowSource(
        workspace="nivu",
        project="indian-license-plate-knte7",
        version=1,
        directory="roboflow-in",
        region=Region.INDIA,
        source_tag="roboflow-in-plates",
        id_prefix="in-",
        licence="CC BY 4.0",
        url="https://universe.roboflow.com/nivu/indian-license-plate-knte7",
    ),
)


def attribution() -> str:
    """Attribution text for the derived dataset. CC BY requires it."""
    lines = ["Derived from Roboflow Universe datasets:"]
    for source in SOURCES:
        if source.redistributable:
            lines.append(f"- {source.project} ({source.licence}) — {source.url}")
    return "\n".join(lines)


def download_sources(raw_dir: str | Path, api_key: str) -> list[Path]:
    """Download every source that is not already present.

    Raises:
        BuildError: when the roboflow client is unavailable or a download fails.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise BuildError("roboflow is not installed — run `pip install roboflow`") from exc

    client = Roboflow(api_key=api_key)
    downloaded = []

    for source in SOURCES:
        target = raw_dir / source.directory
        if target.exists() and any(target.iterdir()):
            downloaded.append(target)
            continue
        try:
            (
                client.workspace(source.workspace)
                .project(source.project)
                .version(source.version)
                .download("yolov8", location=str(target))
            )
        except Exception as exc:
            # Leave nothing half-written: a partial export would ingest as a
            # smaller dataset rather than as an error.
            shutil.rmtree(target, ignore_errors=True)
            raise BuildError(f"downloading {source.project} failed: {exc}") from exc
        downloaded.append(target)

    return downloaded


def ingest_sources(raw_dir: str | Path) -> list[ImageRecord]:
    """Turn downloaded exports into manifest records."""
    raw_dir = Path(raw_dir)
    records: list[ImageRecord] = []

    for source in SOURCES:
        location = raw_dir / source.directory
        if not location.is_dir():
            raise BuildError(f"{location} is missing — download the sources first")
        report = from_roboflow_export(
            location,
            source=source.source_tag,
            region=source.region,
            id_prefix=source.id_prefix,
            path_root=raw_dir,
            strict=False,
        )
        records.extend(report.records)

    if not records:
        raise BuildError(f"no images found under {raw_dir}")
    return records


def build_dataset(
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/yolo",
    manifest_path: str | Path = "data/manifest.jsonl",
    *,
    seed: int = 0,
) -> tuple[Path, DatasetStats]:
    """Ingest, split, verify and export. Returns the data.yaml and the stats."""
    records = ingest_sources(raw_dir)
    write_manifest(records, manifest_path)

    records = read_manifest(manifest_path)
    assignment = split_records(records, seed=seed)
    verify_split(records, assignment)

    result = export_yolo(records, assignment, raw_dir, out_dir)
    return result.data_yaml, compute_stats(records, assignment)


def dataset_ready(data_yaml: str | Path = "data/yolo/data.yaml") -> bool:
    """Whether a usable export already exists."""
    from alpr.train import TrainingError, validate_dataset

    try:
        validate_dataset(data_yaml)
    except TrainingError:
        return False
    return True


def ensure_dataset(
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path = "data/yolo",
    manifest_path: str | Path = "data/manifest.jsonl",
    *,
    api_key: str | None = None,
    seed: int = 0,
    force: bool = False,
) -> Path:
    """Return a usable `data.yaml`, building the dataset if it is missing.

    Safe to call at the top of any notebook: it is a no-op when the dataset is
    already there, and a full rebuild when it is not.

    Args:
        api_key: Roboflow key. Read from the environment or Colab secrets when
            omitted — and only needed if a download is actually required.
        force: rebuild even if an export already exists.
    """
    data_yaml = Path(out_dir) / "data.yaml"

    if not force and dataset_ready(data_yaml):
        return data_yaml

    if not all((Path(raw_dir) / s.directory).is_dir() for s in SOURCES):
        if api_key is None:
            from alpr.env import ROBOFLOW_KEY_VAR, get_credential

            api_key = get_credential(ROBOFLOW_KEY_VAR)
        download_sources(raw_dir, api_key)

    data_yaml, _ = build_dataset(raw_dir, out_dir, manifest_path, seed=seed)
    return data_yaml
