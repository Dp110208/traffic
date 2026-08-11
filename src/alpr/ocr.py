"""Reading plate crops.

Detection localizes; this reads. The two are kept apart deliberately — the
detector is region-agnostic, while everything about *reading* a plate is
country-specific, and that lives in `alpr.plates`.

**Preprocessing turned out not to matter, and that is a measured result.**

The expectation was the opposite: a plate crop is small and low-contrast, often
40 px tall against a recognition model trained on ~48 px text, so upscaling and
contrast normalization looked like free accuracy. Running the ablation in
`alpr.cer` over plates rendered at the dataset's median 119x40 px produced
*identical* CER for every variant — raw, upscaled, greyscaled, contrast-
stretched, sharpened. All 0.1343.

PaddleOCR resizes and normalizes each crop to its own input specification
before inference, so this module's work is redundant with the library's. The
steps are kept, switchable and defaulting on, because they cost microseconds
and would matter for a reader that does *not* normalize internally — but no
claim is made that they help here, because they measurably do not.

That measurement is on clean synthetic renders. Real crops carry motion blur,
oblique angles and poor light, where the answer may differ; the ablation exists
to be re-run on labelled real crops rather than assumed either way.

The general point stands regardless of which way it came out: a step that
cannot be shown to help is a step that should not be trusted.

**One thing this cannot do:** perspective correction. Undoing the slant of a
plate photographed off-axis needs its four corners, and the detector produces
an axis-aligned box — it does not know where the corners are. Rotating by a
guessed angle would be as likely to hurt as help. Proper de-skew needs an
oriented-box or segmentation model, which is a change to Phase 2, not
something preprocessing can fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpr.detect import Detection

# Recognition models are trained on text roughly this tall. Below it, strokes
# merge and characters that differ by a thin bar (8 vs B, 0 vs O) stop being
# separable at all.
TARGET_HEIGHT = 48

# Crops sit tight against the plate edge, and recognition does better with a
# little quiet space around the characters.
DEFAULT_PADDING = 0.08


class OcrError(RuntimeError):
    """Raised when the reader cannot be built or run."""


@dataclass(frozen=True)
class OcrResult:
    """One reading of one crop."""

    text: str
    confidence: float

    def __bool__(self) -> bool:
        return bool(self.text)


@dataclass(frozen=True)
class Preprocess:
    """Which preprocessing steps to apply, and how hard.

    Defaults are on, but see the module docstring: measured against PaddleOCR
    they changed nothing, because it normalizes crops itself. Treat these as
    insurance for a different reader, not as a source of accuracy here.
    """

    padding: float = DEFAULT_PADDING
    upscale: bool = True
    target_height: int = TARGET_HEIGHT
    grayscale: bool = True
    autocontrast: bool = True
    sharpen: bool = False

    @classmethod
    def none(cls) -> Preprocess:
        """The control condition: crop only, nothing applied."""
        return cls(
            padding=0.0,
            upscale=False,
            grayscale=False,
            autocontrast=False,
            sharpen=False,
        )

    def describe(self) -> str:
        on = [
            name
            for name, enabled in (
                ("pad", self.padding > 0),
                (f"upscale>={self.target_height}px", self.upscale),
                ("gray", self.grayscale),
                ("autocontrast", self.autocontrast),
                ("sharpen", self.sharpen),
            )
            if enabled
        ]
        return "+".join(on) if on else "raw"


# Module-level singleton rather than a call in a default argument. `Preprocess`
# is frozen, so the shared instance cannot be mutated by a caller.
DEFAULT_PREPROCESS = Preprocess()


def crop_plate(image, detection: Detection, padding: float = DEFAULT_PADDING):
    """Cut a plate out of a frame.

    Args:
        image: a PIL image of the whole frame.
        detection: normalized box.
        padding: fraction of box size added on each side, clamped to the frame.

    Returns:
        A PIL image of the plate.
    """
    width, height = image.size
    pad_x = detection.width * width * padding
    pad_y = detection.height * height * padding

    left = max(0, int(detection.x1 * width - pad_x))
    top = max(0, int(detection.y1 * height - pad_y))
    right = min(width, int(detection.x2 * width + pad_x))
    bottom = min(height, int(detection.y2 * height + pad_y))

    if right <= left or bottom <= top:
        raise OcrError(f"degenerate crop for detection {detection}")

    return image.crop((left, top, right, bottom))


def prepare(crop, settings: Preprocess | None = None):
    """Apply the enabled preprocessing steps to a crop."""
    from PIL import ImageOps

    settings = settings or DEFAULT_PREPROCESS
    out = crop

    if settings.upscale and out.height < settings.target_height:
        # Upscale rather than let the model see merged strokes. LANCZOS over
        # NEAREST: interpolation that invents smooth edges reads better than
        # one that produces blocky ones.
        from PIL import Image

        scale = settings.target_height / out.height
        out = out.resize(
            (max(1, int(out.width * scale)), settings.target_height),
            Image.Resampling.LANCZOS,
        )

    if settings.grayscale:
        out = out.convert("L")

    if settings.autocontrast:
        # Plates are photographed in headlights, dusk and direct sun; the
        # useful signal is the contrast between characters and background,
        # not the absolute brightness of either.
        out = ImageOps.autocontrast(out.convert("L") if out.mode != "L" else out)

    if settings.sharpen:
        from PIL import ImageFilter

        out = out.filter(ImageFilter.SHARPEN)

    return out


class PlateReader:
    """Reads plate crops with PaddleOCR.

    The model loads on first use, so constructing a reader stays cheap — the
    ablation builds several and only some get used.
    """

    def __init__(
        self,
        *,
        settings: Preprocess | None = None,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.settings = settings or Preprocess()
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    @property
    def model(self):
        if self._model is None:
            try:
                from paddleocr import TextRecognition
            except ImportError as exc:
                raise OcrError(
                    'paddleocr is not installed — run `pip install -e ".[ocr]"`'
                ) from exc

            kwargs: dict[str, Any] = {}
            if self.model_name:
                kwargs["model_name"] = self.model_name
            if self.device:
                kwargs["device"] = self.device
            self._model = TextRecognition(**kwargs)
        return self._model

    def read_image(self, image) -> OcrResult:
        """Read an already-cropped, already-prepared plate image."""
        import numpy as np

        array = np.array(image.convert("RGB"))
        results = self.model.predict(array)
        return _first_result(results)

    def read(self, image, detection: Detection) -> OcrResult:
        """Crop a plate out of a frame, prepare it, and read it."""
        crop = crop_plate(image, detection, self.settings.padding)
        return self.read_image(prepare(crop, self.settings))

    def read_all(self, image, detections: Sequence[Detection]) -> list[OcrResult]:
        return [self.read(image, d) for d in detections]

    def read_path(self, path: str | Path) -> OcrResult:
        """Read a plate image straight from disk, for the ablation."""
        from PIL import Image

        with Image.open(path) as img:
            return self.read_image(prepare(img.convert("RGB"), self.settings))


def _first_result(results) -> OcrResult:
    """Pull text and score out of whatever PaddleOCR returned.

    Defensive on purpose: PaddleOCR's result shape has changed across major
    versions, and a silent shape mismatch would read as "OCR found nothing"
    rather than as the integration error it is.
    """
    if not results:
        return OcrResult("", 0.0)

    first = results[0]
    if isinstance(first, dict):
        text = first.get("rec_text", "")
        score = first.get("rec_score", 0.0)
    else:
        text = getattr(first, "rec_text", "")
        score = getattr(first, "rec_score", 0.0)

    try:
        confidence = float(score)
    except (TypeError, ValueError):
        confidence = 0.0

    return OcrResult(str(text or ""), confidence)
