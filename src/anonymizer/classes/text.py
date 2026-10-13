"""The text detector of the plate path: PP-OCRv6 detector (small) run through ONNX Runtime.

The model is the text detector of the PP-OCRv6 family (Apache-2.0, the file RapidOCR ships); only the file is used: this module
does the resize, the normalisation and the Differentiable Binarization decoding itself, with numpy and OpenCV, so the product
needs no OCR package. The decoding follows RapidOCR's (parity is measured against it in tests/test_text_models.py): a text region
is a connected area of the probability map above `thresh`, boxed by its minimum-area rectangle, scored by the mean probability
inside, and grown by the unclip rule. Regions come back as axis-aligned boxes in the pixels of the picture that was passed in.

Loaded lazily; the file is checked against `models/SHA256SUMS` when the detector is built and again when the bytes are read for
the session (CL3). A run that raises is counted in `run_failures` and taken as "nothing here"; the engine then marks the picture
"not checked" instead of exporting it as clean.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..contract import TEXT, Detection, models_dir
from .heavy import open_session, read_verified, verified_model

log = logging.getLogger(__name__)

TEXT_MODEL = "PP-OCRv6_det_small.onnx"
TEXT_KIND = "text"  # the `kind` ModelLoadError carries for this model
NAME = "PP-OCRv6_det_small"
VERSION = "PP-OCRv6-small"


@dataclass(frozen=True)
class TextThresholds:
    """The numbers of the detector. The box threshold is the `min_score` of each call, because the plate path chooses it (tau_lo)."""

    limit_side: int = 736  # a picture whose short side is below this is enlarged to it (RapidOCR's "min" limit)
    max_side: int = 2048  # and no side grows past this: a thin crop must not become a huge input
    thresh: float = (
        0.2  # a pixel is text above this probability (RapidOCR's default 0.3; 0.2 is the pilot's looser setting)
    )
    unclip_ratio: float = 2.0  # regions grow by area * ratio / perimeter (RapidOCR 1.6; the pilot's looser setting)
    max_candidates: int = 1000
    min_side: int = 3  # a candidate whose short side is below this (network pixels) is noise
    min_side_grown: int = 5  # and so is one that is still below this after growing
    min_region: int = 3  # a region of the picture not wider and taller than this many pixels is dropped
    mean: float = 0.5
    std: float = 0.5


DEFAULT_TEXT_THRESHOLDS = TextThresholds()


def _open_checked(path: str, expected: str, threads: int):
    """An ONNX Runtime session over the bytes of `path`, read once and hashed: what is checked is what is run (CL3, heavy.read_verified)."""
    return open_session(TEXT_KIND, read_verified(TEXT_KIND, path, expected), threads)


class TextDetector:
    """PP-OCRv6 detector (small): text regions of a BGR picture (contract.TextDetector)."""

    name = NAME
    version = VERSION

    def __init__(
        self, threads: int = 0, models: Optional[str] = None, thresholds: TextThresholds = DEFAULT_TEXT_THRESHOLDS
    ):
        self._path, self.sha256 = verified_model(TEXT_KIND, models_dir(models), TEXT_MODEL)
        self._threads = threads
        self.thresholds = thresholds
        self.run_failures = 0
        self._session = None
        self._closed = False
        self._lock = threading.Lock()

    def _open(self):
        if self._closed:
            raise RuntimeError("text detector is closed")
        if self._session is None:
            self._session = _open_checked(self._path, self.sha256, self._threads)
        return self._session

    def _input(self, bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """The network input: the picture enlarged so its short side reaches `limit_side` (and no side passes `max_side`), both sides
        rounded to a multiple of 32, in BGR order as PP-OCR's own pipeline feeds its detector (B: +3 plates on v2 dev), scaled to (-1, 1). Returns it with its height and width."""
        t = self.thresholds
        h, w = bgr.shape[:2]
        ratio = max(1.0, t.limit_side / min(h, w))
        ratio = min(ratio, t.max_side / max(h, w)) if max(h, w) * ratio > t.max_side else ratio
        new_h = max(32, int(round(int(h * ratio) / 32) * 32))
        new_w = max(32, int(round(int(w * ratio) / 32) * 32))
        img = cv2.resize(bgr, (new_w, new_h))
        blob = (img.astype(np.float32) / 255.0 - t.mean) / t.std
        return np.ascontiguousarray(blob.transpose(2, 0, 1)[None], dtype=np.float32), new_h, new_w

    def detect(self, bgr: np.ndarray, min_score: float) -> List[Detection]:
        """Text regions of `bgr` whose mean probability is at least `min_score`, kind TEXT, in the pixels of `bgr`."""
        h, w = bgr.shape[:2]
        if h < 8 or w < 8:
            return []
        blob, new_h, new_w = self._input(bgr)
        session = self._open()  # a model that cannot be loaded raises ModelLoadError: that is not a "nothing here"
        with self._lock:
            try:
                prob = np.asarray(session.run(None, {session.get_inputs()[0].name: blob})[0])[0, 0]
            except Exception as exc:
                self.run_failures += 1
                log.warning("text model failed on a %dx%d picture: %s", w, h, exc)
                return []
        if not np.isfinite(prob).all():  # NaN compares False with every threshold: it would read as "no text here"
            self.run_failures += 1
            log.warning("text model returned values that are not numbers on a %dx%d picture", w, h)
            return []
        return self.decode(prob, w, h, min_score)

    def decode(self, prob: np.ndarray, dest_w: int, dest_h: int, min_score: float) -> List[Detection]:
        """Differentiable Binarization decoding of a probability map into text regions of the destination picture."""
        t = self.thresholds
        map_h, map_w = prob.shape
        mask = cv2.dilate((prob > t.thresh).astype(np.uint8), np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        found: List[Detection] = []
        for contour in contours[: t.max_candidates]:
            (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
            if min(rw, rh) < t.min_side:
                continue
            score = _mean_inside(prob, cv2.boxPoints(((cx, cy), (rw, rh), angle)))
            if score < min_score:
                continue
            grow = rw * rh * t.unclip_ratio / (2.0 * (rw + rh))  # the offset the unclip rule gives a rectangle
            rw, rh = rw + 2 * grow, rh + 2 * grow
            if min(rw, rh) < t.min_side_grown:
                continue
            corners = cv2.boxPoints(((cx, cy), (rw, rh), angle))
            xs = np.clip(np.round(corners[:, 0] / map_w * dest_w), 0, dest_w)
            ys = np.clip(np.round(corners[:, 1] / map_h * dest_h), 0, dest_h)
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            if x1 - x0 > t.min_region and y1 - y0 > t.min_region:
                found.append(Detection(x0, y0, x1, y1, float(score), TEXT))
        return found

    def close(self) -> None:
        self._closed = True
        self._session = None


def _mean_inside(prob: np.ndarray, corners: np.ndarray) -> float:
    """Mean of `prob` inside the quadrilateral `corners` (RapidOCR's "fast" box score)."""
    h, w = prob.shape
    x0 = int(np.clip(np.floor(corners[:, 0].min()), 0, w - 1))
    x1 = int(np.clip(np.ceil(corners[:, 0].max()), 0, w - 1))
    y0 = int(np.clip(np.floor(corners[:, 1].min()), 0, h - 1))
    y1 = int(np.clip(np.ceil(corners[:, 1].max()), 0, h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    shifted = corners - np.array([x0, y0], np.float32)
    cv2.fillPoly(mask, shifted.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(prob[y0 : y1 + 1, x0 : x1 + 1], mask)[0])
