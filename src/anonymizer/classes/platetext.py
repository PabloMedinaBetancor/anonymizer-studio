"""The plate reader: a second, independent opinion on a text region of a vehicle (2.2, round 2).

The text detector says "there is writing here"; it cannot say whether the writing is a plate or the name of the car, a taxi sign or a
company on a van. The reader is PP-OCR's Latin recogniser (Apache-2.0, the file RapidOCR ships), run on the region's crop: what it reads
tells a plate (letters and digits in a short run: "4314 KMJ") from a word ("TAXI", "RENAULT", "TRANSIT"), and a region it reads as nothing
at all is no worse than before. It decides nothing alone: `TextPlateModule.photo_placement` uses it to take a region out of the default
blur (a word) and to bring one in (a plate that the position rule rejected). Like the detector it is checked against `models/SHA256SUMS`
before it is read (CL3), built from the checked bytes, opened at the first region and run at 48 px of height as PaddleOCR does.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from ..contract import models_dir
from .heavy import open_session, read_verified, verified_model

log = logging.getLogger(__name__)

READER_MODEL = "latin_PP-OCRv5_rec_mobile.onnx"
READER_KIND = "plate_reader"  # the `kind` ModelLoadError carries for this model
NAME = "PP-OCRv5_latin_rec_mobile"
HEIGHT = 48  # the input height of the recogniser
MAX_WIDTH = 960
SHRINK = (
    0.12,
    0.22,
)  # the second reading: the region without its margins (the detector's boxes are grown past the writing)


@dataclass(frozen=True)
class Reading:
    """What the recogniser read in a region: the text, its mean confidence (0 when it read nothing) and the characters it is made of."""

    text: str = ""
    conf: float = 0.0

    @property
    def alnum(self) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", self.text)

    @property
    def digits(self) -> int:
        return sum(ch.isdigit() for ch in self.alnum)

    @property
    def letters(self) -> int:
        return sum(ch.isalpha() for ch in self.alnum)


NOTHING = Reading()


class PlateReader:
    """PP-OCR Latin recogniser (ONNX Runtime, CTC greedy decoding). Built as PlateReader(threads=0, models=None)."""

    name = NAME

    def __init__(self, threads: int = 0, models: Optional[str] = None):
        self._path, self.sha256 = verified_model(READER_KIND, models_dir(models), READER_MODEL)
        self._threads = threads
        self.run_failures = 0
        self._session = None
        self._chars = None
        self._closed = False
        self._lock = threading.Lock()

    def _open(self):
        if self._closed:
            raise RuntimeError("plate reader is closed")
        if self._session is None:
            self._session = open_session(
                READER_KIND, read_verified(READER_KIND, self._path, self.sha256), self._threads
            )
            meta = self._session.get_modelmeta().custom_metadata_map.get("character", "")
            self._chars = [""] + meta.split("\n")  # index 0 is the CTC blank
        return self._session

    def read(self, bgr: np.ndarray) -> Tuple[Reading, Reading]:
        """Two readings of `bgr`: the whole crop and the crop without its margins (the detector's boxes are grown past the writing)."""
        h, w = bgr.shape[:2]
        out = []
        for sx, sy in ((0.0, 0.0), SHRINK):
            out.append(self._read_one(bgr[int(h * sy) : h - int(h * sy), int(w * sx) : w - int(w * sx)]))
        return out[0], out[1]

    def _read_one(self, bgr: np.ndarray) -> Reading:
        h, w = bgr.shape[:2]
        if h < 4 or w < 4:
            return NOTHING
        new_w = max(16, min(MAX_WIDTH, int(round(HEIGHT * w / h))))
        img = cv2.resize(bgr, (new_w, HEIGHT), interpolation=cv2.INTER_CUBIC if h < HEIGHT else cv2.INTER_AREA)
        blob = ((img[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
        session = self._open()
        with self._lock:
            try:
                out = np.asarray(session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(blob)})[0])[0]
            except Exception as exc:
                self.run_failures += 1
                log.warning("plate reader failed on a %dx%d crop: %s", w, h, exc)
                return NOTHING
        idx, conf = out.argmax(1), out.max(1)
        chars, scores, prev = [], [], 0
        for i, c in zip(idx, conf, strict=True):
            if i != 0 and i != prev and i < len(self._chars):
                chars.append(self._chars[i])
                scores.append(float(c))
            prev = i
        return Reading("".join(chars), float(np.mean(scores))) if chars else NOTHING

    def close(self) -> None:
        self._closed = True
        self._session = None


def best_of(readings: Tuple[Reading, Reading]) -> Reading:
    """The more confident of two readings."""
    return max(readings, key=lambda r: r.conf)


def plate_like(reading: Reading, min_conf: float, min_digits: int, min_len: int = 4, max_len: int = 9) -> bool:
    """A short run of letters and digits with at least `min_digits` digits, read with at least `min_conf`: what a plate reads like."""
    return reading.conf >= min_conf and min_len <= len(reading.alnum) <= max_len and reading.digits >= min_digits


def word_like(reading: Reading, min_conf: float, min_letters: int) -> bool:
    """Letters only (at least `min_letters`), no dash, read with at least `min_conf`: a name, a sign, lettering. A dash is kept out of this
    on purpose: German plates are written "WOB-VWMKL"."""
    return (
        reading.conf >= min_conf and reading.digits == 0 and reading.letters >= min_letters and "-" not in reading.text
    )
