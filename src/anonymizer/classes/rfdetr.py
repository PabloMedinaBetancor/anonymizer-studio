"""RF-DETR nano as the vehicle and person detector.

The pilot measured it against YOLOX-nano as the vehicle stage of the text path: every EU plate lies on a detected vehicle at 100, 50 and 30 %
of the size, where YOLOX-nano misses vehicles at the small sizes. The model is the Apache-2.0 COCO export of the `open-image-models`
release (RF-DETR nano, 384 x 384, 91 COCO columns). It has the `VehicleDetector` interface of `plates.py` (`detect`, `session`), so the vehicle
stage and the person pass take it where they take YOLOX-nano; nothing switches by itself: `classes.load(..., vehicle_model="rfdetr")` and
`PersonFinder(person_model="rfdetr")` choose it, and the default stays YOLOX-nano until the dev verdict says otherwise.

How it differs from the YOLOX wrapper: it is lazy (the file is hash-checked when the wrapper is built and the session opens at the first picture,
from the very bytes that were checked, and is shared with the other RF-DETR detectors of the process), a run that raises is counted in `run_failures` and taken as "nothing found here" (the plate path
cannot be stopped by a context detector; the count is in the report), and the model has no NMS of its own, which the stage's merge of tiles
supplies as it does for YOLOX-nano.

The preprocessing is the one the pilot validated (`measure_vehicles.py`, class `RFDetr`): the picture or tile is squashed to a square
(no letterbox), RGB, ImageNet mean and deviation; the output is normalised centre boxes and 91 logits per query; the score of a query is the
largest sigmoid over the wanted COCO columns (car, motorcycle, bus, truck or person).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import contract
from ..contract import VEHICLE, Detection, ModelLoadError
from .heavy import open_session, read_verified, verified_model

log = logging.getLogger(__name__)

RFDETR_MODEL = "rf-detr-nano-384-coco.onnx"
RFDETR_SIDE = 384  # the input side of the nano export
VEHICLE_COLUMNS = (3, 4, 6, 8)  # COCO ids of the 91-column export: car, motorcycle, bus, truck
PERSON_COLUMNS = (1,)  # COCO id 1: person
DEFAULT_SCORE = 0.30  # the pilot's threshold, the same as YOLOX-nano's
MIN_REGION = 16  # a cut smaller than this is not looked at (as the YOLOX wrapper)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


_SESSIONS: Dict[Tuple[str, str, int], list] = {}  # (path, SHA-256, threads) -> [session, detectors holding it]
_SESSIONS_LOCK = threading.Lock()


def _acquire(kind: str, path: str, sha256: str, threads: int):
    """The session over the checked bytes of the file, opened by the first detector that asks and shared by the next ones (the vehicle
    pass and the person pass hold one copy of the weights, not two: about 300 MB each on a 3.3 GB PC)."""
    key = (os.path.abspath(path), sha256, threads)
    with _SESSIONS_LOCK:
        entry = _SESSIONS.get(key)
        if entry is None:
            session = open_session(kind, read_verified(kind, path, sha256), threads)
            try:
                names = [o.name for o in session.get_outputs()]
                if len(names) != 2:
                    raise ValueError(f"expected 2 outputs (boxes, logits), found {names}")
            except Exception as exc:  # a file that is not the model it should be
                raise ModelLoadError(kind, f"{os.path.basename(path)}: {exc}") from exc
            entry = _SESSIONS[key] = [session, 0]
        entry[1] += 1
        return key, entry[0]


def _release(key) -> None:
    with _SESSIONS_LOCK:
        entry = _SESSIONS.get(key)
        if entry is not None:
            entry[1] -= 1
            if entry[1] <= 0:
                del _SESSIONS[key]


def _clip_region(bgr: np.ndarray, region: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    height, width = bgr.shape[:2]
    if region is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


class RFDetrDetector:
    """RF-DETR nano on COCO, keeping the classes of `columns` and giving them as `kind` (VEHICLE by default; PERSON for the person pass).
    Built as RFDetrDetector(threads=0, models=None); raises ModelLoadError at build when the file is missing, unlisted or changed, and at
    the first picture when the bytes read then are not the ones checked or the runtime cannot open them."""

    name = "rf-detr-nano-384-coco"

    def __init__(
        self,
        threads: int = 0,
        models: Optional[str] = None,
        columns: Sequence[int] = VEHICLE_COLUMNS,
        kind: str = VEHICLE,
        min_score: float = DEFAULT_SCORE,
        file: str = RFDETR_MODEL,
    ):
        self._path, self.sha256 = verified_model(kind, contract.models_dir(models), file)
        self._threads = threads
        self._columns = tuple(int(c) for c in columns)
        self.kind = kind
        self.min_score = min_score
        self.run_failures = 0
        self._session = None
        self._key = None  # the shared session this detector holds, when it opened it itself
        self._closed = False
        self._lock = threading.Lock()
        self.size = RFDETR_SIDE

    @property
    def session(self):
        """The open session, or None before the first picture and after `close()`. Assigning None closes the detector (what the plate
        modules do in `close()` to every detector they hold); assigning a session is for tests."""
        return self._session

    @session.setter
    def session(self, value) -> None:
        if value is None:
            self.close()
        else:
            self._session = value  # a session of a test: not shared, not released

    def _open(self):
        if self._closed:
            raise RuntimeError(f"{self.kind} detector is closed")
        if self._session is None:
            self._key, self._session = _acquire(self.kind, self._path, self.sha256, self._threads)
        return self._session

    def _blob(self, img_bgr: np.ndarray) -> np.ndarray:
        n = self.size
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        shrink = min(rgb.shape[:2]) > n
        squashed = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        blob = ((squashed.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(blob, dtype=np.float32)

    def detect(
        self, bgr: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None, min_score: Optional[float] = None
    ) -> List[Detection]:
        """The objects of this detector's classes in `bgr` (or in `region`), in the pixels of `bgr`, from `min_score`. A run that raises is
        counted in `run_failures` and gives nothing; a model that cannot be loaded raises ModelLoadError."""
        min_score = self.min_score if min_score is None else min_score
        x0, y0, x1, y1 = _clip_region(bgr, region)
        if x1 - x0 < MIN_REGION or y1 - y0 < MIN_REGION:
            return []
        session = self._open()
        blob = self._blob(bgr[y0:y1, x0:x1])
        with self._lock:
            try:
                boxes, logits = session.run(None, {session.get_inputs()[0].name: blob})
                boxes, logits = np.asarray(boxes)[0], np.asarray(logits)[0]
                if not (np.isfinite(boxes).all() and np.isfinite(logits).all()):
                    raise ValueError(
                        "non-finite output"
                    )  # a corrupt run is a failure, not "nothing here" (RV-RFDETR F2)
                score = (1.0 / (1.0 + np.exp(-logits[:, self._columns].astype(np.float64)))).max(1)
            except Exception as exc:
                self.run_failures += 1
                log.warning("%s model failed on a %dx%d cut: %s", self.kind, x1 - x0, y1 - y0, exc)
                return []
        width, height = x1 - x0, y1 - y0
        found: List[Detection] = []
        for (cx, cy, bw, bh), s in zip(boxes[score >= min_score], score[score >= min_score], strict=True):
            bx0, by0 = max(0.0, (cx - bw / 2) * width), max(0.0, (cy - bh / 2) * height)
            bx1, by1 = min(float(width), (cx + bw / 2) * width), min(float(height), (cy + bh / 2) * height)
            if bx1 > bx0 and by1 > by0:
                found.append(
                    Detection(float(bx0 + x0), float(by0 + y0), float(bx1 + x0), float(by1 + y0), float(s), self.kind)
                )
        return found

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._session = None
            if self._key is not None:
                _release(self._key)
                self._key = None


def load_vehicle_detector(threads: int = 0, models: Optional[str] = None) -> RFDetrDetector:
    """The RF-DETR vehicle detector (car, motorcycle, bus, truck), checked against `SHA256SUMS`. Raises ModelLoadError."""
    return RFDetrDetector(threads, models, VEHICLE_COLUMNS, VEHICLE)
