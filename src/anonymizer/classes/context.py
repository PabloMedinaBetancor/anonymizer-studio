"""Persons as context for the second look.

The head zone of a person who has no face item is visited by the heavy model, so the engine has to know where the persons are.
They are found with the pass the product already uses for vehicles: the same YOLOX-nano file (which holds the 80 COCO classes), whole
picture plus 1,024 px tiles on large pictures, the same merge of repeats. A person is context only: never an item, never blurred,
never in a preview or a report row (SL2).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .. import contract, geometry
from ..contract import Detection, ModelLoadError
from .heavy import open_session, read_verified, verified_model
from .plates import VehicleDetector
from .rfdetr import PERSON_COLUMNS, RFDETR_MODEL, RFDetrDetector

PERSON = "person"  # kind of a person detection; context, never a blurred item
PERSON_CLASSES = (0,)  # COCO: person
PERSON_MODEL = "yolox_nano.onnx"
PERSON_SCORE = 0.30  # persons count from this score
PERSON_MODELS = (
    "yolox",
    "rfdetr",
)  # the detector of the person pass; yolox until the dev verdict says otherwise (PT1b)
TILE_SIDE = 1024  # as the vehicle pass: tiles only when the longer side is over TILE_TRIGGER
TILE_TRIGGER = 1400
TILE_OVERLAP = 0.25
MERGE_IOU = 0.5


class PersonDetector(VehicleDetector):
    """`VehicleDetector` with the person class instead of the four vehicle classes; its boxes carry kind PERSON."""

    def __init__(self, session, min_score: float = PERSON_SCORE):
        super().__init__(session)
        self.classes = np.array(PERSON_CLASSES)
        self.min_score = min_score

    def detect(self, bgr: np.ndarray, region=None, min_score: Optional[float] = None) -> List[Detection]:
        found = super().detect(bgr, region, self.min_score if min_score is None else min_score)
        for d in found:
            d.kind = PERSON
        return found

    def close(self) -> None:
        self.session = None


def load_person_detector(threads: int = 0, models: Optional[str] = None, person_model: str = "yolox"):
    """The person detector of `models` (hash-checked, CL3): from the YOLOX-nano file, or with `person_model="rfdetr"` from the RF-DETR nano
    file (same interface, lazy session, failures counted in `run_failures`). Raises ModelLoadError."""
    if person_model not in PERSON_MODELS:
        raise ValueError(f"person_model must be one of {PERSON_MODELS}, not {person_model!r}")
    if person_model == "rfdetr":
        return RFDetrDetector(threads, models, PERSON_COLUMNS, PERSON, PERSON_SCORE)
    path, digest = verified_model(PERSON, contract.models_dir(models), PERSON_MODEL)
    try:
        return PersonDetector(open_session(PERSON, read_verified(PERSON, path, digest), threads))
    except ModelLoadError:
        raise
    except Exception as exc:  # a file that is not the model it should be
        raise ModelLoadError(PERSON, f"{PERSON_MODEL}: {exc}") from exc


def find_persons(detector: PersonDetector, bgr: np.ndarray, use_tiles: bool = True) -> List[Detection]:
    """Every person in the picture: the whole picture and, on a large one, 1,024 px tiles, repeats merged as the vehicle pass merges them.
    The box of the pass is that of `PlateModule._find_vehicles`; a person is found twice only when two views disagree."""
    height, width = bgr.shape[:2]
    persons = detector.detect(bgr)
    if use_tiles and max(width, height) > TILE_TRIGGER:
        for tile in geometry.tiles(width, height, TILE_SIDE, TILE_OVERLAP):
            persons += detector.detect(bgr, tile)
        persons = geometry.nms(persons, MERGE_IOU, contained=None)
    return persons


class PersonFinder:
    """Finds the persons of a picture for the second look. The YOLOX-nano session opens at the first picture and closes with `close()`
    (SL8: nothing is resident before a zone needs it); the file is checked against `SHA256SUMS` when the finder is built."""

    def __init__(self, threads: int = 0, models: Optional[str] = None, person_model: str = "yolox"):
        if person_model not in PERSON_MODELS:
            raise ValueError(f"person_model must be one of {PERSON_MODELS}, not {person_model!r}")
        verified_model(PERSON, contract.models_dir(models), PERSON_MODEL if person_model == "yolox" else RFDETR_MODEL)
        self._threads, self._models, self._person_model = threads, models, person_model
        self._detector = None
        self._closed_failures = 0

    def persons(self, bgr: np.ndarray) -> List[Detection]:
        if self._detector is None:
            self._detector = load_person_detector(self._threads, self._models, self._person_model)
        return find_persons(self._detector, bgr)

    @property
    def run_failures(self) -> int:
        """Person model runs that raised and were taken as "no persons here" (the RF-DETR detector counts them; YOLOX-nano does not)."""
        return self._closed_failures + getattr(self._detector, "run_failures", 0)

    def close(self) -> None:
        if self._detector is not None:
            self._closed_failures += getattr(self._detector, "run_failures", 0)
            self._detector.close()
            self._detector = None


def head_zone(person: Sequence[float], widen: float = 0.10, share: float = 0.35) -> contract.Box:
    """The head zone of a person box: widened `widen` per side, the top `share` of its height."""
    x0, y0, x1, y1 = person
    w, h = x1 - x0, y1 - y0
    return (x0 - widen * w, y0, x1 + widen * w, y0 + share * h)
