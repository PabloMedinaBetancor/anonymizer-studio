"""Stand-ins for the heavy detector of the second look: they honour `contract.HeavyDetector` and find what the test says is there."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from anonymizer.classes.heavy import HeavyRunError, HeavyThresholds, _ZoneDetector
from anonymizer.contract import FACE, SECOND, Box, Detection

# The thresholds the tests of the rules use, so that they do not move when the tuned constants of the product do.
TEST_TH = HeavyThresholds(h_hi=0.9, h_lo=0.7, target_side=256, look_min=0.5)

Placed = Tuple[Box, float]  # a face of the picture: box and the score the fake gives it


class FakeHeavy:
    """Finds the faces of `placed` whose centre lies in the region it is asked about, at their exact box. Records every call; the calls
    listed in `fail_on` (0-based) raise HeavyRunError; `delay` makes every call that long."""

    name, version, sha256 = "fake-heavy", "0", "0" * 64
    slowest_piece = 0.0
    check = None

    def __init__(self, placed: Sequence[Placed] = (), fail_on: Optional[Set[int]] = None, delay: float = 0.0):
        self.placed = list(placed)
        self.fail_on = fail_on or set()
        self.delay = delay
        self.calls: List[Tuple[Box, float]] = []
        self.closed = False

    def detect(self, bgr, region: Box, scale: float, min_score: float) -> List[Detection]:
        number = len(self.calls)
        self.calls.append((tuple(region), scale))
        started = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        self.slowest_piece = max(self.slowest_piece, time.perf_counter() - started)
        if number in self.fail_on:
            raise HeavyRunError("fake failure")
        found = []
        for box, score in self.placed:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3] and score >= min_score:
                found.append(Detection(*box, score, FACE, SECOND))
        return found

    def close(self) -> None:
        self.closed = True


class Scripted(_ZoneDetector):
    """A zone detector whose `_run` returns what the test says, in the pixels of the piece it is given, and records the pieces."""

    name, version, sha256 = "scripted", "0", "0" * 64

    def __init__(self, rows, max_side: int = 640):
        self.rows, self.max_side, self.pieces = rows, max_side, []

    def _run(self, img: np.ndarray, min_score: float):
        self.pieces.append(img.shape[:2])
        return self.rows(len(self.pieces) - 1, img) if callable(self.rows) else self.rows

    def close(self) -> None:
        pass
