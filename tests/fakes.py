"""Stand-in class modules for the engine tests: they honour contract.ClassModule without any model.

A ColourModule "detects" blobs of one colour, so synthetic clips can carry faces and plates the fakes really find.
Its rules (verdicts, when to verify or retry, margins, shape) are those of the 1.04 release for its kind, copied here as test
data, so the engine can be tested before the real modules exist.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from anonymizer.contract import (
    BOX_EXTEND,
    BOX_GAP,
    BOX_RETRY,
    BOX_SWEEP,
    DEEP,
    FACE,
    PLATE,
    RETRY,
    SECOND,
    SWEEP,
    TEXT,
    VEHICLE,
    Box,
    Detection,
    Judgement,
    RetryPurpose,
    SuspectZone,
    SweepPlan,
    TrackStats,
    Verdict,
)
from anonymizer.geometry import iou, region_around

# The margins of 1.04 (video and photo)
V1_VIDEO_MARGINS = {
    FACE: {BOX_SWEEP: (0.12, 0.18), BOX_RETRY: (0.15, 0.20), BOX_GAP: (0.22, 0.26), BOX_EXTEND: (0.35, 0.35)},
    PLATE: {BOX_SWEEP: (0.10, 0.18), BOX_RETRY: (0.12, 0.22), BOX_GAP: (0.20, 0.30), BOX_EXTEND: (0.35, 0.45)},
}
V1_PHOTO_MARGIN = {FACE: (0.12, 0.18), PLATE: (0.10, 0.18)}
V1_SHAPE = {FACE: "ellipse", PLATE: "rect"}


def v1_video_judgement(t: TrackStats) -> Judgement:
    """The 1.04 rule that confirms a track or leaves it doubtful."""
    if t.kind == FACE:
        if t.best_verify >= 0.84 or (t.n_sweep_views >= 3 and t.max_score >= 0.75 and t.best_verify >= 0.60):
            return Judgement(Verdict.BLUR, "face.reason.confirmed")
        if t.best_verify >= 0.70 or t.max_score >= 0.80:
            return Judgement(Verdict.CHECK, "face.reason.unsure")
        return Judgement(Verdict.DROP, "face.reason.false_positive")
    if t.on_vehicle and (t.best_verify >= 0.60 or (t.n_views >= 3 and t.n_sweep_views >= 2 and t.max_score >= 0.50)):
        return Judgement(Verdict.BLUR, "plate.reason.on_vehicle")
    if t.on_vehicle:
        unsure = t.max_score >= 0.30
    else:
        unsure = (t.n_views >= 8 and t.best_verify >= 0.85) or (t.max_score >= 0.8 and t.best_verify >= 0.8)
    if unsure:
        return Judgement(Verdict.CHECK, "plate.reason.unsure")
    return Judgement(Verdict.DROP, "plate.reason.false_positive")


def v1_wants_verify(t: TrackStats, det: Detection) -> bool:
    """The 1.04 rule for when a sweep detection needs a VERIFY retry."""
    side = max(det.w, det.h)
    if t.kind == PLATE and not t.on_vehicle and t.n_views < 8:
        return False
    if t.n_verify == 0:
        return True
    return t.n_verify < 3 and side >= 1.5 * t.verify_side


def v1_wants_recover(t: TrackStats) -> bool:
    """The 1.04 rule for when a lost track deserves a retry."""
    if t.kind == FACE:
        return t.best_verify >= 0.70 or t.max_score >= 0.80
    return t.on_vehicle


def v1_photo_judgement(det: Detection, verify: Optional[Detection]) -> Judgement:
    """The 1.04 photo rule, for one detection and its VERIFY retry."""
    v = verify.score if verify is not None else 0.0
    if det.kind == FACE:
        if v >= 0.84:
            return Judgement(Verdict.BLUR, "face.reason.confirmed", {"verify": v})
        if v >= 0.70 or det.score >= 0.80:
            return Judgement(Verdict.CHECK, "face.reason.unsure", {"verify": v})
        return Judgement(Verdict.DROP, "face.reason.false_positive", {"verify": v})
    if det.on_vehicle and v >= 0.60:
        return Judgement(Verdict.BLUR, "plate.reason.on_vehicle", {"verify": v})
    if (det.on_vehicle and det.score >= 0.30) or (det.score >= 0.8 and v >= 0.8):
        return Judgement(Verdict.CHECK, "plate.reason.unsure", {"verify": v})
    return Judgement(Verdict.DROP, "plate.reason.false_positive", {"verify": v})


class ColourModule:
    """Finds blobs of `colour` (BGR, within `tolerance`) at least `min_side` px wide and high, and reports them `pad`
    px larger on each side, as loose as the 1.04 stand-in detector was."""

    def __init__(
        self,
        kind: str,
        colour: Tuple[int, int, int],
        tolerance: int = 12,
        score: float = 0.8,
        verify_score: float = 0.9,
        on_vehicle: bool = True,
        min_side: int = 4,
        pad: float = 3.0,
    ):
        self.kind = kind
        self.colour = np.array(colour, np.int16)
        self.tolerance = tolerance
        self.score = score
        self.verify_score = verify_score
        self.on_vehicle = on_vehicle if kind == PLATE else None
        self.min_side = min_side
        self.pad = pad
        self.blur_shape = V1_SHAPE[kind]
        self.video_margins = V1_VIDEO_MARGINS[kind]
        self.photo_margin = V1_PHOTO_MARGIN[kind]
        self.timings: Dict[str, float] = {}
        self.calls: Dict[str, int] = {"sweep": 0, "retry": 0}
        self.model_failures = 0
        self.plans: List[SweepPlan] = []
        self.closed = False

    def _blobs(self, bgr: np.ndarray, region: Optional[Sequence[int]] = None) -> List[Box]:
        import cv2

        x0, y0 = (region[0], region[1]) if region else (0, 0)
        img = bgr if region is None else bgr[region[1] : region[3], region[0] : region[2]]
        if img.size == 0:
            return []
        mask = (np.abs(img.astype(np.int16) - self.colour).max(axis=2) <= self.tolerance).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h = (int(v) for v in stats[i, :4])
            if w >= self.min_side and h >= self.min_side:
                p = self.pad
                boxes.append((float(x0 + x) - p, float(y0 + y) - p, float(x0 + x + w) + p, float(y0 + y + h) + p))
        return boxes

    def sweep(self, bgr, plan: SweepPlan) -> List[Detection]:
        self.calls["sweep"] += 1
        self.plans.append(plan)
        return [Detection(*b, self.score, self.kind, SWEEP, self.on_vehicle) for b in self._blobs(bgr)]

    def retry(self, bgr, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        self.calls["retry"] += 1
        h, w = bgr.shape[:2]
        region = region_around(expected, w, h, 3.0)
        best = max(self._blobs(bgr, region), key=lambda b: iou(b, expected), default=None)
        if best is None or iou(best, expected) < 0.15:
            return None
        return Detection(*best, self.verify_score, self.kind, RETRY, self.on_vehicle)

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        return v1_photo_judgement(det, verify)

    def video_judgement(self, track: TrackStats) -> Judgement:
        return v1_video_judgement(track)

    def wants_verify(self, track: TrackStats, det: Detection) -> bool:
        return v1_wants_verify(track, det)

    def wants_recover(self, track: TrackStats) -> bool:
        return v1_wants_recover(track)

    def close(self) -> None:
        self.closed = True


class FixedVehicles:
    """The vehicle detector's stand-in: always the same vehicles, the ones whose centre lies in the region asked for. `calls` counts
    the regions it was asked about (the whole picture is None)."""

    session = None

    def __init__(self, boxes: Sequence[Box], score: float = 0.9):
        self.boxes = [tuple(float(v) for v in b) for b in boxes]
        self.score = score
        self.calls: List[Optional[Sequence[int]]] = []

    def detect(self, bgr, region=None, min_score=None) -> List[Detection]:
        self.calls.append(region)
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = region or (0, 0, w, h)
        return [
            Detection(*b, self.score, VEHICLE)
            for b in self.boxes
            if x0 <= (b[0] + b[2]) / 2 < x1 and y0 <= (b[1] + b[3]) / 2 < y1
        ]


class ColourTextDetector:
    """The text detector's stand-in: finds blobs of the given colours in the picture it is given (an enlarged vehicle crop) and
    reports each as a text region with its colour's score, `pad` pixels larger on each side. `fail_on` lists the call numbers (from 1)
    that fail the way a model run does: counted in `run_failures`, nothing found. `crops` keeps the shape of every picture it saw."""

    name, version, sha256 = "fake-text", "1", "00" * 32

    def __init__(
        self, colours: Sequence[Tuple[Tuple[int, int, int], float]], tolerance: int = 12, pad: float = 1.0, fail_on=()
    ):
        self.colours = [(np.array(c, np.int16), s) for c, s in colours]
        self.tolerance = tolerance
        self.pad = pad
        self.fail_on = set(fail_on)
        self.run_failures = 0
        self.crops: List[Tuple[int, ...]] = []
        self.min_scores: List[float] = []
        self.closed = False

    def detect(self, bgr, min_score: float) -> List[Detection]:
        import cv2

        self.crops.append(bgr.shape)
        self.min_scores.append(min_score)
        if len(self.crops) in self.fail_on:
            self.run_failures += 1
            return []
        found = []
        for colour, score in self.colours:
            if score < min_score:
                continue
            mask = (np.abs(bgr.astype(np.int16) - colour).max(axis=2) <= self.tolerance).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, w, h = (int(v) for v in stats[i, :4])
                if w >= 4 and h >= 4:
                    p = self.pad
                    found.append(Detection(x - p, y - p, x + w + p, y + h + p, score, TEXT))
        return found

    def close(self) -> None:
        self.closed = True


class ColourHeavy:
    """The heavy detector's stand-in (contract.HeavyDetector): it finds the union of blobs of all its colours inside the region, so a face the
    light module sees only in part (one colour) is found whole. `pad` is 0: its boxes are the truth box. `fail_on` lists call numbers
    (from 1) that raise, as a model that crashes does."""

    name, version, sha256 = "fake-heavy", "1", "00" * 32

    def __init__(self, kind: str, colours: Sequence[Tuple[int, int, int]], score: float = 0.95, tolerance: int = 12):
        self.kind = kind
        self.colours = [np.array(c, np.int16) for c in colours]
        self.score = score
        self.tolerance = tolerance
        self.calls = 0
        self.on_call = None  # called with the call number before the work (a test may stop the job there)
        self.closed = False

    def detect(self, bgr, region, scale: float, min_score: float) -> List[Detection]:
        import cv2

        self.calls += 1
        if self.on_call is not None:
            self.on_call(self.calls)
        h, w = bgr.shape[:2]
        x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
        x1, y1 = min(w, int(region[2])), min(h, int(region[3]))
        if x1 <= x0 or y1 <= y0:
            return []
        img = bgr[y0:y1, x0:x1].astype(np.int16)
        mask = np.zeros(img.shape[:2], np.uint8)
        for colour in self.colours:
            mask |= (np.abs(img - colour).max(axis=2) <= self.tolerance).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        found = []
        for i in range(1, n):
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            if bw >= 4 and bh >= 4 and self.score >= min_score:
                found.append(Detection(x0 + x, y0 + y, x0 + x + bw, y0 + y + bh, self.score, self.kind, SWEEP))
        return found

    def close(self) -> None:
        self.closed = True


class SecondColourModule(ColourModule):
    """A ColourModule with a heavy look (contract.SecondLookModule). The light module finds its own colour; the heavy detector finds the
    union of `heavy_colours`. `zones` keeps every zone it was asked to visit, `deep_calls` counts whole-frame sweeps. Its video rule
    uses `best_second`: a track that a heavy view confirmed (score >= 0.9) is BLUR, as the rule of the real face module will."""

    def __init__(self, kind, colour, heavy_colours, heavy_score: float = 0.95, **kw):
        super().__init__(kind, colour, **kw)
        self.heavy_zone = ColourHeavy(kind, heavy_colours, heavy_score)
        self.heavy_deep = self.heavy_zone
        self.zones: List[SuspectZone] = []
        self.deep_calls = 0
        self.model_failures = 0
        self.fail_second = False

    def second_look(self, bgr, zone: SuspectZone) -> List[Detection]:
        self.zones.append(zone)
        self.calls["second"] = self.calls.get("second", 0) + 1
        h, w = bgr.shape[:2]
        bx0, by0, bx1, by1 = zone.box
        mx, my = (bx1 - bx0) * 0.25, (by1 - by0) * 0.25
        region = (max(0, bx0 - mx), max(0, by0 - my), min(w, bx1 + mx), min(h, by1 + my))
        if self.fail_second:
            self.model_failures += 1
            return []
        return [
            Detection(d.x0, d.y0, d.x1, d.y1, d.score, self.kind, SECOND)
            for d in self.heavy_zone.detect(bgr, region, 1.0, 0.0)
        ]

    def deep_sweep(self, bgr, plan: SweepPlan) -> List[Detection]:
        self.deep_calls += 1
        h, w = bgr.shape[:2]
        return [
            Detection(d.x0, d.y0, d.x1, d.y1, d.score, self.kind, DEEP)
            for d in self.heavy_zone.detect(bgr, (0, 0, w, h), 1.0, 0.0)
        ]

    def second_judgement(self, light, found) -> Judgement:
        return Judgement(Verdict.BLUR, f"{self.kind}.reason.second_found")

    def video_judgement(self, track: TrackStats) -> Judgement:
        if track.best_second >= 0.9:
            return Judgement(Verdict.BLUR, f"{self.kind}.reason.second_confirmed")
        return v1_video_judgement(track)


class DoubtfulColourModule(ColourModule):
    """Like ColourModule, but its class rule is never sure: everything it detects is "Not sure" (Verdict.CHECK), in photos
    and in video. The stand-in for the privacy-first tests. `placed` is what the class rule says about where
    it sits (judgement.params["placed"], read with Options.unsure "placed", the default since 2.2): True, the default, is a doubtful
    item that is blurred by default; False one that is listed and left visible."""

    def __init__(self, *args, placed: bool = True, **kw):
        super().__init__(*args, **kw)
        self.placed = 1.0 if placed else 0.0

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        return Judgement(Verdict.CHECK, f"{self.kind}.reason.unsure", {"placed": self.placed})

    def video_judgement(self, track: TrackStats) -> Judgement:
        return Judgement(Verdict.CHECK, f"{self.kind}.reason.unsure", {"placed": self.placed})


def doubtful_plate_module(**kw) -> DoubtfulColourModule:
    """Plates (white, as plate_module) that the class rule only ever calls "Not sure"."""
    return DoubtfulColourModule(PLATE, (250, 250, 250), **kw)


def plate_module(**kw) -> ColourModule:
    """Plates are white: the moving plate of tests.media.moving_plate_clip."""
    return ColourModule(PLATE, (250, 250, 250), **kw)


def face_module(**kw) -> ColourModule:
    """Faces are pure red ellipses drawn by the tests."""
    return ColourModule(FACE, (0, 0, 255), **kw)


class FixedModule:
    """Always "detects" the same boxes of its kind, and confirms them with a fixed VERIFY score: the photo tests'
    stand-in."""

    def __init__(
        self,
        kind: str,
        boxes: Sequence[Box],
        score: float = 0.9,
        verify_score: Optional[float] = 0.9,
        on_vehicle: bool = True,
        placed: bool = False,
    ):
        self.kind = kind
        self.placed = placed
        self.boxes = list(boxes)
        self.score = score
        self.verify_score = verify_score
        self.on_vehicle = on_vehicle if kind == PLATE else None
        self.blur_shape = V1_SHAPE[kind]
        self.video_margins = V1_VIDEO_MARGINS[kind]
        self.photo_margin = V1_PHOTO_MARGIN[kind]
        self.timings: Dict[str, float] = {}
        self.calls: Dict[str, int] = {"sweep": 0, "retry": 0}
        self.model_failures = 0

    def sweep(self, bgr, plan: SweepPlan) -> List[Detection]:
        self.calls["sweep"] += 1
        return [Detection(*b, self.score, self.kind, SWEEP, self.on_vehicle) for b in self.boxes]

    def retry(self, bgr, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        self.calls["retry"] += 1
        if self.verify_score is None:
            return None
        best = max(self.boxes, key=lambda b: iou(b, expected), default=None)
        if best is None or iou(best, expected) < 0.15:
            return None
        return Detection(*best, self.verify_score, self.kind, RETRY, self.on_vehicle)

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        return v1_photo_judgement(det, verify)

    def photo_placement(self, box, judgement: Judgement, ctx) -> Judgement:
        """What the real modules do with Options.unsure "placed": the stand-in says its doubtful items are placed only when built with `placed=True`."""
        if self.placed:
            return Judgement(judgement.verdict, judgement.reason, {**judgement.params, "placed": 1.0})
        return judgement

    def video_judgement(self, track: TrackStats) -> Judgement:
        j = v1_video_judgement(track)
        return Judgement(j.verdict, j.reason, {**j.params, "placed": 1.0}) if self.placed else j

    def wants_verify(self, track: TrackStats, det: Detection) -> bool:
        return v1_wants_verify(track, det)

    def wants_recover(self, track: TrackStats) -> bool:
        return v1_wants_recover(track)

    def close(self) -> None:
        pass


class Delayed:
    """A detector that takes its time, so a job can still be running when a test acts."""

    def __init__(self, inner, delay: float = 0.05):
        self._inner = inner
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def sweep(self, *args):
        time.sleep(self._delay)
        return self._inner.sweep(*args)


class Failing(Delayed):
    """A detector whose model fails on every run: nothing it was asked to look at was checked."""

    failures = 0

    @property
    def model_failures(self):
        return self.failures

    def sweep(self, *args):
        self.failures += 1
        return []


class Hanging(Delayed):
    """A detector that never comes back from its first sweep."""

    def sweep(self, *args):
        mark = os.environ.get("ANONYMIZER_TEST_HANG_MARK")
        if mark:  # tells the test the hang has begun, so a Stop can be sent to a worker that really is stuck
            open(mark, "w").close()
        time.sleep(3600)


class Crashing(Delayed):
    """A detector that takes the whole process down on its fifth sweep, as a native library might."""

    sweeps = 0

    def sweep(self, *args):
        self.sweeps += 1
        if self.sweeps == 5:
            import os

            os._exit(3)
        return self._inner.sweep(*args)


def plate_modules() -> list:
    """The ANONYMIZER_WORKER_MODULES hooks: what a worker process uses in place of the real detectors."""
    return [plate_module()]


def slow_plate_modules() -> list:
    return [Delayed(plate_module(), 0.08)]


def failing_plate_modules() -> list:
    return [Failing(plate_module())]


def hanging_plate_modules() -> list:
    return [Hanging(plate_module())]


def noisy_plate_modules() -> list:
    """Detectors whose library prints to stdout, as a native one might: the worker's messages must not be hurt."""
    import os

    class Noisy(Delayed):
        def sweep(self, *args):
            print("noise from a library")
            os.write(1, b"raw noise from a native library\n")
            return self._inner.sweep(*args)

    return [Noisy(plate_module(), 0.0)]


def crashing_plate_modules() -> list:
    return [Crashing(plate_module(), 0.0)]
