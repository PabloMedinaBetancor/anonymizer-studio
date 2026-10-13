"""Face detection (YuNet) and the face rules for photos and video.

Ported from the earlier application (its face detector, the face half of its sweep and retry, and its photo
decision). tests/test_faces.py checks it against the frozen copy of that code.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .. import contract, geometry
from ..contract import (
    BOX_EXTEND,
    BOX_GAP,
    BOX_RETRY,
    BOX_SECOND,
    BOX_SWEEP,
    FACE,
    RETRY,
    Box,
    Detection,
    Judgement,
    ModelLoadError,
    PhotoContext,
    Rect,
    RetryPurpose,
    SweepPlan,
    TrackStats,
    Verdict,
)

MODEL_FILE = "face_detection_yunet_2023mar.onnx"

REASON_VERIFIED = "face.reason.verified"  # the second look scored high on its own
REASON_SEEN_OFTEN = "face.reason.seen_often"  # video: seen in many sweeps and not contradicted
REASON_LOW_VERIFY = "face.reason.low_verify"  # the second look found something, not enough to blur unasked
REASON_SWEEP_ONLY = "face.reason.sweep_only"  # a strong first score that the second look did not confirm
REASON_WEAK = "face.reason.weak"  # nothing supports it: dropped


@dataclass(frozen=True)
class FaceThresholds:
    """Every number of the face rules. All come from the earlier application,
    fixed by its rounds of 23/09/2026 on 400 listing photos and on the Calle Preciados clip."""

    # YuNet's own cut. The earlier application builds the network with this score and then asks for 0.3 / 0.35 on top of it, so the
    # lower minimums below never lower anything: no detection under 0.5 is ever returned.
    net_score: float = 0.5
    sweep_min: float = 0.5  # minimum score of a sweep detection
    verify_min: float = 0.3  # minimum of the VERIFY retry (the photo decision of the earlier application)
    recover_min: float = 0.35  # minimum of the RECOVER retry (the earlier application's retry default)
    # photos and tracks: BLUR when the second look scores at least this (plums reach 0.78-0.85, hence 0.84)
    photo_verify: float = 0.84
    # "Not sure" when the second look reaches this, or the first sweep score reached doubt_score
    doubt_verify: float = 0.70
    doubt_score: float = 0.80
    # video: faces are small and backlit, so a track seen in this many sweeps with this top score and a mild
    # second look also gets blurred (Preciados: +15 real faces for 5 smudges over junk in 99 s of another clip)
    video_views: int = 3
    video_score: float = 0.75
    video_verify: float = 0.60
    # second look: square region 3x the face, scaled so the face measures about 96 px, scale kept in range
    retry_region: float = 3.0
    retry_target_side: float = 96.0
    retry_scale: Tuple[float, float] = (0.5, 4.0)
    retry_iou: float = 0.15
    # a track is verified the first time and again (up to this many times) when seen 1.5x bigger
    max_verifications: int = 3
    reverify_growth: float = 1.5
    # YuNet keeps ~85 KB per distinct input size: padding to a multiple of this stops the process growing
    input_multiple: int = 64
    # Where a face can be (Options.unsure "placed", photos; measured on the face photos of the dev part, bench/bench_subtle.py): a face
    # smaller than this share of the long side of the picture is blurred by default only on a person (none of the 623 faces of the
    # dev part is smaller: the first percentile is 0.47 %), and a "Not sure" face is blurred by default only when its centre is in the head
    # part of a person the person finder found with at least `placed_person_score`, or when a second, independent opinion agrees: the heavy
    # model of the second look scored it `placed_second` (RetinaFace) or the first sweep scored it `placed_score` (round 2 of 2.2: the person finder
    # misses people in crowds and at the edge of the picture; 25 of the 623 dev faces were lost by the person rule alone, 22 are back with these two
    # numbers, fitted on one half of the dev photos and holding on the other), and the face is not smaller than `placed_min_side`.
    placed_min_side: float = 0.004
    placed_person_score: float = 0.5
    placed_second: float = 0.85
    placed_score: float = 0.70


ASP_THRESHOLDS = FaceThresholds()


class FaceModule:
    """YuNet behind the `contract.ClassModule` interface. `threads` is accepted for the contract; the network
    runs on OpenCV's own pool, which is process-wide and not set here."""

    kind = FACE
    blur_shape = "ellipse"
    # fraction of the box width, height added on each side, by where the box came from (the earlier application's face margins)
    video_margins: Dict[str, Tuple[float, float]] = {
        BOX_SWEEP: (0.12, 0.18),
        BOX_RETRY: (0.15, 0.20),
        BOX_SECOND: contract.FACE_BOX_SECOND_MARGIN,
        BOX_GAP: (0.22, 0.26),
        BOX_EXTEND: (0.35, 0.35),
    }
    photo_margin: Tuple[float, float] = (0.12, 0.18)  # the earlier application's photo margin for faces (fit="loose")
    PHOTO_MARGIN_TIGHT = (
        0.06,
        0.10,
    )  # fit="tight" (2.2): the same recall of faces on the dev part (94.1 % against 94.2 %), a third less area

    def __init__(
        self,
        threads: int = 0,
        models: Optional[str] = None,
        thresholds: FaceThresholds = ASP_THRESHOLDS,
        fit: str = "tight",
    ):
        self.th = thresholds
        if fit != "loose":
            self.photo_margin = self.PHOTO_MARGIN_TIGHT
        self.timings: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        # A YuNet error is counted, never taken as "nothing found": the frame or photo is then listed as not checked
        self.model_failures = 0
        path = os.path.join(contract.models_dir(models), MODEL_FILE)
        if not os.path.isfile(path):
            raise ModelLoadError(FACE, contract.models_missing(path))
        try:
            self._net = cv2.FaceDetectorYN.create(path, "", (320, 320), thresholds.net_score, 0.3, 5000)
        except cv2.error as e:
            raise ModelLoadError(FACE, f"cannot load {path}: {e}") from e
        self._input_size: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()

    # detection ---------------------------------------------------------------------------------------------

    def _infer(self, img: np.ndarray) -> np.ndarray:
        """Raw YuNet rows for `img`, padded at the bottom and right (so coordinates do not change) to a
        multiple of `input_multiple`."""
        m = self.th.input_multiple
        h0, w0 = img.shape[:2]
        h, w = -(-h0 // m) * m, -(-w0 // m) * m
        if (h, w) != (h0, w0):
            img = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        with self._lock:
            if self._input_size != (w, h):
                self._net.setInputSize((w, h))
                self._input_size = (w, h)
            try:
                _, rows = self._net.detect(img)
            except cv2.error:
                self.model_failures += 1
                return np.zeros((0, 15), np.float32)
        return np.zeros((0, 15), np.float32) if rows is None else rows

    def _detect(self, bgr: np.ndarray, region: Optional[Rect], scale: float, min_score: float) -> List[Detection]:
        """Faces in `region` (the whole image if None) looked at in `scale`: below 1 is faster and loses small
        faces, above 1 finds faces of 10-20 px."""
        height, width = bgr.shape[:2]
        x0, y0, x1, y1 = region or (0, 0, width, height)
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        img = bgr[y0:y1, x0:x1]
        ex = ey = 1.0
        if scale != 1.0:
            nw, nh = max(8, int(round((x1 - x0) * scale))), max(8, int(round((y1 - y0) * scale)))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (nw, nh), interpolation=interpolation)
            ex, ey = (x1 - x0) / nw, (y1 - y0) / nh
        found = []
        for row in self._infer(img):
            score = float(row[14])
            if score < min_score:
                continue
            fx, fy, fw, fh = (float(v) for v in row[:4])
            found.append(Detection(x0 + fx * ex, y0 + fy * ey, x0 + (fx + fw) * ex, y0 + (fy + fh) * ey, score, FACE))
        return found

    def _timed(self, key: str, started: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - started
        self.calls[key] = self.calls.get(key, 0) + 1

    def sweep(self, bgr: np.ndarray, plan: SweepPlan) -> List[Detection]:
        """The whole image downscaled to `plan.general_side` (big faces) plus full-resolution tiles (small ones)."""
        height, width = bgr.shape[:2]
        min_score = self.th.sweep_min if plan.min_score is None else plan.min_score
        found: List[Detection] = []
        started = time.perf_counter()
        scale = min(1.0, plan.general_side / max(width, height))
        found += self._detect(bgr, None, scale, min_score)
        self._timed("face_general", started)
        for tile in geometry.sweep_tiles(width, height, plan.tile_side, plan.tiles):
            started = time.perf_counter()
            found += self._detect(bgr, tile, 1.0, min_score)
            self._timed("face_tiles", started)
        return geometry.nms(found)

    def retry(self, bgr: np.ndarray, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        started = time.perf_counter()
        try:
            return self._second_look(bgr, expected, purpose)
        finally:
            self._timed("face_retry", started)

    def _second_look(self, bgr: np.ndarray, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        th = self.th
        height, width = bgr.shape[:2]
        region = geometry.region_around(expected, width, height, th.retry_region)
        side = max(expected[2] - expected[0], expected[3] - expected[1], 1.0)
        scale = float(np.clip(th.retry_target_side / side, *th.retry_scale))
        min_score = th.verify_min if purpose is RetryPurpose.VERIFY else th.recover_min
        best, best_iou = None, 0.0
        for cand in self._detect(bgr, region, scale, min_score):
            overlap = geometry.iou(cand.box(), expected)
            if overlap > best_iou:
                best, best_iou = cand, overlap
        if best is None or best_iou < th.retry_iou:
            return None
        best.source = RETRY
        return best

    def close(self) -> None:
        self._net = None

    # rules -------------------------------------------------------------------------------------------------

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        th = self.th
        second = verify.score if verify is not None else 0.0
        params = {"score": det.score, "verify": second}
        if second >= th.photo_verify:
            return Judgement(Verdict.BLUR, REASON_VERIFIED, params)
        if second >= th.doubt_verify:
            return Judgement(Verdict.CHECK, REASON_LOW_VERIFY, params)
        if det.score >= th.doubt_score:
            return Judgement(Verdict.CHECK, REASON_SWEEP_ONLY, params)
        return Judgement(Verdict.DROP, REASON_WEAK, params)

    def photo_placement(self, box: Box, judgement: Judgement, ctx: PhotoContext) -> Judgement:
        """Whether a face sits where a face can be (`FaceThresholds.placed_*`): a BLUR smaller than `placed_min_side` of the picture and on
        no person becomes a CHECK; a CHECK is blurred by default (placed 1.0) on a person or when the second look (`second_score`) or the first sweep
        (`item_score`, both put in the params by the engine) is sure enough and not smaller than `placed_min_side` (the pots and poles of a panorama are, and the heavy model is sure of them too); a BLUR that is placed stays one."""
        th = self.th
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        on_person = any(
            p.score >= th.placed_person_score
            and p.x0 - 0.1 * p.w <= cx <= p.x1 + 0.1 * p.w
            and p.y0 - 0.05 * p.h <= cy <= p.y0 + 0.45 * p.h
            for p in ctx.persons
        )
        params = dict(judgement.params)
        if judgement.verdict is Verdict.BLUR:
            small = min(box[2] - box[0], box[3] - box[1]) < th.placed_min_side * max(ctx.width, ctx.height)
            if small and not on_person:
                params["placed"] = 0.0
                return Judgement(Verdict.CHECK, judgement.reason, params)
            params["placed"] = 1.0
            return Judgement(Verdict.BLUR, judgement.reason, params)
        big_enough = min(box[2] - box[0], box[3] - box[1]) >= th.placed_min_side * max(ctx.width, ctx.height)
        agreed = big_enough and (
            params.get("second_score", 0.0) >= th.placed_second or params.get("item_score", 0.0) >= th.placed_score
        )
        params["placed"] = 1.0 if (on_person or agreed) and judgement.verdict is Verdict.CHECK else 0.0
        return Judgement(judgement.verdict, judgement.reason, params)

    def video_judgement(self, track: TrackStats) -> Judgement:
        th = self.th
        params = {"views": float(track.n_sweep_views), "score": track.max_score, "verify": track.best_verify}
        # a "Not sure" face seen in this many sweeps persists like a face does: blurred by default with Options.unsure "placed"
        params["placed"] = 1.0 if track.n_sweep_views >= th.video_views else 0.0
        if track.best_verify >= th.photo_verify:
            return Judgement(Verdict.BLUR, REASON_VERIFIED, params)
        if (
            track.n_sweep_views >= th.video_views
            and track.max_score >= th.video_score
            and track.best_verify >= th.video_verify
        ):
            return Judgement(Verdict.BLUR, REASON_SEEN_OFTEN, params)
        if track.best_verify >= th.doubt_verify:
            return Judgement(Verdict.CHECK, REASON_LOW_VERIFY, params)
        if track.max_score >= th.doubt_score:
            return Judgement(Verdict.CHECK, REASON_SWEEP_ONLY, params)
        return Judgement(Verdict.DROP, REASON_WEAK, params)

    def wants_verify(self, track: TrackStats, det: Detection) -> bool:
        """The first view of a track is verified, and again (up to `max_verifications` times) when the face is
        seen much bigger: a close face is judged better."""
        th = self.th
        if track.n_verify == 0:
            return True
        return track.n_verify < th.max_verifications and max(det.w, det.h) >= th.reverify_growth * track.verify_side

    def wants_recover(self, track: TrackStats) -> bool:
        """Only a track that already looks like a face is worth searching for when a sweep misses it."""
        return track.best_verify >= self.th.doubt_verify or track.max_score >= self.th.doubt_score
