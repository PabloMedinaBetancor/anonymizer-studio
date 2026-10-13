"""Licence plates, with vehicles as context.

Port of the plate and vehicle parts of the earlier application. Thresholds and
geometry are the ones measured there; parity with the frozen copy is checked in tests/test_plates*.py. Change a
number only with a new measurement behind it.

The v1 plate model is YOLOv9-t 384 (open-image-models) run with onnxruntime; it is no longer shipped (licence not cleared, PT4 passed
by the text path) and `PlateModule` is a reference path that needs the file fetched by scripts/fetch_reference_plate_model.py. The
vehicle model is YOLOX-nano (Megvii, Apache 2.0). Vehicles are never blurred: a plate is only trusted on its own when it sits on one.

Version 2 adds a second plate path: the same vehicle stage, then the OCR family's text detector on
each enlarged vehicle crop, and every text region on the vehicle is blurred (or listed as doubtful, which is blurred too). Both
paths coexist behind the contract; the text path is the product (PT4 passed on the sealed plates) and the YOLOv9-t path is a
reference for the paired comparison.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import geometry
from ..contract import (
    BAND,
    BOX_EXTEND,
    BOX_GAP,
    BOX_RETRY,
    BOX_SWEEP,
    PLATE,
    RETRY,
    VEHICLE,
    Box,
    Detection,
    Judgement,
    ModelLoadError,
    PhotoContext,
    RetryPurpose,
    SweepPlan,
    TrackStats,
    Verdict,
    models_dir,
    models_missing,
)
from .text import DEFAULT_TEXT_THRESHOLDS, TextDetector, TextThresholds

log = logging.getLogger(__name__)

PLATE_MODEL = "yolo-v9-t-384-license-plates-end2end.onnx"
VEHICLE_MODEL = "yolox_nano.onnx"
VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck


@dataclass(frozen=True)
class PlateThresholds:
    """Every number the plate path decides with. Where each one comes from is in the notes of the earlier application (rounds of
    23/09/2026)."""

    # Detector
    sweep_score: float = 0.25  # minimum score of a sweep detection (the earlier application's sweep, plate threshold)
    recover_score: float = 0.15  # RECOVER retry (the earlier application's retry)
    verify_score: float = (
        0.10  # VERIFY retry (the photo decision and the video verification of the earlier application)
    )
    retry_region: float = 4.0  # the retry looks at a square this many times the expected box...
    retry_region_min: int = 96  # ...and at least this many pixels wide
    retry_iou: float = 0.15  # a retry result must overlap the expected box this much
    max_height: float = 0.35  # model-input geometry filter: taller than this share of the input is no plate
    max_width: float = 0.90  # wider than this share of the input
    min_aspect: float = 1.1  # width / height below this (plates are wider than tall; motorbikes ~1.1)
    vehicle_score: float = 0.30  # YOLOX-nano minimum score
    vehicle_nms_iou: float = 0.45
    vehicle_margin: float = 0.15  # a plate counts as on a vehicle up to this share outside its box
    vehicle_region: float = 1.15  # plates are searched again inside each vehicle, this many times its size
    vehicle_min_size: Tuple[int, int] = (60, 40)  # only vehicles at least this wide and tall are searched
    # Decision
    confirm_verify: float = 0.60  # BLUR on a vehicle when the VERIFY retry scores this much
    doubt_score: float = 0.30  # on a vehicle but not confirmed: CHECK from this score
    views_to_blur: int = 3  # video: on a vehicle, seen this many times...
    sweep_views_to_blur: int = 2  # ...at least this many by the sweep, not by retries...
    views_score: float = 0.50  # ...with this best score: BLUR without the retry (dark, blurred plates)
    off_vehicle_views: int = 8  # video, no vehicle: seen this long and very verified: CHECK
    off_vehicle_verify: float = 0.85
    strong_score: float = 0.80  # no vehicle, score and VERIFY this high: CHECK
    strong_verify: float = 0.80
    reverify_growth: float = 1.5  # a track is verified again when its view is this much larger...
    max_verifications: int = 3  # ...up to this many times


DEFAULT_THRESHOLDS = PlateThresholds()

# Share of width and height added on each side, by where the box came from (the margins of the earlier application).
VIDEO_MARGINS: Dict[str, Tuple[float, float]] = {
    BOX_SWEEP: (0.10, 0.18),
    BOX_RETRY: (0.12, 0.22),
    BOX_GAP: (0.20, 0.30),
    BOX_EXTEND: (0.35, 0.45),
}
PHOTO_MARGIN = (0.10, 0.18)  # the margin of fit="loose" (2.1)
# Fit "tight" (2.2). The text region is the writing, and the plate is a little bigger than its writing (its frame, the blue strip of a European one, the
# bottom edge): round 1 cut the margin to 0.06 / 0.10 and the hidden share of the readable plates fell with it. Measured on the plate photos of the dev
# part, the streets and OpenALPR EU sets (bench/bench_subtle.py; the plate counts as hidden when 95 % of it lies under the blur): 0.06 / 0.10 hides 44 of
# the 58 readable dev plates and 102 of 108 EU ones, 0.12 / 0.20 hides 46 and 107, 0.15 / 0.25 hides 47 and 108 (a bigger margin than that gains one more
# and costs a third more area). The margin is added to a text region that is already the size of the plate, so it is a few pixels on a small plate.
PHOTO_MARGIN_TIGHT = (0.15, 0.25)


def _open_session(kind: str, path: str, threads: int):
    try:
        import onnxruntime as ort
    except Exception as exc:  # on Windows usually a missing DLL, which is an OSError and not an ImportError
        raise ModelLoadError(kind, f"cannot load onnxruntime: {type(exc).__name__}: {exc}") from exc
    if not os.path.isfile(path):
        raise ModelLoadError(kind, models_missing(path))
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    # No CPU memory arena: it keeps big blocks between calls and the process passed 500 MB on a 3.3 GB PC.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    try:
        return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ModelLoadError(kind, f"{os.path.basename(path)}: {exc}") from exc


def _clip_region(bgr: np.ndarray, region: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = region or (0, 0, width, height)
    return max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1))


class PlateDetector:
    """YOLOv9-t 384 with non-maximum suppression inside the model. Input is RGB in 0..1 on a grey letterbox."""

    def __init__(self, session, thresholds: PlateThresholds = DEFAULT_THRESHOLDS, min_score: Optional[float] = None):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.size = int(session.get_inputs()[0].shape[2])
        self.thresholds = thresholds
        self.min_score = thresholds.sweep_score if min_score is None else min_score
        self.run_failures = 0
        self._lock = threading.Lock()

    def _letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        h, w = img.shape[:2]
        n = self.size
        ratio = min(n / h, n / w)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        pad_w, pad_h = (n - new_w) / 2, (n - new_h) / 2
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, ratio, pad_w, pad_h

    def detect(
        self, bgr: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None, min_score: Optional[float] = None
    ) -> List[Detection]:
        min_score = self.min_score if min_score is None else min_score
        height, width = bgr.shape[:2]
        x0, y0, x1, y1 = _clip_region(bgr, region)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        if self.session is None:
            raise RuntimeError("plate detector is closed")
        img, ratio, pad_w, pad_h = self._letterbox(bgr[y0:y1, x0:x1])
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.0)
        with self._lock:
            try:
                raw = self.session.run(None, {self.input_name: blob})[0]
            except Exception as exc:
                # The model's built-in NMS fails now and then on odd crops. The area is then unchecked, which is a
                # possible miss: it is logged and counted so the engine and the benchmark can report it.
                self.run_failures += 1
                log.warning("plate model failed on a %dx%d region at (%d, %d): %s", x1 - x0, y1 - y0, x0, y0, exc)
                return []
        t = self.thresholds
        n = float(self.size)
        found = []
        for row in np.asarray(raw).reshape(-1, 7):
            score = float(row[6])
            if score < min_score:
                continue
            # The model sometimes returns a box that fills nearly its whole input with a high score, even on noise;
            # that is never a plate. Measured in the model's input space, before scaling back.
            box_w, box_h = float(row[3] - row[1]), float(row[4] - row[2])
            if box_h <= 0 or box_w <= 0 or box_h > t.max_height * n or box_w > t.max_width * n:
                continue
            if box_w / box_h < t.min_aspect:
                continue
            bx0, by0 = (row[1] - pad_w) / ratio + x0, (row[2] - pad_h) / ratio + y0
            bx1, by1 = (row[3] - pad_w) / ratio + x0, (row[4] - pad_h) / ratio + y0
            bx0, by0, bx1, by1 = max(0.0, bx0), max(0.0, by0), min(float(width), bx1), min(float(height), by1)
            if bx1 - bx0 >= 2 and by1 - by0 >= 2:
                found.append(Detection(float(bx0), float(by0), float(bx1), float(by1), score, PLATE))
        return found


class VehicleDetector:
    """YOLOX-nano on COCO, keeping cars, motorcycles, buses and trucks. Plates on window rows, wheels or signs are
    off any vehicle, which is what tells them from real ones."""

    def __init__(self, session, thresholds: PlateThresholds = DEFAULT_THRESHOLDS):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.size = int(session.get_inputs()[0].shape[2])
        self.thresholds = thresholds
        grids, strides = [], []
        for stride in (8, 16, 32):
            k = self.size // stride
            gy, gx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            grids.append(np.stack((gx, gy), 2).reshape(-1, 2))
            strides.append(np.full((k * k, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)
        self.classes = np.array(VEHICLE_CLASSES)
        self._lock = threading.Lock()

    def detect(
        self, bgr: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None, min_score: Optional[float] = None
    ) -> List[Detection]:
        min_score = self.thresholds.vehicle_score if min_score is None else min_score
        x0, y0, x1, y1 = _clip_region(bgr, region)
        if x1 - x0 < 16 or y1 - y0 < 16:
            return []
        if self.session is None:
            raise RuntimeError("vehicle detector is closed")
        img = bgr[y0:y1, x0:x1]
        n = self.size
        ratio = min(n / img.shape[0], n / img.shape[1])
        new_w, new_h = max(1, int(img.shape[1] * ratio)), max(1, int(img.shape[0] * ratio))
        canvas = np.full((n, n, 3), 114, np.uint8)  # YOLOX: top-left, grey 114, BGR 0-255
        canvas[:new_h, :new_w] = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        )
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        with self._lock:
            out = self.session.run(None, {self.input_name: blob})[0][0]
        centres = (out[:, :2] + self.grid) * self.strides
        sizes = np.exp(np.clip(out[:, 2:4], -10, 10)) * self.strides
        class_scores = out[:, 4:5] * out[:, 5 + self.classes]
        best = class_scores.argmax(1)
        score = class_scores[np.arange(len(class_scores)), best]
        keep = score >= min_score
        if not keep.any():
            return []
        centres, sizes, score = centres[keep], sizes[keep], score[keep]
        boxes = np.concatenate([centres - sizes / 2, sizes], 1) / ratio
        picked = cv2.dnn.NMSBoxes(boxes.tolist(), score.tolist(), min_score, self.thresholds.vehicle_nms_iou)
        found = []
        for i in np.array(picked).reshape(-1):
            bx, by, bw, bh = boxes[i]
            found.append(
                Detection(
                    float(max(0, bx) + x0),
                    float(max(0, by) + y0),
                    float(min(img.shape[1], bx + bw) + x0),
                    float(min(img.shape[0], by + bh) + y0),
                    float(score[i]),
                    VEHICLE,
                )
            )
        return found


def on_vehicle(
    plate: Detection, vehicles: Sequence[Detection], margin: float = DEFAULT_THRESHOLDS.vehicle_margin
) -> bool:
    """The plate is inside some vehicle (with a margin) and small next to it."""
    cx, cy = (plate.x0 + plate.x1) / 2, (plate.y0 + plate.y1) / 2
    for v in vehicles:
        mx, my = v.w * margin, v.h * margin
        if v.x0 - mx <= cx <= v.x1 + mx and v.y0 - my <= cy <= v.y1 + my and plate.w <= 0.9 * v.w + mx:
            return True
    return False


def clip_to_vehicle(
    region: Detection, vehicles: Sequence[Detection], margin: float = DEFAULT_THRESHOLDS.vehicle_margin
) -> Optional[Detection]:
    """A text region whose centre lies on a vehicle (with its margin), cut to that vehicle plus margin; None when it lies on none.
    The text path hides every region on a vehicle (PT9), so a region too wide to be small next to it (`on_vehicle`) is clipped to the
    vehicle, not dropped: privacy first (B: a plate on a 159 x 78 vehicle was lost with a region 182 wide, blind-015)."""
    cx, cy = (region.x0 + region.x1) / 2, (region.y0 + region.y1) / 2
    for v in vehicles:
        mx, my = v.w * margin, v.h * margin
        if v.x0 - mx <= cx <= v.x1 + mx and v.y0 - my <= cy <= v.y1 + my:
            box = (
                max(region.x0, v.x0 - mx),
                max(region.y0, v.y0 - my),
                min(region.x1, v.x1 + mx),
                min(region.y1, v.y1 + my),
            )
            return Detection(*box, region.score, region.kind, region.source, True)
    return None


class _VehicleStage:
    """What both plate paths share: the vehicle detector, the timings and the search for vehicles (whole picture, plus
    1,024 px tiles in large photos)."""

    kind = PLATE
    blur_shape = "rect"
    video_margins = dict(VIDEO_MARGINS)
    photo_margin = PHOTO_MARGIN
    thresholds: PlateThresholds
    vehicles: VehicleDetector

    def __init__(self):
        self.last_vehicles: List[Detection] = []  # vehicles found by the latest sweep, for crops and the benchmark
        self.timings: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    @contextmanager
    def _timed(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[key] = self.timings.get(key, 0.0) + time.perf_counter() - start
            self.calls[key] = self.calls.get(key, 0) + 1

    def _find_vehicles(self, bgr: np.ndarray, use_tiles: bool) -> List[Detection]:
        height, width = bgr.shape[:2]
        with self._timed("vehicles"):
            vehicles = self.vehicles.detect(bgr)
            if use_tiles and max(width, height) > 1400:
                for tile in geometry.tiles(width, height, 1024, 0.25):
                    vehicles += self.vehicles.detect(bgr, tile)
                vehicles = geometry.nms(
                    vehicles, 0.5, contained=None
                )  # IoU only: a small vehicle inside a big box stays
        return vehicles


class PlateModule(_VehicleStage):
    """The plate class of the engine (contract.ClassModule). Built as PlateModule(threads=0, models=None)."""

    def __init__(
        self,
        threads: int = 0,
        models: Optional[str] = None,
        thresholds: PlateThresholds = DEFAULT_THRESHOLDS,
        vehicles=None,
    ):
        super().__init__()
        folder = models_dir(models)
        self.thresholds = thresholds
        plate_file = os.path.join(folder, PLATE_MODEL)
        if not os.path.isfile(plate_file):
            raise ModelLoadError(
                PLATE,
                f"{PLATE_MODEL} is not shipped (licence not cleared); the v1 reference path needs it fetched with "
                f"scripts/fetch_reference_plate_model.py. {models_missing(plate_file)}",
            )
        self.plates = PlateDetector(_open_session(PLATE, plate_file, threads), thresholds)
        self.vehicles = vehicles or VehicleDetector(
            _open_session(VEHICLE, os.path.join(folder, VEHICLE_MODEL), threads), thresholds
        )

    @property
    def model_failures(self) -> int:
        """Plate (and, with the RF-DETR vehicle detector, vehicle) model runs that raised and were treated as "nothing here"."""
        return self.plates.run_failures + getattr(self.vehicles, "run_failures", 0)

    def sweep(self, bgr: np.ndarray, plan: SweepPlan) -> List[Detection]:
        """Plates in the whole image (shrunk to plan.general_side) and in full-resolution tiles, plus a second look
        inside the largest vehicles. Each plate knows whether it sits on a detected vehicle."""
        height, width = bgr.shape[:2]
        min_score = self.plates.min_score if plan.min_score is None else plan.min_score
        plates: List[Detection] = []
        if plan.full_image:
            with self._timed("plate_full"):
                plates += self.plates.detect(bgr, None, min_score)
        if plan.plate_tiles:
            for tile in geometry.sweep_tiles(width, height, plan.tile_side, plan.tiles):
                with self._timed("plate_tiles"):
                    plates += self.plates.detect(bgr, tile, min_score)
        vehicles = self._find_vehicles(bgr, plan.vehicle_tiles)
        min_w, min_h = self.thresholds.vehicle_min_size
        for v in sorted(vehicles, key=lambda v: -v.w * v.h)[: plan.max_vehicles]:
            if v.w >= min_w and v.h >= min_h:
                region = geometry.region_around(v.box(), width, height, self.thresholds.vehicle_region, 64)
                with self._timed("plates_in_vehicles"):
                    plates += self.plates.detect(bgr, region, min_score)
        plates = geometry.nms(plates)
        for p in plates:
            p.on_vehicle = on_vehicle(p, vehicles, self.thresholds.vehicle_margin)
        self.last_vehicles = vehicles
        return plates

    def retry(self, bgr: np.ndarray, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        """Look again at the model's own scale around `expected` and return the detection that overlaps it most."""
        t = self.thresholds
        min_score = t.recover_score if purpose is RetryPurpose.RECOVER else t.verify_score
        height, width = bgr.shape[:2]
        region = geometry.region_around(expected, width, height, t.retry_region, t.retry_region_min)
        with self._timed("retry_plate"):
            candidates = self.plates.detect(bgr, region, min_score)
        best, best_iou = None, 0.0
        for d in candidates:
            overlap = geometry.iou(d.box(), expected)
            if overlap > best_iou:
                best, best_iou = d, overlap
        if best is None or best_iou < t.retry_iou:
            return None
        return replace(best, source=RETRY)

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        """BLUR on a vehicle that the VERIFY retry confirms; CHECK on a vehicle without confirmation, or off a
        vehicle with very high scores; else DROP. Off a vehicle a plate is never blurred on its own."""
        t = self.thresholds
        v = verify.score if verify is not None else 0.0
        params = {"score": det.score, "verify": v}
        if det.on_vehicle and v >= t.confirm_verify:
            return Judgement(Verdict.BLUR, "plate.reason.on_vehicle_verified", params)
        if (det.on_vehicle and det.score >= t.doubt_score) or (det.score >= t.strong_score and v >= t.strong_verify):
            key = "plate.reason.unconfirmed" if det.on_vehicle else "plate.reason.off_vehicle"
            return Judgement(Verdict.CHECK, key, params)
        return Judgement(Verdict.DROP, "plate.reason.rejected", params)

    def video_judgement(self, track: TrackStats) -> Judgement:
        t = self.thresholds
        params = {"views": float(track.n_views), "score": track.max_score, "verify": track.best_verify}
        if track.on_vehicle:
            repeated = (
                track.n_views >= t.views_to_blur
                and track.n_sweep_views >= t.sweep_views_to_blur
                and track.max_score >= t.views_score
            )
            if track.best_verify >= t.confirm_verify or repeated:
                return Judgement(Verdict.BLUR, "plate.reason.on_vehicle_confirmed", params)
            if track.max_score >= t.doubt_score:
                return Judgement(Verdict.CHECK, "plate.reason.unconfirmed", params)
        else:
            long_and_verified = track.n_views >= t.off_vehicle_views and track.best_verify >= t.off_vehicle_verify
            strong = track.max_score >= t.strong_score and track.best_verify >= t.strong_verify
            if long_and_verified or strong:
                return Judgement(Verdict.CHECK, "plate.reason.off_vehicle", params)
        return Judgement(Verdict.DROP, "plate.reason.rejected", params)

    def wants_verify(self, track: TrackStats, det: Detection) -> bool:
        """Verify at birth and when the view grows a lot (judging is easier up close). A plate off every vehicle is
        not worth the time until it has been seen for long enough to reach Review."""
        t = self.thresholds
        if not track.on_vehicle and track.n_views < t.off_vehicle_views:
            return False
        if track.n_verify == 0:
            return True
        return track.n_verify < t.max_verifications and max(det.w, det.h) >= t.reverify_growth * track.verify_side

    def wants_recover(self, track: TrackStats) -> bool:
        """Only plates on a vehicle are searched for when missed: the rest are mostly signs and windows."""
        return track.on_vehicle

    def close(self) -> None:
        self.plates.session = None
        self.vehicles.session = None


@dataclass(frozen=True)
class TextPathThresholds:
    """The numbers of the text path. The pilot sets them on dev data (4.3's method); these are its starting values."""

    vehicle_side: int = 960  # T_V: a vehicle crop is enlarged (or reduced) so its longest side measures this
    vehicle_context: float = 1.15  # the crop is the vehicle's box this many times larger, clipped to the picture
    tau_lo: float = 0.30  # a text region scoring from here is listed (blurred by default: privacy first)...
    tau_hi: float = 0.60  # ...and from here it is blurred without a doubt
    views_to_blur: int = 3  # video: a region seen this many times is BLUR whatever its score (PT7)
    # Photos: the detector also gives the regions scoring from `merge_floor`, and they are kept only as part of a region they are united with (a piece of a
    # plate that the detector scored low, a strip of its frame): the united box is the writing and its frame instead of the writing alone, and a plate is
    # covered more often (round 2 of 2.2: 3 of the 58 readable dev plates). A region that is still under `tau_lo` after the union is dropped, as before.
    merge_floor: float = 0.12
    plate_min_aspect: float = 1.8  # `plate_like` label only: width / height of a plate...
    plate_max_aspect: float = 6.5
    plate_min_height: float = (
        6.0  # ...and its height in pixels of the picture. The label never decides what is blurred.
    )
    # The plate-position band (PT4 lever, opt-in, photos): on a vehicle narrower than `band_max_width` px where no text box lies over the
    # band, the band itself is one more region to blur. Fitted on the v1 sets only (bench/fit_plate_band.py): the 2nd and 98th percentiles
    # of the plate edges on vehicles narrower than 250 px, as fractions of the vehicle's box (left, right, top, bottom); the right edge is
    # widened from 0.95 to 0.97 because the exact band (without the photo margin that B's offline estimate gave it) left one EU 30 %
    # plate at 107 of 108. The width limit (400) and the suppression (off) were then set by B's replay on the held-out v2 dev record
    # (measure/simulate_band_variants.py, plates hidden at 100/50/30 % of the size: 86/77/68 with 250 and suppression 0.90, 100/87/69
    # with 400 and none; 3.4/3.6/5.3 % of the picture), because a text box elsewhere in the band hid 8 small vehicles' plates.
    band: bool = False
    band_box: Tuple[float, float, float, float] = (0.13, 0.97, 0.43, 1.04)
    band_max_width: float = 400.0
    band_text_inside: float = 1.01  # a text box with at least this share of its area inside the band means text was found there; above 1: never
    # Where a plate can be (Options.unsure "placed", photos; measured on the plate photos of the dev part, bench/bench_subtle.py): a text region
    # is blurred by default when it scored `tau_hi`, is at most `placed_max_height` of the height of the vehicle it lies on, has its centre
    # below the top `placed_min_y` of that vehicle and is at least `placed_min_width` of its width (a badge is smaller; against the widest vehicle it lies on); a region of
    # `placed_shape_px` or more must also be `placed_min_aspect` times wider than tall (a logo, a wheel hub or a door handle is as tall as
    # it is wide; a smaller box is too coarse to tell). What fails any of these is a "Not sure" item left visible and listed in Review.
    placed_max_height: float = 0.30
    placed_min_y: float = 0.40
    placed_min_width: float = 0.09
    placed_shape_px: float = 90.0
    placed_min_aspect: float = 1.3
    # The second opinion of the plate reader (`classes/platetext.py`, round 2 of 2.2; fitted on the dev part, checked on the street photos and the
    # OpenALPR EU set): a region that passed the position rule is taken out of the default blur when it is at least `read_veto_min_width` px wide and the
    # reader reads a word in it (letters only, at least `read_word_letters`, confidence `read_word_conf`: "TAXI", "RENAULT", lettering on a van); a region that
    # failed the position rule or scored under `tau_hi` is brought in when the reader reads a plate in it (`read_plate_digits` digits in a run of 4 to 9
    # characters, confidence `read_plate_conf`, on the crop without its margins). A region of `read_blank_min_width` px or more in which the reader finds
    # nothing at all (confidence under `read_blank_conf`) is not a plate worth a blur by default either: a plate that wide is legible to it (no readable dev, street or
    # EU plate was lost to this, and it removes the bare patches that the text detector reads as writing). A smaller region is judged by position alone.
    read_blank_min_width: float = 50.0
    read_blank_conf: float = 0.3
    # A region `read_wide_min_width` px wide or more in which the reader does not find `read_wide_digits` digits at `read_blank_conf` is no plate either (a plate that
    # large is legible to it), and a small, roundish region (under `placed_shape_px` wide and less than `placed_min_aspect` times wider than tall, which the shape
    # rule leaves alone as too coarse) in which it reads, with `read_square_conf`, something with no digit in it is an emblem, not a plate: the lion of a Peugeot reads "E".
    read_wide_min_width: float = 120.0
    read_wide_digits: int = 2
    read_square_conf: float = 0.7
    read_veto_min_width: float = 60.0
    read_word_conf: float = 0.5
    read_word_letters: int = 3
    read_plate_conf: float = 0.5
    read_plate_digits: int = 2


DEFAULT_TEXT_PATH = TextPathThresholds()


def unite_overlapping(regions: Sequence[Detection], iou_min: float = 0.4, inside_min: float = 0.8) -> List[Detection]:
    """Text regions that are the same place become one region: the union of their boxes with the higher score. `nms` would drop the
    lower-scored one whole, and the part of it that sticks out would be left uncovered; here no region ever loses coverage (privacy
    first, PT3). Two regions are one place when their IoU is at least `iou_min` or one lies `inside_min` inside the other."""
    merged = [Detection(r.x0, r.y0, r.x1, r.y1, r.score, r.kind, r.source, r.on_vehicle) for r in regions]
    merged.sort(key=lambda r: -r.score)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(merged):
            for b in merged[i + 1 :]:
                if _same_place(a, b, iou_min, inside_min):
                    a.x0, a.y0, a.x1, a.y1 = min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)
                    merged.remove(b)
                    changed = True
                    break
            if changed:
                break
    return merged


def _same_place(a: Detection, b: Detection, iou_min: float, inside_min: float) -> bool:
    if geometry.iou(a.box(), b.box()) >= iou_min:
        return True
    iw, ih = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
    if iw <= 0 or ih <= 0:
        return False
    return iw * ih / max(1e-9, min(a.w * a.h, b.w * b.h)) >= inside_min


def _inside_share(box: Box, region: Box) -> float:
    """The share of `box`'s area that lies inside `region`."""
    iw, ih = min(box[2], region[2]) - max(box[0], region[0]), min(box[3], region[3]) - max(box[1], region[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return 0.0 if iw <= 0 or ih <= 0 or area <= 0 else iw * ih / area


def plate_like(det: Detection, t: TextPathThresholds = DEFAULT_TEXT_PATH) -> bool:
    """Whether a text region is shaped like a plate: the label that separates plates from other writing in the report (PT9)."""
    return det.h >= t.plate_min_height and t.plate_min_aspect <= det.w / max(det.h, 1e-9) <= t.plate_max_aspect


class TextPlateModule(_VehicleStage):
    """The plate class by text on vehicles, behind the same contract as PlateModule: detect the vehicles, run the text
    detector on each enlarged vehicle crop, and give every text region on a vehicle as a plate item (a plate, a phone number, a
    company name: what is written on a car is hidden). A region is BLUR from `tau_hi`, CHECK from `tau_lo`; both are blurred by
    default. Nothing off a vehicle is ever found, by construction. Built as TextPlateModule(threads=0, models=None); `vehicles` and
    `text` replace the detectors (tests)."""

    def __init__(
        self,
        threads: int = 0,
        models: Optional[str] = None,
        thresholds: PlateThresholds = DEFAULT_THRESHOLDS,
        text_path: TextPathThresholds = DEFAULT_TEXT_PATH,
        text_thresholds: TextThresholds = DEFAULT_TEXT_THRESHOLDS,
        vehicles=None,
        text=None,
        fit: str = "tight",
        reader=None,
    ):
        super().__init__()
        self.reader = reader  # the plate reader (classes/platetext.PlateReader); None: the position rule decides alone
        self.reads = 0  # regions given to the reader, over every sweep
        self.thresholds = thresholds
        self.photo_margin = PHOTO_MARGIN if fit == "loose" else PHOTO_MARGIN_TIGHT
        self.text_path = text_path
        self.vehicles = vehicles or VehicleDetector(
            _open_session(VEHICLE, os.path.join(models_dir(models), VEHICLE_MODEL), threads), thresholds
        )
        self.text = text or TextDetector(threads, models, text_thresholds)
        self.vehicles_visited = 0  # vehicles given to the text detector, over every sweep
        self.vehicles_not_visited = 0  # vehicles beyond `max_vehicles`: counted and reported, never silently skipped
        self.vehicles_too_small = (
            0  # vehicles under `vehicle_min_size`: too small to hold readable text, counted all the same (PT6)
        )
        self.bands_added = 0  # plate-position bands added (`text_path.band`), over every sweep
        self.band_area = 0.0  # pixels blurred by those bands...
        self.banded_vehicle_area = (
            0.0  # ...and the pixels of the boxes of the vehicles that got one: the area share is their ratio
        )

    @property
    def model_failures(self) -> int:
        """Text (and, with the RF-DETR vehicle detector, vehicle) model runs that raised and were treated as "nothing here"."""
        return getattr(self.text, "run_failures", 0) + getattr(self.vehicles, "run_failures", 0)

    def sweep(self, bgr: np.ndarray, plan: SweepPlan) -> List[Detection]:
        """Every text region on the `plan.max_vehicles` largest vehicles, as plate detections that sit on a vehicle."""
        vehicles = self._find_vehicles(bgr, plan.vehicle_tiles)
        self.last_vehicles = vehicles
        if plan.max_vehicles <= 0:
            return []  # video: only the vehicles are found here; the text detector is run at key frames (look_at_vehicle, PT7)
        searchable = sorted((v for v in vehicles if self._big_enough(v)), key=lambda v: -v.w * v.h)
        self.vehicles_too_small += len(vehicles) - len(searchable)
        visited, skipped = searchable[: plan.max_vehicles], searchable[plan.max_vehicles :]
        self.vehicles_visited += len(visited)
        self.vehicles_not_visited += len(skipped)
        regions: List[Detection] = []
        for v in visited:
            regions += self._text_on(bgr, v, vehicles, plan.min_score, merge_floor=self.text_path.merge_floor)
        found = unite_overlapping(regions)
        if self.text_path.band:
            found += self._bands(bgr.shape[1], bgr.shape[0], searchable, found)
        return found

    def _bands(
        self, width: int, height: int, vehicles: Sequence[Detection], text: Sequence[Detection]
    ) -> List[Detection]:
        """One band zone on every vehicle narrower than `band_max_width` that has no text box over its band (a box with `band_text_inside`
        of its area inside it), visited or not. The zone is shrunk by the photo margin so that, once the engine adds its margin, the
        blurred area is the band (clipped to the picture). Counted: `bands_added`, `band_area`, `banded_vehicle_area`."""
        p = self.text_path
        mx, my = self.photo_margin
        out = []
        for v in vehicles:
            if v.w >= p.band_max_width:
                continue
            u0, u1, t0, t1 = p.band_box
            band = (
                max(0.0, v.x0 + u0 * v.w),
                max(0.0, v.y0 + t0 * v.h),
                min(float(width), v.x0 + u1 * v.w),
                min(float(height), v.y0 + t1 * v.h),
            )
            if band[2] - band[0] < 2 or band[3] - band[1] < 2:
                continue
            if any(_inside_share(t.box(), band) >= p.band_text_inside for t in text):
                continue
            cx, cy, half_w, half_h = (
                (band[0] + band[2]) / 2,
                (band[1] + band[3]) / 2,
                (band[2] - band[0]) / 2,
                (band[3] - band[1]) / 2,
            )
            half_w, half_h = half_w / (1 + 2 * mx), half_h / (1 + 2 * my)
            out.append(Detection(cx - half_w, cy - half_h, cx + half_w, cy + half_h, 1.0, PLATE, BAND, True))
            self.bands_added += 1
            self.band_area += (band[2] - band[0]) * (band[3] - band[1])
            self.banded_vehicle_area += v.w * v.h
        return out

    def look_at_vehicle(
        self, bgr: np.ndarray, vehicle: Detection, vehicles: Sequence[Detection] = (), min_score: Optional[float] = None
    ) -> List[Detection]:
        """The text regions of one vehicle, on demand: what the video engine runs at the key frames of a tracked vehicle.
        `vehicles` are the ones the regions are checked to lie on (default: this one). A vehicle too small to read is not looked at."""
        if not self._big_enough(vehicle):
            self.vehicles_too_small += 1
            return []
        self.vehicles_visited += 1
        return self._text_on(bgr, vehicle, list(vehicles) or [vehicle], min_score)

    def _big_enough(self, vehicle: Detection) -> bool:
        min_w, min_h = self.thresholds.vehicle_min_size
        return vehicle.w >= min_w and vehicle.h >= min_h

    def _text_on(
        self,
        bgr: np.ndarray,
        vehicle: Detection,
        vehicles: Sequence[Detection],
        min_score: Optional[float],
        merge_floor: Optional[float] = None,
    ) -> List[Detection]:
        """The text regions of one vehicle: its crop enlarged to `vehicle_side`, the detector run on it, boxes back in the picture. With a
        `merge_floor` the detector gives the regions from that score and the ones still under `min_score` after the union are dropped."""
        p = self.text_path
        height, width = bgr.shape[:2]
        cx, cy = (vehicle.x0 + vehicle.x1) / 2, (vehicle.y0 + vehicle.y1) / 2
        half_w, half_h = vehicle.w * p.vehicle_context / 2, vehicle.h * p.vehicle_context / 2
        x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        x1, y1 = int(min(width, cx + half_w)), int(min(height, cy + half_h))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        crop = bgr[y0:y1, x0:x1]
        scale = p.vehicle_side / max(crop.shape[:2])
        enlarged = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )
        wanted = p.tau_lo if min_score is None else min_score
        with self._timed("text_on_vehicles"):
            found = self.text.detect(enlarged, min(wanted, merge_floor) if merge_floor else wanted)
        out = []
        for r in found:
            box = (
                max(0.0, r.x0 / scale + x0),
                max(0.0, r.y0 / scale + y0),
                min(float(width), r.x1 / scale + x0),
                min(float(height), r.y1 / scale + y0),
            )
            region = Detection(*box, r.score, PLATE, r.source, True)
            if on_vehicle(region, vehicles, self.thresholds.vehicle_margin):
                kept = region
            else:  # too wide to be small next to its vehicle: cut to it instead of losing it
                kept = clip_to_vehicle(region, vehicles, self.thresholds.vehicle_margin)
            if kept is not None and kept.w >= 2 and kept.h >= 2:
                out.append(kept)
        united = unite_overlapping(out)
        return [r for r in united if r.score >= wanted] if merge_floor else united

    def retry(self, bgr: np.ndarray, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        """The text path has no second opinion: the score of the region is what decides."""
        return None

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        """BLUR from `tau_hi`, CHECK from `tau_lo`; the region's shape only labels it (`plate_like` in the params). A position band
        (source BAND) is BLUR: a prior, not a doubt, and it never counts as a "Not sure" item."""
        p = self.text_path
        if det.source == BAND:
            return Judgement(Verdict.BLUR, "plate.reason.position_band", {})
        params = {"score": det.score, "plate_like": 1.0 if plate_like(det, p) else 0.0}
        if det.score >= p.tau_hi:
            return Judgement(Verdict.BLUR, "plate.reason.vehicle_text", params)
        if det.score >= p.tau_lo:
            return Judgement(Verdict.CHECK, "plate.reason.vehicle_text", params)
        return Judgement(Verdict.DROP, "plate.reason.rejected", params)

    def photo_placement(self, box: Box, judgement: Judgement, ctx: PhotoContext) -> Judgement:
        """Whether a text region is a plate worth blurring by default, judged against the vehicles the latest sweep found and, with a reader, against
        what is written in it. A BLUR region that does not sit where a plate can be becomes a CHECK, a "Not sure" item that is listed and left visible; one
        that sits there but reads as a word (`read_veto_min_width`) becomes one too; a region that failed the position rule, or scored under `tau_hi`, is
        blurred when the reader reads a plate in it. The position band is a prior, not a reading: left as it is."""
        if judgement.reason == "plate.reason.position_band":
            return judgement
        params = dict(judgement.params)
        sits = judgement.verdict is Verdict.BLUR and self._sits_like_a_plate(box)
        ok = sits
        if self.reader is not None and ctx.image is not None and judgement.verdict is not Verdict.DROP:
            ok = self._second_opinion(box, ctx.image, sits, params)
        params["placed"] = 1.0 if ok else 0.0
        verdict = Verdict.BLUR if ok else (Verdict.CHECK if judgement.verdict is not Verdict.DROP else Verdict.DROP)
        return Judgement(verdict, judgement.reason, params)

    def _second_opinion(self, box: Box, image: np.ndarray, sits: bool, params: Dict[str, float]) -> bool:
        """The reader's say on a region, which `sits` where a plate can be or not. Records what it read in `params` (read_conf, read_digits)."""
        from .platetext import best_of, plate_like, word_like

        p = self.text_path
        x0, y0, x1, y1 = (int(v) for v in box)
        if not self._hosts(box) or x1 - x0 < 8 or y1 - y0 < 4:
            return sits
        width, height = x1 - x0, y1 - y0
        roundish = width < p.placed_shape_px and width < p.placed_min_aspect * height
        if sits and width < min(p.read_veto_min_width, p.read_blank_min_width) and not roundish:
            return True  # too small for the reader to be a judge: the position rule stands
        crop = image[max(0, y0) : y1, max(0, x0) : x1]
        if crop.size == 0:
            return sits
        self.reads += 1
        raw, shrunk = self.reader.read(crop)
        seen = best_of((raw, shrunk))
        params["read_conf"], params["read_digits"] = seen.conf, float(seen.digits)
        if not sits:
            return plate_like(shrunk, p.read_plate_conf, p.read_plate_digits)
        if width >= p.read_veto_min_width and word_like(seen, p.read_word_conf, p.read_word_letters):
            return False
        if width >= p.read_blank_min_width and seen.conf < p.read_blank_conf:
            return False
        if width >= p.read_wide_min_width and not (
            seen.conf >= p.read_blank_conf and seen.digits >= p.read_wide_digits
        ):
            return False
        if roundish and seen.digits == 0 and seen.conf >= p.read_square_conf:
            return False
        return True

    def _hosts(self, box: Box) -> List[Detection]:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return [v for v in self.last_vehicles if v.x0 <= cx <= v.x1 and v.y0 <= cy <= v.y1]

    def _sits_like_a_plate(self, box: Box) -> bool:
        p = self.text_path
        w, h = box[2] - box[0], box[3] - box[1]
        cy = (box[1] + box[3]) / 2
        hosts = self._hosts(box)
        if not hosts or w <= 0 or h <= 0:
            return False
        if min(h / v.h for v in hosts) > p.placed_max_height:
            return False
        if max((cy - v.y0) / v.h for v in hosts) < p.placed_min_y:
            return False
        if min(w / v.w for v in hosts) < p.placed_min_width:
            return False
        return w < p.placed_shape_px or w / h >= p.placed_min_aspect

    def video_judgement(self, track: TrackStats) -> Judgement:
        p = self.text_path
        params = {"views": float(track.n_views), "score": track.max_score}
        if track.max_score >= p.tau_hi or (track.n_views >= p.views_to_blur and track.max_score >= p.tau_lo):
            return Judgement(Verdict.BLUR, "plate.reason.vehicle_text", params)
        if track.max_score >= p.tau_lo:
            return Judgement(Verdict.CHECK, "plate.reason.vehicle_text", params)
        return Judgement(Verdict.DROP, "plate.reason.rejected", params)

    def wants_verify(self, track: TrackStats, det: Detection) -> bool:
        return False

    def wants_recover(self, track: TrackStats) -> bool:
        return False

    def close(self) -> None:
        self.vehicles.session = None
        self.text.close()
        if self.reader is not None:
            self.reader.close()
