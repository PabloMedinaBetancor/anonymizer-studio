"""Heavy face detectors for the second look and deep mode, behind `contract.HeavyDetector`.

Two wrappers live here:

* `RetinaFaceDetector`: RetinaFace ResNet-34 (MIT code and weights, ONNX Runtime, dynamic input). The provisional heavy
  model of the pilot, ported from the measurement adapter as our own code.
* `ZoomedYuNet`: the control C1, the YuNet of the first pass run again on the enlarged zone. No new file. A heavy model that does not beat
  it is not shipped.

Every model file is checked against `models/SHA256SUMS` when a detector is built (CL3): a missing or changed file is a
`ModelLoadError` before any photo is touched, never a silent downgrade (SL7). The ONNX session itself opens at the first call
(SL8: nothing heavy is resident until a zone needs it) and closes with `close()`.

A run that fails raises `HeavyRunError`. The second look counts it in `model_failures`, so the photo is "not checked": a failed look is
never "nothing found" (SL6).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import contract, geometry
from ..contract import (
    DEEP,
    FACE,
    SECOND,
    Box,
    Detection,
    Judgement,
    ModelLoadError,
    SuspectZone,
    SweepPlan,
    TrackStats,
    Verdict,
)
from .faces import FaceModule

RETINAFACE_FILE = "retinaface_r34.onnx"
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
HEAVY_FACE_MODELS = ("retinaface-r34", "yunet-zoomed")
DEFAULT_HEAVY_FACE = "retinaface-r34"
ZONE_CONTEXT = 0.25  # context cut around the zone box, per side
SCALE_RANGE = (0.5, 6.0)  # 4.2: the enlargement is kept in this range
MIN_REGION = 16  # a cut smaller than this is widened (the wrappers refuse anything under 8 px)
DEEP_GENERAL_SIDE = 1024  # deep: the whole picture at this longest side, then full-resolution tiles
DEEP_TILE = 640  # the ceiling of a deep tile (DM1b: memory and the 1.5 s limit)
MIN_DEEP_TILE = 320  # and its floor: tiles are at least 320 px, and a smaller one would defeat the limit of tiles (DM2)
VERDICT_RANK = {Verdict.DROP: 0, Verdict.CHECK: 1, Verdict.BLUR: 2}
REASON_SECOND_FOUND = "face.reason.second_found"
REASON_SECOND_CONFIRMED = "face.reason.second_confirmed"

# The longest side of what one call sees. A zone enlarged beyond it is split into overlapping pieces, not waited for (SL6, DM1b):
# R34 takes about 0.4 s on a 640 px piece of the reference machine, so no call comes near the 1.5 s limit.
MAX_INPUT_SIDE = 640
PIECE_OVERLAP = 0.25
NMS_IOU = 0.4
RETINAFACE_MEAN = (104.0, 117.0, 123.0)  # BGR, subtracted from the input and also the padding colour
PRIOR_CACHE = 8  # prior grids kept (one per input size seen)
YUNET_MULTIPLE = 64  # YuNet keeps memory per distinct input size; padding to a multiple stops the growth


class HeavyRunError(RuntimeError):
    """A heavy model run raised. Never taken as "nothing found"."""


_VERIFIED: Dict[Tuple[str, int, int], str] = {}
_LOCK = threading.Lock()


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_hashes(folder: str) -> Dict[str, str]:
    """`models/SHA256SUMS` as {file name: hash}; empty when the file is missing."""
    path = os.path.join(folder, "SHA256SUMS")
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                out[parts[1].lstrip("*")] = parts[0].lower()
    return out


def verified_model(kind: str, folder: str, name: str) -> Tuple[str, str]:
    """The path and SHA-256 of model file `name` in `folder`, after checking it against `SHA256SUMS` (CL3). Raises ModelLoadError
    when the file is missing, has no entry or differs. The hash of a file that has not changed is not computed twice."""
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise ModelLoadError(kind, contract.models_missing(path))
    expected = expected_hashes(folder).get(name)
    if expected is None:
        raise ModelLoadError(kind, f"{name} has no entry in {os.path.join(folder, 'SHA256SUMS')}")
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns)
    with _LOCK:
        actual = _VERIFIED.get(key)
    if actual is None:
        actual = sha256_of(path)
        with _LOCK:
            _VERIFIED[key] = actual
    if actual != expected:
        raise ModelLoadError(
            kind, f"{name} does not match SHA256SUMS (expected {expected[:12]}..., found {actual[:12]}...)"
        )
    return path, actual


def read_verified(kind: str, path: str, expected: str) -> bytes:
    """The bytes of a model file, after checking THOSE bytes against `expected` (CL3). A session built from them is the file that was checked, even
    when the file on disk was replaced after the wrapper was built."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise ModelLoadError(kind, contract.models_missing(path)) from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ModelLoadError(
            kind,
            f"{os.path.basename(path)} does not match SHA256SUMS (expected {expected[:12]}..., found {actual[:12]}...)",
        )
    return data


def open_session(kind: str, model, threads: int = 0):
    """An ONNX Runtime CPU session with the options of the plate module: no memory arena, no memory pattern. `model` is a path or the bytes
    of the file (`read_verified`)."""
    try:
        import onnxruntime as ort
    except Exception as exc:  # on Windows usually a missing DLL, which is an OSError and not an ImportError
        raise ModelLoadError(kind, f"cannot load onnxruntime: {type(exc).__name__}: {exc}") from exc
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    try:
        return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    except Exception as exc:
        name = os.path.basename(model) if isinstance(model, str) else kind
        raise ModelLoadError(kind, f"{name}: {exc}") from exc


def _pieces(width: int, height: int, side: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Overlapping squares of at most `side` that cover width x height (one piece when it fits)."""

    def cuts(n: int) -> List[int]:
        if n <= side:
            return [0]
        step = side * (1.0 - overlap)
        k = int(np.ceil((n - side) / step)) + 1
        return [int(round(i * (n - side) / (k - 1))) for i in range(k)]

    return [(x, y, min(width, x + side), min(height, y + side)) for y in cuts(height) for x in cuts(width)]


def deep_tile_side(requested: int = 0) -> int:
    """The side of a deep tile: 0 asks for the default (`DEEP_TILE`), anything else is kept between `MIN_DEEP_TILE` and `DEEP_TILE`; a negative
    side is an error."""
    if requested < 0:
        raise ValueError(f"a tile side cannot be negative: {requested}")
    return min(max(requested or DEEP_TILE, MIN_DEEP_TILE), DEEP_TILE)


def pieces_in(width: int, height: int, side: int = MAX_INPUT_SIDE) -> int:
    """How many heavy calls an image of this size takes when it is cut into pieces of at most `side`."""
    return len(_pieces(width, height, side, PIECE_OVERLAP))


def _nms(dets: List[Detection], iou: float) -> List[Detection]:
    if len(dets) < 2:
        return dets
    boxes = [[d.x0, d.y0, d.w, d.h] for d in dets]
    keep = cv2.dnn.NMSBoxes(boxes, [d.score for d in dets], 0.0, iou)
    return [dets[i] for i in np.array(keep, dtype=int).reshape(-1)]


class _ZoneDetector:
    """What the two wrappers share: the region and scale of `detect`, the pieces, the mapping of boxes back to picture pixels.
    A subclass supplies `_run(img) -> [(x0, y0, x1, y1, score)]` in the pixels of `img`."""

    name: str
    version: str
    sha256: str
    max_side: int = MAX_INPUT_SIDE
    check: Optional[Callable[[], None]] = None  # called before every piece: it may raise to stop a long sweep (E8, DM3)
    slowest_piece: float = (
        0.0  # seconds of the slowest model run since the owner last reset it: SL6 asks that none passes 1.5 s
    )

    def _run(self, img: np.ndarray, min_score: float) -> List[Tuple[float, float, float, float, float]]:
        raise NotImplementedError

    def detect(self, bgr: np.ndarray, region: Box, scale: float, min_score: float) -> List[Detection]:
        height, width = bgr.shape[:2]
        x0, y0 = max(0, int(region[0])), max(0, int(region[1]))
        x1, y1 = min(width, int(np.ceil(region[2]))), min(height, int(np.ceil(region[3])))
        if x1 - x0 < 8 or y1 - y0 < 8 or scale <= 0:
            return []
        crop = bgr[y0:y1, x0:x1]
        nw, nh = max(8, int(round((x1 - x0) * scale))), max(8, int(round((y1 - y0) * scale)))
        if (nw, nh) != (x1 - x0, y1 - y0):
            crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        ex, ey = (x1 - x0) / nw, (y1 - y0) / nh
        found: List[Detection] = []
        for px0, py0, px1, py1 in _pieces(nw, nh, self.max_side, PIECE_OVERLAP):
            if self.check is not None:
                self.check()
            started = time.perf_counter()
            rows = self._run(crop[py0:py1, px0:px1], min_score)
            self.slowest_piece = max(self.slowest_piece, time.perf_counter() - started)
            for bx0, by0, bx1, by1, score in rows:
                if score < min_score:
                    continue
                found.append(
                    Detection(
                        min(float(x1), x0 + (px0 + max(0.0, bx0)) * ex),
                        min(float(y1), y0 + (py0 + max(0.0, by0)) * ey),
                        min(float(x1), x0 + (px0 + bx1) * ex),
                        min(float(y1), y0 + (py0 + by1) * ey),
                        float(score),
                        FACE,
                        SECOND,
                    )
                )
        return [d for d in _nms(found, NMS_IOU) if d.w >= 1 and d.h >= 1]


class RetinaFaceDetector(_ZoneDetector):
    """RetinaFace ResNet-34 (yakhyo/retinaface-pytorch, MIT, trained by its author on WIDER FACE). Input BGR minus the dataset mean at any size
    that is a multiple of 32; 16,800 priors per 640 px decoded here; the scores come out of the network already soft-maxed."""

    name = "retinaface-r34"
    version = "yakhyo-v0.0.1"

    def __init__(self, threads: int = 0, models: Optional[str] = None):
        self._path, self.sha256 = verified_model(FACE, contract.models_dir(models), RETINAFACE_FILE)
        self._threads = threads
        self._session = None
        self._closed = False
        self._priors: "OrderedDict[Tuple[int, int], np.ndarray]" = OrderedDict()
        self._lock = threading.Lock()

    def _open(self):
        if self._closed:
            raise HeavyRunError(f"{self.name} is closed")
        if self._session is None:
            self._session = open_session(FACE, read_verified(FACE, self._path, self.sha256), self._threads)
        return self._session

    def _grid(self, h: int, w: int) -> np.ndarray:
        cached = self._priors.get((h, w))
        if cached is not None:
            self._priors.move_to_end((h, w))
            return cached
        rows = []
        for step, sizes in ((8, (16, 32)), (16, (64, 128)), (32, (256, 512))):
            for i in range(-(-h // step)):
                for j in range(-(-w // step)):
                    for size in sizes:
                        rows.append([(j + 0.5) * step / w, (i + 0.5) * step / h, size / w, size / h])
        grid = np.array(rows, np.float32)
        self._priors[(h, w)] = grid
        while len(self._priors) > PRIOR_CACHE:
            self._priors.popitem(last=False)
        return grid

    def _run(self, img: np.ndarray, min_score: float) -> List[Tuple[float, float, float, float, float]]:
        h0, w0 = img.shape[:2]
        h, w = -(-h0 // 32) * 32, -(-w0 // 32) * 32
        canvas = np.empty((h, w, 3), np.float32)
        canvas[:] = RETINAFACE_MEAN
        canvas[:h0, :w0] = img
        canvas -= RETINAFACE_MEAN
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
        with self._lock:
            try:
                session = self._open()
                loc, conf, _ = session.run(None, {"input": blob})
            except (HeavyRunError, ModelLoadError):
                raise
            except Exception as exc:
                raise HeavyRunError(f"{self.name} failed on a {w0}x{h0} piece: {exc}") from exc
            priors = self._grid(h, w)
        if not (np.isfinite(loc).all() and np.isfinite(conf).all()):
            raise HeavyRunError(f"{self.name} returned values that are not numbers on a {w0}x{h0} piece")
        loc, score = loc[0], conf[0][:, 1]
        if len(priors) != len(loc):
            raise HeavyRunError(f"{self.name} returned {len(loc)} boxes for {len(priors)} priors")
        keep = score >= min_score
        if not keep.any():
            return []
        pr, lo = priors[keep], loc[keep]
        centre = pr[:, :2] + lo[:, :2] * 0.1 * pr[:, 2:]
        size = pr[:, 2:] * np.exp(np.clip(lo[:, 2:] * 0.2, -10, 10))
        boxes = np.concatenate([centre - size / 2, centre + size / 2], 1) * np.array([w, h, w, h], np.float32)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w0)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h0)
        return [tuple(float(v) for v in b) + (float(s),) for b, s in zip(boxes, score[keep], strict=True)]

    def close(self) -> None:
        with self._lock:
            self._session = None
            self._closed = True
            self._priors.clear()


class ZoomedYuNet(_ZoneDetector):
    """The control C1: the YuNet of the first pass on the enlarged zone. It adds no model file; it exists so that the heavy model has
    a bar to beat."""

    name = "yunet-zoomed"
    version = "2023mar"

    def __init__(self, threads: int = 0, models: Optional[str] = None):
        self._path, self.sha256 = verified_model(FACE, contract.models_dir(models), YUNET_FILE)
        self._net = None
        self._size: Optional[Tuple[int, int]] = None
        self._closed = False
        self._lock = threading.Lock()

    def _open(self):
        if self._closed:
            raise HeavyRunError(f"{self.name} is closed")
        if self._net is None:
            try:
                data = read_verified(FACE, self._path, self.sha256)
                self._net = cv2.FaceDetectorYN.create(
                    "onnx", np.frombuffer(data, np.uint8), np.empty(0, np.uint8), (320, 320), 0.3, 0.3, 5000
                )
            except cv2.error as exc:
                raise ModelLoadError(FACE, f"cannot load {self._path}: {exc}") from exc
        return self._net

    def _run(self, img: np.ndarray, min_score: float) -> List[Tuple[float, float, float, float, float]]:
        h0, w0 = img.shape[:2]
        h, w = -(-h0 // YUNET_MULTIPLE) * YUNET_MULTIPLE, -(-w0 // YUNET_MULTIPLE) * YUNET_MULTIPLE
        if (h, w) != (h0, w0):
            img = cv2.copyMakeBorder(img, 0, h - h0, 0, w - w0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        with self._lock:
            try:
                net = self._open()
                net.setScoreThreshold(float(min(min_score, 0.99)))
                if self._size != (w, h):
                    net.setInputSize((w, h))
                    self._size = (w, h)
                _, rows = net.detect(img)
            except HeavyRunError:
                raise
            except cv2.error as exc:
                raise HeavyRunError(f"{self.name} failed on a {w0}x{h0} piece: {exc}") from exc
        if rows is None:
            return []
        if not np.isfinite(rows).all():
            raise HeavyRunError(f"{self.name} returned values that are not numbers on a {w0}x{h0} piece")
        return [(float(r[0]), float(r[1]), float(r[0] + r[2]), float(r[1] + r[3]), float(r[14])) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._net = None
            self._closed = True


def load_face_heavy(name: str, threads: int = 0, models: Optional[str] = None):
    """The heavy face detector called `name` (one of HEAVY_FACE_MODELS). Raises ModelLoadError, ValueError for an unknown name."""
    if name == "retinaface-r34":
        return RetinaFaceDetector(threads, models)
    if name == "yunet-zoomed":
        return ZoomedYuNet(threads, models)
    raise ValueError(f"unknown heavy face model: {name}")


def _at_least(low: float, high: float, size: float, limit: float) -> Tuple[float, float]:
    """Widen [low, high] to `size` around its middle, moved back inside [0, limit] if it sticks out."""
    if high - low >= size:
        return low, high
    mid = (low + high) / 2
    low, high = mid - size / 2, mid + size / 2
    shift = max(0.0, -low) - max(0.0, high - limit)
    return max(0.0, low + shift), min(limit, high + shift)


def enlargement(box: Sequence[float], width: int, height: int, side: float) -> Tuple[Box, float]:
    """How a zone is cut and enlarged: the box with 25 % context per side, never beyond the picture and at least
    `MIN_REGION` px each way (a zone of a few pixels is still looked at), and the scale that makes its longest side `side`, kept
    between 0.5 and 6. Returns the region (picture pixels) and the scale."""
    w, h = box[2] - box[0], box[3] - box[1]
    region = (
        max(0.0, box[0] - ZONE_CONTEXT * w),
        max(0.0, box[1] - ZONE_CONTEXT * h),
        min(float(width), box[2] + ZONE_CONTEXT * w),
        min(float(height), box[3] + ZONE_CONTEXT * h),
    )
    x0, x1 = _at_least(region[0], region[2], MIN_REGION, float(width))
    y0, y1 = _at_least(region[1], region[3], MIN_REGION, float(height))
    longest = max(x1 - x0, y1 - y0, 1.0)
    return (x0, y0, x1, y1), float(np.clip(side / longest, *SCALE_RANGE))


@dataclass(frozen=True)
class HeavyThresholds:
    """The numbers of the second look for the face class, chosen on the dev part of the truth only: 1,144 faces in 155 photos
    (36 + 55 public photos of the v1 sets and 64 of WIDER FACE validation), RetinaFace R34 at three enlargements, the light models and the
    control (YuNet zoomed) on the same zones, recorded on the second laptop and again on the development laptop with the same result. Figures and rules: `bench/results/cells-dev-*.json`."""

    h_hi: float = (
        0.98  # lowest heavy score whose detections match a truth face with precision >= 98 % (n = 196 detections): BLUR
    )
    h_lo: float = 0.52  # lowest score whose extra doubtful items add <= 2 false hidden photos per 100 photos without people: CHECK
    target_side: int = 192  # T_H: largest paired gain over the light models (29.7 per 1,000 faces) of 192, 256 and 320 px; also the fastest
    look_min: float = 0.50  # the lowest heavy score worth carrying out of the wrapper


class HeavyFaceModule(FaceModule):
    """The face class with the second look and deep mode (`contract.SecondLookModule`): YuNet for the first pass as before, a heavy
    detector for the zones of the second look and, in deep mode, for the whole picture. Built only for `careful` and `deep`; `fast` uses
    `FaceModule` and never loads a heavy model, and a missing heavy model file is a `ModelLoadError` here, never a silent downgrade (SL7)."""

    def __init__(
        self,
        threads: int = 0,
        models: Optional[str] = None,
        heavy_model: str = DEFAULT_HEAVY_FACE,
        heavy_thresholds: HeavyThresholds = HeavyThresholds(),  # noqa: B008 (frozen)
        heavy_zone=None,
        heavy_deep=None,
        fit: str = "tight",
    ):
        super().__init__(threads, models, fit=fit)
        self.hth = heavy_thresholds
        self.heavy_zone = heavy_zone if heavy_zone is not None else load_face_heavy(heavy_model, threads, models)
        self.heavy_deep = heavy_deep if heavy_deep is not None else self.heavy_zone

    def second_look(self, bgr, zone: SuspectZone) -> List[Detection]:
        """The heavy look at one zone, enlarged with context; detections with source SECOND in picture coordinates. A failed run is
        counted in `model_failures` (the photo is then not checked) and gives no detections."""
        height, width = bgr.shape[:2]
        region, scale = enlargement(zone.box, width, height, self.hth.target_side)
        started = time.perf_counter()
        try:
            found = self.heavy_zone.detect(bgr, region, scale, self.hth.look_min)
        except HeavyRunError:
            self.model_failures += 1
            return []
        finally:
            self._timed("face_second", started)
        for det in found:
            det.source = SECOND
        return found

    def deep_sweep(self, bgr, plan: SweepPlan) -> List[Detection]:
        """The heavy model over the picture: the whole picture shrunk to `plan.general_side` (none when it is 0) and the full-resolution
        tiles `plan.tiles` of side `plan.deep_tile` (default and ceiling `DEEP_TILE`, DM1b). Detections with source DEEP; a failed run is
        counted in `model_failures` and gives none. The engine calls it once for the general pass and once per tile (see `engine.deep`), so
        that Stop is looked at between calls."""
        height, width = bgr.shape[:2]
        side = deep_tile_side(plan.deep_tile)
        found: List[Detection] = []
        started = time.perf_counter()
        try:
            if plan.general_side > 0:
                scale = min(1.0, plan.general_side / max(width, height))
                found += self.heavy_deep.detect(bgr, (0, 0, width, height), scale, self.hth.look_min)
            for tile in geometry.sweep_tiles(width, height, side, plan.tiles):
                found += self.heavy_deep.detect(bgr, tile, 1.0, self.hth.look_min)
        except HeavyRunError:
            self.model_failures += 1
            return []
        finally:
            self._timed("face_deep", started)
        for det in found:
            det.source = DEEP
        return geometry.nms(found)

    def take_slowest_piece(self) -> float:
        """The slowest model run of the heavy detectors since the last call, in seconds, and reset (SL6 times a run, not a whole zone)."""
        slowest = 0.0
        for detector in (self.heavy_zone, self.heavy_deep):
            if detector is not None:
                slowest = max(slowest, float(getattr(detector, "slowest_piece", 0.0)))
                detector.slowest_piece = 0.0
        return slowest

    def set_check(self, check: Optional[Callable[[], None]]) -> None:
        """Have both heavy detectors call `check` before every piece they run; it may raise to stop the sweep."""
        for detector in (self.heavy_zone, self.heavy_deep):
            if detector is not None:
                detector.check = check

    def second_judgement(self, light: Optional[Detection], found: Detection) -> Judgement:
        """The heavy score read by `HeavyThresholds`: BLUR at `h_hi`, CHECK from `h_lo`, else DROP. `light` is the first-pass detection the
        heavy one overlaps (None if it is new); the engine never lets a verdict fall below what the item already had."""
        th = self.hth
        reason = REASON_SECOND_FOUND if light is None else REASON_SECOND_CONFIRMED
        params = {"second": found.score}
        if found.score >= th.h_hi:
            return Judgement(Verdict.BLUR, reason, params)
        if found.score >= th.h_lo:
            return Judgement(Verdict.CHECK, reason, params)
        return Judgement(Verdict.DROP, reason, params)

    def video_judgement(self, track: TrackStats) -> Judgement:
        """The verdict of a finished video track: the light rule of v1 and, when the heavy model looked at the track at key frames, the heavy
        score read as a photo's (`best_second` at or above `h_hi` is BLUR, from `h_lo` CHECK, VK1). The better of the two wins: a heavy
        look never lowers what the light rule decided, and one that found nothing says nothing."""
        light = super().video_judgement(track)
        th = self.hth
        if not track.n_second:
            return light  # a track the heavy model never saw is read as in v1
        params = {
            **light.params,
            "second": track.best_second,
            "best_second": track.best_second,
            "n_second": float(track.n_second),
        }
        heavy = (
            Verdict.BLUR
            if track.best_second >= th.h_hi
            else Verdict.CHECK
            if track.best_second >= th.h_lo
            else Verdict.DROP
        )
        if VERDICT_RANK[heavy] <= VERDICT_RANK[light.verdict]:
            return Judgement(light.verdict, light.reason, params)
        # The heavy model "found" a track no light view supports or the light rule dropped; it "confirms" one the light rule kept in doubt.
        found = track.n_sweep_views == 0 or light.verdict is Verdict.DROP
        return Judgement(heavy, REASON_SECOND_FOUND if found else REASON_SECOND_CONFIRMED, params)

    def close(self) -> None:
        for detector in {id(d): d for d in (self.heavy_zone, self.heavy_deep) if d is not None}.values():
            detector.close()
        super().close()
