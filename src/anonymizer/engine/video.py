"""One video clip: analyse it (sweeps, camera motion, tracks, retries) and render it from the saved analysis.

Ported from the earlier application. Rendering never runs a detector:
everything it needs (tracks, motions, cuts, margins and shapes, verdicts) is in the analysis, which is saved as JSON.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..contract import SWEEP, Box, ClassModule, Detection, Judgement, RetryPurpose, SweepPlan, Verdict
from ..geometry import tile_count
from . import video_io
from .blur import DEFAULT_STRENGTH, GAUSSIAN, RECT, blur, residual
from .events import ClipError, StopFlag
from .keyframes import KeyFrameParams, KeyFrames, refuse_deep_if_too_large
from .photos import thumbnail
from .tracks import (
    BlurSpec,
    CameraMotion,
    FinalBoxes,
    Track,
    Tracker,
    TrackerParams,
    box_to_state,
    carry,
    final_boxes,
    join_split,
    merge_duplicates,
    state_to_box,
    with_second_margin,
)

# step: a full sweep every this many frames, computed boxes in between (and a retry when something expected is
# missing). general_side: the whole frame reduced to this long side, for faces; tile_side: full-size tiles for small
# things; tiles_per_sweep: how many tiles each sweep looks at (they rotate). Plates are also searched inside the
# max_vehicles largest vehicles; plates_full_every: every how many sweeps plates are searched on the whole frame.
# Plate tiles are needed: a 42x19 px plate on the edge, behind a railing, was only seen by its tile (the earlier application, 23/09).
QUALITIES = {
    "standard": dict(
        step=2,
        general_side=960,
        tile_side=640,
        tiles_per_sweep=1,
        plate_tiles=True,
        plates_full_every=2,
        max_vehicles=6,
    ),
    "maximum": dict(
        step=1,
        general_side=1280,
        tile_side=640,
        tiles_per_sweep=99,
        plate_tiles=True,
        plates_full_every=1,
        max_vehicles=6,
    ),
}
MOTION_WIDTH = 480
# Each tile is looked at again at least this often (s). With one per sweep, 4K at 30 fps (32 tiles of 640, step 2)
# was revisited every 2.1 s and the backward extension (1.5 s at most) did not reach: a small face appearing just after
# its tile was looked at stayed visible ~0.6 s (an earlier audit, A15).
REVISIT_MAX_S = 1.5
# Below this much free memory the analysis stops cleanly instead of being killed by the system (the earlier application default).
MEMORY_FLOOR_MB = 350

Progress = Callable[[float, int, int], None]  # fraction 0..1, frame, frames


def model_failures(modules: Sequence[ClassModule]) -> int:
    """Model runs that failed so far, across the modules: when it grows during a frame, that frame is not checked."""
    return sum(m.model_failures for m in modules)


def tiles_per_sweep(n_tiles: int, wanted: int, step: int, fps: float) -> int:
    """Tiles each sweep looks at: the quality's, or more so that each tile is revisited within REVISIT_MAX_S."""
    if not n_tiles or not wanted:
        return 0
    needed = math.ceil(n_tiles * max(1, step) / (REVISIT_MAX_S * max(1.0, fps)) - 1e-9)
    return min(n_tiles, max(wanted, needed))


def available_memory_mb() -> int:
    """MemAvailable on Linux, ullAvailPhys on Windows; -1 when unknown (then nothing is stopped for memory)."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except OSError:
            pass  # no /proc (a sandbox): unknown
        return -1
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("total", "avail", "total_page", "avail_page", "total_virtual", "avail_virtual", "ext")
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail // (1024 * 1024))
    return -1


def check_memory(floor_mb: int = MEMORY_FLOOR_MB) -> None:
    free = available_memory_mb()
    if floor_mb and 0 <= free < floor_mb:
        raise MemoryError(f"only {free} MB of memory left")


def release_memory() -> None:
    """Hand freed memory back to the system: glibc keeps it otherwise (the earlier application: +0.4 MB per frame at 720p, and a 720 MB
    peak for a 20 s piece that dropped to the sum of the two phases with this between them)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        import ctypes

        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass  # not glibc (musl): nothing to trim, the memory floor check still applies


@dataclass
class VideoAnalysis:
    """Everything a render needs, and what Review shows. Saved as JSON next to the job."""

    frames: int
    fps: float
    width: int
    height: int
    motions: List[np.ndarray]
    cuts: Set[int]
    step: int
    params: TrackerParams
    tracks: List[Track]
    judgements: Dict[int, Judgement]
    specs: Dict[str, BlurSpec]
    key_frames: int = 0
    unchecked: List[int] = field(default_factory=list)  # frames where a model run failed: never exported as clean
    seconds: Dict[str, float] = field(default_factory=dict)
    calls: Dict[str, int] = field(default_factory=dict)
    thumbnails: Dict[int, Tuple[np.ndarray, int]] = field(default_factory=dict)  # track id -> (BGR, frame)
    second: Dict[str, int] = field(
        default_factory=dict
    )  # what the heavy model did at the key frames: calls, skipped, by key (VK2)

    def verdict(self, track_id: int) -> Verdict:
        return self.judgements[track_id].verdict

    def listed(self) -> List[Track]:
        """The tracks Review shows: blurred automatically or "Not sure". Detector false positives are dropped."""
        return [t for t in self.tracks if self.verdict(t.id) is not Verdict.DROP]

    def sure_ids(self) -> Set[int]:
        """The tracks the class rules are sure of (BLUR). Not the default blur set: under privacy first the "Not sure" tracks (CHECK) are
        blurred too unless the person turns them off; `listed()` has both."""
        return {t.id for t in self.tracks if self.verdict(t.id) is Verdict.BLUR}

    def to_json(self) -> dict:
        return {
            "frames": self.frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "motions": [np.round(m, 6).ravel().tolist() for m in self.motions],
            "cuts": sorted(self.cuts),
            "step": self.step,
            "key_frames": self.key_frames,
            "unchecked": self.unchecked,
            "params": asdict(self.params),
            "specs": {
                k: {"margins": {o: list(m) for o, m in s.margins.items()}, "shape": s.shape}
                for k, s in self.specs.items()
            },
            "tracks": [_track_json(t, self.judgements[t.id]) for t in self.tracks],
            "seconds": self.seconds,
            "calls": self.calls,
            "second": self.second,
        }

    @classmethod
    def from_json(cls, data: dict) -> "VideoAnalysis":
        known = TrackerParams.__dataclass_fields__
        params = TrackerParams(**{k: v for k, v in data["params"].items() if k in known})
        tracks, judgements = [], {}
        for tj in data["tracks"]:
            track, judgement = _track_of_json(tj)
            tracks.append(track)
            judgements[track.id] = judgement
        specs = {
            kind: BlurSpec({o: tuple(m) for o, m in s["margins"].items()}, s["shape"])
            for kind, s in data["specs"].items()
        }
        return cls(
            frames=data["frames"],
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            motions=[np.array(m, float).reshape(2, 3) for m in data["motions"]],
            cuts=set(data["cuts"]),
            step=data["step"],
            params=params,
            tracks=tracks,
            judgements=judgements,
            specs=specs,
            key_frames=data.get("key_frames", 0),
            unchecked=list(data.get("unchecked", [])),
            seconds=data.get("seconds", {}),
            calls=data.get("calls", {}),
            second=dict(data.get("second", {})),
        )


def _track_json(t: Track, j: Judgement) -> dict:
    return {
        "id": t.id,
        "kind": t.kind,
        "velocity": [round(float(v), 4) for v in t.velocity],
        "shot_end": t.shot_end,
        "best_verify": round(float(t.best_verify), 3),
        "n_verify": t.n_verify,
        "verify_side": round(float(t.verify_side), 1),
        "on_vehicle": bool(t.on_vehicle),
        "best_second": round(float(t.best_second), 3),
        "n_second": t.n_second,
        "second_side": round(float(t.second_side), 1),
        "second_over_sweep": t.second_over_sweep,
        "judgement": {"verdict": j.verdict.value, "reason": j.reason, "params": j.params},
        "views": {
            str(f): [round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1), round(d.score, 3), d.source]
            for f, d in sorted(t.views.items())
        },
    }


def _track_of_json(tj: dict) -> Tuple[Track, Judgement]:
    track = Track(tj["id"], tj["kind"])
    for f, v in tj["views"].items():
        track.views[int(f)] = Detection(v[0], v[1], v[2], v[3], v[4], tj["kind"], v[5])
    track.velocity = np.array(tj.get("velocity", [0.0, 0.0]), float)
    track.shot_end = tj.get("shot_end")
    track.best_verify = float(tj.get("best_verify", 0.0))
    track.n_verify = int(tj.get("n_verify", 0))
    track.verify_side = float(tj.get("verify_side", 0.0))
    track.on_vehicle = bool(tj.get("on_vehicle", False))
    track.best_second = float(tj.get("best_second", 0.0))
    track.n_second = int(tj.get("n_second", 0))
    track.second_side = float(tj.get("second_side", 0.0))
    track.second_over_sweep = int(tj.get("second_over_sweep", 0))
    track.closed = True
    j = tj["judgement"]
    return track, Judgement(Verdict(j["verdict"]), j["reason"], dict(j.get("params", {})))


def analyze_video(
    path: str,
    modules: Sequence[ClassModule],
    quality: str = "standard",
    gap_method: str = "motion",
    progress: Optional[Progress] = None,
    stop: Optional[StopFlag] = None,
    memory_floor_mb: int = MEMORY_FLOOR_MB,
    live: Optional[Callable[[dict], None]] = None,
    detection: str = "fast",
    key_params: Optional[KeyFrameParams] = None,
    persons: Optional[Callable[[object], Sequence[Detection]]] = None,
) -> VideoAnalysis:
    """Analyse a clip. `live`, when given, is called a few times a second with what is in view right now, for a
    progress display: {"found": {kind: tracks seen at least twice so far}, "boxes": [(kind, box, score, sure)]} where
    `sure` says the class rule already confirms the track. These are provisional: the final list of items comes
    with the VideoAnalysis.

    `detection` is the Options value: "fast" runs the light models as in v1; "careful" adds the heavy second look at the key frames
    (births, losses, shot changes, every N frames) for the modules that have a heavy model; "deep" adds the whole-frame
    heavy sweep (K5) after refusing a clip whose estimate is above `key_params.deep_max_seconds_of_work` (VK6). `persons` finds the
    persons of a frame (K3 and K4 look at their heads); without it those key frames are not made. A plate module of the text path
    is driven at key frames in every mode (PT7)."""
    if quality not in QUALITIES:
        raise ValueError(f"unknown quality: {quality}")
    info = video_io.probe(path)
    q = QUALITIES[quality]
    params = TrackerParams(gap_method=gap_method)
    step = max(1, int(q["step"]))
    n_tiles = tile_count(info.width, info.height, q["tile_side"])
    per_sweep = tiles_per_sweep(n_tiles, int(q["tiles_per_sweep"]), step, info.fps)
    if per_sweep:
        # A tile is looked at again every n_tiles / per_sweep sweeps: something small may take that long to be seen,
        # so tracks extend backwards at least that long.
        revisit = (n_tiles / per_sweep) * step / max(1.0, info.fps)
        params.extend_back_s = max(params.extend_back_s, min(REVISIT_MAX_S, revisit))
    tracker = Tracker(info.fps, params)
    kf = KeyFrames(modules, info.fps, detection, key_params, stop, persons)
    if kf.deep:
        refuse_deep_if_too_large(info.frames, info.fps, kf.params)
    scale = MOTION_WIDTH / max(info.width, 1) if info.width > MOTION_WIDTH else 1.0
    motion = CameraMotion(scale)
    by_kind = {m.kind: m for m in modules}
    best_view: Dict[int, Tuple[float, np.ndarray, int]] = {}
    unchecked: List[int] = []
    turn = sweeps = 0
    total = max(1, info.frames)
    seconds = {"read": 0.0, "motion": 0.0}
    t = -1
    mark = time.perf_counter()
    for t, frame, rotation in video_io.frames(path):
        seconds["read"] += time.perf_counter() - mark
        if stop is not None:
            stop.check()
        mark = time.perf_counter()
        w, h = (frame.height, frame.width) if rotation in (90, 270) else (frame.width, frame.height)
        gw, gh = max(16, int(round(w * scale / 2)) * 2), max(16, int(round(h * scale / 2)) * 2)
        if rotation in (90, 270):
            gray = frame.to_ndarray(format="gray", width=gh, height=gw)
        else:
            gray = frame.to_ndarray(format="gray", width=gw, height=gh)
        a, cut, reliable = motion.next(video_io.rotate(gray, rotation))
        seconds["motion"] += time.perf_counter() - mark
        key = t % step == 0 or cut or t == 0
        failures = model_failures(modules)
        dets: List[Detection] = []
        bgr = None
        if key:
            bgr = video_io.bgr_of(frame, rotation)
            which = [(turn + k) % n_tiles for k in range(per_sweep)] if per_sweep else []
            turn += len(which)
            sweeps += 1
            plan = SweepPlan(
                general_side=q["general_side"],
                tile_side=q["tile_side"],
                tiles=which,
                full_image=(sweeps - 1) % max(1, q["plates_full_every"]) == 0,
                plate_tiles=q["plate_tiles"],
                max_vehicles=q["max_vehicles"],
            )
            # One list in module order, faces before plates, as the earlier application swept them: the tracker's stable sort keeps it.
            dets = [d for m in modules for d in m.sweep(bgr, kf.plan_for(m, plan))]
        dets += kf.vehicle_looks(t, bgr, a, cut, reliable, key, info.fps / step)
        for track in tracker.step(t, a, cut, dets, key, reliable):
            module = by_kind[track.kind]
            found = None
            if module.wants_recover(track.stats()):
                found = module.retry(bgr, state_to_box(track.state), RetryPurpose.RECOVER)
            if found is not None:
                tracker.recover(track, t, found)
            else:
                kf.loss(tracker, track, t, bgr)
        if bgr is not None:
            # A second look when a track is born and when it is seen much larger
            for track in tracker.active:
                d = track.views.get(t)
                if d is None or d.source != SWEEP or not by_kind[track.kind].wants_verify(track.stats(), d):
                    continue
                verify = by_kind[track.kind].retry(bgr, d.box(), RetryPurpose.VERIFY)
                track.best_verify = max(track.best_verify, verify.score if verify is not None else 0.0)
                track.n_verify += 1
                track.verify_side = max(d.w, d.h)
        if bgr is not None and kf.active:
            kf.births(tracker, t, bgr)
            kf.heads(tracker, t, bgr, cut)
            kf.deep_sweep(tracker, t, bgr, cut, info.frames)
        tracker.end_step(t, info.fps / step)
        if model_failures(modules) != failures:
            unchecked.append(t)
        if bgr is not None:
            for track in tracker.tracks:
                d = track.views.get(t)
                if d is None:
                    continue
                value = d.score * max(1.0, d.w * d.h) ** 0.5
                if track.id not in best_view or value > best_view[track.id][0]:
                    best_view[track.id] = (value, thumbnail(bgr, d.box()), t)
        if t % 15 == 0:
            if live is not None:
                live(_live_view(tracker, by_kind, t))
            if progress is not None:
                progress(min(1.0, (t + 1) / total), t, total)
            if t % 150 == 0:
                release_memory()
                check_memory(memory_floor_mb)
        mark = time.perf_counter()
    if t < 0:
        raise ClipError("error.video.no_frames", detail=path)
    # The same thing lost for a while (or behind a flash) ends up as ONE track and its gap is filled (A3, A4).
    tracks = join_split(merge_duplicates(tracker.finish()), tracker.motions, tracker.cuts, t + 1, info.fps, params)
    release_memory()
    for m in modules:
        for k, v in m.timings.items():
            seconds[k] = seconds.get(k, 0.0) + v
    analysis = VideoAnalysis(
        frames=t + 1,
        fps=info.fps,
        width=info.width,
        height=info.height,
        motions=tracker.motions,
        cuts=tracker.cuts,
        step=step,
        params=params,
        tracks=tracks,
        judgements={tr.id: by_kind[tr.kind].video_judgement(tr.stats()) for tr in tracks},
        specs={m.kind: BlurSpec(with_second_margin(m.video_margins), m.blur_shape) for m in modules},
        key_frames=len(tracker.keys),
        unchecked=unchecked,
        seconds={k: round(v, 2) for k, v in seconds.items()},
        calls={k: v for m in modules for k, v in m.calls.items()},
        thumbnails={tid: (img, frame) for tid, (_, img, frame) in best_view.items()},
        second=kf.summary(),
    )
    # What is kept is what is saved: the JSON rounds boxes and motions, so a render from the analysis in memory and a
    # render from the file written from it (and read back) give the same video.
    saved = VideoAnalysis.from_json(analysis.to_json())
    saved.thumbnails = analysis.thumbnails
    return saved


def _live_view(tracker: Tracker, by_kind: Dict[str, ClassModule], t: int) -> dict:
    found: Dict[str, int] = {}
    for track in tracker.tracks:
        if len(track.views) >= 2:
            found[track.kind] = found.get(track.kind, 0) + 1
    boxes = []
    for track in tracker.active:
        d = track.views.get(t)
        if d is not None:
            sure = by_kind[track.kind].video_judgement(track.stats()).verdict is Verdict.BLUR
            boxes.append((track.kind, d.box(), d.score, sure))
    return {"found": found, "boxes": boxes}


MANUAL_MARGIN = (0.10, 0.10)  # per side; the user drew the box, so it only needs room for the follow error
MANUAL_MIN_PX = 4


def manual_boxes(
    analysis: VideoAnalysis, area_id: int, frame_from: int, frame_to: int, box: Box, follow: str = "camera"
) -> FinalBoxes:
    """The per-frame boxes of an area added in Review: `box` at `frame_from`, carried by the measured camera motion up
    to `frame_to` ("camera") or left where it is ("fixed"), with the manual margin. Shape: rectangle."""
    if follow not in ("camera", "fixed"):
        raise ValueError(f"unknown follow mode: {follow}")
    if not 0 <= frame_from <= frame_to < analysis.frames:
        raise ValueError(f"frames {frame_from}-{frame_to} outside 0-{analysis.frames - 1}")
    out: FinalBoxes = {}
    e = box_to_state(box)
    minimum = MANUAL_MIN_PX * max(1.0, max(analysis.width, analysis.height) / 1920.0)
    for t in range(frame_from, frame_to + 1):
        if t > frame_from and follow == "camera" and t not in analysis.cuts:
            e = carry(analysis.motions[t], e)  # not across a change of shot: what the camera did there says nothing
        x0, y0, x1, y1 = state_to_box(e)
        dx, dy = max((x1 - x0) * MANUAL_MARGIN[0], minimum), max((y1 - y0) * MANUAL_MARGIN[1], minimum)
        b = (
            max(0.0, x0 - dx),
            max(0.0, y0 - dy),
            min(float(analysis.width), x1 + dx),
            min(float(analysis.height), y1 + dy),
        )
        if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
            out[t] = [(b, RECT, area_id)]
    return out


def boxes_to_blur(
    analysis: VideoAnalysis,
    blur_ids: Set[int],
    extra_margin: float = 0.0,
    gap_method: Optional[str] = None,
    manual: Optional[FinalBoxes] = None,
) -> FinalBoxes:
    """frame -> [(box with margins, shape, id)] for the tracks in `blur_ids` plus the `manual` boxes: what a render
    blurs, and what Review draws."""
    tracks = [t for t in analysis.tracks if t.id in blur_ids]
    specs = {
        kind: BlurSpec({o: (mx + extra_margin, my + extra_margin) for o, (mx, my) in spec.margins.items()}, spec.shape)
        for kind, spec in analysis.specs.items()
    }
    boxes = final_boxes(
        tracks,
        analysis.motions,
        analysis.cuts,
        analysis.frames,
        analysis.fps,
        analysis.params,
        analysis.width,
        analysis.height,
        specs,
        gap_method,
    )
    for t, frame_boxes in (manual or {}).items():
        boxes.setdefault(t, []).extend(frame_boxes)
    return boxes


@dataclass
class RenderResult:
    written: int
    blurred_frames: int
    blurred_tracks: int
    max_residual: Optional[float]
    audio_reencoded: bool
    audio_lost: int
    written_duration: float
    warnings: List[str] = field(default_factory=list)  # i18n keys


PARTIAL_SUFFIX = ".anonymizer-partial"


def partial_path(target: str) -> str:
    """Where `target` is written before it is checked: a hidden name of our own, so cleaning up never touches
    anybody else's half-downloaded files."""
    folder, name = os.path.split(target)
    return os.path.join(folder, f".{name}{PARTIAL_SUFFIX}")


def render_video(
    source: str,
    target: str,
    analysis: VideoAnalysis,
    blur_ids: Set[int],
    style: str = GAUSSIAN,
    strength: float = DEFAULT_STRENGTH,
    gap_method: Optional[str] = None,
    progress: Optional[Progress] = None,
    stop: Optional[StopFlag] = None,
    threads: int = 0,
    extra_margin: float = 0.0,
    manual: Optional[FinalBoxes] = None,
    before_commit: Optional[Callable[[], None]] = None,
    profile: Optional[dict] = None,
) -> RenderResult:
    """Write `target` with the tracks in `blur_ids` blurred, plus the `manual` boxes (areas added in Review). The
    video goes to a hidden partial file, is decoded whole and checked, and only then renamed: a stop or a crash never leaves
    a file under the final name. `extra_margin` widens every class margin (fraction of the box per side).
    `before_commit` runs once the partial file is complete and checked, and may raise to refuse the final name (the API uses it
    to see that the source is still the file that was analysed)."""
    if os.path.exists(target):
        raise ClipError("error.output.exists", {"path": target})
    if analysis.unchecked:
        # A detector failed on these frames: what they show was never looked at, so the clip is not written as clean.
        frames = analysis.unchecked
        raise ClipError(
            "error.analysis.incomplete", {"frames": len(frames), "first": frames[0]}, f"frames {frames[:20]}"
        )
    info = video_io.probe(source)
    if (info.width, info.height) != (analysis.width, analysis.height):
        raise ClipError("error.source.changed", {"path": source})
    boxes = boxes_to_blur(analysis, blur_ids, extra_margin, gap_method, manual)
    partial = partial_path(target)
    total = max(1, analysis.frames)
    residuals: List[float] = []
    try:
        with video_io.Writer(source, partial, info, threads=threads) as writer:
            for t, frame, rotation in writer.frames():
                if stop is not None:
                    stop.check()
                y, u, v = video_io.yuv420_planes(frame, rotation)
                todo = boxes.get(t, [])
                for box, shape, _ in todo:
                    blur(y, box, shape, style, strength, profile)
                    half = [c / 2 for c in box]
                    blur(u, half, shape, style, strength, profile)
                    blur(v, half, shape, style, strength, profile)
                if todo and t % 10 == 0:
                    residuals += [residual(y, b) for b, _, _ in todo]
                writer.write(frame, y, u, v)
                if t % 15 == 0:
                    if progress is not None:
                        progress(min(1.0, (t + 1) / total), t, total)
                    if t % 300 == 0:
                        release_memory()
        written, duration = writer.written, writer.written_duration
        if written != analysis.frames:
            # A frame the analysis never saw would be written unchecked: the source is not the analysed one.
            raise ClipError(
                "error.analysis.incomplete", {"path": source}, f"{written} frames, {analysis.frames} analysed"
            )
        problems = video_io.check_output(partial, info, written, duration)
        if problems:
            raise ClipError("error.output.check_failed", {"path": target}, "; ".join(problems))
        if before_commit is not None:
            before_commit()
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    result = RenderResult(
        written=written,
        blurred_frames=len(boxes),
        blurred_tracks=len({tid for frame in boxes.values() for _, _, tid in frame}),
        max_residual=round(max(residuals), 2) if residuals else None,
        audio_reencoded=writer.audio_reencoded,
        audio_lost=writer.audio_lost,
        written_duration=round(duration, 3),
    )
    if info.pix_fmt not in ("yuv420p", "yuvj420p"):
        result.warnings.append("video.warning.converted_to_8bit")
    if info.odd:
        result.warnings.append("video.warning.odd_size_cropped")
    if writer.audio_reencoded:
        result.warnings.append("video.warning.audio_reencoded")
    if info.video_tracks > 1:
        result.warnings.append("video.warning.extra_video_tracks")
    if writer.audio_lost:
        result.warnings.append("video.warning.audio_tracks_lost")
    if info.track_duration and duration < info.track_duration - max(0.5, 2 / max(1.0, info.fps)):
        result.warnings.append("video.warning.source_truncated")
    return result
