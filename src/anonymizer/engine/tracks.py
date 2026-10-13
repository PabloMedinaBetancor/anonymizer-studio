"""Tracking and gap filling: from detections at some frames to a box at every frame.

Ported from the earlier application (its tracker). The class rules (which tracks are blurred, verified or searched for
again, and their margins and shape) are no longer here: they come from each contract.ClassModule.

The detector does not see an object in every frame (blur, a turn, and sweeps only every few frames to go fast).
The rule: seen, not seen, seen again means the detector missed the middle frame; the object is still there and its
position is computed. Here:

1. CameraMotion measures how the camera moves from one frame to the next (optical flow on a small grey copy, fitted
   to a similarity: rotation, scale and shift). A change of shot cuts everything.
2. Tracker (online, while the video is read) joins each detection to the track that expected it (the previous box
   carried by the camera motion plus the object's own motion) and says which tracks were missed by a sweep, so the
   caller can retry them another way.
3. complete() (at the end) fills the frames a track has no view for: by default "motion", the box before carried
   forward with the camera and the one after carried backwards, mixed by distance; and extends the track a little
   before its first view and after its last.
4. final_boxes() smooths the boxes and adds the margins of the box's origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..contract import (
    BOX_EXTEND,
    BOX_GAP,
    BOX_RETRY,
    BOX_SECOND,
    BOX_SWEEP,
    DEEP,
    SECOND,
    SWEEP,
    VEHICLE,
    Box,
    Detection,
    TrackStats,
)
from ..geometry import iou

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
MOTION, LINEAR, FORWARD, HOLD = "motion", "linear", "forward", "hold"
GAP_METHODS = (MOTION, LINEAR, FORWARD, HOLD)

# Minimum margin per side, in pixels at 1920 px on the long side (scaled with the video): on a 20 px plate 10 % is
# 2 px, less than the error of a computed box (the earlier application: boxes 2-4 px off a truck's plate in the shade).
MIN_MARGIN_PX = {BOX_SWEEP: 3, BOX_RETRY: 4, BOX_SECOND: 3, BOX_GAP: 6, BOX_EXTEND: 8}
# A view from the second look (SECOND) or from deep mode (DEEP) is one origin for margins: BOX_SECOND.
HEAVY_SOURCES = (SECOND, DEEP)


def margin_origin(source: str) -> str:
    return BOX_SECOND if source in HEAVY_SOURCES else source


def with_second_margin(margins: Mapping[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """The margins of a class with BOX_SECOND filled in when the class has none: halfway between its sweep and its retry margin
    (the pilot sets the real value from the heavy model's measured box error). Without it a heavy view would get the
    gap margin, the largest after extension."""
    out = dict(margins)
    if BOX_SECOND not in out and BOX_SWEEP in out and BOX_RETRY in out:
        out[BOX_SECOND] = tuple((a + b) / 2 for a, b in zip(out[BOX_SWEEP], out[BOX_RETRY], strict=True))
    return out


def box_to_state(box: Sequence[float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]], float)


def state_to_box(e: Sequence[float]) -> Box:
    return (e[0] - e[2] / 2, e[1] - e[3] / 2, e[0] + e[2] / 2, e[1] + e[3] / 2)


def carry(a: np.ndarray, e: np.ndarray) -> np.ndarray:
    """A state (cx, cy, w, h) moved by the 2x3 similarity `a`."""
    c = a[:, :2] @ e[:2] + a[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(a[:, :2]))))
    return np.array([c[0], c[1], e[2] * s, e[3] * s])


def inverse(a: np.ndarray) -> np.ndarray:
    ri = np.linalg.inv(a[:, :2])
    return np.hstack([ri, (-ri @ a[:, 2])[:, None]])


class CameraMotion:
    """The similarity of each frame to the previous one, in pixels of the full frame. It is fed a reduced grey copy
    (`scale` = reduced width / real width)."""

    def __init__(self, scale: float):
        import cv2

        self.cv2 = cv2
        self.scale = scale
        self.prev = None
        self.prev_hist = None

    def next(self, gray: np.ndarray) -> Tuple[np.ndarray, bool, bool]:
        """(a, cut, reliable): `a` carries coordinates of the previous frame to this one; cut is a change of shot
        (tracks do not cross it); reliable is False when the motion could not be measured (assumed still)."""
        cv2 = self.cv2
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        prev, prev_hist = self.prev, self.prev_hist
        self.prev, self.prev_hist = gray, hist
        if prev is None or prev.shape != gray.shape:
            return IDENTITY.copy(), False, True
        alike = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
        # 200 corners, window 15, 2 levels: measured in the earlier application on 150 frames against a 720 px reference with 400 corners,
        # 18 ms per frame and 1.4 px median error over 10 frames (300/21/3: 23 ms and 2.8 px).
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7)
        if pts is None or len(pts) < 12:
            return IDENTITY.copy(), alike < 0.5, False
        new, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, winSize=(15, 15), maxLevel=2)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, new, None, winSize=(15, 15), maxLevel=2)
        err = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (err < 1.0)
        if good.sum() < 12:
            return IDENTITY.copy(), alike < 0.6, False
        a, b = pts.reshape(-1, 2)[good], new.reshape(-1, 2)[good]
        m, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=500)
        if m is None or inliers is None or int(inliers.sum()) < 10:
            return IDENTITY.copy(), alike < 0.6, False
        m = m.astype(float)
        m[:, 2] /= self.scale  # the shift, in pixels of the full frame
        zoom = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))
        if not 0.8 < zoom < 1.25:  # a zoom jump like that between two frames is not the camera
            return IDENTITY.copy(), True, False
        return m, False, True


@dataclass
class Track:
    id: int
    kind: str
    views: Dict[int, Detection] = field(default_factory=dict)  # frame -> detection (sweep or retry)
    state: Optional[np.ndarray] = None  # expected box at the current frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # own motion (px/frame), camera removed
    misses: int = 0  # consecutive sweeps without a view
    closed: bool = False
    shot_end: Optional[int] = None  # last frame of the shot it lives in
    best_verify: float = 0.0  # best VERIFY retry score
    n_verify: int = 0
    verify_side: float = 0.0  # longest side of the view verified last
    on_vehicle: bool = False  # plates: any view over a detected vehicle
    best_second: float = 0.0  # best score of a heavy-model view (SECOND or DEEP) so far
    n_second: int = 0
    second_side: float = 0.0  # longest side of the view the heavy model looked at last
    second_over_sweep: int = 0  # frames where a heavy view replaced a sweep view (their sweep view still counts as one)

    @property
    def first(self) -> int:
        return min(self.views)

    @property
    def last(self) -> int:
        return max(self.views)

    @property
    def max_score(self) -> float:
        return max(d.score for d in self.views.values())

    def best_view(self) -> Tuple[int, Detection]:
        """The frame with the largest view among the well-scored ones (for the Review thumbnail)."""
        top = self.max_score
        return max(((t, d) for t, d in self.views.items() if d.score >= top * 0.8), key=lambda x: x[1].w * x[1].h)

    def stats(self) -> TrackStats:
        """What a class rule may know about this track, as of now."""
        return TrackStats(
            kind=self.kind,
            n_views=len(self.views),
            n_sweep_views=sum(1 for d in self.views.values() if d.source == SWEEP) + self.second_over_sweep,
            max_score=self.max_score,
            best_verify=self.best_verify,
            n_verify=self.n_verify,
            verify_side=self.verify_side,
            on_vehicle=self.on_vehicle,
            best_second=self.best_second,
            n_second=self.n_second,
        )


@dataclass
class TrackerParams:
    max_miss_s: float = 1.0  # a track unseen by the sweeps for this long is closed
    extend_back_s: float = 0.3  # before the first view (it may have been there before a sweep looked)
    extend_forward_s: float = 0.3  # after the last (it leaves the shot: blurred a moment too long)
    gap_method: str = MOTION
    iou_min: float = 0.10  # joining a detection to a track
    dist_max: float = 0.8  # centre distance / expected longest side
    # Two tracks of the same shot, one after the other in the same place, are one thing the detector lost for a while
    # when the gap is at most this; and a "shot" of up to shot_min frames (a flash, an exposure jump) does not cut
    # (an earlier audit, A3 and A4).
    join_max_s: float = 4.0
    shot_min: int = 3


def _cost(expected: np.ndarray, d: Detection) -> Optional[float]:
    v = iou(state_to_box(expected), d.box())
    side = max(expected[2], expected[3], 1.0)
    dist = float(np.hypot((d.x0 + d.x1) / 2 - expected[0], (d.y0 + d.y1) / 2 - expected[1])) / side
    size = max(d.w, d.h) / side
    if not 0.4 < size < 2.5:
        return None
    return (1 - v) + dist


class Tracker:
    """Joins detections into tracks while the video is read, and keeps every frame's motion so complete() can fill
    the gaps afterwards."""

    def __init__(self, fps: float, params: Optional[TrackerParams] = None):
        self.params = params or TrackerParams()
        self.fps = max(1.0, float(fps))
        self.motions: List[np.ndarray] = []  # motions[t]: from t-1 to t
        self.reliable: List[bool] = []
        self.cuts: Set[int] = set()  # t: frame t starts a new shot
        self.keys: Set[int] = set()
        self.tracks: List[Track] = []
        self.active: List[Track] = []
        self.t = -1

    def _new(self, t: int, d: Detection) -> Track:
        track = Track(len(self.tracks), d.kind)
        track.views[t] = d
        track.state = box_to_state(d.box())
        track.on_vehicle = bool(d.on_vehicle)
        self.tracks.append(track)
        self.active.append(track)
        return track

    def _observe(self, track: Track, t: int, d: Detection) -> None:
        if track.views:
            t0 = track.last
            # own motion: where it is now minus where the camera alone would have left it
            e = box_to_state(track.views[t0].box())
            for k in range(t0 + 1, t + 1):
                e = carry(self.motions[k], e)
            own = (np.array([(d.x0 + d.x1) / 2, (d.y0 + d.y1) / 2]) - e[:2]) / max(1, t - t0)
            side = max(d.w, d.h, 1.0)
            own = np.clip(own, -side, side)  # an absurd jump (a wrong join) does not send the track flying
            track.velocity = 0.5 * track.velocity + 0.5 * own if len(track.views) > 1 else own * 0.5
        track.views[t] = d
        track.state = box_to_state(d.box())
        track.misses = 0
        if d.on_vehicle:
            track.on_vehicle = True

    def _close(self, track: Track, shot_end: Optional[int] = None) -> None:
        track.closed = True
        track.shot_end = shot_end
        if track in self.active:
            self.active.remove(track)

    def step(
        self, t: int, a: np.ndarray, cut: bool, dets: Sequence[Detection], key: bool, reliable: bool = True
    ) -> List[Track]:
        """Advance to frame t. On key frames, returns the tracks that were expected and not seen by this sweep: the
        caller may try to recover() them another way."""
        if t != self.t + 1:
            raise ValueError(f"frames go one by one: got {t} after {self.t}")
        self.t = t
        self.motions.append(np.asarray(a, float))
        self.reliable.append(bool(reliable))
        if key:
            self.keys.add(t)
        if cut:
            self.cuts.add(t)
            for track in list(self.active):
                self._close(track, shot_end=t - 1)
        else:
            for track in self.active:
                e = carry(self.motions[t], track.state)
                e[:2] += track.velocity
                track.velocity *= 0.95
                track.state = e
        # greedy join by cost, highest scores first (stable: equal scores keep the sweep's order, faces then plates)
        free = list(self.active)
        for d in sorted(dets, key=lambda d: -d.score):
            best, best_cost = None, None
            for track in free:
                if track.kind != d.kind:
                    continue
                c = _cost(track.state, d)
                if c is None:
                    continue
                if iou(state_to_box(track.state), d.box()) < self.params.iou_min and c - 1 > self.params.dist_max:
                    continue
                if best_cost is None or c < best_cost:
                    best, best_cost = track, c
            if best is not None:
                self._observe(best, t, d)
                free.remove(best)
            else:
                self._new(t, d)
        if not key:
            return []
        return [track for track in free if track.kind != VEHICLE]

    def recover(self, track: Track, t: int, d: Detection) -> None:
        """A retry found the track that was missing."""
        self._observe(track, t, d)

    def add_second(self, track: Track, t: int, d: Detection) -> Detection:
        """A heavy-model view (source SECOND or DEEP) of `track` at frame t. Where the track already has a view at t
        the box is the union of the two, never smaller, and the score the larger; where it has none (the track was lost and the heavy
        model found it) the view is observed like a recovery. Returns the view kept."""
        light = track.views.get(t)
        if light is None:
            view = Detection(d.x0, d.y0, d.x1, d.y1, d.score, track.kind, d.source, d.on_vehicle)
            self._observe(track, t, view)
        else:
            view = Detection(
                min(light.x0, d.x0),
                min(light.y0, d.y0),
                max(light.x1, d.x1),
                max(light.y1, d.y1),
                max(light.score, d.score),
                track.kind,
                d.source,
                light.on_vehicle if light.on_vehicle is not None else d.on_vehicle,
            )
            if light.source == SWEEP:
                track.second_over_sweep += 1
            track.views[t] = view
        track.best_second = max(track.best_second, d.score)
        track.n_second += 1
        track.second_side = max(d.w, d.h)
        return view

    def offer(self, t: int, d: Detection) -> Tuple[Track, bool]:
        """A heavy detection that no track asked for (a head zone, a deep sweep): it joins the active track that expects it, or starts
        a track of its own. Returns (track, new)."""
        best, best_cost = None, None
        for track in self.active:
            if track.kind != d.kind:
                continue
            c = _cost(track.state, d)
            if c is None or (
                iou(state_to_box(track.state), d.box()) < self.params.iou_min and c - 1 > self.params.dist_max
            ):
                continue
            if best_cost is None or c < best_cost:
                best, best_cost = track, c
        if best is None:
            track = self._new(t, d)
            track.best_second, track.n_second, track.second_side = d.score, 1, max(d.w, d.h)
            return track, True
        self.add_second(best, t, d)
        return best, False

    def end_step(self, t: int, sweeps_per_second: float) -> None:
        """Count a miss for the tracks still unseen by this sweep and close the ones missing for too long."""
        if t not in self.keys:
            return
        limit = max(1, int(round(self.params.max_miss_s * sweeps_per_second)))
        for track in list(self.active):
            if t not in track.views:
                track.misses += 1
                if track.misses > limit:
                    self._close(track)

    def finish(self) -> List[Track]:
        for track in list(self.active):
            self._close(track, shot_end=self.t)
        return self.tracks


def _absorb(a: Track, b: Track) -> None:
    a.best_verify = max(a.best_verify, b.best_verify)
    a.n_verify += b.n_verify
    a.best_second = max(a.best_second, b.best_second)
    a.n_second += b.n_second
    a.second_side = max(a.second_side, b.second_side)
    a.second_over_sweep += b.second_over_sweep
    a.on_vehicle = a.on_vehicle or b.on_vehicle


def merge_duplicates(tracks: List[Track], iou_min: float = 0.5) -> List[Track]:
    """Two tracks following the same thing (one born from a sweep while the other lived on retries; seen on a truck's
    plate in the earlier application) become one: in Review, turning one off left the other blurring. Same kind and boxes that match
    (IoU > iou_min) in at least half of the frames they share."""
    alive = sorted(tracks, key=lambda p: p.id)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(alive):
            for b in alive[i + 1 :]:
                if a.kind != b.kind:
                    continue
                shared = set(a.views) & set(b.views)
                if not shared:
                    continue
                same = sum(1 for t in shared if iou(a.views[t].box(), b.views[t].box()) > iou_min)
                if same * 2 < len(shared):
                    continue
                for t, d in b.views.items():
                    if t not in a.views or d.score > a.views[t].score:
                        a.views[t] = d
                _absorb(a, b)
                if b.shot_end is not None and (a.shot_end is None or b.shot_end > a.shot_end):
                    a.shot_end = b.shot_end
                alive.remove(b)
                changed = True
                break
            if changed:
                break
    return alive


def effective_cuts(cuts: Set[int], n: int, shot_min: int = 3) -> Set[int]:
    """The cuts that separate real shots. A "shot" shorter than `shot_min` frames (a flash, an exposure jump, a broken
    frame) is not one: its two cuts count neither for joining tracks nor for filling (an earlier audit, A4: the face was
    left visible in exactly the frame of the flash)."""
    cs = sorted(c for c in cuts if 0 < c < n)
    kept = set(cs)
    edges = [0] + cs + [n]
    for k in range(len(edges) - 1):
        start, end = edges[k], edges[k + 1]
        if end - start < shot_min:
            kept.discard(start)
            kept.discard(end)
    return kept


def join_split(
    tracks: List[Track], motions: List[np.ndarray], cuts: Set[int], n: int, fps: float, params: TrackerParams
) -> List[Track]:
    """A face the detector loses for longer than max_miss_s and finds again in the same place used to end up as two
    tracks with the gap between them visible (an earlier audit, A3). Two tracks of the same kind and shot that do not
    overlap are joined when the second's first view falls where the first's last view lands with the camera motion
    (IoU > 0.1 or centres less than a side apart, similar size) and the gap is at most join_max_s; complete() then
    fills the gap with the gap margins. Two different things in the same place get blurred a moment too long,
    which is harmless."""
    real_cuts = sorted(effective_cuts(cuts, n, params.shot_min))
    longest = int(round(params.join_max_s * max(1.0, fps)))
    alive = sorted(tracks, key=lambda p: (p.first, p.id))
    i = 0
    while i < len(alive):
        a = alive[i]
        joined = False
        for b in alive[i + 1 :]:
            if b.first > a.last + longest:
                break
            if b.kind != a.kind or b.first <= a.last:
                continue
            if any(a.last < c <= b.first for c in real_cuts):
                continue
            e = box_to_state(a.views[a.last].box())
            for k in range(a.last + 1, min(b.first, len(motions) - 1) + 1):
                e = carry(motions[k], e)
            db = b.views[b.first]
            side = max(e[2], e[3], 1.0)
            dist = float(np.hypot((db.x0 + db.x1) / 2 - e[0], (db.y0 + db.y1) / 2 - e[1])) / side
            size = max(db.w, db.h) / side
            if not (0.5 < size < 2.0 and (iou(state_to_box(e), db.box()) > 0.1 or dist < 1.0)):
                continue
            a.views.update(b.views)
            _absorb(a, b)
            a.verify_side = max(a.verify_side, b.verify_side)
            a.shot_end = b.shot_end
            alive.remove(b)
            joined = True
            break
        if not joined:
            i += 1
    return sorted(alive, key=lambda p: p.id)


def _shot_bounds(t: int, n: int, cuts: Set[int]) -> Tuple[int, int]:
    """First and last frame of the shot that contains t."""
    start = max([c for c in cuts if c <= t], default=0)
    end = min([c for c in cuts if c > t], default=n) - 1
    return start, end


def complete(
    track: Track,
    motions: List[np.ndarray],
    cuts: Set[int],
    n: int,
    params: TrackerParams,
    fps: float,
    method: Optional[str] = None,
) -> Dict[int, Tuple[np.ndarray, str]]:
    """State (cx, cy, w, h) and origin for every frame of the track: the view's source where it was seen, BOX_GAP
    between two views, BOX_EXTEND before the first and after the last. Never crosses a change of shot."""
    method = method or params.gap_method
    if method not in GAP_METHODS:
        raise ValueError(f"unknown gap method: {method}")
    ts = sorted(track.views)
    out: Dict[int, Tuple[np.ndarray, str]] = {
        t: (box_to_state(track.views[t].box()), margin_origin(track.views[t].source)) for t in ts
    }
    for ta, tb in zip(ts, ts[1:], strict=False):
        if tb - ta < 2:
            continue
        ea, eb = out[ta][0], out[tb][0]
        if method == HOLD:
            for t in range(ta + 1, tb):
                out[t] = (ea.copy(), BOX_GAP)
            continue
        if method == LINEAR:
            for t in range(ta + 1, tb):
                w = (t - ta) / (tb - ta)
                out[t] = ((1 - w) * ea + w * eb, BOX_GAP)
            continue
        forward = {}
        e = ea.copy()
        for t in range(ta + 1, tb):
            e = carry(motions[t], e)
            forward[t] = e.copy()
        if method == FORWARD:
            for t in range(ta + 1, tb):
                out[t] = (forward[t], BOX_GAP)
            continue
        # backwards with the camera from tb, mixed by distance: each side's error grows as it gets further away
        e = eb.copy()
        backward = {}
        for t in range(tb - 1, ta, -1):
            e = carry(inverse(motions[t + 1]), e)
            backward[t] = e.copy()
        for t in range(ta + 1, tb):
            w = (t - ta) / (tb - ta)
            out[t] = ((1 - w) * forward[t] + w * backward[t], BOX_GAP)
    start, _ = _shot_bounds(ts[0], n, cuts)
    _, end = _shot_bounds(ts[-1], n, cuts)
    e = out[ts[0]][0].copy()
    v = track.velocity.copy()
    for t in range(ts[0] - 1, max(start, ts[0] - int(round(params.extend_back_s * fps))) - 1, -1):
        e = carry(inverse(motions[t + 1]), e)
        e[:2] -= v
        v *= 0.9
        out[t] = (e.copy(), BOX_EXTEND)
    e = out[ts[-1]][0].copy()
    v = track.velocity.copy()
    for t in range(ts[-1] + 1, min(end, ts[-1] + int(round(params.extend_forward_s * fps))) + 1):
        e = carry(motions[t], e)
        e[:2] += v
        v *= 0.9
        out[t] = (e.copy(), BOX_EXTEND)
    return out


@dataclass
class BlurSpec:
    """How a kind is blurred in video: margins by box origin (fractions of width and height per side) and shape."""

    margins: Mapping[str, Tuple[float, float]]
    shape: str


FinalBoxes = Dict[int, List[Tuple[Box, str, int]]]


def final_boxes(
    tracks: Sequence[Track],
    motions: List[np.ndarray],
    cuts: Set[int],
    n: int,
    fps: float,
    params: TrackerParams,
    width: int,
    height: int,
    specs: Mapping[str, BlurSpec],
    method: Optional[str] = None,
) -> FinalBoxes:
    """frame -> [(box with margins, shape, track id)] for the tracks to blur (the caller has already applied the
    class verdicts and the user's decisions). A 5-frame mean smooths the jitter without ever leaving out the box of
    that frame."""
    cuts = effective_cuts(cuts, n, params.shot_min)  # a flash does not cut the extensions (A4)
    out: FinalBoxes = {}
    for track in tracks:
        spec = specs[track.kind]
        boxes = complete(track, motions, cuts, n, params, fps, method)
        ts = sorted(boxes)
        est = np.array([boxes[t][0] for t in ts])
        # A computed box cannot grow or shrink without limit (a badly measured scale that accumulates): between half
        # the smallest and 1.5 times the largest of the real views.
        vw = [d.w for d in track.views.values()]
        vh = [d.h for d in track.views.values()]
        est[:, 2] = np.clip(est[:, 2], 0.5 * min(vw), 1.5 * max(vw))
        est[:, 3] = np.clip(est[:, 3], 0.5 * min(vh), 1.5 * max(vh))
        if len(ts) >= 5:
            kernel = np.ones(5) / 5
            padded = np.pad(est, ((2, 2), (0, 0)), mode="edge")
            smooth = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(4)], axis=1)
        else:
            smooth = est
        seen = np.array(sorted(track.views))
        for i, t in enumerate(ts):
            origin = boxes[t][1]
            mx, my = spec.margins.get(origin, spec.margins[BOX_GAP])
            if origin in (BOX_GAP, BOX_EXTEND):
                # the further from a real view, the larger the error: +1 % per frame of distance, at most 20
                far = min(20, int(np.min(np.abs(seen - t))))
                mx, my = mx + 0.01 * far, my + 0.01 * far
            c1 = state_to_box(smooth[i])
            c0 = state_to_box(est[i])
            minimum = MIN_MARGIN_PX.get(origin, 6) * max(1.0, max(width, height) / 1920.0)
            if iou(c0, c1) > 0.2:
                # the smoothed box and this frame's together: what was detected is never left out
                x0, y0 = min(c0[0], c1[0]), min(c0[1], c1[1])
                x1, y1 = max(c0[2], c1[2]), max(c0[3], c1[3])
            else:
                # the box jumps between frames (motion badly measured): joining them would smear across the image;
                # this frame's box is used
                x0, y0, x1, y1 = c0
            w, h = x1 - x0, y1 - y0
            dx, dy = max(w * mx, minimum), max(h * my, minimum)
            box = (max(0.0, x0 - dx), max(0.0, y0 - dy), min(float(width), x1 + dx), min(float(height), y1 + dy))
            if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
                out.setdefault(t, []).append((box, spec.shape, track.id))
    return out
