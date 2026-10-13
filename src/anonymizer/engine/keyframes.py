"""The heavy model in video, at key frames only.

The light models sweep as in v1. Here the heavy second look is added where it pays: when a track is born (K1), when a tracked
thing stops being seen (K2), at a change of shot (K3) and every `period` frames (K4: the heads of persons that have no face
track), and in `deep` mode over the whole frame (K5). A class whose plates are found by text on vehicles (PT7) is driven the same
way: its vehicles are tracked here (never listed, never blurred), and the text detector is run on a vehicle at its birth, when it
has grown 1.5 times, when it is lost and every `period` frames; the regions it finds are given to the tracker as plate views.

Every heavy call goes through one budget (`heavy_per_second`, VK2); what the budget refuses is counted, never silent. Every call is
preceded by the Stop check, so a stop is felt between two calls (VK6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

from .. import contract
from ..classes.heavy import DEEP_GENERAL_SIDE
from ..contract import (
    FACE,
    RETRY,
    SWEEP,
    VEHICLE,
    ZONE_DOUBTFUL,
    ZONE_PERSON_HEAD,
    Box,
    Detection,
    SecondLookModule,
    SuspectZone,
    SweepPlan,
    Verdict,
)
from ..geometry import iou
from .events import ClipError, StopFlag
from .tracks import Track, Tracker, TrackerParams, state_to_box

K1, K2, K3, K4, K5 = "k1", "k2", "k3", "k4", "k5"
# The share of `heavy_per_second` each key frame may use in one second of video: the higher the priority, the more it may take, so a
# crowd of periodic looks never starves a birth or a loss (VK2). K5 (deep) is paced by `deep_period`, not by the budget.
BUDGET_SHARE = {K1: 1.0, K2: 1.0, K3: 0.75, K4: 0.5}
SURE_VERIFY = 0.84  # K1: a track whose light VERIFY scored at least this is already sure (v1's `photo_verify`)
GROWTH = 1.5  # K1: look again at a view this many times larger than the one last looked at (v1's `reverify_growth`)
# A person has a face track when one's centre lies in the upper half of the person's box.
PERSON_FACE_REACH = 0.5
MATCH_IOU = 0.15  # a heavy detection belongs to the track it overlaps this much (v1's retry rule)


@dataclass
class KeyFrameParams:
    """The numbers of the key frames: `period` from the interpolation measurement (VK4) and `heavy_per_second` from the machine,
    both in contract.py with their figures."""

    heavy_per_second: float = contract.HEAVY_PER_SECOND  # VK2
    period: Optional[int] = (
        contract.KEY_PERIOD_FRAMES
    )  # N of K4 and of the vehicle looks (VK4); None = one second of frames
    deep_period: Optional[int] = None  # N_deep of K5; None = three seconds of frames
    deep_ms_per_frame: Optional[float] = (
        None  # the heavy model's cost of one whole-frame sweep, from the calibration; None = measured
    )
    deep_max_seconds_of_work: float = 3600.0  # VK6: more than this is refused with the number

    def period_frames(self, fps: float) -> int:
        return self.period if self.period is not None else max(1, round(fps))

    def deep_period_frames(self, fps: float) -> int:
        return self.deep_period if self.deep_period is not None else max(1, 3 * round(fps))


def head_zone(person: Sequence[float], widen: float = 0.10, share: float = 0.35) -> Box:
    """The place of a person's head: the box widened `widen` per side, its top `share` of the height. (The photo
    engine has the same rule in classes.context; one of the two goes when they are merged.)"""
    w, h = person[2] - person[0], person[3] - person[1]
    return (person[0] - widen * w, person[1], person[2] + widen * w, person[1] + share * h)


def estimate_deep_work(frames: int, fps: float, params: KeyFrameParams) -> Optional[float]:
    """Seconds of heavy-model work that deep mode needs for a clip: sweeps (frames / N_deep, plus the first) times the cost of one
    (VK6). None when the cost of a sweep is not known yet."""
    if params.deep_ms_per_frame is None:
        return None
    sweeps = math.ceil(frames / params.deep_period_frames(fps)) + 1
    return sweeps * params.deep_ms_per_frame / 1000.0


class Budget:
    """Heavy calls per second of video (VK2). `take` says whether a key frame of a kind may still make a call in this second. A look that is
    refused and asked for again later (a birth at the next sweep) is named by `who`: it counts as one look skipped while it stays refused and
    as none once it is served, so `skipped` is the number of looks that never happened, not the number of refusals."""

    def __init__(self, per_second: float, fps: float):
        self.per_second = per_second
        self.fps = max(1.0, fps)
        self.used: Dict[int, int] = {}
        self.calls = 0
        self.refused = 0  # refusals of looks that are not named
        self.waiting: set = set()  # named looks refused and not served yet
        self.by_key: Dict[str, int] = {}
        self.found: Dict[str, int] = {}  # looks that brought a detection back, by key frame: what the calls were worth

    @property
    def skipped(self) -> int:
        return self.refused + len(self.waiting)

    def take(self, t: int, key: str, who=None) -> bool:
        second = int(t // self.fps)
        cap = int(self.per_second * BUDGET_SHARE[key] + 1e-9)
        if self.used.get(second, 0) >= cap:
            if who is None:
                self.refused += 1
            else:
                self.waiting.add(who)
            return False
        self.waiting.discard(who)
        self.used[second] = self.used.get(second, 0) + 1
        self.calls += 1
        self.by_key[key] = self.by_key.get(key, 0) + 1
        return True


def is_heavy(m) -> bool:
    return isinstance(m, SecondLookModule) and getattr(m, "heavy_zone", None) is not None


def is_deep(m) -> bool:
    return isinstance(m, SecondLookModule) and getattr(m, "heavy_deep", None) is not None


def is_vehicle_text(m) -> bool:
    """A plate module of the text path: it finds vehicles when it sweeps and gives text regions on one vehicle on demand."""
    return callable(getattr(m, "look_at_vehicle", None))


class KeyFrames:
    """What the video loop asks at each frame; see the module's description."""

    def __init__(
        self,
        modules: Sequence,
        fps: float,
        detection: str,
        params: Optional[KeyFrameParams] = None,
        stop: Optional[StopFlag] = None,
        persons: Optional[Callable[[object], Sequence[Detection]]] = None,
    ):
        self.params = params or KeyFrameParams()
        self.fps = fps
        self.stop = stop
        self.persons = persons
        careful = detection in ("careful", "deep")
        self.heavy: Dict[str, object] = {m.kind: m for m in modules if careful and is_heavy(m)}
        self.deep: Dict[str, object] = {m.kind: m for m in modules if detection == "deep" and is_deep(m)}
        self.text = [m for m in modules if is_vehicle_text(m)]
        self.budget = Budget(self.params.heavy_per_second, fps)
        self.period = self.params.period_frames(fps)
        self.deep_period = self.params.deep_period_frames(fps)
        self.vehicles = Tracker(fps, TrackerParams())  # vehicles are context: tracked here, never listed
        self.vehicle_state: Dict[int, dict] = {}
        self.state: Dict[int, dict] = {}  # by track id: {"side": last looked at, "lost": a look was made for its loss}
        self.last_heads: Optional[int] = None
        self.last_deep: Optional[int] = None
        self.deep_calls = 0
        self.deep_seconds = 0.0
        self.deep_checked = False
        self.sweeps_per_second = 1.0

    @property
    def active(self) -> bool:
        return bool(self.heavy or self.deep or self.text)

    def plan_for(self, module, plan: SweepPlan) -> SweepPlan:
        """A text-path module only finds vehicles when it sweeps: its text detector is run at the key frames of the vehicles."""
        return replace(plan, max_vehicles=0) if is_vehicle_text(module) else plan

    def _check(self) -> None:
        if self.stop is not None:
            self.stop.check()

    # --- vehicles and the text path (PT7) ---

    def vehicle_looks(
        self, t: int, bgr, a, cut: bool, reliable: bool, key: bool, sweeps_per_second: float
    ) -> List[Detection]:
        """Track the vehicles the text modules found at this sweep and run their text detector on the ones that need it. Returns the
        text regions, plate detections that sit on a vehicle, for this frame's tracker step. Called at every frame."""
        if not self.text:
            return []
        census = [v for m in self.text for v in getattr(m, "last_vehicles", []) if key]
        tracker = self.vehicles
        tracker.step(t, a, cut, census, key, reliable)
        if not key:
            return []
        tracker.end_step(t, sweeps_per_second)
        requests = []
        for track in tracker.active:
            st = self.vehicle_state.setdefault(track.id, {"look": None, "side": 0.0, "lost": False})
            d = track.views.get(t)
            if d is None:
                if st["look"] is not None and not st["lost"]:
                    requests.append((K2, state_to_box(track.state), st, track.id))
                continue
            st["lost"] = False
            side = max(d.w, d.h)
            if st["look"] is None or side >= GROWTH * st["side"]:
                requests.append((K1, d.box(), st, track.id))
            elif t - st["look"] >= self.period:
                requests.append((K4, d.box(), st, track.id))
        requests.sort(key=lambda r: (r[0], -(r[1][2] - r[1][0]) * (r[1][3] - r[1][1])))
        out: List[Detection] = []
        for key_name, box, st, vehicle_id in requests:
            if not self.budget.take(t, key_name, ("vehicle", key_name, vehicle_id)):
                continue
            self._check()
            vehicle = Detection(*box, 0.9, VEHICLE)
            for m in self.text:
                regions = m.look_at_vehicle(bgr, vehicle, [*getattr(m, "last_vehicles", []), vehicle])
                if regions:
                    self.budget.found[key_name] = self.budget.found.get(key_name, 0) + 1
                out += regions
            if key_name == K2:
                st["lost"] = True
            else:
                st["look"], st["side"] = t, max(box[2] - box[0], box[3] - box[1])
        return out

    # --- the heavy face model on tracks (K1, K2) ---

    def _zone_look(self, module, bgr, box: Box, why: int, key: str) -> List[Detection]:
        self._check()
        found = list(module.second_look(bgr, SuspectZone(tuple(box), module.kind, why, 0)))
        if found:
            self.budget.found[key] = self.budget.found.get(key, 0) + 1
        return found

    def _join(self, tracker: Tracker, track: Track, t: int, around: Box, found: Sequence[Detection]) -> None:
        """The heavy detections of a zone: the one that overlaps the track most is its second view; the rest are offered as they are."""
        mine = max(found, key=lambda d: iou(d.box(), around), default=None)
        if mine is not None and iou(mine.box(), around) < MATCH_IOU:
            mine = None
        for d in found:
            if d is mine:
                tracker.add_second(track, t, d)
            else:
                self._offer(tracker, t, d)

    def _offer(self, tracker: Tracker, t: int, d: Detection) -> None:
        """A heavy detection nobody asked for: a new track is already looked at, so K1 does not look at it again."""
        track, new = tracker.offer(t, d)
        if new:
            self.state[track.id] = {"side": max(d.w, d.h), "lost": False, "looked": True}

    def loss(self, tracker: Tracker, track: Track, t: int, bgr) -> bool:
        """K2: a track that was BLUR or CHECK was not seen at this sweep and the light retry did not find it. One heavy look around the
        predicted box. When it finds the track the track goes on, and the next look for it is `period` frames later (the interpolation
        between key frames carries the box meanwhile, VK4); when it does not, the track is not looked for again until a light model
        sees it. True when the track was found again."""
        module = self.heavy.get(track.kind)
        st = self.state.setdefault(track.id, {"side": 0.0, "lost": False})
        if module is None or st["lost"]:
            return False
        if st.get("k2_at") is not None and t - st["k2_at"] < self.period:
            return False
        if module.video_judgement(track.stats()).verdict is Verdict.DROP:
            return False
        if not self.budget.take(t, K2, (K2, track.id)):
            return False
        around = state_to_box(track.state)
        found = self._zone_look(module, bgr, around, ZONE_DOUBTFUL, K2)
        before = track.n_second
        self._join(tracker, track, t, around, found)
        st["k2_at"] = t
        st["lost"] = track.n_second == before
        return not st["lost"]

    def births(self, tracker: Tracker, t: int, bgr) -> None:
        """K1: the first sweep view of a track that its VERIFY did not make sure of, and a view 1.5 times larger than the one last looked at."""
        for track in list(tracker.active):
            d = track.views.get(t)
            module = self.heavy.get(track.kind)
            if d is None:
                continue
            st = self.state.setdefault(track.id, {"side": 0.0, "lost": False})
            if d.source in (SWEEP, RETRY):  # a light model sees it again: its loss is over
                st["lost"], st["k2_at"] = False, None
            if module is None or d.source != SWEEP or track.best_verify >= SURE_VERIFY:
                continue
            side = max(d.w, d.h)
            if st.get("looked") and side < GROWTH * st["side"]:
                continue
            if not self.budget.take(t, K1, (K1, track.id)):
                continue
            found = self._zone_look(module, bgr, d.box(), ZONE_DOUBTFUL, K1)
            st["looked"], st["side"] = True, side
            self._join(tracker, track, t, d.box(), found)

    # --- persons and deep (K3, K4, K5) ---

    def _has_face_track(self, tracker: Tracker, person: Detection) -> bool:
        top, bottom = person.y0, person.y0 + (person.y1 - person.y0) * PERSON_FACE_REACH
        for track in tracker.active:
            if track.kind != FACE:
                continue
            cx, cy = track.state[0], track.state[1]
            if person.x0 <= cx <= person.x1 and top <= cy <= bottom:
                return True
        return False

    def heads(self, tracker: Tracker, t: int, bgr, cut: bool) -> None:
        """K3 at a change of shot (and the first frame) and K4 every `period` frames: the persons found with no face track get their head
        zone visited by the heavy face model; what it finds joins a track or starts one."""
        module = self.heavy.get(FACE)
        if module is None or self.persons is None:
            return
        first = self.last_heads is None
        due = cut or first or t - self.last_heads >= self.period
        if not due:
            return
        key = K3 if (cut or first) else K4
        self.last_heads = t
        self._check()
        people = [p for p in self.persons(bgr) if not self._has_face_track(tracker, p)]
        people.sort(key=lambda p: -p.w * p.h)
        for p in people:
            if not self.budget.take(t, key):
                continue
            zone = head_zone(p.box())
            for d in self._zone_look(module, bgr, zone, ZONE_PERSON_HEAD, key):
                self._offer(tracker, t, d)

    def _whole_frame(self, module, bgr) -> List[Detection]:
        """The heavy model over the whole frame as the deep stage of photos runs it (DM1b): the general pass and then each full-resolution tile
        (640 px at most) as its own call, with the Stop check between calls and no call over a tile."""
        from . import deep

        height, width = bgr.shape[:2]
        side = deep.tile_side(0)
        tiles = deep.call_count(width, height, side)["tiles"]
        plans = [SweepPlan(general_side=DEEP_GENERAL_SIDE, tile_side=side, tiles=[], deep=True, deep_tile=side)]
        plans += [SweepPlan(general_side=0, tile_side=side, tiles=[i], deep=True, deep_tile=side) for i in range(tiles)]
        found: List[Detection] = []
        for one in plans:
            self._check()
            found += module.deep_sweep(bgr, one)
        return found

    def deep_sweep(self, tracker: Tracker, t: int, bgr, cut: bool, frames: int) -> None:
        """K5 (deep only): the heavy model over the whole frame at a change of shot, at the first frame and every `deep_period` frames."""
        if not self.deep:
            return
        due = self.last_deep is None or cut or t - self.last_deep >= self.deep_period
        if not due:
            return
        self.last_deep = t
        import time

        for module in self.deep.values():
            start = time.perf_counter()
            found = self._whole_frame(module, bgr)
            took = time.perf_counter() - start
            self.deep_calls += 1
            self.budget.by_key[K5] = self.budget.by_key.get(K5, 0) + 1
            self.budget.calls += 1
            self.deep_seconds += took
            for d in found:
                self._offer(tracker, t, d)
        if not self.deep_checked:
            self.deep_checked = True
            if self.params.deep_ms_per_frame is None:
                measured = replace(self.params, deep_ms_per_frame=1000.0 * self.deep_seconds / max(1, self.deep_calls))
                refuse_deep_if_too_large(frames, self.fps, measured)

    def summary(self) -> Dict[str, int]:
        """What the heavy model did, for the analysis file and the report (VK2): calls, skipped looks, the calls by key frame (`k1`..`k5`) and,
        of those, the looks that found something (`found_k1`..`found_k4`)."""
        b = self.budget
        return {
            "calls": b.calls,
            "skipped": b.skipped,
            **{k: b.by_key.get(k, 0) for k in (K1, K2, K3, K4, K5)},
            **{f"found_{k}": b.found.get(k, 0) for k in (K1, K2, K3, K4)},
        }


def refuse_deep_if_too_large(frames: int, fps: float, params: KeyFrameParams) -> None:
    """VK6: deep mode in video is refused above `deep_max_seconds_of_work`, with the number of minutes."""
    work = estimate_deep_work(frames, fps, params)
    if work is not None and work > params.deep_max_seconds_of_work:
        raise ClipError(
            "error.deep.too_much_work",
            {"minutes": int(math.ceil(work / 60.0)), "limit": int(params.deep_max_seconds_of_work // 60)},
        )
