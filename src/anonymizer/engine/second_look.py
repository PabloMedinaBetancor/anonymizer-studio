"""The second stage of a photo: the heavy look at the suspect zones.

`second_stage` takes what the first pass left (`LookItem`s), the persons and vehicles of the picture and the class modules, builds the zones
of 4.1, visits them in order up to the budget and merges what the heavy models find by the table of 4.3. It never lowers anything (4.4): an
item keeps at least its verdict and its box only grows. A heavy look that fails is counted on its module (`model_failures`) and the caller
(`decide_photo`) then takes the photo as not checked; a failed look is never "nothing found".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .. import geometry
from ..classes.heavy import HeavyRunError
from ..contract import (
    SECOND,
    ZONE_WHY,
    ClassModule,
    Detection,
    Judgement,
    SecondLookModule,
    SuspectZone,
    Verdict,
)
from .zones import VERDICT_ORDER, LookItem, build_zones, same_place

DEFAULT_MAX_ZONES = 30  # SL5: the smallest multiple of 5 that lets 95 % of the 155 dev photos be visited whole (their 95th percentile is 30 zones)
CALL_LIMIT_S = 1.5  # SL6: a heavy call over this is a design error, counted in `slow_calls`

REASON_CONFIRMED = "reason.second_confirmed"
REASON_FOUND = "reason.second_found"


@dataclass
class ZoneVisit:
    """What one visit did, for the report and the benchmark."""

    zone: SuspectZone
    seconds: float
    found: int  # heavy detections in the zone
    failed: bool = False


@dataclass
class SecondLookResult:
    items: List[LookItem]
    zones_total: int = 0
    zones_visited: int = 0
    zones_skipped: int = 0
    confirmed: int = 0  # doubtful items the heavy model turned into sure ones
    added: int = 0  # items the heavy model brought in (new, or a dropped candidate brought back)
    failures: int = 0  # visits that failed: the photo is not checked
    slow_calls: int = 0
    visits: List[ZoneVisit] = field(default_factory=list)


def _reason(kind: str, key: str) -> str:
    return f"{kind}.{key}"


def _best_match(found: Detection, items: Sequence[LookItem]) -> Optional[LookItem]:
    """The item `found` is the same place as: the one with the highest verdict, then the best overlap. A heavy detection never makes a
    second item for a place that already has one (v1's `nms` rule)."""
    best, key = None, None
    for it in items:
        if it.kind != found.kind or not same_place(it.box, found.box()):
            continue
        k = (VERDICT_ORDER[it.verdict], geometry.iou(it.box, found.box()))
        if key is None or k > key:
            best, key = it, k
    return best


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def merge(
    item: Optional[LookItem],
    found: Detection,
    judgement: Judgement,
    items: List[LookItem],
    result: SecondLookResult,
    origin: str = SECOND,
) -> None:
    """Fold one heavy detection into the item list by the second-look table. `judgement` is the class's reading of the heavy score
    (BLUR, CHECK or DROP); the verdict never goes down and a box only grows. `origin` is what the new items are labelled `found_by`: SECOND
    for a zone, DEEP for the whole-picture sweep (an item the first pass already had keeps its label and notes the heavy score in `params`)."""
    new = judgement.verdict
    if item is None:
        if new is Verdict.DROP:
            return
        params = dict(judgement.params)
        params["second_score"] = found.score
        reason = Judgement(new, _reason(found.kind, REASON_FOUND), params)
        items.append(LookItem(found.box(), found.kind, found.score, new, reason, found, origin, found.score, params))
        result.added += 1
        return
    item.second_score = found.score if item.second_score is None else max(item.second_score, found.score)
    if origin != SECOND:
        item.params[origin] = max(item.params.get(origin, 0.0), found.score)
    if VERDICT_ORDER[new] <= VERDICT_ORDER[item.verdict]:
        return  # the table's "nothing found" rows and a weaker reading: the item stays as it was
    was = item.verdict
    item.verdict = new
    item.box = _union(item.box, found.box())
    params = dict(judgement.params)
    params["second_score"] = found.score
    if was is Verdict.CHECK:
        item.judgement = Judgement(new, _reason(item.kind, REASON_CONFIRMED), params)
        result.confirmed += 1
    else:  # a dropped candidate brought back
        item.judgement = Judgement(new, _reason(item.kind, REASON_FOUND), params)
        item.found_by = origin
        result.added += 1


def set_check_on(module, check: Optional[Callable[[], None]]) -> None:
    """Hand `check` to the module's heavy detectors, which call it before every piece they run, if the module takes one (`HeavyFaceModule.set_check`)."""
    setter = getattr(module, "set_check", None)
    if setter is not None:
        setter(check)


def _slowest_piece(module) -> Optional[float]:
    """The slowest model run of the visit just made, if the module can say (`HeavyFaceModule.take_slowest_piece`); None if it cannot."""
    taker = getattr(module, "take_slowest_piece", None)
    return None if taker is None else taker()


def second_stage(
    bgr,
    items: Sequence[LookItem],
    modules: Sequence[ClassModule],
    persons: Sequence[Detection] = (),
    vehicles: Sequence[Detection] = (),
    max_zones: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    check: Optional[Callable[[], None]] = None,
) -> SecondLookResult:
    """Visit the suspect zones of one photo with the heavy models of `modules` and return the items after the second look.

    `items` is not changed. A zone is visited by the module of its kind if it is a `SecondLookModule`; a class with no heavy look gets no
    zones. At most `max_zones` visits (default `DEFAULT_MAX_ZONES`) in the order of 4.1; the rest are counted as skipped and nothing is claimed
    about them. `progress(done, total)` is called after each visit; `check()` before each visit and, by the heavy detectors, before every piece of
    a zone cut into several, and may raise to stop the run (E8). `slow_calls` counts model runs over `CALL_LIMIT_S`: one piece of a zone, not the
    whole visit, when the module can say which was the slowest."""
    result = SecondLookResult([it.copy() for it in items])
    by_kind: Dict[str, SecondLookModule] = {m.kind: m for m in modules if isinstance(m, SecondLookModule)}
    zones = build_zones(result.items, persons, vehicles, tuple(by_kind))
    cap = DEFAULT_MAX_ZONES if max_zones is None else max_zones
    result.zones_total = len(zones)
    visit, skipped = zones[: max(0, cap)], zones[max(0, cap) :]
    result.zones_skipped = len(skipped)
    for module in by_kind.values():
        set_check_on(module, check)
    try:
        for number, zone in enumerate(visit):
            if check is not None:
                check()
            module = by_kind[zone.kind]
            _slowest_piece(module)  # start the visit with a clean slate
            started = time.perf_counter()
            before = module.model_failures
            try:
                found = module.second_look(bgr, zone)
            except (
                HeavyRunError
            ):  # a module that lets the error through: counted here, so that the photo is not checked
                module.model_failures += 1
                found = []
            seconds = time.perf_counter() - started
            failed = module.model_failures > before
            result.failures += int(failed)
            slowest = _slowest_piece(module)
            result.slow_calls += int((seconds if slowest is None else slowest) > CALL_LIMIT_S)
            for det in found:
                item = _best_match(det, result.items)
                light = item.det if item is not None and item.verdict is Verdict.CHECK else None
                merge(item, det, module.second_judgement(light, det), result.items, result)
            result.zones_visited += 1
            result.visits.append(ZoneVisit(zone, seconds, len(found), failed))
            if progress is not None:
                progress(number + 1, len(visit))
    finally:
        for module in by_kind.values():
            set_check_on(module, None)
    return result


def zone_name(zone: SuspectZone) -> str:
    """The word for why a zone was visited (the report and the benchmark)."""
    return ZONE_WHY[zone.why]
