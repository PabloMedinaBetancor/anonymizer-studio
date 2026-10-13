"""The zones the second look visits and how a zone is cut and enlarged (4.2).

A photo's first pass leaves a list of `LookItem`s: what the light models found and what the class rules made of it (BLUR, CHECK, or DROP
for a candidate the rules dropped). `build_zones` turns that list, the persons and the vehicles of the picture into the ordered list of
`SuspectZone`s of 4.1; the budget (`max_zones`) and the merge of what the heavy model finds belong to `second_look`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .. import geometry
from ..classes.context import head_zone
from ..classes.heavy import enlargement
from ..contract import (
    FACE,
    PLATE,
    ZONE_DOUBTFUL,
    ZONE_DROPPED_STRONG,
    ZONE_DROPPED_WEAK,
    ZONE_PERSON_HEAD,
    ZONE_VEHICLE,
    Box,
    Detection,
    Judgement,
    SuspectZone,
    Verdict,
)

DROPPED_STRONG_SCORE = 0.65  # a dropped candidate at or above this goes before the heads of persons
DUPLICATE_IOU = 0.5  # two zones of one kind overlapping this much are one visit...
DUPLICATE_INSIDE = 0.8  # ...or one lying this much inside the other when they are of similar size
DUPLICATE_SIZE_RATIO = (
    2.5  # (a small zone inside a much bigger one is not its duplicate: the bigger one is a head, say)
)
VERDICT_ORDER = {Verdict.DROP: 0, Verdict.CHECK: 1, Verdict.BLUR: 2}

# `LookItem.found_by` values
LIGHT = "light"


@dataclass
class LookItem:
    """One candidate of the first pass, in the pixels of the photo. `box` is what would be blurred (before the class margin): the union of
    the sweep box and its VERIFY box. A DROP item is a candidate the rules dropped; it stays in the list because the second look may
    bring it back."""

    box: Box
    kind: str
    score: float
    verdict: Verdict
    judgement: Judgement
    det: Optional[Detection] = None  # the sweep detection behind it, for the class's second judgement
    found_by: str = LIGHT
    second_score: Optional[float] = None
    params: Dict[str, float] = field(default_factory=dict)

    def copy(self) -> "LookItem":
        return LookItem(
            self.box,
            self.kind,
            self.score,
            self.verdict,
            self.judgement,
            self.det,
            self.found_by,
            self.second_score,
            dict(self.params),
        )


def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def inside_share(inner: Sequence[float], outer: Sequence[float]) -> float:
    """The share of `inner` that lies within `outer`."""
    iw = min(inner[2], outer[2]) - max(inner[0], outer[0])
    ih = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / max(1e-9, area(inner))


def same_place(a: Sequence[float], b: Sequence[float]) -> bool:
    """Whether two boxes are one place: the `nms` rule of the first pass (IoU over 0.3, or one almost inside the other)."""
    return geometry.iou(a, b) > 0.3 or max(inside_share(a, b), inside_share(b, a)) > 0.8


def _duplicate(a: Sequence[float], b: Sequence[float]) -> bool:
    """Two zones that need one visit between them."""
    if geometry.iou(a, b) >= DUPLICATE_IOU:
        return True
    small, big = (a, b) if area(a) <= area(b) else (b, a)
    if area(big) > DUPLICATE_SIZE_RATIO * max(1e-9, area(small)):
        return False
    return inside_share(small, big) >= DUPLICATE_INSIDE


def _has_face_zone(person: Detection, items: Sequence[LookItem]) -> bool:
    """A person has a face zone when a BLUR or CHECK face item has its centre inside the person's box, in its upper half."""
    top, bottom = person.y0, person.y0 + (person.y1 - person.y0) / 2
    for it in items:
        if it.kind != FACE or it.verdict is Verdict.DROP:
            continue
        cx, cy = (it.box[0] + it.box[2]) / 2, (it.box[1] + it.box[3]) / 2
        if person.x0 <= cx <= person.x1 and top <= cy <= bottom:
            return True
    return False


def build_zones(
    items: Sequence[LookItem],
    persons: Sequence[Detection] = (),
    vehicles: Sequence[Detection] = (),
    kinds: Sequence[str] = (FACE, PLATE),
) -> List[SuspectZone]:
    """The suspect zones of one photo in visiting order: doubtful, dropped and strong, heads of persons with no face zone,
    dropped and weak, vehicles; inside a kind of zone the larger first. Of two zones of one class that are the same place the earlier
    is kept. `kinds` are the classes that have a heavy look: nothing is built for the others. `SuspectZone.item` is the index in
    `items`; `priority` is the place in the visiting order."""
    wanted = set(kinds)
    zones: List[SuspectZone] = []
    for index, it in enumerate(items):
        if it.kind not in wanted:
            continue
        if it.verdict is Verdict.CHECK:
            zones.append(SuspectZone(it.box, it.kind, ZONE_DOUBTFUL, 0, index))
        elif it.verdict is Verdict.DROP:
            why = ZONE_DROPPED_STRONG if it.score >= DROPPED_STRONG_SCORE else ZONE_DROPPED_WEAK
            zones.append(SuspectZone(it.box, it.kind, why, 0, index))
    if FACE in wanted:
        for person in persons:
            if not _has_face_zone(person, items):
                zones.append(SuspectZone(head_zone(person.box()), FACE, ZONE_PERSON_HEAD, 0, None))
    if PLATE in wanted:
        for vehicle in vehicles:
            zones.append(SuspectZone(vehicle.box(), PLATE, ZONE_VEHICLE, 0, None))
    zones.sort(key=lambda z: (z.why, -area(z.box)))
    kept: List[SuspectZone] = []
    for zone in zones:
        if any(k.kind == zone.kind and _duplicate(k.box, zone.box) for k in kept):
            continue
        zone.priority = len(kept)
        kept.append(zone)
    return kept


__all__ = ["LookItem", "build_zones", "enlargement", "same_place"]
