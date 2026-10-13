"""Deep mode on photos: the heavy model over the whole picture, united with the first pass.

`deep_stage` sweeps a picture with the heavy models of the classes that have one, in calls small enough to stop between (DM3), and folds what
they find into the first pass's items by the same table as the second look (4.3). With `heavy_only` the classes that have a heavy model start
from nothing: the light result of those classes is left out and only the heavy sweep decides. `estimate` says before it runs how many heavy
calls and how many seconds a picture needs, and a picture that would take more than `deep_max_tiles` is refused (DM2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import geometry
from ..classes.heavy import DEEP_GENERAL_SIDE, DEEP_TILE, HeavyRunError, deep_tile_side, pieces_in
from ..contract import DEEP, ClassModule, Detection, SecondLookModule, SweepPlan, Verdict
from . import second_look
from .zones import LookItem

NO_LIMIT = 10**9  # a `max_tiles` that no picture reaches (the benchmark runs every photo, whatever its size)
DEEP_MAX_TILES = 120  # DM2: above this many heavy calls (a 64 MP panorama) the picture is refused
DEMO_REFUSAL = "error.demo.deep"
TOO_LARGE = "error.deep.too_large"
CALIBRATION_CALLS = 3  # a calibration is the median of this many calls on a 640 px picture


class DeepRefused(Exception):
    """Deep mode is not allowed or not possible here. `key` is the i18n key of the message and `params` its placeholders."""

    def __init__(self, key: str, params: Optional[Dict[str, float]] = None):
        super().__init__(key)
        self.key = key
        self.params = params or {}


@dataclass
class Estimate:
    calls: int  # heavy calls: the pieces of the general pass plus one per tile
    tiles: int  # full-resolution tiles
    seconds: float  # calls x the measured time of one call on this machine


def tile_side(requested: int = 0) -> int:
    """The side of a deep tile (DM1b): between 320 and 640 px, the default 640; a negative side is a `ValueError`."""
    return deep_tile_side(requested)


def max_tiles_for(side: int, base: int = DEEP_MAX_TILES) -> int:
    """`deep_max_tiles` follows from the tile size (DM1b): a smaller tile is a cheaper call, so more of them are allowed in the same work."""
    return int(round(base * (DEEP_TILE / max(1, side)) ** 2))


def call_count(width: int, height: int, side: int = DEEP_TILE) -> Dict[str, int]:
    """The heavy calls of a deep sweep of a width x height picture: the pieces of the general pass and the tiles."""
    scale = min(1.0, DEEP_GENERAL_SIDE / max(width, height))
    general = pieces_in(max(1, round(width * scale)), max(1, round(height * scale)))
    tiles = geometry.tile_count(width, height, side)
    return {"general": general, "tiles": tiles, "calls": general + tiles}


def estimate(width: int, height: int, ms_per_call: float, side: int = 0, max_tiles: Optional[int] = None) -> Estimate:
    """What a deep sweep costs on this machine (`ms_per_call` is its calibration), or `DeepRefused("error.deep.too_large")` when the picture
    needs more tiles than `max_tiles` (default `max_tiles_for(side)`)."""
    side = tile_side(side)
    counts = call_count(width, height, side)
    limit = max_tiles_for(side) if max_tiles is None else max_tiles
    if counts["tiles"] > limit:
        raise DeepRefused(TOO_LARGE, {"tiles": counts["tiles"], "limit": limit})
    return Estimate(counts["calls"], counts["tiles"], round(counts["calls"] * ms_per_call / 1000.0, 2))


def calibrate(module: SecondLookModule, calls: int = CALIBRATION_CALLS) -> float:
    """Milliseconds per heavy call on this machine: the median of `calls` runs of the deep model on a 640 px picture of noise, after one run that
    opens the session and is not counted. The Studio stores it in the user settings (DM2)."""
    noise = np.random.default_rng(0).integers(0, 255, (DEEP_TILE, DEEP_TILE, 3), dtype=np.uint8)
    plan = SweepPlan(general_side=DEEP_TILE, tile_side=DEEP_TILE, tiles=[], deep=True)
    before = module.model_failures
    module.deep_sweep(noise, plan)
    times = []
    for _ in range(max(1, calls)):
        started = time.perf_counter()
        module.deep_sweep(noise, plan)
        times.append(1000 * (time.perf_counter() - started))
    if module.model_failures > before:
        raise HeavyRunError("the heavy model failed during its calibration: no time per call can be stored")
    return sorted(times)[len(times) // 2]


def allow_detection(detection: str, demo: bool) -> None:
    """The hosted demo refuses deep mode (DM4); careful and fast stay allowed. Raises `DeepRefused("error.demo.deep")`."""
    if demo and detection == "deep":
        raise DeepRefused(DEMO_REFUSAL)


@dataclass
class DeepResult:
    items: List[LookItem]
    calls: int = 0
    tiles: int = 0
    added: int = 0
    confirmed: int = 0
    failures: int = 0
    seconds: float = 0.0
    detections: List[Detection] = field(default_factory=list)


def deep_stage(
    bgr,
    items: Sequence[LookItem],
    modules: Sequence[ClassModule],
    tile: int = 0,
    max_tiles: Optional[int] = None,
    heavy_only: bool = False,
    progress: Optional[Callable[[int, int], None]] = None,
    check: Optional[Callable[[], None]] = None,
) -> DeepResult:
    """Sweep the picture with the heavy model of every `SecondLookModule` in `modules` and merge what it finds into `items` (not changed).

    `check()` is called between heavy calls and by the detectors between the pieces of one call, and may raise to stop (E8); `progress(done,
    total)` is called after each call. Raises `DeepRefused("error.deep.too_large")` before running anything when the picture needs more
    tiles than `max_tiles`. A heavy call that fails is counted in its module's `model_failures` (the photo is then not checked)."""
    height, width = bgr.shape[:2]
    side = tile_side(tile)
    counts = call_count(width, height, side)
    limit = max_tiles_for(side) if max_tiles is None else max_tiles
    if counts["tiles"] > limit:
        raise DeepRefused(TOO_LARGE, {"tiles": counts["tiles"], "limit": limit})
    heavy_modules = {m.kind: m for m in modules if isinstance(m, SecondLookModule)}
    kept = [it.copy() for it in items if not heavy_only or it.kind not in heavy_modules]
    result = DeepResult(kept, calls=counts["calls"], tiles=counts["tiles"])
    tally = second_look.SecondLookResult(kept)  # the merge of the second look counts what it adds and confirms
    done = 0
    total = len(heavy_modules) * (1 + counts["tiles"])
    for module in heavy_modules.values():
        second_look.set_check_on(module, check)
        try:
            passes = [SweepPlan(general_side=DEEP_GENERAL_SIDE, tile_side=side, tiles=[], deep=True, deep_tile=side)]
            passes += [
                SweepPlan(general_side=0, tile_side=side, tiles=[i], deep=True, deep_tile=side)
                for i in range(counts["tiles"])
            ]
            for plan in passes:
                if check is not None:
                    check()
                before = module.model_failures
                started = time.perf_counter()
                try:
                    found = module.deep_sweep(bgr, plan)
                except HeavyRunError:
                    module.model_failures += 1
                    found = []
                result.seconds += time.perf_counter() - started
                result.failures += int(module.model_failures > before)
                for det in found:
                    item = second_look._best_match(det, tally.items)
                    light = item.det if item is not None and item.verdict is Verdict.CHECK else None
                    second_look.merge(item, det, module.second_judgement(light, det), tally.items, tally, DEEP)
                    result.detections.append(det)
                done += 1
                if progress is not None:
                    progress(done, total)
        finally:
            second_look.set_check_on(module, None)
    result.items = tally.items
    result.added, result.confirmed = tally.added, tally.confirmed
    return result


__all__ = [
    "DEEP_MAX_TILES",
    "DeepRefused",
    "DeepResult",
    "Estimate",
    "allow_detection",
    "calibrate",
    "call_count",
    "deep_stage",
    "estimate",
    "max_tiles_for",
    "tile_side",
]
