"""Box geometry shared by the engine and the class modules.

Same behaviour as the earlier application; parity is tested in tests/test_contract.py.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .contract import Detection, Rect


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(dets: List[Detection], threshold: float = 0.4, contained: Optional[float] = 0.8) -> List[Detection]:
    """Drop repeats (the same face seen in two overlapping tiles): keep the higher score. A box with `contained` of its area
    inside a bigger one of the same kind also counts as a repeat (made for faces); `contained=None` merges by IoU alone, which
    vehicles and persons need: a car inside a big box that merged several cars is a car of its own."""
    kept: List[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        repeat = False
        for f in kept:
            if f.kind != d.kind:
                continue
            if iou(f.box(), d.box()) > threshold:
                repeat = True
                break
            if contained is None:
                continue
            iw = min(f.x1, d.x1) - max(f.x0, d.x0)
            ih = min(f.y1, d.y1) - max(f.y0, d.y0)
            if iw > 0 and ih > 0 and iw * ih > contained * min(d.w * d.h, f.w * f.h):
                repeat = True
                break
        if not repeat:
            kept.append(d)
    return kept


def tiles(width: int, height: int, side: int, overlap: float = 0.2) -> List[Rect]:
    """Squares of `side` covering width x height with at least `overlap` (the last one sits on the edge)."""
    if width <= side and height <= side:
        return [(0, 0, width, height)]

    def cuts(n: int) -> List[int]:
        if n <= side:
            return [0]
        step = side * (1 - overlap)
        k = int(math.ceil((n - side) / step)) + 1
        return [int(round(i * (n - side) / (k - 1))) for i in range(k)]

    return [(x, y, min(width, x + side), min(height, y + side)) for y in cuts(height) for x in cuts(width)]


def uses_tiles(width: int, height: int, tile_side: int) -> bool:
    """Tiles are only used when the image is clearly bigger than one tile (as in the earlier application: max(W, H) > tile side * 1.15)."""
    return bool(tile_side) and max(width, height) > tile_side * 1.15


def tile_count(width: int, height: int, tile_side: int) -> int:
    """How many tiles a sweep of this image has (the earlier application counted them the same way)."""
    return len(tiles(width, height, tile_side)) if uses_tiles(width, height, tile_side) else 0


def sweep_tiles(width: int, height: int, tile_side: int, which: Sequence[int] | None) -> List[Rect]:
    """The tiles one sweep looks at: all if `which` is None, else tiles[i % n] for i in `which` (as the earlier application swept)."""
    if not uses_tiles(width, height, tile_side):
        return []
    grid = tiles(width, height, tile_side)
    idx = range(len(grid)) if which is None else which
    return [grid[i % len(grid)] for i in idx]


def region_around(box: Sequence[float], width: int, height: int, factor: float, minimum: int = 48) -> Rect:
    """Square region centred on the box, `factor` times its longest side (second attempts, plates in vehicles)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(minimum, max(box[2] - box[0], box[3] - box[1]) * factor)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return x0, y0, int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
