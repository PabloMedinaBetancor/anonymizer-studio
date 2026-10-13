"""What every results file of the benchmarks says about itself: the truth it was measured against (sha256), how many cases stand behind
a figure (`n`), the 95 % interval, and whether the figure may be quoted (`_publication`).

A results file may hold a top-level `_publication` map {metric name or prefix: {"publishable": false, "why": "..."}}. The `numbers` gate
of the documents refuses to fill a marker whose metric is, or lies under, an entry that is not publishable.
Intervals are Wilson's, in percent; the cases of one photo are not independent, so for faces in photos the interval is a floor of the
real uncertainty, and the `caveat` of the file says so.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth_hashes(relative_paths: Sequence[str]) -> Dict[str, str]:
    """{path relative to the repository: sha256} of the truth files a result was measured against."""
    return {rel: sha256_file(os.path.join(ROOT, rel)) for rel in relative_paths}


def wilson(hits: int, n: int, z: float = 1.96) -> list:
    """95 % Wilson interval of a share, in percent; [None, None] with no cases."""
    if n == 0:
        return [None, None]
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def interval(hits: int, n: int) -> Dict[str, object]:
    """{"n": cases, "hits": hits, "ci95_pct": [low, high]}: what stands behind a percentage."""
    return {"n": n, "hits": hits, "ci95_pct": wilson(hits, n)}


def rate_per_100(events: int, photos_affected: int, n: int) -> Dict[str, object]:
    """What stands behind a "per 100 photos" rate: the events counted, the photos they fell on, the photos looked at, the 95 % Wilson
    interval of the share of photos affected and, when no photo was, the rule of three (the share cannot be shown to be below 300/n
    per 100 photos with these n)."""
    out: Dict[str, object] = {
        "per_100": round(100.0 * events / n, 2) if n else None,
        "events": events,
        "photos_affected": photos_affected,
        "n": n,
        "ci95_photos_pct": wilson(photos_affected, n),
    }
    if n and photos_affected == 0:
        out["rule_of_three_upper_pct"] = round(300.0 / n, 1)
    return out


def not_publishable(why: str) -> Dict[str, object]:
    return {"publishable": False, "why": why}


def flat_interval(results: dict, base: str, hits: int, n: int) -> None:
    """For the files whose keys are flat dotted names: base_n and base_ci95_pct next to the figure `base`_pct."""
    results[f"{base}_n"] = n
    results[f"{base}_ci95_pct"] = wilson(hits, n)


def optional(value: Optional[str]) -> str:
    return value or ""
