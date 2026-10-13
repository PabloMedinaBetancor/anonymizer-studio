"""Ground truth for the plate benchmark: loading, checking and the box tests the metrics use.

The schema is described in bench/truth/plates/README.md; `problems` is what tests/test_plates_truth.py enforces.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "truth", "plates")
LICENCES = {
    "AGPL-3.0",
    "CC0-1.0",
    "CC-BY-2.0",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-2.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "CC-BY-2.5",
    "CC-BY-SA-2.5",
    "PDM",
}


def load(name: str) -> Dict:
    with open(os.path.join(TRUTH_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def problems(doc: Dict) -> List[str]:
    """Everything wrong with a truth document, one line each; empty when it follows the schema."""
    found: List[str] = []
    for key in ("set", "licence", "images"):
        if key not in doc:
            found.append(f"missing top-level key {key!r}")
    if found:
        return found
    if doc["licence"] not in LICENCES | {"mixed"}:
        found.append(f"unknown licence {doc['licence']!r}")
    for name, image in doc["images"].items():
        width, height = image.get("width"), image.get("height")
        if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
            found.append(f"{name}: width and height must be positive integers")
            continue
        for i, entry in enumerate(image.get("boxes", [])):
            where = f"{name} box {i}"
            if set(entry) != {"box", "kind", "readable", "ignore"}:
                found.append(f"{where}: keys are {sorted(entry)}")
                continue
            if entry["kind"] != "plate":
                found.append(f"{where}: kind must be 'plate'")
            if not isinstance(entry["readable"], bool) or not isinstance(entry["ignore"], bool):
                found.append(f"{where}: readable and ignore must be booleans")
            x0, y0, x1, y1 = entry["box"]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                found.append(f"{where}: box {entry['box']} is empty or outside the {width}x{height} image")
    return found


def dump(doc: Dict, path: str) -> None:
    """Write `doc` with one line per image, so a change to one photo is one line in a diff."""
    lines = [
        f"  {json.dumps(name)}: {json.dumps(image, separators=(', ', ': '))}" for name, image in doc["images"].items()
    ]
    head = f'{{"set": {json.dumps(doc["set"])}, "licence": {json.dumps(doc["licence"])}, "images": {{\n'
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + ",\n".join(lines) + "\n}}\n")


def scaled(box: Sequence[float], factor: float) -> List[float]:
    return [v * factor for v in box]


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw * ih if iw > 0 and ih > 0 else 0.0


def area(box: Sequence[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def covered_share(truth_box: Sequence[float], zone: Sequence[float]) -> float:
    """Share of the truth box that the zone covers."""
    return intersection(truth_box, zone) / area(truth_box)


def touches(a: Sequence[float], b: Sequence[float]) -> bool:
    return intersection(a, b) > 0
