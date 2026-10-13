"""Fit and check the plate-position band (PT4 lever, TextPathThresholds.band) on the v1 sets ONLY.

    python3 bench/fit_plate_band.py --records <measure/work of the pilot> --out bench/results/plate-band-fit-v1.json

Input: the pilot's records of the v1 sets (the OpenALPR EU plates at 100, 50 and 30 % and the readable street plates: vehicles, plates and
the text boxes of the engine's run). The v2 dev records are never read: they are B's held-out measure. Three steps:
1. each plate's box as fractions of the box of the narrow vehicle (< 250 px) it lies on;
2. the band edges: the p-th and (100 - p)-th percentile of the plate edges (left, right, top, bottom) for p = 0..10, with the share of the
   plates the band then hides (>= 95 % of the plate inside the band) and its area share of the vehicle box: the table that justifies p = 2;
3. the suppression rule: a band is not added when a text box has at least `inside` of its area inside it; the plates hidden on the EU sets
   by the text boxes of the engine run plus the bands, for several values of `inside`, and the share of small vehicles that get a band.
The result is a table in a JSON file; nothing here sets a constant by itself."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

MARGIN = (
    0.10,
    0.18,
)  # the photo margin of the plate module: a plate counts as hidden at 95 % coverage of the blurred boxes
MIN_VEHICLE = (60, 40)
EU_FILE, STREET_FILE = "eng-eu-rfn-v12-fix2-laptop-a.json", "tp-v1st-rfn-1280loose-vall.json"
PILOT_EU_FILE = "tp-v1eu-rfn-1280loose.json"
Box = Sequence[float]


def distinct_vehicles(record: dict) -> List[Box]:
    """The vehicles of a record without the near-duplicates (the raw list holds the same box at several scores), as large as the plate stage searches."""
    kept: List[Box] = []
    for v in sorted(record["vehicles"], key=lambda v: -v[4]):
        if not any(sum(abs(a - b) for a, b in zip(v[:4], k[:4], strict=True)) < 0.1 * (k[2] - k[0]) for k in kept):
            kept.append(v)
    return [v for v in kept if v[2] - v[0] >= MIN_VEHICLE[0] and v[3] - v[1] >= MIN_VEHICLE[1]]


def band_box(v: Box, edges: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    w, h = v[2] - v[0], v[3] - v[1]
    return (v[0] + edges[0] * w, v[1] + edges[2] * h, v[0] + edges[1] * w, v[1] + edges[3] * h)


def coverage(plate: Box, boxes: Sequence[Box], margin: Tuple[float, float] = (0.0, 0.0)) -> float:
    """The share of the plate's area lying inside the union of the boxes, each grown by `margin` per side."""
    px0, py0, px1, py1 = (int(round(v)) for v in plate[:4])
    px1, py1 = max(px1, px0 + 1), max(py1, py0 + 1)
    mask = np.zeros((py1 - py0, px1 - px0), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mx, my = (x1 - x0) * margin[0], (y1 - y0) * margin[1]
        ax0, ay0, ax1, ay1 = int(x0 - mx), int(y0 - my), int(x1 + mx), int(y1 + my)
        if ax1 <= px0 or ax0 >= px1 or ay1 <= py0 or ay0 >= py1:
            continue
        mask[max(0, ay0 - py0) : max(0, ay1 - py0), max(0, ax0 - px0) : max(0, ax1 - px0)] = 1
    return float(mask.mean())


def inside_share(box: Box, region: Box) -> float:
    iw, ih = min(box[2], region[2]) - max(box[0], region[0]), min(box[3], region[3]) - max(box[1], region[1])
    area = (box[2] - box[0]) * (box[3] - box[1])
    return 0.0 if iw <= 0 or ih <= 0 or area <= 0 else iw * ih / area


def plates_on_small(
    records: Sequence[dict], max_width: float, readable_only: bool
) -> List[Tuple[float, float, float, float]]:
    rows = []
    for r in records:
        vehicles = distinct_vehicles(r)
        for p in r["plates"]:
            if readable_only and not p.get("readable", True):
                continue
            x0, y0, x1, y1 = p["box"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            on = [
                v
                for v in vehicles
                if v[0] - 0.15 * (v[2] - v[0]) <= cx <= v[2] + 0.15 * (v[2] - v[0])
                and v[1] - 0.15 * (v[3] - v[1]) <= cy <= v[3] + 0.15 * (v[3] - v[1])
            ]
            if not on:
                continue
            v = min(on, key=lambda v: (v[2] - v[0]) * (v[3] - v[1]))
            w, h = v[2] - v[0], v[3] - v[1]
            if w < max_width:
                rows.append(((x0 - v[0]) / w, (x1 - v[0]) / w, (y0 - v[1]) / h, (y1 - v[1]) / h))
    return rows


def hidden_by_band(rel: np.ndarray, edges: Sequence[float]) -> int:
    u0, u1, t0, t1 = edges
    iw = np.clip(np.minimum(rel[:, 1], u1) - np.maximum(rel[:, 0], u0), 0, None)
    ih = np.clip(np.minimum(rel[:, 3], t1) - np.maximum(rel[:, 2], t0), 0, None)
    return int((iw * ih / ((rel[:, 1] - rel[:, 0]) * (rel[:, 3] - rel[:, 2])) >= 0.95).sum())


def percentile_table(rel: np.ndarray) -> List[dict]:
    table = []
    for p in (0, 1, 2, 3, 5, 8, 10):
        edges = (
            float(np.percentile(rel[:, 0], p)),
            float(np.percentile(rel[:, 1], 100 - p)),
            float(np.percentile(rel[:, 2], p)),
            float(np.percentile(rel[:, 3], 100 - p)),
        )
        table.append(
            {
                "p": p,
                "edges": [round(e, 3) for e in edges],
                "plates_hidden_by_the_band_alone": hidden_by_band(rel, edges),
                "of": len(rel),
                "area_share": round((edges[1] - edges[0]) * (edges[3] - edges[2]), 3),
            }
        )
    return table


def rule_table(records: Sequence[dict], edges, max_width: float, inside_values: Sequence[float]) -> List[dict]:
    """Plates hidden (>= 95 % covered, on a vehicle) by the engine's text boxes plus the bands that the rule keeps."""
    out = []
    for inside in inside_values:
        plates = text_only = both = small = banded = 0
        for r in records:
            text = [t[:4] for t in r["text"]]
            bands = []
            for v in distinct_vehicles(r):
                if v[2] - v[0] >= max_width:
                    continue
                small += 1
                b = band_box(v, edges)
                if inside is None or not any(inside_share(t, b) >= inside for t in text):
                    bands.append(b)
                    banded += 1
            for p in r["plates"]:
                plates += 1
                by_text = p["on_vehicle"] and p["cover"] >= 0.95
                text_only += by_text
                both += by_text or (p["on_vehicle"] and combined(p["box"], text, bands) >= 0.95)
        out.append(
            {
                "inside": inside,
                "plates": plates,
                "text_only": int(text_only),
                "text_and_bands": int(both),
                "small_vehicles": small,
                "banded": banded,
            }
        )
    return out


def combined(plate: Box, text: Sequence[Box], bands: Sequence[Box]) -> float:
    """Coverage by the text boxes (with the photo margin) and the bands (the engine shrinks a band so that its margin brings it back to the band: no margin here)."""
    px0, py0, px1, py1 = (int(round(v)) for v in plate[:4])
    px1, py1 = max(px1, px0 + 1), max(py1, py0 + 1)
    mask = np.zeros((py1 - py0, px1 - px0), np.uint8)
    for boxes, margin in ((text, MARGIN), (bands, (0.0, 0.0))):
        for x0, y0, x1, y1 in boxes:
            mx, my = (x1 - x0) * margin[0], (y1 - y0) * margin[1]
            ax0, ay0, ax1, ay1 = int(x0 - mx), int(y0 - my), int(x1 + mx), int(y1 + my)
            if ax1 <= px0 or ax0 >= px1 or ay1 <= py0 or ay0 >= py1:
                continue
            mask[max(0, ay0 - py0) : max(0, ay1 - py0), max(0, ax0 - px0) : max(0, ax1 - px0)] = 1
    return float(mask.mean())


def load_records(folder: str, name: str, scale: float | None = None) -> List[dict]:
    with open(os.path.join(folder, name), encoding="utf-8") as f:
        records = json.load(f)["records"]
    return [r for r in records if scale is None or abs(r["scale"] - scale) < 1e-6]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", required=True, help="the pilot's work folder (records of the v1 sets)")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--edges", type=float, nargs=4, default=(0.13, 0.95, 0.43, 1.04), help="the band checked in step 3"
    )
    parser.add_argument("--max-width", type=float, default=250.0)
    args = parser.parse_args()
    eu = load_records(args.records, PILOT_EU_FILE)
    street = load_records(args.records, STREET_FILE)
    rel_eu = plates_on_small(eu, args.max_width, False)
    rel_street = plates_on_small(street, args.max_width, True)
    rel = np.array(rel_eu + rel_street)
    result: Dict[str, object] = {
        "max_width": args.max_width,
        "plates_fitted": {"eu_all_scales": len(rel_eu), "street_readable": len(rel_street)},
        "percentiles": percentile_table(rel),
        "edges_checked": list(args.edges),
        "rule": {},
    }
    for label, scale in (("EU 100 %", 1.0), ("EU 50 %", 0.5), ("EU 30 %", 0.3)):
        records = load_records(args.records, EU_FILE, scale)
        result["rule"][label] = rule_table(records, args.edges, args.max_width, (None, 0.6, 0.7, 0.8, 0.9, 1.0))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["percentiles"], indent=1))
    for label, rows in result["rule"].items():
        print(label)
        for row in rows:
            print("  ", row)


if __name__ == "__main__":
    main()
