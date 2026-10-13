"""What the defaults hide, and how much of the rest of the picture they touch (2.2, "subtle").

The 2.2 defaults hide only the plate itself and the face itself, tightly and softly. Version 2.1's defaults
blurred every text region on a vehicle, the plate-position band and the doubtful faces. This benchmark runs the product on the DEV part
of the labelled photos (never the sealed one) under two option sets and measures, per set of photos:

- recall: the plates (readable ones apart) and the faces the blur really hides (>= 95 % of a plate, >= 80 % of a face under the blur zone with
  its margin, as the earlier benchmarks count);
- false regions per 100 photos: blur zones that touch no labelled object at all (labelled "ignore" objects count as objects: touching one is
  neither a hit nor a false region);
- the blurred area: pixels the render really changes, in total and outside the labelled objects ("collateral"), as a share of the picture.

    python3 -s -m bench.bench_subtle [--sets plates,faces] [--configs v2.1,v2.2] [--limit N] [--out FILE] [--keep-zones FILE]

The options sets: "v2.1" is `Options(unsure="blur", fit="loose", plate_band=True)` (the defaults of 2.1.10) and "v2.2" is `Options()`. The
photos are the ones of the plate truth (`bench/truth/plates/plates-v2.json`) and of the face truths A and B that the split fixes as dev.
Needs `$ANONYMIZER_BENCH_DATA`; heavy (models): book the machine first.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Dict, List, Sequence

import cv2
import numpy as np

from . import bench_cells as bc
from .results_meta import interval

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
SPLIT = os.path.join(HERE, "truth", "split-v2.json")
PLATES_TRUTH = os.path.join(HERE, "truth", "plates", "plates-v2.json")
PLATE_COVER, FACE_COVER = 0.95, 0.80
ALTERED = 2  # a pixel counts as touched when a channel changes by more than this


def configs() -> Dict[str, dict]:
    """Option sets by name. "v2.2r1" is the first round of 2.2 (the default `Options`, with the margins 0.06 / 0.10, no plate reader, no union with
    the weak text regions and no agreement rule for the doubtful faces), made here by `round_one` on the modules of the current code, so that the three rows of the table
    come from one code and one runner."""
    return {
        "v2.1": dict(unsure="blur", fit="loose", plate_band=True),
        "v2.2r1": dict(round1=True),
        "v2.2": dict(),
    }


def round_one(anon) -> None:
    """Put the modules of `anon` back to what 2.2 was after its first round (commits c433dda and 0bb49b9): the plate margin 0.06 / 0.10, no reader,
    and no agreement of the second look or the sweep for a doubtful face."""
    from dataclasses import replace

    for module in anon.modules():
        if module.kind == "plate":
            module.reader = None
            module.photo_margin = (0.06, 0.10)
            module.text_path = replace(
                module.text_path, merge_floor=module.text_path.tau_lo
            )  # no union with the weak regions
        else:
            module.th = replace(module.th, placed_second=9.0, placed_score=9.0)


def covered_share(truth: Sequence[float], boxes: Sequence[Sequence[float]]) -> float:
    x0, y0 = int(round(truth[0])), int(round(truth[1]))
    x1, y1 = max(x0 + 1, int(round(truth[2]))), max(y0 + 1, int(round(truth[3])))
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for bx0, by0, bx1, by1 in boxes:
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        mask[max(0, int(by0) - y0) : max(0, int(by1) - y0 + 1), max(0, int(bx0) - x0) : max(0, int(bx1) - x0 + 1)] = 1
    return float(mask.mean())


def touches(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def dev_keys(prefixes: Sequence[str]) -> List[str]:
    table, _ = bc.read_split(SPLIT)
    return sorted(k for k, v in table.items() if v == "dev" and k.startswith(tuple(prefixes)))


def photo_set(name: str):
    """(key, path, truth boxes [{box, ignore, readable?}], options extras) for the dev photos of "plates" or "faces"."""
    if (
        name in ("streets", "eu")
    ):  # held-out checks: outside the split, never used to choose a number (streets: false regions are measured; EU: recall only)
        folder = "streets" if name == "streets" else "openalpr-eu"
        truth = json.load(open(os.path.join(HERE, "truth", "plates", f"{folder}.json"), encoding="utf-8"))["images"]
        for n in sorted(truth):
            if os.path.isfile(os.path.join(DATA, "plates", folder, n)):
                yield f"{name}/{n}", os.path.join(DATA, "plates", folder, n), truth[n]["boxes"], dict(faces=False)
    elif name == "plates":
        truth = json.load(open(PLATES_TRUTH, encoding="utf-8"))["images"]
        for key in dev_keys(("plates/",)):
            n = key.split("/", 1)[1]
            yield key, os.path.join(DATA, "plates", "v2", n), truth[n]["boxes"], dict(faces=False)
    else:
        tr = {}
        for tag in bc.SETS:
            for n, v in bc.load_truth(tag)["images"].items():
                tr[f"set-{tag}/{n}"] = v
        for key in dev_keys(("set-a/", "set-b/")):
            yield key, bc.image_path(key), tr[key]["boxes"], dict(plates=False)


def altered_pixels(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return (
        (np.abs(before.astype(np.int16) - after.astype(np.int16)).max(axis=2) > ALTERED)
        if before.ndim == 3
        else (np.abs(before.astype(np.int16) - after.astype(np.int16)) > ALTERED)
    )


def measure(set_name: str, config: dict, limit, keep: dict) -> dict:
    import anonymizer
    from anonymizer import api
    from anonymizer.engine.photos import blur_zones, with_margin

    rows = []
    is_plate = set_name in ("plates", "streets", "eu")
    cover = PLATE_COVER if is_plate else FACE_COVER
    kind = "plate" if is_plate else "face"
    items = list(photo_set(set_name))[:limit]
    base = {k: v for k, v in config.items() if k != "round1"}
    extra = items[0][3] if items else {}
    options = anonymizer.Options(**{**base, **extra})
    anon = anonymizer.Anonymizer(options)
    if config.get("round1"):
        round_one(anon)
    for key, path, truth, _ in items:
        if True:
            t0 = time.perf_counter()
            analysis = anon.analyze(path)
            seconds = time.perf_counter() - t0
            zones, specs = api._photo_zones(analysis, options)
            height, width = analysis.height, analysis.width
            blurred = [z for z in zones if z.kind == kind and z.blur]
            boxes = [with_margin(z.box, specs[z.kind].margin, width, height) for z in blurred]
            img = cv2.imread(path)
            out = img.copy()
            blur_zones(out, blurred, specs, options.blur, options.strength, options.profile)
            changed = altered_pixels(img, out)
        objects = [t for t in truth]
        real = [t for t in truth if not t.get("ignore")]
        mask = np.zeros(changed.shape, bool)
        for t in objects:
            b = t["box"]
            mask[max(0, int(b[1])) : int(b[3]) + 1, max(0, int(b[0])) : int(b[2]) + 1] = True
        hit = [covered_share(t["box"], boxes) >= cover for t in real]
        rows.append(
            {
                "photo": key,
                "seconds": round(seconds, 2),
                "truth": len(real),
                "hit": int(sum(hit)),
                "readable": sum(1 for t in real if t.get("readable", True)),
                "readable_hit": int(sum(h for h, t in zip(hit, real, strict=True) if t.get("readable", True))),
                "zones": len(blurred),
                "false_regions": sum(1 for b in boxes if not any(touches(b, t["box"]) for t in objects)),
                "listed_visible": sum(
                    1 for z in zones if z.kind == kind and not z.blur and z.judgement.verdict.value == "check"
                ),
                "altered_share": round(float(changed.mean()), 6),
                "collateral_share": round(float((changed & ~mask).mean()), 6),
            }
        )
        if key not in keep:
            keep[key] = {}
        keep[key][json.dumps({**base, **extra}, sort_keys=True)] = [
            {
                "kind": z.kind,
                "box": [round(v, 1) for v in z.box],
                "blur": z.blur,
                "verdict": z.judgement.verdict.value,
                "score": round(z.score, 3),
            }
            for z in zones
        ]
        print(f"{set_name} {key} {len(blurred)} zones {rows[-1]['hit']}/{rows[-1]['truth']} {seconds:.1f}s", flush=True)
    anon.close()
    n = len(rows)
    truth_n, hit_n = sum(r["truth"] for r in rows), sum(r["hit"] for r in rows)
    out = {
        "photos": n,
        "recall": interval(hit_n, truth_n),
        "recall_pct": round(100.0 * hit_n / truth_n, 1) if truth_n else None,
        "zones_per_photo": round(sum(r["zones"] for r in rows) / max(1, n), 2),
        "false_regions_per_100": round(100.0 * sum(r["false_regions"] for r in rows) / max(1, n), 1),
        "listed_visible_per_100": round(100.0 * sum(r["listed_visible"] for r in rows) / max(1, n), 1),
        "altered_pct_mean": round(100.0 * sum(r["altered_share"] for r in rows) / max(1, n), 3),
        "collateral_pct_mean": round(100.0 * sum(r["collateral_share"] for r in rows) / max(1, n), 3),
        "seconds_median": round(float(np.median([r["seconds"] for r in rows])), 2) if rows else None,
        "rows": rows,
    }
    if is_plate:
        read_n, read_hit = sum(r["readable"] for r in rows), sum(r["readable_hit"] for r in rows)
        out["readable_recall"] = interval(read_hit, read_n)
        out["readable_recall_pct"] = round(100.0 * read_hit / read_n, 1) if read_n else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--sets", default="plates,faces", help="plates, faces (the dev part) and, as held-out checks, streets and eu"
    )
    ap.add_argument("--configs", default="v2.1,v2.2")
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos of each set (smoke runs)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "subtle.json"))
    ap.add_argument("--keep-zones", default=None, help="write every zone of every photo here (research; not committed)")
    ap.add_argument(
        "--merge",
        nargs="+",
        default=None,
        help="no run: put the configs of these result files (one config per machine) into --out",
    )
    args = ap.parse_args(argv)
    if args.merge:
        merged = None
        for path in args.merge:
            with open(path, encoding="utf-8") as f:
                part = json.load(f)
            if merged is None:
                merged, merged["machines"] = part, {}
            for cfg, sets in part["configs"].items():
                merged["configs"].setdefault(cfg, {}).update(
                    sets
                )  # one config can come from several runs (its plates, its faces)
                merged["machines"].setdefault(cfg, {})
                for name in sets:
                    if name != "options":
                        merged["machines"][cfg][name] = part["machine"]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1)
            f.write("\n")
        return 0
    table = configs()
    result: dict = {
        "what": "defaults of 2.1.10 (v2.1) against 2.2 (v2.2) on the dev part of the labelled photos; sealed photos untouched",
        "machine": {"cpu": _cpu(), "python": platform.python_version()},
        "run": {"date": time.strftime("%Y-%m-%d")},
        "configs": {},
    }
    keep: dict = {}
    for cfg in args.configs.split(","):
        result["configs"][cfg] = {"options": dict(table[cfg])}
        for s in args.sets.split(","):
            result["configs"][cfg][s] = measure(s, table[cfg], args.limit, keep)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    if args.keep_zones:
        with open(args.keep_zones, "w", encoding="utf-8") as f:
            json.dump(keep, f)
    return 0


def _cpu() -> str:
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
