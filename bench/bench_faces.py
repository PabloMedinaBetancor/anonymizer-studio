"""Face benchmark on public photos (criteria F5 and F7, data F9).

Runs the photo rule of the earlier application (its photo decision: sweep, VERIFY every candidate, judge, union of the two
boxes, photo margin) with FaceModule, scores it against the labelled photos in bench/truth/faces/ and writes
bench/results/faces.json (the numbers the documentation quotes) and bench/results/faces-details.json (every
miss and every false blur, with a crop saved under $ANONYMIZER_BENCH_DATA/faces/crops/).

    python3 -s bench/bench_faces.py [--limit N] [--no-crops]

decide_photo() is the loop that engine.photos.decide_photo will replace once the engine is integrated; the
scoring around it stays.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from anonymizer.classes.faces import ASP_THRESHOLDS, FaceModule
from anonymizer.contract import Box, RetryPurpose, SweepPlan, Verdict
from bench.results_meta import interval, rate_per_100, truth_hashes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
TRUTH_DIR = os.path.join(HERE, "truth", "faces")
RESULTS_DIR = os.path.join(HERE, "results")
SETS = {"a": "photos with faces", "b": "photos without people"}

PHOTO_PLAN = SweepPlan(general_side=1600, tile_side=640)  # anon_fotos.FOTO
COVERED = 0.95  # share of a face a blur must cover to count
SIZE_CLASSES = (("lt24", 0, 24), ("24_48", 24, 48), ("gt48", 48, 10**6))


@dataclass
class Zone:
    box: Box  # union of the sweep box and the VERIFY box, before the margin
    margin_box: Box  # what is blurred (or listed in Review) once the photo margin is added
    verdict: Verdict
    reason: str
    score: float
    verify: float


def with_margin(box: Box, margin: Tuple[float, float], width: int, height: int) -> Box:
    mx, my = margin
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (
        max(0.0, x0 - w * mx),
        max(0.0, y0 - h * my),
        min(float(width), x1 + w * mx),
        min(float(height), y1 + h * my),
    )


def decide_photo(module: FaceModule, bgr: np.ndarray) -> List[Zone]:
    height, width = bgr.shape[:2]
    zones = []
    for det in module.sweep(bgr, PHOTO_PLAN):
        second = module.retry(bgr, det.box(), RetryPurpose.VERIFY)
        judgement = module.photo_judgement(det, second)
        if judgement.verdict is Verdict.DROP:
            continue
        box = det.box()
        if second is not None:
            box = (min(box[0], second.x0), min(box[1], second.y0), max(box[2], second.x1), max(box[3], second.y1))
        zones.append(
            Zone(
                box,
                with_margin(box, module.photo_margin, width, height),
                judgement.verdict,
                judgement.reason,
                det.score,
                judgement.params["verify"],
            )
        )
    return zones


def coverage(face: Box, zones: Sequence[Zone], ellipse: bool) -> float:
    """Share of the face box (in pixels) inside the union of the zones, as rectangles or as the ellipses that
    are really drawn (an ellipse inscribed in the margin box, like engine blur does for faces)."""
    x0, y0 = int(round(face[0])), int(round(face[1]))
    x1, y1 = max(x0 + 1, int(round(face[2]))), max(y0 + 1, int(round(face[3])))
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for z in zones:
        zx0, zy0, zx1, zy1 = z.margin_box
        if zx1 <= x0 or zx0 >= x1 or zy1 <= y0 or zy0 >= y1:
            continue
        if ellipse:
            center = (int(round((zx0 + zx1) / 2)) - x0, int(round((zy0 + zy1) / 2)) - y0)
            axes = (max(1, int(round((zx1 - zx0) / 2))), max(1, int(round((zy1 - zy0) / 2))))
            cv2.ellipse(canvas, center, axes, 0, 0, 360, 1, -1, cv2.LINE_8)
        else:
            canvas[
                max(0, int(round(zy0)) - y0) : max(0, int(round(zy1)) - y0),
                max(0, int(round(zx0)) - x0) : max(0, int(round(zx1)) - x0),
            ] = 1
    return float(canvas.mean())


def touches(zone: Zone, box: Sequence[float]) -> bool:
    z = zone.margin_box
    return min(z[2], box[2]) > max(z[0], box[0]) and min(z[3], box[3]) > max(z[1], box[1])


def load_truth(tag: str) -> dict:
    with open(os.path.join(TRUTH_DIR, f"set-{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def pct(part: int, whole: int) -> Optional[float]:
    return round(100.0 * part / whole, 2) if whole else None


def hardware() -> dict:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass  # not Linux, or no model name line: the report just says "unknown"
    ram = None
    try:
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass  # same: RAM is informative only
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "ram_gb": ram,
        "os": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
    }


def save_crop(name: str, bgr: np.ndarray, focus: Box, marks: Sequence[Tuple[Box, Tuple[int, int, int]]]) -> str:
    """A crop around `focus` with three times its size of context, marks drawn, enlarged to at least 300 px."""
    height, width = bgr.shape[:2]
    cx, cy = (focus[0] + focus[2]) / 2, (focus[1] + focus[3]) / 2
    half = max(40.0, 1.5 * max(focus[2] - focus[0], focus[3] - focus[1]))
    x0, y0, x1, y1 = (
        max(0, int(cx - half)),
        max(0, int(cy - half)),
        min(width, int(cx + half)),
        min(height, int(cy + half)),
    )
    scale = max(1.0, 300.0 / max(x1 - x0, y1 - y0))
    crop = cv2.resize(bgr[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for box, color in marks:
        p0 = (int((box[0] - x0) * scale), int((box[1] - y0) * scale))
        p1 = (int((box[2] - x0) * scale), int((box[3] - y0) * scale))
        cv2.rectangle(crop, p0, p1, color, 1)
    folder = os.path.join(DATA, "faces", "crops")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return os.path.relpath(path, DATA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=0, help="only the first N photos of each set (smoke test)")
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument(
        "--net-score",
        type=float,
        help="experiment: build YuNet with this cut instead of the earlier application's 0.5; write the results elsewhere with --out",
    )
    args = parser.parse_args()

    thresholds = ASP_THRESHOLDS if args.net_score is None else replace(ASP_THRESHOLDS, net_score=args.net_score)
    module = FaceModule(thresholds=thresholds)
    module.sweep(np.zeros((128, 128, 3), np.uint8), PHOTO_PLAN)  # load everything once, outside the timing
    module.timings.clear()
    module.calls.clear()

    sizes = {
        name: {"faces": 0, "blur": 0, "blur_or_check": 0, "blur_ellipse": 0, "hidden_ellipse": 0}
        for name, _, _ in SIZE_CLASSES
    }
    totals = {
        "faces": 0,
        "blur": 0,
        "blur_or_check": 0,
        "blur_ellipse": 0,
        "hidden_ellipse": 0,
        "ignored": 0,
        "ignored_blurred": 0,
    }
    photos = {"a": 0, "b": 0}
    false_blurs = {"a": 0, "b": 0}
    photos_false = {"a": 0, "b": 0}
    false_hidden = {"a": 0, "b": 0}  # BLUR and CHECK zones that touch no truth box (privacy first: both are blurred)
    photos_false_hidden = {"a": 0, "b": 0}
    misses, falses, per_photo_ms = [], [], []

    for tag in SETS:
        truth = load_truth(tag)
        names = sorted(truth["images"])[: args.limit or None]
        for name in names:
            entry = truth["images"][name]
            bgr = cv2.imread(os.path.join(DATA, "faces", f"set-{tag}", name))
            if bgr is None:
                raise SystemExit(f"cannot read {name}")
            if bgr.shape[:2] != (entry["height"], entry["width"]):
                raise SystemExit(
                    f"{name}: truth says {entry['width']}x{entry['height']}, file is {bgr.shape[1]}x{bgr.shape[0]}"
                )
            started = time.perf_counter()
            zones = decide_photo(module, bgr)
            per_photo_ms.append(1000 * (time.perf_counter() - started))
            photos[tag] += 1
            blur = [z for z in zones if z.verdict is Verdict.BLUR]
            for k, item in enumerate(entry["boxes"]):
                face = item["box"]
                if item.get("ignore"):
                    totals["ignored"] += 1
                    totals["ignored_blurred"] += any(touches(z, face) for z in blur)
                    continue
                side = max(face[2] - face[0], face[3] - face[1])
                klass = next(n for n, lo, hi in SIZE_CLASSES if lo <= side < hi)
                got_blur = coverage(face, blur, False) >= COVERED
                got_any = got_blur or coverage(face, zones, False) >= COVERED
                got_ellipse = coverage(face, blur, True) >= COVERED
                got_hidden_ellipse = got_ellipse or coverage(face, zones, True) >= COVERED
                for bucket in (sizes[klass], totals):
                    bucket["faces"] += 1
                    bucket["blur"] += got_blur
                    bucket["blur_or_check"] += got_any
                    bucket["blur_ellipse"] += got_ellipse
                    bucket["hidden_ellipse"] += got_hidden_ellipse
                if not got_blur:
                    entry_out = {
                        "image": f"set-{tag}/{name}",
                        "box": face,
                        "side_px": round(side, 1),
                        "outcome": "check" if got_any else "missed",
                        "note": item.get("note", ""),
                    }
                    if not args.no_crops:
                        entry_out["crop"] = save_crop(
                            f"miss-{tag}-{os.path.splitext(name)[0][:40]}-{k}.jpg",
                            bgr,
                            tuple(face),
                            [(tuple(face), (0, 200, 0))] + [(z.margin_box, (0, 0, 255)) for z in zones],
                        )
                    misses.append(entry_out)
            falses_here = 0
            for k, z in enumerate(blur):
                if not any(touches(z, item["box"]) for item in entry["boxes"]):
                    false_blurs[tag] += 1
                    falses_here += 1
                    entry_out = {
                        "image": f"set-{tag}/{name}",
                        "zone": [round(v, 1) for v in z.margin_box],
                        "score": round(z.score, 3),
                        "verify": round(z.verify, 3),
                        "reason": z.reason,
                    }
                    if not args.no_crops:
                        entry_out["crop"] = save_crop(
                            f"false-{tag}-{os.path.splitext(name)[0][:40]}-{k}.jpg",
                            bgr,
                            z.margin_box,
                            [(z.margin_box, (0, 0, 255))],
                        )
                    falses.append(entry_out)
            photos_false[tag] += falses_here > 0
            hidden_here = sum(1 for z in zones if not any(touches(z, item["box"]) for item in entry["boxes"]))
            false_hidden[tag] += hidden_here
            photos_false_hidden[tag] += hidden_here > 0
        print(f"set {tag}: {photos[tag]} photos", flush=True)

    def rate(bucket: dict, key: str) -> Optional[float]:
        return pct(bucket[key], bucket["faces"])

    results = {
        "faces": {
            "photo": {
                "recall_blur_pct": rate(totals, "blur"),
                "recall_blur": interval(totals["blur"], totals["faces"]),
                "recall_blur_or_check_pct": rate(totals, "blur_or_check"),
                "recall_blur_or_check": interval(totals["blur_or_check"], totals["faces"]),
                "recall_blur_ellipse_pct": rate(totals, "blur_ellipse"),
                "recall_blur_ellipse": interval(totals["blur_ellipse"], totals["faces"]),
                "false_blur_photos_without_people": photos["b"],
                "false_blur_per_100": round(100.0 * false_blurs["b"] / photos["b"], 2) if photos["b"] else None,
                "false_blur": rate_per_100(false_blurs["b"], photos_false["b"], photos["b"]),
                "false_blur_in_set_a": rate_per_100(false_blurs["a"], photos_false["a"], photos["a"]),
                "false_blur_per_100_in_set_a": round(100.0 * false_blurs["a"] / photos["a"], 2)
                if photos["a"]
                else None,
                # Privacy first: what is blurred by default is a BLUR or a CHECK zone. Same measurement as
                # recall_blur_or_check; the ellipse figure counts the faces against the ellipse that is really drawn.
                "hidden_default_pct": rate(totals, "blur_or_check"),
                "hidden_default": interval(totals["blur_or_check"], totals["faces"]),
                "hidden_default_ellipse_pct": rate(totals, "hidden_ellipse"),
                "hidden_default_ellipse": interval(totals["hidden_ellipse"], totals["faces"]),
                "false_hidden_default": rate_per_100(false_hidden["b"], photos_false_hidden["b"], photos["b"]),
                "false_hidden_default_in_set_a": rate_per_100(false_hidden["a"], photos_false_hidden["a"], photos["a"]),
                "faces_total": totals["faces"],
                "ignored_total": totals["ignored"],
                "ignored_blurred": totals["ignored_blurred"],
                "by_size": {
                    name: {
                        "faces": s["faces"],
                        "recall_blur_pct": rate(s, "blur"),
                        "recall_blur_or_check_pct": rate(s, "blur_or_check"),
                    }
                    for name, s in sizes.items()
                },
            },
            "speed": {
                "ms_per_photo": round(statistics.mean(per_photo_ms), 1) if per_photo_ms else None,
                "ms_per_photo_median": round(statistics.median(per_photo_ms), 1) if per_photo_ms else None,
                "photos": len(per_photo_ms),
                "seconds_by_step": {k: round(v, 2) for k, v in module.timings.items()},
                "calls_by_step": module.calls,
            },
        },
        "data": {
            "photos_with_faces": photos["a"],
            "photos_without_people": photos["b"],
            "size_classes_px": {n: [lo, hi if hi < 10**6 else None] for n, lo, hi in SIZE_CLASSES},
            "covered_share": COVERED,
        },
        "caveat": (
            "the truth starts from YuNet proposals at a low threshold, so faces it cannot find at all are missing from the "
            "count (an optimistic bound); the faces of one photo are not independent, so the 95 % interval is narrower than "
            "the real uncertainty; checked blind on 15 photos (bench/results/truth-check-faces.json)"
        ),
        "truth_sha256": truth_hashes([f"bench/truth/faces/set-{tag}.json" for tag in SETS]),
        "hardware": hardware(),
        "run": {
            "date": datetime.date.today().isoformat(),
            "plan": {"general_side": PHOTO_PLAN.general_side, "tile_side": PHOTO_PLAN.tile_side},
        },
    }
    os.makedirs(args.out, exist_ok=True)
    suffix = "-smoke" if args.limit else ""
    with open(os.path.join(args.out, f"faces{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    with open(os.path.join(args.out, f"faces-details{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"misses": misses, "false_blurs": falses}, f, indent=2)
        f.write("\n")
    print(json.dumps(results["faces"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
