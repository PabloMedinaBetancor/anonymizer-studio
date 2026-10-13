"""Plate benchmark on photos: recall on the OpenALPR EU set at full size, 50 % and 30 %, false blurs on street photos,
speed. Writes bench/results/plates.json (the numbers the documentation quotes) and bench/results/plates-details.json
(every miss and every false blur, with a crop saved under $ANONYMIZER_BENCH_DATA/plates/crops/).

    python3 -s -m bench.bench_plates [--sets openalpr-eu,streets] [--limit N] [--out DIR]

`decide_photo` is the photo decision loop of the earlier application (sweep, VERIFY retry of each candidate, judgement, union of the two
boxes) written against PlateModule. When the engine's own photo path exists (engine.photos.decide_photo) this
function goes away and the benchmark calls that one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from anonymizer.classes.plates import PlateModule
from anonymizer.contract import Box, RetryPurpose, SweepPlan, Verdict

from . import plates_truth
from .results_meta import flat_interval, not_publishable, rate_per_100, truth_hashes

DATA = os.environ.get("ANONYMIZER_BENCH_DATA", "bench/data")
RESULTS = os.path.join(os.path.dirname(plates_truth.HERE), "bench", "results")
PHOTO_PLAN = SweepPlan(general_side=1600, tile_side=640, vehicle_tiles=True)
EU_SCALES = (1.0, 0.5, 0.3)
FULL_COVER = 0.95  # a plate counts as blurred when one blur zone covers this much of its box


@dataclass
class Zone:
    box: Box  # union of the sweep box and the VERIFY box, no margin
    blur_box: Box  # the area that would be blurred: `box` plus the photo margin
    verdict: Verdict
    reason: str
    score: float
    verify: float
    on_vehicle: Optional[bool]


def with_margin(box: Box, margin: Tuple[float, float], width: int, height: int) -> Box:
    mx, my = (box[2] - box[0]) * margin[0], (box[3] - box[1]) * margin[1]
    return (
        max(0.0, box[0] - mx),
        max(0.0, box[1] - my),
        min(float(width), box[2] + mx),
        min(float(height), box[3] + my),
    )


def decide_photo(module: PlateModule, bgr: np.ndarray) -> List[Zone]:
    height, width = bgr.shape[:2]
    zones = []
    for det in module.sweep(bgr, PHOTO_PLAN):
        verify = module.retry(bgr, det.box(), RetryPurpose.VERIFY)
        judgement = module.photo_judgement(det, verify)
        if judgement.verdict is Verdict.DROP:
            continue
        box = det.box()
        if verify is not None:
            box = (min(box[0], verify.x0), min(box[1], verify.y0), max(box[2], verify.x1), max(box[3], verify.y1))
        zones.append(
            Zone(
                box,
                with_margin(box, module.photo_margin, width, height),
                judgement.verdict,
                judgement.reason,
                det.score,
                verify.score if verify else 0.0,
                det.on_vehicle,
            )
        )
    return zones


def covers_like_asp(truth: Box, box: Box) -> bool:
    """The plate-coverage test of the earlier application: IoU >= 0.3, or the plate covered at least 70 %."""
    inter = plates_truth.intersection(truth, box)
    area_t, area_b = plates_truth.area(truth), plates_truth.area(box)
    return inter / max(1e-6, area_t + area_b - inter) >= 0.3 or inter / max(1e-6, area_t) >= 0.7


def hardware() -> Dict[str, object]:
    cpu = "unknown"
    with open("/proc/cpuinfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    with open("/proc/meminfo", encoding="utf-8") as f:
        ram_mb = int(f.readline().split()[1]) // 1024
    import onnxruntime

    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "ram_mb": ram_mb,
        "os": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
        "opencv": cv2.__version__,
    }


def save_crop(bgr: np.ndarray, box: Box, name: str, extra: Optional[Box] = None) -> str:
    """Crop around `box` with room to see the surroundings, enlarged to at least 256 px wide for viewing."""
    height, width = bgr.shape[:2]
    pad_x, pad_y = max(40, (box[2] - box[0]) * 0.6), max(40, (box[3] - box[1]) * 0.6)
    x0, y0 = int(max(0, box[0] - pad_x)), int(max(0, box[1] - pad_y))
    x1, y1 = int(min(width, box[2] + pad_x)), int(min(height, box[3] + pad_y))
    crop = bgr[y0:y1, x0:x1].copy()
    boxes = [(box, (0, 255, 0))] + ([(extra, (0, 0, 255))] if extra else [])
    for b, colour in boxes:
        cv2.rectangle(crop, (int(b[0] - x0), int(b[1] - y0)), (int(b[2] - x0), int(b[3] - y0)), colour, 1)
    if crop.shape[1] < 256:
        factor = 256 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    folder = os.path.join(DATA, "plates", "crops")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    cv2.imwrite(path, crop)
    return os.path.relpath(path, DATA)  # relative to $ANONYMIZER_BENCH_DATA: no machine paths in the results


def load_set(name: str, scale: float = 1.0, limit: Optional[int] = None):
    doc = plates_truth.load(name)
    folder = os.path.join(DATA, "plates", name)
    names = sorted(doc["images"])[:limit]
    for file in names:
        bgr = cv2.imread(os.path.join(folder, file))
        if bgr is None:
            raise FileNotFoundError(os.path.join(folder, file))
        entries = doc["images"][file]["boxes"]
        if scale != 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        yield file, bgr, [(plates_truth.scaled(e["box"], scale), e) for e in entries]


def run_eu(module: PlateModule, scale: float, limit: Optional[int]) -> Dict[str, object]:
    blurred = asp_like = seen = hidden = total = 0
    misses, millis = [], []
    for file, bgr, truth in load_set("openalpr-eu", scale, limit):
        start = time.perf_counter()
        zones = decide_photo(module, bgr)
        millis.append((time.perf_counter() - start) * 1000)
        blur = [z for z in zones if z.verdict is Verdict.BLUR]
        for box, _entry in truth:
            total += 1
            full = any(plates_truth.covered_share(box, z.blur_box) >= FULL_COVER for z in blur)
            blurred += full
            asp_like += any(covers_like_asp(box, z.blur_box) for z in blur)
            seen += any(covers_like_asp(box, z.box) for z in zones)
            hidden += any(
                plates_truth.covered_share(box, z.blur_box) >= FULL_COVER for z in zones
            )  # BLUR or CHECK: v2's default
            if not full:
                crop = save_crop(bgr, tuple(box), f"eu-{scale:g}-{file}")
                misses.append(
                    {
                        "image": file,
                        "scale": scale,
                        "box": box,
                        "crop": crop,
                        "zones": [
                            [
                                z.verdict.value,
                                z.reason,
                                [round(v, 1) for v in z.box],
                                round(z.score, 2),
                                round(z.verify, 2),
                            ]
                            for z in zones
                        ],
                    }
                )
    return {
        "total": total,
        "blurred": blurred,
        "blurred_asp_criterion": asp_like,
        "blur_or_check": seen,
        "hidden_default": hidden,
        "ms": millis,
        "misses": misses,
    }


def run_streets(module: PlateModule, limit: Optional[int]) -> Dict[str, object]:
    photos = false_blurs = plates = plates_blurred = false_checks = 0
    photos_false_blur = photos_false_check = photos_false_hidden = 0
    details = []
    for file, bgr, truth in load_set("streets", 1.0, limit):
        photos += 1
        zones = decide_photo(module, bgr)
        events_before = (false_blurs, false_checks)
        for z in zones:
            if any(plates_truth.touches(z.blur_box, box) for box, _entry in truth):
                continue
            kind = "false_blur" if z.verdict is Verdict.BLUR else "false_check"
            false_blurs += z.verdict is Verdict.BLUR
            false_checks += z.verdict is Verdict.CHECK
            crop = save_crop(bgr, z.box, f"streets-{kind}-{os.path.splitext(file)[0]}-{int(z.box[0])}.jpg")
            details.append(
                {
                    "image": file,
                    "kind": kind,
                    "box": [round(v, 1) for v in z.box],
                    "score": round(z.score, 2),
                    "verify": round(z.verify, 2),
                    "reason": z.reason,
                    "on_vehicle": z.on_vehicle,
                    "crop": crop,
                }
            )
        photos_false_blur += false_blurs > events_before[0]
        photos_false_check += false_checks > events_before[1]
        photos_false_hidden += (false_blurs, false_checks) != events_before
        blur = [z for z in zones if z.verdict is Verdict.BLUR]
        for box, entry in truth:
            if entry["ignore"] or not entry["readable"]:
                continue
            plates += 1
            if any(plates_truth.covered_share(box, z.blur_box) >= FULL_COVER for z in blur):
                plates_blurred += 1
            else:
                crop = save_crop(bgr, tuple(box), f"streets-miss-{os.path.splitext(file)[0]}-{int(box[0])}.jpg")
                details.append({"image": file, "kind": "missed_plate", "box": box, "crop": crop})
    return {
        "photos": photos,
        "false_blurs": false_blurs,
        "false_checks": false_checks,
        "photos_false_blur": photos_false_blur,
        "photos_false_check": photos_false_check,
        "photos_false_hidden": photos_false_hidden,
        "plates": plates,
        "plates_blurred": plates_blurred,
        "details": details,
    }


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", default="openalpr-eu,streets")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images of each set (smoke runs)")
    parser.add_argument("--out", default=RESULTS)
    args = parser.parse_args()
    sets = args.sets.split(",")

    module = PlateModule(threads=2)
    results: Dict[str, object] = {"hardware": hardware(), "limit": args.limit}
    details: Dict[str, object] = {}
    if "openalpr-eu" in sets:
        names = {1.0: "recall_full_pct", 0.5: "recall_half_pct", 0.3: "recall_30_pct"}
        eu_details = []
        for scale in EU_SCALES:
            out = run_eu(module, scale, args.limit)
            key = f"plates.photo.eu.{names[scale]}"
            results[key] = pct(out["blurred"], out["total"])
            results[f"{key[:-4]}_count"] = f"{out['blurred']}/{out['total']}"
            results[f"{key[:-4]}_asp_criterion_pct"] = pct(out["blurred_asp_criterion"], out["total"])
            results[f"{key[:-4]}_blur_or_check_pct"] = pct(out["blur_or_check"], out["total"])
            flat_interval(results, key[:-4], out["blurred"], out["total"])  # recall_full_n, recall_full_ci95_pct
            # Privacy first: what is hidden by default is a BLUR or a CHECK zone covering 95 % of the plate
            hidden_key = f"plates.photo.hidden_default_{names[scale].split('_')[1]}"
            results[f"{hidden_key}_pct"] = pct(out["hidden_default"], out["total"])
            results[f"{hidden_key}_count"] = f"{out['hidden_default']}/{out['total']}"
            flat_interval(results, hidden_key, out["hidden_default"], out["total"])
            if scale == 1.0:
                ms = out["ms"]
                results["plates.speed.ms_per_photo"] = round(float(np.mean(ms)), 1)
                results["plates.speed.ms_per_photo_median"] = round(float(np.median(ms)), 1)
            eu_details += out["misses"]
        details["openalpr-eu"] = eu_details
    if "streets" in sets:
        out = run_streets(module, args.limit)
        results["plates.photo.false_blur_per_100"] = round(100.0 * out["false_blurs"] / max(1, out["photos"]), 2)
        results["plates.photo.false_check_per_100"] = round(100.0 * out["false_checks"] / max(1, out["photos"]), 2)
        results["plates.photo.false_blur"] = rate_per_100(out["false_blurs"], out["photos_false_blur"], out["photos"])
        results["plates.photo.false_check"] = rate_per_100(
            out["false_checks"], out["photos_false_check"], out["photos"]
        )
        results["plates.photo.false_hidden_default"] = rate_per_100(
            out["false_blurs"] + out["false_checks"], out["photos_false_hidden"], out["photos"]
        )
        results["plates.photo.streets.photos"] = out["photos"]
        results["plates.photo.streets.plates"] = out["plates"]
        results["plates.photo.streets.recall_blur_pct"] = pct(out["plates_blurred"], out["plates"])
        flat_interval(results, "plates.photo.streets.recall_blur", out["plates_blurred"], out["plates"])
        details["streets"] = out["details"]
    results["plates.photo.eu.what"] = (
        "plates annotated by OpenALPR (not by us) in 108 photos; the photos are AGPL and stay local"
    )
    results["plates.photo.streets.caveat"] = (
        "16 readable plates in 60 Commons photos, truth by two labellers whose boxes agreed on a minority of plates "
        "(bench/results/truth-check-plates.json): the interval is wide and the truth is the union of both"
    )
    if "streets" in sets:
        why = (
            f"{results.get('plates.photo.streets.plates')} readable plates in {results.get('plates.photo.streets.photos')} photos, truth "
            "from two labellers who agreed on a minority of the plates (bench/results/truth-check-plates.json): not a figure to quote; "
            "the openalpr-eu figures are the ones with a defensible truth"
        )
        results["_publication"] = {
            f"plates.photo.streets.recall_blur{suffix}": not_publishable(why) for suffix in ("_pct", "_n", "_ci95_pct")
        }
    results["truth_sha256"] = truth_hashes(
        [f"bench/truth/plates/{name}.json" for name in ("openalpr-eu", "streets", "streets-author", "streets-verifier")]
    )
    results["run"] = {"date": datetime.date.today().isoformat()}
    results["plates.model_failures"] = module.model_failures
    results["plates.timings_s"] = {k: round(v, 2) for k, v in module.timings.items()}
    results["plates.calls"] = module.calls

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "plates.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
        f.write("\n")
    with open(os.path.join(args.out, "plates-details.json"), "w", encoding="utf-8") as f:
        json.dump(details, f, indent=1)
        f.write("\n")
    print(
        json.dumps(
            {k: v for k, v in results.items() if k not in ("hardware", "plates.timings_s", "plates.calls")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
