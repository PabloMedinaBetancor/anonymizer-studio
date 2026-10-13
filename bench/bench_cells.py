"""Cells of the pilot and results files of the heavy models.

    python3 -s -m bench.bench_cells speed --model retinaface-r34 [--photos 12] [--zone-side 256] [--threads 4] [--out FILE]
    python3 -s -m bench.bench_cells collect --models retinaface-r34,yunet-zoomed --split-file F [--target-side 256] --out RAW
    python3 -s -m bench.bench_cells evaluate --raw RAW --split-file F --out FILE

`speed` times one heavy detector on real suspect-looking zones (a face found by the first pass, cut with 25 % context and enlarged to the registered scale) and on 640 px tiles of whole photos, and records the peak resident memory of the process. One model per run, so the
peak is that model's.

`collect` runs the first pass (the light face module, as `decide_photo` does) and the heavy detectors once over the photos of a split and
records items, persons and what each detector found in each zone; `evaluate` replays that record through the product's own second stage
(bench/cells_eval.py) to score cell C0 (light only), C1 (the control, YuNet zoomed) and C2 (the heavy model on the zones), choose `H_hi`, `H_lo`
and `max_zones` on the split, run the veto study (PF-V) and pair each cell against the one below. The file names the machine and the load it ran under: a speed is quoted from one machine (M7), and the development laptop
only when it was reserved and quiet.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import resource
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from anonymizer.classes import context, heavy
from anonymizer.classes.faces import FaceModule
from anonymizer.classes.heavy import HeavyFaceModule, HeavyThresholds
from anonymizer.contract import FACE, Box, RetryPurpose, SweepPlan
from anonymizer.engine import deep, second_look
from anonymizer.engine.photos import PHOTO_PLAN
from anonymizer.engine.zones import LookItem, build_zones
from bench import cells_eval
from bench.results_meta import truth_hashes

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")
LIGHT_PLAN = SweepPlan(general_side=1600, tile_side=640)
CONTEXT = 0.25  # context per side around the zone box
SCALE_RANGE = (0.5, 6.0)  # enlargement range of the zone crops
TILE = 640  # DM1b: tiles of the deep sweep are 640 px or less


def machine() -> Dict[str, object]:
    """Which machine and how busy: what a speed figure needs next to it."""
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "")
    except OSError:
        pass
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "host": os.environ.get("BENCH_MACHINE_LABEL", "unnamed"),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "load_1min": round(load[0], 2),
    }


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def zone_region(box: Box, width: int, height: int, side: float):
    """The region of the picture to cut (25 % context per side, inside the picture) and the scale that makes its longest side `side`."""
    w, h = box[2] - box[0], box[3] - box[1]
    region = (
        max(0.0, box[0] - CONTEXT * w),
        max(0.0, box[1] - CONTEXT * h),
        min(float(width), box[2] + CONTEXT * w),
        min(float(height), box[3] + CONTEXT * h),
    )
    longest = max(region[2] - region[0], region[3] - region[1], 1.0)
    return region, float(np.clip(side / longest, *SCALE_RANGE))


def summary(ms: Sequence[float], cpu_ms: Sequence[float]) -> Dict[str, object]:
    if not ms:
        return {"n": 0}
    ordered = sorted(ms)
    return {
        "n": len(ms),
        "wall_ms_median": round(statistics.median(ms), 1),
        "wall_ms_p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
        "wall_ms_max": round(ordered[-1], 1),
        "cpu_ms_median": round(statistics.median(cpu_ms), 1),
    }


def timed(call, *args) -> tuple:
    wall, cpu = time.perf_counter(), time.process_time()
    result = call(*args)
    return result, 1000 * (time.perf_counter() - wall), 1000 * (time.process_time() - cpu)


def dev_set_a_files(split_file: str) -> List[str]:
    """The photos of set a that are in the dev part, in a fixed order: what the speed and memory runs use (nothing is run on a sealed photo)."""
    table = load_split(split_file)
    return [image_path(k) for k in sorted(k for k, part in table.items() if part == "dev" and k.startswith("set-a/"))]


def speed(
    model: str, photos: int, zone_side: float, threads: int, out: Optional[str], split_file: str
) -> Dict[str, object]:
    files = dev_set_a_files(split_file)[:photos]
    if not files:
        raise SystemExit(f"no dev photos of set a in {DATA}/faces/set-a (source scripts/dev-env.sh)")
    light = FaceModule(threads=threads)
    detector = heavy.load_face_heavy(model, threads=threads)
    started = machine()
    zone_ms: List[float] = []
    zone_cpu: List[float] = []
    tile_ms: List[float] = []
    tile_cpu: List[float] = []
    zones = tiles = 0
    first_call_ms = None
    for index, path in enumerate(files):
        bgr = cv2.imread(path)
        height, width = bgr.shape[:2]
        for det in sorted(light.sweep(bgr, LIGHT_PLAN), key=lambda d: -d.score)[:8]:
            region, scale = zone_region(det.box(), width, height, zone_side)
            _, wall, cpu = timed(detector.detect, bgr, region, scale, 0.5)
            if first_call_ms is None:
                first_call_ms = wall  # includes opening the session
                continue
            zone_ms.append(wall)
            zone_cpu.append(cpu)
            zones += 1
        if index < 4:
            for tile in speed_tiles(width, height):
                _, wall, cpu = timed(detector.detect, bgr, tile, 1.0, 0.5)
                tile_ms.append(wall)
                tile_cpu.append(cpu)
                tiles += 1
    detector.close()
    light.close()
    result = {
        "what": "speed and memory of one heavy face detector on zones and tiles",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": {"name": detector.name, "version": detector.version, "sha256": detector.sha256},
        "machine": started,
        "load_1min_at_end": machine()["load_1min"],
        "threads": threads,
        "python": sys.version.split()[0],
        "photos": len(files),
        "zone_side_px": zone_side,
        "first_call_ms_includes_session_open": round(first_call_ms or 0.0, 1),
        "zone": summary(zone_ms, zone_cpu),
        "tile_640": summary(tile_ms, tile_cpu),
        "peak_rss_mb": peak_rss_mb(),
        "caveat": "design measurement: zones are the 8 best first-pass faces of each photo, not the suspect zones of the engine; the median "
        "is of calls after the first; quote it only with its machine and load",
        "_publication": {
            "speed": {
                "publishable": False,
                "why": "design measurement; the published speeds come from the pilot cells on a reserved development laptop",
            }
        },
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return result


def speed_tiles(width: int, height: int) -> List[Box]:
    from anonymizer.geometry import tiles

    return [t for t in tiles(width, height, TILE, 0.2)][:6]


TRUTH_DIR = os.path.join(HERE, "truth", "faces")
SETS = ("a", "b")  # a: photos with faces; b: photos without people
WIDER_DIR = os.environ.get(
    "ANONYMIZER_WIDER_DIR", ""
)  # WIDER FACE validation subset: val/<name>.jpg and subset.json (test data, never distributed)
WIDER_MIN_HEIGHT = 10  # WIDER's own rule: a face under 10 px high, or flagged invalid, is neither a hit nor a false


def first_pass(face: FaceModule, bgr: np.ndarray) -> List[LookItem]:
    """What `decide_photo` leaves for the second look: every sweep candidate with its VERIFY retry, the rules' verdict, dropped ones included."""
    items = []
    for det in face.sweep(bgr, PHOTO_PLAN):
        verify = face.retry(bgr, det.box(), RetryPurpose.VERIFY)
        box = det.box()
        if verify is not None:
            box = (min(box[0], verify.x0), min(box[1], verify.y0), max(box[2], verify.x1), max(box[3], verify.y1))
        judgement = face.photo_judgement(det, verify)
        items.append(LookItem(box, FACE, det.score, judgement.verdict, judgement, det))
    return items


def load_truth(tag: str, truth_dir: str = TRUTH_DIR) -> dict:
    with open(os.path.join(truth_dir, f"set-{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def wider_truth(wider_dir: str) -> dict:
    """{name: {"boxes": [...]}} for the WIDER subset in the shape of the other truths: a face flagged invalid or under 10 px high is `ignore`."""
    with open(os.path.join(wider_dir, "subset.json"), encoding="utf-8") as f:
        subset = json.load(f)
    out = {}
    for name, boxes in subset.items():
        out[name.replace("/", "__")] = {
            "boxes": [
                {"box": b["box"], "ignore": bool(b.get("invalid")) or b["box"][3] - b["box"][1] < WIDER_MIN_HEIGHT}
                for b in boxes
            ]
        }
    return out


def image_path(key: str, wider_dir: str = "") -> str:
    """Where the photo `key` ("set-a/x.jpg", "wider/x.jpg") is."""
    kind, name = key.split("/", 1)
    if kind == "wider":
        return os.path.join(wider_dir or WIDER_DIR, "val", name)
    return os.path.join(DATA, "faces", kind, name)


SEALED_REFUSAL = "the sealed part is read once, by bench.read_sealed; the cells of the pilot and the choice of thresholds are run on the dev part only"


def read_split(path: str):
    """The split file as ({photo: part}, {photo: sha256 of its bytes}). Two shapes are read: B's `split-v2.json` (a rule and, per photo, its part and the
    SHA-256 of its image: the hashes let a run check that no image moved) and the flat view {photo: part} (no hashes). A part that is neither
    dev nor sealed is an error."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("items"), dict):
        table = {k: v.get("part") for k, v in data["items"].items()}
        hashes = {k: v["sha256"] for k, v in data["items"].items() if v.get("sha256")}
    else:
        table, hashes = data, {}
    if not isinstance(table, dict) or not table:
        raise SystemExit(f"{path}: not a split file (a map from photo to dev or sealed)")
    bad = sorted({v for v in table.values() if v not in ("dev", "sealed")}, key=str)
    if bad:
        raise SystemExit(f"{path}: unknown part {bad}: a photo is either dev or sealed")
    return table, hashes


def load_split(path: str) -> Dict[str, str]:
    """The split table {photo: dev | sealed}, fixed by file hash before any model sees a photo."""
    return read_split(path)[0]


def read_image(key: str, hashes: Dict[str, str], wider_dir: str = ""):
    """The photo `key` as BGR, after checking its bytes against the split's SHA-256 when the split has one: an image that moved since the split was
    fixed is an error, not a photo to measure."""
    path = image_path(key, wider_dir)
    with open(path, "rb") as f:
        data = f.read()
    expected = hashes.get(key)
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(
            f"{key}: the image is not the one the split was fixed on (SHA-256 differs): do not measure on it"
        )
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {key}")
    return bgr


def dev_part(split_file: str, part: str = "dev") -> Dict[str, str]:
    """The split table, after refusing any part but dev: nothing here reads the sealed photos or their truth."""
    if part != "dev":
        raise SystemExit(SEALED_REFUSAL)
    return load_split(split_file)


def split_names(split_file: str, part: str = "dev") -> set:
    """The photos of the dev part, as "set-a/name.jpg"."""
    return {k for k, v in dev_part(split_file, part).items() if v == "dev"}


class Recorder:
    """A heavy detector that writes down what every call returned (box and score), in call order."""

    def __init__(self, detector):
        self.detector = detector
        self.name, self.version, self.sha256 = detector.name, detector.version, detector.sha256
        self.log: List[List[List[float]]] = []
        self.pieces: List[float] = []  # the slowest model run of each call, in seconds (SL6)
        self.seconds: List[float] = []  # the time of each call, in seconds

    def detect(self, bgr, region, scale, min_score):
        self.detector.slowest_piece = 0.0
        started = time.perf_counter()
        found = self.detector.detect(bgr, region, scale, min_score)
        self.seconds.append(round(time.perf_counter() - started, 4))
        self.pieces.append(round(self.detector.slowest_piece, 4))
        self.log.append(
            [[round(d.x0, 2), round(d.y0, 2), round(d.x1, 2), round(d.y1, 2), round(d.score, 4)] for d in found]
        )
        return found

    def close(self) -> None:
        self.detector.close()


def photo_keys(
    table: Dict[str, str], limit: int, sets: Sequence[str] = SETS, wider_dir: str = "", part: str = "dev"
) -> List[str]:
    """The photos of `part` to run, in a fixed order: the v1 sets "a" and "b" and, with a WIDER folder, "wider". A photo that the split file does not
    know is an error, never taken as dev; the sealed part is refused."""
    if part != "dev":
        raise SystemExit(SEALED_REFUSAL)
    keys: List[str] = []
    for tag in sets:
        if tag == "wider":
            candidates = [f"wider/{n}" for n in sorted(wider_truth(wider_dir or WIDER_DIR))]
        else:
            candidates = [f"set-{tag}/{n}" for n in sorted(load_truth(tag)["images"])]
        unknown = [k for k in candidates if k not in table]
        if unknown:
            raise SystemExit(
                f"{len(unknown)} photos are not in the split file (first: {unknown[0]}): fix the split, do not guess a part"
            )
        keys += [k for k in candidates if table[k] == "dev"][: limit or None]
    return keys


def rss_mb() -> float:
    """The resident set of this process now, in MB (Linux)."""
    with open("/proc/self/statm", encoding="utf-8") as f:
        return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)


def memory(zones: int, long_side: int, threads: int, out: Optional[str], split_file: str) -> dict:
    """SL8: the peak resident set with every model loaded (the light face and plate models, the vehicle and person detector, the heavy
    face model) on a `long_side` px photo that has at least `zones` suspect zones, all of them visited."""
    from anonymizer.classes.plates import PlateModule

    stages = {"start": rss_mb()}
    face = HeavyFaceModule(threads=threads)
    plates = PlateModule(threads=threads)
    finder = context.PersonFinder(threads)
    stages["models_built"] = rss_mb()
    chosen = None
    for path in dev_set_a_files(split_file):
        bgr = cv2.imread(path)
        scale = long_side / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        items = first_pass(face, bgr)
        persons = finder.persons(bgr)
        if len(build_zones(items, persons, plates.last_vehicles, (FACE,))) >= zones:
            chosen = (os.path.basename(path), bgr, items, persons)
            break
    if chosen is None:
        raise SystemExit(f"no photo of set a has {zones} zones at {long_side} px")
    name, bgr, items, persons = chosen
    plates.sweep(bgr, PHOTO_PLAN)
    stages["first_pass_and_plates"] = rss_mb()
    started = time.perf_counter()
    result = second_look.second_stage(bgr, items, [face], persons, max_zones=zones)
    seconds = time.perf_counter() - started
    stages["after_second_look"] = rss_mb()
    face.close()
    plates.close()
    finder.close()
    report = {
        "what": "peak resident set with every model loaded on one photo with the zones asked for",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": machine(),
        "threads": threads,
        "photo": {
            "name": name,
            "long_side_px": long_side,
            "zones_visited": result.zones_visited,
            "zones_total": result.zones_total,
        },
        "second_look_seconds": round(seconds, 2),
        "rss_mb_by_stage": stages,
        "peak_rss_mb": peak_rss_mb(),
        "limit_mb": 1200,
        "within_limit": peak_rss_mb() < 1200,
        "_publication": {"memory": {"publishable": False, "why": "design measurement on a public crowd photo"}},
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    return report


def record_photo(
    bgr: np.ndarray, items: Sequence[LookItem], persons: Sequence, modules: Dict[str, HeavyFaceModule]
) -> dict:
    """What `evaluate` needs of one photo: the items and persons of the first pass, the zones of the second look (built once, so that every
    detector is given the same ones: SL10) and what each detector found in each zone. The modules' heavy detectors are `Recorder`s."""
    height, width = bgr.shape[:2]
    record = {
        "width": width,
        "height": height,
        "items": [
            {
                "box": list(i.box),
                "score": i.score,
                "verdict": i.verdict.value,
                "reason": i.judgement.reason,
                "det": list(i.det.box()),
            }
            for i in items
        ],
        "persons": [[*p.box(), p.score] for p in persons],
        "zones": [{"box": list(z.box), "why": z.why, "item": z.item} for z in build_zones(items, persons, (), (FACE,))],
        "cells": {},
    }
    for model, module in modules.items():
        module.heavy_zone.log.clear()
        module.heavy_zone.pieces.clear()
        result = second_look.second_stage(bgr, items, [module], persons, max_zones=cells_eval.NOTHING)
        record["cells"][model] = {
            "dets": list(module.heavy_zone.log),
            "seconds": [round(v.seconds, 4) for v in result.visits],
            "piece_s": list(module.heavy_zone.pieces),
        }
    return record


def collect(
    models: Sequence[str],
    keys: Sequence[str],
    target_side: int,
    threads: int,
    out: str,
    hashes: Optional[Dict[str, str]] = None,
) -> dict:
    """Run the first pass and each heavy detector once over `keys` and write the record that `evaluate` replays."""
    face = FaceModule(threads=threads)
    finder = context.PersonFinder(threads)
    thresholds = HeavyThresholds(target_side=target_side)
    modules = {
        m: HeavyFaceModule(
            threads=threads, heavy_thresholds=thresholds, heavy_zone=Recorder(heavy.load_face_heavy(m, threads))
        )
        for m in models
    }
    photos: Dict[str, dict] = {}
    for key in keys:
        bgr = read_image(key, hashes or {})
        started = time.perf_counter()
        items = first_pass(face, bgr)
        first_ms = 1000 * (time.perf_counter() - started)
        record = record_photo(bgr, items, finder.persons(bgr), modules)
        record["first_pass_ms"] = round(first_ms, 1)
        photos[key] = record
        print(f"{key}: {len(items)} items, {len(record['zones'])} zones", flush=True)
    for module in modules.values():
        module.close()
    finder.close()
    face.close()
    raw = {
        "what": "first-pass items, persons and the detections of each heavy detector in each zone; replayed by `evaluate`",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": machine(),
        "target_side_px": target_side,
        "look_min": thresholds.look_min,
        "models": {
            m: {"version": mod.heavy_zone.version, "sha256": mod.heavy_zone.sha256} for m, mod in modules.items()
        },
        "photos": photos,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return raw


def load_raw(paths: Sequence[str]) -> dict:
    """One or more recorded runs of the same detectors, put together (a run of the v1 sets and a run of WIDER, say)."""
    merged: Optional[dict] = None
    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if merged is None:
            merged = raw
            continue
        if raw["models"] != merged["models"] or raw["target_side_px"] != merged["target_side_px"]:
            raise SystemExit(f"{path} was recorded with other models or another target side")
        merged["photos"].update(raw["photos"])
    if merged is None:
        raise SystemExit("no recorded run given")
    return merged


def add_adjudicated(truths: Dict[str, dict], path: str) -> int:
    """Put the faces judged real on their crops (the pooled adjudication) into the truth of the photos in `truths`; returns how many."""
    with open(path, encoding="utf-8") as f:
        extra = json.load(f)["additions"]
    added = 0
    for entry in extra:
        if entry["image"] in truths:
            truths[entry["image"]]["boxes"].append(
                {"box": entry["box"], "ignore": bool(entry.get("ignore")), "note": entry.get("note", "")}
            )
            added += 1
    return added


def collect_deep(
    models: Sequence[str], keys: Sequence[str], threads: int, out: str, hashes: Optional[Dict[str, str]] = None
) -> dict:
    """Run the first pass and the deep sweep of each heavy detector over `keys` once (every tile of every photo, no tile limit) and record what
    each call found and how long it took; `evaluate --deep-raw` replays it as C4 (united with the first pass) and C3 (the heavy model alone)."""
    face = FaceModule(threads=threads)
    modules = {}
    for m in models:
        recorder = Recorder(heavy.load_face_heavy(m, threads))
        modules[m] = HeavyFaceModule(threads=threads, heavy_zone=recorder, heavy_deep=recorder)
    photos: Dict[str, dict] = {}
    for key in keys:
        bgr = read_image(key, hashes or {})
        items = first_pass(face, bgr)
        height, width = bgr.shape[:2]
        record = {"width": width, "height": height, "deep": {}}
        for model, module in modules.items():
            recorder = module.heavy_deep
            recorder.log.clear()
            recorder.pieces.clear()
            recorder.seconds.clear()
            result = deep.deep_stage(bgr, items, [module], max_tiles=deep.NO_LIMIT)
            record["deep"][model] = {
                "dets": list(recorder.log),
                "seconds": list(recorder.seconds),
                "piece_s": list(recorder.pieces),
                "calls": result.calls,
                "failures": result.failures,
            }
        photos[key] = record
        print(f"{key}: deep sweep recorded", flush=True)
    for module in modules.values():
        module.close()
    face.close()
    raw = {
        "what": "what each heavy detector found in every call of a deep sweep: replayed by `evaluate --deep-raw`",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": machine(),
        "models": {
            m: {"version": mod.heavy_zone.version, "sha256": mod.heavy_zone.sha256} for m, mod in modules.items()
        },
        "photos": photos,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return raw


def deep_cells(
    report: dict,
    deep_paths: Sequence[str],
    photos: Dict[str, dict],
    truths: Dict[str, dict],
    zone_results: Dict[str, dict],
    base: dict,
    thresholds: Dict[str, HeavyThresholds],
    without: Sequence[str],
    with_people: Sequence[str],
    replayer,
) -> None:
    """Add the deep cells to `report`: C3 (the heavy model alone over the picture), C4 (united with the first pass), each paired against C0 and
    against C2 (the same model on the zones), with the time the calls took; and the verdict of DM6 on whether deep earns its cost."""
    recorded: Dict[str, dict] = {}
    for path in deep_paths:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for key, record in raw["photos"].items():
            recorded.setdefault(key, {}).update(record["deep"])
    keys = [k for k in photos if k in recorded]
    if not keys:
        raise SystemExit("the deep record has none of the photos of the dev part")
    sub = {k: photos[k] for k in keys}
    out: Dict[str, object] = {}
    for model, zone_th in thresholds.items():
        cell = {k: recorded[k][model]["dets"] for k in keys}
        secs = {k: sum(recorded[k][model]["seconds"]) for k in keys}
        # The cuts of the zones are not the cuts of the whole picture: deep is given its own, by the same two rules.
        hi_study = cells_eval.choose_h_hi(cell, truths)
        hi = hi_study["h_hi"] or 0.99
        sub_without = [k for k in without if k in sub]
        lo_study = cells_eval.choose_h_lo_deep(cell, sub, truths, sub_without, hi, replayer, base, secs)
        th = HeavyThresholds(
            h_hi=hi,
            h_lo=lo_study["h_lo"] if lo_study["h_lo"] is not None else hi,
            target_side=zone_th.target_side,
            look_min=zone_th.look_min,
        )
        at_zone_cuts = cells_eval.evaluate_deep(sub, truths, cell, zone_th, False, replayer, secs)
        run = {}
        for name, only in (("C4_union", False), ("C3_heavy_only", True)):
            got = cells_eval.evaluate_deep(sub, truths, cell, th, only, replayer, secs)
            run[name] = got
        c2 = zone_results[model]
        c2_sub = {i: v for i, v in c2["faces"].items() if i.rsplit("#", 1)[0] in sub}
        base_sub = {i: v for i, v in base["faces"].items() if i.rsplit("#", 1)[0] in sub}
        entry: Dict[str, object] = {
            "photos": len(keys),
            "thresholds": {"h_hi": th.h_hi, "h_lo": th.h_lo, "h_hi_study": hi_study, "h_lo_study": lo_study},
            "at_the_cuts_of_the_zones": {
                "h_hi": zone_th.h_hi,
                "h_lo": zone_th.h_lo,
                **cells_eval.summary(at_zone_cuts, sub_without, [k for k in with_people if k in sub]),
            },
        }
        for name, got in run.items():
            summary = cells_eval.summary(got, [k for k in without if k in sub], [k for k in with_people if k in sub])
            summary["calls_per_photo"] = round(got["work"]["calls"] / max(1, len(keys)), 1)
            summary["ms_per_photo"] = round(1000 * got["work"]["seconds"] / max(1, len(keys)), 1)
            summary["paired_over_C0"] = cells_eval.paired(got["faces"], base_sub, 1)
            summary["paired_over_C2"] = cells_eval.paired(got["faces"], c2_sub, 1)
            summary["by_source"] = cells_eval.by_source(got, sub)
            entry[name] = summary
        entry["slowest_model_run_s"] = max((max(recorded[k][model]["piece_s"], default=0.0) for k in keys), default=0.0)
        union = entry["C4_union"]
        zones_only_pct = round(100.0 * sum(1 for v in c2_sub.values() if v[1]) / max(1, len(c2_sub)), 2)
        gain_points = round(union["hidden_default_pct"] - zones_only_pct, 2)
        paired_p = union["paired_over_C2"]["mcnemar_p"]
        entry["dm6"] = {
            "c2_hidden_default_pct": zones_only_pct,
            "c4_hidden_default_pct": union["hidden_default_pct"],
            "gain_points": gain_points,
            "mcnemar_p": paired_p,
            "deep_beats_careful": cells_eval.deep_beats_careful(gain_points, paired_p),
            "rule": "deep beats careful by >= 1.0 point, paired p < 0.05; otherwise it ships as the audit mode and the documents say it did not",
            "heavy_only_equals_union": cells_eval.paired(run["C3_heavy_only"]["faces"], run["C4_union"]["faces"], 1),
        }
        out[model] = entry
    report["deep"] = out


def evaluate(
    raw_paths: Sequence[str],
    split: set,
    cap: Optional[int],
    out: Optional[str],
    fixed: Optional[dict] = None,
    truth_dir: str = TRUTH_DIR,
    additions: Optional[str] = None,
    deep_paths: Sequence[str] = (),
) -> dict:
    """Score C0 and every recorded detector on the photos of `split`, choosing its thresholds on those photos by the registered rules."""
    raw = load_raw(raw_paths)
    outside = sorted(k for k in raw["photos"] if k not in split)
    if outside:
        raise SystemExit(
            f"the record holds {len(outside)} photos outside the dev part (first: {outside[0]}): {SEALED_REFUSAL}"
        )
    photos = dict(raw["photos"])
    truths = {}
    for tag in SETS:
        table = load_truth(tag, truth_dir)["images"]
        for key in photos:
            if key.startswith(f"set-{tag}/"):
                truths[key] = table[key.split("/", 1)[1]]
    if any(k.startswith("wider/") for k in photos):
        table = wider_truth(WIDER_DIR)
        truths.update({k: table[k.split("/", 1)[1]] for k in photos if k.startswith("wider/")})
    adjudicated = add_adjudicated(truths, additions) if additions else 0
    without = [k for k in photos if k.startswith("set-b/")]
    with_people = [k for k in photos if not k.startswith("set-b/")]
    replayer = cells_eval.Replayer()
    default = HeavyThresholds()
    base = cells_eval.evaluate(photos, truths, None, default, 0, replayer)
    out_cells = {"C0": cells_eval.summary(base, without, with_people)}
    out_cells["C0"]["by_source"] = cells_eval.by_source(base, photos)
    counts = cells_eval.choose_max_zones(photos)
    cap = cap or counts["max_zones"]
    results = {}
    used: Dict[str, HeavyThresholds] = {}
    for model in raw["models"]:
        cell = {k: v["cells"][model]["dets"] for k, v in photos.items()}
        if fixed and model in fixed:
            hi, lo = fixed[model]
            chosen = {"h_hi": hi, "h_lo": lo, "source": "given"}
        else:
            hi_study = cells_eval.choose_h_hi(cell, truths)
            hi = hi_study["h_hi"] or 0.99
            lo_study = cells_eval.choose_h_lo(cell, photos, truths, without, hi, replayer, base)
            lo = lo_study["h_lo"] if lo_study["h_lo"] is not None else hi
            chosen = {"h_hi": hi, "h_lo": lo, "h_hi_study": hi_study, "h_lo_study": lo_study}
        th = HeavyThresholds(h_hi=hi, h_lo=lo, target_side=int(raw["target_side_px"]), look_min=raw["look_min"])
        secs = {k: v["cells"][model]["seconds"] for k, v in photos.items()}
        got = cells_eval.evaluate(photos, truths, cell, th, cap, replayer, secs)
        results[model] = got
        used[model] = th
        seconds = got["work"]["seconds"]
        entry = cells_eval.summary(got, without, with_people)
        entry["thresholds"] = chosen
        entry["second_look_ms_per_photo"] = round(1000 * seconds / max(1, len(photos)), 1)
        entry["slowest_call_s"] = round(got["work"]["max_call_s"], 3)
        entry["slowest_model_run_s"] = max(
            (max(v["cells"][model].get("piece_s", [0.0]), default=0.0) for v in photos.values()), default=0.0
        )
        entry["paired_over_C0"] = {
            "blur_or_check": cells_eval.paired(got["faces"], base["faces"], 1),
            "blur": cells_eval.paired(got["faces"], base["faces"], 0),
        }
        entry["veto_study"] = cells_eval.veto_study(cell, photos, truths, th)
        entry["by_source"] = cells_eval.by_source(got, photos)
        entry["cap_study"] = cells_eval.cap_study(photos, truths, cell, th, replayer, secs, without, with_people)
        out_cells[model] = entry
    models = list(raw["models"])
    if "yunet-zoomed" in results:
        for model in models:
            if model != "yunet-zoomed":
                out_cells[model]["paired_over_C1"] = {
                    "blur_or_check": cells_eval.paired(results[model]["faces"], results["yunet-zoomed"]["faces"], 1)
                }
    report = {
        "what": "cells C0 (light only), C1 (control: YuNet zoomed) and C2 (heavy model on the zones) on the photos of one split, replayed from a recorded run",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "photos": {
            "with_people": len(with_people),
            "without_people": len(without),
            "split": "dev",
        },
        "max_zones": counts,
        "cap_used": cap,
        "cells": out_cells,
        "target_side_px": raw["target_side_px"],
        "recorded_on": raw["machine"],
        "truth_sha256": truth_hashes([f"bench/truth/faces/set-{tag}.json" for tag in SETS]),
        "truth_dir": os.path.relpath(truth_dir, os.path.dirname(HERE))
        if truth_dir.startswith(os.path.dirname(HERE))
        else truth_dir,
        "adjudicated_faces_added": adjudicated,
        "caveat": "the truth is v1 (YuNet proposals) unless a v2 truth was given: an optimistic bound that favours the control; the faces of one "
        "photo are not independent",
        "_publication": {
            "cells": {
                "publishable": False,
                "why": "dev numbers choose thresholds; the published ones come from the single sealed read",
            }
        },
    }
    if deep_paths:
        deep_cells(report, deep_paths, photos, truths, results, base, used, without, with_people, replayer)
    replayer.close()
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    return report


def choose(report_paths: Sequence[str], model: str, out: Optional[str]) -> dict:
    """Put the reports of several enlargements side by side and choose `T_H` for `model` by the rule of `cells_eval.choose_target_side`."""
    reports = {}
    for path in report_paths:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        reports[int(report["target_side_px"])] = report
    result = cells_eval.choose_target_side(reports, model)
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("speed", help="time a heavy detector on zones and tiles")
    s.add_argument("--model", choices=heavy.HEAVY_FACE_MODELS, required=True)
    s.add_argument("--split-file", required=True, help="the split of the truth: only its dev photos are run")
    s.add_argument("--photos", type=int, default=12)
    s.add_argument("--zone-side", type=float, default=256.0)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--out")
    c = sub.add_parser("collect", help="run the first pass and the heavy detectors once over the photos of a split")
    c.add_argument("--models", default=",".join(heavy.HEAVY_FACE_MODELS))
    c.add_argument("--split-file", required=True, help="the split of the truth: {photo: dev | sealed}")
    c.add_argument("--split", default="dev", help="only dev is accepted here")
    c.add_argument("--limit", type=int, default=0, help="only the first N photos of each set (smoke test)")
    c.add_argument("--sets", default="a,b", help="a, b and/or wider (wider needs --wider-dir or $ANONYMIZER_WIDER_DIR)")
    c.add_argument("--wider-dir", default="")
    c.add_argument("--target-side", type=int, default=256)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--out", required=True)
    e = sub.add_parser("evaluate", help="score the cells and choose the thresholds from a recorded run")
    e.add_argument("--raw", required=True, nargs="+")
    e.add_argument("--split-file", required=True, help="the split of the truth: {photo: dev | sealed}")
    e.add_argument("--split", default="dev", help="only dev is accepted here")
    e.add_argument(
        "--deep-raw", nargs="*", default=[], help="records of `collect-deep` to score as the deep cells C3 and C4"
    )
    e.add_argument("--cap", type=int)
    e.add_argument("--truth-dir", default=TRUTH_DIR)
    e.add_argument(
        "--additions", help="faces judged real on their crops, to add to the truth (photos-truth-v2-additions.json)"
    )
    e.add_argument("--out", required=True)
    d = sub.add_parser(
        "collect-deep", help="run the deep sweep of each heavy detector once over the photos of the dev part"
    )
    d.add_argument("--models", default=",".join(heavy.HEAVY_FACE_MODELS))
    d.add_argument("--split-file", required=True, help="the split of the truth: {photo: dev | sealed}")
    d.add_argument("--split", default="dev", help="only dev is accepted here")
    d.add_argument("--limit", type=int, default=0, help="only the first N photos of each set (smoke test)")
    d.add_argument("--sets", default="a,b", help="a, b and/or wider")
    d.add_argument("--wider-dir", default="")
    d.add_argument("--threads", type=int, default=4)
    d.add_argument("--out", required=True)
    m = sub.add_parser("memory", help="peak resident set with every model loaded on a photo with many zones (SL8)")
    m.add_argument("--split-file", required=True, help="the split of the truth: only its dev photos are run")
    m.add_argument("--zones", type=int, default=40)
    m.add_argument("--long-side", type=int, default=2048)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--out")
    t = sub.add_parser("choose", help="choose the enlargement T_H from the reports of several target sides")
    t.add_argument("--reports", required=True, nargs="+")
    t.add_argument("--model", default="retinaface-r34")
    t.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "collect-deep":
        keys = photo_keys(
            dev_part(args.split_file, args.split), args.limit, tuple(args.sets.split(",")), args.wider_dir
        )
        collect_deep(args.models.split(","), keys, args.threads, args.out, read_split(args.split_file)[1])
        return 0
    if args.command == "memory":
        print(json.dumps(memory(args.zones, args.long_side, args.threads, args.out, args.split_file), indent=1))
        return 0
    if args.command == "choose":
        print(json.dumps(choose(args.reports, args.model, args.out), indent=1))
        return 0
    if args.command == "collect":
        sets = tuple(args.sets.split(","))
        keys = photo_keys(dev_part(args.split_file, args.split), args.limit, sets, args.wider_dir)
        collect(args.models.split(","), keys, args.target_side, args.threads, args.out, read_split(args.split_file)[1])
        return 0
    if args.command == "evaluate":
        report = evaluate(
            args.raw,
            split_names(args.split_file, args.split),
            args.cap,
            args.out,
            None,
            args.truth_dir,
            args.additions,
            args.deep_raw,
        )
        for cell, row in report["cells"].items():
            th = row.get("thresholds", {})
            false = row["false_hidden_without_people"]["photos_affected_per_100"]
            print(
                f"{cell}: hidden by default {row['hidden_default_pct']} %, false hidden photos per 100 {false}, "
                f"H_hi {th.get('h_hi')} H_lo {th.get('h_lo')}"
            )
        return 0
    result = speed(args.model, args.photos, args.zone_side, args.threads, args.out, args.split_file)
    print(json.dumps({k: result[k] for k in ("model", "machine", "zone", "tile_640", "peak_rss_mb")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
