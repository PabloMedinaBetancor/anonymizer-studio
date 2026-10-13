"""The measurements of the sealed read: one function per registered claim, run by `bench.read_sealed` on the sealed keys of a frozen product.

A measurement takes the list of sealed keys (the split's keys of the part it measures, e.g. "plates/blind-003.jpg") and returns a JSON-able dict. Nothing here opens a sealed photo except
through the keys `read_sealed.read` hands over; a measurement called by hand (the tests, a dry run on dev) is given dev keys and `bench.read_sealed.refuse_sealed` is the guard of
anything that is not the read. Every figure carries its n and its interval (M5), every comparison is paired on the same items (M4), every recall figure has its cost beside it (M6).

Registered here: `plates_v2` (claim C5b: the plate path as shipped on the sealed part of plate set v2, paired with the v1 YOLOv9-t pipeline and with the text path alone, and the cost of the
plate-position band on the same photos).
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from . import read_sealed as rs
from .bench_cells import read_split
from .cells_eval import paired, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
PLATES_V2_TRUTH = os.path.join(HERE, "truth", "plates", "plates-v2.json")
BLIND_FACES_TRUTH = os.path.join(HERE, "truth", "faces", "blind-faces-photos.json")
FACE_MODES = (
    "fast",
    "control",
    "careful",
    "deep_heavy_only",
    "deep",
    "reference",
    "reference_default",
)  # C0 ... C4 and the reference (C5, RM1)
REFERENCE_MODEL = os.environ.get("ANONYMIZER_REFERENCE_MODEL") or os.path.join(DATA, "reference", "centerface.onnx")
REFERENCE_SHA256 = "09189deaaf8646c5c51a68447e3c744ea1e211798155d4728c20507b9f5aefbc"  # bench/SOURCES.csv row of reference/centerface.onnx
REFERENCE_THRESHOLD_FILE = os.path.join(
    HERE, "truth", "reference-threshold.json"
)  # chosen on dev by `choose_reference_threshold`, frozen with the truth
REFERENCE_GRID = tuple(
    round(0.2 + 0.05 * i, 2) for i in range(15)
)  # the score cuts tried on dev: 0.20 (deface's default) to 0.90
REFERENCE_MASK_SCALE = 1.3  # deface blurs each detection scaled by this (its default --mask-scale)
REFERENCE_FALSE_BUDGET = 3.0  # per 100 photos without people on dev: the cut is the lowest of the grid that stays within it (the aim of claim C4)
VIDEO_V2_TRUTH = os.path.join(HERE, "truth", "faces", "truth-v2-video.json")
VIDEO_V2_DIR = os.path.join(DATA, "faces", "video2", "clips")
VIDEO_MODES = ("fast", "careful")  # the two detection modes of the product compared in video (claim C6)
VIDEO_STRICT, VIDEO_LOOSE = 0.8, 0.5  # the cover rule of video frames (M9: not moved)
PLATE_SCALES = (1.0, 0.5, 0.3)  # full size is the headline, the two reduced sizes are reported next to it
COVERED = 0.95  # a plate counts as hidden when this share of its box is covered (M9: not changed)
BAND_REASON = "plate.reason.position_band"
MASK_SHRINK = 4  # the cost masks are a quarter of the picture


def covered_share(plate: Sequence[float], boxes: Sequence[Sequence[float]]) -> float:
    """Share of the plate box (in pixels) inside the union of `boxes`."""
    x0, y0 = int(round(plate[0])), int(round(plate[1]))
    x1, y1 = max(x0 + 1, int(round(plate[2]))), max(y0 + 1, int(round(plate[3])))
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for bx0, by0, bx1, by1 in boxes:
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        mask[max(0, int(by0) - y0) : max(0, int(by1) - y0 + 1), max(0, int(bx0) - x0) : max(0, int(bx1) - x0 + 1)] = 1
    return float(mask.mean())


def with_margin(
    box: Sequence[float], margin: Tuple[float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    mx, my = (box[2] - box[0]) * margin[0], (box[3] - box[1]) * margin[1]
    return (
        max(0.0, box[0] - mx),
        max(0.0, box[1] - my),
        min(float(width), box[2] + mx),
        min(float(height), box[3] + my),
    )


def union_share(boxes: Sequence[Sequence[float]], width: int, height: int) -> float:
    """Share of the picture inside the union of `boxes`, on a quarter-size mask."""
    mask = np.zeros((height // MASK_SHRINK + 1, width // MASK_SHRINK + 1), np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[
            max(0, int(y0 / MASK_SHRINK)) : int(y1 / MASK_SHRINK) + 1,
            max(0, int(x0 / MASK_SHRINK)) : int(x1 / MASK_SHRINK) + 1,
        ] = 1
    return float(mask.mean())


def plate_hidden(plate: Sequence[float], zones: Sequence[Sequence[float]]) -> bool:
    return covered_share(plate, zones) >= COVERED


def bootstrap_by_photo(per_photo: Dict[str, List[int]], seed: int = 7, rounds: int = 1000) -> List[float]:
    """95 % bootstrap interval of the share (per 100) of hidden plates, resampling PHOTOS (the plates of a photo are not independent): `per_photo` is {photo: [hidden, plates]}."""
    names = sorted(per_photo)
    if not names:
        return [None, None]
    rng = random.Random(seed)
    shares = []
    for _ in range(rounds):
        pick = [per_photo[rng.choice(names)] for _ in names]
        n = sum(p[1] for p in pick)
        shares.append(100.0 * sum(p[0] for p in pick) / n if n else 0.0)
    shares.sort()
    return [round(shares[int(0.025 * rounds)], 2), round(shares[int(0.975 * rounds) - 1], 2)]


def read_plate_photo(name: str, hashes: Dict[str, str]):
    """A plate photo as BGR, after checking its bytes against the split's SHA-256 (an image that moved since the split was fixed is an error)."""
    import hashlib

    path = os.path.join(DATA, "plates", "v2", name)
    with open(path, "rb") as f:
        data = f.read()
    expected = hashes.get(f"plates/{name}")
    if expected and hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"plates/{name}: the image is not the one the split was fixed on (SHA-256 differs)")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read plates/{name}")
    return bgr


def shipped_zones(anon, bgr: np.ndarray, tmp: str):
    """What the shipped product blurs on a photo: the plate zones the analysis blurs by default, with the engine's photo margin added, split into the text regions and the position
    bands, and the counts the summary gives. `anon` is `Anonymizer(Options(faces=False))` with every other option at its default."""
    from anonymizer import api
    from anonymizer import contract as c

    path = os.path.join(tmp, "p.png")
    cv2.imwrite(path, bgr)
    analysis = anon.analyze(path)
    zones, specs = api._photo_zones(analysis, anon.options)
    height, width = bgr.shape[:2]
    text, bands = [], []
    for z in zones:
        if z.kind != c.PLATE or not z.blur:
            continue
        box = with_margin(z.box, specs[z.kind].margin, width, height)
        (bands if z.judgement.reason == BAND_REASON else text).append(box)
    return text, bands, analysis.summary(), dict(analysis.options)


def plates_v2(
    keys: Sequence[str],
    scales: Sequence[float] = PLATE_SCALES,
    split_path: str = rs.SPLIT,
    truth_path: str = PLATES_V2_TRUTH,
    allow_sealed: bool = False,
) -> dict:
    """Claim C5b. For the plate photos among `keys` ("plates/<name>"): per scale, the plates hidden by the shipped plate path (text regions plus position band), by the text regions alone, and by
    the v1 YOLOv9-t pipeline (P0, `bench.bench_plates.decide_photo`, one blur zone covering 95 %, BLUR or CHECK = hidden by default), with Wilson and bootstrap-by-photo intervals and the exact
    McNemar pairings (shipped against P0, band against text alone); and the cost of the band: the share of the picture it blurs (union of its zones with the margin), photos that got a band,
    bands on photos without a searchable vehicle. Every plate and photo is counted; the per-plate verdicts are in the result so that the claim can be rechecked."""
    import anonymizer
    from anonymizer.classes.plates import PlateModule

    from .bench_plates import decide_photo

    names = sorted(k.split("/", 1)[1] for k in keys if k.startswith("plates/"))
    if not allow_sealed:
        rs.refuse_sealed([f"plates/{n}" for n in names], split_path)  # a dry run or a test: dev keys only
    _, hashes = read_split(split_path)
    truth = json.load(open(truth_path, encoding="utf-8"))["images"]
    p0_module = PlateModule(threads=0)
    result: dict = {
        "photos": len(names),
        "scales": {},
        "machine": _machine(),
        "options": "Anonymizer(Options(faces=False)), defaults",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    rows: Dict[float, List[dict]] = {s: [] for s in scales}
    with (
        tempfile.TemporaryDirectory(prefix="sealed-plates-") as tmp,
        anonymizer.Anonymizer(anonymizer.Options(faces=False)) as anon,
    ):
        result["analysis_options"] = None
        for name in names:
            bgr0 = read_plate_photo(name, hashes)
            plates = [p for p in truth[name]["boxes"] if not p.get("ignore")]
            for scale in scales:
                bgr = bgr0 if scale == 1.0 else cv2.resize(bgr0, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                height, width = bgr.shape[:2]
                text, bands, summary, options = shipped_zones(anon, bgr, tmp)
                result["analysis_options"] = result["analysis_options"] or options
                p0_zones = decide_photo(p0_module, bgr)
                p0_hidden = [z.blur_box for z in p0_zones]  # BLUR or CHECK: hidden by default (privacy first)
                boxes = []
                for plate in plates:
                    pb = [v * scale for v in plate["box"]]
                    boxes.append(
                        {
                            "box": [round(v, 1) for v in pb],
                            "readable": plate.get("readable", True),
                            "shipped": bool(plate_hidden(pb, text + bands)),
                            "text_alone": bool(plate_hidden(pb, text)),
                            "p0": bool(any(_single(pb, z) >= COVERED for z in p0_hidden)),
                        }
                    )
                searched = summary.get("zones_visited", 0) + summary.get("zones_skipped", 0)
                rows[scale].append(
                    {
                        "photo": name,
                        "plates": boxes,
                        "bands": len(bands),
                        "band_share": union_share(bands, width, height),
                        "text_share": union_share(text, width, height),
                        "both_share": union_share(text + bands, width, height),
                        "vehicles_searched": int(searched),
                        "summary": {
                            k: summary.get(k)
                            for k in (
                                "bands_added",
                                "band_area_share",
                                "zones_visited",
                                "zones_skipped",
                                "vehicles_too_small",
                            )
                        },
                    }
                )
    for scale in scales:
        result["scales"][str(scale)] = _plates_scale(rows[scale])
    result["rows"] = {str(s): rows[s] for s in scales}
    return result


def check_clip_hash(path: str, expected: set) -> None:
    """A clip that moved since the split was fixed is an error (the split holds the SHA-256 of each clip next to every one of its labelled frames)."""
    expected.discard(None)
    if len(expected) != 1 or rs.sha256_file(path) not in expected:
        raise SystemExit(f"{path}: the clip is not the one the split was fixed on (SHA-256 differs or is missing)")


def _video_flags(clip: str, va, frames: dict) -> dict:
    """Per labelled face of `frames` ({frame: truth entry}): [blurred by default >= 80 %, blurred by default >= 50 %, blurred or listed to check >= 80 %, blurred or listed >= 50 %], from the
    zones the engine would draw (`bench_faces_video.zones_of`, ellipses for faces, the class margins), ids "clip#frame:k" so that `paired` resamples CLIPS (the faces of one clip are not independent)."""
    from . import bench_faces_video as bv

    zones = {"blur": bv.zones_of(va, va.sure_ids()), "check": bv.zones_of(va, {t.id for t in va.listed()})}
    out, ignored_blurred = {}, 0
    for frame, entry in frames.items():
        for k, box in enumerate(entry["boxes"]):
            if box.get("ignore"):
                ignored_blurred += bv.covered_share(box["box"], zones["blur"].get(int(frame), [])) >= VIDEO_STRICT
                continue
            blur = bv.covered_share(box["box"], zones["blur"].get(int(frame), []))
            check = bv.covered_share(box["box"], zones["check"].get(int(frame), []))
            out[f"{clip}#{frame}:{k}"] = [
                blur >= VIDEO_STRICT,
                blur >= VIDEO_LOOSE,
                check >= VIDEO_STRICT,
                check >= VIDEO_LOOSE,
            ]
    return {"faces": out, "ignored_blurred": int(ignored_blurred)}


def video_v2(
    keys: Sequence[str],
    modes: Sequence[str] = VIDEO_MODES,
    split_path: str = rs.SPLIT,
    truth_path: str = VIDEO_V2_TRUTH,
    clips_dir: str = VIDEO_V2_DIR,
    allow_sealed: bool = False,
) -> dict:
    """Claims C6 and VK8. For the labelled frames among `keys` ("video/<clip>/<frame>"; a clip is one unit): the clip is analysed by the product through the public API, once per detection mode
    (`Options(detection=mode, plates=False)`, every other option at its default), and each labelled face is scored as the engine would blur it (ellipses with the class margins; covered >= 80 %,
    and >= 50 % next to it). Per mode: the share of faces covered with its Wilson interval and the bootstrap interval by clip; careful against fast paired by face (exact McNemar, gain in points with the
    bootstrap-by-clip interval); the per-clip table; the cost (seconds, ms per frame, tracks, heavy views). VK8 holds only if the gain at 80 % is at least 10 points with p < 0.05."""
    import anonymizer

    wanted: Dict[str, List[str]] = {}
    for key in keys:
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "video":
            wanted.setdefault(parts[1], []).append(parts[2])
    if not allow_sealed:
        rs.refuse_sealed([f"video/{c}/{f}" for c, fs in wanted.items() for f in fs], split_path)
    truth = json.load(open(truth_path, encoding="utf-8"))["clips"]
    _, hashes = read_split(split_path)
    for clip, frames in wanted.items():
        check_clip_hash(os.path.join(clips_dir, truth[clip]["file"]), {hashes.get(f"video/{clip}/{f}") for f in frames})
    flags: Dict[str, Dict[str, list]] = {m: {} for m in modes}
    per_clip: Dict[str, dict] = {}
    errors: List[str] = []
    for clip in sorted(wanted):
        info = truth[clip]
        frames = {int(f): info["frames"][str(f)] for f in wanted[clip]}
        per_clip[clip] = {"labelled_frames": len(frames), "modes": {}}
        for mode in modes:
            try:
                started = time.perf_counter()
                with anonymizer.Anonymizer(anonymizer.Options(detection=mode, plates=False)) as anon:
                    analysis = anon.analyze(os.path.join(clips_dir, info["file"]))
                    va = analysis._video
                    scored = _video_flags(clip, va, frames)
                    flags[mode].update(scored["faces"])
                    seconds = time.perf_counter() - started
                    per_clip[clip]["modes"][mode] = {
                        "faces": len(scored["faces"]),
                        "covered_80": sum(1 for v in scored["faces"].values() if v[0]),
                        "covered_50": sum(1 for v in scored["faces"].values() if v[1]),
                        "blurred_or_listed_80": sum(1 for v in scored["faces"].values() if v[2]),
                        "ignored_boxes_blurred": scored["ignored_blurred"],
                        "seconds": round(seconds, 1),
                        "frames": va.frames,
                        "ms_per_frame": round(1000 * seconds / max(1, va.frames), 1),
                        "tracks": len(va.tracks),
                        "heavy_views": int(sum(getattr(t, "n_second", 0) for t in va.tracks)),
                        "second": dict(getattr(va, "second", {}) or {}),
                        "unchecked_frames": len(getattr(va, "unchecked", []) or []),
                    }
            except Exception as err:  # noqa: BLE001 - recorded, not hidden: the claim that needs this clip and mode is then made on the others
                flags[mode] = {k: v for k, v in flags[mode].items() if not k.startswith(f"{clip}#")}
                per_clip[clip]["modes"][mode] = {"error": f"{type(err).__name__}: {err}"}
                errors.append(f"{clip}/{mode}")

    result: dict = {
        "errors": errors,
        "clips": sorted(wanted),
        "modes": list(modes),
        "machine": _machine(),
        "per_clip": per_clip,
        "cells": {},
    }
    for mode in modes:
        faces = flags[mode]
        n = len(faces)
        by_clip: Dict[str, List[int]] = {}
        for fid, v in faces.items():
            counts = by_clip.setdefault(fid.rsplit("#", 1)[0], [0, 0])
            counts[0] += 1 if v[0] else 0
            counts[1] += 1
        hits = {
            name: sum(1 for v in faces.values() if v[i])
            for i, name in enumerate(("covered_80", "covered_50", "blurred_or_listed_80", "blurred_or_listed_50"))
        }
        result["cells"][mode] = {
            "faces": n,
            "hits": hits,
            "covered_80_pct": round(100.0 * hits["covered_80"] / n, 2) if n else None,
            "covered_80_wilson95": wilson(hits["covered_80"], n),
            "covered_80_bootstrap_by_clip95": bootstrap_by_photo(by_clip),
            "covered_50_pct": round(100.0 * hits["covered_50"] / n, 2) if n else None,
        }
    if "fast" in flags and "careful" in flags:
        gain = paired(flags["careful"], flags["fast"], 0)
        loose = paired(flags["careful"], flags["fast"], 1)
        n = gain["n"]
        points = round(100.0 * (gain["gained"] - gain["lost"]) / n, 2) if n else None
        result["careful_over_fast"] = {
            "covered_80": gain,
            "covered_50": loose,
            "gain_points_80": points,
            "vk8_holds": bool(points is not None and points >= 10.0 and gain["mcnemar_p"] < 0.05),
            "rule": "VK8: careful over fast in video needs at least +10 points covered >= 80 % with paired exact McNemar p < 0.05",
        }
    return result


class _Zone:
    """What `bench_faces.coverage` and `touches` read of a zone: the box that is blurred (the margin included)."""

    def __init__(self, margin_box):
        self.margin_box = margin_box


class CenterFace:
    """CenterFace (Star-Clouds, MIT) as shipped in deface (MIT), run with OpenCV's DNN on the whole picture once, as deface does (RM1: the reference)."""

    name = "centerface"

    def __init__(self, path: str = REFERENCE_MODEL, threads: int = 4):
        if rs.sha256_file(path) != REFERENCE_SHA256:
            raise SystemExit(f"{path}: the reference model is not the file of bench/SOURCES.csv (SHA-256 differs)")
        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(path)

    def detect(self, bgr: np.ndarray, min_score: float = REFERENCE_GRID[0]) -> List[List[float]]:
        h0, w0 = bgr.shape[:2]
        w, h = int(np.ceil(w0 / 32) * 32), int(np.ceil(h0 / 32) * 32)
        sw, sh = w / w0, h / h0
        self.net.setInput(cv2.dnn.blobFromImage(bgr, 1.0, (w, h), (0, 0, 0), swapRB=False, crop=False))
        heat, scale, offset, _ = self.net.forward(["537", "538", "539", "540"])
        ys, xs = np.where(heat[0, 0] > min_score)
        if len(ys) == 0:
            return []
        s0, s1 = np.exp(scale[0, 0, ys, xs]) * 4, np.exp(scale[0, 1, ys, xs]) * 4
        o0, o1 = offset[0, 0, ys, xs], offset[0, 1, ys, xs]
        x1 = np.clip((xs + o1 + 0.5) * 4 - s1 / 2, 0, w)
        y1 = np.clip((ys + o0 + 0.5) * 4 - s0 / 2, 0, h)
        boxes = np.stack(
            [x1 / sw, y1 / sh, np.minimum(x1 + s1, w) / sw, np.minimum(y1 + s0, h) / sh, heat[0, 0][ys, xs]], 1
        )
        order = np.argsort(-boxes[:, 4])
        kept: List[np.ndarray] = []
        for i in order:  # non-maximum suppression at IoU 0.3, as the pilot's adapter does
            b = boxes[i]
            if all(_iou(b, k) <= 0.3 for k in kept):
                kept.append(b)
        return [[float(v) for v in b] for b in kept]


def _iou(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def reference_zones(detections: Sequence[Sequence[float]], threshold: float, width: int, height: int) -> list:
    """The area deface would blur for the detections at or above `threshold`: each box scaled by REFERENCE_MASK_SCALE about its centre, clipped to the picture."""
    zones = []
    for x0, y0, x1, y1, score in detections:
        if score < threshold:
            continue
        cx, cy, hw, hh = (
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            (x1 - x0) * REFERENCE_MASK_SCALE / 2,
            (y1 - y0) * REFERENCE_MASK_SCALE / 2,
        )
        zones.append(
            _Zone((max(0.0, cx - hw), max(0.0, cy - hh), min(float(width), cx + hw), min(float(height), cy + hh)))
        )
    return zones


def choose_reference_threshold(
    scan: Dict[str, dict], without_people: Sequence[str], budget: float = REFERENCE_FALSE_BUDGET
) -> dict:
    """The one score cut of the reference, chosen on DEV (RM1): of REFERENCE_GRID, the lowest cut whose false-hidden photos without people stay within `budget` per 100 (the grid's top is the
    fallback). `scan` is {photo: {"width", "height", "detections", "truth"}} on dev photos only."""
    from . import bench_faces as bf

    rows = []
    for t in REFERENCE_GRID:
        hits = faces = 0
        false_photos = 0
        for key, item in scan.items():
            zones = reference_zones(item["detections"], t, item["width"], item["height"])
            for b in item["truth"]["boxes"]:
                if not b.get("ignore"):
                    faces += 1
                    hits += bf.coverage(b["box"], zones, False) >= bf.COVERED
            if key in without_people and any(
                not any(bf.touches(z, b["box"]) for b in item["truth"]["boxes"]) for z in zones
            ):
                false_photos += 1
        rows.append(
            {
                "threshold": t,
                "faces": faces,
                "hidden": hits,
                "false_photos_without_people": false_photos,
                "photos_without_people": len(without_people),
                "false_per_100": round(100.0 * false_photos / len(without_people), 2) if without_people else None,
            }
        )
    allowed = [r for r in rows if r["false_per_100"] is not None and r["false_per_100"] <= budget]
    chosen = allowed[0]["threshold"] if allowed else REFERENCE_GRID[-1]
    return {
        "threshold": chosen,
        "rule": f"the lowest score cut of the grid {REFERENCE_GRID[0]}-{REFERENCE_GRID[-1]} whose false-hidden photos without people on dev stay within {budget} per 100",
        "grid": rows,
        "mask_scale": REFERENCE_MASK_SCALE,
    }


def _face_truth(key: str, wider_dir: str) -> dict:
    """The truth entry ({"boxes": [{"box", "ignore"}]}) of the photo `key` ("set-a/x.jpg", "set-b/x.jpg", "wider/x.jpg", "blind/x.jpg")."""
    from . import bench_cells as bc

    kind, name = key.split("/", 1)
    if kind == "blind":
        return json.load(open(BLIND_FACES_TRUTH, encoding="utf-8"))["images"][name]
    if kind == "wider":
        return bc.wider_truth(wider_dir or bc.WIDER_DIR)[name]
    return bc.load_truth(kind.split("-", 1)[1])["images"][name]


def _face_cell(mode: str, threads: int = 0):
    """`(Options kwargs, modules)` of the cell `mode`: C0 fast, C1 the control (careful with the light model zoomed on the zones instead of the heavy one), C2 careful, C3 deep with the heavy
    model alone, C4 deep. Modules are None when the product builds them itself."""
    import anonymizer

    if mode == "control":
        from anonymizer.classes.heavy import HeavyFaceModule, load_face_heavy

        module = HeavyFaceModule(threads=threads, heavy_zone=load_face_heavy("yunet-zoomed", threads))
        return {"detection": "careful"}, [module]
    del anonymizer
    return {
        "fast": {"detection": "fast"},
        "careful": {"detection": "careful"},
        "deep_heavy_only": {"detection": "deep", "heavy_only": True},
        "deep": {"detection": "deep"},
    }[mode], None


def faces_v2(
    keys: Sequence[str],
    modes: Sequence[str] = FACE_MODES,
    split_path: str = rs.SPLIT,
    wider_dir: str = "",
    allow_sealed: bool = False,
) -> dict:
    """Claims C1-C4. For the face photos among `keys` ("set-a/", "set-b/", "wider/", "blind/"): each photo is analysed by the product through the public API once per cell (C0 fast, C1 control,
    C2 careful, C3 deep with the heavy model alone, C4 deep; `unsure="blur"`, plates off, every other option at its default) and each labelled face is scored as the engine would blur it: hidden by
    default when the blurred zones (BLUR or doubtful, with the engine's photo margin) cover >= 95 % of its box (M9: the rule is not moved; the ellipses that are really drawn are given next to it). Per
    part (blind, WIDER, set a, set b) and for the independent parts together: the share hidden with its Wilson and bootstrap-by-photo intervals, the paired comparisons (careful against fast and against the
    control, deep against careful, C3 against C4; exact McNemar, gain per 1,000 faces with the bootstrap by photo), and the false hidden zones per 100 photos without people (zones that touch no truth box;
    ignore boxes neutral) with the second look's own addition (careful minus fast), and the cost (seconds per photo from the same run, M6)."""
    import anonymizer
    from anonymizer import api
    from anonymizer import contract as c
    from anonymizer.contract import Verdict

    from . import bench_cells as bc
    from . import bench_faces as bf

    keys = sorted(k for k in keys if k.split("/", 1)[0] in ("set-a", "set-b", "wider", "blind"))
    if not allow_sealed:
        rs.refuse_sealed(keys, split_path)
    _, hashes = read_split(split_path)
    truths = {k: _face_truth(k, wider_dir) for k in keys}
    counted = {k: sum(1 for b in t["boxes"] if not b.get("ignore")) for k, t in truths.items()}
    without_people = [k for k in keys if counted[k] == 0]
    result: dict = {
        "photos": len(keys),
        "modes": list(modes),
        "machine": _machine(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cells": {},
    }
    flags: Dict[str, Dict[str, list]] = {}
    chosen = (
        json.load(open(REFERENCE_THRESHOLD_FILE, encoding="utf-8"))["threshold"]
        if any(m.startswith("reference") for m in modes)
        else None
    )
    detector = CenterFace() if chosen is not None else None
    scans: Dict[str, List[list]] = {}

    def run_mode(mode: str) -> None:
        flags[mode] = {}
        false_by_photo: Dict[str, int] = {}
        seconds: List[float] = []
        if mode.startswith("reference"):
            threshold = chosen if mode == "reference" else REFERENCE_GRID[0]
            for key in keys:
                bgr = bc.read_image(key, hashes, wider_dir)
                height, width = bgr.shape[:2]
                if key not in scans:
                    started = time.perf_counter()
                    scans[key] = detector.detect(bgr)
                    seconds.append(time.perf_counter() - started)
                zones = reference_zones(scans[key], threshold, width, height)
                for i, box in enumerate(truths[key]["boxes"]):
                    if box.get("ignore"):
                        continue
                    covered = bf.coverage(box["box"], zones, False) >= bf.COVERED
                    flags[mode][f"{key}#{i}"] = [covered, covered, bf.coverage(box["box"], zones, True) >= bf.COVERED]
                false_by_photo[key] = sum(
                    1 for z in zones if not any(bf.touches(z, b["box"]) for b in truths[key]["boxes"])
                )
            result["cells"][mode] = _face_cell_summary(flags[mode], false_by_photo, without_people, seconds)
            result["cells"][mode]["threshold"] = threshold
            return
        kwargs, modules = _face_cell(mode)
        with (
            tempfile.TemporaryDirectory(prefix="sealed-faces-") as tmp,
            anonymizer.Anonymizer(anonymizer.Options(plates=False, **kwargs), modules=modules) as anon,
        ):
            for key in keys:
                bgr = bc.read_image(key, hashes, wider_dir)
                height, width = bgr.shape[:2]
                path = os.path.join(tmp, "p.png")
                cv2.imwrite(path, bgr)
                started = time.perf_counter()
                analysis = anon.analyze(path)
                seconds.append(time.perf_counter() - started)
                zones, specs = api._photo_zones(analysis, anon.options)
                blurred = [
                    _Zone(bf.with_margin(z.box, specs[z.kind].margin, width, height))
                    for z in zones
                    if z.kind == c.FACE and z.blur
                ]
                sure = [
                    _Zone(bf.with_margin(z.box, specs[z.kind].margin, width, height))
                    for z in zones
                    if z.kind == c.FACE and z.blur and z.judgement.verdict is Verdict.BLUR
                ]
                for i, box in enumerate(truths[key]["boxes"]):
                    if box.get("ignore"):
                        continue
                    flags[mode][f"{key}#{i}"] = [
                        bf.coverage(box["box"], blurred, False) >= bf.COVERED,
                        bf.coverage(box["box"], sure, False) >= bf.COVERED,
                        bf.coverage(box["box"], blurred, True) >= bf.COVERED,
                    ]
                false_by_photo[key] = sum(
                    1 for z in blurred if not any(bf.touches(z, b["box"]) for b in truths[key]["boxes"])
                )
        result["cells"][mode] = _face_cell_summary(flags[mode], false_by_photo, without_people, seconds)

    for (
        mode
    ) in modes:  # a failing cell must not take the others with it: a read cannot be repeated without a recorded reason
        try:
            run_mode(mode)
        except Exception as err:  # noqa: BLE001 - recorded, not hidden: the claim that needs this cell is then not made
            flags.pop(mode, None)
            result["cells"][mode] = {"error": f"{type(err).__name__}: {err}"}
            result.setdefault("errors", []).append(mode)
    for name, (better, base) in {
        "careful_over_fast": ("careful", "fast"),
        "careful_over_control": ("careful", "control"),
        "deep_over_careful": ("deep", "careful"),
        "deep_heavy_only_vs_deep": ("deep_heavy_only", "deep"),
        "reference_over_careful": ("reference", "careful"),
        "careful_over_reference": ("careful", "reference"),
    }.items():
        if better in flags and base in flags:
            result[name] = {
                "hidden_by_default": paired(flags[better], flags[base], 0),
                "blurred_with_certainty": paired(flags[better], flags[base], 1),
            }
    result["speed"] = _speed_ratio(result["cells"])
    result["load_at_end"] = os.getloadavg()
    result["parts"] = {part: _face_part(flags, part) for part in ("blind", "wider", "set-a", "set-b")}
    result["independent_parts"] = _face_part(flags, ("blind", "wider"))
    return result


def _face_cell_summary(
    faces: Dict[str, list], false_by_photo: Dict[str, int], without_people: Sequence[str], seconds: Sequence[float]
) -> dict:
    n = len(faces)
    hidden = sum(1 for v in faces.values() if v[0])
    per_photo: Dict[str, List[int]] = {}
    for fid, v in faces.items():
        counts = per_photo.setdefault(fid.rsplit("#", 1)[0], [0, 0])
        counts[0] += 1 if v[0] else 0
        counts[1] += 1
    photos_false = sum(1 for k in without_people if false_by_photo.get(k))
    return {
        "faces": n,
        "hidden_by_default": hidden,
        "hidden_by_default_pct": round(100.0 * hidden / n, 2) if n else None,
        "hidden_by_default_wilson95": wilson(hidden, n),
        "hidden_by_default_bootstrap_by_photo95": bootstrap_by_photo(per_photo),
        "blurred_with_certainty_pct": round(100.0 * sum(1 for v in faces.values() if v[1]) / n, 2) if n else None,
        "hidden_by_default_ellipse_pct": round(100.0 * sum(1 for v in faces.values() if v[2]) / n, 2) if n else None,
        "false_hidden_photos_without_people": {
            "photos": len(without_people),
            "photos_affected": photos_false,
            "per_100_photos": round(100.0 * photos_false / len(without_people), 2) if without_people else None,
            "wilson95_per_100": wilson(photos_false, len(without_people)),
            "events": sum(false_by_photo.get(k, 0) for k in without_people),
        },
        "false_hidden_events_in_photos_with_people": sum(
            v for k, v in false_by_photo.items() if k not in set(without_people)
        ),
        "seconds_by_photo": [round(v, 3) for v in seconds],
        "seconds_per_photo_median": round(float(np.median(seconds)), 2) if seconds else None,
        "seconds_per_photo_mean": round(float(np.mean(seconds)), 2) if seconds else None,
    }


def _speed_ratio(cells: Dict[str, dict], seed: int = 7, rounds: int = 1000) -> dict:
    """Claim C7: what careful costs against fast, per photo on the same machine and in the same run: the median of the per-photo ratios with a bootstrap-by-photo interval, and the ratio of the
    medians. "<= 3x" is claimed only if the median ratio is at most 3 and the upper bound at most 3.3. The machine's load is in the result (quiet means a load under 1)."""
    if "seconds_by_photo" not in cells.get("fast", {}) or "seconds_by_photo" not in cells.get("careful", {}):
        return {}  # a cell that failed has no timings: the claim needing it is not made
    fast, careful = cells["fast"]["seconds_by_photo"], cells["careful"]["seconds_by_photo"]
    ratios = [c / f for f, c in zip(fast, careful, strict=False) if f > 0]
    if not ratios:
        return {}
    rng = random.Random(seed)
    boots = sorted(float(np.median([ratios[rng.randrange(len(ratios))] for _ in ratios])) for _ in range(rounds))
    return {
        "photos": len(ratios),
        "median_ratio": round(float(np.median(ratios)), 3),
        "bootstrap95": [round(boots[int(0.025 * rounds)], 3), round(boots[int(0.975 * rounds) - 1], 3)],
        "ratio_of_medians": round(float(np.median(careful) / np.median(fast)), 3),
        "claim_3x_holds": bool(np.median(ratios) <= 3.0 and boots[int(0.975 * rounds) - 1] <= 3.3),
    }


def _face_part(flags: Dict[str, Dict[str, list]], part) -> dict:
    parts = (part,) if isinstance(part, str) else tuple(part)
    out = {}
    for mode, faces in flags.items():
        mine = {fid: v for fid, v in faces.items() if fid.split("/", 1)[0] in parts}
        n = len(mine)
        hidden = sum(1 for v in mine.values() if v[0])
        per_photo: Dict[str, List[int]] = {}
        for fid, v in mine.items():
            counts = per_photo.setdefault(fid.rsplit("#", 1)[0], [0, 0])
            counts[0] += 1 if v[0] else 0
            counts[1] += 1
        out[mode] = {
            "faces": n,
            "hidden_by_default": hidden,
            "pct": round(100.0 * hidden / n, 2) if n else None,
            "wilson95": wilson(hidden, n),
            "bootstrap_by_photo95": bootstrap_by_photo(per_photo),
        }
    return out


def memory_v2(mode: str, zones: int = 40, long_side: int = 2048, threads: int = 4, split_path: str = rs.SPLIT) -> dict:
    """Claim C8, NOT part of the read (it opens no sealed photo): the peak resident set of ONE process running the product in `mode` ("careful" or "deep", every model loaded, plates on) on a DEV photo of set a
    enlarged to `long_side` px that has at least `zones` suspect zones, all of them visited (`max_zones`). One mode per process, so the peak is that mode's. The machine and its power profile are in the result."""
    import resource

    import anonymizer

    from . import bench_cells as bc

    table, hashes = read_split(split_path)
    dev = sorted(k for k, v in table.items() if v == "dev" and k.startswith("set-a/"))
    rss = {"start": bc.rss_mb()}
    with (
        tempfile.TemporaryDirectory(prefix="sealed-memory-") as tmp,
        anonymizer.Anonymizer(anonymizer.Options(detection=mode, max_zones=zones), threads=threads) as anon,
    ):
        anon.modules()
        rss["models_built"] = bc.rss_mb()
        chosen = None
        for key in dev:
            bgr = bc.read_image(key, hashes)
            scale = long_side / max(bgr.shape[:2])
            bgr = cv2.resize(
                bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            )
            path = os.path.join(tmp, "p.png")
            cv2.imwrite(path, bgr)
            summary = anon.analyze(path).summary()
            visited = int(summary.get("zones_visited", 0))
            if (
                visited >= zones or mode == "deep"
            ):  # deep looks at the whole picture in tiles: it has no zones to count, its first photo is the photo
                chosen = (key, visited)
                break
        rss["after_analysis"] = bc.rss_mb()
    peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    limit = {"careful": 1200, "deep": 1600}[mode]
    return {
        "mode": mode,
        "photo": chosen[0] if chosen else None,
        "zones_visited": chosen[1] if chosen else None,
        "long_side_px": long_side,
        "threads": threads,
        "rss_mb_by_stage": rss,
        "peak_rss_mb": peak,
        "limit_mb": limit,
        "within_limit": bool(chosen and peak < limit),
        "machine": _machine(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="measurements that are not part of the sealed read")
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser(
        "memory", help="claim C8: the peak resident set of the product in one mode on a dev photo with many zones"
    )
    m.add_argument("--mode", choices=["careful", "deep"], required=True)
    m.add_argument("--zones", type=int, default=40)
    m.add_argument("--out")
    args = ap.parse_args(argv)
    result = memory_v2(args.mode, args.zones)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


def _single(plate: Sequence[float], zone: Sequence[float]) -> float:
    ix, iy = (
        max(0.0, min(plate[2], zone[2]) - max(plate[0], zone[0])),
        max(0.0, min(plate[3], zone[3]) - max(plate[1], zone[1])),
    )
    return ix * iy / max(1e-6, (plate[2] - plate[0]) * (plate[3] - plate[1]))


def _plates_scale(rows: Sequence[dict]) -> dict:
    items = [(r["photo"], i, p) for r in rows for i, p in enumerate(r["plates"])]
    n = len(items)
    per = {cell: sum(1 for _, _, p in items if p[cell]) for cell in ("shipped", "text_alone", "p0")}
    per_photo = {cell: {} for cell in per}
    for photo, _, p in items:
        for cell in per:
            counts = per_photo[cell].setdefault(photo, [0, 0])
            counts[0] += 1 if p[cell] else 0
            counts[1] += 1
    flags = {cell: {f"{photo}#{i}": [bool(p[cell]), bool(p[cell])] for photo, i, p in items} for cell in per}
    cost = [r for r in rows]
    banded = [r for r in cost if r["bands"]]
    no_vehicle = [r for r in cost if r["vehicles_searched"] == 0]
    out = {
        "plates": n,
        "hidden": {
            cell: {
                "k": per[cell],
                "pct": round(100.0 * per[cell] / n, 2) if n else None,
                "wilson95": wilson(per[cell], n),
                "bootstrap_by_photo95": bootstrap_by_photo(per_photo[cell]),
            }
            for cell in per
        },
        "shipped_vs_p0": paired(flags["shipped"], flags["p0"], 1),
        "shipped_vs_text_alone": paired(flags["shipped"], flags["text_alone"], 1),
        "difference_shipped_minus_p0_points": round(100.0 * (per["shipped"] - per["p0"]) / n, 2) if n else None,
        "readable": {
            cell: [
                sum(1 for _, _, p in items if p[cell] and p["readable"]),
                sum(1 for _, _, p in items if p["readable"]),
            ]
            for cell in per
        },
        "cost": {
            "photos": len(cost),
            "photos_with_a_band": len(banded),
            "photos_without_a_searchable_vehicle": len(no_vehicle),
            "bands_on_photos_without_a_searchable_vehicle": sum(r["bands"] for r in no_vehicle),
            "band_share_of_picture_mean_all_photos": round(float(np.mean([r["band_share"] for r in cost])), 5)
            if cost
            else None,
            "band_share_of_picture_mean_banded_photos": round(float(np.mean([r["band_share"] for r in banded])), 5)
            if banded
            else None,
            "band_share_of_picture_max": round(max([r["band_share"] for r in cost] or [0.0]), 5),
            "text_share_of_picture_mean_all_photos": round(float(np.mean([r["text_share"] for r in cost])), 5)
            if cost
            else None,
        },
    }
    return out


def _machine() -> dict:
    """The machine of the figure (M7): host, cpu, load and the power profile (a timing is only comparable within one profile)."""
    import subprocess

    from .bench_cells import machine

    out = dict(machine())
    try:
        out["power_profile"] = (
            subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True, timeout=5).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        out["power_profile"] = "unknown"
    return out


rs.MEASUREMENTS["faces_v2"] = lambda keys: faces_v2(
    [k for k in keys if k.split("/", 1)[0] in ("set-a", "set-b", "wider", "blind")], allow_sealed=True
)
rs.MEASUREMENTS["video_v2"] = lambda keys: video_v2([k for k in keys if k.startswith("video/")], allow_sealed=True)
rs.MEASUREMENTS["plates_v2"] = lambda keys: plates_v2([k for k in keys if k.startswith("plates/")], allow_sealed=True)


if __name__ == "__main__":
    raise SystemExit(_main())
