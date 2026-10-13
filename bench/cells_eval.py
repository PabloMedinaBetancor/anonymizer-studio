"""Scoring the cells of the pilot from what the heavy detectors saw, with no model in the loop.

`bench_cells collect` runs the first pass and the heavy detectors once and records, for every photo, the first-pass items, the persons and what
each detector found in each zone. Everything here replays that record through the product's own `second_stage`, so a threshold, a cap or a
decision rule can be changed and scored again without running a model, and two cells are always compared on the same zones (SL10).

Faces count as in `bench_faces`: hidden when the union of the BLUR and CHECK zones (with the photo margin) covers 95 % of the truth box
(`COVERED`, not moved: M9). A false hidden zone is a BLUR or CHECK zone that touches no truth box at all; a box marked `ignore` is neither a hit
nor a false.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from anonymizer.classes.heavy import HeavyFaceModule, HeavyThresholds
from anonymizer.contract import FACE, SECOND, Detection, Judgement, Verdict
from anonymizer.engine import deep, second_look
from anonymizer.engine.zones import LookItem
from bench.bench_faces import COVERED, Zone, coverage, touches, with_margin

NOTHING = 1000  # a cap that no photo reaches: "every zone is visited"


class Replay:
    """A heavy detector that returns what a recorded run found, zone by zone in the order the second look visits them."""

    name, version, sha256 = "replay", "0", "0" * 64

    def __init__(self):
        self.queue: List[List[Detection]] = []
        self.calls = 0

    def load(self, per_zone: Sequence[Sequence[Sequence[float]]]) -> None:
        self.queue = [[Detection(*d[:4], d[4], FACE, SECOND) for d in zone] for zone in per_zone]
        self.calls = 0

    def detect(self, bgr, region, scale, min_score):
        zone = self.queue[self.calls] if self.calls < len(self.queue) else []
        self.calls += 1
        return [Detection(d.x0, d.y0, d.x1, d.y1, d.score, FACE, SECOND) for d in zone if d.score >= min_score]

    def close(self) -> None:
        pass


def look_items(raw_items: Sequence[dict]) -> List[LookItem]:
    out = []
    for r in raw_items:
        verdict = Verdict(r["verdict"])
        out.append(
            LookItem(
                tuple(r["box"]),
                FACE,
                r["score"],
                verdict,
                Judgement(verdict, r["reason"]),
                Detection(*r["det"], r["score"], FACE),
            )
        )
    return out


def persons_of(raw: Sequence[Sequence[float]]) -> List[Detection]:
    return [Detection(*p[:4], p[4], "person") for p in raw]


class Replayer:
    """One module and one `Replay` detector, reused for every photo and every threshold."""

    def __init__(self, models: Optional[str] = None):
        self.heavy = Replay()
        self.module = HeavyFaceModule(models=models, heavy_zone=self.heavy)
        self.margin = self.module.photo_margin

    def run(
        self, photo: dict, dets: Optional[Sequence], th: HeavyThresholds, cap: Optional[int]
    ) -> second_look.SecondLookResult:
        """The second stage of one recorded photo with the thresholds `th` and at most `cap` zones (0: the first pass alone)."""
        self.module.hth = th
        self.heavy.load(dets or [])
        bgr = np.zeros((1, 1, 3), np.uint8)  # the replay never reads the picture
        return second_look.second_stage(
            bgr, look_items(photo["items"]), [self.module], persons_of(photo["persons"]), max_zones=cap
        )

    def close(self) -> None:
        self.module.close()


class Shape:
    """A picture's size and nothing else: the replay never reads pixels, and a deep sweep only asks how big the picture is."""

    def __init__(self, width: int, height: int):
        self.shape = (height, width, 3)


def run_deep(
    replayer: "Replayer", photo: dict, dets: Sequence, th: HeavyThresholds, heavy_only: bool
) -> deep.DeepResult:
    """The deep stage of one recorded photo: the heavy model's calls replayed in the order the sweep makes them, merged into the first pass
    (C4, the union) or on their own (C3, `heavy_only`)."""
    replayer.module.hth = th
    replayer.heavy.load(dets)
    return deep.deep_stage(
        Shape(photo["width"], photo["height"]),
        look_items(photo["items"]),
        [replayer.module],
        max_tiles=deep.NO_LIMIT,
        heavy_only=heavy_only,
    )


def evaluate_deep(
    photos: Dict[str, dict],
    truths: Dict[str, dict],
    cell: Dict[str, Sequence],
    th: HeavyThresholds,
    heavy_only: bool,
    replayer: "Replayer",
    seconds: Dict[str, float],
) -> Dict[str, object]:
    """One deep cell over the photos: per-face outcomes, false hidden per photo and the work (calls, the seconds the model took when recorded)."""
    outcome: Dict[str, List] = {}
    false: Dict[str, int] = {}
    work = {"calls": 0, "added": 0, "confirmed": 0, "seconds": 0.0}
    for name, photo in photos.items():
        result = run_deep(replayer, photo, cell[name], th, heavy_only)
        scored = score_photo(truths[name], zones_of(result.items, replayer.margin, photo["width"], photo["height"]))
        for k, got_blur, got_any, side in scored["faces"]:
            outcome[f"{name}#{k}"] = [got_blur, got_any, side]
        false[name] = scored["false"]
        work["calls"] += result.calls
        work["added"] += result.added
        work["confirmed"] += result.confirmed
        work["seconds"] += seconds.get(name, 0.0)
    return {"faces": outcome, "false": false, "work": work}


def zones_of(items: Iterable[LookItem], margin, width: int, height: int) -> List[Zone]:
    zones = []
    for it in items:
        if it.verdict is Verdict.DROP:
            continue
        zones.append(
            Zone(
                it.box,
                with_margin(it.box, margin, width, height),
                it.verdict,
                it.judgement.reason,
                it.score,
                float(it.second_score or 0.0),
            )
        )
    return zones


def score_photo(truth: dict, zones: Sequence[Zone]) -> Dict[str, object]:
    """The faces of one photo, hidden or not, and the false hidden zones: what the cells are compared on."""
    blur = [z for z in zones if z.verdict is Verdict.BLUR]
    faces = []
    for k, item in enumerate(truth["boxes"]):
        if item.get("ignore"):
            continue
        box = item["box"]
        got_blur = coverage(box, blur, False) >= COVERED
        faces.append(
            (k, got_blur, got_blur or coverage(box, zones, False) >= COVERED, max(box[2] - box[0], box[3] - box[1]))
        )
    false = sum(1 for z in zones if not any(touches(z, item["box"]) for item in truth["boxes"]))
    return {"faces": faces, "false": false}


def evaluate(
    photos: Dict[str, dict],
    truths: Dict[str, dict],
    cell: Optional[Dict[str, Sequence]],
    th: HeavyThresholds,
    cap: Optional[int],
    replayer: Replayer,
    seconds: Optional[Dict[str, Sequence[float]]] = None,
) -> Dict[str, object]:
    """One cell over the photos of a split: per-face outcomes (for paired tests), false hidden per photo and the second look's work.
    `cell` is {photo name: detections per zone}; None is C0, the light models alone. `seconds` is {photo name: seconds of each recorded zone
    call}: the time of the visits that the cap allows, as the model took when it was recorded (the replay itself takes none)."""
    outcome: Dict[str, List] = {}
    false: Dict[str, int] = {}
    work = {
        "zones_total": 0,
        "zones_visited": 0,
        "zones_skipped": 0,
        "confirmed": 0,
        "added": 0,
        "seconds": 0.0,
        "max_call_s": 0.0,
    }
    for name, photo in photos.items():
        result = replayer.run(photo, None if cell is None else cell.get(name), th, 0 if cell is None else cap)
        scored = score_photo(truths[name], zones_of(result.items, replayer.margin, photo["width"], photo["height"]))
        for k, got_blur, got_any, side in scored["faces"]:
            outcome[f"{name}#{k}"] = [got_blur, got_any, side]
        false[name] = scored["false"]
        work["zones_total"] += result.zones_total
        work["zones_visited"] += result.zones_visited
        work["zones_skipped"] += result.zones_skipped
        work["confirmed"] += result.confirmed
        work["added"] += result.added
        taken = list((seconds or {}).get(name, ()))[: result.zones_visited]
        work["seconds"] += sum(taken)
        work["max_call_s"] = max([work["max_call_s"], *taken])
    return {"faces": outcome, "false": false, "work": work}


def wilson(hits: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 2), round(100 * (centre + half), 2)]


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p of the discordant pairs: faces hidden by one cell and not by the other."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(better: Dict[str, List], base: Dict[str, List], index: int, seed: int = 7) -> Dict[str, object]:
    """`better` against `base` on the same faces (`index` 0: hidden by BLUR, 1: hidden by BLUR or CHECK): the discordant counts, the exact
    McNemar p and the gain per 1,000 faces with a 95 % bootstrap interval resampling photos (the faces of one photo are not independent)."""
    ids = sorted(set(better) & set(base))
    gained = [i for i in ids if better[i][index] and not base[i][index]]
    lost = [i for i in ids if base[i][index] and not better[i][index]]
    by_photo: Dict[str, List[int]] = {}
    for i in ids:
        by_photo.setdefault(i.rsplit("#", 1)[0], [0, 0, 0])
    for i in gained:
        by_photo[i.rsplit("#", 1)[0]][0] += 1
    for i in lost:
        by_photo[i.rsplit("#", 1)[0]][1] += 1
    for i in ids:
        by_photo[i.rsplit("#", 1)[0]][2] += 1
    rng = random.Random(seed)
    names = list(by_photo)
    diffs = []
    for _ in range(1000):
        pick = [by_photo[rng.choice(names)] for _ in names] if names else []
        n = sum(p[2] for p in pick)
        diffs.append(1000.0 * (sum(p[0] - p[1] for p in pick)) / n if n else 0.0)
    diffs.sort()
    return {
        "n": len(ids),
        "gained": len(gained),
        "lost": len(lost),
        "mcnemar_p": round(mcnemar_exact(len(gained), len(lost)), 5),
        "gain_per_1000": round(1000.0 * (len(gained) - len(lost)) / len(ids), 1) if ids else None,
        "ci95_per_1000": [round(diffs[24], 1), round(diffs[974], 1)] if diffs else [None, None],
    }


def summary(
    result: Dict[str, object], photos_without_people: Sequence[str], photos_with_people: Sequence[str]
) -> Dict[str, object]:
    faces = result["faces"]
    n = len(faces)
    blur = sum(1 for v in faces.values() if v[0])
    any_ = sum(1 for v in faces.values() if v[1])
    false = result["false"]
    b_events = sum(false[p] for p in photos_without_people)
    b_photos = sum(1 for p in photos_without_people if false[p])
    a_events = sum(false[p] for p in photos_with_people)
    return {
        "faces": n,
        "blur_pct": round(100.0 * blur / n, 2) if n else None,
        "hidden_default_pct": round(100.0 * any_ / n, 2) if n else None,
        "hidden_default_ci95_pct": wilson(any_, n),
        "false_hidden_without_people": {
            "photos": len(photos_without_people),
            "events": b_events,
            "photos_affected": b_photos,
            "photos_affected_per_100": round(100.0 * b_photos / len(photos_without_people), 2)
            if photos_without_people
            else None,
        },
        "false_hidden_events_in_photos_with_people": a_events,
        "second_look": result["work"],
    }


def choose_target_side(reports: Dict[int, dict], model: str, allowed_extra_per_100: float = 2.0) -> Dict[str, object]:
    """`T_H`: of the enlargements measured (`reports` is {target side: the report of `evaluate`}), the one that gives `model` the largest
    paired gain over the light models alone, as long as its false hidden photos without people stay within `allowed_extra_per_100` of the
    light models'; the smaller side wins a tie (it is the faster)."""
    rows = []
    for side, report in sorted(reports.items()):
        cell, base = report["cells"][model], report["cells"]["C0"]
        gain = cell["paired_over_C0"]["blur_or_check"]
        extra = (
            cell["false_hidden_without_people"]["photos_affected_per_100"]
            - base["false_hidden_without_people"]["photos_affected_per_100"]
        )
        rows.append(
            {
                "target_side": side,
                "gain_per_1000": gain["gain_per_1000"],
                "mcnemar_p": gain["mcnemar_p"],
                "extra_false_hidden_photos_per_100": round(extra, 2),
                "ms_per_photo": cell["second_look_ms_per_photo"],
                "allowed": extra <= allowed_extra_per_100,
            }
        )
    ok = [r for r in rows if r["allowed"] and r["gain_per_1000"] is not None]
    best = max(ok, key=lambda r: (r["gain_per_1000"], -r["target_side"])) if ok else None
    return {
        "target_side": best["target_side"] if best else None,
        "rows": rows,
        "rule": "largest paired gain, extra false hidden <= 2 per 100",
    }


def cap_study(
    photos: Dict[str, dict],
    truths: Dict[str, dict],
    cell: Dict[str, Sequence],
    th: HeavyThresholds,
    replayer: Replayer,
    seconds: Dict[str, Sequence[float]],
    without: Sequence[str],
    with_people: Sequence[str],
    caps: Sequence[int] = (10, 20, 30, 40, 60, 100, NOTHING),
) -> List[Dict[str, object]]:
    """What the zone budget costs and buys: hidden by default, false hidden and time per photo at each cap (SL5, `max_zones`)."""
    rows = []
    for cap in caps:
        got = evaluate(photos, truths, cell, th, cap, replayer, seconds)
        s = summary(got, without, with_people)
        rows.append(
            {
                "cap": "all" if cap == NOTHING else cap,
                "hidden_default_pct": s["hidden_default_pct"],
                "false_hidden_photos_per_100": s["false_hidden_without_people"]["photos_affected_per_100"],
                "zones_skipped": got["work"]["zones_skipped"],
                "second_look_ms_per_photo": round(1000 * got["work"]["seconds"] / max(1, len(photos)), 1),
            }
        )
    return rows


def v2_markers(
    report: Dict[str, object], model: str = "retinaface-r34", control: str = "yunet-zoomed"
) -> Dict[str, object]:
    """The figures of one scored split under the names the documents use (`faces.photo.v2.*`: a figure on truth v2 is a different
    measurement from v1's and has its own name). Every recall figure comes with its false-hidden figure, n and interval, and the machine and the
    time of the same run (M6, M7); `fast` is the light models alone (C0), `careful` the heavy model on the zones (C2), `control` the zoomed YuNet (C1).
    The caller decides what may be quoted (`_publication`): this function only names what was measured."""
    cells = report["cells"]
    out: Dict[str, object] = {}

    def put(mode: str, cell: Dict[str, object]) -> None:
        base = f"faces.photo.v2.{mode}"
        out[f"{base}.hidden_default_pct"] = cell["hidden_default_pct"]
        out[f"{base}.hidden_default_n"] = cell["faces"]
        out[f"{base}.hidden_default_ci95_pct"] = cell["hidden_default_ci95_pct"]
        out[f"{base}.blur_pct"] = cell["blur_pct"]
        out[f"{base}.false_hidden_default"] = cell["false_hidden_without_people"]
        if "second_look_ms_per_photo" in cell:
            out[f"{base}.second_look_ms_per_photo"] = cell["second_look_ms_per_photo"]
        out[f"{base}.by_source"] = cell["by_source"]

    put("fast", cells["C0"])
    put("careful", cells[model])
    if control in cells:
        put("control", cells[control])
    gain = cells[model]["paired_over_C0"]["blur_or_check"]
    out["faces.photo.v2.careful.over_fast"] = gain
    if "paired_over_C1" in cells[model]:
        out["faces.photo.v2.careful.over_control"] = cells[model]["paired_over_C1"]["blur_or_check"]
    thresholds = cells[model]["thresholds"]
    out["faces.photo.v2.careful.thresholds"] = {k: thresholds[k] for k in ("h_hi", "h_lo")}
    out["faces.photo.v2.careful.zones_skipped"] = cells[model]["second_look"]["zones_skipped"]
    if "deep" in report and model in report["deep"]:
        deep = report["deep"][model]
        put("deep", {**deep["C4_union"], "second_look_ms_per_photo": deep["C4_union"]["ms_per_photo"]})
        out["faces.photo.v2.deep.over_careful"] = deep["C4_union"]["paired_over_C2"]
        out["faces.photo.v2.deep.verdict"] = deep["dm6"]
    out["faces.photo.v2.machine"] = report["recorded_on"]
    out["faces.photo.v2.truth_sha256"] = report["truth_sha256"]
    return out


def deep_beats_careful(gain_points: float, mcnemar_p: float) -> bool:
    """DM6: deep earns its cost only if it hides at least 1.0 point more than careful, paired p < 0.05; otherwise it ships as the audit mode."""
    return gain_points >= 1.0 and mcnemar_p < 0.05


def by_source(result: Dict[str, object], photos: Dict[str, dict]) -> Dict[str, object]:
    """Hidden by default per source of photos: the v1 sets (truth made of YuNet proposals: an optimistic bound that favours the control) and
    WIDER FACE (human truth, independent of every model) must never be pooled into one figure without saying so."""
    out: Dict[str, object] = {}
    for source in sorted({k.split("/")[0] for k in photos}):
        faces = {i: v for i, v in result["faces"].items() if i.startswith(source + "/")}
        n = len(faces)
        any_ = sum(1 for v in faces.values() if v[1])
        out[source] = {
            "photos": sum(1 for k in photos if k.startswith(source + "/")),
            "faces": n,
            "hidden_default_pct": round(100.0 * any_ / n, 2) if n else None,
            "hidden_default_ci95_pct": wilson(any_, n),
            "false_hidden_events": sum(v for k, v in result["false"].items() if k.startswith(source + "/")),
        }
    return out


def choose_h_hi(
    cell: Dict[str, Sequence],
    truths: Dict[str, dict],
    minimum: float = 0.98,
    grid: Sequence[float] = tuple(x / 100 for x in range(50, 100)),
) -> Dict[str, object]:
    """`H_hi` is the lowest score whose heavy detections match a truth item with precision >= 98 % on dev. A detection matches an
    item when it overlaps a non-ignored truth box (IoU 0.3 or its centre inside); one that matches only an `ignore` box is neutral."""
    from anonymizer import geometry

    rows = []
    for name, per_zone in cell.items():
        boxes = truths[name]["boxes"]
        for zone in per_zone:
            for d in zone:
                hit = any(
                    not b.get("ignore") and (geometry.iou(d[:4], b["box"]) >= 0.3 or _centre_in(d[:4], b["box"]))
                    for b in boxes
                )
                neutral = not hit and any(b.get("ignore") and geometry.iou(d[:4], b["box"]) >= 0.3 for b in boxes)
                if not neutral:
                    rows.append((d[4], hit))
    table = []
    for cut in grid:
        kept = [hit for score, hit in rows if score >= cut]
        table.append({"cut": cut, "n": len(kept), "precision": round(sum(kept) / len(kept), 4) if kept else None})
    ok = [t for t in table if t["precision"] is not None and t["precision"] >= minimum and t["n"] >= 20]
    return {
        "h_hi": ok[0]["cut"] if ok else None,
        "table": table[:: max(1, len(table) // 12)],
        "rule": f"precision >= {minimum}, n >= 20",
    }


def _centre_in(box: Sequence[float], other: Sequence[float]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return other[0] <= cx <= other[2] and other[1] <= cy <= other[3]


def choose_h_lo(
    cell: Dict[str, Sequence],
    photos: Dict[str, dict],
    truths: Dict[str, dict],
    without_people: Sequence[str],
    h_hi: float,
    replayer: Replayer,
    base: Dict[str, object],
    allowed_per_100: float = 2.0,
    grid: Sequence[float] = tuple(x / 100 for x in range(50, 100)),
) -> Dict[str, object]:
    """`H_lo` is the lowest score for which the extra doubtful items add at most +2 false hidden photos per 100 photos without
    people on dev, over the light models alone (`base`)."""
    base_photos = sum(1 for p in without_people if base["false"][p])
    table = []
    chosen = None
    for cut in grid:
        if cut > h_hi:
            break
        th = HeavyThresholds(h_hi=h_hi, h_lo=cut)
        got = evaluate({p: photos[p] for p in without_people}, truths, cell, th, NOTHING, replayer)
        affected = sum(1 for p in without_people if got["false"][p])
        extra = 100.0 * (affected - base_photos) / len(without_people) if without_people else 0.0
        table.append({"cut": cut, "photos_affected": affected, "extra_per_100": round(extra, 2)})
        if chosen is None and extra <= allowed_per_100:
            chosen = cut
    return {
        "h_lo": chosen,
        "table": table[:: max(1, len(table) // 12)],
        "rule": f"extra false hidden photos <= {allowed_per_100} per 100",
    }


def choose_h_lo_deep(
    cell: Dict[str, Sequence],
    photos: Dict[str, dict],
    truths: Dict[str, dict],
    without_people: Sequence[str],
    h_hi: float,
    replayer: Replayer,
    base: Dict[str, object],
    seconds: Dict[str, float],
    allowed_per_100: float = 2.0,
    grid: Sequence[float] = tuple(x / 100 for x in range(50, 100)),
) -> Dict[str, object]:
    """The same rule as `choose_h_lo` for the deep sweep: the heavy model looks at the whole picture, so it meets far more false faces than in a
    zone, and the cut that was right for zones is not the cut for deep."""
    base_photos = sum(1 for p in without_people if base["false"][p])
    sub = {p: photos[p] for p in without_people}
    table, chosen = [], None
    for cut in grid:
        if cut > h_hi:
            break
        got = evaluate_deep(sub, truths, cell, HeavyThresholds(h_hi=h_hi, h_lo=cut), False, replayer, seconds)
        affected = sum(1 for p in without_people if got["false"][p])
        extra = 100.0 * (affected - base_photos) / len(without_people) if without_people else 0.0
        table.append({"cut": cut, "photos_affected": affected, "extra_per_100": round(extra, 2)})
        if chosen is None and extra <= allowed_per_100:
            chosen = cut
    return {
        "h_lo": chosen,
        "table": table[:: max(1, len(table) // 12)],
        "rule": f"extra false hidden photos <= {allowed_per_100} per 100",
    }


def choose_max_zones(photos: Dict[str, dict], coverage_share: float = 0.95, step: int = 5) -> Dict[str, object]:
    """SL5: the smallest cap (a multiple of `step`) that lets `coverage_share` of the dev photos be visited whole."""
    counts = sorted(len(p["zones"]) for p in photos.values())
    if not counts:
        return {"max_zones": None, "zones_per_photo": []}
    need = counts[min(len(counts) - 1, math.ceil(coverage_share * len(counts)) - 1)]
    return {
        "max_zones": int(math.ceil(max(need, 1) / step) * step),
        "zones_per_photo": {"median": counts[len(counts) // 2], "p95": need, "max": counts[-1], "n": len(counts)},
    }


def veto_study(
    cell: Dict[str, Sequence], photos: Dict[str, dict], truths: Dict[str, dict], th: HeavyThresholds
) -> Dict[str, object]:
    """PF-V: among the doubtful zones the heavy model found nothing in, how many were a real face? The veto of a doubtful item
    stays off unless this is <= 1 % with an upper 95 % bound <= 3 % on n >= 100 doubtful zones."""
    from anonymizer import geometry

    n = real = 0
    for name, per_zone in cell.items():
        photo = photos[name]
        items = look_items(photo["items"])
        for zone, dets in zip(photo["zones"], per_zone, strict=False):
            if zone["why"] != 1 or zone["item"] is None:
                continue
            item = items[zone["item"]]
            if any(d[4] >= th.h_lo for d in dets):
                continue
            n += 1
            real += any(not b.get("ignore") and geometry.iou(item.box, b["box"]) >= 0.3 for b in truths[name]["boxes"])
    upper = wilson(real, n)[1]
    return {
        "doubtful_zones_with_nothing_found": n,
        "real_faces": real,
        "real_pct": round(100.0 * real / n, 2) if n else None,
        "upper95_pct": upper,
        "veto_allowed": bool(n >= 100 and real / n <= 0.01 and upper is not None and upper <= 3.0),
        "rule": "n >= 100, share <= 1 %, upper 95 % bound <= 3 %",
    }


__all__ = [
    "NOTHING",
    "Replayer",
    "choose_h_hi",
    "choose_h_lo",
    "choose_max_zones",
    "evaluate",
    "mcnemar_exact",
    "paired",
    "summary",
    "veto_study",
]
