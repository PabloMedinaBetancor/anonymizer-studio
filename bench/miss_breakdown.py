"""Why the anonymizer misses the faces it misses: a post-hoc breakdown of the sealed faces read (registered claims C1-C4), per truth face.

The sealed read (`bench.sealed_measure.faces_v2`) stored totals, not which face each mode missed. This script runs the SAME code path on the same
photos and truth with the frozen engine (nothing of the engine, the thresholds or the measurement is changed or imported-and-patched), for the modes
fast, careful and deep, and keeps one record per labelled face: its size (and, outside WIDER, its box), whether each mode hid it by the rule of the read (>= 95 % of the box
covered by the blurred zones, the engine's margin included), whether the blur was certain or "unsure", and the zone that came closest. The tables are
then computed from that log alone, and `check_totals` compares the log with the totals of `results/sealed-v2.json`: if they differ, the breakdown does
not describe the published figures and nothing from it may be quoted.

WIDER FACE is research-only data and its annotation is not redistributed: `anonymise_wider` runs at the end of the log, so the WIDER rows of the written log have
no photo file name, no box and no picture size, carry opaque ids (`wider#0001`, ...) and are in an order taken from their own values. What the tables need stays:
the face height, the five attribute codes and what each mode did.

    python3 -s -m bench.miss_breakdown log    --out bench/results/miss-breakdown-log.json --wider-dir <WIDER subset> [--parts blind,wider,set-a,set-b]
    python3 -s -m bench.miss_breakdown tables --log bench/results/miss-breakdown-log.json --out bench/results/miss-breakdown.json [--wider-split <zip>]

It reads the sealed keys of the split only because the brief of the post-hoc breakdown asks for it (`--post-hoc`); it changes no published figure.
Numbers only: no crop or picture of a face is written by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile
from typing import Dict, List, Optional, Sequence

import cv2

from . import bench_cells as bc
from . import bench_faces as bf
from . import read_sealed as rs
from . import sealed_measure as sm
from .cells_eval import wilson

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SEALED_RESULT = os.path.join(RESULTS, "sealed-v2.json")
LOG_MODES = ("fast", "careful", "deep")
INDEPENDENT = ("blind", "wider")  # the parts that are the headline of the read (their truth is not the engine's)
PARTS = ("blind", "wider", "set-a", "set-b")
HEIGHT_BINS_PX = ((0, 10), (10, 20), (20, 40), (40, 80), (80, 10**9))
HEIGHT_BINS_FRAC = (
    (0.0, 0.01),
    (0.01, 0.02),
    (0.02, 0.04),
    (0.04, 0.08),
    (0.08, 10.0),
)  # face height as a share of the picture height
CUTS_PX = (20, 30, 40, 60, 80)  # "smaller than X px" statements
WIDER_ATTRIBUTES = {
    "blur": {0: "clear", 1: "normal blur", 2: "heavy blur"},
    "occlusion": {0: "none", 1: "partial", 2: "heavy"},
    "pose": {0: "typical", 1: "atypical"},
    "illumination": {0: "normal", 1: "extreme"},
    "expression": {0: "typical", 1: "exaggerated"},
}


# the log: the faces of the read, one record each


def _wider_attributes(split_zip: str) -> Dict[str, List[dict]]:
    """{"<folder>__<file>": [attributes per face in the order of the annotation file]} from the local copy of the WIDER FACE annotation (nothing is fetched)."""
    if not split_zip or not os.path.isfile(split_zip):
        return {}
    with zipfile.ZipFile(split_zip) as z:
        name = next(n for n in z.namelist() if n.endswith("wider_face_val_bbx_gt.txt"))
        lines = z.read(name).decode().splitlines()
    out: Dict[str, List[dict]] = {}
    i = 0
    while i < len(lines):
        fname = lines[i].strip()
        n = int(lines[i + 1])
        rows = lines[i + 2 : i + 2 + max(n, 1)]
        i += 2 + max(n, 1)
        faces = []
        for row in rows[:n]:
            v = [int(x) for x in row.split()]
            faces.append(
                {
                    "box": [v[0], v[1], v[0] + v[2], v[1] + v[3]],
                    "blur": v[4],
                    "expression": v[5],
                    "illumination": v[6],
                    "invalid": v[7],
                    "occlusion": v[8],
                    "pose": v[9],
                }
            )
        out[fname.replace("/", "__")] = faces
    return out


def _round_box(box: Sequence[float]) -> List[float]:
    return [round(float(v), 1) for v in box]


def anonymise_wider(faces: Sequence[dict]) -> List[dict]:
    """The log without the annotation of WIDER FACE (research data, not redistributed). The WIDER faces lose the photo's file name, the box, the box width and the picture size, get opaque ids
    (`wider#0001`, ...) and opaque photo labels (`wider-photo-001`, ...), and are put in an order taken from their own values (height, attributes, results; ties by a hash of the old id), so
    that neither the names nor the order of the annotation file can be recovered. What the published tables need stays: the height in px and as a share of the picture, the five attribute
    codes and what each mode did. The result does not depend on the order of the input. The faces of the other parts are returned as they are, in their order."""
    others = [f for f in faces if f["part"] != "wider"]

    def sort_key(f: dict):
        w = f.get("wider") or {}
        return (
            f["height_px"],
            f["height_frac"],
            [w.get(a, -1) for a in WIDER_ATTRIBUTES],
            json.dumps(f["modes"], sort_keys=True),
            hashlib.sha256(f["id"].encode("utf-8")).hexdigest(),
        )

    photos: Dict[str, str] = {}
    out: List[dict] = []
    for n, f in enumerate(sorted((f for f in faces if f["part"] == "wider"), key=sort_key), 1):
        label = photos.setdefault(f["photo"], f"wider-photo-{len(photos) + 1:03d}")
        rec = {
            "id": f"wider#{n:04d}",
            "part": "wider",
            "photo": label,
            "height_px": f["height_px"],
            "height_frac": f["height_frac"],
            "modes": f["modes"],
        }
        if "wider" in f:
            rec["wider"] = f["wider"]
        out.append(rec)
    return others + out


def log_faces(
    keys: Sequence[str],
    modes: Sequence[str] = LOG_MODES,
    wider_dir: str = "",
    split_path: str = rs.SPLIT,
    split_zip: str = "",
    allow_sealed: bool = False,
    say=print,
) -> dict:
    """One record per counted truth face of the photos `keys`, with what each of `modes` did to it. Same loop as `sealed_measure.faces_v2`: the photo goes through the public API once per
    mode (`Options(plates=False, ...)`, `unsure="blur"` by default), the zones are the blurred face zones with the engine's photo margin, a face is hidden by default at >= 95 % of its box covered."""
    import anonymizer
    from anonymizer import api
    from anonymizer import contract as c
    from anonymizer.contract import Verdict

    keys = sorted(k for k in keys if k.split("/", 1)[0] in PARTS)
    if not allow_sealed:
        rs.refuse_sealed(keys, split_path)
    _, hashes = sm.read_split(split_path)
    truths = {k: sm._face_truth(k, wider_dir) for k in keys}
    wider_attrs = _wider_attributes(split_zip)
    faces: Dict[str, dict] = {}
    for mode in modes:
        kwargs, modules = sm._face_cell(mode)
        started = time.time()
        with (
            tempfile.TemporaryDirectory(prefix="miss-breakdown-") as tmp,
            anonymizer.Anonymizer(anonymizer.Options(plates=False, **kwargs), modules=modules) as anon,
        ):
            for n, key in enumerate(keys, 1):
                bgr = bc.read_image(key, hashes, wider_dir)
                height, width = bgr.shape[:2]
                path = os.path.join(tmp, "p.png")
                cv2.imwrite(path, bgr)
                analysis = anon.analyze(path)
                zones, specs = api._photo_zones(analysis, anon.options)
                face_zones = [z for z in zones if z.kind == c.FACE]
                wrapped = [
                    (z, sm._Zone(bf.with_margin(z.box, specs[z.kind].margin, width, height))) for z in face_zones
                ]
                blurred = [w for z, w in wrapped if z.blur]
                sure = [w for z, w in wrapped if z.blur and z.judgement.verdict is Verdict.BLUR]
                kind, name = key.split("/", 1)
                for i, box in enumerate(truths[key]["boxes"]):
                    if box.get("ignore"):
                        continue
                    fid = f"{key}#{i}"
                    b = box["box"]
                    rec = faces.setdefault(
                        fid,
                        {
                            "id": fid,
                            "part": kind,
                            "photo": key,
                            "box": _round_box(b),
                            "photo_w": width,
                            "photo_h": height,
                            "height_px": round(float(b[3] - b[1]), 1),
                            "width_px": round(float(b[2] - b[0]), 1),
                            "height_frac": round(float(b[3] - b[1]) / height, 5),
                            "modes": {},
                        },
                    )
                    if kind == "wider":
                        attrs = wider_attrs.get(name, [])
                        if i < len(attrs) and [int(v) for v in attrs[i]["box"]] == [int(v) for v in b]:
                            rec["wider"] = {k: attrs[i][k] for k in WIDER_ATTRIBUTES}
                    share = bf.coverage(
                        b, blurred, False
                    )  # the share of the face box inside the blurred zones: the read's rule is share >= 95 %
                    share_ellipse = bf.coverage(b, blurred, True)
                    hidden = share >= bf.COVERED
                    certain = bf.coverage(b, sure, False) >= bf.COVERED
                    ellipse = share_ellipse >= bf.COVERED
                    best = None
                    touching = 0
                    for z, w in wrapped:
                        if bf.touches(w, b):
                            touching += 1
                        single = bf.coverage(b, [w], False)
                        if single > 0 and (best is None or single > best["cover"]):
                            best = {
                                "cover": round(single, 3),
                                "verdict": z.judgement.verdict.value,
                                "found_by": z.found_by,
                                "score": round(float(z.score), 3),
                                "second_score": None if z.second_score is None else round(float(z.second_score), 3),
                                "blur": bool(z.blur),
                            }
                    rec["modes"][mode] = {
                        "hidden": hidden,
                        "certain": certain,
                        "ellipse": ellipse,
                        "share": round(share, 4),
                        "share_ellipse": round(share_ellipse, 4),
                        "touching": touching,
                        "best": best,
                    }
                if n % 25 == 0:
                    say(f"{mode}: {n}/{len(keys)} photos, {time.time() - started:.0f} s")
        say(f"{mode}: done, {time.time() - started:.0f} s")
    return {
        "what": "per-face log of the sealed faces read, post-hoc, same code path as sealed_measure.faces_v2; the WIDER FACE rows have opaque ids and no file name, box or picture size",
        "modes": list(modes),
        "photos": len(keys),
        "machine": sm._machine(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "faces": anonymise_wider(list(faces.values())),
    }


# the tables: everything below reads the log only


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(hits: int, n: int) -> Optional[float]:
    return round(100.0 * hits / n, 2) if n else None


def _mode_cell(faces: Sequence[dict], mode: str, all_misses: int) -> dict:
    n = len(faces)
    hidden = sum(1 for f in faces if f["modes"][mode]["hidden"])
    miss = n - hidden
    return {
        "hidden": hidden,
        "missed": miss,
        "pct": _pct(hidden, n),
        "wilson95": wilson(hidden, n),
        "share_of_all_misses_pct": _pct(miss, all_misses),
    }


def _bins(faces: Sequence[dict], modes: Sequence[str], edges, key: str) -> List[dict]:
    total_miss = {m: sum(1 for f in faces if not f["modes"][m]["hidden"]) for m in modes}
    rows = []
    for lo, hi in edges:
        sub = [f for f in faces if lo <= f[key] < hi]
        row = {"from": lo, "to": hi if hi < 10**8 else None, "faces": len(sub)}
        for m in modes:
            row[m] = _mode_cell(sub, m, total_miss[m])
        rows.append(row)
    return rows


def _cuts(faces: Sequence[dict], modes: Sequence[str]) -> List[dict]:
    """For a cut X: how many of the misses of each mode are under X px tall, and what the mode hides on the faces at least X px tall."""
    rows = []
    for cut in CUTS_PX:
        small = [f for f in faces if f["height_px"] < cut]
        big = [f for f in faces if f["height_px"] >= cut]
        row = {"cut_px": cut, "faces_small": len(small), "faces_at_least": len(big)}
        for m in modes:
            misses = [f for f in faces if not f["modes"][m]["hidden"]]
            small_miss = sum(1 for f in misses if f["height_px"] < cut)
            big_hidden = sum(1 for f in big if f["modes"][m]["hidden"])
            row[m] = {
                "misses": len(misses),
                "misses_under_cut": small_miss,
                "misses_under_cut_pct": _pct(small_miss, len(misses)),
                "hidden_on_small_pct": _pct(sum(1 for f in small if f["modes"][m]["hidden"]), len(small)),
                "hidden_at_least_cut": big_hidden,
                "hidden_at_least_cut_pct": _pct(big_hidden, len(big)),
                "hidden_at_least_cut_wilson95": wilson(big_hidden, len(big)),
            }
        rows.append(row)
    return rows


def _attributes(faces: Sequence[dict], modes: Sequence[str]) -> dict:
    wider = [f for f in faces if f.get("wider")]
    out: dict = {"faces": len(wider)}
    for attr, names in WIDER_ATTRIBUTES.items():
        total_miss = {m: sum(1 for f in wider if not f["modes"][m]["hidden"]) for m in modes}
        rows = []
        for value, label in names.items():
            sub = [f for f in wider if f["wider"][attr] == value]
            row = {"value": value, "label": label, "faces": len(sub)}
            for m in modes:
                row[m] = _mode_cell(sub, m, total_miss[m])
            rows.append(row)
        out[attr] = rows
    return out


def _causes(faces: Sequence[dict], modes: Sequence[str]) -> dict:
    """What happened to each missed face: no face zone touched it ("not found"), or a zone touched it but covered under 95 % of it ("found, covered partly"), split by how sure the engine was.
    And what the hidden faces were hidden by: a sure zone or an unsure one."""
    out = {}
    for m in modes:
        misses = [f for f in faces if not f["modes"][m]["hidden"]]
        not_found = [f for f in misses if f["modes"][m]["best"] is None]
        partial = [f for f in misses if f["modes"][m]["best"] is not None]
        partial_by_verdict: Dict[str, int] = {}
        for f in partial:
            v = f["modes"][m]["best"]["verdict"]
            partial_by_verdict[v] = partial_by_verdict.get(v, 0) + 1
        covers = sorted(f["modes"][m]["best"]["cover"] for f in partial)
        hidden = [f for f in faces if f["modes"][m]["hidden"]]
        out[m] = {
            "misses": len(misses),
            "not_found": len(not_found),
            "not_found_median_height_px": _median([f["height_px"] for f in not_found]),
            "found_but_covered_partly": len(partial),
            "found_but_covered_partly_by_verdict": partial_by_verdict,
            "found_but_covered_partly_median_height_px": _median([f["height_px"] for f in partial]),
            "partial_cover_median": covers[len(covers) // 2] if covers else None,
            "misses_unsure_and_not_hidden": sum(
                1 for f in partial if f["modes"][m]["best"]["verdict"] == "check" and not f["modes"][m]["best"]["blur"]
            ),
            "hidden": len(hidden),
            "hidden_only_as_unsure": sum(1 for f in hidden if not f["modes"][m]["certain"]),
            "hidden_certain": sum(1 for f in hidden if f["modes"][m]["certain"]),
        }
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    s = sorted(values)
    return round(float(s[len(s) // 2]), 1) if s else None


def _paired_rescue(faces: Sequence[dict]) -> dict:
    """The misses of careful that another mode hides (and the other way round): what the extra levels bring, by size."""
    out = {}
    for base, better in (("fast", "careful"), ("careful", "deep")):
        if base not in faces[0]["modes"] or better not in faces[0]["modes"]:
            continue
        gained = [f for f in faces if f["modes"][better]["hidden"] and not f["modes"][base]["hidden"]]
        lost = [f for f in faces if f["modes"][base]["hidden"] and not f["modes"][better]["hidden"]]
        out[f"{better}_over_{base}"] = {
            "gained": len(gained),
            "lost": len(lost),
            "gained_median_height_px": _median([f["height_px"] for f in gained]),
        }
    return out


def _hard(face: dict) -> bool:
    """A WIDER face with a recorded hindrance: heavy blur, heavy occlusion or an atypical pose."""
    w = face["wider"]
    return w["blur"] == 2 or w["occlusion"] == 2 or w["pose"] == 1


HARD_EDGES = ((0, 20), (20, 40), (40, 10**9))


def _hardness(faces: Sequence[dict], modes: Sequence[str]) -> List[dict]:
    """WIDER faces split by size (under 20 px, 20-40, 40 and over) and by whether WIDER records a hindrance (heavy blur, heavy occlusion, atypical pose): is the size the whole story?"""
    wider = [f for f in faces if f.get("wider")]
    rows = []
    for hard in (False, True):
        for lo, hi in HARD_EDGES:
            sub = [f for f in wider if _hard(f) == hard and lo <= f["height_px"] < hi]
            row = {"hindrance": hard, "from": lo, "to": hi if hi < 10**8 else None, "faces": len(sub)}
            for m in modes:
                row[m] = _mode_cell(sub, m, sum(1 for f in wider if not f["modes"][m]["hidden"]))
            rows.append(row)
    return rows


COVER_EDGES = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.8), (0.8, 0.95))


def _loose(faces: Sequence[dict], modes: Sequence[str]) -> dict:
    """How far from hidden is a miss: the share of the labelled face inside the blurred zones, for the faces under the 95 % rule, and what the figure would be under a looser rule. Needs the `share` field of the log."""
    out = {}
    for m in modes:
        if "share" not in faces[0]["modes"][m]:
            continue
        misses = [f for f in faces if not f["modes"][m]["hidden"]]
        bins = []
        for lo, hi in COVER_EDGES:
            bins.append({"from": lo, "to": hi, "misses": sum(1 for f in misses if lo <= f["modes"][m]["share"] < hi)})
        n = len(faces)
        out[m] = {
            "faces": n,
            "misses": len(misses),
            "bins": bins,
            "at_least_50_pct": _pct(sum(1 for f in faces if f["modes"][m]["share"] >= 0.5), n),
            "at_least_80_pct": _pct(sum(1 for f in faces if f["modes"][m]["share"] >= 0.8), n),
            "at_least_95_pct": _pct(sum(1 for f in faces if f["modes"][m]["share"] >= 0.95), n),
            "untouched": sum(1 for f in misses if f["modes"][m]["share"] == 0),
            "median_share_of_misses": _median_float([f["modes"][m]["share"] for f in misses]),
        }
    return out


def _median_float(values: Sequence[float]) -> Optional[float]:
    s = sorted(values)
    return round(float(s[len(s) // 2]), 3) if s else None


def _per_photo(faces: Sequence[dict], modes: Sequence[str]) -> List[dict]:
    rows: Dict[str, dict] = {}
    for f in faces:
        row = rows.setdefault(
            f["photo"], {"photo": f["photo"], "photo_size": [f["photo_w"], f["photo_h"]], "faces": 0, "heights": []}
        )
        row["faces"] += 1
        row["heights"].append(f["height_px"])
        for m in modes:
            row[m] = row.get(m, 0) + (1 if f["modes"][m]["hidden"] else 0)
    out = []
    for row in sorted(rows.values(), key=lambda r: r["photo"]):
        heights = sorted(row.pop("heights"))
        row["median_height_px"] = round(heights[len(heights) // 2], 1)
        out.append(row)
    return out


TAGS = {
    "small": "small or far: under 40 px, upright and in the light, nothing else wrong",
    "profile": "turned: profile, three-quarter, looking down or the back of the head",
    "occluded": "partly hidden by a hand, hair, a hood, another person or a rail",
    "dark": "dark or grainy scene (dusk, shade)",
    "blur": "motion blur or out of focus",
    "screen": "face on a screen, poster or reflection",
    "other": "other",
}


def _blind_tags(log: dict, tags: Optional[dict], modes: Sequence[str]) -> Optional[dict]:
    """The by-eye tags of the missed faces of the 30 blind photos (`miss-blind-tags.json`, written by looking at the crops; judgements, not measurements): per mode, how many misses carry each tag, as the main cause and as any cause."""
    if not tags:
        return None
    given = tags["faces"]
    out: dict = {"tagged": len(given), "tags": TAGS, "modes": {}}
    for m in modes:
        misses = [f for f in log["faces"] if f["part"] == "blind" and not f["modes"][m]["hidden"]]
        untagged = [f["id"] for f in misses if f["id"] not in given]
        main = {t: 0 for t in TAGS}
        anyc = {t: 0 for t in TAGS}
        for f in misses:
            g = given.get(f["id"])
            if not g:
                continue
            main[g["primary"]] += 1
            for t in g["tags"]:
                anyc[t] += 1
        out["modes"][m] = {"misses": len(misses), "untagged": untagged, "main": main, "any": anyc}
    return out


def headline(t: dict) -> dict:
    """The few figures the README quotes, as flat numbers a `{marker}` can hold."""
    ind = t["sets"]["independent"]
    blind = t["sets"]["blind"]
    wider = t["sets"]["wider"]
    out: dict = {}
    for m in ("fast", "careful"):
        cuts = {r["cut_px"]: r[m] for r in ind["cuts"]}
        out[m] = {
            "misses": ind["overall"][m]["missed"],
            "faces": ind["faces"],
            "misses_under_20_px_pct": cuts[20]["misses_under_cut_pct"],
            "misses_under_30_px_pct": cuts[30]["misses_under_cut_pct"],
            "misses_under_40_px_pct": cuts[40]["misses_under_cut_pct"],
            "hidden_at_least_20_px_pct": cuts[20]["hidden_at_least_cut_pct"],
            "hidden_at_least_40_px_pct": cuts[40]["hidden_at_least_cut_pct"],
            "hidden_at_least_40_px_wilson95": cuts[40]["hidden_at_least_cut_wilson95"],
            "hidden_at_least_80_px_pct": cuts[80]["hidden_at_least_cut_pct"],
            "hidden_under_20_px_pct": cuts[20]["hidden_on_small_pct"],
            "not_found": ind["causes"][m]["not_found"],
            "found_but_covered_partly": ind["causes"][m]["found_but_covered_partly"],
        }
        bcuts = {r["cut_px"]: r[m] for r in blind["cuts"]}
        out[m]["blind_hidden_pct"] = blind["overall"][m]["pct"]
        out[m]["blind_faces_under_40_px"] = next(r["faces_small"] for r in blind["cuts"] if r["cut_px"] == 40)
        out[m]["blind_hidden_under_40_px_pct"] = bcuts[40]["hidden_on_small_pct"]
        hind = wider["by_hindrance"]
        n_hind = sum(r["faces"] for r in hind if r["hindrance"])
        n_clean = sum(r["faces"] for r in hind if not r["hindrance"])
        out[m]["hidden_with_hindrance_pct"] = _pct(sum(r[m]["hidden"] for r in hind if r["hindrance"]), n_hind)
        out[m]["hidden_without_hindrance_pct"] = _pct(sum(r[m]["hidden"] for r in hind if not r["hindrance"]), n_clean)
        out[m]["wider_misses_with_hindrance_pct"] = _pct(
            sum(r[m]["missed"] for r in hind if r["hindrance"]), sum(r[m]["missed"] for r in hind)
        )
        if t.get("blind_tags"):
            tg = t["blind_tags"]["modes"][m]
            out[m]["blind_misses"] = tg["misses"]
            out[m]["blind_misses_small_only"] = tg["main"]["small"]
            out[m]["blind_misses_turned_or_covered"] = tg["main"]["profile"] + tg["main"]["occluded"]
            out[m]["blind_misses_dark_any"] = tg["any"]["dark"]
        if "loose" in ind and m in ind["loose"]:
            out[m]["hidden_at_least_80_share_pct"] = ind["loose"][m]["at_least_80_pct"]
            out[m]["hidden_at_least_50_share_pct"] = ind["loose"][m]["at_least_50_pct"]
    return out


def tables(log: dict, sealed: Optional[dict] = None, tags: Optional[dict] = None) -> dict:
    """Everything the MISSES write-up quotes, from the per-face log alone."""
    modes = [m for m in LOG_MODES if m in log["modes"]]
    all_faces = log["faces"]
    sets = {
        "independent": [f for f in all_faces if f["part"] in INDEPENDENT],
        "blind": [f for f in all_faces if f["part"] == "blind"],
        "wider": [f for f in all_faces if f["part"] == "wider"],
        "set-a": [f for f in all_faces if f["part"] == "set-a"],
    }
    out: dict = {"modes": modes, "totals": check_totals(log, sealed) if sealed else None, "sets": {}}
    for name, faces in sets.items():
        if not faces:
            continue
        cell = {
            "faces": len(faces),
            "photos": len({f["photo"] for f in faces}),
            "overall": {m: _mode_cell(faces, m, sum(1 for f in faces if not f["modes"][m]["hidden"])) for m in modes},
            "by_height_px": _bins(faces, modes, HEIGHT_BINS_PX, "height_px"),
            "by_height_frac": _bins(faces, modes, HEIGHT_BINS_FRAC, "height_frac"),
            "cuts": _cuts(faces, modes),
            "causes": _causes(faces, modes),
            "rescue": _paired_rescue(faces),
        }
        loose = _loose(faces, modes)
        if loose:
            cell["loose"] = loose
        if name in ("independent", "wider"):
            cell["wider_attributes"] = _attributes(faces, modes)
            cell["by_hindrance"] = _hardness(faces, modes)
        if name == "blind":
            cell["by_photo"] = _per_photo(faces, modes)
        out["sets"][name] = cell
    # the misses of the 30 blind photos, one per line, for the by-eye tagging (ids and numbers only)
    blind_miss = {}
    for m in modes:
        blind_miss[m] = [
            {
                "id": f["id"],
                "height_px": f["height_px"],
                "height_frac": f["height_frac"],
                "box": f["box"],
                "photo_size": [f["photo_w"], f["photo_h"]],
                "best": f["modes"][m]["best"],
            }
            for f in sets["blind"]
            if not f["modes"][m]["hidden"]
        ]
    out["blind_misses"] = blind_miss
    # the misses of careful on faces at least 40 px tall, with their WIDER attributes: what is left when size is not the reason
    big = [f for f in sets["wider"] if f["height_px"] >= 40 and not f["modes"]["careful"]["hidden"]]
    out["blind_tags"] = _blind_tags(log, tags, modes)
    out["misses_headline"] = headline(out)
    out["wider_careful_misses_at_least_40px"] = [
        {
            "id": f["id"],
            "height_px": f["height_px"],
            "hindrance": _hard(f) if f.get("wider") else None,
            "best": f["modes"]["careful"]["best"],
        }
        for f in big
    ]
    return out


def check_totals(log: dict, sealed: dict) -> dict:
    """The log's totals against the sealed read's (`results.faces_v2.parts` and `independent_parts`): per mode and part, hidden faces and faces. `match` is True only if every one agrees."""
    ref = sealed["results"]["faces_v2"]
    modes = [m for m in LOG_MODES if m in log["modes"]]
    got: Dict[str, dict] = {}
    for part in PARTS + ("independent",):
        faces = [f for f in log["faces"] if (f["part"] in INDEPENDENT if part == "independent" else f["part"] == part)]
        got[part] = {
            m: {"hidden_by_default": sum(1 for f in faces if f["modes"][m]["hidden"]), "faces": len(faces)}
            for m in modes
        }
    certain = {m: sum(1 for f in log["faces"] if f["modes"][m]["certain"]) for m in modes}
    ellipse = {m: sum(1 for f in log["faces"] if f["modes"][m]["ellipse"]) for m in modes}
    rows = []
    ok = True
    for part in PARTS + ("independent",):
        stored = ref["independent_parts"] if part == "independent" else ref["parts"][part]
        for m in modes:
            a, b = got[part][m], stored[m]
            same = a["hidden_by_default"] == b["hidden_by_default"] and a["faces"] == b["faces"]
            ok = ok and same
            rows.append(
                {
                    "part": part,
                    "mode": m,
                    "log": a,
                    "sealed": {"hidden_by_default": b["hidden_by_default"], "faces": b["faces"]},
                    "same": same,
                }
            )
    cells = ref["cells"]
    for m in modes:
        n = len(log["faces"])
        sure_same = abs(100.0 * certain[m] / n - cells[m]["blurred_with_certainty_pct"]) < 0.006
        ellipse_same = abs(100.0 * ellipse[m] / n - cells[m]["hidden_by_default_ellipse_pct"]) < 0.006
        ok = ok and sure_same and ellipse_same
    return {"match": ok, "rows": rows, "certain": certain, "ellipse": ellipse}


# the document: its tables and key sentences are written from the tables above, never by hand

BEGIN, END = "<!-- BEGIN {} -->", "<!-- END {} -->"
MODE_NAMES = {"fast": "Fast", "careful": "Careful", "deep": "Deep"}


def _cell(row: dict, mode: str) -> str:
    c = row[mode]
    if not row["faces"]:
        return "-"
    lo, hi = c["wilson95"]
    return f"{c['pct']:.1f} % ({c['hidden']} of {row['faces']}; {lo:.1f}-{hi:.1f})"


def _range(row: dict, unit: str) -> str:
    lo, hi = row["from"], row["to"]
    if unit == "px":
        return f"under {hi} px" if lo == 0 else (f"{lo} px and over" if hi is None else f"{lo}-{hi} px")
    lo_p, hi_p = lo * 100, None if hi is None or hi >= 10 else hi * 100
    return f"under {hi_p:g} %" if lo == 0 else (f"{lo_p:g} % and over" if hi_p is None else f"{lo_p:g}-{hi_p:g} %")


def _md(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _size_table(rows: Sequence[dict], unit: str, modes: Sequence[str]) -> str:
    body = [
        [_range(r, unit), r["faces"]]
        + [_cell(r, m) for m in modes]
        + [f"{r['careful']['share_of_all_misses_pct']:.1f} %"]
        for r in rows
        if r["faces"]
    ]
    return _md(
        ["Face height" + (" (px)" if unit == "px" else " (share of the picture height)"), "Faces"]
        + [f"{MODE_NAMES[m]} hides" for m in modes]
        + ["Share of all of Careful's misses"],
        body,
    )


def render_blocks(t: dict) -> Dict[str, str]:
    """{block name: markdown} for every generated block of docs/MISSES.md."""
    modes = t["modes"]
    ind, blind, wider = t["sets"]["independent"], t["sets"]["blind"], t["sets"]["wider"]
    h = t["misses_headline"]
    c20, c30, c40, c80 = ({r["cut_px"]: r for r in ind["cuts"]}[x] for x in (20, 30, 40, 80))
    out: Dict[str, str] = {}
    cf, ff = h["careful"], h["fast"]
    out["answer"] = (
        f"- **Careful** (the default) leaves {cf['misses']} of {cf['faces']} faces on the independent parts without the 95 % cover the read asks for. "
        f"Of those {cf['misses']}, **{cf['misses_under_20_px_pct']:.1f} % are under 20 px tall, {cf['misses_under_30_px_pct']:.1f} % under 30 px and {cf['misses_under_40_px_pct']:.1f} % under 40 px**.\n"
        f"- On faces at least 20 px tall ({c20['faces_at_least']} faces) Careful hides {cf['hidden_at_least_20_px_pct']:.1f} %; at least 40 px ({c40['faces_at_least']}), "
        f"{cf['hidden_at_least_40_px_pct']:.1f} % (Wilson {cf['hidden_at_least_40_px_wilson95'][0]:.1f}-{cf['hidden_at_least_40_px_wilson95'][1]:.1f}); at least 80 px ({c80['faces_at_least']}), {cf['hidden_at_least_80_px_pct']:.1f} %. Under 20 px ({c20['faces_small']} faces) it hides {cf['hidden_under_20_px_pct']:.1f} %.\n"
        f"- **Fast**: of its {ff['misses']} misses, {ff['misses_under_20_px_pct']:.1f} % are under 20 px and {ff['misses_under_40_px_pct']:.1f} % under 40 px; on faces at least 40 px tall it hides {ff['hidden_at_least_40_px_pct']:.1f} %."
    )
    if t.get("totals"):
        rows = []
        for part in ("independent", "blind", "wider", "set-a"):
            row = [
                {
                    "independent": "independent parts (blind + WIDER)",
                    "blind": "blind photos",
                    "wider": "WIDER, sealed half",
                    "set-a": "set a",
                }[part]
            ]
            for m in modes:
                r = next(x for x in t["totals"]["rows"] if x["part"] == part and x["mode"] == m)
                row.append(
                    f"{r['log']['hidden_by_default']} of {r['log']['faces']} / {r['sealed']['hidden_by_default']} of {r['sealed']['faces']}"
                    + ("" if r["same"] else " DIFFERS")
                )
            rows.append(row)
        out["reproduction"] = _md(
            ["Faces hidden by default: this log / `sealed-v2.json`"] + [MODE_NAMES[m] for m in modes], rows
        )
    out["size-px"] = _size_table(ind["by_height_px"], "px", modes)
    out["size-frac"] = _size_table(ind["by_height_frac"], "frac", modes)
    out["cuts"] = _md(
        ["Cut", "Faces under it", "Faces at least that tall"]
        + [
            x
            for m in ("fast", "careful")
            for x in (f"{MODE_NAMES[m]}: misses under the cut", f"{MODE_NAMES[m]} hides at least that tall")
        ],
        [
            [f"{r['cut_px']} px", r["faces_small"], r["faces_at_least"]]
            + [
                x
                for m in ("fast", "careful")
                for x in (
                    f"{r[m]['misses_under_cut_pct']:.1f} % of {r[m]['misses']}",
                    f"{r[m]['hidden_at_least_cut_pct']:.1f} % ({r[m]['hidden_at_least_cut']} of {r['faces_at_least']})",
                )
            ]
            for r in ind["cuts"]
        ],
    )
    hind = wider["by_hindrance"]
    out["hindrance"] = _md(
        ["WIDER faces", "Face height", "Faces"] + [f"{MODE_NAMES[m]} hides" for m in modes],
        [
            ["with a hindrance" if r["hindrance"] else "no hindrance recorded", _range(r, "px"), r["faces"]]
            + [_cell(r, m) for m in modes]
            for r in hind
        ],
    )
    big_hind = sum(r["careful"]["missed"] for r in hind if r["hindrance"])
    all_hind = sum(r["careful"]["missed"] for r in hind)
    n_hind = sum(r["faces"] for r in hind if r["hindrance"])
    n_clean = sum(r["faces"] for r in hind if not r["hindrance"])
    clean_hidden = sum(r["careful"]["hidden"] for r in hind if not r["hindrance"])
    hard_hidden = sum(r["careful"]["hidden"] for r in hind if r["hindrance"])
    out["hindrance-summary"] = (
        f"- WIDER records for each face how blurred ({'/'.join(WIDER_ATTRIBUTES['blur'].values())}), how occluded and what pose it has. Call a face one **with a hindrance** if it is heavily blurred, heavily occluded or in an atypical pose: "
        f"{n_hind} of the {n_hind + n_clean} WIDER faces on the sealed part are.\n"
        f"- Careful hides {100.0 * hard_hidden / n_hind:.1f} % of the faces with a hindrance and {100.0 * clean_hidden / n_clean:.1f} % of the faces without, and {big_hind} of its {all_hind} misses on WIDER ({100.0 * big_hind / all_hind:.1f} %) are faces with a hindrance."
    )
    attr_rows = []
    for attr in ("blur", "occlusion", "pose"):
        for r in wider["wider_attributes"][attr]:
            attr_rows.append(
                [
                    attr,
                    r["label"],
                    r["faces"],
                    f"{r['careful']['pct']:.1f} % ({r['careful']['hidden']} of {r['faces']})",
                    f"{r['careful']['share_of_all_misses_pct']:.1f} %",
                ]
            )
    out["attributes"] = _md(
        ["WIDER attribute", "Value", "Faces", "Careful hides", "Share of Careful's misses on WIDER"], attr_rows
    )
    cz = ind["causes"]
    rows = []
    for m in modes:
        z = cz[m]
        rows.append(
            [
                MODE_NAMES[m],
                z["misses"],
                f"{z['not_found']} (median height {z['not_found_median_height_px']:g} px)",
                f"{z['found_but_covered_partly']} (median height {z['found_but_covered_partly_median_height_px']:g} px)",
                z["misses_unsure_and_not_hidden"],
                f"{z['hidden_only_as_unsure']} of {z['hidden']}",
            ]
        )
    out["causes"] = _md(
        [
            "Level",
            "Misses",
            "No face zone touches the face",
            "A face zone touches it but covers under 95 %",
            "Misses that were unsure and left unblurred",
            'Faces hidden only as "unsure"',
        ],
        rows,
    )
    if "loose" in ind:
        lo = ind["loose"]
        out["loose"] = _md(
            [
                "Level",
                "Hidden by the read's rule (at least 95 %)",
                "At least 80 % of the face blurred",
                "At least 50 % blurred",
                "Misses with nothing blurred on the face",
            ],
            [
                [
                    MODE_NAMES[m],
                    f"{lo[m]['at_least_95_pct']:.1f} %",
                    f"{lo[m]['at_least_80_pct']:.1f} %",
                    f"{lo[m]['at_least_50_pct']:.1f} %",
                    f"{lo[m]['untouched']} of {lo[m]['misses']}",
                ]
                for m in modes
            ],
        )
    out["blind-photos"] = _md(
        ["Photo", "Picture size", "Faces", "Median face height", "Fast hides", "Careful hides", "Deep hides"],
        [
            [
                r["photo"].split("/", 1)[1],
                f"{r['photo_size'][0]}x{r['photo_size'][1]}",
                r["faces"],
                f"{r['median_height_px']:g} px",
            ]
            + [f"{r[m]} of {r['faces']}" for m in modes]
            for r in blind["by_photo"]
        ],
    )
    bt = t["blind_tags"]
    if bt:
        rows = []
        for tag, text in bt["tags"].items():
            if any(bt["modes"][m]["any"][tag] for m in modes):
                rows.append(
                    [
                        text,
                        bt["modes"]["careful"]["main"][tag],
                        bt["modes"]["careful"]["any"][tag],
                        bt["modes"]["fast"]["any"][tag],
                        bt["modes"]["deep"]["any"][tag],
                    ]
                )
        out["blind-tags"] = _md(
            [
                "What the face looks like (by eye)",
                "Careful misses: main cause",
                "Careful misses: any cause",
                "Fast misses: any cause",
                "Deep misses: any cause",
            ],
            rows
            + [
                [
                    "**Misses in all**",
                    bt["modes"]["careful"]["misses"],
                    bt["modes"]["careful"]["misses"],
                    bt["modes"]["fast"]["misses"],
                    bt["modes"]["deep"]["misses"],
                ]
            ],
        )
    if bt:
        cm = bt["modes"]["careful"]
        out["blind-summary"] = (
            f"- Of the {cm['misses']} faces Careful misses on the blind photos, {cm['main']['profile']} are mainly turned (profile, three-quarter, looking down, the back of the head), {cm['main']['occluded']} partly hidden, "
            f"{cm['main']['dark']} mainly lost in a dark scene, {cm['main']['blur']} blurred and only {cm['main']['small']} just small; {cm['any']['dark']} of the {cm['misses']} sit in a dark or grainy scene.\n"
            f"- The blind faces are small ({blind['cuts'][2]['faces_small']} of {blind['faces']} are under 40 px) and Careful hides {blind['cuts'][2]['careful']['hidden_on_small_pct']:.1f} % of them, against "
            f"{next(r for r in hind if not r['hindrance'] and r['from'] == 20)['careful']['pct']:.1f} % for the WIDER faces of 20-40 px with no hindrance recorded: on these photos the small size comes with dim light, turned heads and people in front of people, so size is not the only thing at work."
        )
    if "loose" in ind:
        lo = ind["loose"]["careful"]
        out["loose-summary"] = (
            f"- Most misses are faces with nothing blurred on them: Careful leaves {lo['untouched']} of its {lo['misses']} misses "
            f"({100.0 * lo['untouched'] / lo['misses']:.1f} %) with no blur at all and the other {lo['misses'] - lo['untouched']} partly blurred. "
            f"Counting a face as hidden at 80 % of its box instead of 95 % would move Careful from {lo['at_least_95_pct']:.1f} % to {lo['at_least_80_pct']:.1f} %, and at half of its box to {lo['at_least_50_pct']:.1f} %."
        )
    d = ind["overall"]["deep"]
    out["deep"] = (
        f"Deep hides {d['pct']:.1f} % ({d['hidden']} of {ind['faces']}) of the independent faces; it leaves {d['missed']} misses, {ind['causes']['deep']['not_found']} of them with no face zone on them. "
        f"Of those {d['missed']}, {({r['cut_px']: r for r in ind['cuts']}[20])['deep']['misses_under_cut_pct']:.1f} % are under 20 px and {({r['cut_px']: r for r in ind['cuts']}[40])['deep']['misses_under_cut_pct']:.1f} % under 40 px. "
        f"It hides {({r['cut_px']: r for r in ind['cuts']}[40])['deep']['hidden_at_least_cut_pct']:.1f} % of the faces at least 40 px tall. Against Careful it hides {ind['rescue']['deep_over_careful']['gained']} faces more and {ind['rescue']['deep_over_careful']['lost']} fewer, "
        f"and the faces it gains have a median height of {ind['rescue']['deep_over_careful']['gained_median_height_px']:g} px: its extra effort goes to the smallest faces."
    )
    return out


def write_docs(path: str, blocks: Dict[str, str], check: bool = False) -> bool:
    """Replace the text between `<!-- BEGIN name -->` and `<!-- END name -->` of the document by the generated blocks. Returns True if the file already had them (with `check`, it writes nothing)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    new = text
    for name, body in blocks.items():
        b, e = BEGIN.format(name), END.format(name)
        if b not in new:
            continue
        head, rest = new.split(b, 1)
        _, tail = rest.split(e, 1)
        new = f"{head}{b}\n{body}\n{e}{tail}"
    if new != text and not check:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return new == text


# command line


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("log", help="run the modes on the photos and write the per-face log")
    a.add_argument("--out", required=True)
    a.add_argument("--wider-dir", default=os.environ.get("ANONYMIZER_WIDER_DIR", ""))
    a.add_argument(
        "--wider-split",
        default="",
        help="wider_face_split.zip, the local copy of WIDER's annotation (blur, occlusion, pose, illumination, expression)",
    )
    a.add_argument("--parts", default=",".join(PARTS))
    a.add_argument("--modes", default=",".join(LOG_MODES))
    a.add_argument(
        "--sealed",
        action="store_true",
        help="use the sealed keys (post-hoc breakdown of the sealed read) instead of the dev keys",
    )
    a.add_argument("--limit", type=int, default=0, help="only the first N photos (a trial)")
    b = sub.add_parser("tables", help="compute the tables from a log")
    b.add_argument("--log", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--sealed-result", default=SEALED_RESULT)
    b.add_argument(
        "--tags",
        default=os.path.join(RESULTS, "miss-blind-tags.json"),
        help="the by-eye tags of the missed blind faces",
    )
    c = sub.add_parser("docs", help="write the generated blocks of docs/MISSES.md from the tables")
    c.add_argument("--tables", default=os.path.join(RESULTS, "miss-breakdown.json"))
    c.add_argument("--doc", default=os.path.join(os.path.dirname(HERE), "docs", "MISSES.md"))
    c.add_argument("--check", action="store_true", help="exit 1 if the document is not what the tables give")
    args = ap.parse_args(argv)
    if args.cmd == "docs":
        with open(args.tables, encoding="utf-8") as f:
            tables_json = json.load(f)
        same = write_docs(args.doc, render_blocks(tables_json), args.check)
        print("up to date" if same else ("out of date" if args.check else f"written {args.doc}"))
        return 0 if (same or not args.check) else 1
    if args.cmd == "log":
        parts = tuple(args.parts.split(","))
        keys = [k for k in (rs.sealed_keys() if args.sealed else rs.dev_keys()) if k.split("/", 1)[0] in parts]
        if args.limit:
            keys = keys[: args.limit]
        log = log_faces(
            keys,
            tuple(args.modes.split(",")),
            wider_dir=args.wider_dir,
            split_zip=args.wider_split,
            allow_sealed=args.sealed,
        )
        log["keys"] = "sealed" if args.sealed else "dev"
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(log, f, separators=(",", ":"), sort_keys=True)
        print(f"{len(log['faces'])} faces, {log['photos']} photos -> {args.out}")
        return 0
    log = _load_json(args.log)
    sealed = (
        _load_json(args.sealed_result) if os.path.isfile(args.sealed_result) and log.get("keys") == "sealed" else None
    )
    tags = _load_json(args.tags) if os.path.isfile(args.tags) else None
    result = tables(log, sealed, tags)
    result["log"] = os.path.basename(args.log)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    if sealed:
        print("totals reproduce the sealed read:", result["totals"]["match"])
        return 0 if result["totals"]["match"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(_main())
