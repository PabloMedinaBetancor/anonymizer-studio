"""Compare the face truth of the author with a blind verifier's, box by box (the bias of the author's truth).

    python3 -s -m bench.compare_faces_truth --verifier V.json [--author A.json B.json] [--images LIST.txt]
                                            [--iou 0.3] [--details bench/results/faces-details.json] [--json OUT.json]

The author's truth (bench/truth/faces/set-a.json and set-b.json by default) was built from YuNet proposals and one
reviewer. The verifier labelled a sample without seeing proposals or labels. Only the images both name are compared
(with --images, only those in the list: `set-a/name.jpg` or `name.jpg` per line, `#` comments). Boxes are paired
greedily by IoU, best first, a pair needs IoU >= --iou (0.3: faces are small and a 2 px shift ruins the IoU of a
10 px box; the report also gives the concordance at 0.25 / 0.3 / 0.5). What it counts:

  agree      a pair that neither side marks `ignore`
  uncertain  a pair where one side or both mark `ignore`: the box is unclear, nobody is wrong
  only in A  a counted box of the author the verifier does not have: the author may have over-labelled
  only in V  a counted box of the verifier the author does not have: the author's denominator is too small
  ignored    unpaired `ignore` boxes, listed apart

The numbers that say how much to trust the author's truth, by box size (longest side: < 24, 24-48, > 48 px):
  author_precision = agree / (agree + only in A)   share of the author's counted faces the verifier also sees
  author_coverage  = agree / (agree + only in V)   share of the faces the verifier sees that the author counted
With --details, the misses of bench/results/faces-details.json inside the compared images are checked too: how many
the verifier confirms as faces (a miss nobody confirms is a questionable label, not a real leak). The exit status is
0 whatever the result: the report is read by a person.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set

from bench.compare_truth import iou, pair_boxes, read_image_list

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.join(HERE, "truth", "faces")
DEFAULT_AUTHOR = [os.path.join(TRUTH_DIR, "set-a.json"), os.path.join(TRUTH_DIR, "set-b.json")]
SIZE_BUCKETS = (("<24", 0, 24), ("24-48", 24, 48), (">48", 48, 10**9))
SENSITIVITY = (0.25, 0.3, 0.5)


def problems(doc: Dict) -> List[str]:
    """Why a truth document does not follow bench/truth/faces/README.md (empty when it does)."""
    found: List[str] = []
    images = doc.get("images")
    if not isinstance(images, dict):
        return ["no `images` object"]
    for name, entry in images.items():
        for i, b in enumerate(entry.get("boxes", [])):
            box = b.get("box")
            if not (isinstance(box, list) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]):
                found.append(f"{name}[{i}]: bad box {box}")
            if not isinstance(b.get("ignore"), bool):
                found.append(f"{name}[{i}]: `ignore` must be true or false")
    return found


def size_of(box: Sequence[float]) -> float:
    return max(box[2] - box[0], box[3] - box[1])


def bucket(box: Sequence[float]) -> str:
    s = size_of(box)
    return next(label for label, lo, hi in SIZE_BUCKETS if lo <= s < hi)


def merge(docs: List[Dict]) -> Dict:
    """Several truth documents (set a, set b) as one; image names are unique across the sets."""
    images: Dict[str, dict] = {}
    for d in docs:
        for name, entry in d["images"].items():
            if name in images:
                raise SystemExit(f"{name} is in two truth files")
            images[name] = entry
    return {"images": images}


def strip_set(entry: str) -> str:
    return os.path.basename(entry.strip())


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def compare(
    a: Dict,
    v: Dict,
    threshold: float = 0.3,
    only: Optional[Set[str]] = None,
    misses: Sequence[dict] = (),
    sensitivity: bool = True,
) -> Dict:
    names = sorted(set(a["images"]) & set(v["images"]))
    if only is not None:
        names = [n for n in names if n in only]
    zero = {"agree": 0, "uncertain": 0, "only_a": 0, "only_v": 0, "ignored_a": 0, "ignored_v": 0}
    total = dict(zero)
    by_size = {label: dict(zero) for label, _, _ in SIZE_BUCKETS}
    discrepancies: List[dict] = []
    confirmed: Dict[str, Optional[bool]] = {}
    for name in names:
        ba, bv = a["images"][name]["boxes"], v["images"][name]["boxes"]
        pairs, left_a, left_v = pair_boxes(ba, bv, threshold)

        def count(key: str, box: Sequence[float]) -> None:
            total[key] += 1
            by_size[bucket(box)][key] += 1

        for i, j, score in pairs:
            x, y = ba[i], bv[j]
            if x["ignore"] or y["ignore"]:
                count("uncertain", x["box"])
                if x["ignore"] != y["ignore"]:
                    discrepancies.append(_row("ignore_differs", 2, name, x, y, score))
            else:
                count("agree", x["box"])
                if score < 0.5:
                    discrepancies.append(_row("poor_overlap", 4, name, x, y, score))
        for i in left_a:
            if ba[i]["ignore"]:
                count("ignored_a", ba[i]["box"])
            else:
                count("only_a", ba[i]["box"])
                discrepancies.append(_row("only_in_author", 1, name, ba[i], None, 0.0))
        for j in left_v:
            if bv[j]["ignore"]:
                count("ignored_v", bv[j]["box"])
            else:
                count("only_v", bv[j]["box"])
                discrepancies.append(_row("only_in_verifier", 1, name, None, bv[j], 0.0))
        for m in misses:
            if strip_set(m["image"]) == name:
                best = max(((iou(m["box"], y["box"]), j) for j, y in enumerate(bv)), default=(0.0, -1))
                confirmed[f"{name}{m['box']}"] = best[0] >= threshold and not bv[best[1]]["ignore"]
    discrepancies.sort(key=lambda r: (r["rank"], -r["size"], r["image"]))
    report = {
        "images": len(names),
        "iou": threshold,
        **total,
        "author_precision": _ratio(total["agree"], total["agree"] + total["only_a"]),
        "author_coverage": _ratio(total["agree"], total["agree"] + total["only_v"]),
        "by_size": {
            k: {
                **c,
                "author_precision": _ratio(c["agree"], c["agree"] + c["only_a"]),
                "author_coverage": _ratio(c["agree"], c["agree"] + c["only_v"]),
            }
            for k, c in by_size.items()
        },
        "discrepancies": discrepancies,
    }
    report["sensitivity"] = {}
    for t in SENSITIVITY if sensitivity else ():
        if t != threshold:
            r = compare(a, v, t, only, sensitivity=False)
            report["sensitivity"][str(t)] = {
                "author_precision": r["author_precision"],
                "author_coverage": r["author_coverage"],
            }
    if misses:
        report["misses_in_sample"] = len(confirmed)
        report["misses_confirmed_by_verifier"] = sum(1 for ok in confirmed.values() if ok)
    return report


def _row(kind: str, rank: int, image: str, x: Optional[dict], y: Optional[dict], score: float) -> dict:
    ref = (x or y)["box"]
    return {
        "kind": kind,
        "rank": rank,
        "image": image,
        "box_author": None if x is None else x["box"],
        "box_verifier": None if y is None else y["box"],
        "iou": round(score, 3),
        "size": size_of(ref),
        "ignore_author": None if x is None else x["ignore"],
        "ignore_verifier": None if y is None else y["ignore"],
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f} %"


def render(r: Dict, show: int = 40) -> str:
    lines = [
        f"images compared: {r['images']}  (pairing at IoU >= {r['iou']})",
        f"agree {r['agree']} | uncertain (ignore on a side) {r['uncertain']} | only in author {r['only_a']} | "
        f"only in verifier {r['only_v']} | ignored boxes author {r['ignored_a']}, verifier {r['ignored_v']}",
        f"author precision {_pct(r['author_precision'])} | author coverage {_pct(r['author_coverage'])}",
    ]
    for label, c in r["by_size"].items():
        lines.append(
            f"  {label:>6} px: agree {c['agree']}, only author {c['only_a']}, only verifier {c['only_v']}, "
            f"precision {_pct(c['author_precision'])}, coverage {_pct(c['author_coverage'])}"
        )
    for t, c in r["sensitivity"].items():
        lines.append(f"  at IoU {t}: precision {_pct(c['author_precision'])}, coverage {_pct(c['author_coverage'])}")
    if "misses_in_sample" in r:
        lines.append(
            f"misses of the bench in these images: {r['misses_in_sample']}, confirmed as faces by the verifier: "
            f"{r['misses_confirmed_by_verifier']}"
        )
    lines.append("")
    rows = r["discrepancies"]
    lines += [
        f"{d['kind']:<17} {d['image']}  A={d['box_author']}  V={d['box_verifier']}  iou={d['iou']}" for d in rows[:show]
    ]
    if len(rows) > show:
        lines.append(f"... {len(rows) - show} more in the JSON")
    if not rows:
        lines.append("no discrepancies")
    return "\n".join(lines)


def load_path(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    found = problems(doc)
    if found:
        raise SystemExit(f"{path} does not follow the schema: {found[:3]}")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verifier", required=True)
    p.add_argument("--author", nargs="+", default=DEFAULT_AUTHOR)
    p.add_argument("--images", help="text file with the images to compare (one per line)")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--details", help="bench/results/faces-details.json, to check the misses against the verifier")
    p.add_argument("--json", help="also write the full report here")
    args = p.parse_args(argv)
    misses: List[dict] = []
    if args.details:
        with open(args.details, encoding="utf-8") as f:
            misses = json.load(f)["misses"]
    report = compare(
        merge([load_path(x) for x in args.author]),
        load_path(args.verifier),
        args.iou,
        read_image_list(args.images) if args.images else None,
        misses,
    )
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
