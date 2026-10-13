"""Compare two plate truths (the author's and a blind verifier's) box by box.

    python3 -s -m bench.compare_truth A.json B.json [--images LIST.txt] [--iou 0.5] [--json OUT.json]

Both files follow the schema of bench/truth/plates/README.md. Only the images that both name are compared (and, with
--images, only those in the list: one path per line, `streets/street_003.jpg` or just `street_003.jpg`, `#` comments).
Boxes are paired greedily by IoU, best first, and a pair needs IoU >= --iou. What the report counts:

  agree      a pair that neither side marks `ignore`
  uncertain  a pair where one side or both mark `ignore`: nobody is wrong, the box is just unclear
  only in A / only in B   a box with no partner and not `ignore`: the real disagreements
  ignored    unpaired boxes marked `ignore`: not a disagreement, listed apart

concordance = agree / (agree + only in A + only in B). The discrepancies come sorted by how much they matter: boxes
only on one side (larger first), then pairs whose `ignore` or `readable` flag differs, then pairs that overlap
poorly (IoU under 0.7). The exit status is 0 whatever the result: the report is read by a person.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

from bench import plates_truth

Box = Sequence[float]
GOOD_IOU = 0.7


def iou(a: Box, b: Box) -> float:
    inter = plates_truth.intersection(a, b)
    union = plates_truth.area(a) + plates_truth.area(b) - inter
    return inter / union if union > 0 else 0.0


def pair_boxes(
    a: List[dict], b: List[dict], threshold: float
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy pairing by descending IoU. Returns (pairs as (i, j, iou), unpaired in a, unpaired in b)."""
    scored = sorted(((iou(x["box"], y["box"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), reverse=True)
    used_a: Set[int] = set()
    used_b: Set[int] = set()
    pairs: List[Tuple[int, int, float]] = []
    for score, i, j in scored:
        if score < threshold:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((i, j, score))
    return pairs, [i for i in range(len(a)) if i not in used_a], [j for j in range(len(b)) if j not in used_b]


def normalise_name(entry: str) -> str:
    return os.path.basename(entry.strip())


def read_image_list_text(lines: Sequence[str]) -> Set[str]:
    cleaned = [ln.split("#")[0].strip() for ln in lines]
    return {normalise_name(ln) for ln in cleaned if ln}


def read_image_list(path: str) -> Set[str]:
    with open(path, encoding="utf-8") as f:
        return read_image_list_text(f.read().splitlines())


def compare(a: Dict, b: Dict, threshold: float = 0.5, only: Optional[Set[str]] = None) -> Dict:
    """The comparison of two truth documents, as a dict ready for printing or for JSON."""
    names = sorted(set(a["images"]) & set(b["images"]))
    if only is not None:
        names = [n for n in names if n in only]
    counts = {"agree": 0, "uncertain": 0, "only_a": 0, "only_b": 0, "ignored_a": 0, "ignored_b": 0}
    discrepancies: List[dict] = []
    for name in names:
        boxes_a, boxes_b = a["images"][name]["boxes"], b["images"][name]["boxes"]
        pairs, left_a, left_b = pair_boxes(boxes_a, boxes_b, threshold)
        for i, j, score in pairs:
            x, y = boxes_a[i], boxes_b[j]
            if x["ignore"] or y["ignore"]:
                counts["uncertain"] += 1
                if x["ignore"] != y["ignore"]:
                    discrepancies.append(_row("ignore_differs", 2, name, x["box"], y["box"], score, x, y))
                continue
            counts["agree"] += 1
            if x["readable"] != y["readable"]:
                discrepancies.append(_row("readable_differs", 3, name, x["box"], y["box"], score, x, y))
            elif score < GOOD_IOU:
                discrepancies.append(_row("poor_overlap", 4, name, x["box"], y["box"], score, x, y))
        for i in left_a:
            if boxes_a[i]["ignore"]:
                counts["ignored_a"] += 1
            else:
                counts["only_a"] += 1
                discrepancies.append(_row("only_in_a", 1, name, boxes_a[i]["box"], None, 0.0, boxes_a[i], None))
        for j in left_b:
            if boxes_b[j]["ignore"]:
                counts["ignored_b"] += 1
            else:
                counts["only_b"] += 1
                discrepancies.append(_row("only_in_b", 1, name, None, boxes_b[j]["box"], 0.0, None, boxes_b[j]))
    discrepancies.sort(key=lambda r: (r["rank"], -r["area"], r["image"]))
    both = counts["agree"] + counts["only_a"] + counts["only_b"]
    return {
        "images": len(names),
        "iou": threshold,
        **counts,
        "concordance": round(counts["agree"] / both, 4) if both else None,
        "discrepancies": discrepancies,
    }


def _row(kind: str, rank: int, image: str, box_a, box_b, score: float, x, y) -> dict:
    area = plates_truth.area(box_a if box_a is not None else box_b)
    return {
        "kind": kind,
        "rank": rank,
        "image": image,
        "box_a": box_a,
        "box_b": box_b,
        "iou": round(score, 3),
        "area": round(area, 1),
        "flags_a": None if x is None else {"readable": x["readable"], "ignore": x["ignore"]},
        "flags_b": None if y is None else {"readable": y["readable"], "ignore": y["ignore"]},
    }


def render(report: Dict) -> str:
    c = report
    lines = [
        f"images compared: {c['images']}  (pairing at IoU >= {c['iou']})",
        f"agree {c['agree']} | uncertain (ignore on a side) {c['uncertain']} | only in A {c['only_a']} | only in B {c['only_b']}"
        f" | ignored boxes A {c['ignored_a']}, B {c['ignored_b']}",
        "concordance: " + ("n/a" if c["concordance"] is None else f"{c['concordance'] * 100:.1f} %"),
        "",
    ]
    if not c["discrepancies"]:
        lines.append("no discrepancies")
    for r in c["discrepancies"]:
        lines.append(f"{r['kind']:<17} {r['image']}  A={r['box_a']}  B={r['box_b']}  iou={r['iou']}")
    return "\n".join(lines)


def load_path(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    found = plates_truth.problems(doc)
    if found:
        raise SystemExit(f"{path} does not follow the schema: {found[:3]}")
    return doc


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--images", help="text file with the images to compare (one per line)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--json", help="also write the full report here")
    args = p.parse_args(argv)
    report = compare(
        load_path(args.a), load_path(args.b), args.iou, read_image_list(args.images) if args.images else None
    )
    print(render(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
