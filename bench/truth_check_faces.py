"""Truth check of the face truth: blind verifier against the author, with the discrepancies read by eye.

    python3 -s -m bench.truth_check_faces --verifier V.json --images LIST.txt [--merge] [--out bench/results/truth-check-faces.json]

Runs bench.compare_faces_truth and joins the visual review in bench/truth/faces/truth-check-review.json (made by
looking at the crops of every unpaired box). It writes the report with the reviewed counts: faces the author missed
(a real face the blind verifier saw), faces only the author has that are real, and framed portraits (ambiguous).
With --merge the faces the author missed are added to set-a.json as counted faces (the truth becomes the union, as
for plates); it is idempotent. After a merge, rerun bench/bench_faces.py: the denominator changed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import List, Optional

from bench import compare_faces_truth as cmp

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW = os.path.join(HERE, "truth", "faces", "truth-check-review.json")
SET_A = os.path.join(HERE, "truth", "faces", "set-a.json")


def merge_missed(review: dict, set_a_path: str = SET_A) -> int:
    with open(set_a_path, encoding="utf-8") as f:
        doc = json.load(f)
    added = 0
    for it in review["items"]:
        if it["verdict"] != "author_missed":
            continue
        entry = doc["images"][it["image"].split("/", 1)[1]]
        if any(b["box"] == it["box"] for b in entry["boxes"]):
            continue
        entry["boxes"].append({"box": it["box"], "kind": "face", "ignore": False, "note": "added_from_blind_verifier"})
        entry["boxes"].sort(key=lambda e: (e["box"][1] // 50, e["box"][0]))
        added += 1
    if added:
        with open(set_a_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
    return added


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verifier", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--details", default=os.path.join(HERE, "results", "faces-details.json"))
    p.add_argument("--merge", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "results", "truth-check-faces.json"))
    args = p.parse_args(argv)
    tmp = args.out + ".tmp"
    # The comparison is always against the author's own truth, so rerunning after a merge gives the same report.
    authors = []
    for i, path in enumerate(cmp.DEFAULT_AUTHOR):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for entry in doc["images"].values():
            entry["boxes"] = [b for b in entry["boxes"] if b.get("note") != "added_from_blind_verifier"]
        authors.append(args.out + f".author{i}.tmp")
        with open(authors[-1], "w", encoding="utf-8") as f:
            json.dump(doc, f)
    cmp.main(
        ["--verifier", args.verifier, "--images", args.images, "--json", tmp, "--author"]
        + authors
        + (["--details", args.details] if args.details else [])
    )
    for path in authors:
        os.remove(path)
    with open(tmp, encoding="utf-8") as f:
        report = json.load(f)
    os.remove(tmp)
    with open(REVIEW, encoding="utf-8") as f:
        review = json.load(f)
    counts = Counter(it["verdict"] for it in review["items"])
    out = {"report": report, "review": review, "reviewed_counts": dict(counts)}
    if args.merge:
        out["merged_into_set_a"] = merge_missed(review)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("reviewed:", dict(counts), "| merged:", out.get("merged_into_set_a", "no"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
