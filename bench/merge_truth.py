"""Merge a blind verifier's plate truth into the author's, for the images the verifier labelled.

    python3 -s -m bench.merge_truth AUTHOR.json VERIFIER.json OUT.json [--images LIST.txt] [--iou 0.3]

On every image of the verifier's list the result keeps all the author's boxes and adds each verifier box that has no
partner at IoU >= --iou (the same plate drawn tighter or looser is one plate; the author's box wins). Both sides miss
plates the other one sees, so the union is the better estimate of what is really in the photo; the boxes that were
added are listed on stdout so that a person can look at each one. Images outside the list are copied unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from typing import List, Optional

from bench import compare_truth, plates_truth


def merge(author: dict, verifier: dict, only: Optional[set] = None, threshold: float = 0.3) -> tuple[dict, List[str]]:
    merged = copy.deepcopy(author)
    added: List[str] = []
    for name, entry in verifier["images"].items():
        if name not in merged["images"] or (only is not None and name not in only):
            continue
        mine = merged["images"][name]["boxes"]
        theirs = entry["boxes"]
        _, _, unpaired = compare_truth.pair_boxes(mine, theirs, threshold)
        for j in unpaired:
            mine.append(copy.deepcopy(theirs[j]))
            added.append(f"{name} {theirs[j]['box']} readable={theirs[j]['readable']} ignore={theirs[j]['ignore']}")
    return merged, added


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("author")
    parser.add_argument("verifier")
    parser.add_argument("out")
    parser.add_argument("--images")
    parser.add_argument("--iou", type=float, default=0.3)
    args = parser.parse_args(argv)
    only = compare_truth.read_image_list(args.images) if args.images else None
    merged, added = merge(compare_truth.load_path(args.author), compare_truth.load_path(args.verifier), only, args.iou)
    problems = plates_truth.problems(merged)
    if problems:
        raise SystemExit("merged truth breaks the schema:\n" + "\n".join(problems))
    with open(args.out, "w") as handle:
        json.dump(merged, handle, indent=1)
        handle.write("\n")
    print(f"{len(added)} boxes added from the verifier:")
    print("\n".join(added))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
