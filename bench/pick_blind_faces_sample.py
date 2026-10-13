"""Pick, by code, a small sample of photos with few faces for a blind verifier (no detector, no author truth).

    python3 -s -m bench.pick_blind_faces_sample [--total 15] [--seed 20261003] [--out FILE]

Candidates are photos of set a with 1 to 8 counted faces (no crowds) and no more than 8 boxes in all (ignore ones included). Half of the
sample (rounded up) comes from the candidates that have a miss or a false blur in bench/results/faces-details.json,
the other half at random (fixed seed) from the rest. Lines are `set-a/name.jpg`, sorted. The verifier gets only this
list: never the truth, the boxes or the details file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILS = os.path.join(HERE, "results", "faces-details.json")
TRUTH_A = os.path.join(HERE, "truth", "faces", "set-a.json")
MAX_FACES = 8
MAX_BOXES = 8


def pick(total: int, seed: int, details_path: str = DETAILS, truth_path: str = TRUTH_A) -> List[str]:
    with open(truth_path, encoding="utf-8") as f:
        images = json.load(f)["images"]
    with open(details_path, encoding="utf-8") as f:
        details = json.load(f)
    flagged = {d["image"] for d in details["misses"] + details["false_blurs"]}
    pool = []
    for name, v in images.items():
        faces = sum(1 for x in v["boxes"] if not x["ignore"])
        ignored = len(v["boxes"]) - faces
        if 1 <= faces <= MAX_FACES and faces + ignored <= MAX_BOXES:
            pool.append("set-a/" + name)
    rng = random.Random(seed)
    bad = sorted(p for p in pool if p in flagged)
    good = sorted(p for p in pool if p not in flagged)
    n_bad = min(len(bad), (total + 1) // 2)
    chosen = rng.sample(bad, n_bad)
    chosen += rng.sample(good, min(len(good), total - n_bad))
    return sorted(chosen)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=15)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--out")
    args = p.parse_args(argv)
    text = "\n".join(pick(args.total, args.seed)) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
