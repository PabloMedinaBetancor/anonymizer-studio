"""Pick, by code, the photos a blind verifier labels again (the sample for bench/compare_faces_truth.py).

    python3 -s -m bench.pick_faces_verify_sample [--total 60] [--crowds 8] [--set-b 12] [--seed 20261003] [--out FILE]

The sample is every photo with a miss or a false blur in bench/results/faces-details.json, plus photos drawn at
random (fixed seed) until it holds --total: first --crowds photos of set a with 30 or more counted faces (crowds
are where the author's truth is weakest), then --set-b photos of set b (the verifier looks for faces the author
called absent), then photos of set a. Lines are `set-a/name.jpg`, sorted. The verifier gets only this list.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILS = os.path.join(HERE, "results", "faces-details.json")
TRUTH_DIR = os.path.join(HERE, "truth", "faces")
CROWD_FACES = 30


def _load(tag: str) -> Dict:
    with open(os.path.join(TRUTH_DIR, f"set-{tag}.json"), encoding="utf-8") as f:
        return json.load(f)["images"]


def pick(total: int, crowds: int, set_b: int, seed: int, details_path: str = DETAILS) -> List[str]:
    a, b = _load("a"), _load("b")
    with open(details_path, encoding="utf-8") as f:
        details = json.load(f)
    flagged = {d["image"] for d in details["misses"] + details["false_blurs"]}
    rng = random.Random(seed)
    chosen = {"set-a/" + n for n in a if "set-a/" + n in flagged} | {"set-b/" + n for n in b if "set-b/" + n in flagged}

    def add(pool: List[str], count: int) -> None:
        pool = sorted(p for p in pool if p not in chosen)
        chosen.update(rng.sample(pool, min(count, len(pool))))

    crowd_pool = ["set-a/" + n for n, v in a.items() if sum(1 for x in v["boxes"] if not x["ignore"]) >= CROWD_FACES]
    have = len([c for c in chosen if c in crowd_pool])
    add(crowd_pool, max(0, crowds - have))
    have = len([c for c in chosen if c.startswith("set-b/")])
    add(["set-b/" + n for n in b], max(0, set_b - have))
    add(["set-a/" + n for n in a], total - len(chosen))
    return sorted(chosen)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=60)
    p.add_argument("--crowds", type=int, default=8)
    p.add_argument("--set-b", type=int, default=12)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--out")
    args = p.parse_args(argv)
    text = "\n".join(pick(args.total, args.crowds, args.set_b, args.seed)) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
