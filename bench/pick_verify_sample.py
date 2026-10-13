"""Pick, by code, the street photos a blind verifier labels again (the sample for bench/compare_truth.py).

    python3 -s -m bench.pick_verify_sample [--total 30] [--seed 20261003] [--out FILE]

The sample is every street photo that has a miss, a false blur or a false check in bench/results/plates-details.json,
plus photos drawn at random (fixed seed) from the others until it holds --total. The OpenALPR photos are not sampled:
their truth is the dataset's own. Paths are relative to $ANONYMIZER_BENCH_DATA/plates/ and sorted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Optional

from bench import plates_truth

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILS = os.path.join(HERE, "results", "plates-details.json")


def pick(total: int, seed: int) -> List[str]:
    streets = sorted(plates_truth.load("streets")["images"])
    with open(DETAILS, encoding="utf-8") as f:
        flagged = {d["image"] for d in json.load(f)["streets"]}
    chosen = [n for n in streets if n in flagged]
    others = [n for n in streets if n not in flagged]
    chosen += random.Random(seed).sample(others, max(0, min(len(others), total - len(chosen))))
    return ["streets/" + n for n in sorted(chosen)]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--out")
    args = p.parse_args(argv)
    lines = pick(args.total, args.seed)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
