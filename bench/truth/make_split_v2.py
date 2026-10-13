"""Fix the dev / sealed split of the photo truth v2 by file hash.

    python3 -s bench/truth/make_split_v2.py            # writes bench/truth/split-v2.json (refuses to overwrite)

Rule, written before any pilot cell runs on dev and applied to the SHA-256 of the image bytes, never to a name or a score:
  (a) the 128 photos of v1 (sets a and b): sealed when int(sha256[:8], 16) % 100 < 30, else dev (about 30 % sealed);
  (c) the WIDER FACE validation subset of 150 images: sealed when int(sha256[8:16], 16) % 2 == 0, else dev (about half);
  (b) the blind photos of the real-estate kind (added by `--add-blind LIST`): all sealed;
  (p) the plate photos of truth v2 (added by `--add-plates KEY.json`, the kit's key with the sha256 of every photo): sealed when
      int(sha256[:8], 16) % 100 < 30, else dev, fixed before any cell runs on them.
The file keeps the rule, the sha256 of every image and its part, so that a later run can check that no image moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "split-v2.json")
MAIN = os.environ.get("ANONYMIZER_MAIN", "")  # the main checkout (scripts/dev-env.sh sets it)
WIDER = os.environ.get("WIDER_VAL", "")  # the folder of the WIDER FACE validation subset (test data, never distributed)
if not (MAIN and WIDER):
    sys.exit("set ANONYMIZER_MAIN (source scripts/dev-env.sh) and WIDER_VAL (the WIDER FACE validation folder)")

RULE = {
    "a": "sealed if int(sha256[:8],16) % 100 < 30 else dev",
    "c": "sealed if int(sha256[8:16],16) % 2 == 0 else dev",
    "b": "all sealed",
    "p": "sealed if int(sha256[:8],16) % 100 < 30 else dev",
}


def sha256_of(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def part_a(digest: str) -> str:
    return "sealed" if int(digest[:8], 16) % 100 < 30 else "dev"


def part_c(digest: str) -> str:
    return "sealed" if int(digest[8:16], 16) % 2 == 0 else "dev"


def build() -> dict:
    items = {}
    for part, folder in (("a", "set-a"), ("a", "set-b")):
        root = os.path.join(MAIN, "bench", "data", "faces", folder)
        for name in sorted(os.listdir(root)):
            digest = sha256_of(os.path.join(root, name))
            items[f"{folder}/{name}"] = {"origin": part, "sha256": digest, "part": part_a(digest)}
    for name in sorted(os.listdir(WIDER)):
        digest = sha256_of(os.path.join(WIDER, name))
        items[f"wider/{name}"] = {"origin": "c", "sha256": digest, "part": part_c(digest)}
    return {"rule": RULE, "made": time.strftime("%Y-%m-%d %H:%M:%S"), "items": items}


def add_blind(doc: dict, listing: str) -> None:
    with open(listing, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest = sha256_of(line)
            doc["items"][f"blind/{os.path.basename(line)}"] = {"origin": "b", "sha256": digest, "part": "sealed"}
    doc["blind_added"] = time.strftime("%Y-%m-%d %H:%M:%S")


def add_plates(doc: dict, key_file: str) -> None:
    with open(key_file, encoding="utf-8") as fh:
        key = json.load(fh)
    for name, meta in sorted(key.items()):
        doc["items"][f"plates/{name}"] = {"origin": "p", "sha256": meta["sha256"], "part": part_a(meta["sha256"])}
    doc["plates_added"] = time.strftime("%Y-%m-%d %H:%M:%S")


def add_video(doc: dict, truth_file: str, clips_dir: str) -> None:
    """The 40 truth frames of the video truth v2, keyed `video/<clip>/<frame>`. A clip is one unit (its frames are not independent): the clips are sorted
    by the SHA-256 of the clip file and the first 30 % (3 of 10, rounded up) are sealed, the rest dev. A hash threshold per clip would put 6 of 10 clips in
    the sealed part with these hashes; the rank rule is fixed before any cell runs on a frame."""
    with open(truth_file, encoding="utf-8") as fh:
        clips = json.load(fh)["clips"]
    hashes = {name: sha256_of(os.path.join(clips_dir, os.path.basename(c["file"]))) for name, c in clips.items()}
    order = sorted(clips, key=lambda name: hashes[name])
    sealed = set(order[: -(-3 * len(order) // 10)])
    for name, c in sorted(clips.items()):
        for frame in sorted(c["frames"], key=int):
            doc["items"][f"video/{name}/{frame}"] = {
                "origin": "v",
                "sha256": hashes[name],
                "part": "sealed" if name in sealed else "dev",
            }
    doc["video_added"] = time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-blind", help="text file with one image path per line (the blind set b)")
    ap.add_argument("--add-plates", help="key.json of the plate kit (name -> {sha256, ...})")
    ap.add_argument(
        "--add-video",
        nargs=2,
        metavar=("TRUTH_JSON", "CLIPS_DIR"),
        help="truth-v2-video.json and the folder of its clips",
    )
    a = ap.parse_args()
    if a.add_plates:
        doc = json.load(open(OUT, encoding="utf-8"))
        if "plates_added" in doc:
            sys.exit("the plate photos were already added; the split is fixed")
        add_plates(doc, a.add_plates)
    elif a.add_video:
        doc = json.load(open(OUT, encoding="utf-8"))
        if "video_added" in doc:
            sys.exit("the video frames were already added; the split is fixed")
        add_video(doc, *a.add_video)
    elif a.add_blind:
        doc = json.load(open(OUT, encoding="utf-8"))
        if "blind_added" in doc:
            sys.exit("the blind set was already added; the split is fixed")
        add_blind(doc, a.add_blind)
    else:
        if os.path.exists(OUT):
            sys.exit(f"{OUT} exists: the split is fixed, not remade")
        doc = build()
    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=1, sort_keys=True)
    counts: dict = {}
    for item in doc["items"].values():
        key = (item["origin"], item["part"])
        counts[key] = counts.get(key, 0) + 1
    print(sorted(counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
