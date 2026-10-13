"""Build the truth-label dataset (labels only, no images) from bench/truth and bench/SOURCES.csv.

    python3 scripts/make_truth_dataset.py            # write docs/truth-dataset/
    python3 scripts/make_truth_dataset.py --check    # exit 1 if docs/truth-dataset/ is out of date

The package holds the boxes and how they were made; the images stay where they came from. IMAGES.csv gives, for every labelled
image or clip, its page, licence and the SHA-256 of the file the labels were made on, so anybody can fetch it and check.
This script only reads bench/truth and bench/SOURCES.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH = os.path.join(ROOT, "bench", "truth")
SOURCES = os.path.join(ROOT, "bench", "SOURCES.csv")
OUT = os.path.join(ROOT, "docs", "truth-dataset")

# The licence of the labels this project made itself.
OWN_LICENCE = "CC-BY-4.0"
OPENALPR_LICENCE = "AGPL-3.0"

# label file -> (folder of the images in bench/SOURCES.csv, kind of entry, licence of the label file, set name)
FILES = [
    ("faces/set-a.json", "faces/set-a", "images", OWN_LICENCE),
    ("faces/set-b.json", "faces/set-b", "images", OWN_LICENCE),
    ("faces/blind-faces-photos.json", "faces/blind", "images", OWN_LICENCE),
    ("faces/blind-video-frames.json", None, "images", OWN_LICENCE),
    ("faces/video.json", "faces/video", "clips", OWN_LICENCE),
    ("faces/truth-v2-video.json", "faces/video2/clips", "clips", OWN_LICENCE),
    ("plates/streets.json", "plates/streets", "images", OWN_LICENCE),
    ("plates/streets-author.json", "plates/streets", "images", OWN_LICENCE),
    ("plates/streets-verifier.json", "plates/streets", "images", OWN_LICENCE),
    ("plates/plates-v2.json", "plates/v2", "images", OWN_LICENCE),
    ("plates/openalpr-eu.json", "plates/openalpr-eu", "images", OPENALPR_LICENCE),
]
COLUMNS = [
    "set",
    "label_file",
    "name",
    "width",
    "height",
    "boxes",
    "ignored_boxes",
    "source_url",
    "licence",
    "author",
    "source_sha256",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def sources() -> dict:
    with open(SOURCES, encoding="utf-8", newline="") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def entries(document: dict, how: str):
    """(name, width, height, boxes) of every image, or of every clip with the boxes of all its frames."""
    if how == "images":
        for name, item in sorted(document["images"].items()):
            yield name, item.get("width", ""), item.get("height", ""), item.get("boxes", [])
    else:
        for clip in document["clips"].values():
            boxes = [b for frame in clip["frames"].values() for b in frame["boxes"]]
            yield clip["file"], clip.get("width", ""), clip.get("height", ""), boxes


def build() -> dict:
    """{relative path: bytes} of the whole package."""
    table = sources()
    out: dict = {}
    manifest = [["file", "bytes", "sha256", "licence_of_labels", "source_sha256_recorded_in_the_file"]]
    rows = []
    for rel, folder, how, licence in FILES:
        data = read(os.path.join(TRUTH, rel))
        document = json.loads(data)
        place = "labels-agpl" if licence == OPENALPR_LICENCE else "labels"
        target = f"{place}/{rel}"
        out[target] = data
        manifest.append([target, str(len(data)), sha256(data), licence, str(document.get("source_sha256", ""))])
        for name, width, height, boxes in entries(document, how):
            src = table.get(f"{folder}/{name}", {}) if folder else {}
            ignored = sum(1 for b in boxes if b.get("ignore"))
            rows.append(
                [
                    os.path.basename(rel)[:-5],
                    target,
                    name,
                    width,
                    height,
                    len(boxes) - ignored,
                    ignored,
                    quote(src.get("source_url", ""), safe=":/?&=%#,()'!*~+;@$"),
                    src.get("licence", ""),
                    src.get("author", ""),
                    src.get("sha256", ""),
                ]
            )
    # the AGPL-3.0 text travels with the AGPL labels (the licence asks for a copy, not a link)
    out["labels-agpl/LICENSE"] = read(os.path.join(TRUTH, "plates", "AGPL-3.0.txt"))
    split = read(os.path.join(TRUTH, "split-v2.json"))
    out["split/split-v2.json"] = split
    manifest.append(["split/split-v2.json", str(len(split)), sha256(split), OWN_LICENCE, ""])
    for name, header, body in (("MANIFEST.csv", manifest[0], manifest[1:]), ("IMAGES.csv", COLUMNS, rows)):
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        w.writerows(body)
        out[name] = buf.getvalue().encode("utf-8")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "truth_dataset_readme.md"), "rb") as f:
        out["README.md"] = f.read()
    return out


def unmatched() -> list:
    """Rows of IMAGES.csv with no source: the frames cut from a clip (their clip is in the table of the clips)."""
    table = csv.DictReader(io.StringIO(build()["IMAGES.csv"].decode("utf-8")))
    return [row["name"] for row in table if not row["source_sha256"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    package = build()
    if args.check:
        stale = [
            p
            for p, data in package.items()
            if not os.path.exists(os.path.join(OUT, p)) or read(os.path.join(OUT, p)) != data
        ]
        for p in stale:
            print("out of date:", p)
        return 1 if stale else 0
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for p, data in package.items():
        os.makedirs(os.path.dirname(os.path.join(OUT, p)), exist_ok=True)
        with open(os.path.join(OUT, p), "wb") as f:
            f.write(data)
    print(f"wrote {len(package)} files to {OUT}; {len(unmatched())} entries without a source row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
