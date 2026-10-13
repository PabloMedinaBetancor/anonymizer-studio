"""Build the truth file and the SOURCES.csv rows for the OpenALPR EU benchmark kept in
$ANONYMIZER_BENCH_DATA/plates/openalpr-eu/ (a local copy of github.com/openalpr/benchmarks, endtoend/eu).

Each .txt holds one tab-separated line: file, x, y, width, height, plate text. Usage:
    python3 -s -m bench.convert_openalpr_eu
"""

from __future__ import annotations

import csv
import glob
import hashlib
import os

import cv2

from . import plates_truth

DATA = os.path.join(os.environ["ANONYMIZER_BENCH_DATA"], "plates", "openalpr-eu")
URL = "https://github.com/openalpr/benchmarks/tree/master/endtoend/eu"
RAW = "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/eu/"
LICENCE_URL = "https://www.gnu.org/licenses/agpl-3.0.html"
DOWNLOADED = "2026-09-23"
SOURCES_HEADER = [
    "file",
    "kind",
    "source_url",
    "author",
    "licence",
    "licence_url",
    "downloaded",
    "sha256",
    "used_for",
    "notes",
]


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_truth() -> dict:
    images = {}
    for txt in sorted(glob.glob(os.path.join(DATA, "*.txt"))):
        with open(txt, encoding="utf-8") as f:
            name, x, y, w, h, _text = f.read().strip().split("\t")
        height, width = cv2.imread(os.path.join(DATA, name)).shape[:2]
        box = [int(x), int(y), int(x) + int(w), int(y) + int(h)]
        images[name] = {
            "width": width,
            "height": height,
            "boxes": [{"box": box, "kind": "plate", "readable": True, "ignore": False}],
        }
    return {"set": "openalpr-eu", "licence": "AGPL-3.0", "images": images}


def source_rows() -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "*.jpg")) + glob.glob(os.path.join(DATA, "*.txt"))):
        name = os.path.basename(path)
        kind = "image" if name.endswith(".jpg") else "annotation"
        rows.append(
            [
                f"plates/openalpr-eu/{name}",
                kind,
                RAW + name,
                "OpenALPR Inc. and benchmark contributors",
                "AGPL-3.0",
                LICENCE_URL,
                DOWNLOADED,
                sha256(path),
                "plate recall benchmark (P4, P6)",
                "local benchmark only: never committed, never in the demo",
            ]
        )
    return rows


def main() -> None:
    truth = build_truth()
    bad = plates_truth.problems(truth)
    if bad:
        raise SystemExit("\n".join(bad))
    plates_truth.dump(truth, os.path.join(plates_truth.TRUTH_DIR, "openalpr-eu.json"))
    sources = os.path.join(os.path.dirname(plates_truth.HERE), "bench", "SOURCES.csv")
    with open(sources, encoding="utf-8", newline="") as f:
        existing = [r for r in csv.reader(f) if r and r[0] != "file" and not r[0].startswith("plates/openalpr-eu/")]
    with open(sources, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(SOURCES_HEADER)
        out.writerows(existing + source_rows())
    print(f"{len(truth['images'])} images, source: {URL}")


if __name__ == "__main__":
    main()
