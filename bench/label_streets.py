"""Helpers for labelling the street photos by eye (the labels are then written by hand into
bench/truth/plates/streets.json; see the README there).

    python3 -s -m bench.label_streets propose            # run the detector on every street photo -> proposals.json
    python3 -s -m bench.label_streets view NAME [grid]   # NAME with the proposals drawn (and a coordinate grid)
    python3 -s -m bench.label_streets crop NAME x0 y0 x1 y1 [zoom]   # an enlarged crop with a finer grid

Output goes to a temporary folder (anonymizer-label under the system temp dir). Proposals only help to find plates quickly: the truth is
what a person sees, including plates the detector misses.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2
import numpy as np

from anonymizer.classes.plates import PlateModule

from .bench_plates import decide_photo

DATA = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", "bench/data"), "plates")
OUT = os.path.join(tempfile.gettempdir(), "anonymizer-label")
COLOURS = {"blur": (0, 200, 0), "check": (0, 200, 255)}


def grid(img: np.ndarray, step: int, origin=(0, 0), zoom: float = 1.0) -> None:
    height, width = img.shape[:2]
    ox, oy = origin
    first_x, first_y = (-ox) % step, (-oy) % step
    for gx in range(first_x, int(width / zoom) + 1, step):
        x = int(gx * zoom)
        cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1)
        cv2.putText(img, str(gx + ox), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for gy in range(first_y, int(height / zoom) + 1, step):
        y = int(gy * zoom)
        cv2.line(img, (0, y), (width, y), (255, 255, 255), 1)
        cv2.putText(img, str(gy + oy), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def propose() -> None:
    module = PlateModule(threads=2)
    folder = os.path.join(DATA, "streets")
    result = {}
    for name in sorted(os.listdir(folder)):
        bgr = cv2.imread(os.path.join(folder, name))
        zones = decide_photo(module, bgr)
        result[name] = [
            {
                "box": [round(v) for v in z.box],
                "verdict": z.verdict.value,
                "score": round(z.score, 2),
                "verify": round(z.verify, 2),
                "on_vehicle": z.on_vehicle,
            }
            for z in zones
        ]
        print(name, bgr.shape[1], bgr.shape[0], [(z["verdict"], z["box"]) for z in result[name]], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "proposals.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)


def view(name: str, with_grid: bool = False) -> None:
    with open(os.path.join(OUT, "proposals.json"), encoding="utf-8") as f:
        proposals = json.load(f)[name]
    bgr = cv2.imread(os.path.join(DATA, "streets", name))
    if with_grid:
        grid(bgr, 100)
    for i, p in enumerate(proposals):
        x0, y0, x1, y1 = p["box"]
        cv2.rectangle(bgr, (x0, y0), (x1, y1), COLOURS.get(p["verdict"], (255, 0, 0)), 2)
        cv2.putText(bgr, str(i), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    path = os.path.join(OUT, name)
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(path, bgr.shape[1], bgr.shape[0])
    for i, p in enumerate(proposals):
        print(i, p)


def crop(name: str, x0: int, y0: int, x1: int, y1: int, zoom: float = 3.0) -> None:
    bgr = cv2.imread(os.path.join(DATA, "streets", name))[y0:y1, x0:x1]
    bgr = cv2.resize(bgr, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    grid(bgr, 20, (x0, y0), zoom)
    path = os.path.join(OUT, f"{os.path.splitext(name)[0]}-crop.jpg")
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(path, bgr.shape[1], bgr.shape[0])


if __name__ == "__main__":
    command, rest = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if command == "propose":
        propose()
    elif command == "view" and len(rest) in (1, 2):
        view(rest[0], len(rest) == 2)
    elif command == "crop" and len(rest) in (5, 6):
        crop(rest[0], *(int(v) for v in rest[1:5]), *(float(v) for v in rest[5:]))
    else:
        raise SystemExit(__doc__)
