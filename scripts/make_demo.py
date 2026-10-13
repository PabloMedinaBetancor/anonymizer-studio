"""Build demo/ (the public sample clips of the demo mode and of the screenshots) from the benchmark data.

    python3 -s scripts/make_demo.py

Only public material from bench/SOURCES.csv (Wikimedia Commons, CC0 / CC BY / CC BY-SA) is used. The two videos are cut
and shrunk with ffmpeg (the exact commands are the ones below), the two photos are shrunk to the 720 px short side the demo mode accepts (DEMO_LIMITS).  demo/SOURCES.csv is
written from the benchmark rows, with the sha256 of the file that lands in demo/ and a 'derived from' note.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(ROOT, "bench", "data")
DEMO = os.path.join(ROOT, "demo")
FIELDS = ["file", "kind", "source_url", "author", "licence", "licence_url", "downloaded", "sha256", "used_for", "notes"]

# (output, source in bench/data, start s, length s or None for a copy, why)
ITEMS = [
    ("street-madrid.mp4", "faces/video/preciados.ogv", 12, 15, "demo clip: pedestrians, faces"),
    ("street-hongkong.mp4", "faces/video/hongkong.webm", 0, 10, "demo clip: street scene, nothing to blur"),
    ("street-taipei.jpg", "plates/streets/street_005.jpg", None, None, "demo photo: dense traffic, plates"),
    (
        "street-westminster.jpg",
        "faces/set-a/London-Westminster---Bridge-Street-Pedestrian-Crossing---geograph.org..jpg",
        None,
        None,
        "demo photo: pedestrians and cars",
    ),
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cut(source: str, target: str, start: float, length: float) -> None:
    cmd = [
        "ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", source,
        "-an", "-vf", "scale=960:-2:flags=lanczos", "-c:v", "libx264", "-preset", "slow", "-crf", "30",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", target,
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


def shrink(source: str, target: str) -> None:
    short = "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)':flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-vf", short, "-q:v", "3", target], check=True)


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found", file=sys.stderr)
        return 1
    with open(os.path.join(ROOT, "bench", "SOURCES.csv"), encoding="utf-8", newline="") as f:
        origin = {row["file"]: row for row in csv.DictReader(f)}
    os.makedirs(DEMO, exist_ok=True)
    rows = []
    for name, rel, start, length, why in ITEMS:
        row = dict(origin[rel])
        source = os.path.join(BENCH, rel)
        target = os.path.join(DEMO, name)
        if length is None:
            shrink(source, target)
            note = "derived from bench/%s (short side 720 px, JPEG quality 3 of ffmpeg)" % rel
        else:
            cut(source, target, start, length)
            note = "derived from bench/%s (%.0f s from %.0f s, 960 px wide, H.264 crf 30, no audio)" % (
                rel,
                length,
                start,
            )
        rows.append(
            {
                "file": name,
                "kind": "video" if length else "image",
                "source_url": row["source_url"],
                "author": row["author"],
                "licence": row["licence"],
                "licence_url": row["licence_url"],
                "downloaded": row["downloaded"],
                "sha256": sha256(target),
                "used_for": why,
                "notes": note,
            }
        )
    with open(os.path.join(DEMO, "SOURCES.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %d files in %s" % (len(rows), DEMO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
