"""Face recall per frame on two public street clips (criterion F4), Standard against Maximum.

    python3 -s -m bench.bench_faces_video [--qualities standard maximum] [--out bench/results]

Runs engine.video.analyze_video with FaceModule on the clips of bench/truth/faces/video.json (fetched by
bench/fetch_faces_video.py) and scores 52 frames labelled by hand (28 of Calle Preciados, 24 of a Hong Kong corridor).
For each labelled face, a frame counts it when the blur that render_video would draw there (the tracks the analysis
blurs by default, with the class margins, as ellipses) covers at least COVERED of the box (0.8; also given at 0.5).
Two readings: `blur` is what is blurred without anybody looking, `blur_or_check` adds the tracks Review lists as
"Not sure". Hong Kong has no face in its frames: it only counts false blurs (the tracks BLUR gives and the frames
with a blur). Writes bench/results/faces-video.json; the hardware is in the file.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np

from anonymizer.classes.faces import FaceModule
from anonymizer.engine.tracks import final_boxes
from anonymizer.engine.video import analyze_video
from bench.bench_faces import hardware
from bench.results_meta import interval, not_publishable, truth_hashes, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
TRUTH = os.path.join(HERE, "truth", "faces", "video.json")
COVERED = 0.8
LOOSE = 0.5
SIZE_CLASSES = (("<24", 0, 24), ("24-48", 24, 48), (">48", 48, 10**9))


def load_truth() -> Dict:
    with open(TRUTH, encoding="utf-8") as f:
        return json.load(f)


def covered_share(truth_box: Sequence[float], zones: Sequence[Tuple[Sequence[float], str]]) -> float:
    """Share of the truth box that the union of the zones covers (ellipses for faces, rectangles otherwise)."""
    x0, y0, x1, y1 = (int(round(v)) for v in truth_box)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    mask = np.zeros((h, w), np.uint8)
    for (bx0, by0, bx1, by1), shape in zones:
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        if shape == "ellipse":
            centre = (int(round((bx0 + bx1) / 2 - x0)), int(round((by0 + by1) / 2 - y0)))
            axes = (max(1, int(round((bx1 - bx0) / 2))), max(1, int(round((by1 - by0) / 2))))
            cv2.ellipse(mask, centre, axes, 0, 0, 360, 1, -1)
        else:
            mask[max(0, int(by0) - y0) : max(0, int(by1) - y0), max(0, int(bx0) - x0) : max(0, int(bx1) - x0)] = 1
    return float(mask.sum()) / float(w * h)


def zones_of(analysis, track_ids: Set[int]) -> Dict[int, List[Tuple[list, str]]]:
    tracks = [t for t in analysis.tracks if t.id in track_ids]
    boxes = final_boxes(
        tracks,
        analysis.motions,
        analysis.cuts,
        analysis.frames,
        analysis.fps,
        analysis.params,
        analysis.width,
        analysis.height,
        analysis.specs,
    )
    return {t: [(list(b), shape) for b, shape, _ in v] for t, v in boxes.items()}


def size_class(box: Sequence[float]) -> str:
    side = max(box[2] - box[0], box[3] - box[1])
    return next(n for n, lo, hi in SIZE_CLASSES if lo <= side < hi)


def score_clip(analysis, truth: Dict) -> Dict:
    blur = analysis.sure_ids()
    listed = {t.id for t in analysis.listed()}
    readings = {"blur": zones_of(analysis, blur), "blur_or_check": zones_of(analysis, listed)}
    faces = ignored = 0
    hits = {k: {"strict": 0, "loose": 0} for k in readings}
    by_size = {n: {"faces": 0, **{k: 0 for k in readings}} for n, _, _ in SIZE_CLASSES}
    misses: List[dict] = []
    ignored_blurred = 0
    for frame, entry in truth["frames"].items():
        t = int(frame)
        for b in entry["boxes"]:
            if b["ignore"]:
                ignored += 1
                ignored_blurred += covered_share(b["box"], readings["blur"].get(t, [])) >= COVERED
                continue
            faces += 1
            by_size[size_class(b["box"])]["faces"] += 1
            for k, zones in readings.items():
                share = covered_share(b["box"], zones.get(t, []))
                hits[k]["strict"] += share >= COVERED
                hits[k]["loose"] += share >= LOOSE
                by_size[size_class(b["box"])][k] += share >= COVERED
                if k == "blur" and share < COVERED:
                    misses.append(
                        {
                            "frame": t,
                            "box": b["box"],
                            "share": round(share, 2),
                            "check_share": round(covered_share(b["box"], readings["blur_or_check"].get(t, [])), 2),
                        }
                    )
    pct = lambda n: round(100.0 * n / faces, 2) if faces else None  # noqa: E731
    all_frames = readings["blur"]
    return {
        "labelled_frames": len(truth["frames"]),
        "faces": faces,
        "ignored_boxes": ignored,
        "ignored_boxes_blurred": int(ignored_blurred),
        **{f"recall_{k}_pct": pct(v["strict"]) for k, v in hits.items()},
        **{f"recall_{k}_loose_pct": pct(v["loose"]) for k, v in hits.items()},
        "recall_blur_interval_pct": wilson(hits["blur"]["strict"], faces),
        "recall_blur_loose_interval_pct": wilson(hits["blur"]["loose"], faces),
        "faces_hit_blur": hits["blur"]["strict"],
        "faces_hit_blur_loose": hits["blur"]["loose"],
        "by_size": by_size,
        "misses": misses,
        "tracks": len(analysis.tracks),
        "tracks_blur": len(blur),
        "tracks_check": len(listed - blur),
        "frames_in_clip": analysis.frames,
        "frames_with_a_blur": len(all_frames),
    }


def run_clip(path: str, quality: str, truth: Dict) -> Dict:
    module = FaceModule()
    start = time.perf_counter()
    analysis = analyze_video(path, [module], quality=quality)
    elapsed = time.perf_counter() - start
    result = score_clip(analysis, truth)
    result["analysis_seconds"] = round(elapsed, 1)
    result["ms_per_frame"] = round(1000 * elapsed / max(1, analysis.frames), 1)
    result["seconds_by_step"] = {k: round(v, 2) for k, v in analysis.seconds.items()}
    result["unchecked_frames"] = len(analysis.unchecked)
    return result


WHY_NOT = (
    "22 faces of one street clip, labelled by one reviewer with no second pass (95 % interval 23-61), nearly all of them profiles in "
    "deep shadow at 25-45 px: it says nothing about faces that look at the camera, and Standard against Maximum is inside the noise "
    "(Maximum blurred 8 and then 7 of the 22 faces in two runs of the same code). bench/truth/faces/README.md, section Video (F4)."
)
PUBLICATION = {"faces.video": not_publishable(WHY_NOT), "faces_video": not_publishable(WHY_NOT)}


def summary(runs: Dict) -> Dict:
    """The figures the documents quote (`faces.video.*`), from the per-clip runs: face-frames covered over all the
    labelled faces of the clips that have some, for each quality (blurred by default, and blurred or listed to check)."""
    video: Dict = {}
    for quality, clips in runs.items():
        faces = sum(r["faces"] for r in clips.values())
        hits = sum(r["faces_hit_blur"] for r in clips.values())
        video[f"frame_recall_{quality}_pct"] = round(100.0 * hits / faces, 2)
        video[f"frame_recall_{quality}"] = interval(hits, faces)
        video[f"frame_recall_{quality}_blur_or_check_pct"] = round(
            100.0
            * sum(round(r["recall_blur_or_check_pct"] * r["faces"] / 100.0) for r in clips.values() if r["faces"])
            / faces,
            2,
        )
        video[f"labelled_faces_{quality}"] = faces
    return {"faces": {"video": video}, "_publication": PUBLICATION}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--qualities", nargs="+", default=["standard", "maximum"])
    p.add_argument("--clips", nargs="+", default=["preciados", "hongkong"])
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--summary-only", action="store_true", help="rewrite the `faces.video` block of an existing result")
    args = p.parse_args()
    if args.summary_only:
        path = os.path.join(args.out, "faces-video.json")
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        results.update(summary(results["faces_video"]))
        results["truth_sha256"] = truth_hashes(["bench/truth/faces/video.json"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        return 0
    truth = load_truth()
    out: Dict = {}
    for quality in args.qualities:
        out[quality] = {}
        for clip in args.clips:
            spec = truth["clips"][clip]
            path = os.path.join(DATA, "faces", "video", spec["file"])
            print(f"{quality} {clip} ...", flush=True)
            out[quality][clip] = run_clip(path, quality, spec)
            r = out[quality][clip]
            print(
                f"  faces {r['faces']}: BLUR {r['recall_blur_pct']} %, BLUR or CHECK {r['recall_blur_or_check_pct']} %; "
                f"tracks BLUR {r['tracks_blur']} CHECK {r['tracks_check']}; {r['ms_per_frame']} ms/frame",
                flush=True,
            )
    results = {
        "faces_video": out,
        "data": {"covered_share": COVERED, "loose_share": LOOSE, "truth": "bench/truth/faces/video.json"},
        "truth_sha256": truth_hashes(["bench/truth/faces/video.json"]),
        "hardware": hardware(),
        "run": {"date": datetime.date.today().isoformat()},
    }
    results.update(summary(out))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "faces-video.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
