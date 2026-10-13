"""Engine numbers quoted in the documentation: gap filling, Standard against Maximum, speed, memory, re-render.

    python3 -s -m bench.bench_engine [--runs 2] [--out bench/results] [--skip-speed]

Writes bench/results/engine.json. Where each figure comes from:

- C5 `engine.gaps.synthetic`: synthetic clips with exact truth (the generators of tests/test_engine_tracks.py: a still,
  panning or walking camera, plates fixed in the scene, faces moving, a noisy detector with misses, retries and verification
  simulated from the truth). The share of object-frames whose whole visible box is covered by the blur, for each gap method,
  over several seeds and cameras. Deterministic: the same numbers on any machine.
- C6 `engine.standard_vs_maximum`: computed from bench/results/faces-video.json (run bench.bench_faces_video first; both
  qualities, same clips, same machine, per-frame hand-labelled truth). `coverage_pct` is the share of the faces that
  Maximum blurred (covering 80 % of the box) that Standard blurred too; `speedup` is the analysis time of Maximum over that of
  Standard. If the file or the labels are missing the figure is left out and the JSON says why.
- C7-C9 `engine.speed`, `.memory`, `.rerender`: a child process runs the public API on a public 1080p clip (faces and plates,
  Standard): analyze, render, change one decision, render again. Speed is (analyze + render) per second of video; memory is the
  peak resident set of the child right after the analysis (models included); the re-render speedup is (analyze + render)
  over the second render, which runs no detector. One machine, named in the file; `--runs` fresh processes, the median is kept.
  The same is measured on a second, busy clip (a street with many faces, 720p) because the 1080p clip has almost nothing to blur:
  `engine.speed_busy`, `engine.memory_busy`, `engine.rerender_busy`, each with the share of frames that have a blurred box and the
  resolution in its name. `engine.photo_speed` is the time per photo with faces and plates together on ONE set (the street photos).
- `engine.stop.latency_s`: seconds from `CancelToken.cancel()` to `Cancelled`, 10 stops during the analysis and 10 during the render of
  the 1080p clip, in this process. The requirement (SPEC E8) is 2 s on a free machine: the file says whether it was met.
- C10 `repo.tests.count`: the tests that `unittest discover` loads in this tree (with the libraries of this machine).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional

from bench.results_meta import interval, not_publishable, truth_hashes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("ANONYMIZER_BENCH_DATA") or os.path.join(HERE, "data")
SPEED_CLIP = os.path.join(DATA, "faces", "video", "hongkong.webm")
BUSY_CLIP = os.path.join(DATA, "faces", "video", "preciados.ogv")
STREET_PHOTOS = os.path.join(DATA, "plates", "streets")
STOPS_PER_PHASE = 10
STOP_REQUIREMENT_S = 2.0
GAP_CASES = (("pan", 4), ("walk", 3), ("walk", 4))
GAP_SEEDS = 10
FULL_COVERAGE = 0.99


def pct(part: int, whole: int) -> Optional[float]:
    return round(100.0 * part / whole, 2) if whole else None


def gap_round() -> Dict:
    """C5: coverage of the object-frames by gap method on synthetic clips with exact truth."""
    import numpy as np

    from anonymizer.contract import RETRY, SWEEP, Detection
    from anonymizer.engine import tracks as tk
    from tests import test_engine_tracks as tt
    from tests.fakes import v1_video_judgement, v1_wants_recover, v1_wants_verify

    def run_round(seed: int, camera_kind: str, step: int):
        rng = np.random.default_rng(seed)
        n = 300
        motions = tt.camera(n, rng, camera_kind)
        objects = tt.objects(n, motions, rng)
        by_t = tt.detections(n, objects, rng, step, 0.75, 5, 0.04)
        tracker = tk.Tracker(tt.FPS, tk.TrackerParams())

        def truth_near(kind, t, box):
            best = None
            for k, path in objects:
                if k == kind and t in path:
                    b = tk.state_to_box(path[t])
                    if best is None or tt.coverage(b, box) > tt.coverage(best, box):
                        best = b
            return best if best is not None and tt.coverage(best, box) > 0.2 else None

        for t in range(n):
            key = t % step == 0
            raw = by_t[t] if key else []
            lost = tracker.step(t, motions[t], False, [Detection(*d[0], d[1], d[2], SWEEP, d[3]) for d in raw], key)
            for track in lost:
                if not v1_wants_recover(track.stats()):
                    continue
                truth = truth_near(track.kind, t, tk.state_to_box(track.state))
                if truth is not None:
                    tracker.recover(track, t, Detection(*truth, 0.6, track.kind, RETRY, track.kind == tt.PLATE))
            if key:
                for track in tracker.active:
                    view = track.views.get(t)
                    if view is None or view.source != SWEEP:
                        continue
                    if v1_wants_verify(track.stats(), view):
                        track.best_verify = max(track.best_verify, 0.95 if view.score > 0.35 else 0.0)
                        track.n_verify += 1
                        track.verify_side = max(view.w, view.h)
            tracker.end_step(t, tt.FPS / step)
        tracks = tk.join_split(
            tk.merge_duplicates(tracker.finish()), tracker.motions, tracker.cuts, n, tt.FPS, tk.TrackerParams()
        )
        return tracker, tracks, objects, n

    methods = (tk.MOTION, tk.LINEAR, tk.HOLD)
    total = {m: [0, 0] for m in methods}
    by_camera = {}
    for cam, step in GAP_CASES:
        cases = {m: [0, 0] for m in methods}
        for seed in range(GAP_SEEDS):
            tracker, tracks, objects, n = run_round(seed, cam, step)
            blurred = [p for p in tracks if v1_video_judgement(p.stats()).verdict.value == "blur"]
            for m in methods:
                boxes = tk.final_boxes(
                    blurred, tracker.motions, tracker.cuts, n, tt.FPS, tk.TrackerParams(), tt.W, tt.H, tt.SPECS, m
                )
                for _, path in objects:
                    for t, e in path.items():
                        truth = tk.state_to_box(e)
                        seen = (max(0, truth[0]), max(0, truth[1]), min(tt.W, truth[2]), min(tt.H, truth[3]))
                        if (seen[2] - seen[0]) * (seen[3] - seen[1]) < 0.3 * e[2] * e[3]:
                            continue
                        cases[m][1] += 1
                        if max((tt.coverage(seen, b) for b, _, _ in boxes.get(t, [])), default=0) >= FULL_COVERAGE:
                            cases[m][0] += 1
        for m in methods:
            total[m][0] += cases[m][0]
            total[m][1] += cases[m][1]
        by_camera[f"{cam}/{step}"] = {f"{m}_pct": pct(*cases[m]) for m in methods}
    return {
        "motion_pct": pct(*total[tk.MOTION]),
        "linear_pct": pct(*total[tk.LINEAR]),
        "hold_pct": pct(*total[tk.HOLD]),
        "object_frames": total[tk.MOTION][1],
        "seeds": GAP_SEEDS,
        "cameras": [f"{c}/{s}" for c, s in GAP_CASES],
        "by_camera": by_camera,
        "criterion": f"object-frames whose visible box is covered at least {FULL_COVERAGE:.0%} by the blur, gap method in the name",
        "oracle": (
            "the retries and the second looks are simulated from the truth (a retry returns the real box, a verification "
            "always confirms): this is the share of object-frames covered with perfect recovery, a ceiling, not recall of the "
            "whole pipeline. State it with the figure."
        ),
    }


def standard_vs_maximum() -> Dict:
    """C6, from bench/results/faces-video.json and the labels it was scored with."""
    path = os.path.join(HERE, "results", "faces-video.json")
    try:
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        with open(os.path.join(HERE, "truth", "faces", "video.json"), encoding="utf-8") as f:
            truth = json.load(f)
    except OSError as e:
        return {"reason": f"no per-frame results to compare ({e.__class__.__name__}: {e.filename})"}
    hits: Dict[str, set] = {}
    seconds: Dict[str, float] = {}
    frames = 0
    for quality in ("standard", "maximum"):
        hits[quality], seconds[quality] = set(), 0.0
        for clip, spec in truth["clips"].items():
            run = results["faces_video"][quality][clip]
            missed = {(m["frame"], tuple(m["box"])) for m in run["misses"]}
            for frame, entry in spec["frames"].items():
                for b in entry["boxes"]:
                    if b["kind"] == "face" and not b["ignore"] and (int(frame), tuple(b["box"])) not in missed:
                        hits[quality].add((clip, int(frame), tuple(b["box"])))
            seconds[quality] += run["analysis_seconds"]
            if quality == "standard":
                frames += run["faces"]
        if len(hits[quality]) != sum(results["faces_video"][quality][c]["faces_hit_blur"] for c in truth["clips"]):
            return {"reason": f"the misses of {quality} do not add up to its hits in faces-video.json"}
    both = hits["standard"] & hits["maximum"]
    if not hits["maximum"]:
        return {"reason": "Maximum blurred no labelled face: nothing to compare against"}
    return {
        "coverage_pct": pct(len(both), len(hits["maximum"])),
        "speedup": round(seconds["maximum"] / seconds["standard"], 2),
        "coverage": interval(len(both), len(hits["maximum"])),
        "faces": frames,
        "faces_blurred_by_maximum": len(hits["maximum"]),
        "faces_blurred_by_standard": len(hits["standard"]),
        "faces_blurred_by_both": len(both),
        "analysis_seconds": {q: round(s, 1) for q, s in seconds.items()},
        "speedup_what": (
            "analysis time of Maximum over Standard, the face detector alone: what Maximum costs, not an advantage of Standard"
        ),
        "truth_sha256": truth_hashes(["bench/truth/faces/video.json"]),
        "source": "bench/results/faces-video.json",
        "scope": "face detector only, two public clips, 52 frames labelled by hand",
        "hardware": results["hardware"],
        "date": results["run"]["date"],
    }


def coverage_publication(comparison: Dict) -> Dict:
    """The coverage of Maximum by Standard is not to be quoted: a handful of faces from one clip. The text carries this run's numbers."""
    both, maximum = comparison.get("faces_blurred_by_both"), comparison.get("faces_blurred_by_maximum")
    low, high = comparison.get("coverage", {}).get("ci95_pct", [None, None])
    why = (
        f"{both} of the {maximum} faces that Maximum blurred, in one street clip with {comparison.get('faces')} labelled faces: "
        f"a handful of cases (95 % interval {low}-{high}), inside the noise; the same sample cannot say whether Maximum finds more faces than "
        f"Standard ({comparison.get('faces_blurred_by_standard')} against {maximum}, and Maximum itself gave 8 and 7 faces in two "
        "runs of the same code). bench/truth/faces/README.md."
    )
    if "reason" in comparison:
        why = "no comparison was made in this run: " + comparison["reason"]
    return {
        "engine.standard_vs_maximum.coverage_pct": not_publishable(why),
        "engine.standard_vs_maximum.coverage": not_publishable(why),
    }


def rerender_publication(speed: Dict) -> Dict:
    """The re-render of the 1080p clip is not to be quoted when that clip has nothing to blur: no decision was changed, so the second
    render is a copy of the first with nothing to do. The same measure on the busy clip (`rerender_busy`) is the one that stands."""
    if speed.get("blurred_frames_pct", 0) > 50 and speed.get("items", 0) > 0:
        return {}
    why = (
        f"measured on a clip with {speed.get('items')} items and {speed.get('blurred_frames_pct')} % of its frames blurred: no decision was "
        "changed and the second render has no blur to redo; the figure of the clip with a blurred box in most frames is engine.rerender_busy"
    )
    return {"engine.rerender": not_publishable(why)}


def child_run(clip: str, work: str) -> Dict:
    """Inside the child: analyze, render, change a decision, render again. Times in seconds, peak memory in MB."""
    import anonymizer

    out: Dict = {}
    options = anonymizer.Options()
    with anonymizer.Anonymizer(options) as studio:
        start = time.perf_counter()
        analysis = studio.analyze(clip)
        out["analyze_s"] = time.perf_counter() - start
        out["peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        out["items"] = len(analysis.items)
        rows = analysis.boxes(0, analysis.frames - 1)
        out["blurred_frames_pct"] = round(100.0 * sum(1 for row in rows.values() if row) / max(1, len(rows)), 1)
        start = time.perf_counter()
        studio.render(analysis, os.path.join(work, "first.mp4"))
        out["render_s"] = time.perf_counter() - start
        if analysis.items:
            item = analysis.items[0]
            analysis.set_blur(item.item_id, not item.blur)
        start = time.perf_counter()
        studio.render(analysis, os.path.join(work, "second.mp4"))
        out["rerender_s"] = time.perf_counter() - start
    info = anonymizer.media_info(clip)
    out["duration_s"], out["width"], out["height"], out["frames"] = (
        info.duration_s,
        info.width,
        info.height,
        info.frames,
    )
    return out


def child_runs(clip: str, runs: int) -> Dict:
    done: List[Dict] = []
    for i in range(runs):
        with tempfile.TemporaryDirectory(prefix="bench-engine-") as work:
            print(f"speed run {i + 1}/{runs} on {os.path.basename(clip)} ...", flush=True)
            cmd = [sys.executable, "-s", "-m", "bench.bench_engine", "--child", clip, work]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                return {"reason": "the measuring process failed: " + proc.stderr.strip().splitlines()[-1:][0]}
            done.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    return {"runs": done}


def figures(done: List[Dict], label: str) -> Dict:
    """speed, memory and re-render of one clip from its runs (the median is kept)."""
    first = done[0]
    duration = first["duration_s"]
    full = [r["analyze_s"] + r["render_s"] for r in done]
    med = statistics.median
    common = {
        "video_seconds": round(duration, 2),
        "frames": first["frames"],
        "resolution": f"{first['width']}x{first['height']}",
        "items": first["items"],
        "blurred_frames_pct": first["blurred_frames_pct"],
        "hardware": hardware(),
    }
    return {
        "speed": {
            f"s_per_video_s_{label}": round(med(full) / duration, 2),
            "analyze_s_per_video_s": round(med(r["analyze_s"] for r in done) / duration, 2),
            "render_s_per_video_s": round(med(r["render_s"] for r in done) / duration, 2),
            "runs_s_per_video_s": [round(x / duration, 2) for x in full],
            "settings": "faces and plates, Standard, default margins; analyze + render, one process",
            **common,
        },
        "memory": {
            f"peak_mb_{label}": round(med(r["peak_mb"] for r in done)),
            "runs_mb": [round(r["peak_mb"]) for r in done],
            "what": "peak resident set of the process right after the analysis, models loaded",
            "hardware": hardware(),
        },
        "rerender": {
            "speedup": round(med(f / r["rerender_s"] for f, r in zip(full, done, strict=True)), 2),
            "rerender_s_per_video_s": round(med(r["rerender_s"] for r in done) / duration, 2),
            "what": "(analyze + render) over a second render after set_blur on one item; no detector runs in it",
            "blurred_frames_pct": first["blurred_frames_pct"],
            "hardware": hardware(),
        },
    }


def speed_runs(runs: int) -> Dict:
    out: Dict = {}
    for clip, label, suffix in ((SPEED_CLIP, "1080p", ""), (BUSY_CLIP, "720p", "_busy")):
        if not os.path.isfile(clip):
            out.update(
                {k + suffix: {"reason": f"the public clip is missing: {clip}"} for k in ("speed", "memory", "rerender")}
            )
            continue
        measured = child_runs(clip, runs)
        if "reason" in measured:
            out.update({k + suffix: measured for k in ("speed", "memory", "rerender")})
            continue
        got = figures(measured["runs"], label)
        resolution = got["speed"]["resolution"]
        expected = {"1080p": "1920x1080", "720p": "1280x720"}[label]
        if resolution != expected:
            out.update({k + suffix: {"reason": f"the clip is {resolution}, not {expected}"} for k in got})
            continue
        got["speed"]["clip"] = (
            f"faces/video/{os.path.basename(clip)} ({resolution}, Wikimedia Commons, bench/SOURCES.csv)"
        )
        out.update({k + suffix: v for k, v in got.items()})
    return out


def photo_speed() -> Dict:
    """The time to analyze, and to analyze and write, one photo with faces and plates together, over one set of photos (the 60 street photos)."""
    from PIL import Image

    import anonymizer

    if not os.path.isdir(STREET_PHOTOS):
        return {"reason": f"the street photos are missing: {STREET_PHOTOS}"}
    files = sorted(f for f in os.listdir(STREET_PHOTOS) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    sizes, analyze_ms, both_ms = [], [], []
    with (
        tempfile.TemporaryDirectory(prefix="bench-photos-") as work,
        anonymizer.Anonymizer(anonymizer.Options()) as studio,
    ):
        studio.analyze(os.path.join(STREET_PHOTOS, files[0]))  # the models are loaded once, outside the timing
        for name in files:
            path = os.path.join(STREET_PHOTOS, name)
            with Image.open(path) as im:
                sizes.append(im.size)
            start = time.perf_counter()
            analysis = studio.analyze(path)
            middle = time.perf_counter()
            studio.render(analysis, os.path.join(work, f"{os.path.splitext(name)[0]}.out{os.path.splitext(name)[1]}"))
            end = time.perf_counter()
            analyze_ms.append(1000 * (middle - start))
            both_ms.append(1000 * (end - start))
    widths, heights = sorted(w for w, _ in sizes), sorted(h for _, h in sizes)
    return {
        "set": "plates/streets (Wikimedia Commons street photos, bench/SOURCES.csv)",
        "photos": len(files),
        "resolution_px": {
            "median": [widths[len(widths) // 2], heights[len(heights) // 2]],
            "smallest_width": widths[0],
            "largest_width": widths[-1],
        },
        "analyze_ms_per_photo": round(statistics.mean(analyze_ms), 1),
        "analyze_ms_per_photo_median": round(statistics.median(analyze_ms), 1),
        "analyze_and_write_ms_per_photo": round(statistics.mean(both_ms), 1),
        "analyze_and_write_ms_per_photo_median": round(statistics.median(both_ms), 1),
        "settings": "faces and plates together, Standard, through the public API (analyze, then render)",
        "hardware": hardware(),
    }


def stop_latency() -> Dict:
    """Seconds from cancel() to Cancelled, STOPS_PER_PHASE stops during the analysis and as many during the render of the 1080p clip.
    The stop comes from another thread at a moment that is not a frame boundary (a timer), as a person's click does."""
    import threading

    import anonymizer

    if not os.path.isfile(SPEED_CLIP):
        return {"reason": f"the public 1080p clip is missing: {SPEED_CLIP}"}
    seconds: Dict[str, List[float]] = {"analyze": [], "render": []}
    with (
        tempfile.TemporaryDirectory(prefix="bench-stop-") as work,
        anonymizer.Anonymizer(anonymizer.Options()) as studio,
    ):
        analysis = studio.analyze(SPEED_CLIP)
        for phase in seconds:
            for i in range(STOPS_PER_PHASE):
                token = anonymizer.CancelToken()
                asked: List[float] = []

                def stop(token=token, asked=asked):
                    asked.append(time.perf_counter())
                    token.cancel()

                timer = threading.Timer(
                    (2.0 if phase == "analyze" else 0.5) + (1.13 if phase == "analyze" else 0.8) * i, stop
                )
                timer.start()
                try:
                    if phase == "analyze":
                        studio.analyze(SPEED_CLIP, None, token)
                    else:
                        studio.render(analysis, os.path.join(work, "stopped.mp4"), None, token)
                except anonymizer.Cancelled:
                    seconds[phase].append(time.perf_counter() - asked[0])
                else:
                    timer.cancel()
                    return {"reason": f"{phase} finished before the stop at stop {i + 1}"}
    every = seconds["analyze"] + seconds["render"]
    worst = max(every)
    return {
        "latency_s": {"median": round(statistics.median(every), 2), "max": round(worst, 2)},
        "latency_median_s": round(statistics.median(every), 2),
        "latency_max_s": round(worst, 2),
        "by_phase_max_s": {p: round(max(v), 2) for p, v in seconds.items()},
        "stops": len(every),
        "requirement_s": STOP_REQUIREMENT_S,
        "met": worst <= STOP_REQUIREMENT_S,
        "what": (
            "seconds from CancelToken.cancel(), called from another thread at an arbitrary moment, to Cancelled being raised, on the "
            "1080p clip, in one process; the requirement holds on a free machine"
        ),
        "hardware": hardware(),
    }


def hardware() -> Dict:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass  # not Linux, or no model name line: the file just says "unknown"
    ram = None
    try:
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass  # RAM is informative only
    return {
        "cpu": cpu,
        "cores": os.cpu_count(),
        "ram_gb": ram,
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def count_tests() -> int:
    """The test cases `python3 -m unittest discover -s tests -t .` finds, counted by a fresh process as that command would: a class or a
    module that skips as a whole (no Node, no AVIF writer, a timing test without ANONYMIZER_TIMING) still counts each of its cases here, while
    the `Ran N tests` of a run on a machine that skips them says fewer."""
    code = "import unittest; print(unittest.TestLoader().discover('tests', top_level_dir='.').countTestCases())"
    done = subprocess.run([sys.executable, "-s", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return int(done.stdout.strip().splitlines()[-1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=int, default=2, help="fresh processes for speed, memory and re-render")
    p.add_argument("--out", default=os.path.join(HERE, "results"))
    p.add_argument("--skip-speed", action="store_true")
    p.add_argument("--child", nargs=2, metavar=("CLIP", "WORKDIR"), help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        print(json.dumps(child_run(*args.child)))
        return 0
    engine: Dict = {}
    print("gap round ...", flush=True)
    engine["gaps"] = {"synthetic": gap_round()}
    engine["standard_vs_maximum"] = standard_vs_maximum()
    if args.skip_speed:
        reason = {"reason": "not measured in this run (--skip-speed)"}
        engine.update(
            {
                name: reason
                for name in (
                    "speed",
                    "memory",
                    "rerender",
                    "speed_busy",
                    "memory_busy",
                    "rerender_busy",
                    "photo_speed",
                    "stop",
                )
            }
        )
    else:
        engine.update(speed_runs(args.runs))
        print("photos ...", flush=True)
        engine["photo_speed"] = photo_speed()
        print("stop latency ...", flush=True)
        engine["stop"] = stop_latency()
    results = {
        "engine": engine,
        "_publication": {
            **coverage_publication(engine["standard_vs_maximum"]),
            **rerender_publication(engine["speed"]),
        },
        "repo": {
            "tests": {
                "count": count_tests(),
                "what": "test cases that `unittest discover -s tests -t .` finds (a class that skips as a whole still counts each of its cases); a run reports fewer when this machine skips some",
            }
        },
        "hardware": hardware(),
        "run": {"date": datetime.date.today().isoformat()},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "engine.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(json.dumps(results["engine"]["gaps"]["synthetic"], indent=1)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
