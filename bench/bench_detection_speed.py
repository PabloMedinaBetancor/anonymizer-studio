"""SL9: the time per photo of `careful` against `fast`, end to end through the public API (faces and plates together), on dev photos only.

    python3 -s bench/bench_detection_speed.py --split-file bench/split-v2.json --plates 12 --faces 12 --threads 4 --out FILE.json

Run it from the tree under test (its `src` on the path); `--tree` names that tree so that the file records its commit. Every photo is analysed
once per mode in the same process, the models loaded and one warm-up photo analysed outside the timing, and the photos alternate between the
two modes so that a change in the load of the machine hits both. The criterion is the median of `careful` over the median of `fast` (<= 3).
A figure is quoted with its machine and load; a run on a loaded machine is a design measurement, not a published speed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

DATA = os.environ.get("ANONYMIZER_BENCH_DATA", "")


def dev_files(split_file: str, plates: int, faces: int) -> List[str]:
    with open(split_file, encoding="utf-8") as f:
        items = json.load(f)["items"]
    dev = sorted(k for k, v in items.items() if v["part"] == "dev")
    chosen = [os.path.join(DATA, "plates", "v2", k.split("/", 1)[1]) for k in dev if k.startswith("plates/")][:plates]
    chosen += [os.path.join(DATA, "faces", "set-a", k.split("/", 1)[1]) for k in dev if k.startswith("set-a/")][:faces]
    return chosen


def machine() -> Dict[str, object]:
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), "")
    except OSError:
        pass
    return {
        "host": os.environ.get("BENCH_MACHINE_LABEL", "unnamed"),  # a label, never the computer's own name
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "load_1min": round(os.getloadavg()[0], 2),
    }


def tree_commit(tree: Optional[str]) -> str:
    if not tree:
        return ""
    try:
        return subprocess.run(
            [os.environ.get("GIT", "git"), "-C", tree, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:  # the commit is a note, not a condition
        return ""


def run(files: List[str], threads: int, tree: Optional[str], out: Optional[str]) -> Dict[str, object]:
    import anonymizer

    zones: Dict[str, List[int]] = {"fast": [], "careful": []}
    ms: Dict[str, List[float]] = {"fast": [], "careful": []}
    per_photo: List[Dict[str, object]] = []
    studios = {
        mode: anonymizer.Anonymizer(anonymizer.Options(detection=mode), threads=threads) for mode in ("fast", "careful")
    }
    started = machine()
    try:
        for studio in studios.values():
            studio.analyze(files[0])  # models loaded and the first call made outside the timing
        for path in files:
            row: Dict[str, object] = {"photo": os.path.basename(path)}
            for mode in ("fast", "careful"):  # fast first, then careful, photo by photo
                seen: List[object] = []
                start = time.perf_counter()
                studios[mode].analyze(path, on_event=seen.append)
                elapsed = 1000 * (time.perf_counter() - start)
                ms[mode].append(elapsed)
                visited = next((e.zones_visited for e in seen if getattr(e, "type", "") == "clip.analyzed"), 0)
                zones[mode].append(visited)
                row[f"{mode}_ms"] = round(elapsed, 1)
                row[f"{mode}_zones_visited"] = visited
            per_photo.append(row)
    finally:
        for studio in studios.values():
            studio.close()
    ratios = [r["careful_ms"] / r["fast_ms"] for r in per_photo]
    fast_median, careful_median = statistics.median(ms["fast"]), statistics.median(ms["careful"])
    result = {
        "what": "SL9: median ms per photo, careful against fast, end to end through the public API, faces and plates, dev photos only",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "tree": tree or "",
        "tree_commit": tree_commit(tree),
        "machine": started,
        "load_1min_at_end": machine()["load_1min"],
        "threads": threads,
        "python": sys.version.split()[0],
        "photos": len(files),
        "fast_ms_median": round(fast_median, 1),
        "careful_ms_median": round(careful_median, 1),
        "ratio_of_medians": round(careful_median / fast_median, 2),
        "ratio_median_of_photos": round(statistics.median(ratios), 2),
        "ratio_worst_photo": round(max(ratios), 2),
        "bar": 3.0,
        "met": careful_median / fast_median <= 3.0,
        "zones_visited_median_careful": statistics.median(zones["careful"]),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "per_photo": per_photo,
        "caveat": "dev photos only; the two modes alternate photo by photo in one process; quote it only with its machine and load",
        "_publication": {
            "speed": {"publishable": False, "why": "design measurement, not a reserved quiet-machine cell"}
        },
    }
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split-file", default="bench/split-v2.json")
    parser.add_argument("--plates", type=int, default=12)
    parser.add_argument("--faces", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tree")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    files = dev_files(args.split_file, args.plates, args.faces)
    if not files:
        raise SystemExit("no dev photos: source scripts/dev-env.sh of the tree under test")
    result = run(files, args.threads, args.tree, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "per_photo"}, indent=1))


if __name__ == "__main__":
    main()
