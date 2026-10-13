"""Speed and memory of the context detectors (vehicles and persons) on dev photos only: YOLOX-nano against RF-DETR nano (PT1b).

    python3 -s -m bench.bench_context --model yolox|rfdetr --threads 4 --photos 12 --split-file bench/split-v2.json --out FILE.json

One model per process, so the peak resident set is that of the model alone. The vehicle pass is the stage of the plate path (whole picture, plus
1,024 px tiles above 1,400 px, tiles merged) on the dev plate photos; the person pass is `context.find_persons` on the dev photos of set a.
Nothing is run on a sealed photo. A figure is quoted with its machine and load; it is a design measurement, not a published speed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional

import cv2

from anonymizer.classes import context, plates, rfdetr
from bench.bench_cells import DATA, dev_part, dev_set_a_files, machine, peak_rss_mb, rss_mb, summary, timed


def dev_plate_files(split_file: str, photos: int) -> List[str]:
    table = dev_part(split_file)
    keys = sorted(k for k, part in table.items() if part == "dev" and k.startswith("plates/"))
    return [os.path.join(DATA, "plates", "v2", k.split("/", 1)[1]) for k in keys][:photos]


def run(model: str, photos: int, threads: int, split_file: str, out: Optional[str]) -> Dict[str, object]:
    stage = plates._VehicleStage()
    if model == "rfdetr":
        stage.vehicles = rfdetr.load_vehicle_detector(threads)
        people = context.load_person_detector(threads, None, "rfdetr")
    else:
        stage.vehicles = plates.VehicleDetector(
            plates._open_session(
                plates.VEHICLE, os.path.join(os.environ["ANONYMIZER_MODELS"], plates.VEHICLE_MODEL), threads
            )
        )
        people = context.load_person_detector(threads)
    started, before = machine(), rss_mb()
    vehicle_ms: List[float] = []
    vehicle_cpu: List[float] = []
    counts: List[int] = []
    first_call = None
    for path in dev_plate_files(split_file, photos):
        bgr = cv2.imread(path)
        found, wall, cpu = timed(stage._find_vehicles, bgr, True)
        if first_call is None:
            first_call = wall  # includes opening the session
            continue
        vehicle_ms.append(wall)
        vehicle_cpu.append(cpu)
        counts.append(len(found))
    person_ms: List[float] = []
    person_cpu: List[float] = []
    person_counts: List[int] = []
    for path in dev_set_a_files(split_file)[:photos]:
        bgr = cv2.imread(path)
        found, wall, cpu = timed(context.find_persons, people, bgr)
        person_ms.append(wall)
        person_cpu.append(cpu)
        person_counts.append(len(found))
    result = {
        "what": f"speed and memory of the {model} vehicle and person detector on dev photos (PT1b)",
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "machine": started,
        "load_1min_at_end": machine()["load_1min"],
        "threads": threads,
        "python": sys.version.split()[0],
        "rss_mb_before_first_picture": before,
        "first_vehicle_call_ms_includes_session_open": round(first_call or 0.0, 1),
        "vehicle_pass_per_photo": summary(vehicle_ms, vehicle_cpu),
        "vehicles_found_per_photo_mean": round(sum(counts) / max(1, len(counts)), 2),
        "person_pass_per_photo": summary(person_ms, person_cpu),
        "persons_found_per_photo_mean": round(sum(person_counts) / max(1, len(person_counts)), 2),
        "peak_rss_mb": peak_rss_mb(),
        "run_failures": getattr(stage.vehicles, "run_failures", 0) + getattr(people, "run_failures", 0),
        "caveat": "design measurement on dev photos; the median excludes the first vehicle call; quote it only with its machine and load",
        "_publication": {
            "speed": {"publishable": False, "why": "design measurement, not a reserved quiet-machine cell"}
        },
    }
    stage.vehicles.session = None
    people.close()
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("yolox", "rfdetr"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--photos", type=int, default=12)
    parser.add_argument("--split-file", default="bench/split-v2.json")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.model, args.photos, args.threads, args.split_file, args.out), indent=1))


if __name__ == "__main__":
    main()
