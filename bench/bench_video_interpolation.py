"""VK4: how far the blur computed between two heavy key frames stays on the face the heavy model sees.

    python3 -s -m bench.bench_video_interpolation collect --clips DIR --truth-video FILE --split-file SPLIT [--clip-names a,b] [--windows 20] [--truth-frames] --out RAW
    python3 -s -m bench.bench_video_interpolation evaluate --raw RAW [--min-score 0.52] [--out FILE]

The claim under test: between heavy key frames the box is "mostly covered by calculation" (the camera motion carries it, the measured gap
margins cover the rest). `collect` takes windows of 49 consecutive frames from the public clips (away from the truth frames and the
changes of shot), measures the camera motion of every frame of a window with the engine's own `CameraMotion`, and runs the heavy model
over the whole frame (the deep stage of photos) at the frames of a grid (offsets 0, 2, 4, ... 48 from the window start). `evaluate` then
replays it offline: for every spacing N of the grid, the faces found at the key frames 0 and N that are the same face (the tracker's own join
rule, with the camera motion) become a track of two heavy views, `tracks.final_boxes` gives its blur at every frame in between (gap
interpolation, margins that grow with the distance, smoothing), and each heavy detection at an in-between grid frame that lies on that track
is compared with it: the share of the heavy box left uncovered. Reported by distance to the nearest key frame; `N` is the largest spacing
whose 95th percentile uncovered share is at most 1 % and for which every smaller candidate holds too (the tail, the p99, the maximum and the shares
over 10 % and 50 % are in the table and in the claim text; N is in frames and in seconds by the frame rate of each clip). A face the heavy model sees between the key frames and that no track carries is not an
interpolation error (it is a face born after the key frame, the periodic look's job): it is counted apart, never dropped. A spacing at
which fewer than half of the faces of the first key frame can be joined to a face of the second is not claimed either: the faces that
moved too far to be joined are exactly the ones the interpolation would not carry.

`collect` needs the split (`--split-file`, keys "video/<clip>/<frame>", a clip is one unit): only the clips the split marks dev are opened, a
frame the split does not name stops the run, a sealed clip is never decoded and its labels are never read. With `--truth-frames` it also
runs the heavy model on the dev truth frames, and `evaluate` reads from them the box error of the heavy model: the per-side margin that makes
its box contain the labelled box, and its 90th percentile is the proposed `BOX_SECOND` margin.

The file names the machine and the load; a speed is quoted from one machine (M7). No number is written into the engine by this script: the
proposals (`period`, `box_second_margin`, `heavy_per_second`) are in the file and the maintainer sets the constants.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from anonymizer.contract import BOX_SECOND as SECOND_ORIGIN
from anonymizer.contract import FACE, SECOND, ZONE_DOUBTFUL, Detection, SuspectZone
from anonymizer.engine import tracks as tk
from anonymizer.geometry import iou

GRID = (0, 2, 4, 6, 8, 12, 16, 24, 32, 48)  # frame offsets in a window at which the heavy model runs
SPACINGS = (4, 6, 8, 12, 16, 24, 32, 48)  # candidate spacings N: key frames at offsets 0 and N, both in GRID
WINDOW = max(GRID) + 1
MAX_STEP = 16  # the largest gap between two grid frames: a face unseen at one grid frame is still followed across it
TRUTH_MARGIN = 15  # a window keeps this many frames away from a truth frame
P95_LIMIT = 0.01  # VK4: the 95th percentile of the uncovered share must be at most 1 %
MIN_CASES = 30  # a spacing is judged only on at least this many (face, in-between frame) cases
# and only if at least this share of the faces at the first key frame can be joined to a face at the second
MIN_PAIR_RATE = 0.5
MARGIN_PERCENTILE = 90
HEAVY_SHARE_OF_REAL_TIME = 0.5  # proposal for heavy_per_second: at most this much heavy work per second of video
Box = Tuple[float, float, float, float]


# --- planning ---


def plan_windows(
    clips: Dict[str, dict], total: int, truth_frames: Dict[str, Sequence[int]], seed: int = 20261003
) -> List[Tuple[str, int]]:
    """(clip, first frame) of `total` windows of WINDOW frames, spread over the clips in proportion to their length, never within TRUTH_MARGIN
    frames of a truth frame of that clip. `clips` maps a name to {"total_frames": n}. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    names = sorted(clips)
    frames = {n: int(clips[n]["total_frames"]) for n in names}
    weight = sum(frames.values())
    quota = {n: max(1, round(total * frames[n] / weight)) for n in names}
    while sum(quota.values()) > total:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < total:
        quota[min(quota, key=quota.get)] += 1
    windows: List[Tuple[str, int]] = []
    for name in names:
        last = frames[name] - WINDOW
        if last <= 0 or quota[name] <= 0:
            continue
        taken: List[int] = []
        for slot in np.linspace(0, last, quota[name] + 2)[1:-1]:
            start = max(0, min(last, int(slot) + int(rng.integers(-3, 4))))
            while start <= last and (
                any(abs(start - t) < WINDOW for t in taken) or _near_truth(start, truth_frames.get(name, ()))
            ):
                start += 5
            if start <= last:
                taken.append(start)
                windows.append((name, start))
    return windows


def _near_truth(start: int, truth: Sequence[int]) -> bool:
    return any(start - TRUTH_MARGIN <= t <= start + WINDOW + TRUTH_MARGIN for t in truth)


# --- collecting (needs video, the light motion and the heavy model) ---


def machine() -> Dict[str, object]:
    from bench.bench_cells import machine as machine_of

    return machine_of()


def window_record(
    frames: Dict[int, np.ndarray], gray_scale: float, sweep, offsets: Sequence[int] = GRID, zone_look=None
) -> dict:
    """The raw record of one window. `frames` maps an offset to its BGR frame (every offset 0..WINDOW-1 present for the motion, the grid ones
    for the heavy model); `sweep(bgr) -> List[Detection]` is the heavy model over the whole frame. Motion is measured with the engine's
    `CameraMotion` on a reduced grey copy, as `analyze_video` does. The window ends at a change of shot."""
    import cv2

    motion = tk.CameraMotion(gray_scale)
    motions, cut_at = [tk.IDENTITY.tolist()], None
    last = max(frames)
    for offset in range(0, last + 1):
        bgr = frames[offset]
        h, w = bgr.shape[:2]
        small = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (max(16, int(w * gray_scale)), max(16, int(h * gray_scale)))
        )
        a, cut, _ = motion.next(small)
        if offset == 0:
            continue
        if cut and cut_at is None:
            cut_at = offset
        motions.append(np.round(a, 6).tolist())
    detections, ms, zone_ms = {}, [], []
    for offset in offsets:
        if offset not in frames or (cut_at is not None and offset >= cut_at):
            continue
        started = time.perf_counter()
        found = sweep(frames[offset])
        ms.append(1000 * (time.perf_counter() - started))
        detections[str(offset)] = [
            [round(d.x0, 1), round(d.y0, 1), round(d.x1, 1), round(d.y1, 1), round(d.score, 4)] for d in found
        ]
        if zone_look is not None:
            for d in sorted(found, key=lambda d: -d.score)[
                :2
            ]:  # the call a key frame makes: one look at a zone around a face
                started = time.perf_counter()
                zone_look(frames[offset], d)
                zone_ms.append(1000 * (time.perf_counter() - started))
    return {"motions": motions, "cut_at": cut_at, "detections": detections, "sweep_ms": ms, "zone_ms": zone_ms}


def read_clip_windows(path: str, starts: Sequence[int]):
    """Yield (start, {offset: BGR frame}) for each window start (windows do not overlap), decoding the clip once."""
    import av

    ordered = sorted(starts)
    held: Dict[int, np.ndarray] = {}
    current = 0
    with av.open(path) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if current >= len(ordered):
                break
            start = ordered[current]
            if index < start:
                continue
            held[index - start] = frame.to_ndarray(format="bgr24")
            if index - start == WINDOW - 1:
                yield start, held
                held, current = {}, current + 1
    if held:
        yield ordered[current], held


def dev_truth_frames(truth: dict, table: Dict[str, str]) -> Dict[str, List[int]]:
    """The truth frames the split marks dev, as {clip: [frame, ...]}. The split keys are "video/<clip>/<frame>". A frame the split does not
    name stops the run (the measurement role fixes the split by hash before any model sees a frame); a sealed frame is never read."""
    missing, dev = [], {}
    for clip, info in truth["clips"].items():
        for frame in info["frames"]:
            part = table.get(f"video/{clip}/{frame}")
            if part is None:
                missing.append(f"video/{clip}/{frame}")
            elif part == "dev":
                dev.setdefault(clip, []).append(int(frame))
            elif part != "sealed":
                raise SystemExit(
                    f"split: unknown part {part!r} for video/{clip}/{frame}: a frame is either dev or sealed"
                )
    if missing:
        raise SystemExit(
            f"the split has no entry for {len(missing)} video truth frames (first: {missing[0]}): ask the measurement role to add them, "
            "by hash, before any model sees them"
        )
    return dev


def dev_clip_names(table: Dict[str, str]) -> List[str]:
    """The clips the split marks dev: a clip is one unit, so one sealed frame makes the whole clip sealed and a clip split between the two
    parts is an error. The names come from the split, never from a typed list (RV-VK4 W1)."""
    parts: Dict[str, set] = {}
    for key, part in table.items():
        pieces = key.split("/")
        if len(pieces) == 3 and pieces[0] == "video":
            parts.setdefault(pieces[1], set()).add(part)
    split = sorted(n for n, p in parts.items() if len(p) > 1)
    if split:
        raise SystemExit(f"split: clip {split[0]} has frames in more than one part: a clip is one unit")
    return sorted(n for n, p in parts.items() if p == {"dev"})


def select_clips(truth: dict, dev: Dict[str, List[int]], names: Optional[Sequence[str]]) -> Dict[str, dict]:
    """The clips a tool may open: dev clips (`dev` = {clip: truth frames the split marks dev}) that have at least one labelled face, and, when
    `names` is given, only those. A name that is not a dev clip (sealed or unknown) stops the run: a sealed clip is never decoded (RV-VK4
    W1, W5)."""
    refused = [n for n in (names or ()) if n not in dev]
    if refused:
        raise SystemExit(f"clip {refused[0]} is not a dev clip of the split (sealed or unknown): it is not opened")
    return {
        n: truth["clips"][n]
        for n in dev
        if (not names or n in names)
        and any(not b.get("ignore") for f in truth["clips"][n]["frames"].values() for b in f["boxes"])
    }


def clip_fps(path: str) -> float:
    """The frame rate of a clip, from its container."""
    import av

    with av.open(path) as container:
        rate = container.streams.video[0].average_rate
    return float(rate) if rate else 25.0


def collect(
    clips_dir: str,
    truth_path: str,
    out: str,
    total_windows: int,
    names: Optional[Sequence[str]],
    threads: int,
    models: Optional[str],
    split_file: str,
    use_truth_frames: bool = False,
) -> dict:
    from anonymizer.classes.heavy import HeavyFaceModule
    from anonymizer.engine import deep
    from bench.results_meta import sha256_file

    truth = json.load(open(truth_path, encoding="utf-8"))
    with open(split_file, encoding="utf-8") as f:
        dev = dev_truth_frames(
            truth, json.load(f)
        )  # only the keys of the frames are read: a sealed clip is not touched
    clips = select_clips(truth, dev, names)
    truth_frames = {n: dev[n] for n in clips}
    module = HeavyFaceModule(threads=threads, models=models)

    def sweep(bgr) -> List[Detection]:
        return deep.deep_stage(bgr, [], [module], max_tiles=deep.NO_LIMIT).detections

    def zone_look(bgr, det) -> None:
        module.second_look(bgr, SuspectZone(det.box(), FACE, ZONE_DOUBTFUL, 0))

    windows = plan_windows(clips, total_windows, truth_frames)
    record = {
        "what": "VK4 raw record: windows of consecutive frames with their camera motion and the heavy detections at the grid offsets",
        "grid": list(GRID),
        "window": WINDOW,
        "machine": machine(),
        "heavy_model": {"name": module.heavy_zone.name, "sha256": module.heavy_zone.sha256},
        "thresholds": {"h_hi": module.hth.h_hi, "h_lo": module.hth.h_lo, "look_min": module.hth.look_min},
        "clips": {},
        "truth_frames": {},
        "truth_file_sha256": sha256_file(truth_path),
    }
    by_clip: Dict[str, List[int]] = {}
    for name, start in windows:
        by_clip.setdefault(name, []).append(start)
    for name, starts in sorted(by_clip.items()):
        info = clips[name]
        scale = 480 / info["width"]
        record["clips"][name] = {
            "width": info["width"],
            "height": info["height"],
            "total_frames": info["total_frames"],
            "fps": clip_fps(os.path.join(clips_dir, info["file"])),
            "windows": {},
        }
        for start, frames in read_clip_windows(os.path.join(clips_dir, info["file"]), starts):
            record["clips"][name]["windows"][str(start)] = window_record(frames, scale, sweep, GRID, zone_look)
            print(name, start, len(record["clips"][name]["windows"]), flush=True)
    record["split_sha256"] = sha256_file(split_file)
    record["dev_clips"] = sorted(clips)
    if use_truth_frames:
        record["truth_frames"] = collect_truth_frames(clips_dir, truth, clips, sweep, dev)
    every = [ms for clip in record["clips"].values() for w in clip["windows"].values() for ms in w["zone_ms"]]
    record["heavy_ms_per_zone_call"] = round(statistics.median(every), 1) if every else None
    record["heavy_ms_per_frame_sweep"] = (
        round(
            statistics.median(
                [ms for c in record["clips"].values() for w in c["windows"].values() for ms in w["sweep_ms"]]
            ),
            1,
        )
        if every
        else None
    )
    module.close()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f)
    return record


def collect_truth_frames(clips_dir: str, truth: dict, clips: dict, sweep, allowed: Dict[str, List[int]]) -> dict:
    """The heavy model on the dev truth frames only (`allowed`: {clip: frames}); nothing else of the truth is decoded or swept."""
    import av

    out: Dict[str, dict] = {}
    for name, info in clips.items():
        wanted = {
            int(k): v["boxes"] for k, v in truth["clips"][name]["frames"].items() if int(k) in allowed.get(name, ())
        }
        if not wanted:
            continue
        found = {}
        with av.open(os.path.join(clips_dir, info["file"])) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    dets = sweep(frame.to_ndarray(format="bgr24"))
                    found[str(index)] = {
                        "truth": [b["box"] for b in wanted[index] if not b.get("ignore")],
                        "heavy": [[d.x0, d.y0, d.x1, d.y1, d.score] for d in dets],
                    }
                if index >= max(wanted):
                    break
        out[name] = found
    return out


# --- evaluating (pure computation on the raw record) ---


def carry_box(box: Sequence[float], motions: Sequence[np.ndarray], start: int, end: int) -> Box:
    e = tk.box_to_state(box)
    for t in range(start + 1, end + 1):
        e = tk.carry(motions[t], e)
    return tk.state_to_box(e)


def same_face(a: Sequence[float], b: Sequence[float], motions, start: int, end: int, params: tk.TrackerParams) -> bool:
    """The tracker's join rule for a face seen at `start` (box `a`) and at `end` (box `b`): `a` carried by the camera motion falls on `b`."""
    expected = tk.box_to_state(carry_box(a, motions, start, end))
    cost = tk._cost(expected, Detection(*b[:4], 1.0, FACE))
    if cost is None:
        return False
    if iou(tk.state_to_box(expected), b[:4]) < params.iou_min and cost - 1 > params.dist_max:
        return False
    return True


def uncovered(heavy: Sequence[float], blur: Sequence[float]) -> float:
    """The share of the heavy box that the blur box does not cover."""
    ix = max(0.0, min(heavy[2], blur[2]) - max(heavy[0], blur[0]))
    iy = max(0.0, min(heavy[3], blur[3]) - max(heavy[1], blur[1]))
    area = max(1e-9, (heavy[2] - heavy[0]) * (heavy[3] - heavy[1]))
    return 1.0 - ix * iy / area


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    return None if not values else float(np.percentile(values, q))


def interpolate_pair(
    a: Sequence[float], b: Sequence[float], spacing: int, motions, width: int, height: int, fps: float, margins
) -> Dict[int, Box]:
    """The blur boxes of a face seen by the heavy model at frame 0 (box a) and at `spacing` (box b): the engine's own gap filling, margins and
    smoothing (`tracks.final_boxes`), frames 0 to `spacing`."""
    track = tk.Track(0, FACE)
    track.views[0] = Detection(*a[:4], a[4], FACE, SECOND)
    track.views[spacing] = Detection(*b[:4], b[4], FACE, SECOND)
    spec = tk.BlurSpec(tk.with_second_margin(margins), "ellipse")
    boxes = tk.final_boxes(
        [track], motions[: spacing + 1], set(), spacing + 1, fps, tk.TrackerParams(), width, height, {FACE: spec}
    )
    return {t: items[0][0] for t, items in boxes.items()}


def build_chains(window: dict, min_score: float, params: tk.TrackerParams) -> List[Dict[int, list]]:
    """The faces of a window followed through its heavy frames: {offset: box} per face. A face is followed from one grid frame to the next with
    the camera motion between them (the tracker's own join rule, one-to-one, best overlap first) and may be missed at one grid frame in a row;
    a heavy box that continues no face starts a new one. This is the reference: who is who comes from the heavy model's own frames, so that
    two faces passing each other in a crowd are not taken for one."""
    motions = [np.array(m, float).reshape(2, 3) for m in window["motions"]]
    offsets = sorted(int(k) for k in window["detections"])
    chains: List[dict] = []
    for offset in offsets:
        found = [d for d in window["detections"][str(offset)] if d[4] >= min_score]
        candidates = []
        for ci, chain in enumerate(chains):
            last = max(chain)
            if offset - last > 2 * MAX_STEP or last == offset:
                continue
            for di, d in enumerate(found):
                if same_face(chain[last], d, motions, last, offset, params):
                    candidates.append((1.0 - iou(carry_box(chain[last], motions, last, offset), d[:4]), ci, di))
        used_c, used_d = set(), set()
        for _, ci, di in sorted(candidates):
            if ci in used_c or di in used_d:
                continue
            chains[ci][offset] = found[di]
            used_c.add(ci)
            used_d.add(di)
        for di, d in enumerate(found):
            if di not in used_d:
                chains.append({offset: d})
    return chains


def evaluate_window(
    window: dict, spacing: int, width: int, height: int, fps: float, margins, min_score: float, params
) -> dict:
    """Cases of one window at one spacing: {"cases": [(distance, uncovered share)], "unmatched": n, "tracked": n, "key_faces": n}. Key frames
    are offsets 0 and `spacing`; a face counts when the heavy model's chain has it at both; the cases are its heavy boxes in between."""
    det = window["detections"]
    if str(spacing) not in det or "0" not in det:
        return {"cases": [], "unmatched": 0, "tracked": 0, "key_faces": 0}
    motions = [np.array(m, float).reshape(2, 3) for m in window["motions"]]
    chains = build_chains(window, min_score, params)
    key_faces = sum(1 for c in chains if 0 in c)
    spanning = [c for c in chains if 0 in c and spacing in c]
    cases, followed = [], 0
    for chain in spanning:
        blur = interpolate_pair(chain[0], chain[spacing], spacing, motions, width, height, fps, margins)
        for offset in GRID:
            if 0 < offset < spacing and offset in chain and offset in blur:
                cases.append((min(offset, spacing - offset), uncovered(chain[offset][:4], blur[offset])))
                followed += 1
    between = sum(1 for o in GRID if 0 < o < spacing for d in det.get(str(o), []) if d[4] >= min_score)
    return {"cases": cases, "unmatched": between - followed, "tracked": len(spanning), "key_faces": key_faces}


def box_second_margin(truth_frames: dict, min_iou: float = 0.3) -> Optional[dict]:
    """The margin the heavy box needs to contain the labelled box, per side as a fraction of its width and height: the 90th percentile over the
    faces the heavy model found. None without truth frames."""
    need_x, need_y = [], []
    for frames in truth_frames.values():
        for record in frames.values():
            for t in record["truth"]:
                best = max(record["heavy"], key=lambda h: iou(h[:4], t), default=None)
                if best is None or iou(best[:4], t) < min_iou:
                    continue
                w, h = max(1.0, best[2] - best[0]), max(1.0, best[3] - best[1])
                need_x.append(max(0.0, (best[0] - t[0]) / w, (t[2] - best[2]) / w))
                need_y.append(max(0.0, (best[1] - t[1]) / h, (t[3] - best[3]) / h))
    if not need_x:
        return None
    return {
        "n": len(need_x),
        "percentile": MARGIN_PERCENTILE,
        "x": round(percentile(need_x, MARGIN_PERCENTILE), 3),
        "y": round(percentile(need_y, MARGIN_PERCENTILE), 3),
        "median": [round(percentile(need_x, 50), 3), round(percentile(need_y, 50), 3)],
    }


def propose_heavy_per_second(zone_ms: Optional[float]) -> Optional[dict]:
    if not zone_ms:
        return None
    calls = int(HEAVY_SHARE_OF_REAL_TIME * 1000.0 / zone_ms)
    return {
        "value": max(1, min(12, calls)),
        "rule": f"at most {HEAVY_SHARE_OF_REAL_TIME} s of heavy work per second of video",
        "zone_ms": zone_ms,
    }


def largest_holding_prefix(holds: Dict[int, bool], spacings: Sequence[int] = SPACINGS) -> Optional[int]:
    """The largest spacing such that it and every smaller candidate hold: a spacing above one that fails is not chosen even if it passes
    by luck on its own cases (RV-VK4 W3)."""
    chosen = None
    for n in sorted(spacings):
        if not holds.get(n, False):
            break
        chosen = n
    return chosen


def clip_rates(raw: dict, fps: float, given: Optional[Dict[str, float]]) -> Dict[str, dict]:
    """The frame rate of every clip and where it comes from: recorded by `collect`, given by the caller (read from the clip file), or assumed
    (`fps`, the old raw files that did not record it). N is in frames; the claim is in seconds (RV-VK4 W4)."""
    out = {}
    for name, clip in raw["clips"].items():
        if clip.get("fps"):
            out[name] = {"fps": float(clip["fps"]), "source": "recorded"}
        elif given and name in given:
            out[name] = {"fps": float(given[name]), "source": "clip file"}
        else:
            out[name] = {"fps": float(fps), "source": "assumed"}
    return out


def tail_of(shares: Sequence[float]) -> dict:
    """What a percentile hides: the 99th percentile, the maximum, and the share of cases left more than 10 % and more than 50 % uncovered."""
    if not shares:
        return {"p99": None, "max": None, "over_10pct": None, "over_50pct": None}
    return {
        "p99": round(percentile(shares, 99), 4),
        "max": round(max(shares), 4),
        "over_10pct": round(sum(1 for x in shares if x > 0.10) / len(shares), 4),
        "over_50pct": round(sum(1 for x in shares if x > 0.50) / len(shares), 4),
    }


def evaluate(
    raw: dict,
    min_score: Optional[float] = None,
    fps: float = 25.0,
    given_fps: Optional[Dict[str, float]] = None,
    second_margin: Optional[Tuple[float, float]] = None,
) -> dict:
    """The replay. `second_margin` (x, y) replaces the placeholder BOX_SECOND margin of the faces (`None`: the engine's own, as
    `FaceModule.video_margins` gives it); every clip is replayed with its own frame rate."""
    from anonymizer.classes.faces import FaceModule

    min_score = raw["thresholds"]["h_lo"] if min_score is None else min_score
    margins = dict(FaceModule.video_margins) if hasattr(FaceModule, "video_margins") else {}
    if second_margin is not None:
        margins[SECOND_ORIGIN] = (float(second_margin[0]), float(second_margin[1]))
    rates = clip_rates(raw, fps, given_fps)
    params = tk.TrackerParams()
    by_spacing: Dict[int, dict] = {
        n: {"cases": [], "unmatched": 0, "tracked": 0, "key_faces": 0, "windows": 0, "by_clip": {}} for n in SPACINGS
    }
    for name, clip in raw["clips"].items():
        for window in clip["windows"].values():
            for n in SPACINGS:
                got = evaluate_window(
                    window, n, clip["width"], clip["height"], rates[name]["fps"], margins, min_score, params
                )
                cell = by_spacing[n]
                cell["cases"] += got["cases"]
                cell["unmatched"] += got["unmatched"]
                cell["tracked"] += got["tracked"]
                cell["key_faces"] += got["key_faces"]
                cell["windows"] += 1
                cell["by_clip"].setdefault(name, []).extend(c[1] for c in got["cases"])
    table, holds = {}, {}
    for n in SPACINGS:
        cell = by_spacing[n]
        shares = [c[1] for c in cell["cases"]]
        distances: Dict[int, List[float]] = {}
        for d, s in cell["cases"]:
            distances.setdefault(d, []).append(s)
        p95 = percentile(shares, 95)
        pair_rate = cell["tracked"] / cell["key_faces"] if cell["key_faces"] else 0.0
        seconds = [n / rates[name]["fps"] for name in raw["clips"]]
        table[str(n)] = {
            "n": len(shares),
            "seconds": {"min": round(min(seconds), 3), "max": round(max(seconds), 3)},
            "tracked_pairs": cell["tracked"],
            "pair_rate": round(pair_rate, 3),
            "unmatched_heavy_boxes": cell["unmatched"],
            "mean": None if not shares else round(float(np.mean(shares)), 4),
            "p50": None if not shares else round(percentile(shares, 50), 4),
            "p95": None if p95 is None else round(p95, 4),
            **tail_of(shares),
            "by_distance": {
                str(d): {"n": len(v), "p95": round(percentile(v, 95), 4)} for d, v in sorted(distances.items())
            },
            "by_clip": {
                k: {"n": len(v), "p95": round(percentile(v, 95), 4), **tail_of(v)}
                for k, v in sorted(cell["by_clip"].items())
                if v
            },
            "holds": bool(
                len(shares) >= MIN_CASES and pair_rate >= MIN_PAIR_RATE and p95 is not None and p95 <= P95_LIMIT
            ),
        }
        holds[n] = table[str(n)]["holds"]
    chosen = largest_holding_prefix(holds)
    return {
        "what": "VK4: share of the heavy box left uncovered by the blur interpolated between two heavy key frames",
        "min_score": min_score,
        "p95_limit": P95_LIMIT,
        "min_cases": MIN_CASES,
        "min_pair_rate": MIN_PAIR_RATE,
        "spacings": table,
        "period_proposal": chosen,
        "period_proposal_seconds": None
        if chosen is None
        else {name: round(chosen / r["fps"], 3) for name, r in rates.items()},
        "claim_made": chosen is not None,
        "claim": claim_text(table, chosen, rates),
        "clip_rates": rates,
        "margins_used": {
            "second": list(margins[SECOND_ORIGIN]) if SECOND_ORIGIN in margins else "halfway between sweep and retry"
        },
        "box_second_margin": box_second_margin(raw.get("truth_frames", {})),
        "heavy_per_second_proposal": propose_heavy_per_second(raw.get("heavy_ms_per_zone_call")),
        "machine": raw["machine"],
        "heavy_model": raw["heavy_model"],
        "windows": sum(len(c["windows"]) for c in raw["clips"].values()),
        "caveat": "The reference is the heavy model itself (not a labelled truth), 8 clips: results are per clip as well as pooled, and the pooled interval is not a bootstrap by clip. If no spacing holds, the margins are widened and the claim is not made (VK4).",
    }


def claim_text(table: Dict[str, dict], chosen: Optional[int], rates: Dict[str, dict]) -> str:
    """The sentence the claim may be quoted as: pooled figures, the tail, and every clip that alone is over the limit."""
    if chosen is None:
        return "No spacing holds: the claim is not made (VK4)."
    cell = table[str(chosen)]
    secs = cell["seconds"]
    over = [f"{k} {100 * v['p95']:.1f} %" for k, v in cell["by_clip"].items() if v["p95"] > P95_LIMIT]
    text = (
        f"A key frame every {chosen} frames ({secs['min']}-{secs['max']} s by clip frame rate) holds pooled over {len(rates)} clips: "
        f"p95 {100 * cell['p95']:.1f} %, p99 {100 * cell['p99']:.1f} %, max {100 * cell['max']:.1f} %, "
        f"{100 * cell['over_10pct']:.1f} % of the cases more than 10 % uncovered, {100 * cell['over_50pct']:.1f} % more than 50 %."
    )
    return text + (f" Clips that alone are over the {100 * P95_LIMIT:.0f} % limit: {', '.join(over)}." if over else "")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--clips", required=True)
    c.add_argument("--truth-video", required=True)
    c.add_argument("--clip-names", default="")
    c.add_argument("--windows", type=int, default=20)
    c.add_argument("--threads", type=int, default=0)
    c.add_argument("--models", default=None)
    c.add_argument(
        "--split-file", required=True, help="the flat split (split-v2-parts.json): only its dev clips are used"
    )
    c.add_argument(
        "--truth-frames",
        action="store_true",
        help="also run the heavy model on the dev truth frames (BOX_SECOND margin)",
    )
    c.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--raw", required=True)
    e.add_argument("--min-score", type=float, default=None)
    e.add_argument(
        "--fps", type=float, default=25.0, help="for clips that recorded no frame rate and have no --clips file"
    )
    e.add_argument(
        "--clips", default=None, help="the clips folder: read the frame rate of a clip the raw file does not record"
    )
    e.add_argument("--truth-video", default=None, help="with --clips: the file names of the clips")
    e.add_argument(
        "--second-margin",
        default=None,
        help="x,y of BOX_SECOND to replay with ('measured': the 90th percentile of the raw file's truth frames)",
    )
    e.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.command == "collect":
        names = [n for n in args.clip_names.split(",") if n]
        collect(
            args.clips,
            args.truth_video,
            args.out,
            args.windows,
            names,
            args.threads,
            args.models,
            args.split_file,
            args.truth_frames,
        )
        return 0
    raw = json.load(open(args.raw, encoding="utf-8"))
    given = None
    if args.clips:
        files = {n: raw["clips"][n].get("file") or f"{n}.webm" for n in raw["clips"]}
        if args.truth_video:
            files = {n: json.load(open(args.truth_video, encoding="utf-8"))["clips"][n]["file"] for n in raw["clips"]}
        given = {n: clip_fps(os.path.join(args.clips, f)) for n, f in files.items()}
    margin = None
    if args.second_margin == "measured":
        got = box_second_margin(raw.get("truth_frames", {}))
        margin = (got["x"], got["y"]) if got else None
    elif args.second_margin:
        margin = tuple(float(v) for v in args.second_margin.split(","))
    result = evaluate(raw, args.min_score, args.fps, given, margin)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
