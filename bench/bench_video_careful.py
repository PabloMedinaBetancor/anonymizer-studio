"""VK2: the real heavy face model at the key frames of a video, on the public clips, next to the light-only run.

    python3 -s -m bench.bench_video_careful --clips DIR --truth-video FILE --split-file SPLIT [--clip-names a,b] [--seconds 20] [--threads 4] [--out FILE]

For each dev clip (its first `--seconds`, cut to a temporary file) the engine runs twice with the real models: `fast` (the light face and plate
models only, v1's behaviour) and `careful` (the heavy face model at the key frames, the person finder for the heads). The file
records, per clip and mode, the wall time, the tracks listed and who found them (`found_by`), the heavy calls the key frames made and the
ones the budget refused (`second`), the frames that could not be checked, and the peak memory. It is a diagnostic of the integration (does
the budget hold, do heavy views reach the tracks, what does it cost): the coverage figures of VK8 are measured on the sealed truth by the
measurement role, never here. The file names the machine and the load. Like the interpolation tool it reads the split (`--split-file`): only the clips the split marks dev,
that have a labelled face, are opened; a name the split marks sealed stops the run (RV-VK4 W1). `seconds` of a row includes loading the
models; `peak_rss_mb` is the largest of the process so far, not of that run; the cut is re-encoded (CRF 18), so the run sees a slightly
different video than the clip.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from typing import Dict, Optional, Sequence


def trim(source: str, target: str, seconds: float) -> int:
    """Write the first `seconds` of `source` to `target` (H.264, no audio); returns the frames written."""
    import av

    written = 0
    with av.open(source) as src, av.open(target, "w") as out:
        stream = src.streams.video[0]
        rate = stream.average_rate or 25
        video = out.add_stream("libx264", rate=rate)
        video.width, video.height, video.pix_fmt = stream.codec_context.width, stream.codec_context.height, "yuv420p"
        video.options = {"crf": "18", "preset": "ultrafast"}
        limit = int(seconds * float(rate))
        for frame in src.decode(video=0):
            if written >= limit:
                break
            out_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")
            out_frame.pts = written
            for packet in video.encode(out_frame):
                out.mux(packet)
            written += 1
        for packet in video.encode(None):
            out.mux(packet)
    return written


def run_clip(path: str, detection: str, threads: int, models: Optional[str]) -> Dict[str, object]:
    import anonymizer

    started = time.perf_counter()
    options = anonymizer.Options(detection=detection)
    with anonymizer.Anonymizer(options, models_dir=models, threads=threads) as anon:
        analysis = anon.analyze(path)
        loaded = time.perf_counter() - started
        found_by: Dict[str, int] = {}
        for item in analysis.items:
            found_by[item.found_by] = found_by.get(item.found_by, 0) + 1
        va = analysis._video
        return {
            "seconds": round(loaded, 1),
            "frames": va.frames,
            "fps": va.fps,
            "tracks": len(analysis.items),
            "by_kind": {k: sum(1 for i in analysis.items if i.kind == k) for k in {i.kind for i in analysis.items}},
            "by_verdict": {
                v: sum(1 for i in analysis.items if i.verdict == v) for v in {i.verdict for i in analysis.items}
            },
            "found_by": found_by,
            "heavy_views": sum(t.n_second for t in va.tracks),
            "second": dict(va.second),
            "unchecked_frames": len(va.unchecked),
            "options": analysis.options,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    from bench.bench_cells import machine

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", required=True)
    ap.add_argument(
        "--truth-video", required=True, help="the video truth file: its clip names, files and labelled faces"
    )
    ap.add_argument(
        "--split-file", required=True, help="the flat split (split-v2-parts.json): only its dev clips are opened"
    )
    ap.add_argument("--clip-names", default="")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from bench.bench_video_interpolation import dev_truth_frames, select_clips

    with open(args.truth_video, encoding="utf-8") as f:
        truth = json.load(f)
    with open(args.split_file, encoding="utf-8") as f:
        dev = dev_truth_frames(
            truth, json.load(f)
        )  # only the keys of the frames are read: a sealed clip is not touched
    clips = select_clips(truth, dev, [n for n in args.clip_names.split(",") if n])
    result: Dict[str, object] = {
        "what": "VK2 integration run: fast against careful on public clips, real models",
        "machine": machine(),
        "seconds_per_clip": args.seconds,
        "dev_clips": sorted(clips),
        "clips": {},
    }
    for name in sorted(clips):
        with tempfile.TemporaryDirectory() as tmp:
            piece = os.path.join(tmp, f"{name}.mp4")
            frames = trim(os.path.join(args.clips, clips[name]["file"]), piece, args.seconds)
            result["clips"][name] = {"frames_cut": frames}
            for detection in ("fast", "careful"):
                result["clips"][name][detection] = run_clip(piece, detection, args.threads, args.models)
                print(name, detection, result["clips"][name][detection]["seconds"], flush=True)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
