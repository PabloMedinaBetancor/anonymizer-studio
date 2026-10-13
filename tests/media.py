"""Synthetic videos for the engine tests, written with PyAV (no ffmpeg binary). Ported from the helpers of the 1.04 tests,
so the same cases are checked."""

from __future__ import annotations

import fractions
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


def plate_box(t: int) -> Box:
    """Where moving_plate_clip draws its plate at frame t, with a 3 px margin."""
    return (20 + 2 * t - 3, 97, 20 + 2 * t + 103, 133)


def moving_plate_clip(path: str, n: int = 60, width: int = 320, height: int = 240, fps: int = 30, audio: bool = True):
    """A white plate reading "ABC 123" that moves 2 px per frame over a striped background, and a 440 Hz tone."""
    import av
    import cv2

    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = width, height, "yuv420p"
        vs.options = {"crf": "18", "preset": "ultrafast"}
        aus = None
        if audio:
            aus = out.add_stream("aac", rate=44100)
            aus.layout = "mono"
        samples = 0
        for t in range(n):
            img = np.full((height, width, 3), (60, 90, 120), np.uint8)
            img[::16, :] = 200  # stripes give the optical flow something to hold on to
            img[:, ::20] = 30
            x = 20 + 2 * t
            cv2.rectangle(img, (x, 100), (x + 100, 130), (250, 250, 250), -1)
            cv2.putText(img, "ABC 123", (x + 4, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = t, fractions.Fraction(1, fps)
            for p in vs.encode(frame):
                out.mux(p)
            while aus is not None and samples < (t + 1) * 44100 // fps:
                k = 1024
                tone = (0.2 * np.sin(2 * np.pi * 440 * (np.arange(samples, samples + k) / 44100))).astype(np.float32)
                af = av.AudioFrame.from_ndarray(tone[None, :], format="fltp", layout="mono")
                af.sample_rate, af.pts = 44100, samples
                for p in aus.encode(af):
                    out.mux(p)
                samples += k
        for stream in (vs, aus):
            for p in stream.encode(None) if stream is not None else ():
                out.mux(p)


def odd_mjpeg_avi(path: str, n: int = 20):
    """MJPEG at 321x241, full range (yuvj420p), with PCM sound that MP4 cannot hold as it is."""
    import av

    with av.open(path, "w") as out:
        vs = out.add_stream("mjpeg", rate=25)
        vs.width, vs.height, vs.pix_fmt = 321, 241, "yuvj420p"
        aus = out.add_stream("pcm_s16le", rate=16000)
        aus.layout = "mono"
        for t in range(n):
            img = np.full((241, 321, 3), 40 + 5 * t, np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuvj420p")
            frame.pts, frame.time_base = t, fractions.Fraction(1, 25)
            for p in vs.encode(frame):
                out.mux(p)
            wave = (np.sin(np.arange(640) / 5) * 8000).astype(np.int16)[None, :]
            af = av.AudioFrame.from_ndarray(wave, format="s16", layout="mono")
            af.sample_rate, af.pts = 16000, t * 640
            for p in aus.encode(af):
                out.mux(p)
        for stream in (vs, aus):
            for p in stream.encode(None):
                out.mux(p)


def full_range_mp4(path: str, n: int = 12):
    """yuv420p tagged as full range, as some Android phones write it."""
    import av

    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=25)
        vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
        vs.codec_context.color_range = 2
        for t in range(n):
            frame = av.VideoFrame.from_ndarray(np.full((120, 160, 3), 30 + 15 * t, np.uint8), format="bgr24")
            frame = frame.reformat(format="yuv420p", dst_color_range=2)
            frame.pts, frame.time_base = t, fractions.Fraction(1, 25)
            for p in vs.encode(frame):
                out.mux(p)
        for p in vs.encode(None):
            out.mux(p)


def short_clip(
    path: str,
    n: int = 20,
    container: Optional[str] = None,
    audio: Sequence[Tuple[str, float]] = (("aac", 0.7),),
    start: float = 0.0,
    options: Optional[Dict[str, str]] = None,
):
    """n frames at 25 fps from `start` seconds, and one stereo track per (codec, seconds) of `audio`, also from
    `start`."""
    import av

    out = av.open(path, "w", format=container, options=options or {})
    vs = out.add_stream("libx264" if container != "mpegts" else "mpeg2video", rate=25)
    vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p"
    vs.codec_context.time_base = fractions.Fraction(1, 25)
    tracks = []
    for codec, seconds in audio:
        aus = out.add_stream(codec, rate=44100)
        aus.layout = "stereo"
        tracks.append((aus, seconds))
    for t in range(n):
        img = np.full((120, 160, 3), (40 + 7 * t) % 255, np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        frame.pts, frame.time_base = int(round(start * 25)) + t, fractions.Fraction(1, 25)
        for p in vs.encode(frame):
            out.mux(p)
    for p in vs.encode(None):
        out.mux(p)
    for aus, seconds in tracks:
        size, rate, done, total = aus.codec_context.frame_size or 1152, 44100, 0, int(seconds * 44100)
        fmt = aus.codec_context.format.name
        while done < total:
            m = min(size, total - done)
            wave = (0.2 * np.sin(np.arange(done, done + m) / rate * 2 * np.pi * 440)).astype(np.float32)
            arr = np.tile(wave, (2, 1)) if fmt.endswith("p") else np.tile(wave[:, None], (1, 2)).reshape(1, -1)
            if fmt.startswith("s16"):
                arr = (arr * 32767).astype(np.int16)
            af = av.AudioFrame.from_ndarray(arr, format=fmt, layout="stereo")
            af.sample_rate, af.pts, af.time_base = rate, int(round(start * rate)) + done, fractions.Fraction(1, rate)
            for p in aus.encode(af):
                out.mux(p)
            done += m
        for p in aus.encode(None):
            out.mux(p)
    out.close()


def stream_starts(path: str) -> Dict[str, float]:
    """Time of the first packet of each stream type, in seconds."""
    import av

    starts: Dict[str, float] = {}
    with av.open(path) as container:
        for p in container.demux():
            if p.pts is not None and p.stream.type not in starts:
                starts[p.stream.type] = float(p.pts * p.stream.time_base)
    return starts


def decoded_frame(path: str, index: int) -> Optional[np.ndarray]:
    from anonymizer.engine import video_io

    for i, frame, rotation in video_io.frames(path):
        if i == index:
            return video_io.bgr_of(frame, rotation)
    return None


def rewrite(source: str, target: str, boxes: Callable[[int], Sequence[Box]] = lambda t: (), **blur_options):
    """Run a clip through Writer, blurring `boxes(t)` on Y and on U, V at half size, as a render does."""
    from anonymizer.engine import video_io
    from anonymizer.engine.blur import blur

    info = video_io.probe(source)
    with video_io.Writer(source, target, info) as writer:
        for t, frame, rotation in writer.frames():
            y, u, v = video_io.yuv420_planes(frame, rotation)
            for box in boxes(t):
                half = [c / 2 for c in box]
                blur(y, box, **blur_options)
                blur(u, half, **blur_options)
                blur(v, half, **blur_options)
            writer.write(frame, y, u, v)
    return info, writer


def panning_clip(path: str, n: int = 40, width: int = 320, height: int = 240, fps: int = 25, speed: int = 3):
    """A still scene filmed by a camera panning right `speed` px per frame: textured background and a parked plate
    at scene x 120-220, y 100-130 (in frame t it is at x - speed * t)."""
    import av
    import cv2

    rng = np.random.default_rng(5)
    scene = rng.integers(40, 200, (height // 8, (width + speed * n) // 8 + 1, 3), dtype=np.uint8)
    scene = cv2.resize(scene, (scene.shape[1] * 8, height), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(scene, (120, 100), (220, 130), (250, 250, 250), -1)
    cv2.putText(scene, "ABC 123", (124, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = width, height, "yuv420p"
        vs.options = {"crf": "18", "preset": "ultrafast"}
        for t in range(n):
            crop = np.ascontiguousarray(scene[:, speed * t : speed * t + width])
            frame = av.VideoFrame.from_ndarray(crop, format="bgr24")
            frame.pts, frame.time_base = t, fractions.Fraction(1, fps)
            for p in vs.encode(frame):
                out.mux(p)
        for p in vs.encode(None):
            out.mux(p)


def ten_bit_clip(path: str, n: int = 12, level: int = 120):
    """Lossless 10-bit 4:2:0 video (FFV1 in Matroska), a flat grey `level` with a white plate, as HDR phones give."""
    import av
    import cv2

    with av.open(path, "w", format="matroska") as out:
        vs = out.add_stream("ffv1", rate=25)
        vs.width, vs.height, vs.pix_fmt = 160, 120, "yuv420p10le"
        for t in range(n):
            img = np.full((120, 160, 3), level, np.uint8)
            cv2.rectangle(img, (40, 50), (120, 70), (250, 250, 250), -1)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24").reformat(format="yuv420p10le")
            frame.pts, frame.time_base = t, fractions.Fraction(1, 25)
            for p in vs.encode(frame):
                out.mux(p)
        for p in vs.encode(None):
            out.mux(p)


def probe_size(path: str) -> Tuple[int, int]:
    """(width, height) of a video, from its container."""
    from anonymizer.engine import video_io

    info = video_io.probe(path)
    return info.width, info.height
