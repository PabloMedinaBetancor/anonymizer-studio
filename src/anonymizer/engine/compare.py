"""The comparison video: the original on the left, the anonymized video on the right, each with a caption,
as long as the original and with its audio.

Written like a render: to a hidden partial file, checked by decoding it whole, then renamed. The captions are drawn on
the planes of the frame (luma only, the bar's colour is neutral), so no frame goes through an RGB conversion.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Callable, Optional, Tuple

import numpy as np

from . import video_io
from .events import ClipError, StopFlag
from .video import partial_path, release_memory

CAPTION_BAR = 0.35  # the bar keeps this share of the picture's brightness under the text
TEXT_LEVEL = 235  # white in limited-range luma
Progress = Callable[[float, int, int], None]


def draw_caption(y: np.ndarray, u: np.ndarray, v: np.ndarray, x0: int, width: int, text: str) -> None:
    """A dark bar along the top of the picture that starts at column `x0` and is `width` wide, with `text` in it."""
    import cv2

    bar = max(16, y.shape[0] // 12)
    bar -= bar % 2
    y[:bar, x0 : x0 + width] = (y[:bar, x0 : x0 + width] * CAPTION_BAR).astype(np.uint8)
    u[: bar // 2, x0 // 2 : (x0 + width) // 2] = 128
    v[: bar // 2, x0 // 2 : (x0 + width) // 2] = 128
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = bar / 40.0
    thickness = max(1, round(scale * 2))
    while scale > 0.2 and cv2.getTextSize(text, font, scale, thickness)[0][0] > width - 16:
        scale *= 0.9
        thickness = max(1, round(scale * 2))
    (_, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    origin = (x0 + 8, (bar + text_height - baseline) // 2)
    cv2.putText(y, text, origin, font, scale, TEXT_LEVEL, thickness, cv2.LINE_AA)


def compare_video(
    original: str,
    anonymized: str,
    target: str,
    captions: Tuple[str, str] = ("Original", "Anonymized"),
    progress: Optional[Progress] = None,
    stop: Optional[StopFlag] = None,
    threads: int = 0,
) -> int:
    """Write `target` and return the number of frames in it. Both videos must have the same size and frame count."""
    if os.path.exists(target):
        raise ClipError("error.output.exists", {"path": target})
    info = video_io.probe(original)
    other = video_io.probe(anonymized)
    if (other.width, other.height) != (info.width, info.height):
        raise ClipError("error.output.check_failed", {"path": target}, "the two videos differ in size")
    wide = replace(info, width=info.width * 2)
    total = max(1, info.frames)
    partial = partial_path(target)
    try:
        with video_io.Writer(original, partial, wide, threads=threads) as writer:
            others = video_io.frames(anonymized)
            for t, frame, rotation in writer.frames():
                if stop is not None:
                    stop.check()
                twin = next(others, None)
                if twin is None:
                    raise ClipError("error.output.check_failed", {"path": target}, "the anonymized video is shorter")
                left = video_io.yuv420_planes(frame, rotation)
                right = video_io.yuv420_planes(twin[1], twin[2])
                y, u, v = (np.ascontiguousarray(np.hstack(pair)) for pair in zip(left, right, strict=True))
                draw_caption(y, u, v, 0, info.width, captions[0])
                draw_caption(y, u, v, info.width, info.width, captions[1])
                writer.write(frame, y, u, v)
                if t % 15 == 0:
                    if progress is not None:
                        progress(min(1.0, (t + 1) / total), t, total)
                    if t % 300 == 0:
                        release_memory()
            if next(others, None) is not None:
                raise ClipError("error.output.check_failed", {"path": target}, "the anonymized video is longer")
        written, duration = writer.written, writer.written_duration
        problems = video_io.check_output(partial, wide, written, duration)
        if problems:
            raise ClipError("error.output.check_failed", {"path": target}, "; ".join(problems))
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return written
