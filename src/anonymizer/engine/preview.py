"""One frame of a clip as Review shows it: the original, the blurred result, the boxes, or both halves side by side."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

ORIGINAL, ANONYMIZED, BOXES, SPLIT = "original", "anonymized", "boxes", "split"
VIEWS = (ORIGINAL, ANONYMIZED, BOXES, SPLIT)
JPEG_QUALITY = 85
MIN_WIDTH = 160
DASH = 8
COLOURS = {"face": (80, 200, 120), "plate": (60, 170, 240), "manual": (230, 140, 90)}  # BGR


def source_bgr(path: str, media: str, frame: int) -> Optional[np.ndarray]:
    """The upright BGR frame of a video, or the whole photo (frame 0); None past the end of a video."""
    import cv2

    from . import photos, video_io

    if media == "video":
        return video_io.read_frame(path, frame)
    arr, _ = photos._to_rgb(photos.read(path)[0])
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def draw_boxes(bgr: np.ndarray, boxes: Sequence[Tuple[str, Tuple[float, float, float, float], str, bool]]) -> None:
    """Outline each (kind, box, shape, blurred) in place: solid for items that are blurred, dashed for "Not sure"."""
    import cv2

    thickness = max(2, round(max(bgr.shape[:2]) / 400))
    for kind, (x0, y0, x1, y1), shape, blurred in boxes:
        colour = COLOURS.get(kind, COLOURS["manual"])
        a, b = (int(x0), int(y0)), (int(x1), int(y1))
        if blurred:
            if shape == "ellipse":
                centre = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                cv2.ellipse(bgr, centre, ((b[0] - a[0]) // 2, (b[1] - a[1]) // 2), 0, 0, 360, colour, thickness)
            else:
                cv2.rectangle(bgr, a, b, colour, thickness)
        else:
            _dashed_rectangle(bgr, a, b, colour, thickness)


def _dashed_rectangle(bgr: np.ndarray, a, b, colour, thickness: int) -> None:
    import cv2

    step = DASH * thickness
    for x in range(a[0], b[0], 2 * step):
        end = min(x + step, b[0])
        cv2.line(bgr, (x, a[1]), (end, a[1]), colour, thickness)
        cv2.line(bgr, (x, b[1]), (end, b[1]), colour, thickness)
    for y in range(a[1], b[1], 2 * step):
        end = min(y + step, b[1])
        cv2.line(bgr, (a[0], y), (a[0], end), colour, thickness)
        cv2.line(bgr, (b[0], y), (b[0], end), colour, thickness)


def split(original: np.ndarray, anonymized: np.ndarray) -> np.ndarray:
    """The left half as it was, the right half anonymized, and a line between them."""
    out = anonymized.copy()
    middle = out.shape[1] // 2
    out[:, :middle] = original[:, :middle]
    out[:, max(0, middle - 1) : middle + 1] = (255, 255, 255)
    return out


def encode_jpeg(bgr: np.ndarray, width: int) -> bytes:
    """JPEG bytes, scaled down to `width` pixels when the frame is wider (never scaled up)."""
    import cv2

    width = max(MIN_WIDTH, int(width))
    height, source_width = bgr.shape[:2]
    if source_width > width:
        bgr = cv2.resize(bgr, (width, max(1, round(height * width / source_width))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("the frame could not be encoded as JPEG")
    return buffer.tobytes()
