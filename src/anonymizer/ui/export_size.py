"""How big the anonymized file will be, as a range: the Export screen's "about" figure.

The file is written again, not copied: video is re-encoded (H.264, crf 20) and a photo is saved again, so its size follows the
picture and not the original file (a heavily compressed original comes out two or three times larger, a high-bitrate phone video
smaller). `scripts/measure_export_size.py` measures it; on the four demo files (2 videos, 2 photos) the output was 0.039 and 0.081
bits per pixel and frame for the videos and 1.3 and 1.4 times the original for the photos. The ranges below are wider than those
points on purpose: an estimate shown as one number was 2.5 times too low on the videos (QA report, 04/10/2026).
"""

from __future__ import annotations

from typing import Tuple

VIDEO_BITS_PER_PIXEL_FRAME = (0.025, 0.12)
AUDIO_BYTES_PER_SECOND = 16_000  # 128 kbit/s, copied as it is
PHOTO_RATIO = (0.5, 1.6)


def estimate(info) -> Tuple[int, int]:
    """(low, high) bytes for the output of `info` (an anonymizer.MediaInfo)."""
    if info.media == "photo":
        low, high = PHOTO_RATIO
        return int(info.size_bytes * low), int(info.size_bytes * high)
    pixels = info.width * info.height * max(1, info.frames)
    audio = AUDIO_BYTES_PER_SECOND * info.duration_s if info.has_audio else 0
    low, high = VIDEO_BITS_PER_PIXEL_FRAME
    return int(pixels * low / 8 + audio), int(pixels * high / 8 + audio)
