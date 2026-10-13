"""The kinds of file the anonymizer takes, and the fingerprint that says a source is still the file that was analysed.

Of the folder job ported from the earlier application nothing is left: the Studio and the API do that work with the same
guarantees (hidden partial file, checked, renamed; the source checked again before the final name), and a second render path
that nobody used was a place for the first one's guarantees to go missing.
"""

from __future__ import annotations

import hashlib
import os

PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".jfif", ".jpe", ".mpo", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif")
"""Extensions the API takes as photos: a tuple of lower-case strings with the dot, such as `".jpg"` and `".avif"`.
The kind of a file is told from its extension only, never from its content. Test with `os.path.splitext(name)[1].lower()`.
"""
VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".avi",
    ".webm",
    ".3gp",
    ".mts",
    ".m2ts",
    ".ts",
    ".m2t",
    ".wmv",
    ".ogv",
    ".mpg",
    ".mpeg",
    ".flv",
)
"""Extensions the API takes as videos: a tuple of lower-case strings with the dot, such as `".mp4"` and `".mkv"`.
The kind of a file is told from its extension only. Videos are decoded by PyAV, which carries FFmpeg.
"""
# Listed and reported, never silently skipped: nobody may believe these were blurred.
UNSUPPORTED_EXTENSIONS = (
    ".heic",
    ".heif",
    ".dng",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".orf",
    ".rw2",
    ".gif",
    ".jp2",
    ".j2k",
    ".psd",
    ".svg",
)
"""Extensions that are known and refused: HEIC and HEIF, camera RAW, GIF, JPEG 2000, PSD and SVG.
A tuple of lower-case strings with the dot. A file with one of them raises `UnsupportedMedia` at once and is never
skipped silently, so that nobody believes it was blurred. Convert it to a supported format first.
"""

SAMPLE = 64 * 1024
SAMPLES_IN_BETWEEN = 8


def source_fingerprint(path: str) -> str:
    """Size, modification time to the nanosecond and a hash of the content. A photo is hashed whole (a few megabytes: a retouched copy
    that keeps size and time cannot pass for the file that was analysed). A video is sampled, at the start, the end and eight places in
    between: reading about a megabyte is cheap even for a long film, and a copy that keeps the time and the size differs in those bytes."""
    st = os.stat(path)
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        if os.path.splitext(path)[1].lower() in PHOTO_EXTENSIONS:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        else:
            digest.update(f.read(SAMPLE))
            if st.st_size > 2 * SAMPLE:
                for k in range(1, SAMPLES_IN_BETWEEN + 1):
                    f.seek(st.st_size * k // (SAMPLES_IN_BETWEEN + 1))
                    digest.update(f.read(SAMPLE // 8))
                f.seek(st.st_size - SAMPLE)
                digest.update(f.read(SAMPLE))
    return f"{st.st_size}-{st.st_mtime_ns}-{digest.hexdigest()[:16]}"
