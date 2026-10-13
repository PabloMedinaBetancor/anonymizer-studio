"""What the Studio says about a file at the door: why a format is not taken, and whether the file can really be read.

A file is refused when it is imported, with its reason in the person's words, so that nothing broken reaches the list of
clips (a truncated JPEG used to be accepted, then stopped the analysis with "not a supported file"). The checks read the
file with the engine's own readers, so a file that passes here is one the analysis can open; they decide nothing about
what is found in it.
"""

from __future__ import annotations

import os
import re
from typing import Optional

import anonymizer

# The reason of each format the engine names as not supported (anonymizer.UNSUPPORTED_EXTENSIONS), as a translation key.
HEIC = (".heic", ".heif")
RAW = (".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2")
GIF = (".gif",)


def unsupported_code(name: str) -> str:
    """The translation key that says why `name` is not taken: its format when it has a reason of its own, else the plain one."""
    ext = os.path.splitext(name)[1].lower()
    if ext in HEIC:
        return "error.media.heic"
    if ext in RAW:
        return "error.media.raw"
    if ext in GIF:
        return "error.media.gif"
    if ext in anonymizer.UNSUPPORTED_EXTENSIONS:
        return "error.media.format"
    return "error.media.unsupported"


def unsupported_error(path: str) -> "anonymizer.UnsupportedMedia":
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    return anonymizer.UnsupportedMedia(unsupported_code(name), {"name": name, "extension": ext or "-"})


UNREADABLE = {"photo": "error.media.unreadable_photo", "video": "error.media.unreadable_video"}


def unreadable(path: str, media: str, detail: str = "") -> "anonymizer.UnsupportedMedia":
    if os.path.exists(path) and not os.access(
        path, os.R_OK
    ):  # it is the computer that says no, not the file that is broken
        return anonymizer.UnsupportedMedia("error.media.denied", {"name": os.path.basename(path)}, detail)
    return anonymizer.UnsupportedMedia(UNREADABLE[media], {"name": os.path.basename(path)}, detail)


def image_count(path: str) -> int:
    """How many images a photo file holds (an animation, pages of a TIFF); an MPO counts as one, as the analysis counts it."""
    from PIL import Image

    try:
        with Image.open(path) as picture:
            return 1 if picture.format == "MPO" else int(getattr(picture, "n_frames", 1) or 1)
    except Exception:
        return 1  # reading it for real comes next and fails with the actual error


def too_large_error(name: str, error: "Exception | str") -> Optional["anonymizer.UnsupportedMedia"]:
    """The refusal of a photo with more pixels than Pillow's safety limit allows (about 179 MP: a 200 MP phone photo, a panorama): it is a
    valid file that is too big, not a damaged one. `error` is the exception, or the "Type: text" of one that media_info already put in
    the detail of its own. None for any other error."""
    from PIL import Image

    text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    if not (isinstance(error, Image.DecompressionBombError) or text.startswith("DecompressionBombError")):
        return None
    found = re.search(r"\((\d+) pixels\)", text)
    pixels = int(found.group(1)) if found else 0
    limit = round(2 * (Image.MAX_IMAGE_PIXELS or 0) / 1e6)
    return anonymizer.UnsupportedMedia(
        "error.media.too_large",
        {"name": name, "megapixels": round(pixels / 1e6) if pixels else "?", "limit": limit},
        text,
    )


def readable_error(path: str, media: str) -> Optional["anonymizer.UnsupportedMedia"]:
    """None when the analysis will be able to open `path` (a photo or a video), else the typed error to show for it."""
    name = os.path.basename(path)
    try:
        if os.path.getsize(path) == 0:
            return anonymizer.UnsupportedMedia("error.media.empty", {"name": name})
        if media == "photo":
            count = image_count(path)
            if count > 1:
                return anonymizer.UnsupportedMedia("error.photo.several_images", {"name": name, "count": count})
        # The public frame reader is the analysis's own: a whole photo is decoded (a truncated one fails here, as it would there)
        # and a video gives its first frame.
        anonymizer.frame_jpeg(path, 0, 64)
        return None
    except MemoryError:
        return None  # too big to open on this machine is the analysis's message (error.photo.not_enough_memory), not a damaged file
    except (
        Exception
    ) as e:  # PyAV, Pillow and the file system raise many kinds; none of them is the person's fault to read
        return too_large_error(name, e) or unreadable(path, media, f"{type(e).__name__}: {e}")
