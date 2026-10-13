"""Photos: read upright by their EXIF, sweep whole and in full-size tiles, judge every candidate with a second look,
blur and save in the same format and resolution.

Ported from the earlier application. With nothing to blur the photo is copied byte for byte, except the GPS location,
which is removed without recompressing (only the JPEG's EXIF and XMP segments are rewritten). With something blurred
it is recompressed once: JPEG at quality 95 with the source's chroma subsampling, colour profile and EXIF kept
(without GPS, and with orientation 1 because the pixels are now upright).
"""

from __future__ import annotations

import io
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..contract import (
    DETECTION_MODES,
    MAX_VEHICLES,
    UNSURE_MODES,
    Box,
    ClassModule,
    Detection,
    Judgement,
    PhotoContext,
    RetryPurpose,
    SecondLookModule,
    SweepPlan,
    Verdict,
)
from .blur import DEFAULT_STRENGTH, GAUSSIAN, blur, residual
from .containers import (
    NO_LENGTH,
    container_leaks,
    exif_blob_leaks,
    exif_segments,
    icc_leaks,
    jpeg_icc_profile,
    jpeg_kept_payload,
    jpeg_units,
)
from .events import ClipError

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
ORIENTATION = 0x0112
XMP_IN_EXIF = 700  # the XMLPacket tag: an XMP packet inside the EXIF (an earlier audit, A8)

# The whole image at 1600 px on the long side and every 640 px tile at full size; vehicles in tiles too.
PHOTO_PLAN = SweepPlan(general_side=1600, tile_side=640, tiles=None, vehicle_tiles=True)
THUMBNAIL_SIDE = 160


def photo_plan(detection: str = "careful") -> SweepPlan:
    """The sweep plan of a photo: PHOTO_PLAN, with the number of vehicles the plate path looks into set by the detection mode
    (contract.MAX_VEHICLES: 12 in fast, 40 (all in practice) in careful and deep)."""
    return replace(PHOTO_PLAN, max_vehicles=MAX_VEHICLES[detection])


# What calibrate_light assumes of a photo besides the sweep over a picture of noise (which finds nothing): the vehicles the text path looks into
# (a real-estate photo shows few; B's parity runs on plate photos visited 1.2 on average). A placeholder, as the other estimate constants.
EXPECTED_VEHICLES = 1
LIGHT_WARM_SIDE = 320  # the picture that opens the sessions before the timing
LIGHT_MAX_SIDE = (
    8000  # a noise picture is never made larger than this: the time of a larger photo is scaled by its area
)


def calibrate_light(modules: Sequence[ClassModule], width: int, height: int, detection: str = "careful") -> float:
    """Milliseconds the light first pass takes on this machine for a photo of `width` x `height`: every module sweeps a picture of noise of that
    size once (after one sweep of a small picture that opens the sessions), plus, for a plate module with a text detector, `EXPECTED_VEHICLES`
    runs of that detector on a vehicle crop. Noise finds nothing, so this is the cost of looking, not of what is found. The sweeps are not
    analyses: nothing is kept but the time."""
    rng = np.random.default_rng(0)
    plan = photo_plan(detection)
    scale = min(1.0, LIGHT_MAX_SIDE / max(width, height, 1))
    w, h = max(8, int(width * scale)), max(8, int(height * scale))
    warm = rng.integers(0, 255, (LIGHT_WARM_SIDE * 3 // 4, LIGHT_WARM_SIDE, 3), dtype=np.uint8)
    noise = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for m in modules:
        m.sweep(warm, plan)
    started = time.perf_counter()
    for m in modules:
        m.sweep(noise, plan)
    ms = 1000 * (time.perf_counter() - started) / (scale * scale)
    for m in modules:
        text = getattr(m, "text", None)
        if text is not None and hasattr(text, "detect"):
            crop = rng.integers(
                0, 255, (640, 960, 3), dtype=np.uint8
            )  # the text path enlarges a vehicle to T_V = 960 px
            text.detect(warm, 0.5)
            started = time.perf_counter()
            text.detect(crop, 0.5)
            ms += EXPECTED_VEHICLES * 1000 * (time.perf_counter() - started)
    return ms


@dataclass
class Zone:
    box: Box  # as detected, without margin, in pixels of the upright photo
    kind: str
    score: float
    blur: bool  # the decision: True for BLUR, and for CHECK unless unsure="keep" (privacy first, PF1); the person may flip it
    judgement: Judgement
    residual: Optional[float] = None
    found_by: str = "light"  # contract.FOUND_BY: which stage found it (the second look and deep mode set their own)
    second_score: Optional[float] = None  # the heavy model's score when it looked at this zone


@dataclass
class PhotoSpec:
    """How a kind is blurred in photos: margin (fraction of width and height per side) and shape."""

    margin: Tuple[float, float]
    shape: str


@dataclass
class PhotoResult:
    zones: List[Zone] = field(default_factory=list)
    copied: bool = False  # nothing to blur: an exact copy (perhaps without GPS)
    gps_removed: bool = False
    warnings: List[str] = field(default_factory=list)  # i18n keys
    thumbnails: List[np.ndarray] = field(default_factory=list)  # BGR, one per zone, for Review


def specs_of(modules: Sequence[ClassModule]) -> Dict[str, PhotoSpec]:
    return {m.kind: PhotoSpec(tuple(m.photo_margin), m.blur_shape) for m in modules}


def blurred_by_default(verdict: Verdict, unsure: str = "blur", params: Optional[Mapping[str, float]] = None) -> bool:
    """The automatic decision for an item: BLUR is blurred. CHECK ("Not sure") is blurred when `unsure` is "blur" (2.1's privacy first,
    PF1/PF2), never when it is "keep", and when it is "placed" (the default since 2.2) only if the class module judged it `placed`
    (`params["placed"]` is 1.0: it is plate- or face-sized and sits where such a thing can be). DROP is not listed at all."""
    if verdict is Verdict.BLUR:
        return True
    if verdict is not Verdict.CHECK:
        return False
    return unsure == "blur" or (unsure == "placed" and bool(params and params.get("placed", 0.0) >= 1.0))


@dataclass
class SecondTally:
    """What the stages after the first pass did to one photo (A10, A11): `stage` is the last stage that ran ("first" when
    only the light models did), the rest are the counts of the second look or of the deep sweep."""

    stage: str = "first"
    zones_visited: int = 0
    zones_skipped: int = 0
    confirmed: int = 0
    added: int = 0
    failures: int = 0


Persons = Callable[[np.ndarray], Sequence[Detection]]
StageProgress = Callable[[str, int, int], None]  # stage ("second" | "deep"), done, total


def heavy_modules(modules: Sequence[ClassModule]) -> List[SecondLookModule]:
    """The modules that have a heavy model loaded: the ones the second look and deep mode can use."""
    return [m for m in modules if isinstance(m, SecondLookModule) and getattr(m, "heavy_zone", None) is not None]


def decide_photo(
    bgr: np.ndarray,
    modules: Sequence[ClassModule],
    unsure: str = "blur",
    detection: str = "careful",
    persons: Optional[Persons] = None,
    max_zones: Optional[int] = None,
    heavy_only: bool = False,
    progress: Optional[StageProgress] = None,
    check: Optional[Callable[[], None]] = None,
    tally: Optional[SecondTally] = None,
) -> List[Zone]:
    """Sweep with every module (in the given order: faces, then plates) and give each candidate a VERIFY retry;
    the class rule decides with both. The zone covers the detection and its retry together. In "careful" the heavy models of the
    modules that have one then look at the suspect zones; in "deep" they sweep the whole picture and the result is
    united with the first pass (5; `heavy_only` leaves out the light result of the classes that have a heavy model). A stage never
    lowers what the first pass decided. `persons(bgr)` finds the persons whose heads are zones; `check()` may raise to stop; `progress(stage,
    done, total)` follows the heavy stage; `tally` receives what the stage did. If a model run fails meanwhile, the photo was not
    checked and ClipError("error.analysis.incomplete") is raised. With `unsure="placed"` (the default of Options since 2.2) each item then goes
    through its class module's `photo_placement`, if it has one: a BLUR where nothing of the class can be becomes a "Not sure" item, and a
    "Not sure" item is blurred by default only if the module says it sits where such a thing can be."""
    if unsure not in UNSURE_MODES:
        raise ValueError(f"unsure must be one of {UNSURE_MODES}, not {unsure!r}")
    if detection not in DETECTION_MODES:
        raise ValueError(f"detection must be one of {DETECTION_MODES}, not {detection!r}")
    failures = sum(m.model_failures for m in modules)
    items = _first_pass(bgr, modules, photo_plan(detection))
    tally = tally if tally is not None else SecondTally()
    found_persons = list(persons(bgr)) if persons is not None else []
    if detection != "fast" and heavy_modules(modules):
        if progress is not None:
            progress("first", 1, 1)  # the light pass is done; what follows is the heavy stage
        items = _heavy_stage(
            bgr, items, modules, detection, found_persons, max_zones, heavy_only, progress, check, tally
        )
    if sum(m.model_failures for m in modules) != failures:
        raise ClipError("error.analysis.incomplete", {"photos": 1})
    by_kind = {m.kind: m for m in modules}
    context = PhotoContext(bgr.shape[1], bgr.shape[0], found_persons, bgr)
    zones = []
    for it in items:
        if it.verdict is not Verdict.DROP:
            judgement, verdict = it.judgement, it.verdict
            place = getattr(by_kind.get(it.kind), "photo_placement", None)
            if unsure == "placed" and place is not None:
                facts = dict(judgement.params)
                facts["item_score"] = it.score
                facts["second_score"] = max(facts.get("second_score", 0.0), it.second_score or 0.0)
                judgement = place(it.box, Judgement(judgement.verdict, judgement.reason, facts), context)
                verdict = judgement.verdict
            blur = blurred_by_default(verdict, unsure, judgement.params)
            zones.append(Zone(it.box, it.kind, it.score, blur, judgement, None, it.found_by, it.second_score))
    return zones


def _first_pass(bgr: np.ndarray, modules: Sequence[ClassModule], plan: SweepPlan) -> list:
    """The light result as `LookItem`s, the candidates the class rules dropped included: the second look may bring one back."""
    from .zones import LookItem

    by_kind = {m.kind: m for m in modules}
    items = []
    for d in (d for m in modules for d in m.sweep(bgr, plan)):
        module = by_kind[d.kind]
        verify = module.retry(bgr, d.box(), RetryPurpose.VERIFY)
        box = d.box()
        if verify is not None:
            box = (min(box[0], verify.x0), min(box[1], verify.y0), max(box[2], verify.x1), max(box[3], verify.y1))
        judgement = module.photo_judgement(d, verify)
        items.append(LookItem(box, d.kind, d.score, judgement.verdict, judgement, d))
    return items


def _heavy_stage(
    bgr, items, modules, detection, found_persons, max_zones, heavy_only, progress, check, tally: SecondTally
):
    from . import deep, second_look

    if detection == "deep":
        done = deep.deep_stage(
            bgr,
            items,
            modules,
            heavy_only=heavy_only,
            progress=(lambda n, total: progress("deep", n, total)) if progress else None,
            check=check,
        )
        tally.stage, tally.zones_visited, tally.added, tally.confirmed = "deep", done.calls, done.added, done.confirmed
        return done.items
    vehicles = [v for m in modules for v in getattr(m, "last_vehicles", [])]
    result = second_look.second_stage(
        bgr,
        items,
        modules,
        found_persons,
        vehicles,
        max_zones,
        (lambda n, total: progress("second", n, total)) if progress else None,
        check,
    )
    tally.stage = "second"
    tally.zones_visited, tally.zones_skipped = result.zones_visited, result.zones_skipped
    tally.confirmed, tally.added = result.confirmed, result.added
    return result.items


@dataclass
class PhotoAnalysis:
    width: int
    height: int
    zones: List[Zone]
    thumbnails: List[np.ndarray]
    second: SecondTally = field(default_factory=SecondTally)


def check_photo_memory(path: str) -> None:
    """A 64 MP panorama needs about 700 MB more (measured in the earlier application): its size is read without decoding it and, if it does not fit in
    the free memory, that photo fails with what it needs. A picture over Pillow's safety limit of pixels is refused as unsupported."""
    from PIL import Image

    from .video import available_memory_mb

    try:
        with Image.open(path) as im:
            width, height = im.size
    except Image.DecompressionBombError as e:
        raise ClipError("error.media.unsupported", {"name": os.path.basename(path)}, f"too many pixels: {e}") from e
    except (OSError, ValueError, SyntaxError):
        return  # not an image Pillow knows: reading it fails next with the real error
    needed = int(width * height * 3 * 3.8 / 1e6) + 150
    free = available_memory_mb()
    if 0 <= free < needed:
        raise ClipError(
            "error.photo.not_enough_memory",
            {"needed_mb": needed, "free_mb": free, "megapixels": round(width * height / 1e6)},
        )


def analyze_photo(
    path: str,
    modules: Sequence[ClassModule],
    unsure: str = "blur",
    detection: str = "careful",
    persons: Optional[Persons] = None,
    max_zones: Optional[int] = None,
    heavy_only: bool = False,
    progress: Optional[StageProgress] = None,
    check: Optional[Callable[[], None]] = None,
) -> PhotoAnalysis:
    """Detection only: the zones of the upright photo and their Review thumbnails. Nothing is written. The optional arguments are those
    of `decide_photo`; `PhotoAnalysis.second` says what the heavy stage did."""
    import cv2

    check_photo_memory(path)
    count = image_count(path)
    if count > 1:
        raise ClipError("error.photo.several_images", {"count": count})
    im = read(path)[0]
    arr, _ = _to_rgb(im)
    del im
    cv2.cvtColor(arr, cv2.COLOR_RGB2BGR, dst=arr)
    tally = SecondTally()
    zones = decide_photo(arr, modules, unsure, detection, persons, max_zones, heavy_only, progress, check, tally)
    height, width = arr.shape[:2]
    return PhotoAnalysis(width, height, zones, [thumbnail(arr, z.box) for z in zones], tally)


def with_margin(box: Box, margin: Tuple[float, float], width: int, height: int) -> Box:
    mx, my = margin
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (
        max(0.0, x0 - w * mx),
        max(0.0, y0 - h * my),
        min(float(width), x1 + w * mx),
        min(float(height), y1 + h * my),
    )


def thumbnail(bgr: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """A small crop around the box with the box drawn in red, for Review."""
    import cv2

    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = box
    m = max(x1 - x0, y1 - y0) * 0.6 + 6
    a, b = int(max(0, x0 - m)), int(max(0, y0 - m))
    c, d = int(min(width, x1 + m)), int(min(height, y1 + m))
    crop = bgr[b:d, a:c].copy()
    if crop.size == 0:
        return np.zeros((8, 8, 3), np.uint8)
    line = max(1, int(max(crop.shape[:2]) / 120))
    cv2.rectangle(crop, (int(x0 - a), int(y0 - b)), (int(x1 - a), int(y1 - b)), (0, 0, 255), line)
    s = THUMBNAIL_SIDE / max(crop.shape[:2])
    if s < 1:
        size = (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return crop


# What an output photo keeps of its metadata is a short list, not "everything but the location" (review 1, H1): a MakerNote can hold
# a full-size preview of the picture as it was before the blur, and the serial number, the owner, the comments, an XMP packet, IPTC
# and Photoshop blocks identify the person who took it. Kept: the resolution, the orientation, the EXIF version, the colour space, the
# pixel dimensions and, when the user chose to keep the location, the GPS block. The ICC profile is kept as well (colour).
KEEP_IFD0 = (0x011A, 0x011B, 0x0128, 0x0213)  # XResolution, YResolution, ResolutionUnit, YCbCrPositioning
KEEP_EXIF = (0x9000, 0x9101, 0xA000, 0xA001)  # ExifVersion, ComponentsConfiguration, FlashpixVersion, ColorSpace
PIXEL_DIMENSIONS = (0xA002, 0xA003)
# What Pillow lists in `im.info` that is not metadata about the photo (anything else there is, and is not carried over).
TECHNICAL_INFO = frozenset(
    "dpi gamma icc_profile transparency srgb chromaticity aspect interlace interlace_type progressive progression compression"
    " resolution background loop duration lossless exif jfif jfif_version jfif_unit jfif_density adobe adobe_transform"
    " dpi_unit timestamp anonymizer_tiff_tags".split()
)
TIFF_METADATA_TAGS = (700, 33723, 34377)  # XMP, IPTC, Photoshop image resources
# In a TIFF the "EXIF" Pillow lists is the file's own directory: sizes, strips, compression, colour. A white list (review 2, N2):
# a tag that is not here (DocumentName, PageName, the Windows XP* tags, PrintIM, SubIFDs, ...) is carried metadata.
TIFF_STRUCTURE = frozenset(
    (
        254,
        256,
        257,
        258,
        259,
        262,
        266,
        273,
        274,
        277,
        278,
        279,
        282,
        283,
        284,
        296,
        317,
        320,
        322,
        323,
        324,
        325,
        338,
        339,
        34665,
        34675,
    )
)


def _number(value, low: float, high: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return low <= number <= high and number == number


def _version(value) -> bool:
    return isinstance(value, (bytes, bytearray)) and len(value) == 4 and bytes(value).isalnum()


# What each tag of the short list may hold (review 2, N3): the value is copied, so a tag with another type or size, which could carry a
# payload, is dropped instead of refused as a leak.
EXIF_VALUE_OK = {
    0x011A: lambda v: _number(v, 0, 1e9),
    0x011B: lambda v: _number(v, 0, 1e9),
    0x0128: lambda v: isinstance(v, int) and 1 <= v <= 3,
    0x0213: lambda v: isinstance(v, int) and 1 <= v <= 2,
    0x9000: _version,
    0x9101: lambda v: isinstance(v, (bytes, bytearray)) and len(v) == 4 and all(b <= 6 for b in v),
    0xA000: _version,
    0xA001: lambda v: isinstance(v, int) and v in (1, 2, 0xFFFF),
    0xA002: lambda v: isinstance(v, int) and 0 <= v < 2**31,
    0xA003: lambda v: isinstance(v, int) and 0 <= v < 2**31,
    ORIENTATION: lambda v: isinstance(v, int) and 1 <= v <= 8,
}


def _extra_exif_tags(exif, fmt: Optional[str], keep_gps: bool) -> List[int]:
    """The EXIF tags of a photo that are not on the short list (see KEEP_*), whichever IFD they are in."""
    allowed = set(KEEP_IFD0) | {ORIENTATION, EXIF_IFD} | ({GPS_IFD} if keep_gps else set())
    keys = set(exif.keys())
    extra = (keys - TIFF_STRUCTURE - ({GPS_IFD} if keep_gps else set())) if fmt == "TIFF" else (keys - allowed)
    inner = exif.get_ifd(EXIF_IFD)
    extra |= set(inner.keys()) - set(KEEP_EXIF) - set(PIXEL_DIMENSIONS)
    if fmt != "TIFF":
        extra |= {tag for tag in KEEP_IFD0 + (ORIENTATION,) if tag in exif and not _value_fits(tag, exif[tag])}
    extra |= {tag for tag in KEEP_EXIF + PIXEL_DIMENSIONS if tag in inner and not _value_fits(tag, inner[tag])}
    return sorted(extra)


def _value_fits(tag: int, value) -> bool:
    check = EXIF_VALUE_OK.get(tag)
    return check is None or bool(check(value))


def clean_exif(exif, size=None, orientation: Optional[int] = 1, keep_gps: bool = False):
    """A new EXIF with only what the output keeps (see KEEP_*). `size` writes the pixel dimensions of the saved picture;
    without it the source's are copied. `orientation` None copies the source's (a photo that is not turned)."""
    from PIL import Image

    fresh = Image.Exif()
    for tag in KEEP_IFD0:
        if tag in exif and _value_fits(tag, exif[tag]):
            fresh[tag] = exif[tag]
    if orientation is not None:
        fresh[ORIENTATION] = orientation
    elif ORIENTATION in exif and _value_fits(ORIENTATION, exif[ORIENTATION]):
        fresh[ORIENTATION] = exif[ORIENTATION]
    source = exif.get_ifd(EXIF_IFD)
    inner = fresh.get_ifd(EXIF_IFD)
    for tag in KEEP_EXIF:
        if tag in source and _value_fits(tag, source[tag]):
            inner[tag] = source[tag]
    if size is not None:
        inner[PIXEL_DIMENSIONS[0]], inner[PIXEL_DIMENSIONS[1]] = int(size[0]), int(size[1])
    else:
        for tag in PIXEL_DIMENSIONS:
            if tag in source and _value_fits(tag, source[tag]):
                inner[tag] = source[tag]
    if inner:
        fresh[EXIF_IFD] = inner
    if keep_gps and GPS_IFD in exif:
        gps = fresh.get_ifd(GPS_IFD)
        gps.update(exif.get_ifd(GPS_IFD))
        fresh[GPS_IFD] = gps
    return fresh


def _carries_metadata(im, exif, keep_gps: bool, fmt: Optional[str], data: bytes = b"") -> bool:
    """Does a photo (not a JPEG, whose segments are rewritten instead) hold metadata beyond the short list? `im` is what read()
    returned: its info says which TIFF blocks the source had."""
    if container_leaks(data) or icc_leaks(im.info.get("icc_profile")):
        return True  # what Pillow does not show: private chunks, EXIF thumbnails, trailing data, a profile that carries more than colour
    if any(key not in TECHNICAL_INFO for key in im.info):
        return True  # XMP, text chunks, comments, Photoshop blocks...
    if im.info.get("anonymizer_tiff_tags"):
        return True  # IPTC, Photoshop blocks or XMP of a TIFF
    return exif is not None and bool(len(exif)) and bool(_extra_exif_tags(exif, fmt, keep_gps))


# GPS without recompressing. Any property whose name starts with "gps", with any prefix (an "ns1:" bound to the EXIF
# namespace, the drones' "drone-dji:GpsLatitude"), as an attribute (double or single quotes), an element or empty
# (an earlier audit, A7: only a few prefixes were removed and the location stayed).
_XMP_GPS_ATTRIBUTE = re.compile(rb"\s[\w.-]+:gps\w*\s*=\s*(?:\"[^\"]*\"|'[^']*')", re.I)
_XMP_GPS_ELEMENT = re.compile(rb"<([\w.-]+:gps\w*)\b[^>]*?(?:/>|>.*?</\1\s*>)", re.I | re.S)


def _gps_in_xmp(data: bytes) -> bool:
    return bool(_XMP_GPS_ATTRIBUTE.search(data) or _XMP_GPS_ELEMENT.search(data))


def _xmp_without_gps(xmp: bytes) -> Optional[bytes]:
    """The XMP without location; None if something like it is still there (then the whole XMP goes: a photo does
    not need it to be seen)."""
    cleaned = _XMP_GPS_ELEMENT.sub(b"", _XMP_GPS_ATTRIBUTE.sub(b"", xmp))
    return None if _gps_in_xmp(cleaned) else cleaned


def _strip_gps_exif(exif) -> bool:
    """Remove the location from a Pillow EXIF: its IFD and an embedded XMP (tag 700) that carries GPS (an earlier audit, A8).
    True if something changed."""
    changed = False
    if GPS_IFD in exif:
        del exif[GPS_IFD]
        changed = True
    xmp = exif.get(XMP_IN_EXIF)
    if xmp is not None:
        raw = xmp if isinstance(xmp, (bytes, bytearray)) else str(xmp).encode("utf-8", "replace")
        if _gps_in_xmp(bytes(raw)):
            del exif[XMP_IN_EXIF]
            changed = True
    return changed


def _exif_without_gps(segment: bytes) -> Optional[bytes]:
    """An APP1 "Exif" segment without the location (None if it had none)."""
    from PIL import Image

    exif = Image.Exif()
    exif.load(segment)
    return exif.tobytes() if _strip_gps_exif(exif) else None


def _exif_segments(data: bytes) -> Iterator[bytes]:
    """Every EXIF block (APP1 "Exif") in the file, also those of the images after the first one (an MPF JPEG: the
    second image of an MPO, the gain map of phone HDR photos)."""
    i = 0
    while True:
        j = data.find(b"\xff\xe1", i)
        if j < 0 or j + 10 > len(data):
            return
        length = int.from_bytes(data[j + 2 : j + 4], "big")
        if data[j + 4 : j + 10] == b"Exif\x00\x00" and length >= 8 and j + 2 + length <= len(data):
            yield data[j + 4 : j + 2 + length]
        i = j + 2


def has_gps(data: bytes) -> bool:
    """Is there a location left anywhere: any EXIF (every image of a JPEG/MPO; Pillow's for other formats) or the
    XMP. An EXIF that cannot be read counts as possibly having one."""
    from PIL import Image

    if _gps_in_xmp(data):
        return True
    if data[:2] == b"\xff\xd8":
        for segment in _exif_segments(data):
            try:
                exif = Image.Exif()
                exif.load(segment)
            except Exception:
                return True  # unreadable: assume it holds a location, so the photo is re-saved or refused
            if GPS_IFD in exif:
                return True
        return False
    try:
        with Image.open(io.BytesIO(data)) as im:
            return GPS_IFD in im.getexif()
    except Exception:
        return True  # same conservative answer


def jpeg_without_gps(data: bytes) -> Tuple[bytes, bool]:
    """Remove GPS from a JPEG's EXIF and XMP by rewriting only those segments; the image data is not touched. When
    the file does not follow the format, the original comes back unchanged with False (the caller then checks
    has_gps() and re-saves)."""
    if data[:2] != b"\xff\xd8":
        return data, False
    out = [b"\xff\xd8"]
    i, n, changed = 2, len(data), False
    seen_mpf = False  # the MPF index of the images that follow
    try:
        while i < n:
            if data[i] != 0xFF:
                return data, False
            marker = data[i + 1]
            if marker == 0xFF:  # fill byte
                i += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                out.append(data[i : i + 2])
                i += 2
                continue
            if marker == 0xD9:  # end of image
                out.append(data[i:])
                break
            length = int.from_bytes(data[i + 2 : i + 4], "big")
            if length < 2 or i + 2 + length > n:  # a cut segment: the file is broken, leave it alone
                return data, False
            segment = data[i + 4 : i + 2 + length]
            if marker == 0xDA:  # compressed data starts: the rest goes as it is
                out.append(data[i:])
                break
            if marker == 0xE2 and segment.startswith(b"MPF\x00"):
                seen_mpf = True
            if marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
                cleaned = _exif_without_gps(segment)
                if cleaned is not None and len(cleaned) + 2 <= 0xFFFF:
                    if seen_mpf and len(cleaned) != len(segment):
                        return data, False  # it would move the images behind the MPF index: re-saved instead
                    segment, changed = cleaned, True
            elif marker == 0xE1 and segment.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                cleaned = _xmp_without_gps(segment)
                if cleaned != segment:
                    if seen_mpf:
                        return data, False
                    changed = True
                    if cleaned is None:  # could not be cleaned: the whole XMP goes
                        i += 2 + length
                        continue
                    segment = cleaned
            out.append(bytes([0xFF, marker]) + (len(segment) + 2).to_bytes(2, "big") + segment)
            i += 2 + length
    except Exception:
        return data, False  # anything odd in the segments: unchanged, and has_gps() decides what happens next
    return (b"".join(out), True) if changed else (data, False)


def jpeg_with_clean_metadata(data: bytes, keep_gps: bool = False) -> Optional[bytes]:
    """The JPEG of the first image with only the metadata the output keeps, by rewriting its segments (the image data is not
    touched): the JFIF header without its thumbnail, the ICC profile, Adobe's colour flag and an EXIF reduced by clean_exif.
    Everything else goes: XMP, IPTC and Photoshop blocks, comments, the other application segments. All of the file is walked,
    the segments between the scans of a progressive JPEG too (review 2, N2). None when the file does not follow the format (the
    caller then saves the photo again)."""
    from PIL import Image

    out = []
    profile_ok = not icc_leaks(jpeg_icc_profile(data))  # a profile that carries more than colour goes whole
    try:
        for marker, start, end in jpeg_units(data):
            if marker is None or marker in NO_LENGTH:
                if marker != -1:
                    out.append(data[start:end])
                continue
            payload = data[start + 4 : end]
            if marker == 0xE2 and not profile_ok:
                continue
            if marker == 0xE1:
                if not payload.startswith(b"Exif\x00\x00"):
                    continue
                try:
                    exif = Image.Exif()
                    exif.load(payload)
                    rebuilt = clean_exif(exif, None, None, keep_gps).tobytes()
                except Exception:
                    continue  # an EXIF that cannot be read is not carried over
                if len(rebuilt) + 2 <= 0xFFFF:
                    out.append(bytes([0xFF, marker]) + (len(rebuilt) + 2).to_bytes(2, "big") + rebuilt)
                continue
            kept = jpeg_kept_payload(marker, payload)
            if kept is not None:
                out.append(bytes([0xFF, marker]) + (len(kept) + 2).to_bytes(2, "big") + kept)
    except ValueError:
        return None
    return b"".join(out)


def _first_image_end(data: bytes) -> Optional[int]:
    """Where the first image of a JPEG ends (after its EOI), walking its segments and its entropy-coded data (there
    0xFF is followed by 0x00 or an RST; the first 0xFF followed by anything else is the next marker; a progressive
    JPEG has several SOS). None if the file does not follow the format."""
    if data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0xD9:
            return i + 2
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return None
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if length < 2 or i + 2 + length > n:
            return None
        i += 2 + length
        if m == 0xDA:
            while True:
                j = data.find(b"\xff", i)
                if j < 0 or j + 1 >= n:
                    return None
                b = data[j + 1]
                if b == 0x00 or 0xD0 <= b <= 0xD7:
                    i = j + 2
                    continue
                if b == 0xFF:
                    i = j + 1
                    continue
                i = j
                break
    return None


def jpeg_first_image(data: bytes) -> Optional[bytes]:
    """The JPEG with only its first image: nothing after its EOI and no MPF index (APP2) pointing at the images that
    were there. The same bytes if there was nothing else; None if the file cannot be understood."""
    end = _first_image_end(data)
    if end is None:
        return None
    out = [b"\xff\xd8"]
    i = 2
    while i < end:
        if data[i] != 0xFF:
            return None
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0xDA, 0xD9):
            out.append(data[i:end])
            break
        if m == 0x01 or 0xD0 <= m <= 0xD7:
            out.append(data[i : i + 2])
            i += 2
            continue
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        segment = data[i : i + 2 + length]
        if not (m == 0xE2 and segment[4:8] == b"MPF\x00"):
            out.append(segment)
        i += 2 + length
    trimmed = b"".join(out)
    return data if trimmed == data else trimmed


def image_count(path: str) -> int:
    """How many images the file holds (animation, pages). An MPO counts as one: its first image is anonymised and
    the rest removed (an earlier audit, A2)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            if im.format == "MPO":
                return 1
            return int(getattr(im, "n_frames", 1) or 1)
    except Exception:
        return 1  # reading it for real comes next and fails with the actual error


def read(path: str):
    """(upright Pillow image, original EXIF, format, JPEG subsampling or -1, ICC profile). Opened from disk: holding
    the file's bytes too was one more copy of the photo."""
    from PIL import Image, ImageOps

    with open(path, "rb") as f:
        im = Image.open(f)
        im.load()
    fmt = im.format
    subsampling = -1
    if fmt in ("JPEG", "MPO"):  # Pillow calls a JPEG with several images (MPF) "MPO"
        from PIL import JpegImagePlugin

        try:
            subsampling = JpegImagePlugin.get_sampling(im)
        except (AttributeError, KeyError, TypeError, ValueError):
            subsampling = -1  # unknown: saved at 4:4:4, which only costs size
    exif = im.getexif()
    icc = im.info.get("icc_profile")
    if icc_leaks(icc):
        icc = None  # a profile that does not measure up is not carried over (R3-4)
    tiff_blocks = [tag for tag in TIFF_METADATA_TAGS if tag in getattr(im, "tag_v2", ())]
    im = ImageOps.exif_transpose(im)
    if tiff_blocks:
        im.info["anonymizer_tiff_tags"] = tiff_blocks  # the turned copy has no tag_v2 of its own
    return im, exif, fmt, subsampling, icc


# Modes over 8 bits (16-bit grey PNG/TIFF, integer, float). convert("RGB") clipped them at 255 (the photo came out
# white and the detectors saw nothing): detection runs on an 8-bit copy and the blur on the real data, which is saved
# at its own depth (an earlier audit, A1).
DEEP_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def _deep(im) -> np.ndarray:
    """A deep-mode photo as uint16 HxW (0-65535); floating point is stretched from its minimum to its maximum."""
    a = np.asarray(im)
    if im.mode == "F":
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (np.nan_to_num(a) - lo) * (65535.0 / (hi - lo) if hi > lo else 0.0)
    return np.ascontiguousarray(np.clip(a, 0, 65535).astype(np.uint16))


def _icc_matches(icc: bytes, mode: str) -> bool:
    """Is the colour profile for the colour space being saved? (A9: an RGB file with the source's CMYK profile)"""
    if not icc or len(icc) < 40 or bytes(icc[36:40]) != b"acsp":
        return True  # not a profile we understand: left as it was
    space = bytes(icc[16:20])
    if mode in ("RGB", "RGBA", "RGBX", "P", "PA"):
        return space == b"RGB "
    if mode in ("L", "LA", "1") or mode in DEEP_MODES:
        return space == b"GRAY"
    if mode == "CMYK":
        return space == b"CMYK"
    return True


def _to_rgb(im) -> Tuple[np.ndarray, Optional[object]]:
    """The photo as ONE RGB uint8 array, and its alpha channel apart if it has one. A 64 MP panorama is 192 MB per
    copy: the earlier application once made up to five (+900 MB peak); now there are the photo and this."""
    if im.mode in DEEP_MODES:
        a8 = (_deep(im) >> 8).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(a8[:, :, None], 3, axis=2)), None
    alpha = None
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        alpha = rgba.getchannel("A")
        arr = np.array(rgba.convert("RGB"))
        del rgba
    else:
        # np.array is the safe copy: pasting through a frombuffer view made Pillow copy and leave the array empty.
        arr = np.array(im if im.mode == "RGB" else im.convert("RGB"))
    return arr, alpha


def blur_zones(
    arr: np.ndarray,
    zones: Sequence[Zone],
    specs: Mapping[str, PhotoSpec],
    style: str = GAUSSIAN,
    strength: float = DEFAULT_STRENGTH,
    profile: Optional[dict] = None,
) -> None:
    """Blur the zones to blur in place (HxWxC or HxW) and record each one's residual."""
    height, width = arr.shape[:2]
    for zone in zones:
        if not zone.blur:
            continue
        spec = specs[zone.kind]
        blur(arr, with_margin(zone.box, spec.margin, width, height), spec.shape, style, strength, profile)
        zone.residual = round(residual(arr, zone.box), 2)


def _valid_dpi(dpi) -> bool:
    """A resolution of two numbers (Pillow gives a TIFF's as IFDRational, a PNG's as float) that a picture can have."""
    try:
        return isinstance(dpi, (tuple, list)) and len(dpi) == 2 and all(1 <= float(v) <= 100_000 for v in dpi)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def save(
    image,
    target: str,
    fmt: str,
    exif,
    subsampling: int,
    icc,
    info: dict,
    strip_gps: bool = True,
    same_pixels: bool = False,
) -> bool:
    """Save in the source's format. Returns False when its EXIF could not be carried over (saved without it). The resolution is
    carried over; with `same_pixels` (the picture is the source's, not one blurred and rebuilt) so is the transparency of a paletted,
    grey or RGB PNG, which Pillow keeps in `info` and the clean copy below would lose."""
    kw = {}
    if _valid_dpi(info.get("dpi")):
        kw["dpi"] = tuple(float(v) for v in info["dpi"])
    if same_pixels and "transparency" in info and image.mode in ("P", "L", "RGB", "1"):
        kw["transparency"] = info["transparency"]
    if image.info or hasattr(image, "tag_v2"):
        # A picture opened from a file carries its comments, XMP, IPTC and Photoshop blocks with it and Pillow writes them back
        # on its own: save a plain copy, so only what is passed below is written.
        image = image.copy()
        image.info.clear()
    final_mode = image.mode if fmt not in ("JPEG", "MPO") or image.mode in ("RGB", "L", "CMYK") else "RGB"
    if icc and _icc_matches(icc, final_mode):
        kw["icc_profile"] = icc
    exif_kept = True
    if exif is not None and len(exif):
        try:
            kw["exif"] = clean_exif(exif, image.size, 1, not strip_gps).tobytes()
        except Exception:
            # An EXIF Pillow cannot write back: the photo is saved without it (no location either) and the caller
            # reports it.
            exif_kept = False
    if fmt in ("JPEG", "MPO"):
        # Pillow's "MPO" (a JPEG with a second image: iPhone HDR photos, 3D cameras) is saved as a single-image JPEG
        # with the usual quality, subsampling, profile and EXIF.
        if image.mode not in ("RGB", "L", "CMYK"):
            image = image.convert("RGB")
        image.save(
            target, "JPEG", quality=95, subsampling=subsampling if subsampling in (0, 1, 2) else 0, optimize=True, **kw
        )
    elif fmt == "PNG":
        image.save(
            target,
            "PNG",
            optimize=False,
            **{k: v for k, v in kw.items() if k in ("icc_profile", "exif", "dpi", "transparency")},
        )
    elif fmt == "WEBP":
        lossless = bool(info.get("lossless"))
        extra = (
            {"exact": True} if lossless and "A" in image.mode else {}
        )  # the colour under a transparent pixel stays as it was
        image.save(
            target,
            "WEBP",
            quality=95,
            lossless=lossless,
            **{k: v for k, v in kw.items() if k != "transparency"},
            **extra,
        )
    elif fmt == "AVIF":
        image.save(target, "AVIF", quality=90, **{k: v for k, v in kw.items() if k in ("icc_profile", "exif")})
    elif fmt == "TIFF":
        image.save(
            target, "TIFF", compression="tiff_lzw", **{k: v for k, v in kw.items() if k in ("icc_profile", "dpi")}
        )
    else:
        image.save(target, fmt or "PNG")
    return exif_kept


def metadata_leaks(path: str, keep_gps: bool = False) -> List[str]:
    """What an output photo still carries beyond the short list it keeps (KEEP_*): comments, XMP, IPTC and Photoshop blocks,
    text chunks, a MakerNote or any other EXIF tag, an image embedded in a metadata segment. Empty when clean. The container is read
    from the bytes (containers.py) and not only through Pillow: a segment between scans, a private chunk or a thumbnail is seen."""
    from PIL import Image

    with open(path, "rb") as f:
        data = f.read()
    leaks = container_leaks(data)
    if data[:2] == b"\xff\xd8":  # every EXIF segment, not only the one Pillow reads
        for segment in exif_segments(data):
            leaks += exif_blob_leaks(segment)
            try:
                exif = Image.Exif()
                exif.load(segment)
                leaks += [f"EXIF tag {tag:#06x}" for tag in _extra_exif_tags(exif, "JPEG", keep_gps)]
            except Exception:
                leaks.append("EXIF that cannot be read")
    try:
        with Image.open(io.BytesIO(data)) as im:
            leaks += [f"metadata {key!r}" for key in im.info if key not in TECHNICAL_INFO]
            leaks += icc_leaks(im.info.get("icc_profile"))
            tags = getattr(im, "tag_v2", None)
            leaks += [f"TIFF tag {tag}" for tag in TIFF_METADATA_TAGS if tags is not None and tag in tags]
            leaks += [f"EXIF tag {tag:#06x}" for tag in _extra_exif_tags(im.getexif(), im.format, keep_gps)]
    except Exception as e:
        leaks.append(f"unreadable: {type(e).__name__}")
    return list(dict.fromkeys(leaks))


def _check_clean(target: str, keep_gps: bool) -> None:
    """What was promised is checked: a saved photo that still carries camera or descriptive data is not delivered."""
    leaks = metadata_leaks(target, keep_gps)
    if leaks:
        raise ClipError("error.photo.metadata_not_removed", detail="; ".join(leaks[:8]))


def _webp_is_lossless(path: str) -> bool:
    with open(path, "rb") as f:
        return b"VP8L" in f.read(
            64
        )  # the first image chunk: VP8L lossless, VP8 lossy, or VP8X (extended) then one of them


def _check_no_gps(target: str) -> None:
    """What was promised is checked: a saved photo that still has a location is an error and is not delivered."""
    with open(target, "rb") as f:
        if has_gps(f.read()):
            raise ClipError("error.photo.gps_not_removed", detail=target)


def process(
    path: str,
    target: str,
    modules: Sequence[ClassModule] = (),
    specs: Optional[Mapping[str, PhotoSpec]] = None,
    strip_gps: bool = True,
    zones: Optional[List[Zone]] = None,
    visible: Set[int] = frozenset(),
    thumbnails: bool = False,
    style: str = GAUSSIAN,
    strength: float = DEFAULT_STRENGTH,
    unsure: str = "blur",
    detection: str = "careful",
    profile: Optional[dict] = None,
) -> PhotoResult:
    """Anonymise one photo into `target` (a partial file the caller renames). With `zones` already decided (a re-render after
    Review) no detector runs and `modules` may be empty; `visible` holds indices of zones left unblurred. With `thumbnails`, each
    zone's thumbnail comes back in the result. What was written is checked: no location when strip_gps, and no camera or
    descriptive data beyond the short list the output keeps (metadata_leaks). `profile` is the blur profile (blur.PROFILE, or
    blur.SUBTLE_PROFILE of fit="tight")."""
    result = _process(
        path, target, modules, specs, strip_gps, zones, visible, thumbnails, style, strength, unsure, detection, profile
    )
    _check_clean(target, not strip_gps)
    return result


def _process(
    path: str,
    target: str,
    modules: Sequence[ClassModule] = (),
    specs: Optional[Mapping[str, PhotoSpec]] = None,
    strip_gps: bool = True,
    zones: Optional[List[Zone]] = None,
    visible: Set[int] = frozenset(),
    thumbnails: bool = False,
    style: str = GAUSSIAN,
    strength: float = DEFAULT_STRENGTH,
    unsure: str = "blur",
    detection: str = "careful",
    profile: Optional[dict] = None,
) -> PhotoResult:
    """Anonymise one photo into `target` (a partial file the caller renames). With `zones` already decided (a
    re-render after Review) no detector runs and `modules` may be empty; `visible` holds indices of zones left
    unblurred. With `thumbnails`, each zone's thumbnail comes back in the result."""
    import cv2
    from PIL import Image

    check_photo_memory(path)
    count = image_count(path)
    if count > 1:
        # An animation (WebP, PNG) or a multi-page TIFF: only the first image was looked at and the rest was
        # delivered unchecked or lost (an earlier audit, A2).
        raise ClipError("error.photo.several_images", {"count": count})
    specs = specs if specs is not None else specs_of(modules)
    try:
        im, exif, fmt, subsampling, icc = read(path)
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as e:  # cut short or damaged since it was analysed
        raise ClipError("error.media.damaged", {"name": os.path.basename(path)}, f"{type(e).__name__}: {e}") from e
    info = dict(im.info)
    if fmt == "WEBP":
        info["lossless"] = _webp_is_lossless(
            path
        )  # Pillow does not say, and a lossless photo saved lossy changes its pixels
    grey = im.mode in ("L", "1")
    result = PhotoResult()
    arr = alpha = None
    if zones is None:
        arr, alpha = _to_rgb(im)
        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR, dst=arr)  # in place: detectors want BGR
        zones = decide_photo(arr, modules, unsure, detection)
        if thumbnails:
            result.thumbnails = [thumbnail(arr, z.box) for z in zones]
        cv2.cvtColor(arr, cv2.COLOR_BGR2RGB, dst=arr)
    for i in visible:
        if 0 <= i < len(zones):
            zones[i].blur = False
    result.zones = zones
    if not any(z.blur for z in zones):
        del arr
        return _copy(path, target, im, exif, fmt, subsampling, icc, info, strip_gps, result)
    if im.mode in DEEP_MODES or im.mode == "CMYK":
        # A1 and A9: the blur goes on the real data (16-bit grey; the four CMYK channels with their profile), not on
        # the 8-bit RGB copy the detectors saw.
        del arr
        mode = im.mode
        data = _deep(im) if mode in DEEP_MODES else np.array(im)
        del im
        blur_zones(data, zones, specs, style, strength, profile)
        height, width = data.shape[:2]
        if data.ndim == 2:
            image = Image.fromarray(data)
        else:
            image = Image.frombuffer("CMYK", (width, height), data, "raw", "CMYK", 0, 1)
    else:
        if arr is None:
            arr, alpha = _to_rgb(im)
        del im
        blur_zones(arr, zones, specs, style, strength, profile)
        height, width = arr.shape[:2]
        image = Image.frombuffer("RGB", (width, height), arr, "raw", "RGB", 0, 1)  # no copy of the array
        if grey:
            image = image.convert("L")
        if alpha is not None:
            image = image.convert("RGBA")
            image.putalpha(alpha)
    result.gps_removed = bool(strip_gps and exif is not None and GPS_IFD in exif)
    if not save(image, target, fmt, exif, subsampling, icc, info, strip_gps):
        result.warnings.append("photo.warning.metadata_dropped")
    if strip_gps:
        _check_no_gps(target)
    return result


def _copy(path, target, im, exif, fmt, subsampling, icc, info, strip_gps, result: PhotoResult) -> PhotoResult:
    """Nothing to blur: the pixels as they are. A JPEG (or MPO) has only its metadata segments rewritten, by a white list of segments,
    down to the short list the output keeps and without the location when strip_gps; the image data is a byte copy. A format without
    loss (PNG, TIFF, lossless WebP) is ALWAYS written again from the decoded picture and the short list of metadata: the pixels come
    out the same and nothing the analysers do not know can be carried (review 3: going case by case does not close the class). A lossy
    WebP, an AVIF and the rest are saved again when they hold anything beyond the list and are a byte copy when not."""
    result.copied = True
    with open(path, "rb") as f:
        data = f.read()
    lossless = fmt in ("PNG", "TIFF") or (fmt == "WEBP" and bool(info.get("lossless")))
    if fmt in ("JPEG", "MPO"):
        # What follows the first image (the video of an Android "motion photo", Samsung's trailer, the second image of
        # an MPO) has not been looked at, so it is not delivered (an earlier audit, A2).
        first = jpeg_first_image(data)
        if first is None:
            result.copied = False
            result.warnings.append("photo.warning.resaved_trailing_data")
            if not save(im, target, fmt, exif, subsampling, icc, info, strip_gps):
                result.warnings.append("photo.warning.metadata_dropped")
            if strip_gps:
                _check_no_gps(target)
            return result
        if len(first) != len(data):
            result.warnings.append("photo.warning.trailing_data_removed")
            data = first
        cleaned = jpeg_with_clean_metadata(data, not strip_gps)
        if cleaned is not None and not (strip_gps and has_gps(cleaned)):
            with open(target, "wb") as f:
                f.write(cleaned)
            result.gps_removed = bool(strip_gps and has_gps(data))
            return result
    elif (
        not lossless and not _carries_metadata(im, exif, not strip_gps, fmt, data) and not (strip_gps and has_gps(data))
    ):
        with open(target, "wb") as f:
            f.write(data)
        return result
    result.copied = False
    result.gps_removed = bool(strip_gps and has_gps(data))
    if not lossless:
        result.warnings.append("photo.warning.resaved_for_metadata")
    try:
        exif_kept = save(im, target, fmt, exif, subsampling, icc, info, strip_gps, same_pixels=True)
    except (OSError, KeyError, ValueError) as e:  # Pillow reads a mode it cannot write in this format
        if lossless and not _carries_metadata(im, exif, not strip_gps, fmt, data) and not (strip_gps and has_gps(data)):
            with open(target, "wb") as f:  # nothing to remove: the file as it is
                f.write(data)
            result.copied, result.gps_removed = True, False
            return result
        raise ClipError(
            "error.photo.metadata_not_removed", detail=f"cannot be written again: {type(e).__name__}: {e}"
        ) from e
    if not exif_kept:
        result.warnings.append("photo.warning.metadata_dropped")
    if strip_gps:
        _check_no_gps(target)
    return result
