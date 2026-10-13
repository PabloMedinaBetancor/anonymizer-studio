"""Types shared by the engine and the per-class modules (faces, plates).

Boxes are (x0, y0, x1, y1) floats in pixels of the image that was passed in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

Box = Tuple[float, float, float, float]
Rect = Tuple[int, int, int, int]

FACE = "face"
PLATE = "plate"
VEHICLE = "vehicle"  # context for plates only, never blurred
MANUAL = "manual"  # an area the user added in Review; handled by the engine, no class module

TEXT = "text"  # a text region found on a vehicle; becomes a PLATE item, never listed under this name

# Where a detection came from. A detection found by the second look or by deep mode also carries the origin of the item
# in the report (`found_by`: "light" | "second" | "deep" | "manual"; see FOUND_BY).
SWEEP = "sweep"
RETRY = "retry"
SECOND = "second"  # a heavy model looked at a suspect zone
DEEP = "deep"  # a heavy model swept the whole picture in tiles
BAND = "band"  # a plate-position band on a small vehicle where no text was found (opt-in PT4 lever, TextPathThresholds.band)
FOUND_BY = ("light", SECOND, DEEP, MANUAL)
DECIDED_BY = ("engine", "person")
DETECTION_MODES = ("fast", "careful", "deep")  # Options.detection; careful is the default
UNSURE_MODES = (
    "placed",
    "blur",
    "keep",
)  # Options.unsure; placed is the default since 2.2, blur was the default of 2.1
FIT_MODES = (
    "tight",
    "loose",
)  # Options.fit; tight is the default since 2.2 (small margins, soft edge), loose is the wide fit of 2.1

# Where a final video box came from; keys of ClassModule.video_margins. The further a box is from a real
# detection, the larger the margin it gets.
BOX_SWEEP = "sweep"
BOX_RETRY = "retry"
BOX_GAP = "gap"  # interpolated between two views with the camera motion
BOX_EXTEND = "extend"  # before the first view or after the last one
BOX_SECOND = "second"  # seen by a heavy model at a key frame: between BOX_RETRY and BOX_SWEEP in margin

# Why a suspect zone is visited by the second look, in priority order: the lower the number, the sooner.
ZONE_DOUBTFUL = 1  # a CHECK item
ZONE_DROPPED_STRONG = 2  # a dropped candidate with sweep score >= 0.65
ZONE_PERSON_HEAD = 3  # the head zone of a person with no face item
ZONE_DROPPED_WEAK = 4  # the other dropped candidates (score 0.5-0.65)
ZONE_VEHICLE = 5  # a vehicle: goes to the text path
ZONE_WHY = {
    ZONE_DOUBTFUL: "doubtful",
    ZONE_DROPPED_STRONG: "dropped_strong",
    ZONE_PERSON_HEAD: "person_head",
    ZONE_DROPPED_WEAK: "dropped_weak",
    ZONE_VEHICLE: "vehicle",
}

# Key frames of the video second look, measured on the five development clips (bench/results/video-interpolation-laptop-b.json, replayed
# per clip with its own fps): a heavy look every 8 frames (0.27-0.33 s) leaves p95 0.0 % of a face uncovered between two looks, pooled
# (the hardest clip alone 1.9 %; 12 frames fail: 7.5 % on that clip). The heavy-call budget is the second laptop's 78 ms per zone look.
KEY_PERIOD_FRAMES = 8
HEAVY_PER_SECOND = 6.0
# BOX_SECOND margin of a face seen by the heavy model: p90 of the box error over 50 truth cases, fraction of width, height per side.
FACE_BOX_SECOND_MARGIN = (0.125, 0.152)

# Vehicles given to the text path, per detection mode (PT2, PT6); the rest are counted as not visited. PT4 verdict (B10,
# DECISIONS-V2-B): 6 loses plates against YOLOv9-t on v2 dev (64 against 82, p = .002), 12 does not differ (76) and all vehicles gives 81.
MAX_VEHICLES = {"fast": 12, "careful": 40, "deep": 40}


@dataclass
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    kind: str
    source: str = SWEEP
    on_vehicle: Optional[bool] = None  # plates; None means vehicles were not looked for

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def box(self) -> Box:
        return (self.x0, self.y0, self.x1, self.y1)


class Verdict(str, Enum):
    BLUR = "blur"  # blurred without asking
    CHECK = "check"  # "Not sure": always listed in Review. Blurred by default when Options.unsure is "blur" (2.1), or when the
    # class module says it is `placed` (params["placed"] == 1, Options.unsure "placed", the default since 2.2); left visible with "keep"
    DROP = "drop"  # a detector false positive: neither blurred nor listed


class RetryPurpose(str, Enum):
    RECOVER = "recover"  # the sweep missed a track that was expected here
    VERIFY = "verify"  # a second opinion on a fresh sweep detection
    SECOND_LOOK = "second_look"  # a heavy model on a suspect zone or a video key frame


@dataclass
class Judgement:
    verdict: Verdict
    reason: str  # i18n key, e.g. "plate.reason.off_vehicle"; shown in Review and in the privacy report
    params: Dict[str, float] = field(
        default_factory=dict
    )  # "placed": 1.0 / 0.0 is read by the engine when Options.unsure is "placed"


@dataclass
class PhotoContext:
    """What a class module may use to decide whether a photo detection sits where such a thing can be (`ClassModule.photo_placement`):
    the size of the upright picture, the persons found in it (empty when no person finder ran) and the picture itself (a BGR array; None when
    the caller has none: a module that reads the picture then decides on position alone)."""

    width: int
    height: int
    persons: Sequence["Detection"] = ()
    image: Any = None


@dataclass
class TrackStats:
    """The facts about a video track that a class rule may use. The engine fills it from its track at the
    moment it asks the question."""

    kind: str
    n_views: int  # observations from sweeps and retries
    n_sweep_views: int  # observations from sweeps only
    max_score: float
    best_verify: float  # best VERIFY score so far; 0.0 if never verified or nothing found
    n_verify: int
    verify_side: float  # longest side of the view that was verified last
    on_vehicle: bool  # plates: any view over a detected vehicle
    best_second: float = (
        0.0  # best score of a heavy-model view (source SECOND) so far; 0.0 if none; used as best_verify is
    )
    n_second: int = 0  # views from the second look


@dataclass
class SweepPlan:
    """How to sweep one image."""

    general_side: int  # whole image downscaled to this longest side
    tile_side: int  # full-resolution square tiles; 0 disables tiles
    tiles: Optional[Sequence[int]] = None  # indices into geometry.tiles(); None = all, [] = none
    min_score: Optional[float] = None  # None = the class default
    full_image: bool = True  # plates: also run on the whole image this time
    plate_tiles: bool = True  # plates: also run on the tiles
    vehicle_tiles: bool = False  # plates: also look for vehicles in 1024 px tiles (large photos)
    max_vehicles: int = 6  # plates: search inside the N largest vehicles
    deep: bool = False  # sweep the whole picture with the class's deep-tier model; False = the light plan
    deep_tile: int = 0  # deep: side in pixels of the heavy model's tiles; 0 = the model's own choice


@dataclass
class SuspectZone:
    """A place the second look has to visit: a box in picture pixels, what class of detection it is for,
    why it was chosen (ZONE_* number), and the item it belongs to when it comes from one (index in the first-pass list)."""

    box: Box
    kind: str
    why: int
    priority: int
    item: Optional[int] = None


@runtime_checkable
class HeavyDetector(Protocol):
    """A heavier detector of one class, used on zones or on tiles. Loaded lazily and hash-checked at load (CL3); raises
    ModelLoadError when the file is missing or its SHA-256 differs from `models/SHA256SUMS`."""

    name: str
    version: str
    sha256: str

    def detect(self, bgr, region: Box, scale: float, min_score: float) -> List[Detection]:
        """Detections inside `region` of `bgr`, with the crop enlarged by `scale`; boxes in picture coordinates."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class TextDetector(Protocol):
    """The text detector of the plate path (PP-OCRv6 det through ONNX Runtime). Same load rules as HeavyDetector."""

    name: str
    version: str
    sha256: str

    def detect(self, bgr, min_score: float) -> List[Detection]:
        """Text regions of `bgr` (a vehicle crop already enlarged), kind TEXT, boxes in the pixels of `bgr`."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class ClassModule(Protocol):
    """One detectable class. Built as Module(threads=0, models=None); raises ModelLoadError if a model
    cannot be loaded."""

    kind: str
    blur_shape: str  # "ellipse" or "rect"
    video_margins: Dict[str, Tuple[float, float]]  # by box origin (BOX_*): fraction of width, height per side
    photo_margin: Tuple[float, float]
    timings: Dict[str, float]  # seconds per step, for the benchmark and the report
    calls: Dict[str, int]
    # Model runs that raised and were taken as "nothing here". It only grows; when it grows while a frame or a
    # photo is analysed, the engine marks that frame or photo "not checked" instead of exporting it as clean.
    model_failures: int

    def sweep(self, bgr, plan: SweepPlan) -> List[Detection]:
        """Every detection of this class in the image, after non-maximum suppression."""
        ...

    def retry(self, bgr, expected: Box, purpose: RetryPurpose) -> Optional[Detection]:
        """Look again around `expected` at another scale. Returns the best match (IoU >= 0.15) with
        source=RETRY, or None."""
        ...

    def photo_judgement(self, det: Detection, verify: Optional[Detection]) -> Judgement:
        """Verdict for a photo detection, given its VERIFY retry."""
        ...

    def video_judgement(self, track: TrackStats) -> Judgement:
        """Verdict for a finished video track."""
        ...

    # Optional (not part of the protocol check): `photo_placement(self, box, judgement, ctx: PhotoContext) -> Judgement`. With Options.unsure
    # "placed" the engine calls it for every photo item of the class: it returns the judgement it wants kept, a BLUR that sits where nothing
    # of its class can be turned into a CHECK, and `params["placed"]` says whether a CHECK is blurred by default (1.0) or left visible (0.0).

    def wants_verify(self, track: TrackStats, det: Detection) -> bool:
        """Whether this new sweep view is worth a VERIFY retry."""
        ...

    def wants_recover(self, track: TrackStats) -> bool:
        """Whether a missed track is worth a RECOVER retry."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class SecondLookModule(ClassModule, Protocol):
    """A class module that also has the second look and deep mode. Kept apart from ClassModule so that a module without a
    heavy model (the v1 modules, test fakes) is still a ClassModule; the engine asks `isinstance(m, SecondLookModule)`."""

    heavy_zone: Optional[HeavyDetector]  # the model for zones (careful); None until it is loaded
    heavy_deep: Optional[HeavyDetector]  # the model for the whole picture (deep); may be the same object as heavy_zone

    def second_look(self, bgr, zone: SuspectZone) -> List[Detection]:
        """The heavy look at one zone (enlarged with context); detections with source SECOND, in picture coordinates."""
        ...

    def deep_sweep(self, bgr, plan: SweepPlan) -> List[Detection]:
        """The heavy model over the whole picture in tiles; detections with source DEEP."""
        ...

    def second_judgement(self, light: Optional[Detection], found: Detection) -> Judgement:
        """The result of the second-look table for one heavy detection and the light one it overlaps (None if new)."""
        ...


class ModelLoadError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def _default_model_folders() -> List[str]:
    """Where the models are looked for when nothing says: `models/` in the working directory (a checkout, or a folder the person
    made), then `models/` of the source tree. After a plain `pip install` the package sits in site-packages and the second one is
    not the checkout's: the first, ANONYMIZER_MODELS and --models are what work there."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.normpath(os.path.join(os.getcwd(), "models")),
        os.path.normpath(os.path.join(here, "..", "..", "models")),
    ]


def models_dir(override: Optional[str] = None) -> str:
    """`override`, else $ANONYMIZER_MODELS, else the first existing of ./models and <source tree>/models."""
    if override:
        return override
    env = os.environ.get("ANONYMIZER_MODELS")
    if env:
        return env
    folders = _default_model_folders()
    return next((f for f in folders if os.path.isdir(f)), folders[-1])


def models_missing(path: str) -> str:
    """What a missing model file says: where it looked and the three ways to say where the models are."""
    looked = [os.path.dirname(path)] + [f for f in _default_model_folders() if f != os.path.dirname(path)]
    return (
        f"model file not found: {path}. Looked in: {', '.join(looked)}. Put the .onnx files of MODELS.md in one of them, "
        "or set ANONYMIZER_MODELS to the folder that holds them, or pass --models <folder> (models_dir= in Python)."
    )


def kinds_of(modules: Sequence[ClassModule]) -> List[str]:
    return [m.kind for m in modules]
