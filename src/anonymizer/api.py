"""The Python API. What `anonymizer.__all__` lists is defined here and is public; the rest of the package is internal."""

from __future__ import annotations

import copy
import errno
import functools
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import contract
from .classes import PLATE_PATHS, VEHICLE_MODELS
from .contract import DETECTION_MODES, FACE, FIT_MODES, MANUAL, PLATE, UNSURE_MODES, ClassModule, Judgement, Verdict
from .engine import events
from .engine import preview as previews
from .engine.blur import DEFAULT_STRENGTH, MAX_STRENGTH, MIN_STRENGTH, PROFILE, RECT, STYLES, SUBTLE_PROFILE
from .engine.blur import blur as blur_region
from .engine.compare import compare_video as write_comparison
from .engine.files import PHOTO_EXTENSIONS, UNSUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS, source_fingerprint
from .engine.keyframes import KeyFrameParams
from .engine.photos import (
    PhotoSpec,
    Zone,
    analyze_photo,
    blurred_by_default,
    heavy_modules,
    process,
    specs_of,
    with_margin,
)
from .engine.tracks import HOLD, LINEAR, MOTION, FinalBoxes
from .engine.video import (
    MANUAL_MARGIN,
    VideoAnalysis,
    analyze_video,
    boxes_to_blur,
    manual_boxes,
    partial_path,
    render_video,
)

# Report files: 2 adds found_by, decision and decided_by to every row, and the options and summary.
FORMAT_VERSION = 2
# analysis.json: 3 adds the options fingerprint, found_by and decided_by; 2 was v1's.
ANALYSIS_VERSION = 3
# What estimate() assumes: the zones a photo gives the second look on average over the 128 development photos (19 in crowds, 3 without
# people). The light pass of a photo is not assumed: `calibrate_light` times it on this machine (RF-DETR and the text path made it several
# times longer than v1's 378-510 ms).
EXPECTED_ZONES = 10
SECOND_LOOK_COUNTS = {
    "zones_visited": 0,
    "zones_skipped": 0,
    "confirmed": 0,
    "added": 0,
    "vehicles_too_small": 0,
    "bands_added": 0,
    "band_area_share": 0.0,
}
GAP_FILL = {"camera": MOTION, "straight": LINEAR, "hold": HOLD}
THOROUGHNESS = ("standard", "maximum")
Box = Tuple[float, float, float, float]


class _Missing(dict):
    """The parameters of a sentence: one that was not given stays as its `{name}` instead of failing."""

    def __missing__(self, key):
        return "{" + key + "}"


@functools.lru_cache(maxsize=1)
def _english_sentences() -> Dict[str, str]:
    """The Studio's English text of every error code (`ui/i18n/en.json`), read once; empty where the file cannot be read."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "i18n", "en.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k.startswith("error.") and isinstance(v, str)}


# Where the Studio's text speaks of the Studio ("not imported"), the API says it its own way.
_API_SENTENCES = {"error.media.unsupported": "{name} is not a photo or a video that can be processed."}


def _sentence(code: str, params: dict, detail: str) -> str:
    """What `str(error)` says: the English sentence of the code with its parameters filled in, and the technical `detail` after it
    when there is one. A code with no sentence says itself, as it always did."""
    if code == "error.media.unsupported" and detail in ("not a video", "not a photo"):
        # The one-call functions refuse a file of the other kind: it is a photo or a video, and the Studio's text ("not a photo or a
        # video this tool can open, so it was not imported") would say the opposite.
        return f"{params.get('name', 'The file')} is {detail}."
    text = _API_SENTENCES.get(code) or _english_sentences().get(code)
    if text is None:
        return f"{code}: {detail}" if detail else code
    try:
        text = text.format_map(_Missing(params))
    except (ValueError, IndexError, AttributeError):  # a brace or a field the text does not expect: say it as it is
        pass
    return f"{text} ({detail})" if detail else text


def _destination_folder_exists(target: str) -> None:
    """Refuse a target whose folder is not there, naming the folder: a write would stop on the hidden working file instead
    (`.name.anonymizer-partial`), which the person never made, and after a long analysis."""
    folder = os.path.dirname(os.path.abspath(target))
    if os.path.isdir(folder):
        return
    if os.path.exists(folder):
        raise NotADirectoryError(errno.ENOTDIR, "The destination folder is not a folder", folder)
    raise FileNotFoundError(errno.ENOENT, "The destination folder does not exist", folder)


class AnonymizerError(Exception):
    """The base of the errors the Python API raises for a file, and the class itself for errors that have no subclass.

    Every error carries a `code`, the key of the sentence the Studio shows for it, and the `params` that fill that sentence.
    `str(error)` is the English sentence with the parameters filled in, then `detail` in brackets. Six errors have a
    subclass, so that you can catch them by class; the others are raised as `AnonymizerError` itself and told apart by
    `code`: `error.source.missing`, `error.analysis.version`, `error.output.check_failed`, `error.photo.metadata_not_removed`,
    `error.photo.gps_not_removed`, `error.photo.not_enough_memory`, `error.video.no_frames`, `error.deep.too_large`,
    `error.deep.too_much_work` and `error.demo.deep`. Some mistakes raise a builtin instead, such as `ValueError` for a bad
    option.

    Attributes:
        code: The error key, such as `error.output.exists`. The constructor's `code=None` means `default_code`.
        params: A dict with the values that fill the placeholders of the sentence, such as `{"name": "x.heic"}` (a copy).
        detail: Technical text for a log, not written for a person and not translated, or an empty string.
        default_code: Class attribute. The code a subclass has when none is given.

    Example:
        ```python
        import anonymizer

        try:
            anonymizer.media_info("notes.txt")
        except anonymizer.AnonymizerError as error:
            print(error.code)    # error.media.unsupported
            print(error.params)  # {'name': 'notes.txt'}
            print(error)         # notes.txt is not a photo or a video that can be processed. (.txt: not a photo or a video)
        ```
    """

    default_code = "error.anonymizer"

    def __init__(self, code: Optional[str] = None, params: Optional[dict] = None, detail: str = ""):
        self.code = code or self.default_code
        self.params = dict(params or {})
        self.detail = detail
        super().__init__(_sentence(self.code, self.params, detail))


class UnsupportedMedia(AnonymizerError):
    """The file is not a photo or a video that can be processed.

    Raised by `media_info`, `frame_jpeg`, `thumbnail_jpeg`, `compare_video`, `anonymize_image`, `anonymize_video`,
    `Anonymizer.estimate` and `Anonymizer.analyze` when the extension is not one of `PHOTO_EXTENSIONS` or
    `VIDEO_EXTENSIONS` (HEIC, RAW, GIF, a text file, a folder), or when the file is empty or its header cannot be read;
    the kind is told from the extension only. `anonymize_image` also raises it for a video and `anonymize_video` for a photo. Two more codes use
    this class: `error.media.damaged`, a file that stops decoding partway, while it is analysed or rendered (nothing is
    written under the target); and `error.photo.several_images`, a photo that holds more than one image, such as an
    animation.

    Attributes:
        code: `error.media.unsupported`, `error.media.damaged` or `error.photo.several_images`.
        params: `{"name": ...}`, the file's base name, or `{"count": ...}` for several images.
        detail: What was found, such as `.heic: a format that is not supported`.
    """

    default_code = "error.media.unsupported"


class ModelLoadError(AnonymizerError):
    """A detection model, or the runtime that runs it, could not be loaded.

    Raised by the first call that loads the models: a model file that is missing, not listed in the models folder's
    `SHA256SUMS` or changed is refused before any file is touched. This is not `anonymizer.contract.ModelLoadError`, an
    internal class: catching that one catches nothing the public calls raise.

    Attributes:
        code: `error.model.load`.
        params: `{"kind": ...}`, the model or runtime that failed (for example `heavy`).
        detail: The folders that were searched and the file that failed.
    """

    default_code = "error.model.load"


class OutputExists(AnonymizerError):
    """The target already exists. Nothing is ever overwritten.

    Raised by `Anonymizer.render` and `compare_video`, which check it first, and by `anonymize_image` and
    `anonymize_video`, which find it only after the whole analysis: in a loop, test `os.path.exists(target)` before the
    call. It is also raised when the target appears while the output is written. `Report.to_csv`, `Report.to_json` and
    `Analysis.save` replace an existing file instead.

    Attributes:
        code: `error.output.exists`.
        params: `{"path": ...}`, the target.
        detail: An empty string.
    """

    default_code = "error.output.exists"


class SourceChanged(AnonymizerError):
    """The original is no longer the file that was analysed.

    Raised by `Anonymizer.render`, which compares the original's size, modification time and a hash of its content with
    the analysis at the start and immediately before the output gets its name, and by `Analysis.preview`. Analyse it
    again.

    Attributes:
        code: `error.source.changed`.
        params: `{"path": ...}`, the original, when raised by `Anonymizer.render`.
        detail: An empty string.
    """

    default_code = "error.source.changed"


class Cancelled(AnonymizerError):
    """The job was stopped through its `CancelToken`. Nothing was written.

    Raised by `Anonymizer.analyze`, `Anonymizer.render`, `compare_video` and `anonymize_video` after `CancelToken.cancel`,
    once a `Finished` event with status `"stopped"` has been sent. Nothing is left under the target. A video stops within a
    frame or two; a photo's first pass with the light models cannot be broken off.

    Attributes:
        code: `error.cancelled`.
        params: An empty dict.
        detail: An empty string.
    """

    default_code = "error.cancelled"


class IncompleteAnalysis(AnonymizerError):
    """Part of the file could not be checked, so it is not delivered as clean.

    Raised by `Anonymizer.render` for a video whose `Analysis.unchecked_frames` is not empty, because a detector failed on
    those frames, and by `Anonymizer.analyze` for a photo on which a model run or the person finder failed. The Studio
    says "Part of this file could not be checked, so it is not exported as clean."

    Attributes:
        code: `error.analysis.incomplete`.
        params: `{"frames": ..., "first": ...}` for a video (how many, and the first), `{"photos": 1}` for a photo.
        detail: The failure underneath, when there is one.
    """

    default_code = "error.analysis.incomplete"


_ERRORS = {
    "error.output.exists": OutputExists,
    "error.source.changed": SourceChanged,
    "error.analysis.incomplete": IncompleteAnalysis,
    "error.photo.several_images": UnsupportedMedia,
    "error.media.damaged": UnsupportedMedia,
}


def _ffmpeg_error():
    """FFmpeg's error class, or an empty tuple (catches nothing) where PyAV is not installed."""
    try:
        import av
    except ImportError:
        return ()
    return av.error.FFmpegError


def _still_analysed(a: "Analysis") -> None:
    """The source is still the file the analysis was made from; checked again just before an output gets its final name,
    so a file replaced while it was being written is never delivered as anonymised."""
    try:
        same = source_fingerprint(a.source) == a.fingerprint
    except OSError:
        same = False
    if not same:
        raise SourceChanged(params={"path": a.source})


def _damaged(name: str, e: Exception) -> UnsupportedMedia:
    return UnsupportedMedia("error.media.damaged", {"name": name}, f"{type(e).__name__}: {e}")


def _api_error(e: events.ClipError) -> AnonymizerError:
    return _ERRORS.get(e.code, AnonymizerError)(e.code, e.params, e.detail)


class CancelToken:
    """A request to stop a running job, to pause it, and to continue it. Safe to use from another thread.

    Pass it as `cancel=` and call `cancel()` from any thread: a video stops within a frame or two, the call raises
    `Cancelled`, and nothing is written. A token stays cancelled, so make a new one for each job. A photo's first pass
    cannot be broken off, and `anonymize_image` takes no token.

    Example:
        ```python
        import threading

        import anonymizer

        token = anonymizer.CancelToken()
        threading.Timer(1.0, token.cancel).start()
        with anonymizer.Anonymizer() as anon:
            try:
                anon.analyze("examples/data/street-madrid-5s.mp4", cancel=token)
            except anonymizer.Cancelled:
                print("stopped", token.cancelled)  # stopped True
        ```

    See also:
        `Cancelled`, `Finished`.
    """

    def __init__(self):
        self._flag = events.StopFlag()

    def cancel(self) -> None:
        """Ask the job that holds this token to stop, and release it if it is paused. It does not wait."""
        self._flag.request()

    @property
    def cancelled(self) -> bool:
        """Whether `cancel()` has been called. Once True it stays True."""
        return self._flag.requested

    def pause(self) -> None:
        """Hold the job after the current frame, until `resume()` or `cancel()`. Nothing done so far is lost."""
        self._flag.pause()

    def resume(self) -> None:
        """Let a paused job go on. It does nothing when the job is not paused."""
        self._flag.resume()

    @property
    def paused(self) -> bool:
        """Whether the job is held by `pause()` and not yet released by `resume()`."""
        return self._flag.paused


def _flag(token: Optional[CancelToken]) -> Optional[events.StopFlag]:
    return token._flag if token is not None else None


@dataclass(frozen=True)
class Options:
    """What to look for, how hard to look, and how to hide what is found.

    `Options()` is the default behaviour of the Python API and of the Studio. It is a frozen dataclass: each field is checked
    when it is made, and you change one by making a copy with `dataclasses.replace`.

    **Which fields act when.** `faces`, `plates`, `detection`, `plate_path`, `vehicle_model`, `plate_band` and `fit` are
    read once, when an `Anonymizer` loads its detectors: make a new `Anonymizer` to change them. `thoroughness`, `unsure`,
    `max_zones`, `heavy_only` and `gap_fill` are read by each `analyze`. `blur`, `strength`, `extra_margin`, `remove_gps`,
    `gap_fill` and the soft edge of `fit` are read by each `render`, `preview` and `boxes`, so you can assign a new
    `Options` to `Anonymizer.options` between an analysis and its render.

    "The 2.1 settings" are `Options(unsure="blur", fit="loose", plate_band=True)`, and "the v1 settings"
    `Options(detection="fast", unsure="keep", fit="loose")`.

    Attributes:
        faces: Look for faces. A class left out is not blurred, and only the JSON report says it was not searched.
        plates: Look for licence plates. At least one of `faces` and `plates` must be True.
        thoroughness: `"standard"` (every second frame of a video) or `"maximum"` (every frame, at a larger size): slower,
            not repeatable bit for bit, no better in the measurements, and not in the Studio.
        gap_fill: `"camera"`, `"straight"` or `"hold"`, for videos: how a box is placed on the frames between two
            detections. `"camera"` follows the camera's estimated motion and stops at a change of shot, `"straight"` draws a
            line between the two boxes, `"hold"` keeps the last box.
        blur: `"gaussian"` (blocks about the size of a character, then smoothed), `"pixelate"` (large blocks cut from the
            blurred content) or `"solid"` (the average colour). None of them can be undone.
        strength: From 1.0 to 2.0; a larger value blurs more. 1.0 is the floor, the weakest strength at which the tests
            for residual detail pass.
        extra_margin: From 0.0 to 0.5: added to every margin, as a fraction of the box on each side.
        remove_gps: Photos: True drops the GPS block. Other metadata is dropped in every case, and a video never carries
            the container's metadata.
        detection: `"fast"` (the light models only), `"careful"` (adds the second look with the heavy model) or `"deep"`
            (the heavy model over the whole picture in tiles: experimental, refused over its work limit and in demo mode).
        heavy_only: Deep only: leave the light models' items out of the result.
        unsure: What happens to a doubtful item before anyone decides: `"placed"` blurs it only when it is placed like a
            plate or a face, `"blur"` blurs them all (2.1), `"keep"` leaves them all visible (v1).
        max_zones: The most zones the second look visits in one photo: None for the built-in cap of 30, 0 for no heavy
            call. Zones beyond it keep the first pass's result and are counted in `Analysis.second["zones_skipped"]`.
        plate_path: `"text"` (a vehicle detector, a text detector inside each vehicle, and a plate reader) or `"yolo"`
            (the v1 plate model, kept for comparison; its weights are not shipped, so it raises `ModelLoadError`).
        vehicle_model: The vehicle detector of the text path: `"rfdetr"` (RF-DETR nano) or `"yolox"` (YOLOX-nano).
        plate_band: Photos: True also blurs where a plate usually sits on a vehicle narrower than 400 pixels with no text
            found there. Off, because in 2.1 it blurred bare walls and car sides. Needs `plate_path="text"`.
        fit: `"tight"` (small margins and a soft edge close to the box) or `"loose"` (the 2.1 margins and a wide soft edge).

    Raises:
        ValueError: A field has a value it does not accept, or two fields do not go together. The message names the field.

    Example:
        ```python
        from dataclasses import replace

        import anonymizer

        v1 = anonymizer.Options(detection="fast", unsure="keep", fit="loose")
        print(replace(v1, blur="pixelate", strength=1.5).kinds())  # ['face', 'plate']
        ```

    See also:
        `Anonymizer`, `Item`.
    """

    faces: bool = True
    plates: bool = True
    thoroughness: str = "standard"
    gap_fill: str = "camera"
    blur: str = "gaussian"
    strength: float = DEFAULT_STRENGTH
    extra_margin: float = 0.0
    remove_gps: bool = True
    detection: str = "careful"
    heavy_only: bool = False
    unsure: str = "placed"
    max_zones: Optional[int] = None
    plate_path: str = "text"
    vehicle_model: str = "rfdetr"
    plate_band: Optional[bool] = None
    fit: str = "tight"

    def __post_init__(self):
        if not (self.faces or self.plates):
            raise ValueError("choose faces, plates or both")
        if self.thoroughness not in THOROUGHNESS:
            raise ValueError(f"thoroughness must be one of {THOROUGHNESS}, not {self.thoroughness!r}")
        if self.gap_fill not in GAP_FILL:
            raise ValueError(f"gap_fill must be one of {tuple(GAP_FILL)}, not {self.gap_fill!r}")
        if self.blur not in STYLES:
            raise ValueError(f"blur must be one of {STYLES}, not {self.blur!r}")
        if not MIN_STRENGTH <= self.strength <= MAX_STRENGTH:
            raise ValueError(f"strength must be between {MIN_STRENGTH} and {MAX_STRENGTH}, not {self.strength}")
        if not 0.0 <= self.extra_margin <= 0.5:
            raise ValueError(f"extra_margin must be between 0 and 0.5, not {self.extra_margin}")
        if self.detection not in DETECTION_MODES:
            raise ValueError(f"detection must be one of {DETECTION_MODES}, not {self.detection!r}")
        if self.unsure not in UNSURE_MODES:
            raise ValueError(f"unsure must be one of {UNSURE_MODES}, not {self.unsure!r}")
        if self.heavy_only and self.detection != "deep":
            raise ValueError('heavy_only needs detection="deep"')
        if self.max_zones is not None and (
            isinstance(self.max_zones, bool) or not isinstance(self.max_zones, int) or self.max_zones < 0
        ):
            raise ValueError(f"max_zones must be None or a whole number of zones, 0 or more, not {self.max_zones!r}")
        if self.plate_path not in PLATE_PATHS:
            raise ValueError(f"plate_path must be one of {PLATE_PATHS}, not {self.plate_path!r}")
        if self.vehicle_model not in VEHICLE_MODELS:
            raise ValueError(f"vehicle_model must be one of {VEHICLE_MODELS}, not {self.vehicle_model!r}")
        if self.plate_band is not None and not isinstance(self.plate_band, bool):
            raise ValueError(f"plate_band must be None, True or False, not {self.plate_band!r}")
        if self.plate_band and self.plate_path != "text":
            raise ValueError('plate_band needs plate_path="text"')
        if self.fit not in FIT_MODES:
            raise ValueError(f"fit must be one of {FIT_MODES}, not {self.fit!r}")

    @property
    def profile(self) -> dict:
        """The numbers of the soft edge that `fit` selects, as a dict. The engine reads it; you rarely need it."""
        return SUBTLE_PROFILE if self.fit == "tight" else PROFILE

    def kinds(self) -> List[str]:
        """The classes to look for, in detection order: `"face"` and `"plate"`, less the one whose option is False."""
        return [k for k, on in ((FACE, self.faces), (PLATE, self.plates)) if on]


@dataclass
class Item:
    """One thing the analysis found in a file, or an area a person added: a face, a plate or an added area.

    `Analysis.items` lists them. Decide through `Analysis.set_blur` and `Analysis.add_area`, which also record that a person
    decided. In a video an item spans frames and its box moves; `Analysis.boxes` gives the boxes frame by frame.

    Attributes:
        item_id: The engine's number of the item, stable for the analysis. It can skip numbers; the Studio and the PDF
            report number the items 1 to n in time order instead.
        kind: `"face"`, `"plate"` or `"manual"` (an added area).
        verdict: `"blur"` when the detectors are sure, `"check"` when they are not (a doubtful item; **Not sure** in the
            Studio).
        reason: The key of the sentence that says why, such as `plate.reason.vehicle_text`. The reference page "Errors and
            codes" lists every key with the Studio's English sentence.
        first_frame: The first frame of the item; 0 for a photo. Divide by `Analysis.fps` for seconds.
        last_frame: The last frame of the item; 0 for a photo.
        max_score: The best certainty any detector gave, rounded to 3 places; 1.0, and not a certainty, for an added area
            and a plate band.
        blur: Whether the item is blurred in the output.
        params: The numbers behind the verdict, such as `score` and `placed`; plates found by text also carry
            `plate_like`, `read_conf` and `read_digits`. The keys depend on the module that judged it: use `params.get`.
        found_by: `"light"` (the first pass), `"second"` (the second look), `"deep"` or `"manual"`.
        second_score: The heavy model's score when the second look checked this item, otherwise None.
        decided_by: `"engine"`, or `"person"` once a person decided it or added it.


    See also:
        `Analysis`, `ReportRow`.
    """

    item_id: int
    kind: str
    verdict: str
    reason: str
    first_frame: int
    last_frame: int
    max_score: float
    blur: bool
    params: dict = field(default_factory=dict)
    found_by: str = "light"
    second_score: Optional[float] = None
    decided_by: str = "engine"


@dataclass
class Started:
    """Sent when the analysis of a file starts.

    For a video it comes before the work. For a photo it comes before the heavy stage, or after the whole analysis when only
    the light models run, so such a photo sends no `Progress` at all.

    Attributes:
        clip_id: The base name of the file, such as `street-madrid.mp4`. Every event has it first.
        media: `"photo"` or `"video"`.
        frames: The number of frames (1 for a photo); for a video the container's count, which can differ by one from
            `Analysis.frames`.
        width: The width in pixels, as displayed.
        height: The height in pixels, as displayed.
        fps: The frame rate. 0.0 for a photo.
        type: The type of its JSON message.
    """

    clip_id: str
    media: str
    frames: int
    width: int
    height: int
    fps: float
    type = "job.started"


@dataclass
class Progress:
    """How far a phase has got. Sent many times; `fraction` never goes back within a phase.

    `anonymize_video` sends a sequence for the phase `"analyze"`, then one for `"render"`; `compare_video` sends
    `"compare"`. A photo's render sends none.

    Attributes:
        clip_id: The base name of the file.
        phase: `"analyze"`, `"render"` or `"compare"`.
        fraction: From 0 to 1, rounded to 4 places.
        frame: The frame the work is at.
        frames: The number of frames.
        live: While a video is analysed, the boxes in view at `frame` (`kind`, `box`, `score`, `state`), for a display.
            Otherwise None.
        found: While a video is analysed, the things seen at least twice so far, by kind. Otherwise None.
        stage: For a photo with a heavy model: `"first"`, `"second"` or `"deep"`. Otherwise None.
        zones: During the `"second"` stage, `done` and `total`. Otherwise None.
        type: The type of its JSON message.
    """

    clip_id: str
    phase: str
    fraction: float
    frame: int
    frames: int
    live: Optional[list] = None
    found: Optional[dict] = None
    stage: Optional[str] = None
    zones: Optional[dict] = None
    type = "clip.progress"


@dataclass
class ItemFound:
    """Sent once for each item, in time order, after the analysis has ended and before `Analyzed`.

    Attributes:
        clip_id: The base name of the file.
        item: The `Item`.
        type: The type of its JSON message.
    """

    clip_id: str
    item: Item
    type = "clip.item"


@dataclass
class Analyzed:
    """Sent once, at the end of `Anonymizer.analyze`, after every `ItemFound`. The analysis sends no `Finished`.

    Attributes:
        clip_id: The base name of the file.
        items: How many items were listed.
        to_check: How many of them are doubtful, decided or not.
        zones_visited: How many zones the second look visited. For a photo it also counts the vehicles the plate path
            looked into; for a video, the heavy looks at key frames.
        zones_skipped: How many zones the budget (`Options.max_zones`) left out.
        second_added: How many items the heavy model brought in that the first pass had not found.
        type: The type of its JSON message.
    """

    clip_id: str
    items: int
    to_check: int
    zones_visited: int = 0
    zones_skipped: int = 0
    second_added: int = 0
    type = "clip.analyzed"


@dataclass
class Estimate:
    """What `Anonymizer.estimate` expects an analysis to cost, with the options in force.

    A guess from timings of this machine. It does not cover the light pass of a video, and under load the real time can be
    several times the estimate.

    Attributes:
        seconds: Seconds of work expected: the light pass of a photo plus the heavy model's calls; for a video only the heavy
            calls (0.0 with `detection="fast"`).
        zones_expected: For a photo, the zones the second look is expected to visit; for a video, the heavy calls it may make
            or the whole-frame sweeps of the deep level.
        tiles: The full-resolution tiles of a deep sweep. 0 at the other levels.
    """

    seconds: float
    zones_expected: int
    tiles: int


@dataclass
class Rendered:
    """Sent once, after the output was written, checked and given its final name.

    Attributes:
        clip_id: The base name of the original.
        output: The path of the output, as given to `render`.
        seconds: How long the render took, rounded to 2 places.
        type: The type of its JSON message.
    """

    clip_id: str
    output: str
    seconds: float
    type = "clip.rendered"


@dataclass
class JobWarning:
    """Something the person should know. The work went on.

    Sent zero or more times. The codes are listed, with their sentences, at the end of the reference page "Errors and
    codes": the second look's budget, the render's `photo.warning.*` and `video.warning.*`, and
    `warning.review.check_left_visible` when doubtful items stay visible. `Result.warnings` keeps the codes of the render.

    Attributes:
        clip_id: The base name of the file.
        code: The key of the warning.
        params: The values for its sentence, such as `count`. Often empty.
        type: The type of its JSON message.
    """

    clip_id: str
    code: str
    params: dict = field(default_factory=dict)
    type = "job.warning"


@dataclass
class Finished:
    """Sent when a call ends: by `Anonymizer.render` with status `"done"`, after `Rendered`; by `Anonymizer.analyze`,
    `Anonymizer.render` and `compare_video` with status `"stopped"`, immediately before `Cancelled` is raised.

    Attributes:
        clip_id: The base name of the file.
        status: `"done"` or `"stopped"`.
        type: The type of its JSON message.
    """

    clip_id: str
    status: str
    type = "job.finished"


Event = Union[Started, Progress, ItemFound, Analyzed, Rendered, JobWarning, Finished]
"""An event sent to the handler you pass as `on_event`: one of `Started`, `Progress`, `ItemFound`, `Analyzed`, `Rendered`,
`JobWarning` and `Finished`, each a dataclass whose first field is `clip_id`.

The handler runs in the thread that runs the call, so keep it quick; an exception in it stops the call. A video's
analysis sends `Started`, `Progress` (many), `ItemFound` (one per item) and `Analyzed`; a render sends `Progress`,
`JobWarning` (zero or more), `Rendered` and `Finished("done")`. A cancelled job sends `Finished("stopped")`, then raises
`Cancelled`.
"""
EventHandler = Callable[[Event], None]


def event_json(event: Event) -> dict:
    """Turn an event into a JSON-ready dict, the message the Studio's page receives.

    The dict has `"type"` (such as `"job.finished"`), `"v"` (the message version, 1) and the event's fields. An `ItemFound`
    carries its item as a plain dict.

    Example:
        ```python
        import json

        import anonymizer

        event = anonymizer.Finished("street.jpg", "done")
        print(json.dumps(anonymizer.event_json(event)))
        # {"type": "job.finished", "v": 1, "clip_id": "street.jpg", "status": "done"}
        ```
    """
    return {"type": event.type, "v": 1, **asdict(event)}


def _csv_line(values) -> str:
    """One CSV line as the `csv` module writes it (RFC 4180, `\\r\\n`), with a field also quoted when it holds a semicolon or a tab: the
    Text Import of LibreOffice splits at those as well as at the comma by default, so a file named `beach;house.jpg` moved the rest of its
    row one column to the right."""
    cells = []
    for value in values:
        text = "" if value is None else str(value)
        if any(c in text for c in ',;\t"\r\n'):
            text = '"' + text.replace('"', '""') + '"'
        cells.append(text)
    return ",".join(cells) + "\r\n"


def _csv_safe(value):
    """A text that a spreadsheet would run as a formula (a file name starting with =, +, - or @) is written as text."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


@dataclass
class ReportRow:
    """One line of the privacy report: one item, with what was found and what was done.

    The CSV report has these fields as its 14 columns, in this order, and the JSON report has them as the keys of each row.

    Attributes:
        clip: The base name of the file.
        item_id: The item's number (see `Item.item_id`).
        kind: `"face"`, `"plate"` or `"manual"`.
        verdict: `"blur"` or `"check"`.
        blurred: Whether the item is blurred in the output.
        reason: The key of the reason (see `Item.reason`).
        first_s: Seconds from the start of a video to the item's first frame, rounded to 3 places. 0.0 for a photo.
        last_s: Seconds from the start to its last frame. 0.0 for a photo.
        frames: How many frames the item spans. 1 for a photo.
        max_score: The best certainty (see `Item.max_score`).
        found_by: `"light"`, `"second"`, `"deep"` or `"manual"`.
        decision: `"blurred"` or `"left visible"`: what the output shows.
        decided_by: `"engine"` or `"person"`.
        plate_like: For a plate found by text, whether the region is shaped like a plate; False means it is listed because
            it may still be one. None otherwise.
    """

    clip: str
    item_id: int
    kind: str
    verdict: str
    blurred: bool
    reason: str
    first_s: float
    last_s: float
    frames: int
    max_score: float
    found_by: str = "light"  # "light" | "second" | "deep" | "manual"
    decision: str = "blurred"  # "blurred" | "left visible"
    decided_by: str = "engine"  # "engine" | "person"
    plate_like: Optional[bool] = (
        None  # plates by text: whether the region is shaped like a plate (False: other writing); else None
    )


def _doubtful_counts(entries) -> dict:
    """What happened to the "Not sure" items, from (verdict, blurred, decided_by) triples: how many there were, how many were
    blurred by default and left that way, how many a person confirmed, how many a person turned off, and how many stay visible
    because the options said unsure="keep" or "placed" and the item is not placed."""
    counts = {
        "doubtful": 0,
        "blurred_by_default": 0,
        "confirmed_by_person": 0,
        "turned_off_by_person": 0,
        "left_visible_by_default": 0,
    }
    for verdict, blurred, decided_by in entries:
        if verdict != Verdict.CHECK.value:
            continue
        counts["doubtful"] += 1
        if decided_by == "person":
            counts["confirmed_by_person" if blurred else "turned_off_by_person"] += 1
        elif blurred:
            counts["blurred_by_default"] += 1
        else:
            counts["left_visible_by_default"] += 1
    return counts


@dataclass
class Report:
    """The privacy report: one row for each item, the items left visible and the added areas included.

    `Result.report` and `Analysis.report` give one. **It does not say what the detectors missed**, and the PDF and CSV
    reports do not say which classes were searched; the JSON report's `options` does.

    Attributes:
        rows: The `ReportRow` list, in time order.
        options: What the analysis was made with: `faces`, `plates`, `thoroughness`, `detection` (asked for),
            `detection_ran` (what ran), `heavy_only`, `unsure`, `max_zones`, `plate_path`, `vehicle_model`, `plate_band`,
            and `models`, the name and SHA-256 of each heavy or text model loaded. Not the blur settings.
        second: The second look's counts (see `Analysis.second`).


    See also:
        `ReportRow`, `Result`.
    """

    rows: List[ReportRow]
    options: dict = field(default_factory=dict)
    second: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Count what happened to the items, and what the second look did.

        Returns:
            A dict: `items` (the rows) and `blurred`; for the doubtful items, `doubtful`, `blurred_by_default` (no person touched
            it), `confirmed_by_person`, `turned_off_by_person` and `left_visible_by_default`; then the keys of `second`.
        """
        counts = _doubtful_counts((r.verdict, r.blurred, r.decided_by) for r in self.rows)
        counts["items"] = len(self.rows)
        counts["blurred"] = sum(1 for r in self.rows if r.blurred)
        return {**counts, **self.second}

    def to_csv(self, path: str) -> None:
        """Write the rows as a CSV file with a header line. An existing file at `path` is replaced.

        The file is UTF-8 with a byte-order mark (read it with `encoding="utf-8-sig"`), and a text that starts like a formula
        gets a `'` in front. The reference page "Files written" has the whole format.

        Args:
            path: Where to write the file. The folder must exist.

        Raises:
            OSError: The file cannot be written.
        """
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write(_csv_line([f.name for f in ReportRow.__dataclass_fields__.values()]))
            for r in self.rows:
                f.write(_csv_line([_csv_safe(v) for v in asdict(r).values()]))

    def to_json(self, path: str) -> None:
        """Write the report as a JSON file. An existing file at `path` is replaced.

        The file is an object with `"version"` (2: the version of the report file), `"options"`, `"summary"` (what `summary`
        returns) and `"items"` (one object per row, with the keys of `ReportRow`).

        Args:
            path: Where to write the file. The folder must exist.

        Raises:
            OSError: The file cannot be written.
        """
        with open(path, "w", encoding="utf-8") as f:
            data = {"version": FORMAT_VERSION, "options": self.options, "summary": self.summary()}
            json.dump({**data, "items": [asdict(r) for r in self.rows]}, f, indent=1)
            f.write("\n")


@dataclass
class Result:
    """What `Anonymizer.render`, `anonymize_image` and `anonymize_video` return.

    Always read `warnings` and `visible_check`. With the default settings, doubtful items that are not placed like a plate
    or a face stay visible, and the call still succeeds: `demo/street-taipei.jpg` comes out with nothing blurred, and only
    these two fields say so.

    Attributes:
        output: The path of the output, as you gave it.
        report: The `Report` of the analysis as it was rendered.
        warnings: The warning codes of the render, then `warning.review.check_left_visible` when doubtful items stay
            visible.
        seconds: How long the render took, rounded to 2 places.
        visible_check: How many doubtful items stay visible in this output.
    """

    output: str
    report: Report
    warnings: List[str]
    seconds: float
    visible_check: int = 0


@dataclass
class BoxInfo:
    """A box on one frame, margins included: where an item is blurred, or would be.

    `Analysis.boxes` returns them. The coordinates are in pixels of the original (a video's can be numpy floats: use
    `float()` before writing JSON).

    Attributes:
        item_id: The item the box belongs to.
        kind: `"face"`, `"plate"` or `"manual"`.
        box: `(x0, y0, x1, y1)` in pixels: a video's rounded to 1 place, a photo's as the engine computed them.
        shape: `"ellipse"` for a face, `"rect"` for a plate and for an added area.
        blur: Whether the item is blurred in the output.

    See also:
        `Analysis.boxes`.
    """

    item_id: int
    kind: str
    box: Box
    shape: str
    blur: bool


@dataclass
class MediaInfo:
    """What a file is, read from its header without analysing it. `media_info` returns it.

    Attributes:
        media: `"photo"` or `"video"`, from the extension.
        width: The width in pixels, as displayed (a photo's orientation is applied).
        height: The height in pixels, as displayed.
        frames: 1 for a photo; for a video the container's count, which can differ by one from `Analysis.frames`.
        fps: The frame rate. 0.0 for a photo.
        duration_s: The length in seconds. 0.0 for a photo.
        codec: A photo's format name in lower case, such as `jpeg`; a video's codec, such as `h264`.
        has_audio: Whether the file has an audio track.
        has_gps: Whether the file carries a location (a photo's GPS block, a video's location key).
        size_bytes: The file's size in bytes.
    """

    media: str
    width: int
    height: int
    frames: int
    fps: float
    duration_s: float
    codec: str
    has_audio: bool
    has_gps: bool
    size_bytes: int


@dataclass
class _Area:
    frame_from: int
    frame_to: int
    box: Box
    follow: str


class Analysis:
    """What `Anonymizer.analyze` found in one file, with the decisions made on it.

    It holds the items, the decisions, the added areas and what `Anonymizer.render` needs, so no detector runs again.
    `Anonymizer.analyze` and `load_analysis` return it. Decide with `set_blur`, `add_area` and `remove_area`, look with
    `preview` and `boxes`, and keep it with `save`. Calling `analyze` again makes a new analysis without your decisions.

    Attributes:
        source: The path of the original **as you gave it to `analyze`**. A relative one is resolved against the current
            folder, so analyse an absolute path when the analysis will be loaded from somewhere else.
        media: `"photo"` or `"video"`.
        fingerprint: The original's size, modification time and a hash of its content, which `render` compares.
        frames: The exact number of frames the decoder gave. 1 for a photo.
        fps: The frame rate. 0.0 for a photo.
        width: The width in pixels, as displayed.
        height: The height in pixels, as displayed.
        options: What the analysis was made with (see `Report.options`).
        second: The second look's counts: `zones_visited`, `zones_skipped`, `confirmed`, `added`, `vehicles_too_small`
            (vehicles under the text path's minimum size, never searched), `bands_added` and `band_area_share`. All 0 on a
            video at `detection="fast"`. On a photo at Fast, `zones_visited`, `zones_skipped` and `vehicles_too_small`
            count the vehicles the text path looked into, left out or found too small, and the others are 0.

    Example:
        ```python
        import os

        import anonymizer

        os.makedirs("out", exist_ok=True)
        with anonymizer.Anonymizer() as anon:
            analysis = anon.analyze(os.path.abspath("demo/street-westminster.jpg"))
            for item in analysis.items:
                if item.verdict == "check":
                    analysis.set_blur(item.item_id, True)  # blur every doubtful item
            analysis.save("out/westminster.analysis.json")
            result = anon.render(analysis, "out/westminster-checked.jpg")
        print(result.visible_check)  # 0
        ```

    See also:
        `Anonymizer`, `load_analysis`, `Item`.
    """

    def __init__(self, source: str, media: str, fingerprint: str, frames: int, fps: float, width: int, height: int):
        self.source = source
        self.media = media
        self.fingerprint = fingerprint
        self.frames = frames
        self.fps = fps
        self.width = width
        self.height = height
        self._items: Dict[int, Item] = {}
        self._areas: Dict[int, _Area] = {}
        self._video: Optional[VideoAnalysis] = None
        self._zones: List[Zone] = []
        self._photo_specs: Dict[str, PhotoSpec] = {}
        self._thumbnails: Dict[int, object] = {}
        self._thumb_dir: Optional[str] = None  # where the thumbnails of a loaded analysis are
        self._track_boxes: Optional[Tuple[Tuple[float, str], FinalBoxes]] = None
        self._area_boxes: Dict[int, FinalBoxes] = {}
        self.options: dict = {}  # the options fingerprint this was analysed with; analyze() fills it
        self.second: dict = dict(SECOND_LOOK_COUNTS)  # what the second look did; the engine of the second look fills it

    @property
    def items(self) -> List[Item]:
        """Every item, in order of appearance (by first frame, then by number). A list that is built at each access."""
        return sorted(self._items.values(), key=lambda i: (i.first_frame, i.item_id))

    @property
    def unchecked_frames(self) -> List[int]:
        """The frames of a video on which a detector failed. `render` refuses a video that has any.

        Always empty for a photo, whose failed model run already raised `IncompleteAnalysis` at `analyze`.
        """
        return list(self._video.unchecked) if self._video is not None else []

    def item(self, item_id: int) -> Item:
        """The item with this number, itself and not a copy.

        Args:
            item_id: An `Item.item_id` of this analysis.

        Returns:
            The `Item`.

        Raises:
            KeyError: The analysis has no such item.
        """
        try:
            return self._items[item_id]
        except KeyError:
            raise KeyError(f"no item {item_id} in this analysis") from None

    def thumbnail(self, item_id: int):
        """A small crop of an item, to show it to a person, as a numpy array in BGR order.

        Args:
            item_id: An `Item.item_id` of this analysis.

        Returns:
            The crop, or None for an added area, an item without a saved crop, or an unknown item.
        """
        crop = self._thumbnails.get(item_id)
        if crop is None:
            data = self._saved_thumbnail(item_id)
            if data is not None:
                import cv2
                import numpy as np

                crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return crop

    def thumbnail_jpeg(self, item_id: int) -> Optional[bytes]:
        """`thumbnail` as JPEG bytes, or None when the item has no crop.

        Args:
            item_id: An `Item.item_id` of this analysis.

        Returns:
            The JPEG bytes, or None.
        """
        crop = self._thumbnails.get(item_id)
        if crop is None:
            return self._saved_thumbnail(item_id)
        return previews.encode_jpeg(crop, previews.MIN_WIDTH)

    def _saved_thumbnail(self, item_id: int) -> Optional[bytes]:
        if self._thumb_dir is None:
            return None
        try:
            with open(os.path.join(self._thumb_dir, f"{item_id}.jpg"), "rb") as f:
                return f.read()
        except OSError:
            return None  # an added area, or a thumbnail that was never written

    def boxes(
        self, frame_from: int, frame_to: int, options: Optional["Options"] = None, only_blurred: bool = True
    ) -> Dict[int, List["BoxInfo"]]:
        """The final boxes of a span of frames, margins included, as `render` would blur them. No detector runs.

        Args:
            frame_from: The first frame.
            frame_to: The last frame, from `frame_from` to `frames - 1`.
            options: The settings to take the margins and gap filling from. None means `Options()`, **not** the options of
                your `Anonymizer`: pass those if you changed `extra_margin`, `gap_fill` or `fit`.
            only_blurred: True lists only the blurred items; False lists the ones left visible too.

        Returns:
            A dict that maps each frame number to a list of `BoxInfo`. A photo has the one key 0.

        Raises:
            ValueError: The frames are out of range, or `frame_from` is after `frame_to`.

        Example:
            ```python
            import anonymizer

            with anonymizer.Anonymizer() as anon:
                analysis = anon.analyze("demo/street-westminster.jpg")
                for box in analysis.boxes(0, 0, only_blurred=False)[0]:
                    print(box.item_id, box.kind, [round(v, 1) for v in box.box], box.blur)  # first line: 0 face [191.4, 335.2, 202.9, 351.7] True
            ```
        """
        o = options or Options()
        if not 0 <= frame_from <= frame_to < self.frames:
            raise ValueError(f"frames {frame_from}-{frame_to} outside 0-{self.frames - 1}")
        if self.media == "photo":
            return {0: [b for b in self._photo_boxes(o) if b.blur or not only_blurred]}
        tracks = self._all_track_boxes(o)
        out: Dict[int, List[BoxInfo]] = {}
        for t in range(frame_from, frame_to + 1):
            row = []
            for box, shape, item_id in tracks.get(t, []) + [
                b for a in self._area_boxes_of().values() for b in a.get(t, [])
            ]:
                item = self._items.get(item_id)
                if item is not None and (item.blur or not only_blurred):
                    row.append(BoxInfo(item_id, item.kind, tuple(round(v, 1) for v in box), shape, item.blur))
            out[t] = row
        return out

    def _all_track_boxes(self, o: "Options") -> FinalBoxes:
        """The boxes of every listed track, whatever its decision, cached while the options that shape them stay put."""
        key = (o.extra_margin, o.gap_fill)
        if self._track_boxes is None or self._track_boxes[0] != key:
            ids = {t.id for t in self._video.listed()}
            self._track_boxes = (key, boxes_to_blur(self._video, ids, o.extra_margin, GAP_FILL[o.gap_fill]))
        return self._track_boxes[1]

    def _area_boxes_of(self) -> Dict[int, FinalBoxes]:
        for area_id, area in self._areas.items():
            if area_id not in self._area_boxes:
                self._area_boxes[area_id] = manual_boxes(
                    self._video, area_id, area.frame_from, area.frame_to, area.box, area.follow
                )
        return self._area_boxes

    def _photo_boxes(self, o: "Options") -> List["BoxInfo"]:
        zones, specs = _photo_zones(self, o)
        height, width = self.height, self.width
        boxes = []
        for i, z in enumerate(zones):
            spec = specs[z.kind]
            item_id = i if i < len(self._zones) else _area_id_of(self, i - len(self._zones))
            boxes.append(BoxInfo(item_id, z.kind, with_margin(z.box, spec.margin, width, height), spec.shape, z.blur))
        return boxes

    def preview(
        self, frame: int, view: str = "anonymized", width: int = 960, options: Optional["Options"] = None
    ) -> bytes:
        """One frame as JPEG bytes, with the current decisions drawn on it. No detector runs.

        Args:
            frame: The frame.
            view: `"anonymized"` (the blur of the current decisions), `"original"`, `"boxes"` (every item outlined, dashed when
                it is left visible) or `"split"` (the original on the left, the anonymised frame on the right).
            width: The most pixels wide the JPEG is.
            options: The blur settings to draw with. None means `Options()`, **not** the options of your `Anonymizer`.

        Returns:
            The JPEG bytes.

        Raises:
            ValueError: `view` is not one of the four names, or the frame is out of range or could not be read.
            SourceChanged: The original changed after the analysis.
            FileNotFoundError: The original is not found from the current folder (a loaded analysis whose `source` is
                relative).
        """
        o = options or Options()
        if view not in previews.VIEWS:
            raise ValueError(f"view must be one of {previews.VIEWS}, not {view!r}")
        if not 0 <= frame < self.frames:
            raise ValueError(f"frame {frame} outside 0-{self.frames - 1}")
        if source_fingerprint(self.source) != self.fingerprint:
            raise SourceChanged(params={"path": self.source})
        original = previews.source_bgr(self.source, self.media, frame)
        if original is None:
            raise ValueError(f"frame {frame} could not be read")
        if view == previews.ORIGINAL:
            return previews.encode_jpeg(original, width)
        if view == previews.BOXES:
            shown = original.copy()
            found = self.boxes(frame, frame, o, only_blurred=False)[frame if self.media == "video" else 0]
            previews.draw_boxes(shown, [(b.kind, b.box, b.shape, b.blur) for b in found])
            return previews.encode_jpeg(shown, width)
        shown = original.copy()
        for b in self.boxes(frame, frame, o)[frame if self.media == "video" else 0]:
            blur_region(shown, b.box, b.shape, o.blur, o.strength, o.profile)
        if view == previews.SPLIT:
            shown = previews.split(original, shown)
        return previews.encode_jpeg(shown, width)

    def report(self, clip: Optional[str] = None) -> "Report":
        """The privacy report of the current decisions, as `Anonymizer.render` would return it.

        It is built at each call. `clip` is the name to put in the `clip` column of every row; None means the original's base
        name.
        """
        return _report(self, clip or os.path.basename(self.source))

    def set_blur(self, item_id: int, blur: bool) -> None:
        """Decide an item: `blur` True blurs it, False leaves it visible.

        The item then counts as a person's decision (`decided_by` becomes `"person"`), and the report records it. Verdicts,
        reasons and scores do not change. Raises `KeyError` when the analysis has no item with that number.
        """
        item = self.item(item_id)
        item.blur = bool(blur)
        item.decided_by = "person"

    def summary(self) -> dict:
        """The dict that `Report.summary` returns for the current decisions, with the second look's counts."""
        return {**self.report().summary(), **self.second}

    def add_area(self, frame_from: int, frame_to: int, box: Box, follow: str = "camera") -> Item:
        """Blur an area the detectors missed.

        The new item has kind `"manual"`, verdict `"blur"`, reason `manual.reason.added`, `decided_by` `"person"`, and the
        number after the highest one in use.

        Args:
            frame_from: The first frame the area is on.
            frame_to: The last frame; in a video the area is carried from `frame_from` to here.
            box: `(x0, y0, x1, y1)` as the thing appears at `frame_from`. Only the part inside the picture is kept.
            follow: `"camera"` carries the area with the camera's motion, and stays put at a change of shot until the new
                shot moves; `"fixed"` keeps it still. These are not the `gap_fill` names.

        Returns:
            The new `Item`.

        Raises:
            ValueError: The box is not finite, under a pixel or outside the picture, or `follow` or a frame is not valid.
        """
        x0, y0, x1, y1 = (float(v) for v in box)
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            raise ValueError(f"the box must be four finite numbers: {box}")
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise ValueError(f"empty box: {box}")
        # Only the part inside the picture can be hidden: a box outside it (coordinates of another size) would be reported as
        # blurred and draw nothing.
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(self.width), x1), min(float(self.height), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise ValueError(f"the box {tuple(box)} is outside the {self.width}x{self.height} picture")
        if follow not in ("camera", "fixed"):
            raise ValueError(f"follow must be 'camera' or 'fixed', not {follow!r}")
        if not 0 <= frame_from <= frame_to < self.frames:
            raise ValueError(f"frames {frame_from}-{frame_to} outside 0-{self.frames - 1}")
        item_id = max(self._items, default=-1) + 1
        self._areas[item_id] = _Area(frame_from, frame_to, (x0, y0, x1, y1), follow)
        self._area_boxes.pop(item_id, None)
        item = Item(
            item_id,
            MANUAL,
            Verdict.BLUR.value,
            "manual.reason.added",
            frame_from,
            frame_to,
            1.0,
            True,
            {},
            MANUAL,
            None,
            "person",
        )
        self._items[item_id] = item
        return item

    def remove_area(self, item_id: int) -> None:
        """Remove an area added with `add_area`; a detected item is turned off with `set_blur` instead.

        Args:
            item_id: The number of an added area.

        Raises:
            KeyError: The item is not an added area.
        """
        if item_id not in self._areas:
            raise KeyError(f"item {item_id} is not an added area")
        del self._areas[item_id]
        del self._items[item_id]
        self._area_boxes.pop(item_id, None)

    def save(self, path: str) -> None:
        """Write the analysis, its decisions and its thumbnails, so that `load_analysis` can read them back.

        It replaces an existing file, and writes the item crops to a folder beside it: the path without `.json`, plus `.thumbs`.

        Args:
            path: Where to write the JSON. The folder must exist.

        Raises:
            OSError: The file or the folder cannot be written.
        """
        data = {
            "version": ANALYSIS_VERSION,
            "source": self.source,
            "media": self.media,
            "fingerprint": self.fingerprint,
            "frames": self.frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "options": self.options,
            "second": self.second,
            "items": [asdict(i) for i in self.items],
            "areas": {str(k): asdict(a) for k, a in self._areas.items()},
        }
        if self._video is not None:
            data["video"] = self._video.to_json()
        else:
            data["zones"] = [_zone_json(z) for z in self._zones]
            data["photo_specs"] = {
                k: {"margin": list(s.margin), "shape": s.shape} for k, s in self._photo_specs.items()
            }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
        self._save_thumbnails(thumbnails_folder(path))

    def _save_thumbnails(self, folder: str) -> None:
        """Write the thumbnails of this analysis to `folder` and nothing else: what an earlier analysis left there (other items
        with the same numbers) is removed, so a reloaded analysis never shows another object's crop"""
        crops = {i: self.thumbnail_jpeg(i) for i in self._items}  # read first: they may come from this very folder
        crops = {i: data for i, data in crops.items() if data is not None}
        if not crops and not os.path.isdir(folder):
            return
        # The new folder is written whole beside the old one and then put in its place: a death half way leaves the earlier thumbnails
        # as they were (or, between the two renames, as `<folder>.old`, which load_analysis falls back to), never a half-written set.
        fresh, old = folder + ".new", folder + ".old"
        shutil.rmtree(fresh, ignore_errors=True)
        os.makedirs(fresh)
        for item_id, data in crops.items():
            with open(os.path.join(fresh, f"{item_id}.jpg"), "wb") as f:
                f.write(data)
        shutil.rmtree(old, ignore_errors=True)
        if os.path.isdir(folder):
            os.replace(folder, old)
        os.replace(fresh, folder)
        shutil.rmtree(old, ignore_errors=True)
        self._thumb_dir = folder


def thumbnails_folder(path: str) -> str:
    """Where Analysis.save(path) keeps the thumbnails: next to the file, named after it."""
    return (path[: -len(".json")] if path.endswith(".json") else path) + ".thumbs"


def load_analysis(path: str) -> Analysis:
    """Read an analysis that `Analysis.save` wrote. Nothing is analysed, and the original is not opened.

    Whether the original is still there is found out by `Anonymizer.render` or `Analysis.preview`. A file edited by hand
    renders as edited.

    Args:
        path: The JSON file that `Analysis.save` wrote.

    Returns:
        The `Analysis`, with its items, decisions and added areas.

    Raises:
        AnonymizerError: `error.analysis.version`: the file has another format version (it has been 3 since 2.1, so only
            older files are refused); `params["version"]` is the one found.
        FileNotFoundError: There is no such file.
        json.JSONDecodeError: The file is not JSON.
        KeyError: The file claims the right version but lacks a field.

    See also:
        `Analysis.save`, `Anonymizer.render`.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != ANALYSIS_VERSION:
        raise AnonymizerError("error.analysis.version", {"version": data.get("version")}, path)
    a = Analysis(
        data["source"], data["media"], data["fingerprint"], data["frames"], data["fps"], data["width"], data["height"]
    )
    a.options = dict(data.get("options", {}))
    a.second = {**SECOND_LOOK_COUNTS, **data.get("second", {})}
    a._items = {i["item_id"]: Item(**i) for i in data["items"]}
    a._areas = {
        int(k): _Area(v["frame_from"], v["frame_to"], tuple(v["box"]), v["follow"]) for k, v in data["areas"].items()
    }
    folder = thumbnails_folder(path)
    if not os.path.isdir(folder) and os.path.isdir(folder + ".old"):
        folder += ".old"  # the save was cut between its two renames: the earlier thumbnails are still there
    a._thumb_dir = folder if os.path.isdir(folder) else None
    if "video" in data:
        a._video = VideoAnalysis.from_json(data["video"])
    else:
        a._zones = [_zone_of_json(z) for z in data["zones"]]
        a._photo_specs = {k: PhotoSpec(tuple(s["margin"]), s["shape"]) for k, s in data["photo_specs"].items()}
    return a


def _zone_json(z: Zone) -> dict:
    j = z.judgement
    return {
        "box": list(z.box),
        "kind": z.kind,
        "score": z.score,
        "blur": z.blur,
        "found_by": z.found_by,
        "second_score": z.second_score,
        "judgement": {"verdict": j.verdict.value, "reason": j.reason, "params": j.params},
    }


def _zone_of_json(z: dict) -> Zone:
    j = z["judgement"]
    return Zone(
        tuple(z["box"]),
        z["kind"],
        z["score"],
        z["blur"],
        Judgement(Verdict(j["verdict"]), j["reason"], j["params"]),
        found_by=z.get("found_by", "light"),
        second_score=z.get("second_score"),
    )


def _photo_zones(a: Analysis, o: Options) -> Tuple[List[Zone], Dict[str, PhotoSpec]]:
    """The zones of a photo with the user's decisions, areas added by hand last, and the margins with `o.extra_margin`."""
    zones = [replace(z, blur=a._items[i].blur) for i, z in enumerate(a._zones)]
    for area_id, area in a._areas.items():
        zones.append(
            Zone(area.box, MANUAL, 1.0, a._items[area_id].blur, Judgement(Verdict.BLUR, "manual.reason.added"))
        )
    specs = dict(a._photo_specs)
    specs[MANUAL] = PhotoSpec(MANUAL_MARGIN, RECT)
    specs = {
        k: PhotoSpec((s.margin[0] + o.extra_margin, s.margin[1] + o.extra_margin), s.shape) for k, s in specs.items()
    }
    return zones, specs


def _area_id_of(a: Analysis, index: int) -> int:
    return list(a._areas)[index]


def media_info(path: str) -> MediaInfo:
    """Read what a file is, without analysing it: size, length, codec, audio and location.

    Use it to check a file before the long work, or to see that a render removed the location.

    Args:
        path: A photo or a video. The kind is told from the extension.

    Returns:
        The `MediaInfo`.

    Raises:
        UnsupportedMedia: The extension is not supported, or the file is empty, damaged or cannot be opened.
        FileNotFoundError: The file does not exist.

    Example:
        ```python
        import anonymizer

        info = anonymizer.media_info("demo/street-hongkong.mp4")
        print(info.width, info.height, info.fps, info.has_gps)  # 960 540 29.97002997002997 True
        ```
    """
    media = media_of(path)
    name = os.path.basename(path)
    size = os.path.getsize(path)
    if media == "video":
        import av

        from .engine import video_io

        try:
            info = video_io.probe(path)
            with av.open(path) as container:
                gps = any("location" in k.lower() for k in container.metadata) or any(
                    "location" in k.lower() for s in container.streams.video for k in s.metadata
                )
        except (av.error.FFmpegError, OSError, ValueError) as e:
            raise UnsupportedMedia(params={"name": name}, detail=f"{type(e).__name__}: {e}") from e
        frames = info.frames
        return MediaInfo(
            "video",
            info.width,
            info.height,
            frames,
            info.fps,
            round(info.duration, 3),
            info.codec,
            info.has_audio,
            gps,
            size,
        )
    from PIL import Image

    from .engine import photos

    try:
        with Image.open(path) as im:
            fmt = (im.format or "").lower()
            width, height = im.size  # the size comes from the header: the picture is not decoded to know it
            if im.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                width, height = height, width
        with open(path, "rb") as f:
            gps = photos.has_gps(f.read())
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
        raise UnsupportedMedia(params={"name": name}, detail=f"{type(e).__name__}: {e}") from e
    return MediaInfo("photo", width, height, 1, 0.0, 0.0, fmt, False, gps, size)


def frame_jpeg(path: str, frame: int = 0, width: int = 960) -> bytes:
    """One frame of a photo or video as it is, unblurred, as JPEG bytes.

    Args:
        path: A photo or a video.
        frame: The frame. A photo is frame 0.
        width: The most pixels wide the JPEG is; a smaller picture is not enlarged.

    Returns:
        The JPEG bytes.

    Raises:
        UnsupportedMedia: The extension is not supported.
        ValueError: The frame does not exist or could not be read.
    """
    image = previews.source_bgr(path, media_of(path), frame)
    if image is None:
        raise ValueError(f"frame {frame} could not be read")
    return previews.encode_jpeg(image, width)


def thumbnail_jpeg(path: str, width: int = 240) -> bytes:
    """A small JPEG of a file, to show it in a list: the photo, or the frame at a tenth of a video's length.

    Args:
        path: A photo or a video.
        width: The most pixels wide the JPEG is.

    Returns:
        The JPEG bytes. A video whose frame at a tenth cannot be read gives its first frame.

    Raises:
        UnsupportedMedia: The extension is not supported, the file cannot be opened, or no frame can be read.
        FileNotFoundError: The file does not exist.
    """
    info = media_info(path)
    frame = int(info.frames * 0.1) if info.media == "video" else 0
    image = previews.source_bgr(path, info.media, frame)
    if image is None and frame:
        image = previews.source_bgr(path, info.media, 0)
    if image is None:
        raise UnsupportedMedia(params={"name": os.path.basename(path)}, detail="no frame could be read")
    return previews.encode_jpeg(image, width)


def media_of(path: str) -> str:
    """ "video" or "photo", from the file extension; raises UnsupportedMedia for anything else."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in PHOTO_EXTENSIONS:
        return "photo"
    kind = "a format that is not supported" if ext in UNSUPPORTED_EXTENSIONS else "not a photo or a video"
    raise UnsupportedMedia(params={"name": os.path.basename(path)}, detail=f"{ext or 'no extension'}: {kind}")


def _guarded(persons: Callable) -> Callable:
    """A person finder that raises is a failed look, never "nobody there": the analysis is not complete (and a client gets a code, not
    the finder's own exception). A stop is not a failure."""

    def find(bgr):
        try:
            return persons(bgr)
        except events.Cancelled:
            raise
        except Exception as exc:
            raise events.ClipError(
                "error.analysis.incomplete", {"photos": 1}, f"person finder: {type(exc).__name__}: {exc}"
            ) from exc

    return find


def _upright_size(path: str) -> Optional[Tuple[int, int]]:
    """Width and height of a photo as it is analysed (upright by its EXIF orientation), without decoding it; None if that cannot be told."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            width, height = im.size
            orientation = im.getexif().get(0x0112, 1)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return None
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def _track_found_by(track) -> str:
    """Who found a video track: the light models when any of its views came from a sweep (a heavy view that replaced one counts
    for it), else the heavy model that gave its first view."""
    if track.second_over_sweep or any(d.source not in (contract.SECOND, contract.DEEP) for d in track.views.values()):
        return "light"
    return track.views[track.first].source


class Anonymizer:
    """Finds faces and plates in photos and videos, and writes the anonymised copies. Detectors are loaded once.

    Make one, use it in a `with` block, and call `analyze` and `render` for each file: `analyze` writes nothing, and
    `render` writes the output without running a detector. The models load at the first call that needs them, from
    `models_dir`, else `ANONYMIZER_MODELS`, else `./models`, else the `models` folder next to the source tree, and are
    checked against that folder's `SHA256SUMS`.

    Each argument below is kept as an attribute of the same name; `calibrate` sets `calibration_ms`. An `Anonymizer` is not
    safe to share between threads: give each thread its own. `analyze`, `render` and the event handlers run in the calling
    thread. `CancelToken` is the one object meant to be used from another thread.

    Args:
        options: The `Options`. None means the defaults. Kept as the attribute `options`; assign a new one between calls
            to change the settings read at render.
        models_dir: The folder with the model files. None means the search above.
        threads: ONNX Runtime threads per model. 0 lets the runtime decide.
        modules: Detectors of your own instead of the built-in ones, for tests; `close` does not free them.
        persons: A function that returns the persons in a BGR image, for tests. None means the built-in person finder.
        demo: True refuses `detection="deep"` (code `error.demo.deep`), as the Studio's demo mode does.
        calibration_ms: A stored time of one heavy call on this machine, in milliseconds, for `estimate`.
        light_ms: A stored result of `calibrate_light`: the time of the light first pass of a photo, in milliseconds.

    Example:
        ```python
        import os

        import anonymizer

        os.makedirs("out", exist_ok=True)
        with anonymizer.Anonymizer() as anon:
            for path in ["demo/street-westminster.jpg", "demo/street-taipei.jpg"]:
                analysis = anon.analyze(os.path.abspath(path))
                result = anon.render(analysis, os.path.join("out", os.path.basename(path)))
                print(path, len(analysis.items), "items,", result.visible_check, "doubtful left visible")
        ```

    See also:
        `Options`, `Analysis`, `anonymize_image`.
    """

    def __init__(
        self,
        options: Optional[Options] = None,
        *,
        models_dir: Optional[str] = None,
        threads: int = 0,
        modules: Optional[Sequence[ClassModule]] = None,
        persons: Optional[Callable] = None,
        demo: bool = False,
        calibration_ms: Optional[float] = None,
        light_ms: Optional[float] = None,
    ):
        self.options = options or Options()
        self.models_dir = models_dir
        self.threads = threads
        self.demo = demo
        self.calibration_ms = calibration_ms
        self.light_ms = light_ms
        self._light_measured: Dict[Tuple[int, int], float] = {}
        self._modules: Optional[List[ClassModule]] = list(modules) if modules is not None else None
        self._own_modules = modules is None
        self._persons = persons
        self._person_finder = None

    def __enter__(self) -> "Anonymizer":
        """Return the `Anonymizer` itself, so that it can be used in a `with` block."""
        return self

    def __exit__(self, *exc) -> None:
        """Free the detectors at the end of a `with` block, as `close` does.

        The arguments are the exception type, value and traceback when the block ended in an exception. They are not used, and the
        exception is not suppressed.
        """
        self.close()

    def close(self) -> None:
        """Free the detectors this `Anonymizer` loaded. It can be used again: the models then load again."""
        if self._modules is not None and self._own_modules:
            for m in self._modules:
                m.close()
            self._modules = None
        if self._person_finder is not None:
            self._person_finder.close()
            self._person_finder = self._persons = None

    def modules(self) -> List[ClassModule]:
        """The detectors in use, loading them now if this is the first call: a way to find a missing model early.

        Returns:
            The list of detectors.

        Raises:
            ModelLoadError: A model file is missing, not listed in `SHA256SUMS`, changed, or its runtime failed.
        """
        if self._modules is None:
            from . import classes

            try:
                self._modules = classes.load(
                    self.options.kinds(),
                    self.threads,
                    self.models_dir,
                    plate_path=self.options.plate_path,
                    detection=self.options.detection,
                    vehicle_model=self.options.vehicle_model,
                    plate_band=self.options.plate_band,
                    fit=self.options.fit,
                )
            except contract.ModelLoadError as e:
                raise ModelLoadError(params={"kind": e.kind}, detail=e.detail) from e
        return self._modules

    def _find_persons(self):
        """The person finder of the second look, or None when it has no use (no heavy face model in use, or "fast"/"deep" detection)."""
        if self._persons is not None:
            return _guarded(self._persons)
        faces = [m for m in heavy_modules(self.modules()) if m.kind == FACE]
        if self.options.detection == "fast" or not faces:
            return None
        from .classes.context import PersonFinder

        try:
            self._person_finder = PersonFinder(self.threads, self.models_dir)
        except contract.ModelLoadError as e:
            raise ModelLoadError(params={"kind": e.kind}, detail=e.detail) from e
        self._persons = self._person_finder.persons
        return _guarded(self._persons)

    def calibrate(self) -> float:
        """Time one call of the heavy model on this machine, keep it for `estimate`, and return it.

        The median of three runs on a 640 pixel picture, stored in `calibration_ms`. It loads the models.

        Returns:
            The time in milliseconds.

        Raises:
            ModelLoadError: No detector in use has a heavy model (always the case with `detection="fast"`).
        """
        from .engine import deep

        heavy = heavy_modules(self.modules())
        if not heavy:
            raise ModelLoadError(params={"kind": "heavy"}, detail="no module in use has a heavy model")
        self.calibration_ms = deep.calibrate(heavy[0])
        return self.calibration_ms

    def calibrate_light(self, width: int, height: int) -> float:
        """Time the light first pass of a photo of this size on this machine, keep it, and return it.

        It loads the models. `light_ms`, when set, replaces it for every size.

        Args:
            width: The photo's width in pixels.
            height: The photo's height in pixels.

        Returns:
            The time in milliseconds.
        """
        from .engine.photos import calibrate_light

        key = (int(width), int(height))
        if key not in self._light_measured:
            self._light_measured[key] = calibrate_light(self.modules(), width, height, self.options.detection)
        return self._light_measured[key]

    def estimate(self, path: str) -> Estimate:
        """What analysing a file is expected to cost with the options in force.

        It may time this machine first (`calibrate`, `calibrate_light`), and so load the models. A video's estimate counts only
        the heavy calls of its key frames.

        Args:
            path: A photo or a video.

        Returns:
            The `Estimate`.

        Raises:
            UnsupportedMedia: The extension is not supported or the file cannot be opened.
            AnonymizerError: The file is not there (`error.source.missing`), a picture is over the deep level's tile limit
                (`error.deep.too_large`), a clip over its work limit (`error.deep.too_much_work`), or the deep level is asked
                for in demo mode (`error.demo.deep`).
            ModelLoadError: A model could not be loaded, or no detector in use has the heavy model the estimate needs.
        """
        from .engine import deep, keyframes
        from .engine.second_look import DEFAULT_MAX_ZONES

        if not os.path.isfile(path):
            raise AnonymizerError("error.source.missing", {"path": path})
        info = media_info(path)
        mode = self.options.detection
        self._allow_detection()
        if info.media == "photo":
            width, height = info.width, info.height

            def light_seconds() -> float:  # after the deep refusals: nothing runs for a picture that will be refused
                ms = self.light_ms if self.light_ms is not None else self.calibrate_light(width, height)
                return ms / 1000.0

            if mode == "fast":
                return Estimate(round(light_seconds(), 2), 0, 0)
            ms = self.calibration_ms if self.calibration_ms is not None else self.calibrate()
            if mode == "careful":
                cap = DEFAULT_MAX_ZONES if self.options.max_zones is None else self.options.max_zones
                zones = min(cap, EXPECTED_ZONES)
                return Estimate(round(light_seconds() + zones * ms / 1000.0, 2), zones, 0)
            try:
                found = deep.estimate(width, height, ms)
            except deep.DeepRefused as e:
                raise AnonymizerError(e.key, e.params) from e
            return Estimate(round(light_seconds() + found.seconds, 2), 0, found.tiles)
        if mode == "fast":
            return Estimate(0.0, 0, 0)
        ms = self.calibration_ms if self.calibration_ms is not None else self.calibrate()
        params = keyframes.KeyFrameParams(deep_ms_per_frame=ms)
        if mode == "deep":
            try:
                keyframes.refuse_deep_if_too_large(info.frames, info.fps, params)
            except events.ClipError as e:
                raise _api_error(e) from e
            sweeps = math.ceil(info.frames / params.deep_period_frames(info.fps)) + 1
            return Estimate(round(sweeps * ms / 1000.0, 2), sweeps, 0)
        seconds = info.frames / max(1.0, info.fps)
        calls = int(seconds * params.heavy_per_second)
        return Estimate(round(calls * ms / 1000.0, 2), calls, 0)

    def _allow_detection(self) -> None:
        from .engine import deep

        try:
            deep.allow_detection(self.options.detection, self.demo)
        except deep.DeepRefused as e:
            raise AnonymizerError(e.key, e.params) from e

    def analyze(
        self, path: str, on_event: Optional[EventHandler] = None, cancel: Optional[CancelToken] = None
    ) -> Analysis:
        """Find the faces and plates in a photo or a video. Nothing is written, and the file is only read.

        It checks the file before it loads the models at their first use. A photo's first pass cannot be broken off: a cancel
        is honoured only between the heavy calls of the second look.

        Args:
            path: A photo or a video. Use an absolute path if the analysis will be saved and loaded from another folder.
            on_event: A function that takes an `Event`. It runs in this thread, and an exception in it stops the call.
            cancel: A `CancelToken`.

        Returns:
            The `Analysis`.

        Raises:
            UnsupportedMedia: The extension is not supported or the file cannot be opened (`error.media.unsupported`), a video
                stops decoding partway (`error.media.damaged`), or a photo holds several images (`error.photo.several_images`).
            AnonymizerError: The file is not there (`error.source.missing`), or a deep or demo code of `Anonymizer.estimate`.
            ModelLoadError: A model could not be loaded.
            IncompleteAnalysis: A photo's model run or the person finder failed.
            Cancelled: The job was cancelled.
        """
        emit = on_event or (lambda e: None)
        media = media_of(path)
        clip = os.path.basename(path)
        if not os.path.isfile(path):
            raise AnonymizerError("error.source.missing", {"path": path})
        fingerprint = source_fingerprint(path)
        self._allow_detection()
        try:
            if media == "video":
                return self._analyze_video(path, clip, fingerprint, emit, cancel)
            return self._analyze_photo(path, clip, fingerprint, emit, cancel)
        except events.Cancelled as e:
            emit(Finished(clip, "stopped"))
            raise Cancelled() from e
        except events.ClipError as e:
            raise _api_error(e) from e
        except (
            _ffmpeg_error()
        ) as e:  # the file stops being readable partway: a typed error of this clip, nothing written
            raise _damaged(clip, e) from e

    def _analyze_video(self, path, clip, fingerprint, emit, cancel) -> Analysis:
        import av

        from .engine import video_io

        try:
            info = video_io.probe(path)
        except (av.error.FFmpegError, OSError, ValueError) as e:  # FFmpeg cannot open it, or it has no video
            raise UnsupportedMedia(params={"name": clip}, detail=f"{type(e).__name__}: {e}") from e
        emit(Started(clip, "video", info.frames, info.width, info.height, info.fps))
        progress, seen = _progress(emit, clip, "analyze"), {}
        vehicles_before = _vehicle_counts(self.modules())
        va = analyze_video(
            path,
            self.modules(),
            self.options.thoroughness,
            GAP_FILL[self.options.gap_fill],
            lambda fraction, frame, frames: progress(fraction, frame, frames, seen),
            _flag(cancel),
            live=seen.update,
            detection=self.options.detection,
            key_params=KeyFrameParams(deep_ms_per_frame=self.calibration_ms),
            persons=self._find_persons() if self.options.detection != "fast" else None,
        )
        progress(1.0, va.frames - 1, va.frames)
        a = Analysis(path, "video", fingerprint, va.frames, va.fps, va.width, va.height)
        a._video = va
        a.second["zones_visited"] += va.second.get("calls", 0)  # heavy looks at the key frames
        a.second["zones_skipped"] += va.second.get("skipped", 0)
        a.second["vehicles_too_small"] += _vehicle_counts(self.modules())[2] - vehicles_before[2]
        for track in va.listed():
            j = va.judgements[track.id]
            item = Item(
                track.id,
                track.kind,
                j.verdict.value,
                j.reason,
                track.first,
                track.last,
                round(track.max_score, 3),
                blurred_by_default(j.verdict, self.options.unsure, j.params),
                dict(j.params),
                _track_found_by(track),
                round(track.best_second, 3) if track.n_second else None,
            )
            a._items[track.id] = item
            a._thumbnails[track.id] = va.thumbnails.get(track.id, (None, 0))[0]
        skipped = va.second.get("skipped", 0)
        if skipped:  # the budget refused some heavy looks: said, never silent
            emit(JobWarning(clip, "warning.second_look.zones_skipped", {"count": skipped}))
        return self._announce(a, clip, emit)

    def _analyze_photo(self, path, clip, fingerprint, emit, cancel=None) -> Analysis:
        from PIL import Image

        from .engine import deep

        flag = _flag(cancel)
        progress = _progress(emit, clip, "analyze")
        started = []

        def stage_progress(stage: str, done: int, total: int) -> None:
            if stage == "first":
                progress(0.5, 0, 1, stage="first")
            else:
                zones = {"done": done, "total": total} if stage == "second" else None
                progress(0.5 + 0.5 * done / max(1, total), 0, 1, stage=stage, zones=zones)

        try:
            if self.options.detection == "deep":
                # A picture the deep sweep would refuse is refused before the first pass is spent on it (USAGE: "before anything runs").
                size = _upright_size(path)
                if size is not None:
                    deep.estimate(size[0], size[1], 0.0)
            seen = _vehicle_counts(self.modules())
            bands_before = _band_counts(self.modules())
            if self.options.detection != "fast" and heavy_modules(self.modules()):
                size = _upright_size(path)  # Started goes first, so that the progress of the heavy stage follows it
                if size is not None:
                    emit(Started(clip, "photo", 1, size[0], size[1], 0.0))
                    started.append(size)
            persons = self._find_persons() if self.options.detection == "careful" else None
            pa = analyze_photo(
                path,
                self.modules(),
                self.options.unsure,
                self.options.detection,
                persons,
                self.options.max_zones,
                self.options.heavy_only,
                stage_progress,
                flag.check if flag is not None else None,
            )
        except deep.DeepRefused as e:
            raise AnonymizerError(e.key, e.params) from e
        except (
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as e:  # Pillow cannot identify or decode it
            raise UnsupportedMedia(params={"name": clip}, detail=f"{type(e).__name__}: {e}") from e
        if not started:
            emit(Started(clip, "photo", 1, pa.width, pa.height, 0.0))
        if pa.second.stage != "first":
            progress(1.0, 0, 1, stage=pa.second.stage)
        a = Analysis(path, "photo", fingerprint, 1, 0.0, pa.width, pa.height)
        a._zones = pa.zones
        a._photo_specs = specs_of(self.modules())
        visited, skipped, too_small = (
            now - before for now, before in zip(_vehicle_counts(self.modules()), seen, strict=True)
        )
        # zones visited and skipped: the vehicles the plate path looked into or beyond its budget, and the second look's zones
        left_out = skipped + pa.second.zones_skipped
        a.second["zones_visited"] += visited + pa.second.zones_visited
        a.second["zones_skipped"] += left_out
        a.second["confirmed"] += pa.second.confirmed
        a.second["added"] += pa.second.added
        a.second["vehicles_too_small"] += too_small
        bands, band_area, banded_area = (
            now - before for now, before in zip(_band_counts(self.modules()), bands_before, strict=True)
        )
        a.second["bands_added"] += int(bands)
        a.second["band_area_share"] = round(band_area / banded_area, 3) if banded_area else 0.0
        if left_out:  # said, as for a video: the places the budget left out keep the first pass's result
            emit(JobWarning(clip, "warning.second_look.zones_skipped", {"count": left_out}))
        for i, z in enumerate(pa.zones):
            j = z.judgement
            a._items[i] = Item(
                i,
                z.kind,
                j.verdict.value,
                j.reason,
                0,
                0,
                round(z.score, 3),
                z.blur,
                dict(j.params),
                z.found_by,
                z.second_score,
            )
            a._thumbnails[i] = pa.thumbnails[i]
        return self._announce(a, clip, emit)

    def _announce(self, a: Analysis, clip: str, emit) -> Analysis:
        a.options = options_fingerprint(self.options, self.modules())
        for item in a.items:
            emit(ItemFound(clip, item))
        emit(
            Analyzed(
                clip,
                len(a._items),
                sum(1 for i in a._items.values() if i.verdict == Verdict.CHECK.value),
                a.second["zones_visited"],
                a.second["zones_skipped"],
                a.second["added"],
            )
        )
        return a

    def render(
        self,
        analysis: Analysis,
        target: str,
        on_event: Optional[EventHandler] = None,
        cancel: Optional[CancelToken] = None,
    ) -> Result:
        """Write the anonymised file from an analysis and its decisions. No detector runs.

        It checks that `target` does not exist, that its folder does and that the original is the file that was analysed, then
        writes a hidden partial file, checks it and renames it to `target` when it is whole; if anything fails, nothing is left
        under that name. A photo keeps the **format of the original** whatever the name of `target` says, and a video is an MP4
        (the reference page "Files written" has what each keeps). Doubtful items left visible are counted in
        `Result.visible_check`.

        Args:
            analysis: An `Analysis` from `analyze` or `load_analysis`. A relative `Analysis.source` is looked for from the
                current folder.
            target: Where to write the output. It must not exist, and its folder must.
            on_event: A function that takes an `Event`.
            cancel: A `CancelToken`.

        Returns:
            The `Result`.

        Raises:
            OutputExists: `target` exists, or appeared while writing.
            FileNotFoundError: The folder of `target` does not exist, or a loaded analysis's relative source is not found.
            NotADirectoryError: The folder of `target` is a file.
            SourceChanged: The original differs from the one analysed.
            IncompleteAnalysis: A video has `Analysis.unchecked_frames`.
            UnsupportedMedia: The original stopped decoding partway (`error.media.damaged`).
            AnonymizerError: A written file failed its final check and was discarded (`error.output.check_failed`,
                `error.photo.metadata_not_removed`, `error.photo.gps_not_removed`), or a photo does not fit in free memory
                (`error.photo.not_enough_memory`).
            Cancelled: The job was cancelled.
        """
        emit = on_event or (lambda e: None)
        clip = os.path.basename(analysis.source)
        if os.path.exists(target):
            raise OutputExists(params={"path": target})
        _destination_folder_exists(target)
        if source_fingerprint(analysis.source) != analysis.fingerprint:
            raise SourceChanged(params={"path": analysis.source})
        start = time.monotonic()
        try:
            if analysis.media == "video":
                warnings = self._render_video(analysis, target, clip, emit, cancel)
            else:
                warnings = self._render_photo(analysis, target)
        except events.Cancelled as e:
            emit(Finished(clip, "stopped"))
            raise Cancelled() from e
        except events.ClipError as e:
            raise _api_error(e) from e
        except _ffmpeg_error() as e:  # unreadable source or a failed write: the partial file is already gone
            raise _damaged(clip, e) from e
        for code in warnings:
            emit(JobWarning(clip, code))
        left = sum(1 for i in analysis._items.values() if i.verdict == Verdict.CHECK.value and not i.blur)
        if left:
            # Only with unsure="keep", with "placed" for an item that is not placed (or a person's decision) does a "Not sure" item leave visible: said every time.
            warnings = [*warnings, "warning.review.check_left_visible"]
            emit(JobWarning(clip, "warning.review.check_left_visible", {"count": left}))
        seconds = round(time.monotonic() - start, 2)
        emit(Rendered(clip, target, seconds))
        emit(Finished(clip, "done"))
        return Result(target, _report(analysis, clip), warnings, seconds, left)

    def _render_video(self, a: Analysis, target, clip, emit, cancel) -> List[str]:
        va = a._video
        blur_ids = {i.item_id for i in a._items.values() if i.blur and i.item_id not in a._areas}
        manual = {}
        for area_id, area in a._areas.items():
            if a._items[area_id].blur:
                for t, boxes in manual_boxes(
                    va, area_id, area.frame_from, area.frame_to, area.box, area.follow
                ).items():
                    manual.setdefault(t, []).extend(boxes)
        o = self.options
        progress = _progress(emit, clip, "render")
        result = render_video(
            a.source,
            target,
            va,
            blur_ids,
            o.blur,
            o.strength,
            GAP_FILL[o.gap_fill],
            progress,
            _flag(cancel),
            self.threads,
            o.extra_margin,
            manual,
            lambda: _still_analysed(a),
            o.profile,
        )
        progress(1.0, va.frames - 1, va.frames)
        return result.warnings

    def _render_photo(self, a: Analysis, target) -> List[str]:
        o = self.options
        zones, specs = _photo_zones(a, o)
        partial = partial_path(target)
        try:
            result = process(
                a.source,
                partial,
                zones=copy.deepcopy(zones),
                specs=specs,
                strip_gps=o.remove_gps,
                style=o.blur,
                strength=o.strength,
                profile=o.profile,
            )
            _still_analysed(a)
            if os.path.exists(target):
                raise OutputExists(params={"path": target})
            os.replace(partial, target)
        finally:
            if os.path.exists(partial):
                os.remove(partial)
        return result.warnings


def _vehicle_counts(modules: Sequence[ClassModule]) -> Tuple[int, int, int]:
    """(vehicles looked into, vehicles beyond the budget, vehicles too small to read) so far, summed over the plate modules that count
    them."""
    return (
        sum(getattr(m, "vehicles_visited", 0) for m in modules),
        sum(getattr(m, "vehicles_not_visited", 0) for m in modules),
        sum(getattr(m, "vehicles_too_small", 0) for m in modules),
    )


def _band_counts(modules: Sequence[ClassModule]) -> Tuple[float, float, float]:
    """(plate-position bands added, pixels they blur, pixels of the boxes of the vehicles that got one) so far, summed over the plate modules."""
    return (
        float(sum(getattr(m, "bands_added", 0) for m in modules)),
        sum(getattr(m, "band_area", 0.0) for m in modules),
        sum(getattr(m, "banded_vehicle_area", 0.0) for m in modules),
    )


def options_fingerprint(options: Options, modules: Sequence[ClassModule]) -> dict:
    """What an analysis was made with, as plain data: the detection options that change what is found or decided, and the
    name and SHA-256 of every heavy model loaded (the second look's, deep mode's, the text detector's). `detection_ran` is what was
    really done: "fast" when the detection asked for was "careful" or "deep" but no module in use has a heavy model (modules given
    by the caller); `heavy_only` is recorded only when a deep sweep really ran. Saved in the analysis file and shown in the report, so a result says which models and which privacy rule
    produced it."""
    models = {}
    for m in modules:
        for attribute in ("heavy_zone", "heavy_deep", "text"):
            model = getattr(m, attribute, None)
            if model is not None:
                models[(model.name, model.sha256)] = {"name": model.name, "sha256": model.sha256}
    ran = options.detection if options.detection != "fast" and heavy_modules(modules) else "fast"
    plate = _plate_record(modules)
    return {
        "faces": options.faces,
        "plates": options.plates,
        "thoroughness": options.thoroughness,
        "detection": options.detection,
        "detection_ran": ran,
        "heavy_only": options.heavy_only and ran == "deep",
        "unsure": options.unsure,
        "max_zones": options.max_zones,
        **plate,
        "models": [models[k] for k in sorted(models)],
    }


def _plate_record(modules: Sequence[ClassModule]) -> dict:
    """How plates were found by the modules in use, read from the modules themselves (so modules given by the caller are described as
    they are, not as the options ask): `plate_path` "text" or "yolo", `vehicle_model` "rfdetr" or "yolox", `plate_band` whether the
    band was on. All None when no plate module is in use. `bands_added == 0` can then tell "off" from "nothing banded"."""
    for m in modules:
        if getattr(m, "kind", None) != PLATE:
            continue
        text_path = getattr(m, "text_path", None)
        vehicles_name = str(getattr(getattr(m, "vehicles", None), "name", ""))
        return {
            "plate_path": "text" if text_path is not None else "yolo",
            "vehicle_model": "rfdetr" if vehicles_name.startswith("rf-detr") else "yolox",
            "plate_band": bool(getattr(text_path, "band", False)),
        }
    return {"plate_path": None, "vehicle_model": None, "plate_band": None}


def _progress(emit, clip: str, phase: str):
    last = [-1.0]

    def report(
        fraction: float,
        frame: int,
        frames: int,
        view: Optional[dict] = None,
        stage: Optional[str] = None,
        zones: Optional[dict] = None,
    ) -> None:
        if fraction > last[0]:  # monotonic per phase
            last[0] = fraction
            live = found = None
            if view:
                live = [
                    {
                        "kind": kind,
                        "box": [round(v, 1) for v in box],
                        "score": round(score, 3),
                        "state": "found" if sure else "unsure",
                    }
                    for kind, box, score, sure in view["boxes"]
                ]
                found = dict(view["found"])
            emit(Progress(clip, phase, round(fraction, 4), frame, frames, live, found, stage, zones))

    return report


def _report(a: Analysis, clip: str) -> Report:
    fps = a.fps or 1.0
    rows = []
    for i in a.items:
        frames = i.last_frame - i.first_frame + 1
        first_s, last_s = (
            (round(i.first_frame / fps, 3), round(i.last_frame / fps, 3)) if a.media == "video" else (0.0, 0.0)
        )
        decision = "blurred" if i.blur else "left visible"
        shaped = i.params.get("plate_like")
        rows.append(
            ReportRow(
                clip,
                i.item_id,
                i.kind,
                i.verdict,
                i.blur,
                i.reason,
                first_s,
                last_s,
                frames,
                i.max_score,
                i.found_by,
                decision,
                i.decided_by,
                None if shaped is None else bool(shaped),
            )
        )
    return Report(rows, dict(a.options), dict(a.second))


def compare_video(
    original: str,
    anonymized: str,
    target: str,
    captions: Tuple[str, str] = ("Original", "Anonymized"),
    on_event: Optional[EventHandler] = None,
    cancel: Optional[CancelToken] = None,
) -> str:
    """Write a video with the original on the left and the anonymised video on the right, for checking.

    **It shows the original, unblurred: never publish it.** The output is as long as the original, has its audio, and
    carries a caption over each side. The two videos must have the same size and length. `Progress` events have the phase
    `"compare"`.

    Args:
        original: The original video.
        anonymized: The anonymised video made from it.
        target: Where to write the comparison video. It must not exist.
        captions: Two non-empty ASCII texts, one for each side.
        on_event: A function that takes an `Event`.
        cancel: A `CancelToken`.

    Returns:
        `target`.

    Raises:
        UnsupportedMedia: Either input is not a video.
        AnonymizerError: `error.output.check_failed`: the two videos differ in size or length.
        ValueError: The captions are not two non-empty ASCII texts.
        OutputExists: `target` exists.
        FileNotFoundError: The folder of `target` does not exist.
        Cancelled: The job was cancelled.

    See also:
        `anonymize_video`.
    """
    if media_of(original) != "video" or media_of(anonymized) != "video":
        raise UnsupportedMedia(params={"name": os.path.basename(original)}, detail="not a video")
    if len(captions) != 2 or not all(c.isascii() and c.strip() for c in captions):
        raise ValueError("captions must be two non-empty ASCII texts")
    if not os.path.exists(target):
        _destination_folder_exists(target)
    emit = on_event or (lambda e: None)
    clip = os.path.basename(original)
    progress = _progress(emit, clip, "compare")
    try:
        frames = write_comparison(original, anonymized, target, captions, progress, _flag(cancel))
        progress(1.0, frames - 1, frames)
    except events.Cancelled as e:
        emit(Finished(clip, "stopped"))
        raise Cancelled() from e
    except events.ClipError as e:
        raise _api_error(e) from e
    return target


def anonymize_video(
    source: str,
    target: str,
    options: Optional[Options] = None,
    on_event: Optional[EventHandler] = None,
    cancel: Optional[CancelToken] = None,
) -> Result:
    """Analyse and render a video in one call, with the automatic decisions. Nothing is reviewed.

    It checks that `source` is a video and that the folder of `target` exists, then does what
    `with Anonymizer(options) as anon: anon.render(anon.analyze(source), target)` does. It loads the models at every call:
    for many files, use one `Anonymizer`. With the default options the doubtful items that are not placed stay visible:
    read `Result.visible_check`.

    Args:
        source: The original video.
        target: Where to write the output, an MP4. It must not exist, and its folder must.
        options: The `Options`. None means the defaults.
        on_event: A function that takes an `Event`; it gets the events of the analysis, then those of the render.
        cancel: A `CancelToken`.

    Returns:
        The `Result`.

    Raises:
        UnsupportedMedia: `source` is not a video, or it cannot be opened.
        FileNotFoundError: The folder of `target` does not exist.
        NotADirectoryError: The folder of `target` is a file.
        OutputExists: `target` exists. It is found after the analysis: test `os.path.exists(target)` first.
        SourceChanged, IncompleteAnalysis, ModelLoadError, Cancelled: As for `Anonymizer.analyze` and `Anonymizer.render`.

    See also:
        `anonymize_image`, `Anonymizer`.
    """
    if media_of(source) != "video":
        raise UnsupportedMedia(params={"name": os.path.basename(source)}, detail="not a video")
    _destination_folder_exists(target)  # before the analysis, which can take minutes
    with Anonymizer(options) as anon:
        return anon.render(anon.analyze(source, on_event, cancel), target, on_event, cancel)


def anonymize_image(
    source: str, target: str, options: Optional[Options] = None, on_event: Optional[EventHandler] = None
) -> Result:
    """Analyse and render a photo in one call, with the automatic decisions. Nothing is reviewed.

    It checks that `source` is a photo and that the folder of `target` exists, then does what
    `with Anonymizer(options) as anon: anon.render(anon.analyze(source), target)` does. It loads the models at every call:
    for many files, use one `Anonymizer`. The photo keeps the **format of the original**, so give `target` its extension.
    With the default options the doubtful items that are not placed stay visible: read `Result.visible_check`. It takes no
    `CancelToken`.

    Args:
        source: The original photo.
        target: Where to write the output. It must not exist, and its folder must.
        options: The `Options`. None means the defaults.
        on_event: A function that takes an `Event`.

    Returns:
        The `Result`.

    Raises:
        UnsupportedMedia: `source` is not a photo, or it cannot be opened.
        FileNotFoundError: The folder of `target` does not exist.
        NotADirectoryError: The folder of `target` is a file.
        OutputExists: `target` exists. It is found after the analysis: test `os.path.exists(target)` first.
        ModelLoadError, IncompleteAnalysis, AnonymizerError: As for `Anonymizer.analyze` and `Anonymizer.render`.

    Example:
        ```python
        import os

        import anonymizer

        os.makedirs("out", exist_ok=True)
        result = anonymizer.anonymize_image("demo/street-westminster.jpg", "out/westminster-oneshot.jpg")
        print(result.output, result.visible_check, result.warnings)  # out/westminster-oneshot.jpg 4 ['warning.review.check_left_visible']
        ```

    See also:
        `anonymize_video`, `Anonymizer`, `Result`.
    """
    if media_of(source) != "photo":
        raise UnsupportedMedia(params={"name": os.path.basename(source)}, detail="not a photo")
    _destination_folder_exists(target)  # before the analysis, which can take minutes
    with Anonymizer(options) as anon:
        return anon.render(anon.analyze(source, on_event), target, on_event)
