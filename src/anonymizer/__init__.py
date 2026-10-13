"""Anonymizer Studio: offline face and licence-plate blurring for photos and video.

The Python API of Anonymizer Studio. `import anonymizer` gives the names below, and only these are public: everything else
in the package (`anonymizer.engine`, `anonymizer.contract`, `anonymizer.classes`, `anonymizer.ui`) is internal and may
change. The package imports eagerly, loads no model until the first analysis, and makes no network connection. It sets
the environment variable `OPENCV_LOG_LEVEL` to `ERROR` unless you have set it.

For one file, one call:

    import anonymizer
    anonymizer.anonymize_video("in.mp4", "out.mp4")

For several files, or to review what was found before anything is written, use an `Anonymizer`: `analyze` returns an
`Analysis` whose items you can decide on, and `render` writes the output.

Settings and control:
    `Options` (what to look for and how to hide it), `CancelToken` (stop, pause and resume a job from another thread).

Doing the work:
    `Anonymizer` (loads the detectors once; `analyze`, `render`, `estimate`), `anonymize_image` and `anonymize_video`
    (analyse and render in one call), `compare_video` (a side-by-side video for checking; it shows the original).

What it returns:
    `Analysis` (the items and your decisions; `save` and `load_analysis`), `Item`, `BoxInfo`, `Estimate`, `Result`,
    `Report` and `ReportRow`.

Looking at a file:
    `media_info` and `MediaInfo`, `frame_jpeg`, `thumbnail_jpeg`.

Events, for progress and warnings:
    `Event` (the union of the seven), `Started`, `Progress`, `ItemFound`, `Analyzed`, `Rendered`, `JobWarning`,
    `Finished`, and `event_json` (an event as a JSON-ready dict).

Errors:
    `AnonymizerError` is the base, with the subclasses `UnsupportedMedia`, `ModelLoadError`, `OutputExists`,
    `SourceChanged`, `Cancelled` and `IncompleteAnalysis`. Some errors are builtins that the API lets through, such as
    `FileNotFoundError` and `ValueError`; `AnonymizerError` lists them.

Constants:
    `PHOTO_EXTENSIONS`, `VIDEO_EXTENSIONS` and `UNSUPPORTED_EXTENSIONS`: the extensions by which the kind of a file is told.

`__version__` is the version string, `"2.2.0"`.
"""

from . import _quiet  # first: it sets OpenCV's log level before anything imports OpenCV
from .api import (
    PHOTO_EXTENSIONS,
    UNSUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    Analysis,
    Analyzed,
    Anonymizer,
    AnonymizerError,
    BoxInfo,
    Cancelled,
    CancelToken,
    Estimate,
    Event,
    Finished,
    IncompleteAnalysis,
    Item,
    ItemFound,
    JobWarning,
    MediaInfo,
    ModelLoadError,
    Options,
    OutputExists,
    Progress,
    Rendered,
    Report,
    ReportRow,
    Result,
    SourceChanged,
    Started,
    UnsupportedMedia,
    anonymize_image,
    anonymize_video,
    compare_video,
    event_json,
    frame_jpeg,
    load_analysis,
    media_info,
    thumbnail_jpeg,
)

__version__ = "2.2.0"
_OPENCV_LOG_LEVEL = _quiet.LEVEL  # kept so that the import above is used (it must stay first)

__all__ = [
    "Analysis",
    "Analyzed",
    "Anonymizer",
    "AnonymizerError",
    "BoxInfo",
    "CancelToken",
    "Cancelled",
    "Estimate",
    "Event",
    "Finished",
    "IncompleteAnalysis",
    "Item",
    "ItemFound",
    "MediaInfo",
    "ModelLoadError",
    "Options",
    "OutputExists",
    "Progress",
    "Rendered",
    "Report",
    "ReportRow",
    "Result",
    "SourceChanged",
    "Started",
    "UnsupportedMedia",
    "PHOTO_EXTENSIONS",
    "UNSUPPORTED_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "JobWarning",
    "anonymize_image",
    "anonymize_video",
    "compare_video",
    "event_json",
    "frame_jpeg",
    "load_analysis",
    "media_info",
    "thumbnail_jpeg",
]
