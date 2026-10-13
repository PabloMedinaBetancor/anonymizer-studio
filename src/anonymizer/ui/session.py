"""Everything the Studio page can do, as plain calls: clips, analysis jobs, Review decisions, previews and rendering.

The HTTP layer (server.py) only decodes requests and encodes answers. This module talks to the engine through the public
`anonymizer` package and nothing else, and reports through the events hub. One job runs at a time, in its own thread.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import re
import shutil
import string
import tempfile
import threading
import time
import traceback
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anonymizer

from . import browse, export_size, intake, pdf
from .engine import EngineFailure, LocalEngine, ProcessEngine
from .hub import Hub
from .settings import InvalidSetting, Settings, folder_path

DEMO_LIMITS = {"max_photos": 10, "max_clips": 10, "max_seconds": 20, "max_height": 720, "max_upload_mb": 100}
PROGRESS_INTERVAL = 0.25  # at most four progress events a second
OUTPUT_FOLDER = "anonymized"
# File systems take 255 bytes of name. The engine writes a render under a hidden name of ".<name>.anonymizer-partial" before it renames it,
# so the longest name that can be written is shorter by what that adds.
NAME_LIMIT = 255
PARTIAL_EXTRA = len(".") + len(".anonymizer-partial")
MAX_OUTPUT_NAME = NAME_LIMIT - PARTIAL_EXTRA
DEFAULT_PATTERN = "{name}_anonymized"
ANALYZE_FIELDS = ("faces", "plates", "thoroughness", "detection", "gap_fill")
PREVIEW_VIEWS = ("original", "anonymized", "boxes", "split")
MAX_BOX_FRAMES = 600
STATES_WITH_ANALYSIS = ("analyzed", "rendering", "rendered")
SECOND_LOOK_SKIPPED = "warning.second_look.zones_skipped"  # the engine's warning for what its budget left out
VEHICLES_SKIPPED = (
    "warning.vehicles.skipped"  # what the page says of it when nothing was looked at a second time (Fast)
)


class ApiError(Exception):
    """A refused request: an HTTP status plus an i18n code, as the protocol's error body."""

    def __init__(self, status: int, code: str, params: Optional[dict] = None, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.status = status
        self.code = code
        self.params = dict(params or {})
        self.detail = detail


PATTERN_FIELDS = {"name": re.compile(r"^$"), "index": re.compile(r"^(0?\d{0,2}d)?$")}


def has_control(text: str) -> bool:
    """Whether `text` holds a control character (a NUL cannot be in a path, and none of them belongs in a name)."""
    return any(ord(c) < 32 for c in text)


def pattern_error(pattern: str) -> Optional[str]:
    """Why a file name pattern is not allowed, or None. Only {name} and {index} (with a number format such as 03d) may appear:
    a replacement field that reaches into attributes or items ({name.__class__}) is refused."""
    try:
        parts = list(string.Formatter().parse(pattern))
    except ValueError as e:
        return str(e)
    for _, field_name, spec, conversion in parts:
        if field_name is None:
            continue
        if field_name not in PATTERN_FIELDS or conversion or not PATTERN_FIELDS[field_name].match(spec or ""):
            return f"{{{field_name}}} is not allowed"
    return None


def remove_leftover_partial(target: str) -> None:
    """Delete the hidden file that a render to `target` writes before it renames it (".<name>.anonymizer-partial"). Called when
    the render has ended: a finished render has already renamed it, a stopped one has removed it, and one whose process was killed
    has left it behind. It holds only the frames written so far, never under the final name."""
    folder, name = os.path.split(target)
    try:
        os.remove(os.path.join(folder, f".{name}.anonymizer-partial"))
    except OSError:
        pass  # nothing left, or not ours to remove


def rate_key(options: "anonymizer.Options") -> str:
    """What decides how long a frame takes: the thoroughness and the detection tier ("standard/careful")."""
    return f"{options.thoroughness}/{options.detection}"


def invalid(field_name: str, detail: str = "") -> ApiError:
    return ApiError(400, "error.request.invalid", {"field": field_name}, detail)


def only_keys(body: dict, allowed: Sequence[str], where: str) -> None:
    """A key the server does not read is a page-server mismatch that would otherwise do nothing and say nothing."""
    for key in body:
        if key not in allowed:
            raise invalid(f"{where}{key}", "unknown key")


RENDER_OPTION_KEYS = (
    "name_pattern",
    "destination",
    "reports",
    "comparison",
    "blur",
    "strength",
    "extra_margin",
    "remove_gps",
    "gap_fill",
    "language",
    "paper",
)


@dataclass
class Clip:
    clip_id: str
    name: str
    path: str
    origin: str  # "path" or "upload"
    info: "anonymizer.MediaInfo"
    stamp: Tuple[int, int]  # size and mtime of the source when imported
    status: str = "imported"
    analysis: Optional["anonymizer.Analysis"] = None
    analysis_key: Optional[tuple] = None
    rev: int = 0
    reviewed: set = field(default_factory=set)
    thumb: Optional[bytes] = None
    area_boxes: Dict[int, tuple] = field(
        default_factory=dict
    )  # item id of an area the person added -> its box, for its thumbnail
    area_thumbs: Dict[int, bytes] = field(default_factory=dict)
    seconds_per_frame: Dict[str, float] = field(default_factory=dict)  # by rate_key(options), as measured here
    # What a window opened late (or reloaded) in the middle of a job needs to show the clip as it is now, not as new: the last progress
    # event of its analysis, the items found so far, and why a clip failed. Cleared when the clip is analysed again.
    progress: Optional[dict] = None
    found_items: List[dict] = field(default_factory=list)
    failure: Optional[dict] = None
    # Where the export of the clip is, for a window reloaded in the middle of it: {"phase": "queued" | "render" | "compare", "fraction"} while
    # it runs, {"phase": "done", "outputs", "report", "comparison", "seconds"} once written. Cleared when the clip is analysed again.
    render: Optional[dict] = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    saver: Optional[threading.Timer] = None


@dataclass
class Job:
    job_id: str
    kind: str
    clips: List[Clip]
    token: "anonymizer.CancelToken" = field(default_factory=anonymizer.CancelToken)
    state: str = "running"
    current: Optional[str] = None


def stamp_of(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


def file_identity(path: str) -> Optional[Tuple[int, int, int, int]]:
    """Which file a name leads to (device, inode, size and time, links followed), or None when it cannot be told (a file system with no
    inode numbers, a file gone). The size and the time guard against a file system that gives every file one inode number: it must not
    make a whole folder one clip."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) if st.st_ino else None


WORK_PREFIX = "anonymizer-studio-"
OWNER_FILE = ".studio-pid"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError: it exists, and is somebody else's
        return True
    return True


def remove_stale_work_folders(root: Optional[str] = None) -> List[str]:
    """Delete the work folders of Studios that are gone (killed, or the machine lost power): they hold the person's pictures'
    copies of what was found (face and plate crops, analyses). Only a folder this account owns, whose name starts with the Studio's
    prefix and that names the process that made it, and only when that process is no longer running. Returns what it removed."""
    root = root or tempfile.gettempdir()
    removed: List[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return removed
    for name in names:
        folder = os.path.join(root, name)
        if not name.startswith(WORK_PREFIX) or os.path.islink(folder):
            continue
        try:
            if os.stat(folder).st_uid != os.getuid():
                continue
            with open(os.path.join(folder, OWNER_FILE), encoding="ascii") as f:
                pid = int(f.read().strip())
        except (
            OSError,
            ValueError,
            AttributeError,
        ):  # no marker (an older Studio's folder, or not ours at all): left alone
            continue
        if pid != os.getpid() and not _alive(pid):
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder)
    return removed


def demo_scrub(data: dict) -> dict:
    """What the hosted demo says to its visitors about a failure: the reason and the names of their files, never the technical text of the
    server (tracebacks, its folders, its account) and never a folder of the server in a parameter."""
    data = dict(data)
    if data.get("detail"):
        data["detail"] = ""
    params = data.get("params")
    if isinstance(params, dict) and isinstance(params.get("path"), str):
        data["params"] = {**params, "path": os.path.basename(params["path"].rstrip("/\\"))}
    return data


MAX_UPLOAD_NAME = 120  # bytes: a long name plus the suffixes the Studio adds (-2, _anonymized, a hidden working name) must fit a file system's 255


def shortened(name: str, limit: int = MAX_UPLOAD_NAME) -> str:
    """`name` cut to `limit` bytes in its stem (the extension and whole characters stay)."""
    if len(os.fsencode(name)) <= limit:
        return name
    stem, ext = os.path.splitext(name)
    room = max(1, limit - len(os.fsencode(ext)))
    while stem and len(os.fsencode(stem)) > room:
        stem = stem[:-1]
    return (stem or "upload") + ext


def same_name_key(path: str) -> str:
    """What two paths must differ in to be two files on every file system: Windows, macOS and the USB disks everyone copies to
    (exFAT, FAT, NTFS) do not tell "IMG.JPG" from "img.jpg", and macOS does not tell an accented letter written as one character from the letter followed by its accent."""
    return unicodedata.normalize("NFC", path).casefold()


def report_path(target: str, kind: str) -> str:
    """Where a report of the file written at `target` goes: `<name>.report.<kind>` next to it. Two sources that differ only in their
    extension (`x.bmp`, `x.tif`) come out as `x_anonymized.bmp` and `x_anonymized.tif` but share that name, so the later report
    keeps the media's extension (`x_anonymized.tif.report.csv`) and is not told from the first only by a `-2`."""
    plain = f"{os.path.splitext(target)[0]}.report.{kind}"
    if free_name(plain) == plain:
        return plain
    return free_name(f"{target}.report.{kind}")


def free_name(path: str, taken: Sequence[str] = (), listings: Optional[Dict[str, set]] = None) -> str:
    """`path`, or with -2, -3... before the extension until it names nothing that exists or is already taken (the case of the
    letters does not make a name different: a second photo called "img_1" next to "IMG_1" would replace its output on a disk that
    ignores case). `listings` keeps what a folder holds between calls, for a plan of hundreds of names in one folder."""
    stem, ext = os.path.splitext(path)
    folder = os.path.dirname(path)
    if listings is not None and folder in listings:
        beside = listings[folder]
    else:
        try:
            beside = {same_name_key(os.path.join(folder, name)) for name in os.listdir(folder or ".")}
        except OSError:
            beside = set()
        if listings is not None:
            listings[folder] = beside
    used = {same_name_key(t) for t in taken}
    candidate, n = path, 1
    while os.path.exists(candidate) or same_name_key(candidate) in beside or same_name_key(candidate) in used:
        n += 1
        candidate = f"{stem}-{n}{ext}"
    return candidate


def nearest_existing(path: str) -> str:
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def name_order(name: str) -> Tuple[str, str]:
    """The order of names in a folder import: as a file manager lists them (case ignored: `madrid` before `Taipei`), the exact name
    breaking a tie. The folder dialog already lists them so."""
    return (name.casefold(), name)


def can_create_in(folder: str) -> bool:
    """Whether `folder` exists and can be written, or can be created below a folder that can."""
    anchor = nearest_existing(folder)
    if not (os.path.isdir(anchor) and os.access(anchor, os.W_OK)):
        return False
    # what is yet to be created must be a name a file system takes: a folder of 300 letters was "writable" in the plan, and the render then
    # failed with "not writable" after it had started
    below = os.path.relpath(os.path.abspath(folder), anchor)
    return (
        all(len(os.fsencode(part)) <= 255 for part in below.split(os.sep))
        and len(os.fsencode(os.path.abspath(folder))) < 4000
    )


class Session:
    def __init__(
        self,
        settings: Settings,
        hub: Hub,
        *,
        demo: bool = False,
        work_dir: Optional[str] = None,
        modules: Optional[list] = None,
        models_dir: Optional[str] = None,
        demo_dir: Optional[str] = None,
        languages: Sequence[str] = ("en",),
        find_browser=pdf.find_browser,
        texts_for=lambda language: {},
        engine: str = "thread",
        engine_options: Optional[dict] = None,
    ):
        self.settings = settings
        self.hub = hub
        self.demo = demo
        if demo:
            hub.scrub = demo_scrub
        self.modules = modules
        self.models_dir = models_dir
        self.demo_dir = demo_dir
        self.languages = list(languages)
        self.find_browser = find_browser
        self.texts_for = texts_for
        self.engine_kind = engine
        self.engine_options = dict(engine_options or {})
        self._own_work = work_dir is None
        self.work_dir = work_dir or tempfile.mkdtemp(prefix=WORK_PREFIX)
        os.makedirs(self.work_dir, exist_ok=True)
        if self._own_work:
            try:
                with open(os.path.join(self.work_dir, OWNER_FILE), "w", encoding="ascii") as f:
                    f.write(
                        str(os.getpid())
                    )  # who the folder belongs to: a later Studio removes it if this one is killed
            except OSError:
                pass
        self.lock = threading.RLock()
        self._upload_lock = threading.Lock()
        self._import_lock = (
            threading.Lock()
        )  # two imports of one folder at once (a double drop, two windows) must not both add its files
        self.clips: Dict[str, Clip] = {}
        self._counter = 0
        # The version counter of a clip starts at the moment this server started (microseconds, below 2**53 so the page's numbers
        # hold it): a restarted server on a fixed port gives c1 a version no earlier run used, and the browser's cached thumbnails and
        # 304s of the old c1 do not match (review 2, N7).
        self._rev_base = time.time_ns() // 1000
        self._jobs = 0
        self.job: Optional[Job] = None
        self._thread: Optional[threading.Thread] = None
        self.frame_seconds: Dict[str, float] = {}  # rate_key(options) -> seconds per frame measured on this machine
        self.downloads: Dict[str, Tuple[str, str]] = {}
        self.last_seen = time.monotonic()

    def close(self) -> None:
        job = self.job
        if job is not None:
            job.token.cancel()
            if self._thread is not None:
                self._thread.join(10)
        for clip in self.clips.values():
            if clip.saver is not None:
                clip.saver.cancel()
        if self._own_work:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def capabilities(self) -> dict:
        found = self.find_browser() is not None
        return {
            "pdf_report": {"available": found, "reason": None if found else "report.pdf.no_browser", "params": {}},
            "comparison_video": {"available": True},
            "browse": not self.demo,
            "upload": True,
        }

    def summaries(self) -> List[dict]:
        """The summary of every clip, taken while no import or removal is changing the list (an unlocked walk over it was "dictionary
        changed size during iteration", a 500, for a page that asked while a second window imported)."""
        with self.lock:
            return [self.summary(c) for c in self.clips.values()]

    def state(self) -> dict:
        with self.lock:
            clips = [self.summary(c) for c in self.clips.values()]
            job = self.job
            job_json = (
                None
                if job is None
                else {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "state": job.state,
                    "clips": [c.clip_id for c in job.clips],
                    "current": job.current,
                }
            )
        return {
            "v": 1,
            "version": anonymizer.__version__,
            "mode": "demo" if self.demo else "local",
            "languages": self.languages,
            "formats": self.formats(),
            "capabilities": self.capabilities(),
            "demo_limits": dict(DEMO_LIMITS) if self.demo else None,
            "slowdown": self.slowdowns(),
            "settings": self.public_settings(),
            "clips": clips,
            "job": job_json,
            "seq": self.hub.seq,
        }

    @staticmethod
    def formats() -> dict:
        """What the door takes, for a page that must not send a file only to have it refused (a folder dropped on Import): the extensions of
        photos and videos, those of the formats known not to be supported with the reason of each, and the folder the Studio writes into."""
        return {
            "photo": sorted(anonymizer.PHOTO_EXTENSIONS),
            "video": sorted(anonymizer.VIDEO_EXTENSIONS),
            "unsupported": {
                ext: intake.unsupported_code(f"x{ext}") for ext in sorted(anonymizer.UNSUPPORTED_EXTENSIONS)
            },
            "output_folder": OUTPUT_FOLDER,
        }

    def public_settings(self, values: Optional[dict] = None) -> dict:
        """The settings as a page is told them. In the hosted demo the folder a local Studio saves to (the server account's home) is not told."""
        values = dict(self.settings.get() if values is None else values)
        if self.demo and "default_destination" in values:
            values["default_destination"] = ""
        return values

    def slowdowns(self) -> dict:
        """How many times slower "careful" ran than "fast" detection on a clip analysed both ways in this session. None while no clip
        has been analysed both ways (a number nobody measured is never shown)."""
        pairs = {"careful": ("standard/fast", "standard/careful")}
        found: Dict[str, Optional[float]] = {name: None for name in pairs}
        for clip in list(self.clips.values()):
            for name, (base, other) in pairs.items():
                fast, slow = clip.seconds_per_frame.get(base), clip.seconds_per_frame.get(other)
                if found[name] is None and fast and slow:
                    found[name] = round(slow / fast, 1)
        return found

    def clip(self, clip_id: str) -> Clip:
        try:
            return self.clips[clip_id]
        except KeyError:
            raise ApiError(404, "error.not_found", {"clip_id": clip_id}) from None

    def summary(self, clip: Clip) -> dict:
        info = clip.info
        base = f"/api/clips/{clip.clip_id}"
        return {
            "clip_id": clip.clip_id,
            "name": clip.name,
            "media": info.media,
            "origin": clip.origin,
            "source_dir": os.path.dirname(clip.path) if clip.origin == "path" else None,
            "duration_s": info.duration_s,
            "frames": info.frames,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "codec": info.codec,
            "has_audio": info.has_audio,
            "has_gps": info.has_gps,
            "size_bytes": info.size_bytes,
            # `v` is this server's start: a restart on the same port gives c1 to another file, and the browser kept the old c1's picture for an hour
            "thumb_url": f"{base}/thumb?v={self._rev_base}",
            "source_url": f"{base}/source",
            "status": clip.status,
            "analysis": self.analysis_summary(clip),
            **self._in_flight(clip),
        }

    def _in_flight(self, clip: Clip) -> dict:
        """For a clip that is being analysed, how far it is and what it has found; for one that failed, why (the page of a window that was
        reloaded in the middle of the job shows these, not a clip at 0 % or \"Failed\" with no reason)."""
        if clip.status == "analyzing":
            return {"progress": clip.progress, "items_so_far": list(clip.found_items)}
        if clip.status == "failed" and clip.failure is not None:
            failure = clip.failure
            if self.demo:  # a visitor is told the name of the file that is gone, not the folder of the server it was in
                failure = {**failure, "params": demo_scrub({"params": failure.get("params")})["params"]}
            return {"failure": failure}
        if clip.render is not None:
            return {"render": clip.render}
        return {}

    def analysis_summary(self, clip: Clip) -> Optional[dict]:
        a = clip.analysis
        if a is None:
            return None
        try:
            unchanged = stamp_of(clip.path) == clip.stamp
        except OSError:
            unchanged = False
        return {
            "items": len(a.items),
            "to_check": self.to_check(clip),
            "unchecked_frames": len(a.unchecked_frames),
            "rev": clip.rev,
            # the analysis options this analysis was made with, so that the page counts only the clips that "Analyze" would do again
            # (None when the file changed since: the next Analyze does it again)
            "settings": dict(zip(ANALYZE_FIELDS, clip.analysis_key, strict=True))
            if clip.analysis_key and unchanged
            else None,
        }

    @staticmethod
    def to_check(clip: Clip) -> int:
        return sum(1 for i in clip.analysis.items if i.verdict == "check" and i.item_id not in clip.reviewed)

    def item_json(self, clip: Clip, item: "anonymizer.Item") -> dict:
        video = clip.info.media == "video"
        fps = clip.info.fps or 1.0
        manual = item.kind == "manual"
        return {
            "item_id": item.item_id,
            "kind": item.kind,
            "judgement": {"verdict": item.verdict, "reason": item.reason, "params": item.params},
            "first_frame": item.first_frame,
            "last_frame": item.last_frame,
            "first_s": round(item.first_frame / fps, 3) if video else 0.0,
            "last_s": round(item.last_frame / fps, 3) if video else 0.0,
            "frames": item.last_frame - item.first_frame + 1,
            "max_score": item.max_score,
            "blur": item.blur,
            "found_by": item.found_by,
            "second_score": item.second_score,
            "decided_by": item.decided_by,
            "reviewed": manual or item.item_id in clip.reviewed,
            "manual": manual,
            "thumb_url": (
                None
                if manual and item.item_id not in clip.area_boxes
                else f"/api/clips/{clip.clip_id}/items/{item.item_id}/thumb?rev={clip.rev}"
            ),
        }

    def _new_clip(self, path: str, origin: str) -> Clip:
        try:
            info = anonymizer.media_info(path)
        except anonymizer.UnsupportedMedia as e:  # the header cannot be read: say why, in the words of the file's kind
            raise self._unreadable(path, e) from e
        problem = intake.readable_error(path, info.media)
        if problem is not None:
            raise problem
        self._check_demo_limits(info)
        with self.lock:
            self._counter += 1
            clip = Clip(
                f"c{self._counter}", os.path.basename(path), path, origin, info, stamp_of(path), rev=self._rev_base
            )
            self.clips[clip.clip_id] = clip
        return clip

    def _check_demo_limits(self, info: "anonymizer.MediaInfo") -> None:
        if not self.demo:
            return
        with self.lock:
            photos = sum(1 for c in self.clips.values() if c.info.media == "photo")
            clips = len(self.clips)
        if (
            clips >= DEMO_LIMITS["max_clips"]
        ):  # videos too: the photo limit alone left any number of videos (each up to 100 MB) to fill the disk
            raise ApiError(422, "error.demo.limit", {"limit": "clips", "max": DEMO_LIMITS["max_clips"]})
        if info.media == "photo" and photos >= DEMO_LIMITS["max_photos"]:
            raise ApiError(422, "error.demo.limit", {"limit": "photos", "max": DEMO_LIMITS["max_photos"]})
        if info.media == "video" and info.duration_s > DEMO_LIMITS["max_seconds"] + 0.5:
            raise ApiError(422, "error.demo.limit", {"limit": "seconds", "max": DEMO_LIMITS["max_seconds"]})
        short, long = sorted((info.width, info.height))
        if short > DEMO_LIMITS["max_height"] or long > DEMO_LIMITS["max_height"] * 16 // 9 + 1:
            raise ApiError(422, "error.demo.limit", {"limit": "height", "max": DEMO_LIMITS["max_height"]})

    def _require_idle(self) -> None:
        if self.job is not None:
            raise ApiError(409, "error.job.busy")

    @staticmethod
    def _unreadable(path: str, error: "anonymizer.UnsupportedMedia") -> "anonymizer.UnsupportedMedia":
        """What media_info could not open, as the reason a person reads: nothing in it, or a photo or a video that is broken."""
        if error.code != "error.media.unsupported":
            return error
        try:
            if os.path.getsize(path) == 0:
                return anonymizer.UnsupportedMedia("error.media.empty", {"name": os.path.basename(path)})
        except OSError:
            pass
        media = "video" if browse.kind_of(path) == "video" else "photo"
        return intake.too_large_error(os.path.basename(path), error.detail) or intake.unreadable(
            path, media, error.detail
        )

    @staticmethod
    def _rejection(path: str, error: "anonymizer.AnonymizerError") -> dict:
        return {"path": path, "code": error.code, "params": error.params, "detail": error.detail}

    def import_paths(self, paths: Sequence[str], recursive: bool = True) -> dict:
        if self.demo:
            raise ApiError(403, "error.demo.disabled")
        self._require_idle()
        files, rejected, ignored = self._expand(paths, recursive)
        clips = []
        added = []
        with (
            self._import_lock
        ):  # the check for a file that is already a clip and the adding of the new ones are one step
            known: Dict[
                Tuple[int, int, int, int], Clip
            ] = {}  # file identity -> clip: two links (or two names) of one file are one clip, not two
            for c in list(self.clips.values()):
                if c.origin == "path" and (identity := file_identity(c.path)) is not None:
                    known.setdefault(identity, c)
            for path in files:
                identity = file_identity(path)
                existing = known.get(identity) if identity is not None else None
                if existing is None:
                    existing = next(
                        (c for c in list(self.clips.values()) if c.origin == "path" and c.path == path), None
                    )
                if existing is not None:
                    if existing not in clips:
                        clips.append(existing)
                    continue
                try:
                    clips.append(self._new_clip(path, "path"))
                    added.append(clips[-1])
                    if identity is not None:
                        known[identity] = clips[-1]
                except anonymizer.AnonymizerError as e:
                    rejected.append(self._rejection(path, e))
                except (
                    Exception
                ) as e:  # a hostile or enormous file must not cost the others: it is listed, with what happened to it
                    media = "video" if browse.kind_of(path) == "video" else "photo"
                    rejected.append(self._rejection(path, intake.unreadable(path, media, f"{type(e).__name__}: {e}")))
        if ignored:
            rejected.append(
                {
                    "path": "",
                    "code": "warning.import.ignored",
                    "params": {
                        "count": len(ignored),
                        "extensions": ", ".join(sorted({e or "(none)" for e in ignored.values()})[:8]),
                    },
                    "detail": "; ".join(sorted(ignored)[:20]),
                }
            )
        self._announce_added(added)
        return {"clips": [self.summary(c) for c in clips], "rejected": rejected}

    def _announce_added(self, clips: Sequence[Clip]) -> None:
        """Tell every open page about clips that were just added, so that a second window of the same Studio shows them too."""
        if clips:
            self.hub.emit("clips.added", clips=[self.summary(c) for c in clips])

    def _expand(self, paths: Sequence[str], recursive: bool) -> Tuple[List[str], List[dict], Dict[str, str]]:
        """The files to import, those that are listed as rejected, and {path: extension} of the files in a folder that are neither
        a photo nor a video nor a format known not to be supported: not imported, and said so, never skipped in silence."""
        files: List[str] = []
        rejected: List[dict] = []
        ignored: Dict[str, str] = {}
        hidden = 0
        for raw in paths:
            path = os.path.abspath(os.path.expanduser(raw))
            if os.path.isdir(path):

                def cannot_open(
                    error: OSError, top: str = path
                ) -> None:  # a folder the person may not read: said, not skipped in silence
                    where = error.filename if isinstance(error.filename, str) else top
                    rejected.append(
                        {"path": where, "code": "error.browse.denied", "params": {"path": where}, "detail": ""}
                    )

                seen = {
                    os.path.realpath(path)
                }  # a link to a folder is followed, once: a link back up the tree is not a way round for ever
                for folder, subs, names in os.walk(path, onerror=cannot_open, followlinks=True):
                    hidden += sum(1 for s in subs if s.startswith("."))
                    subs[:] = sorted((s for s in subs if not s.startswith(".") and s != OUTPUT_FOLDER), key=name_order)
                    subs[:] = [s for s in subs if os.path.realpath(os.path.join(folder, s)) not in seen]
                    seen.update(os.path.realpath(os.path.join(folder, s)) for s in subs)
                    for name in sorted(names, key=name_order):
                        kind = browse.kind_of(name)
                        if name.startswith("."):
                            hidden += 1
                            continue
                        full = os.path.join(folder, name)
                        if kind == "other":
                            ignored[full] = os.path.splitext(name)[1].lower()
                        elif kind == "unsupported":
                            rejected.append(self._unsupported(full))
                        elif not os.path.isfile(full):
                            # a pipe, a socket, a device or a link to nothing named like a photo: opening a pipe waits for ever, and the
                            # import (which holds the lock of every import) with it
                            rejected.append(
                                {"path": full, "code": "error.not_found", "params": {"path": full}, "detail": ""}
                            )
                        else:
                            files.append(full)
                    if not recursive:
                        break
            elif os.path.isfile(path):
                if browse.kind_of(path) in ("video", "photo"):
                    files.append(path)
                else:
                    rejected.append(self._unsupported(path))
            else:
                rejected.append({"path": raw, "code": "error.not_found", "params": {"path": raw}, "detail": ""})
        if hidden:  # hidden entries are left out on purpose (editors' and system's files), and said so
            rejected.append({"path": "", "code": "warning.import.hidden", "params": {"count": hidden}, "detail": ""})
        return files, rejected, ignored

    def _unsupported(self, path: str) -> dict:
        return self._rejection(path, intake.unsupported_error(path))

    def upload_target(self, name: str) -> str:
        """Where the bytes of an uploaded file are to be written (a name of our own inside the session folder)."""
        safe = os.path.basename("".join(c for c in name if ord(c) >= 32).replace("\\", "/")).strip() or "upload"
        safe = shortened(safe)
        folder = os.path.join(self.work_dir, "uploads")
        os.makedirs(folder, exist_ok=True)
        with (
            self._upload_lock
        ):  # choosing the name and making the file are one step: two uploads of one name were given the same file
            target = free_name(os.path.join(folder, safe))
            with open(target, "xb"):
                pass
        return target

    def register_upload(self, path: str) -> dict:
        """Add an uploaded file the server has finished writing; it is deleted again when it is not usable."""
        try:
            if browse.kind_of(path) in ("other", "unsupported"):
                raise intake.unsupported_error(path)
            clip = self._new_clip(path, "upload")
        except anonymizer.AnonymizerError as e:
            os.remove(path)
            if isinstance(e, anonymizer.UnsupportedMedia):
                return {"clips": [], "rejected": [self._upload_rejection(path, e)]}
            raise
        except ApiError:
            os.remove(path)
            raise
        except (
            Exception
        ) as e:  # a file the readers choke on in a way nobody foresaw: refused with its reason, and not kept
            os.remove(path)
            media = "video" if browse.kind_of(path) == "video" else "photo"
            error = intake.unreadable(path, media, f"{type(e).__name__}: {e}")
            return {"clips": [], "rejected": [self._upload_rejection(path, error)]}
        self._announce_added([clip])
        return {"clips": [self.summary(clip)], "rejected": []}

    def _upload_rejection(self, path: str, error: "anonymizer.AnonymizerError") -> dict:
        """The refusal of an uploaded file, with the name the person gave it and not the server's folder it was written into (the hosted
        demo is somebody else's computer)."""
        rejection = self._rejection(path, error)
        rejection["path"] = os.path.basename(path)
        rejection["detail"] = rejection["detail"].replace(os.path.dirname(path) + os.sep, "")
        return rejection

    def samples(self) -> List[dict]:
        """The public clips of demo/SOURCES.csv that are on disk, with their licences."""
        if not self.demo_dir:
            return []
        try:
            with open(os.path.join(self.demo_dir, "SOURCES.csv"), encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            return []
        found = []
        for row in rows:
            path = os.path.join(self.demo_dir, row.get("file", ""))
            kind = browse.kind_of(path)
            if kind in ("video", "photo") and os.path.isfile(path):
                try:
                    info = anonymizer.media_info(path)
                except anonymizer.AnonymizerError:
                    continue
                found.append(
                    {
                        "sample_id": os.path.splitext(row["file"])[0].replace("/", "-"),
                        "name": os.path.basename(path),
                        "media": info.media,
                        "duration_s": info.duration_s,
                        "licence": row.get("licence", ""),
                        "author": row.get("author", ""),
                        "source_url": row.get("source_url", ""),
                        "_path": path,
                    }
                )
        return found

    def import_samples(self, sample_ids: Sequence[str]) -> dict:
        self._require_idle()
        by_id = {s["sample_id"]: s for s in self.samples()}
        clips = []
        for sample_id in sample_ids:
            if sample_id not in by_id:
                raise ApiError(404, "error.not_found", {"sample_id": sample_id})
            source = by_id[sample_id]["_path"]
            target = self.upload_target(os.path.basename(source))
            shutil.copyfile(source, target)  # the demo works on a copy, like an upload
            clips.append(self._new_clip(target, "upload"))
        self._announce_added(clips)
        return {"clips": [self.summary(c) for c in clips], "rejected": []}

    def delete_clip(self, clip_id: str) -> None:
        with self.lock:
            clip = self.clip(clip_id)
            if self.job is not None and clip in self.job.clips:
                raise ApiError(409, "error.job.busy")
            del self.clips[clip_id]
        if clip.saver is not None:
            clip.saver.cancel()
        if clip.origin == "upload":
            try:
                os.remove(clip.path)
            except OSError:
                pass  # already gone
        self.hub.emit("clip.removed", clip_id=clip_id)

    def thumb(self, clip_id: str) -> bytes:
        clip = self.clip(clip_id)
        if clip.thumb is None:
            try:
                clip.thumb = anonymizer.thumbnail_jpeg(clip.path, 240)
            except FileNotFoundError:
                raise self.gone(clip) from None
            except anonymizer.UnsupportedMedia:
                if not os.path.exists(clip.path):  # the engine puts a missing file in the words of an unreadable one
                    raise self.gone(clip) from None
                raise
        return clip.thumb

    @staticmethod
    def gone(clip: Clip) -> ApiError:
        """The refusal for a picture of an original that has been moved, renamed or deleted since it was imported."""
        return ApiError(422, "error.source.missing", {"path": clip.path})

    def _analyzed(self, clip_id: str) -> Clip:
        clip = self.clip(clip_id)
        if clip.analysis is None or clip.status not in STATES_WITH_ANALYSIS:
            raise ApiError(409, "error.clip.state", {"clip_id": clip_id, "status": clip.status})
        return clip

    def analysis_json(self, clip_id: str) -> dict:
        clip = self._analyzed(clip_id)
        with clip.lock:
            a = clip.analysis
            return {
                "clip_id": clip_id,
                "rev": clip.rev,
                "frames": a.frames,
                "fps": a.fps,
                "width": a.width,
                "height": a.height,
                "unchecked_frames": len(a.unchecked_frames),
                "to_check": self.to_check(clip),
                "items": [self.item_json(clip, i) for i in a.items],
            }

    def analysis_file(self, clip_id: str) -> str:
        clip = self._analyzed(clip_id)
        with clip.lock:
            return self._save(clip)

    def item_thumb(self, clip_id: str, item_id: int) -> bytes:
        clip = self._analyzed(clip_id)
        with clip.lock:
            try:
                clip.analysis.item(item_id)
            except KeyError:
                raise ApiError(404, "error.not_found", {"item_id": item_id}) from None
            if item_id in clip.area_boxes:
                try:
                    return self._area_thumb(clip, item_id)
                except FileNotFoundError:
                    raise self.gone(clip) from None
            data = clip.analysis.thumbnail_jpeg(item_id)
        if data is None:
            raise ApiError(404, "error.not_found", {"item_id": item_id})
        return data

    @staticmethod
    def _area_thumb(clip: Clip, item_id: int, width: int = 240) -> bytes:
        """The picture inside an area the person added, as the detected items have theirs: the frame the area starts at, cropped to
        its box with a little context around it, at most `width` pixels wide."""
        cached = clip.area_thumbs.get(item_id)
        if cached is not None:
            return cached
        import io

        from PIL import Image

        x0, y0, x1, y1 = clip.area_boxes[item_id]
        item = clip.analysis.item(item_id)
        whole = anonymizer.frame_jpeg(clip.path, item.first_frame, max(clip.info.width, clip.info.height))
        with Image.open(io.BytesIO(whole)) as picture:
            scale = picture.width / max(1, clip.info.width)
            pad = 0.15 * max(x1 - x0, y1 - y0)
            box = (
                max(0, int((x0 - pad) * scale)),
                max(0, int((y0 - pad) * scale)),
                min(picture.width, int((x1 + pad) * scale) + 1),
                min(picture.height, int((y1 + pad) * scale) + 1),
            )
            crop = picture.crop(box).convert("RGB")
        crop.thumbnail((width, width))
        out = io.BytesIO()
        crop.save(out, "JPEG", quality=85)
        clip.area_thumbs[item_id] = out.getvalue()
        return clip.area_thumbs[item_id]

    def preview_tag(self, clip_id: str) -> str:
        """What a preview depends on besides its address: the decisions of the clip and the settings that shape the blur."""
        clip = self.clip(clip_id)
        o = self.settings.options()
        try:
            source = stamp_of(
                clip.path
            )  # a file changed or moved since the analysis is not the picture the browser has kept
        except OSError:
            source = None
        shaping = (o.blur, o.strength, o.extra_margin, o.gap_fill, o.faces, o.plates, source)
        return f"{clip.rev}.{hashlib.sha1(repr(shaping).encode()).hexdigest()[:10]}"

    def preview(self, clip_id: str, frame: int, view: str, width: int) -> bytes:
        if view not in PREVIEW_VIEWS:
            raise invalid("view")
        clip = self.clip(clip_id)
        if not 0 <= frame < clip.info.frames:
            raise invalid("frame")
        if view == "original":
            return self._call(lambda: anonymizer.frame_jpeg(clip.path, frame, width), clip)
        clip = self._analyzed(clip_id)
        with clip.lock:
            return self._call(lambda: clip.analysis.preview(frame, view, width, self.settings.options()), clip)

    def _call(self, function, clip: Clip):
        try:
            return function()
        except anonymizer.SourceChanged as e:
            raise ApiError(422, e.code, e.params) from e
        except FileNotFoundError:
            raise self.gone(clip) from None
        except ValueError as e:
            raise invalid("frame", str(e)) from e

    def boxes(self, clip_id: str, frame_from: int, frame_to: int) -> dict:
        clip = self._analyzed(clip_id)
        if not 0 <= frame_from < clip.info.frames:
            raise invalid("from")
        if not frame_from <= frame_to < clip.info.frames or frame_to - frame_from >= MAX_BOX_FRAMES:
            raise invalid("to")  # it was always "from", even when `to` was the number out of range
        with clip.lock:
            try:
                found = clip.analysis.boxes(frame_from, frame_to, self.settings.options())
            except (
                ValueError
            ) as e:  # frames outside the analysis: a request that cannot be answered, not an internal error
                raise invalid("from", str(e)) from e
            return {
                "rev": clip.rev,
                "frame_from": frame_from,
                "frame_to": frame_to,
                "fps": clip.info.fps,
                "frames": {
                    str(t): [{"item_id": b.item_id, "kind": b.kind, "box": list(b.box), "shape": b.shape} for b in row]
                    for t, row in found.items()
                },
            }

    def report(self, clip_id: str, fmt: str) -> Tuple[str, bytes]:
        clip = self._analyzed(clip_id)
        if fmt not in ("json", "csv"):
            raise invalid("format")
        folder = tempfile.mkdtemp(dir=self.work_dir)
        try:
            with clip.lock:
                report = clip.analysis.report(clip.name)
            path = os.path.join(folder, f"report.{fmt}")
            if fmt == "json":
                report.to_json(path)
            else:
                report.to_csv(path)
            with open(path, "rb") as f:
                return fmt, f.read()
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def _changed(self, clip: Clip) -> None:
        clip.rev += 1
        if clip.status == "rendered":
            clip.status = "analyzed"
        if clip.saver is not None:
            clip.saver.cancel()
        clip.saver = threading.Timer(1.0, self._save_quietly, args=(clip,))
        clip.saver.daemon = True
        clip.saver.start()

    def _save(self, clip: Clip) -> str:
        path = os.path.join(self.work_dir, f"{clip.clip_id}.analysis.json")
        clip.analysis.save(path)
        return path

    def _save_quietly(self, clip: Clip) -> None:
        with clip.lock:
            if clip.analysis is not None and clip.clip_id in self.clips:
                try:
                    self._save(clip)
                except OSError:  # the analysis stays in memory; a full disk will show when exporting
                    self.hub.emit("job.warning", code="warning.analysis.not_saved", clip_id=clip.clip_id, params={})

    def set_item(self, clip_id: str, item_id: int, blur: Any) -> dict:
        if not isinstance(blur, bool):
            raise invalid("blur")
        clip = self._reviewable(clip_id)
        with clip.lock:
            try:
                clip.analysis.set_blur(item_id, blur)
            except KeyError:
                raise ApiError(404, "error.not_found", {"item_id": item_id}) from None
            clip.reviewed.add(item_id)
            self._changed(clip)
            answer = {
                "item": self.item_json(clip, clip.analysis.item(item_id)),
                "rev": clip.rev,
                "to_check": self.to_check(clip),
            }
            self.hub.emit("item.changed", clip_id=clip_id, **answer)
            return answer

    def add_area(self, clip_id: str, body: dict) -> dict:
        only_keys(body, ("frame_from", "frame_to", "box", "follow"), "")
        clip = self._reviewable(clip_id)
        frames = []
        for name in ("frame_from", "frame_to"):
            value = body.get(name)
            try:
                if isinstance(value, bool) or value is None:
                    raise TypeError(name)
                frames.append(int(value))
            except (TypeError, ValueError, OverflowError):
                raise invalid(name, "a whole number of a frame was expected") from None
        frame_from, frame_to = frames
        if not 0 <= frame_from < clip.info.frames:
            raise invalid("frame_from", f"frames run from 0 to {clip.info.frames - 1}")
        if not frame_from <= frame_to < clip.info.frames:
            raise invalid("frame_to", f"frame_to must lie from frame_from to {clip.info.frames - 1}")
        follow = body.get("follow", "camera")
        if follow not in ("camera", "fixed"):
            raise invalid("follow", "follow must be 'camera' or 'fixed'")
        try:
            box = tuple(float(v) for v in body["box"])
            if len(box) != 4:
                raise ValueError("a box has four numbers")
        except (KeyError, TypeError, ValueError, OverflowError) as e:
            raise invalid("box", str(e)) from e
        with clip.lock:
            try:
                item = clip.analysis.add_area(frame_from, frame_to, box, follow)
            except ValueError as e:
                raise invalid("box", str(e)) from e
            clip.reviewed.add(item.item_id)
            clip.area_boxes[item.item_id] = box
            clip.area_thumbs.pop(item.item_id, None)
            self._changed(clip)
            found = clip.analysis.boxes(frame_from, frame_to, self.settings.options())
            area_boxes = {
                str(t): [list(b.box) for b in row if b.item_id == item.item_id]
                for t, row in found.items()
                if any(b.item_id == item.item_id for b in row)
            }
            self.hub.emit("area.added", clip_id=clip_id, item=self.item_json(clip, item), rev=clip.rev)
            return {"item": self.item_json(clip, item), "boxes": area_boxes, "rev": clip.rev}

    def set_doubtful(self, clip_id: str, blur: Any) -> dict:
        """Bulk decision for every doubtful (Not sure) item of a clip: "Keep all doubtful visible" / "Blur all doubtful"."""
        if not isinstance(blur, bool):
            raise invalid("blur")
        clip = self._reviewable(clip_id)
        with clip.lock:
            changed = []
            for item in clip.analysis.items:
                if item.verdict != "check":
                    continue
                clip.analysis.set_blur(item.item_id, blur)
                clip.reviewed.add(item.item_id)
                changed.append(self.item_json(clip, clip.analysis.item(item.item_id)))
            if changed:
                self._changed(clip)
            answer = {"items": changed, "rev": clip.rev, "to_check": self.to_check(clip)}
            if changed:
                self.hub.emit("items.changed", clip_id=clip_id, **answer)
            return answer

    def remove_area(self, clip_id: str, item_id: int) -> dict:
        clip = self._reviewable(clip_id)
        with clip.lock:
            try:
                clip.analysis.remove_area(item_id)
            except KeyError:
                raise ApiError(404, "error.not_found", {"item_id": item_id}) from None
            clip.reviewed.discard(item_id)
            clip.area_boxes.pop(item_id, None)
            clip.area_thumbs.pop(item_id, None)
            self._changed(clip)
            self.hub.emit("area.removed", clip_id=clip_id, item_id=item_id, rev=clip.rev, to_check=self.to_check(clip))
            return {"ok": True, "rev": clip.rev, "to_check": self.to_check(clip)}

    def _reviewable(self, clip_id: str) -> Clip:
        clip = self._analyzed(clip_id)
        if clip.status == "rendering":
            raise ApiError(409, "error.clip.state", {"clip_id": clip_id, "status": clip.status})
        return clip

    def _start(self, kind: str, clips: List[Clip], target, *args) -> Job:
        with self.lock:
            self._require_idle()
            self._jobs += 1
            job = Job(f"j{self._jobs}", kind, clips)
            self.job = job
        self._thread = threading.Thread(target=self._run, args=(job, target) + args, name=job.job_id, daemon=True)
        self._thread.start()
        return job

    def _clips_of(self, ids: Any) -> List[Clip]:
        if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
            raise invalid("clips")
        return [self.clip(i) for i in dict.fromkeys(ids)]

    def start_analyze(self, body: dict) -> dict:
        only_keys(body, ("clips", "options", "discard"), "")
        clips = self._clips_of(body.get("clips"))
        discard = body.get("discard", False)
        if not isinstance(discard, bool):
            raise invalid("discard")
        wanted = body.get("options") or {}
        if not isinstance(wanted, dict):
            raise invalid("options")
        only_keys(wanted, ANALYZE_FIELDS, "options.")
        try:
            options = self.settings.options(**{k: v for k, v in wanted.items() if k in ANALYZE_FIELDS})
        except (InvalidSetting, ValueError) as e:
            field = e.field if isinstance(e, InvalidSetting) else "options"
            raise invalid(field, str(e)) from e
        if not discard:
            # A new analysis replaces the old one with its decisions and added areas: the person is asked before that happens.
            redone = [c for c in clips if c.analysis is not None and c.reviewed and not self._is_current(c, options)]
            if redone:
                raise ApiError(
                    409,
                    "error.analysis.discards",
                    {"count": len(redone), "decisions": sum(len(c.reviewed) for c in redone)},
                )
        estimate = self._estimate(clips, options)
        job = self._start("analyze", clips, self._analyze, options)
        return {"job_id": job.job_id, "clips": [c.clip_id for c in clips], "estimate_s": estimate}

    def _estimate(self, clips: List[Clip], options: "anonymizer.Options") -> Optional[int]:
        rate = self.frame_seconds.get(rate_key(options))
        if rate is None:
            return None
        todo = [c for c in clips if not self._is_current(c, options)]
        return int(round(rate * sum(c.info.frames for c in todo)))

    @staticmethod
    def _key(options: "anonymizer.Options") -> tuple:
        return tuple(getattr(options, f) for f in ANALYZE_FIELDS)

    def _is_current(self, clip: Clip, options: "anonymizer.Options") -> bool:
        try:
            unchanged = stamp_of(clip.path) == clip.stamp
        except OSError:
            unchanged = False
        return clip.analysis is not None and clip.analysis_key == self._key(options) and unchanged

    def job_command(self, command: str) -> dict:
        with self.lock:
            job = self.job
            if job is None:
                raise ApiError(409, "error.clip.state", {"status": "idle"})
            if command == "pause" and job.state == "running":
                job.token.pause()
                job.state = "paused"
            elif command == "resume" and job.state == "paused":
                job.token.resume()
                job.state = "running"
            elif command == "stop" and job.state != "stopping":
                job.state = "stopping"
                job.token.cancel()
            state = job.state
        self.hub.emit("job.state", state=state)
        return {"ok": True, "state": state}

    def _run(self, job: Job, work, *args) -> None:
        self.hub.emit(
            "job.started",
            job_id=job.job_id,
            kind=job.kind,
            clips=[
                {
                    "clip_id": c.clip_id,
                    "name": c.name,
                    "media": c.info.media,
                    "duration_s": c.info.duration_s,
                    "frames": c.info.frames,
                    "width": c.info.width,
                    "height": c.info.height,
                    "fps": c.info.fps,
                    "has_audio": c.info.has_audio,
                    "has_gps": c.info.has_gps,
                }
                for c in job.clips
            ],
        )
        started = time.monotonic()
        counts = {"analyzed": 0, "rendered": 0, "failed": 0}
        status = "done"
        try:
            status = work(job, counts, *args)
        except EngineFailure as e:  # the worker was killed or died: said as it is, the server and the analyses stay
            status = "failed"
            self.hub.emit("job.error", code=e.code, clip_id=job.current, params=e.params, detail=e.detail)
            counts["failed"] += self._fail_running(job, e.code, e.params)
        except Exception as e:  # the thread must report and end, whatever went wrong
            status = "failed"
            self.hub.emit(
                "job.error",
                code="error.engine.crashed",
                clip_id=job.current,
                params={},
                detail="".join(traceback.format_exception(e)),
            )
            counts["failed"] += self._fail_running(job, "error.engine.crashed", {})
        finally:
            for clip in job.clips:
                if clip.render is not None and clip.render.get("phase") != "done":
                    clip.render = None  # a render that failed, was stopped or never began has nothing to show
                if clip.status in ("queued", "stopped"):
                    clip.status = "analyzed" if clip.analysis is not None else "imported"
            with self.lock:
                self.job = None
            summary = {
                "clips": len(job.clips),
                "analyzed": counts["analyzed"],
                "rendered": counts["rendered"],
                "failed": counts["failed"],
                "to_check": sum(self.to_check(c) for c in job.clips if c.analysis is not None),
                "seconds": round(time.monotonic() - started, 2),
            }
            self.hub.emit("job.finished", job_id=job.job_id, status=status, summary=summary)

    @staticmethod
    def _fail_running(job: Job, code: str, params: dict) -> int:
        """The clip a job was working on when it ended badly: not analysed (failed, with the reason), or analysed and not rendered. How many
        there were: the job's summary counts them as failed."""
        failed = 0
        for clip in job.clips:
            if clip.status == "analyzing":
                clip.status = "failed"
                clip.failure = {"code": code, "params": params}
                failed += 1
            elif clip.status == "rendering":
                clip.status = "analyzed"
                failed += 1
        return failed

    def _open_engine(self, options: "anonymizer.Options"):
        threads = self.settings.get()["threads"]
        if self.engine_kind == "process":
            return ProcessEngine(options, threads, self.models_dir, self.work_dir, **self.engine_options)
        return LocalEngine(options, threads, self.models_dir, self.modules)

    def _progress_handler(self, clip: Clip, phase: str, detection: Optional[str] = None):
        started = time.monotonic()
        last = [0.0]

        def handle(event) -> None:
            kind = type(event).__name__
            if kind == "Progress":
                now = time.monotonic()
                if event.fraction < 1.0 and now - last[0] < PROGRESS_INTERVAL:
                    return
                last[0] = now
                elapsed = now - started
                eta = round(elapsed * (1 - event.fraction) / event.fraction) if event.fraction >= 0.02 else None
                wanted = (
                    ("live", event.live),
                    ("found", event.found),
                    ("stage", getattr(event, "stage", None)),
                    ("zones", getattr(event, "zones", None)),
                )
                extra = {k: v for k, v in wanted if v is not None}
                if phase == "analyze":
                    clip.progress = {"fraction": event.fraction, "frame": event.frame, "eta_s": eta, **extra}
                elif event.phase in ("render", "compare"):
                    clip.render = {
                        "phase": event.phase,
                        "fraction": event.fraction,
                        **({"output": clip.render["output"]} if clip.render and "output" in clip.render else {}),
                    }
                self.hub.emit(
                    "clip.progress",
                    clip_id=clip.clip_id,
                    phase=event.phase,
                    fraction=event.fraction,
                    frame=event.frame,
                    frames=event.frames,
                    eta_s=eta,
                    **extra,
                )
            elif kind == "JobWarning":
                code = event.code
                if code == SECOND_LOOK_SKIPPED and detection == "fast":
                    # Fast looks at nothing a second time: what its count holds is the vehicles beyond the plate search's budget.
                    code = VEHICLES_SKIPPED
                self.hub.emit("job.warning", code=code, clip_id=clip.clip_id, params=event.params)
            elif kind == "ItemFound" and phase == "analyze":
                item = self.item_json(clip, event.item)
                clip.found_items.append(item)
                self.hub.emit("clip.item", clip_id=clip.clip_id, item=item)

        return handle

    def _error(self, clip: Clip, error: "anonymizer.AnonymizerError") -> None:
        if clip.status == "failed":  # the analysis of this clip failed (a failed render leaves the clip analysed)
            clip.failure = {"code": error.code, "params": error.params}
        self.hub.emit("job.error", code=error.code, clip_id=clip.clip_id, params=error.params, detail=error.detail)

    def _analyze(self, job: Job, counts: dict, options: "anonymizer.Options") -> str:
        for clip in job.clips:
            clip.status = "queued" if clip.analysis is None else clip.status
        with self._open_engine(options) as engine:
            for clip in job.clips:
                if job.token.cancelled:
                    return "stopped"
                job.current = clip.clip_id
                if self._is_current(clip, options):
                    if (
                        clip.status == "failed"
                    ):  # an analysis that held when a later run of the clip failed: it is the clip's again
                        clip.status, clip.failure = "analyzed", None
                    self._announce_analyzed(clip)
                    counts["analyzed"] += 1
                    continue
                clip.progress, clip.found_items, clip.failure = None, [], None
                clip.status = "analyzing"
                try:
                    stamp_before = stamp_of(
                        clip.path
                    )  # what the analysis is of: a file that changes during it is caught at export
                except OSError:  # the original went away since the import (an unmounted disk, a renamed folder): this clip fails, the others go on
                    clip.status = "failed"
                    counts["failed"] += 1
                    self._error(clip, anonymizer.AnonymizerError("error.source.missing", {"path": clip.path}))
                    continue
                self.hub.emit("clip.started", clip_id=clip.clip_id, phase="analyze")
                began = time.monotonic()
                try:
                    analysis = engine.analyze(
                        clip.clip_id, clip.path, self._progress_handler(clip, "analyze", options.detection), job.token
                    )
                except anonymizer.Cancelled:
                    clip.status = "stopped"
                    return "stopped"
                except anonymizer.ModelLoadError as e:
                    clip.status = "failed"
                    counts["failed"] += 1
                    self._error(clip, e)
                    return "failed"  # every other clip would fail the same way
                except anonymizer.AnonymizerError as e:
                    clip.status = "failed"
                    counts["failed"] += 1
                    self._error(clip, e)
                    continue
                with clip.lock:
                    if clip.info.media == "video" and analysis.frames and analysis.frames != clip.info.frames:
                        # The count read from the container's header (duration times rate: Matroska, WebM and AVI) is an estimate that can be
                        # one over what the decoder gives; the frames of the analysis are the real ones, and every bound below is theirs.
                        clip.info = replace(clip.info, frames=analysis.frames)
                    clip.analysis, clip.analysis_key = analysis, self._key(options)
                    clip.stamp = stamp_before
                    clip.reviewed = set()
                    clip.area_boxes, clip.area_thumbs = {}, {}  # the areas of the old analysis are gone with it
                    clip.progress, clip.found_items = (
                        None,
                        [],
                    )  # what a late window needed while the analysis ran; the analysis holds the items now
                    clip.rev += 1
                    clip.render = None  # what was written was of the old analysis
                    clip.status = "analyzed"
                if analysis.frames:
                    per_frame = (time.monotonic() - began) / analysis.frames
                    self.frame_seconds[rate_key(options)] = per_frame
                    clip.seconds_per_frame[rate_key(options)] = per_frame
                counts["analyzed"] += 1
                self._announce_analyzed(clip)
        return "done"

    def _announce_analyzed(self, clip: Clip) -> None:
        summary = self.analysis_summary(clip)
        self.hub.emit(
            "clip.analyzed",
            clip_id=clip.clip_id,
            items=summary["items"],
            to_check=summary["to_check"],
            unchecked_frames=summary["unchecked_frames"],
            rev=summary["rev"],
            settings=summary["settings"],
            frames=clip.info.frames,
            analysis_url=f"/api/clips/{clip.clip_id}/analysis.json",
        )
        if summary["unchecked_frames"]:
            self.hub.emit(
                "job.warning",
                code="warning.analysis.unchecked",
                clip_id=clip.clip_id,
                params={"frames": summary["unchecked_frames"]},
            )

    def _render_settings(self, wanted: dict) -> dict:
        reports = wanted.get("reports", {})
        if not isinstance(reports, dict):
            raise invalid("reports")
        pattern = wanted.get("name_pattern", DEFAULT_PATTERN)
        if not isinstance(pattern, str) or not pattern or "/" in pattern or "\\" in pattern or has_control(pattern):
            raise invalid("name_pattern")
        problem = pattern_error(pattern)
        if problem is not None:
            raise invalid("name_pattern", problem)
        if not pattern.strip(" ."):
            raise invalid(
                "name_pattern", "a name of dots and spaces only"
            )  # "." gave "..jpg", a hidden file and no name
        literal = "".join(
            text for text, *_ in string.Formatter().parse(pattern)
        )  # the fields, such as {index:03d}, hold a colon
        if any(c in literal for c in '<>:"|?*'):
            raise invalid("name_pattern", 'a file name cannot hold any of < > : " | ? * (Windows cannot save it)')
        language = wanted.get(
            "language"
        )  # the page's own language, which the PDF is written in when none was ever chosen
        if language is not None and (not isinstance(language, str) or language not in self.languages):
            raise invalid("language")
        paper = wanted.get(
            "paper", "a4"
        )  # the paper of the PDF report: the page says "letter" for a browser in US English
        if paper not in pdf.PAPERS:
            raise invalid("paper")
        destination = wanted.get("destination")
        if destination is not None and (
            not isinstance(destination, str) or not destination.strip() or has_control(destination)
        ):
            raise invalid("destination")
        overrides = {
            k: wanted[k] for k in ("blur", "strength", "extra_margin", "remove_gps", "gap_fill") if k in wanted
        }
        try:
            options = self.settings.options(**overrides)
        except InvalidSetting as e:
            raise invalid(e.field, e.detail) from e
        return {
            "pattern": pattern,
            "destination": folder_path(destination) if destination else None,
            "reports": {k: bool(reports.get(k, k != "pdf")) for k in ("pdf", "csv", "json")},
            "comparison": bool(wanted.get("comparison", False)),
            "language": language,
            "paper": paper,
            "options": options,
        }

    def _blocked(self, clip: Clip, output: Optional[str] = None) -> Optional[dict]:
        if clip.analysis is None or clip.status not in STATES_WITH_ANALYSIS:
            return {"code": "error.clip.state", "params": {"status": clip.status}}
        if output is not None and len(os.fsencode(os.path.basename(output))) > MAX_OUTPUT_NAME:
            return {"code": "error.output.name_too_long", "params": {"limit": MAX_OUTPUT_NAME}}
        frames = len(clip.analysis.unchecked_frames)
        if frames:
            return {"code": "error.analysis.incomplete", "params": {"frames": frames}}
        try:
            if stamp_of(clip.path) != clip.stamp:
                return {"code": "error.source.changed", "params": {}}
        except FileNotFoundError:
            return {
                "code": "error.source.missing",
                "params": {"path": os.path.basename(clip.path) if self.demo else clip.path},
            }
        except OSError:
            return {"code": "error.source.changed", "params": {}}
        return None

    def _destination(self, clip: Clip, chosen: Optional[str]) -> str:
        if self.demo:
            return os.path.join(self.work_dir, "out")
        if chosen:
            return chosen
        if clip.origin == "path":
            return os.path.join(os.path.dirname(clip.path), OUTPUT_FOLDER)
        return self.settings.get()["default_destination"]

    def plan(self, body: dict) -> dict:
        return self._hide_server_paths(self._plan(body)[0])

    def _hide_server_paths(self, plan: dict) -> dict:
        """The hosted demo is somebody else's computer: its visitors get the names of their files, not the folder of the server's session."""
        if not self.demo:
            return plan
        hidden = dict(plan, destination="", folders=[], unwritable=[])
        hidden["clips"] = [dict(entry, output=os.path.basename(entry["output"])) for entry in plan["clips"]]
        return hidden

    def _plan(self, body: dict) -> Tuple[dict, dict, Dict[str, str]]:
        """The public plan, the render settings behind it and the output path of each clip."""
        only_keys(body, ("clips", "options"), "")
        clips = self._clips_of(body.get("clips"))
        wanted = body.get("options") or {}
        if not isinstance(wanted, dict):
            raise invalid("options")
        only_keys(wanted, RENDER_OPTION_KEYS, "options.")
        render = self._render_settings(wanted)
        taken: List[str] = []
        listings: Dict[str, set] = {}
        entries = []
        total = total_low = 0
        folders = []
        for index, clip in enumerate(clips, 1):
            folder = self._destination(clip, render["destination"])
            folders.append(folder)
            stem = "".join(
                "_" if ord(c) < 32 else c for c in os.path.splitext(clip.name)[0]
            )  # a line feed in a source name is legal on Linux, not in the output
            ext = ".mp4" if clip.info.media == "video" else os.path.splitext(clip.name)[1].lower()
            wanted_path = os.path.join(folder, render["pattern"].format(name=stem, index=index) + ext)
            output = free_name(wanted_path, taken, listings)
            taken.append(output)
            low, high = export_size.estimate(clip.info)
            total += high
            total_low += low
            entries.append(
                {
                    "clip_id": clip.clip_id,
                    "output": output,
                    "exists": output != wanted_path,
                    "estimate_bytes": high,  # the top of the range, which is what the free space has to hold
                    "estimate_low_bytes": low,
                    "blocked": self._blocked(clip, output),
                    "to_check": self.to_check(clip) if clip.analysis is not None else 0,
                }
            )
        distinct = list(dict.fromkeys(folders))  # "next to each source" is one folder per source folder
        unwritable = [f for f in distinct if not can_create_in(f)]
        free = []
        for folder in distinct:
            anchor = nearest_existing(folder)
            free.append(shutil.disk_usage(anchor if os.path.isdir(anchor) else os.path.dirname(anchor)).free)
        public = {
            "destination": distinct[0]
            if len(distinct) == 1
            else None,  # None: the outputs go to several folders (see `folders`)
            "folders": distinct,
            "unwritable": unwritable,
            "writable": not unwritable,
            "free_bytes": min(free),
            "clips": entries,
            "total_bytes": total,
            "total_low_bytes": total_low,
        }
        return public, render, {e["clip_id"]: e["output"] for e in entries}

    def start_render(self, body: dict) -> dict:
        plan, render, targets = self._plan(body)
        for entry in plan["clips"]:
            if entry["blocked"] is not None:
                params = dict(entry["blocked"]["params"], clip_id=entry["clip_id"])
                raise ApiError(422, entry["blocked"]["code"], params)
        if not plan["writable"]:
            raise ApiError(422, "error.output.not_writable", {"path": plan["unwritable"][0]})
        clips = [self.clip(e["clip_id"]) for e in plan["clips"]]
        job = self._start("render", clips, self._render, render, targets)
        outputs = {k: os.path.basename(v) for k, v in targets.items()} if self.demo else targets
        return {"job_id": job.job_id, "clips": [c.clip_id for c in clips], "outputs": outputs}

    def _publish(self, path: str) -> str:
        """What the page is told for a written file: its path, or in demo mode a download link."""
        if not self.demo:
            return path
        file_id = f"f{len(self.downloads) + 1}"
        self.downloads[file_id] = (path, os.path.basename(path))
        return f"/api/downloads/{file_id}/{os.path.basename(path)}"

    def _render(self, job: Job, counts: dict, render: dict, targets: Dict[str, str]) -> str:
        options = render["options"]
        for clip in job.clips:
            clip.status = "queued"
            clip.render = {"phase": "queued", "fraction": 0, "output": os.path.basename(targets[clip.clip_id])}
        with self._open_engine(options) as engine:
            for clip in job.clips:
                if job.token.cancelled:
                    clip.status = "analyzed"
                    continue
                job.current = clip.clip_id
                target = targets[clip.clip_id]
                clip.status = "rendering"
                clip.render = {"phase": "render", "fraction": 0, "output": os.path.basename(target)}
                self.hub.emit("clip.started", clip_id=clip.clip_id, phase="render")
                try:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with clip.lock:
                        try:
                            result = engine.render(
                                clip.clip_id, clip.analysis, target, self._progress_handler(clip, "render"), job.token
                            )
                        finally:
                            remove_leftover_partial(target)  # a worker that was killed cannot clean up after itself
                        comparison, comparison_stopped = self._comparison(
                            engine, job, clip, result.output, target, render
                        )
                        reports = self._write_reports(clip, result, target, render, comparison)
                except anonymizer.Cancelled:
                    clip.status = "analyzed"
                    return "stopped"
                except anonymizer.AnonymizerError as e:
                    clip.status = "analyzed"
                    counts["failed"] += 1
                    self._error(clip, self._blame_a_full_disk(e, clip, target))
                    continue
                except OSError as e:
                    clip.status = "analyzed"
                    counts["failed"] += 1
                    self.hub.emit(
                        "job.error",
                        code="error.output.disk_full" if e.errno == errno.ENOSPC else "error.output.not_writable",
                        clip_id=clip.clip_id,
                        params={"path": os.path.dirname(target)},
                        detail=f"{type(e).__name__}: {e}",
                    )
                    continue
                clip.status = "rendered"
                counts["rendered"] += 1
                extra = {} if comparison is None else {"comparison": self._publish(comparison)}
                if comparison_stopped:
                    extra["comparison_stopped"] = (
                        True  # the person stopped the export while the comparison video was being made
                    )
                written = dict(
                    outputs=[self._publish(result.output)],
                    report={k: self._publish(v) for k, v in reports.items()},
                    seconds=result.seconds,
                    **extra,
                )
                clip.render = {
                    "phase": "done",
                    "outputs": written["outputs"],
                    "report": written["report"],
                    "comparison": written.get("comparison"),
                    "comparison_stopped": comparison_stopped,
                    "seconds": result.seconds,
                }
                self.hub.emit("clip.rendered", clip_id=clip.clip_id, **written)
        return "stopped" if job.token.cancelled else "done"

    @staticmethod
    def _blame_a_full_disk(
        error: "anonymizer.AnonymizerError", clip: Clip, target: str
    ) -> "anonymizer.AnonymizerError":
        """A render that failed on a destination with less room than the file it was copying says what it knows (a photo written short
        is "metadata not removed", from its check of the file): when the folder has less free space than the original, the cause is the
        space, and that is what the person is told."""
        try:
            folder = os.path.dirname(target)
            if shutil.disk_usage(folder).free < os.path.getsize(clip.path):
                return anonymizer.AnonymizerError("error.output.disk_full", {"path": folder}, error.detail)
        except OSError:
            pass
        return error

    def _comparison(
        self, engine, job: Job, clip: Clip, output: str, target: str, render: dict
    ) -> "tuple[Optional[str], bool]":
        """The side-by-side video when it was asked for, and whether the person stopped the export while it was being made. The
        render is already done: a failure here is reported and costs only the comparison; a stop leaves the clip written without it and
        says so (the clip is not simply \"done\")."""
        if not render["comparison"]:
            return None, False
        if clip.info.media != "video":
            self.hub.emit("job.warning", code="comparison.photo_skipped", clip_id=clip.clip_id, params={})
            return None, False
        path = free_name(f"{os.path.splitext(target)[0]}.compare.mp4")
        try:
            written = engine.compare(
                clip.clip_id, clip.path, output, path, self._progress_handler(clip, "compare"), job.token
            )
        except anonymizer.Cancelled:
            self.hub.emit(
                "job.warning", code="warning.comparison.stopped", clip_id=clip.clip_id, params={"name": clip.name}
            )
            return None, True
        except anonymizer.AnonymizerError as e:
            self._error(clip, e)
            return None, False
        # Its left half is the original, not blurred: it is not a file to publish, and every report of this clip says so.
        self.hub.emit(
            "job.warning",
            code="warning.comparison.contains_original",
            clip_id=clip.clip_id,
            params={"file": os.path.basename(written)},
        )
        return written, False

    def _make_pdf(
        self,
        clip: Clip,
        path: str,
        comparison_file: Optional[str] = None,
        language: Optional[str] = None,
        paper: str = "a4",
    ) -> bool:
        browser = self.find_browser()
        if browser is None:
            return False
        language = language or self.settings.get()["language"] or "en"
        texts = self.texts_for(language)
        rows = clip.analysis.report(clip.name).rows
        page = pdf.report_html(
            clip.name,
            rows,
            texts,
            clip.info.media == "video",
            date.today(),
            comparison_file,
            paper,
            pdf.decimal_for(language),
        )
        page_path = os.path.join(self.work_dir, f"{clip.clip_id}.report.html")
        try:
            with open(page_path, "w", encoding="utf-8") as f:
                f.write(page)
            pdf.html_to_pdf(browser, page_path, path)
        except (RuntimeError, OSError):
            return False
        finally:
            if os.path.exists(page_path):
                os.remove(page_path)
        return True

    def _write_reports(
        self, clip: Clip, result: "anonymizer.Result", target: str, render: dict, comparison: Optional[str] = None
    ) -> Dict[str, str]:
        """The reports of a render. When a comparison video was written (it shows the original without blur) the JSON and the PDF
        say so, with its name; the CSV is rows only."""
        written: Dict[str, str] = {}
        note = os.path.basename(comparison) if comparison else None
        for fmt in ("csv", "json"):
            if render["reports"][fmt]:
                path = report_path(target, fmt)
                result.report.to_csv(path) if fmt == "csv" else result.report.to_json(path)
                if fmt == "json" and note:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    data["notes"] = [{"code": "report.note.comparison_video", "params": {"file": note}}]
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=1)
                written[fmt] = path
        if render["reports"]["pdf"]:
            path = report_path(target, "pdf")
            if self._make_pdf(clip, path, note, render.get("language"), render.get("paper", "a4")):
                written["pdf"] = path
            else:
                self.hub.emit("job.warning", code="report.pdf.unavailable", clip_id=clip.clip_id, params={})
        return written
