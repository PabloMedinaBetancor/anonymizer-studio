"""What the engine tells the UI, and how the UI stops it.

Every event is a JSON object {"type": ..., "v": 1, ...} that carries i18n keys and parameters, never text.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

VERSION = 1


class Level(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class Event:
    type: str  # job.started, clip.progress, clip.item, clip.analyzed, clip.rendered, job.warning, job.error, ...
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {"type": self.type, "v": VERSION, **self.data}


Sink = Callable[[Event], None]


class Cancelled(Exception):
    """Stop was requested. Never raised by a timeout."""


class StopFlag:
    """The Stop and Pause buttons: set from any thread, checked by the engine between frames. A paused check() waits
    until resume() or a stop; nothing is lost meanwhile."""

    def __init__(self):
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._lock = threading.Lock()  # a pause racing a stop must never leave the worker waiting forever

    def request(self) -> None:
        with self._lock:
            self._stop.set()
            self._running.set()  # a paused worker wakes up to stop

    @property
    def requested(self) -> bool:
        return self._stop.is_set()

    def pause(self) -> None:
        with self._lock:
            if not self._stop.is_set():
                self._running.clear()

    def resume(self) -> None:
        self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def check(self) -> None:
        self._running.wait()
        if self._stop.is_set():
            raise Cancelled()

    def wait(self, seconds: float) -> None:
        """Sleep, but raise Cancelled as soon as Stop is requested."""
        if self._stop.wait(max(0.0, seconds)):
            raise Cancelled()


class ClipError(Exception):
    """A clip or photo that cannot be processed. `code` is an i18n key with its `params`; `detail` is the technical
    text for the log, never shown as it is."""

    def __init__(self, code: str, params: Optional[Dict[str, Any]] = None, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.params = params or {}
        self.detail = detail
