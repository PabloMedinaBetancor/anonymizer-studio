"""The events of the session: a numbered ring that the page's event stream reads, and can replay after a reconnect."""

from __future__ import annotations

import collections
import json
import threading
from typing import Any, Callable, Deque, Dict, Iterator, Optional, Tuple

RING = 5000
PING_SECONDS = 15.0


class Hub:
    def __init__(self, ring: int = RING):
        self._ring: Deque[Tuple[int, str]] = collections.deque(maxlen=ring)
        self._seq = 0
        self._cond = threading.Condition()
        self._closed = False
        self.scrub: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = (
            None  # applied to every event's data before it is kept
        )

    @property
    def seq(self) -> int:
        with self._cond:
            return self._seq

    def emit(self, type: str, **data: Any) -> int:
        """Add an event ({"type", "v": 1, ...}) and wake the streams; returns its id."""
        if self.scrub is not None:
            data = self.scrub(data)
        message = json.dumps({"type": type, "v": 1, **data}, separators=(",", ":"))
        with self._cond:
            self._seq += 1
            self._ring.append((self._seq, message))
            self._cond.notify_all()
            return self._seq

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def stream(self, after: Optional[int] = None, ping: float = PING_SECONDS) -> Iterator[Optional[Tuple[int, str]]]:
        """(id, JSON text) for each event after `after`, then new ones as they come; None every `ping` seconds without
        events. With no `after` the stream starts at the present. When `after` is older than the ring (or newer than the
        last event: the server restarted) the first item is a state.resync. The starting point is taken when this is
        called, not at the first step of the iterator: a caller that subscribes before it sends its headers cannot lose an
        event that is emitted between the headers and the first read."""
        with self._cond:
            current = self._seq
            if after is None:
                after = current
            oldest = self._ring[0][0] if self._ring else current + 1
            resync = after > current or (after < current and after + 1 < oldest)
        return self._events(current, after, resync, ping)

    def _events(self, current: int, after: int, resync: bool, ping: float) -> Iterator[Optional[Tuple[int, str]]]:
        if resync:
            yield (current, json.dumps({"type": "state.resync", "v": 1}))
            after = current
        while True:
            with self._cond:
                pending = [item for item in self._ring if item[0] > after]
                if not pending and not self._closed:
                    self._cond.wait(ping)
                    pending = [item for item in self._ring if item[0] > after]
                closed = self._closed
            if pending:
                for item in pending:
                    yield item
                after = pending[-1][0]
            elif closed:
                return
            else:
                yield None


def event_dict(text: str) -> Dict[str, Any]:
    return json.loads(text)
