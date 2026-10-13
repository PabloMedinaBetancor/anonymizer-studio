"""How a job reaches the engine: in this process (LocalEngine) or in a worker process that can be killed (ProcessEngine).

Both have the same three calls. The server uses the process engine: a detector that hangs is killed after ten silent
minutes, one that crashes the whole interpreter (a native library) costs one job and not the server, and the memory a
job used goes back to the system when it ends.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Deque, Dict, Optional, Sequence

import anonymizer

SILENT_SECONDS = 600.0  # an engine silent for 10 minutes is killed
WATCH_SECONDS = 30.0  # how often the watchdog looks
KILL_AFTER_CANCEL_SECONDS = 30.0  # a stop that is not obeyed in 30 s is a kill
STDERR_LINES = 40

EventHandler = Callable[[Any], None]


def no_window() -> Dict[str, Any]:
    """On Windows a child of a GUI-less process must not open a console window of its own."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


class EngineFailure(Exception):
    """The engine could not finish: it was silent for too long, or it died. `code` is an i18n key."""

    def __init__(self, code: str, params: Optional[dict] = None, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.params = dict(params or {})
        self.detail = detail


class LocalEngine:
    """The engine in this process: the detectors are loaded here, and a hang cannot be undone."""

    def __init__(
        self, options: "anonymizer.Options", threads: int = 0, models_dir: Optional[str] = None, modules=None, **_
    ):
        self.anon = anonymizer.Anonymizer(options, models_dir=models_dir, threads=threads, modules=modules)

    def analyze(self, clip_id: str, path: str, on_event: EventHandler, token) -> "anonymizer.Analysis":
        return self.anon.analyze(path, on_event, token)

    def render(
        self, clip_id: str, analysis: "anonymizer.Analysis", target: str, on_event: EventHandler, token
    ) -> "anonymizer.Result":
        return self.anon.render(analysis, target, on_event, token)

    def compare(self, clip_id: str, original: str, anonymized: str, target: str, on_event: EventHandler, token) -> str:
        return anonymizer.compare_video(original, anonymized, target, on_event=on_event, cancel=token)

    def close(self) -> None:
        self.anon.close()

    def __enter__(self) -> "LocalEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def event_of(message: Dict[str, Any]) -> Optional[Any]:
    """The API event object a worker's {"ev": ...} message stands for."""
    fields = {k: v for k, v in message.items() if k not in ("type", "v")}
    kind = message.get("type")
    if kind == "clip.progress":
        return anonymizer.Progress(**fields)
    if kind == "clip.item":
        return anonymizer.ItemFound(fields["clip_id"], anonymizer.Item(**fields["item"]))
    if kind == "job.warning":
        return anonymizer.JobWarning(**fields)
    if kind == "job.started":
        return anonymizer.Started(**fields)
    if kind == "clip.analyzed":
        return anonymizer.Analyzed(**fields)
    if kind == "clip.rendered":
        return anonymizer.Rendered(**fields)
    if kind == "job.finished":
        return anonymizer.Finished(**fields)
    return None


class ProcessEngine:
    """The engine in a worker process (ui/worker.py), started on the first call and ended with the job."""

    def __init__(
        self,
        options: "anonymizer.Options",
        threads: int = 0,
        models_dir: Optional[str] = None,
        work_dir: str = ".",
        silent_after: float = SILENT_SECONDS,
        watch_every: float = WATCH_SECONDS,
        kill_after_cancel: float = KILL_AFTER_CANCEL_SECONDS,
        env: Optional[Dict[str, str]] = None,
        **_,
    ):
        self.options, self.threads, self.models_dir, self.work_dir = options, threads, models_dir, work_dir
        self.silent_after, self.watch_every, self.kill_after_cancel = silent_after, watch_every, kill_after_cancel
        self.env = env
        self.proc: Optional[subprocess.Popen] = None
        self._messages: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._stderr: Deque[str] = collections.deque(maxlen=STDERR_LINES)
        self._calls = 0
        self._last_beat = time.monotonic()
        self._busy = False
        self._paused = False
        self._killed_for: Optional[str] = None
        self._closing = threading.Event()

    def _command(self) -> Sequence[str]:
        flags = ["-s"] if sys.flags.no_user_site else []
        return [sys.executable, *flags, "-m", "anonymizer.ui.worker"]

    def _start(self) -> None:
        env = dict(os.environ)
        env.update(self.env or {})
        self.proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            bufsize=1,
            **no_window(),
        )
        threading.Thread(target=self._read_stdout, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
        options = {k: getattr(self.options, k) for k in self.options.__dataclass_fields__}
        self._send({"cmd": "hello", "options": options, "models_dir": self.models_dir, "threads": self.threads})
        self._last_beat = time.monotonic()
        while True:
            message = self._messages.get()
            if message is None:
                raise self._failure()
            if message.get("ready"):
                return

    def _send(self, command: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass  # the worker is gone: the reader sees the end of its output and the call fails with the reason

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self._last_beat = time.monotonic()
            try:
                self._messages.put(json.loads(line))
            except ValueError:
                self._stderr.append(line.rstrip())  # not ours: kept for the report if the worker dies
        self._messages.put(None)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            self._stderr.append(line.rstrip())

    def _watch(self, proc: subprocess.Popen) -> None:
        """Every `watch_every` seconds: a busy worker that has said nothing for `silent_after` is killed. A paused
        worker is quiet on purpose."""
        while not self._closing.wait(self.watch_every):
            if proc.poll() is not None:
                return
            quiet = time.monotonic() - self._last_beat
            if self._busy and not self._paused and quiet > self.silent_after:
                self._kill("silent")
                return

    def _kill(self, reason: str) -> None:
        self._killed_for = self._killed_for or reason
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()  # this process, by its pid

    def _failure(self) -> Exception:
        """What ended the worker, as the exception for the call that was waiting."""
        if self.proc is not None:
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        tail = "\n".join(self._stderr)
        if self._killed_for == "silent":
            return EngineFailure("error.engine.silent", {"minutes": round(self.silent_after / 60)}, tail)
        if self._killed_for == "cancel":
            return anonymizer.Cancelled()
        code = self.proc.returncode if self.proc is not None else None
        return EngineFailure("error.engine.crashed", {}, f"exit code {code}\n{tail}")

    def _forward(self, token, finished: threading.Event) -> None:
        """Pass the user's Pause, Resume and Stop on to the worker, and kill it if a Stop is not obeyed in time."""
        sent_pause = False
        cancelled_at = None
        while not finished.wait(0.1):
            if token.paused != sent_pause:
                sent_pause = token.paused
                self._paused = sent_pause
                self._last_beat = time.monotonic()  # the silence of a pause is not the engine's
                self._send({"cmd": "pause" if sent_pause else "resume"})
            if token.cancelled and cancelled_at is None:
                cancelled_at = time.monotonic()
                self._send({"cmd": "cancel"})
            if cancelled_at is not None and time.monotonic() - cancelled_at > self.kill_after_cancel:
                self._kill("cancel")
                return

    def _call(self, command: dict, on_event: EventHandler, token) -> dict:
        if self.proc is None:
            self._start()
        elif self.proc.poll() is not None:
            raise self._failure()
        self._calls += 1
        call = self._calls
        finished = threading.Event()
        self._busy, self._paused = True, False
        self._last_beat = time.monotonic()
        threading.Thread(target=self._forward, args=(token, finished), daemon=True).start()
        try:
            self._send({**command, "id": call})
            while True:
                message = self._messages.get()
                if message is None:
                    raise self._failure()
                if "ev" in message:
                    event = event_of(message["ev"])
                    if event is not None:
                        on_event(event)
                elif "done" in message:
                    return message["done"]
                elif "error" in message:
                    raise error_of(message["error"])
                elif "crash" in message:
                    self._messages.put(None)  # the worker exits after this: the next read says so
                    self._stderr.append(message["crash"])
        finally:
            finished.set()
            self._busy = False

    def analyze(self, clip_id: str, path: str, on_event: EventHandler, token) -> "anonymizer.Analysis":
        out = os.path.join(self.work_dir, f"{clip_id}.analysis.json")
        self._call({"cmd": "analyze", "path": path, "out": out}, on_event, token)
        return anonymizer.load_analysis(out)

    def render(
        self, clip_id: str, analysis: "anonymizer.Analysis", target: str, on_event: EventHandler, token
    ) -> "anonymizer.Result":
        saved = os.path.join(self.work_dir, f"{clip_id}.render.json")
        analysis.save(saved)
        done = self._call({"cmd": "render", "analysis": saved, "target": target}, on_event, token)
        return anonymizer.Result(
            done["output"], analysis.report(os.path.basename(analysis.source)), done["warnings"], done["seconds"]
        )

    def compare(self, clip_id: str, original: str, anonymized: str, target: str, on_event: EventHandler, token) -> str:
        command = {"cmd": "compare", "original": original, "anonymized": anonymized, "target": target}
        return self._call(command, on_event, token)["output"]

    def close(self) -> None:
        self._closing.set()
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            self._send({"cmd": "quit"})
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass  # already closed by the reader

    def __enter__(self) -> "ProcessEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def error_of(error: dict) -> Exception:
    """The AnonymizerError a worker's {"error": ...} message stands for."""
    cls = getattr(anonymizer, error.get("kind", ""), None)
    if not (isinstance(cls, type) and issubclass(cls, anonymizer.AnonymizerError)):
        cls = anonymizer.AnonymizerError
    return cls(error["code"], error.get("params"), error.get("detail", ""))
