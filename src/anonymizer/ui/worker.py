"""The engine process: `python -m anonymizer.ui.worker`. The server starts one per job and talks to it in JSON lines, so a
detector that hangs or crashes can be killed without taking the server (and the user's review) with it.

Parent to worker, one JSON object per line on stdin:
    {"cmd": "hello", "options": {...}, "models_dir": null, "threads": 0}     first, once
    {"cmd": "analyze", "id": n, "path": ..., "out": ...}                      analyse a file, save the analysis to `out`
    {"cmd": "render", "id": n, "analysis": ..., "target": ...}                render a saved analysis
    {"cmd": "compare", "id": n, "original": ..., "anonymized": ..., "target": ...}
    {"cmd": "pause"} {"cmd": "resume"} {"cmd": "cancel"}                      act on the call that is running
    {"cmd": "quit"}
Worker to parent, on stdout (anything else a library prints goes to stderr):
    {"ready": true}                                                           after hello
    {"ev": {...}}                                                             an API event, as anonymizer.event_json
    {"done": {"id": n, ...}}                                                  the call ended well
    {"error": {"id": n, "kind": "Cancelled", "code": ..., "params": ..., "detail": ...}}   an AnonymizerError
    {"crash": "traceback"}                                                    anything else; the process exits

Every line the worker writes counts as a sign of life for the server's watchdog. The environment variable
ANONYMIZER_WORKER_MODULES="package.module:function" names a function that returns the class modules to use instead of
the built-in detectors (tests and extra classes).
"""

from __future__ import annotations

import importlib
import json
import os
import queue
import sys
import threading
import traceback
from typing import Any, Dict, Optional

import anonymizer

MODULES_VARIABLE = "ANONYMIZER_WORKER_MODULES"


class Channel:
    """The protocol on the real stdout. The file descriptor is duplicated and fd 1 pointed at stderr, so a library that
    prints cannot corrupt a message."""

    def __init__(self) -> None:
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self._lock = threading.Lock()

    def send(self, message: Dict[str, Any]) -> None:
        line = json.dumps(message, separators=(",", ":"))
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()


def custom_modules() -> Optional[list]:
    spec = os.environ.get(MODULES_VARIABLE)
    if not spec:
        return None
    module, _, function = spec.partition(":")
    return getattr(importlib.import_module(module), function)()


class Worker:
    def __init__(self, channel: Channel):
        self.channel = channel
        self.commands: "queue.Queue[Optional[dict]]" = queue.Queue()
        self.token = anonymizer.CancelToken()
        self.anon: Optional[anonymizer.Anonymizer] = None

    def listen(self) -> None:
        """Read stdin: pause, resume and cancel act at once on the running call; everything else is queued."""
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except ValueError:
                continue
            name = command.get("cmd")
            if name == "pause":
                self.token.pause()
            elif name == "resume":
                self.token.resume()
            elif name == "cancel":
                self.token.cancel()
            else:
                self.commands.put(command)
        self.token.cancel()  # the server is gone: stop what is running and leave
        self.commands.put(None)

    def emit(self, event: Any) -> None:
        self.channel.send({"ev": anonymizer.event_json(event)})

    def run(self) -> int:
        hello = json.loads(sys.stdin.readline())
        self.anon = anonymizer.Anonymizer(
            anonymizer.Options(**hello["options"]),
            models_dir=hello.get("models_dir"),
            threads=int(hello.get("threads", 0)),
            modules=custom_modules(),
        )
        threading.Thread(target=self.listen, daemon=True).start()
        self.channel.send({"ready": True})
        try:
            while True:
                command = self.commands.get()
                if command is None or command.get("cmd") == "quit":
                    return 0
                self.token = anonymizer.CancelToken()
                self.handle(command)
        finally:
            self.anon.close()

    def handle(self, command: dict) -> None:
        call = command.get("id")
        try:
            name = command["cmd"]
            if name == "analyze":
                analysis = self.anon.analyze(command["path"], self.emit, self.token)
                analysis.save(command["out"])
                done: Dict[str, Any] = {}
            elif name == "render":
                result = self.anon.render(
                    anonymizer.load_analysis(command["analysis"]), command["target"], self.emit, self.token
                )
                done = {"output": result.output, "warnings": result.warnings, "seconds": result.seconds}
            elif name == "compare":
                path = anonymizer.compare_video(
                    command["original"], command["anonymized"], command["target"], on_event=self.emit, cancel=self.token
                )
                done = {"output": path}
            else:
                raise ValueError(f"unknown command {name!r}")
            self.channel.send({"done": {"id": call, **done}})
        except anonymizer.AnonymizerError as e:
            self.channel.send(
                {
                    "error": {
                        "id": call,
                        "kind": type(e).__name__,
                        "code": e.code,
                        "params": e.params,
                        "detail": e.detail,
                    }
                }
            )
        except Exception:  # anything else is a bug or a broken library: say so and leave, the server decides what next
            self.channel.send({"crash": traceback.format_exc()})
            os._exit(1)


def main() -> int:
    return Worker(Channel()).run()


if __name__ == "__main__":
    sys.exit(main())
