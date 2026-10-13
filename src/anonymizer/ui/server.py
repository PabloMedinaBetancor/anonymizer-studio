"""The local server of Anonymizer Studio: the page and the JSON, event and media routes of docs/UI-PROTOCOL.md.

Standard library only. It listens on 127.0.0.1, answers only requests that carry the session cookie, and keeps no
state of its own: everything the page does is a call on session.Session.
"""

from __future__ import annotations

import argparse
import glob
import hmac
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

import anonymizer
from anonymizer import contract

from . import browse
from .hub import Hub
from .session import DEMO_LIMITS, ApiError, Session, demo_scrub, invalid, only_keys, remove_stale_work_folders
from .settings import InvalidSetting, Settings, user_folder

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_JSON = 1 << 20
CHUNK = 1 << 16
# How long the server waits with no page open before it stops by itself. A tablet that locks its screen or a browser that freezes a
# background tab loses its connection for minutes, so the wait is long; and while clips are loaded (their analyses and the person's
# decisions are in memory and nowhere else) it is very long.
IDLE_SECONDS = 600.0
IDLE_SECONDS_WITH_CLIPS = 6 * 3600.0
COOKIE = "anon_session"  # the name carries the port: cookies are shared by every port of 127.0.0.1, and a second Studio would log the first out


def cookie_name(port: int) -> str:
    return f"{COOKIE}_{port}"


MEDIA_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
PLACEHOLDER = (
    "<!doctype html><meta charset=utf-8><title>Anonymizer Studio</title>"
    "<p>The Studio server is running, but the page files are not installed (src/anonymizer/ui/static).</p>"
)


def content_disposition(name: str) -> str:
    """An attachment header for a file name that may come from a visitor: no quotes, line breaks or path separators in the plain
    name, and the real name, percent-encoded, in filename* (RFC 5987)."""
    plain = "".join(c if 32 <= ord(c) < 127 and c not in '"\\/;' else "_" for c in name) or "download"
    return f"attachment; filename=\"{plain}\"; filename*=UTF-8''{quote(name, safe='')}"


class Request:
    """What a route handler receives."""

    def __init__(self, handler: "Handler", match: "re.Match[str]", query: Dict[str, List[str]]):
        self.handler = handler
        self.groups = match.groups()
        self.query = query

    def arg(self, name: str, default: Optional[str] = None) -> Optional[str]:
        values = self.query.get(name)
        return values[0] if values else default

    def int_arg(self, name: str, default: Optional[int] = None) -> int:
        raw = self.arg(name)
        if raw is None:
            if default is None:
                raise invalid(name)
            return default
        try:
            return int(raw)
        except ValueError:
            raise invalid(name) from None

    def json(self) -> dict:
        return self.handler.read_json()


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, session: Session, hub: Hub, token: str, static_dir: str, i18n_dir: str, port: int = 0):
        super().__init__(("127.0.0.1", port), Handler)
        self.session = session
        self.hub = hub
        self.token = token
        self.static_dir = os.path.realpath(static_dir)
        self.i18n_dir = i18n_dir
        self.streams = 0
        self.told_settings_not_saved = False
        self.last_request = time.monotonic()
        self._streams_lock = threading.Lock()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.token}"

    def stream_opened(self, delta: int) -> None:
        with self._streams_lock:
            self.streams += delta

    def idle_for(self) -> float:
        """Seconds since the page was last seen (an open event stream counts as seen), 0 while a job runs."""
        if self.streams or self.session.job is not None:
            return 0.0
        return time.monotonic() - self.last_request

    def idle_limit(self) -> float:
        """How long the page may be gone before the server stops: long while clips are loaded."""
        return IDLE_SECONDS_WITH_CLIPS if self.session.clips else IDLE_SECONDS

    def idle_expired(self, seconds: Optional[float] = None) -> bool:
        return self.idle_for() > (self.idle_limit() if seconds is None else seconds)

    def stop_later(self) -> None:
        threading.Thread(target=self.shutdown, daemon=True).start()


def languages_in(folder: str) -> List[str]:
    try:
        found = sorted(f[: -len(".json")] for f in os.listdir(folder) if f.endswith(".json"))
    except OSError:
        found = []
    return ["en"] + [lang for lang in found if lang != "en"]


# Sent with every answer. The page loads nothing from outside (scripts, styles, pictures, video and its own calls are all this origin, plus
# data: and blob: for the picture of an area and the favicon), so the policy says exactly that; it also keeps the page from being framed
# and keeps the address (and with it the session's token, while it is in it) from going along with a click on a link to another site.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
        "media-src 'self' blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
}
REQUEST_TIMEOUT = 120.0  # seconds a connection may sit without sending or taking a byte: a half-sent request must not hold a thread for ever


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AnonymizerStudio"
    server: StudioServer
    timeout = REQUEST_TIMEOUT

    def end_headers(self) -> None:
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # the demo keeps no logs, the local app has its own
        pass

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True  # the page went away in the middle of an answer

    def send_bytes(self, status: int, body: bytes, content_type: str, headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send_bytes(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def send_error_body(self, error: ApiError) -> None:
        if self.command in ("POST", "PUT", "PATCH", "DELETE"):
            self.close_connection = True  # the request body may be unread: this connection cannot be reused
        body = {"type": "error", "v": 1, "code": error.code, "params": error.params, "detail": error.detail}
        if self.server.session.demo:
            body = demo_scrub(body)  # somebody else's computer: no folder of the server, no technical text
        self.send_json(body, error.status)

    def read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise invalid("content-length") from None
        if length < 0:
            raise invalid("content-length")
        if length > MAX_JSON:
            raise ApiError(413, "error.upload.too_large", {"max_mb": 1})
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (
            UnicodeDecodeError,
            ValueError,
            RecursionError,
        ) as e:  # RecursionError: lists or objects nested thousands deep
            raise invalid("body", str(e)[:200]) from e
        if not isinstance(body, dict):
            raise invalid("body", "a JSON object was expected")
        return body

    def host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.server.port}", f"localhost:{self.server.port}")

    def origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        return origin.lower() in (f"http://127.0.0.1:{self.server.port}", f"http://localhost:{self.server.port}")

    def cookie_ok(self) -> bool:
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == cookie_name(self.server.port) and hmac.compare_digest(value, self.server.token):
                return True
        return False

    def do_GET(self) -> None:
        self.dispatch("GET")

    def do_HEAD(self) -> None:
        self.dispatch("GET")

    def do_POST(self) -> None:
        self.dispatch("POST")

    def do_PUT(self) -> None:
        self.dispatch("PUT")

    def do_PATCH(self) -> None:
        self.dispatch("PATCH")

    def do_DELETE(self) -> None:
        self.dispatch("DELETE")

    def dispatch(self, method: str) -> None:
        self.server.last_request = time.monotonic()
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        try:
            if not self.host_ok() or (method != "GET" and not self.origin_ok()):
                raise ApiError(403, "error.request.forbidden")
            if path == "/":
                return self.page(query)
            if path.startswith(("/static/", "/i18n/")):
                return self.public_file(path)
            if not path.startswith("/api/"):
                raise ApiError(404, "error.not_found")
            if not self.cookie_ok():
                raise ApiError(403, "error.request.forbidden")
            for route_method, pattern, name in ROUTES:
                match = pattern.fullmatch(path)
                if match and route_method == method:
                    return getattr(self, name)(Request(self, match, query))
            if any(pattern.fullmatch(path) for _, pattern, _ in ROUTES):
                raise ApiError(405, "error.request.invalid", {"field": "method"})
            raise ApiError(404, "error.not_found")
        except ApiError as e:
            self.send_error_body(e)
        except InvalidSetting as e:
            self.send_error_body(invalid(e.field, e.detail))
        except (
            BrokenPipeError,
            ConnectionResetError,
            socket.timeout,
        ):  # the other end went away or stopped: not an error of ours
            raise
        except Exception as e:  # one bad request must not take the server down
            if not self.server.session.demo:
                print(f"error.internal: {type(e).__name__}: {e}", file=sys.stderr)
            self.send_error_body(ApiError(500, "error.internal", {}, f"{type(e).__name__}: {e}"))

    def page(self, query: Dict[str, List[str]]) -> None:
        token = (query.get("t") or [""])[0]
        if token and hmac.compare_digest(token, self.server.token):
            self.send_response(302)
            # The token goes from the address bar; what else was asked for stays (`?t=TOKEN&lang=de` was the page in English, with the language lost)
            rest = [(k, v) for k, values in query.items() if k != "t" for v in values]
            self.send_header("Location", "/" + (f"?{urlencode(rest)}" if rest else ""))
            # Behind a proxy that ends TLS (the hosted demo) the browser is on https: the cookie then says Secure
            secure = (
                "; Secure" if self.headers.get("X-Forwarded-Proto", "").split(",")[0].strip().lower() == "https" else ""
            )
            self.send_header(
                "Set-Cookie", f"{cookie_name(self.server.port)}={token}; HttpOnly; SameSite=Strict; Path=/{secure}"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        index = os.path.join(self.server.static_dir, "index.html")
        if os.path.isfile(index):
            return self.send_file(index, "text/html; charset=utf-8")
        self.send_bytes(200, PLACEHOLDER.encode("utf-8"), "text/html; charset=utf-8")

    def public_file(self, path: str) -> None:
        if path.startswith("/i18n/"):
            lang = unquote(path[len("/i18n/") :])
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,11}\.json", lang):
                raise ApiError(404, "error.not_found")
            target = os.path.join(self.server.i18n_dir, lang)
            if not os.path.isfile(target):
                raise ApiError(404, "error.not_found")
            return self.send_file(target, "application/json; charset=utf-8")
        relative = unquote(path[len("/static/") :])
        try:
            target = os.path.realpath(os.path.join(self.server.static_dir, relative))
            inside = os.path.commonpath([target, self.server.static_dir]) == self.server.static_dir
        except ValueError:  # a NUL in the name (%00): no such file, not an internal error
            raise ApiError(404, "error.not_found") from None
        if not inside or not os.path.isfile(target):
            raise ApiError(404, "error.not_found")
        kind = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind in ("application/javascript", "application/json"):
            kind += "; charset=utf-8"
        self.send_file(target, kind)

    def send_file(self, target: str, content_type: str, headers: Optional[Dict[str, str]] = None) -> None:
        with open(target, "rb") as f:
            body = f.read()
        self.send_bytes(200, body, content_type, {"Cache-Control": "no-cache", **(headers or {})})

    def get_state(self, req: Request) -> None:
        state = self.server.session.state()
        state["idle_stop"] = {"seconds": IDLE_SECONDS, "seconds_with_clips": IDLE_SECONDS_WITH_CLIPS}
        self.send_json(state)

    def get_settings(self, req: Request) -> None:
        self.send_json(self.server.session.public_settings())

    def put_settings(self, req: Request) -> None:
        settings = self.server.session.settings
        before = settings.get()
        values = settings.update(req.json())
        if values != before:
            # every other open window shows the change; the one that made it is named, so that it does not take it back as news
            by = "".join(c for c in (self.headers.get("X-Studio-Client") or "") if c.isalnum() or c == "-")[:40]
            self.server.hub.emit("settings.changed", settings=self.server.session.public_settings(values), by=by)
        if settings.save_failed and not self.server.told_settings_not_saved:
            self.server.told_settings_not_saved = True  # once: every later change would say it again
            self.server.hub.emit("job.warning", code="warning.settings.not_saved", clip_id=None, params={})
        self.send_json(self.server.session.public_settings(values))

    def get_clips(self, req: Request) -> None:
        session = self.server.session
        self.send_json({"clips": session.summaries()})

    def get_browse(self, req: Request) -> None:
        if self.server.session.demo:
            raise ApiError(403, "error.demo.disabled")
        path = req.arg("path")
        try:
            self.send_json(browse.listing(path))
        except NotADirectoryError:
            raise ApiError(404, "error.not_found", {"path": path}) from None
        except (
            PermissionError
        ):  # the person's own folder that the computer does not let them read: not a refused request
            raise ApiError(403, "error.browse.denied", {"path": path}) from None

    def get_samples(self, req: Request) -> None:
        samples = [{k: v for k, v in s.items() if not k.startswith("_")} for s in self.server.session.samples()]
        self.send_json(samples)

    def post_clips(self, req: Request) -> None:
        body = req.json()
        session = self.server.session
        only_keys(body, ("paths", "recursive", "samples"), "")
        if "samples" in body:
            samples = body["samples"]
            if not isinstance(samples, list) or not all(isinstance(s, str) for s in samples):
                raise invalid("samples")
            return self.send_json(session.import_samples(samples))
        paths = body.get("paths")
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p.strip() for p in paths):
            raise invalid("paths")  # an empty path would stand for the server's own working directory
        self.send_json(session.import_paths(paths, bool(body.get("recursive", True))))

    def put_upload(self, req: Request) -> None:
        session = self.server.session
        name = req.arg("name")
        if not name:
            raise invalid("name")
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            raise invalid("content-length") from None
        if length < 0:
            raise invalid("content-length")
        limit = DEMO_LIMITS["max_upload_mb"] * 1024 * 1024 if session.demo else shutil.disk_usage(session.work_dir).free
        if length > limit:
            raise ApiError(413, "error.upload.too_large", {"max_mb": limit // (1024 * 1024)})
        target = session.upload_target(name)
        remaining = length
        try:
            with open(target, "wb") as f:
                while remaining:
                    chunk = self.rfile.read(min(CHUNK, remaining))
                    if not chunk:
                        raise invalid("body", "the upload ended early")
                    f.write(chunk)
                    remaining -= len(chunk)
        except BaseException:
            if os.path.exists(target):
                os.remove(target)
            raise
        self.send_json(session.register_upload(target))

    def delete_clip(self, req: Request) -> None:
        self.server.session.delete_clip(req.groups[0])
        self.send_json({"ok": True})

    def get_source(self, req: Request) -> None:
        clip = self.server.session.clip(req.groups[0])
        try:
            size = os.path.getsize(clip.path)
            f = open(clip.path, "rb")
        except OSError:
            raise ApiError(404, "error.not_found", {"clip_id": clip.clip_id}) from None
        try:
            start, end = 0, size - 1
            status = 200
            spec = self.headers.get("Range")
            headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
            if spec:
                parsed = parse_range(spec, size)
                if parsed is None:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start, end = parsed
                status = 206
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            ext = os.path.splitext(clip.path)[1].lower()
            content_type = MEDIA_TYPES.get(ext) or mimetypes.guess_type(clip.path)[0] or "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(end - start + 1))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command == "HEAD":
                return
            f.seek(start)
            left = end - start + 1
            while left:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)
        finally:
            f.close()

    def get_thumb(self, req: Request) -> None:
        self.send_bytes(
            200, self.server.session.thumb(req.groups[0]), "image/jpeg", {"Cache-Control": "private, max-age=3600"}
        )

    def get_preview(self, req: Request) -> None:
        width = req.int_arg("w", 960)
        clip_id, frame, view = req.groups[0], req.int_arg("frame"), req.arg("view", "original") or "original"
        # What is drawn depends on the decisions (rev) and on the settings (blur style, strength, margin, gap filling): the browser
        # asks again with the tag it has and is told when nothing changed, so a change of setting is never shown with the old blur.
        tag = f'"{self.server.session.preview_tag(clip_id)}-{frame}-{view}-{width}"'
        if self.headers.get("If-None-Match") == tag:
            self.send_response(304)
            self.send_header("ETag", tag)
            self.send_header("Cache-Control", "private, no-cache")
            self.end_headers()
            return
        data = self.server.session.preview(clip_id, frame, view, width)
        self.send_bytes(200, data, "image/jpeg", {"Cache-Control": "private, no-cache", "ETag": tag})

    def get_boxes(self, req: Request) -> None:
        self.send_json(self.server.session.boxes(req.groups[0], req.int_arg("from"), req.int_arg("to")))

    def get_item_thumb(self, req: Request) -> None:
        data = self.server.session.item_thumb(req.groups[0], int(req.groups[1]))
        cache = (
            "private, max-age=3600" if req.arg("rev") is not None else "no-cache"
        )  # the address says which analysis it is of
        self.send_bytes(200, data, "image/jpeg", {"Cache-Control": cache})

    def get_analysis(self, req: Request) -> None:
        self.send_json(self.server.session.analysis_json(req.groups[0]))

    def get_analysis_file(self, req: Request) -> None:
        session = self.server.session
        path = session.analysis_file(req.groups[0])
        headers = {"Content-Disposition": 'attachment; filename="analysis.json"'}
        if not session.demo:
            return self.send_file(path, "application/json", headers)
        with open(
            path, encoding="utf-8"
        ) as f:  # the hosted demo is somebody else's computer: the file's name, not the server's folder
            data = json.load(f)
        if isinstance(data.get("source"), str):
            data["source"] = os.path.basename(data["source"])
        self.send_bytes(
            200,
            json.dumps(data, indent=1).encode("utf-8"),
            "application/json",
            {"Cache-Control": "no-cache", **headers},
        )

    def patch_item(self, req: Request) -> None:
        body = req.json()
        only_keys(body, ("blur",), "")
        self.send_json(self.server.session.set_item(req.groups[0], int(req.groups[1]), body.get("blur")))

    def post_doubtful(self, req: Request) -> None:
        body = req.json()
        only_keys(body, ("blur",), "")
        self.send_json(self.server.session.set_doubtful(req.groups[0], body.get("blur")))

    def post_area(self, req: Request) -> None:
        self.send_json(self.server.session.add_area(req.groups[0], req.json()), 201)

    def delete_area(self, req: Request) -> None:
        self.send_json(self.server.session.remove_area(req.groups[0], int(req.groups[1])))

    def get_report(self, req: Request) -> None:
        fmt = req.arg("format", "json") or "json"
        _, data = self.server.session.report(req.groups[0], fmt)
        kind = "application/json" if fmt == "json" else "text/csv; charset=utf-8"
        self.send_bytes(200, data, kind, {"Cache-Control": "no-store"})

    def post_analyze(self, req: Request) -> None:
        self.send_json(self.server.session.start_analyze(req.json()), 202)

    def post_job_command(self, req: Request) -> None:
        self.send_json(self.server.session.job_command(req.groups[0]))

    def post_plan(self, req: Request) -> None:
        self.send_json(self.server.session.plan(req.json()))

    def post_render(self, req: Request) -> None:
        self.send_json(self.server.session.start_render(req.json()), 202)

    def get_download(self, req: Request) -> None:
        entry = self.server.session.downloads.get(req.groups[0])
        if entry is None or not os.path.isfile(entry[0]):
            raise ApiError(404, "error.not_found")
        path, name = entry
        kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
        self.send_file(path, kind, {"Content-Disposition": content_disposition(name)})

    def post_quit(self, req: Request) -> None:
        self.send_json({"ok": True})
        self.server.stop_later()

    def get_events(self, req: Request) -> None:
        last = self.headers.get("Last-Event-ID")
        after = (
            int(last) if last and last.isascii() and last.isdigit() and len(last) <= 15 else None
        )  # a longer number is no event of ours
        events = self.server.hub.stream(
            after
        )  # subscribe before the headers go out: nothing emitted after them is lost
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.server.stream_opened(1)
        try:
            for item in events:
                if item is None:
                    self.wfile.write(b": ping\n\n")
                else:
                    self.wfile.write(f"id: {item[0]}\ndata: {item[1]}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass  # the page closed the stream
        finally:
            self.server.stream_opened(-1)
            self.server.last_request = time.monotonic()


def range_number(text: str) -> int:
    """The number of a Range header; more than 18 digits is bigger than any file (and than Python's own limit on integer strings)."""
    return int(text) if len(text) <= 18 else 10**18


def parse_range(spec: str, size: int) -> Optional[Tuple[int, int]]:
    """(first, last) bytes of a single `bytes=a-b`, `a-` or `-n` range, or None when it cannot be satisfied."""
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", spec.strip(), flags=re.ASCII)
    if not match or size == 0:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":
        count = range_number(last)
        if count == 0:
            return None
        return max(0, size - count), size - 1
    start = range_number(first)
    end = min(range_number(last), size - 1) if last else size - 1
    if start >= size or end < start:
        return None
    return start, end


ID = r"(c\d{1,9})"
ROUTES: List[Tuple[str, "re.Pattern[str]", str]] = [
    (method, re.compile(pattern), name)
    for method, pattern, name in [
        ("GET", r"/api/state", "get_state"),
        ("GET", r"/api/settings", "get_settings"),
        ("PUT", r"/api/settings", "put_settings"),
        ("GET", r"/api/clips", "get_clips"),
        ("POST", r"/api/clips", "post_clips"),
        ("GET", r"/api/browse", "get_browse"),
        ("GET", r"/api/samples", "get_samples"),
        ("PUT", r"/api/upload", "put_upload"),
        ("DELETE", rf"/api/clips/{ID}", "delete_clip"),
        ("GET", rf"/api/clips/{ID}/source", "get_source"),
        ("GET", rf"/api/clips/{ID}/thumb", "get_thumb"),
        ("GET", rf"/api/clips/{ID}/preview", "get_preview"),
        ("GET", rf"/api/clips/{ID}/boxes", "get_boxes"),
        ("GET", rf"/api/clips/{ID}/items/(\d{{1,9}})/thumb", "get_item_thumb"),
        ("GET", rf"/api/clips/{ID}/analysis", "get_analysis"),
        ("GET", rf"/api/clips/{ID}/analysis\.json", "get_analysis_file"),
        ("GET", rf"/api/clips/{ID}/report", "get_report"),
        ("PATCH", rf"/api/clips/{ID}/items/(\d{{1,9}})", "patch_item"),
        ("POST", rf"/api/clips/{ID}/doubtful", "post_doubtful"),
        ("POST", rf"/api/clips/{ID}/areas", "post_area"),
        ("DELETE", rf"/api/clips/{ID}/areas/(\d{{1,9}})", "delete_area"),
        ("POST", r"/api/job/analyze", "post_analyze"),
        ("POST", r"/api/job/(pause|resume|stop)", "post_job_command"),
        ("POST", r"/api/render/plan", "post_plan"),
        ("POST", r"/api/render", "post_render"),
        ("GET", r"/api/downloads/(f\d+)/[^/]+", "get_download"),
        ("GET", r"/api/events", "get_events"),
        ("POST", r"/api/quit", "post_quit"),
    ]
]


def make_server(
    *,
    demo: bool = False,
    port: int = 0,
    settings_folder: Optional[str] = None,
    static_dir: Optional[str] = None,
    i18n_dir: Optional[str] = None,
    modules: Optional[list] = None,
    models_dir: Optional[str] = None,
    demo_dir: Optional[str] = None,
    work_dir: Optional[str] = None,
    token: Optional[str] = None,
    find_browser: Optional[Callable[[], Optional[str]]] = None,
    engine: Optional[str] = None,
    engine_options: Optional[dict] = None,
) -> StudioServer:
    """A server ready to serve_forever(). Settings are saved in `settings_folder` (never in demo mode, and not at all
    when it is None). `modules` replaces the real detectors (tests)."""
    static_dir = static_dir or os.path.join(HERE, "static")
    i18n_dir = i18n_dir or os.path.join(HERE, "i18n")
    languages = languages_in(i18n_dir)
    settings = Settings(None if demo else settings_folder, languages)
    hub = Hub()
    extra: Dict[str, Any] = {"find_browser": find_browser} if find_browser else {}

    def texts_for(language: str) -> Dict[str, str]:
        """The page's texts in `language`, each missing key filled in from English."""
        merged: Dict[str, str] = {}
        for folder, name in ((os.path.join(HERE, "i18n"), "en"), (i18n_dir, "en"), (i18n_dir, language)):
            try:
                with open(os.path.join(folder, f"{name}.json"), encoding="utf-8") as f:
                    merged.update(json.load(f))
            except (OSError, ValueError):
                pass  # no such file or not readable: what is already there stays
        return merged

    session = Session(
        settings,
        hub,
        demo=demo,
        work_dir=work_dir,
        modules=modules,
        models_dir=models_dir,
        demo_dir=demo_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "demo"),
        languages=languages,
        texts_for=texts_for,
        engine=engine or ("thread" if modules is not None else "process"),
        engine_options=engine_options,
        **extra,
    )
    return StudioServer(session, hub, token or secrets.token_hex(16), static_dir, i18n_dir, port)


def watch_idle(server: StudioServer, seconds: Optional[float] = None) -> None:
    """Stop the server when the page has been gone for `seconds` (by default IDLE_SECONDS, or IDLE_SECONDS_WITH_CLIPS while clips
    are loaded) and nothing is running."""

    def loop() -> None:
        while True:
            time.sleep(min(5.0, (seconds or IDLE_SECONDS) / 4))
            if server.idle_expired(seconds):
                server.shutdown()
                return

    threading.Thread(target=loop, daemon=True).start()


def models_warning(models_dir: Optional[str]) -> Optional[str]:
    """One line for the terminal when no detection model is where the Studio will look for them, or None. The usual first-run mistake is to
    start the Studio from a folder with no `models/` (after `pip install` the models stay in the checkout): without this line nothing says
    so until Analyze fails with "a detection model could not be loaded"."""
    folder = contract.models_dir(models_dir)
    if glob.glob(os.path.join(glob.escape(folder), "*.onnx")):
        return None
    where = (
        folder if models_dir or os.environ.get("ANONYMIZER_MODELS") else ", ".join(contract._default_model_folders())
    )
    return (
        f"No detection models found (looked in: {where}), so Analyze cannot work yet. Fetch them with scripts/fetch_models.py from the checkout, "
        "then start the Studio from the checkout's folder, or set ANONYMIZER_MODELS to the folder that holds them (or pass --models FOLDER)."
    )


def display_available() -> bool:
    """Whether a window can be shown. On Linux that is a display (X11 or Wayland) named in the environment; elsewhere the system
    always has one."""
    if not sys.platform.startswith("linux"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def window_size(webview) -> tuple:
    """The size of the Studio's own window: 1360x860, and never more than the screen leaves once the system's bars and the window's
    title are taken off (a 1366x768 laptop is 100 px shorter than 860, and the Analyze button sat below the screen). Where the screen
    cannot be asked, 1280x720, which every screen of a laptop holds."""
    width, height = 1360, 860
    try:
        screen = webview.screens[0]
        width, height = min(width, int(screen.width) - 40), min(height, int(screen.height) - 120)
    except Exception:  # an older pywebview, or no screen to ask
        width, height = 1280, 720
    return max(width, 800), max(height, 560)


def serve(server: StudioServer, url: str, *, open_page: bool) -> None:
    """Serve until the Studio stops. With `open_page` the page is shown too: in a pywebview window when that is installed, else in
    the default browser. pywebview must run its window on the main thread, so then the server goes on a thread and the main thread
    runs the window; closing the window ends the Studio. Where the window cannot open (no GTK or Qt, no display) the page opens in
    the browser and says why: the Studio never goes on serving to nobody."""
    if not open_page:
        server.serve_forever()
        return
    try:
        import webview  # optional: not a dependency
    except ImportError:
        webbrowser.open(url)
        server.serve_forever()
        return
    if not display_available():
        # pywebview with GTK and no display neither raises nor returns: the Studio would serve to nobody and print nothing.
        print(
            "No display found (DISPLAY and WAYLAND_DISPLAY are not set), so the Studio's own window is not used: "
            "open the address above in a browser.",
            file=sys.stderr,
            flush=True,
        )
        webbrowser.open(url)
        server.serve_forever()
        return
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        width, height = window_size(webview)
        webview.create_window("Anonymizer Studio", url, width=width, height=height)
        webview.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:  # no toolkit or no display: pywebview says so in its own words
        print(
            f"The Studio's own window could not be opened ({e}). Opening the page in the browser; "
            "the Studio runs until you press Ctrl+C.",
            file=sys.stderr,
            flush=True,
        )
        webbrowser.open(url)
        try:
            serving.join()
        except KeyboardInterrupt:
            pass
    if serving.is_alive():  # the window was closed, or Ctrl+C: the Studio goes with it
        server.shutdown()
    serving.join()


def port_number(text: str) -> int:
    """A TCP port for --port: 0 (a free one) to 65535."""
    try:
        number = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError("a port is a number from 0 to 65535")
    return number


def stop_on_signals(server: StudioServer) -> None:
    """A terminal closed (SIGHUP) or `kill` (SIGTERM) ends the Studio the way Ctrl+C does, so that its work folder (the copies of
    what was found in the person's pictures) is deleted and not left in the temporary folder."""
    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, lambda *_: server.stop_later())
            except (ValueError, OSError):  # not the main thread, or not allowed here: the default stays
                pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m anonymizer.ui", description="Anonymizer Studio, on this computer only."
    )
    parser.add_argument("--version", action="version", version=f"Anonymizer Studio {anonymizer.__version__}")
    parser.add_argument("--demo", action="store_true", help="hosted-demo limits: nothing saved, temporary files only")
    parser.add_argument("--port", type=port_number, default=0, help="port on 127.0.0.1 (default: a free one)")
    parser.add_argument("--no-browser", action="store_true", help="print the address instead of opening it")
    parser.add_argument("--models", default=None, help="folder with the detection models")
    parser.add_argument(
        "--engine",
        choices=("process", "thread"),
        default="process",
        help="run the engine in a worker process that can be killed (default) or inside this one",
    )
    args = parser.parse_args(argv)
    remove_stale_work_folders()
    try:
        server = make_server(
            demo=args.demo, port=args.port, settings_folder=user_folder(), models_dir=args.models, engine=args.engine
        )
    except OSError as e:  # most often: the port is taken, because the Studio is already running
        print(
            f"Anonymizer Studio could not start on port {args.port}: {e.strerror or e}. If it is already running, use its address;"
            " otherwise start it on another port (--port N), or without --port to take a free one.",
            file=sys.stderr,
        )
        return 2
    stop_on_signals(server)  # before the address is printed: whoever reads it may stop the Studio at once
    url = server.url()
    print(f"Anonymizer Studio: {url}", flush=True)
    print(
        f"It stops by itself after {IDLE_SECONDS / 60:.0f} minutes without a page open "
        f"({IDLE_SECONDS_WITH_CLIPS / 3600:.0f} hours while clips are loaded); Ctrl+C stops it now.",
        flush=True,
    )
    warning = models_warning(args.models)
    if warning:
        print(warning, file=sys.stderr, flush=True)
    watch_idle(server)
    try:
        serve(server, url, open_page=not args.no_browser)
    except KeyboardInterrupt:
        pass
    finally:
        server.hub.close()
        server.session.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
