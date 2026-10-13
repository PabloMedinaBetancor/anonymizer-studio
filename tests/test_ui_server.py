"""The Studio server (docs/UI-PROTOCOL.md): security, files, import, analysis jobs, Review, rendering and the demo mode,
over real HTTP with stand-in detectors."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    import cv2
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.ui import hub as hub_module
    from anonymizer.ui import server as server_module
    from tests import media
    from tests.fakes import Delayed, Failing, FixedModule, plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

TOKEN = "studio-test-token"


class EventReader:
    """The page's event stream, read in a thread."""

    def __init__(self, port: int, last_event_id=None):
        self.events = []
        self.ids = []
        headers = {"Cookie": f"anon_session_{port}={TOKEN}"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        self.conn.request("GET", "/api/events", headers=headers)
        self.response = self.conn.getresponse()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        event_id = None
        try:
            for raw in self.response:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("id: "):
                    event_id = int(line[4:])
                elif line.startswith("data: "):
                    self.events.append(json.loads(line[6:]))
                    self.ids.append(event_id)
        except (OSError, ValueError):
            pass  # closed by the test

    def types(self):
        return [e["type"] for e in list(self.events)]

    def wait_for(self, kind: str, timeout: float = 60.0, after: int = 0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for e in list(self.events)[after:]:
                if e["type"] == kind:
                    return e
            time.sleep(0.02)
        raise AssertionError(f"no {kind} event in {timeout} s; got {self.types()}")

    def close(self):
        self.conn.close()


class StudioCase(unittest.TestCase):
    demo = False
    modules = None
    real = False  # True: the real detectors, loaded from $ANONYMIZER_MODELS, instead of the stand-ins
    engine = None  # "process" runs the engine in a worker; the stand-in detectors then come from worker_modules
    engine_options = None
    fixed_port = 0  # a test that restarts the server on the same port sets it
    browser = None  # the Chromium-based browser the server finds: none, unless a test brings a stand-in

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        for folder in ("static", "i18n", "work", "settings", "media"):
            os.makedirs(os.path.join(self.dir, folder))
        self.write("static/index.html", "<!doctype html><title>Studio</title>")
        self.write("static/app.js", "export const x = 1;")
        self.write("i18n/en.json", '{"app.title": "Anonymizer Studio"}')
        self.write("i18n/es.json", '{"app.title": "Anonymizer Studio"}')
        self.start_server()

    def write(self, relative: str, text: str) -> str:
        path = os.path.join(self.dir, relative)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def start_server(self):
        self.server = server_module.make_server(
            demo=self.demo,
            settings_folder=os.path.join(self.dir, "settings"),
            static_dir=os.path.join(self.dir, "static"),
            i18n_dir=os.path.join(self.dir, "i18n"),
            work_dir=os.path.join(self.dir, "work"),
            modules=None if (self.engine == "process" or self.real) else (self.modules or [plate_module()]),
            engine=self.engine,
            engine_options=self.engine_options,
            token=TOKEN,
            demo_dir=os.path.join(self.dir, "demo"),
            find_browser=lambda: self.browser,
            port=self.fixed_port,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.port = self.server.port

    def stop_server(self):
        self.server.hub.close()
        self.server.shutdown()
        self.server.session.close()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None, cookie=True, raw=None):
        h = dict(headers or {})
        if cookie:
            h.setdefault("Cookie", f"anon_session_{self.port}={TOKEN}")
        payload = raw
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            h.setdefault("Content-Type", "application/json")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            conn.request(method, path, body=payload, headers=h)
            response = conn.getresponse()
            data = response.read()
            return response.status, response, data
        finally:
            conn.close()

    def get(self, path, **kw):
        status, response, data = self.request("GET", path, **kw)
        return status, response, data

    def api(self, method, path, body=None, expect=200):
        status, response, data = self.request(method, path, body)
        self.assertEqual(status, expect, data[:300])
        return json.loads(data) if response.getheader("Content-Type", "").startswith("application/json") else data

    def make_clip(self, name="clip.mp4", **kw) -> str:
        path = os.path.join(self.dir, "media", name)
        media.moving_plate_clip(path, audio=kw.pop("audio", False), **kw)
        return path

    def import_clip(self, name="clip.mp4", **kw) -> str:
        path = self.make_clip(name, **kw)
        answer = self.api("POST", "/api/clips", {"paths": [path]})
        self.assertEqual(answer["rejected"], [])
        return answer["clips"][0]["clip_id"]

    def wait_idle(self, timeout=60.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.api("GET", "/api/state")["job"] is None:
                return
            time.sleep(0.05)
        raise AssertionError("the job did not end")

    def analyze(self, *clip_ids):
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": list(clip_ids)}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        return reader


class Security(StudioCase):
    def test_api_needs_the_session_cookie(self):
        self.assertEqual(self.get("/api/state", cookie=False)[0], 403)
        self.assertEqual(self.get("/api/state", headers={"Cookie": f"anon_session_{self.port}=nope"})[0], 403)
        status, _, data = self.get("/api/state", cookie=False)
        self.assertEqual(json.loads(data)["code"], "error.request.forbidden")
        self.assertEqual(self.get("/api/state")[0], 200)

    def test_the_token_in_the_address_becomes_a_cookie_and_leaves_the_address(self):
        status, response, _ = self.get(f"/?t={TOKEN}", cookie=False)
        self.assertEqual((status, response.getheader("Location")), (302, "/"))
        cookie = response.getheader("Set-Cookie")
        self.assertIn(f"anon_session_{self.port}={TOKEN}", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)
        status, response, _ = self.get("/?t=wrong", cookie=False)
        self.assertEqual(status, 200)
        self.assertIsNone(response.getheader("Set-Cookie"))

    def test_a_foreign_host_or_origin_is_refused(self):
        self.assertEqual(self.get("/api/state", headers={"Host": "evil.example:80"})[0], 403)
        self.assertEqual(self.get("/", headers={"Host": "evil.example"}, cookie=False)[0], 403)
        for origin in ("http://evil.example", "http://127.0.0.1:1"):
            status, _, _ = self.request("PUT", "/api/settings", {"blur": "solid"}, {"Origin": origin})
            self.assertEqual(status, 403)
        status, _, _ = self.request(
            "PUT", "/api/settings", {"blur": "solid"}, {"Origin": f"http://127.0.0.1:{self.port}"}
        )
        self.assertEqual(status, 200)

    def test_the_server_listens_on_loopback_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_malformed_requests_get_typed_errors(self):
        status, _, data = self.request("POST", "/api/clips", raw=b"{not json")
        self.assertEqual((status, json.loads(data)["code"]), (400, "error.request.invalid"))
        self.assertEqual(self.request("POST", "/api/clips", {"paths": "x"})[0], 400)
        self.assertEqual(self.get("/api/nothing")[0], 404)
        self.assertEqual(self.request("DELETE", "/api/state")[0], 405)
        self.assertEqual(self.get("/api/clips/c9/thumb")[0], 404)


class Files(StudioCase):
    def test_page_static_and_i18n_files(self):
        status, response, data = self.get("/", cookie=False)
        self.assertEqual((status, b"<title>Studio</title>" in data), (200, True))
        status, response, _ = self.get("/static/app.js", cookie=False)
        self.assertEqual(status, 200)
        self.assertIn("javascript", response.getheader("Content-Type"))
        status, _, data = self.get("/i18n/es.json", cookie=False)
        self.assertEqual((status, json.loads(data)["app.title"]), (200, "Anonymizer Studio"))
        self.assertEqual(self.get("/i18n/xx.json", cookie=False)[0], 404)
        self.assertEqual(self.get("/i18n/..%2fsecret.json", cookie=False)[0], 404)

    def test_static_cannot_leave_its_folder(self):
        self.write("secret.txt", "secret")
        for path in ("/static/../secret.txt", "/static/..%2fsecret.txt", "/static/%2e%2e/secret.txt"):
            self.assertEqual(self.get(path, cookie=False)[0], 404, path)

    def test_state_lists_languages_and_capabilities(self):
        state = self.api("GET", "/api/state")
        self.assertEqual(
            (state["mode"], state["languages"], state["clips"], state["job"]), ("local", ["en", "es"], [], None)
        )
        self.assertFalse(state["capabilities"]["pdf_report"]["available"])
        self.assertEqual(state["capabilities"]["pdf_report"]["reason"], "report.pdf.no_browser")
        self.assertTrue(state["capabilities"]["browse"])
        self.assertIsNone(state["demo_limits"])

    def test_settings_are_validated_saved_and_read_back_by_a_new_server(self):
        self.assertEqual(self.api("GET", "/api/settings")["blur"], "gaussian")
        saved = self.api("PUT", "/api/settings", {"blur": "pixelate", "strength": 1.5, "language": "es"})
        self.assertEqual((saved["blur"], saved["strength"], saved["language"]), ("pixelate", 1.5, "es"))
        for bad, field in (
            ({"strength": 0.5}, "strength"),
            ({"blur": "swirl"}, "blur"),
            ({"nope": 1}, "nope"),
            ({"faces": "yes"}, "faces"),
            ({"language": "xx"}, "language"),
        ):
            status, _, data = self.request("PUT", "/api/settings", bad)
            self.assertEqual((status, json.loads(data)["params"]["field"]), (400, field))
        self.assertEqual(self.api("GET", "/api/settings")["blur"], "pixelate")
        self.stop_server()
        self.start_server()
        self.assertEqual(self.api("GET", "/api/settings")["strength"], 1.5)


class Browsing(StudioCase):
    def test_listing_kinds_order_and_hidden_files(self):
        root = os.path.join(self.dir, "tree")
        os.makedirs(os.path.join(root, "Zeta"))
        os.makedirs(os.path.join(root, "alpha"))
        os.makedirs(os.path.join(root, ".hidden"))
        for name in ("b.mp4", "A.jpg", "IMG.HEIC", "notes.txt", ".secret.jpg"):
            with open(os.path.join(root, name), "wb") as f:
                f.write(b"xx")
        listing = self.api("GET", "/api/browse?path=" + root)
        self.assertEqual(listing["path"], root)
        self.assertEqual(listing["parent"], self.dir)
        self.assertEqual(
            [(e["name"], e["kind"]) for e in listing["entries"]],
            [("alpha", "dir"), ("Zeta", "dir"), ("A.jpg", "photo"), ("b.mp4", "video"), ("IMG.HEIC", "unsupported")],
        )
        self.assertEqual(listing["entries"][0]["items"], 0)
        self.assertFalse(listing["truncated"])

    def test_without_a_path_the_roots_are_listed_and_a_file_is_not_a_folder(self):
        listing = self.api("GET", "/api/browse")
        self.assertIsNone(listing["path"])
        self.assertTrue(listing["roots"])
        self.assertEqual(self.get("/api/browse?path=" + self.make_clip())[0], 404)
        self.assertEqual(self.get("/api/browse?path=/no/such/folder")[0], 404)

    def test_a_long_folder_is_cut_and_says_so(self):
        root = os.path.join(self.dir, "many")
        os.makedirs(root)
        for i in range(5):
            open(os.path.join(root, f"{i}.jpg"), "wb").close()
        with mock.patch.object(server_module.browse, "MAX_ENTRIES", 3):
            listing = self.api("GET", "/api/browse?path=" + root)
        self.assertEqual((len(listing["entries"]), listing["truncated"]), (3, True))


class Importing(StudioCase):
    def test_files_and_folders_are_imported_and_what_is_not_supported_is_listed(self):
        clip = self.make_clip("one.mp4")
        folder = os.path.join(self.dir, "batch")
        os.makedirs(os.path.join(folder, "sub"))
        os.makedirs(os.path.join(folder, "anonymized"))
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(os.path.join(folder, "p.png"))
        shutil.copy(clip, os.path.join(folder, "sub", "two.mp4"))
        shutil.copy(clip, os.path.join(folder, "anonymized", "already.mp4"))
        for name in ("x.heic", "broken.jpg"):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(b"not an image")
        answer = self.api("POST", "/api/clips", {"paths": [clip, folder, os.path.join(self.dir, "missing.mp4")]})
        names = sorted(c["name"] for c in answer["clips"])
        self.assertEqual(names, ["one.mp4", "p.png", "two.mp4"])
        rejected = {os.path.basename(r["path"]): r["code"] for r in answer["rejected"]}
        self.assertEqual(
            rejected,
            {
                "x.heic": "error.media.heic",
                "broken.jpg": "error.media.unreadable_photo",
                "missing.mp4": "error.not_found",
            },
        )
        video = next(c for c in answer["clips"] if c["name"] == "one.mp4")
        self.assertEqual(
            {
                k: video[k]
                for k in (
                    "media",
                    "origin",
                    "frames",
                    "width",
                    "height",
                    "fps",
                    "status",
                    "analysis",
                    "has_audio",
                    "source_dir",
                )
            },
            {
                "media": "video",
                "origin": "path",
                "frames": 60,
                "width": 320,
                "height": 240,
                "fps": 30.0,
                "status": "imported",
                "analysis": None,
                "has_audio": False,
                "source_dir": os.path.dirname(clip),
            },
        )
        self.assertEqual(next(c for c in answer["clips"] if c["media"] == "photo")["frames"], 1)
        again = self.api("POST", "/api/clips", {"paths": [clip]})
        self.assertEqual(again["clips"][0]["clip_id"], video["clip_id"])
        self.assertEqual(len(self.api("GET", "/api/clips")["clips"]), 3)

    def test_not_recursive_stays_in_the_folder(self):
        folder = os.path.join(self.dir, "flat")
        os.makedirs(os.path.join(folder, "sub"))
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(os.path.join(folder, "a.png"))
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(os.path.join(folder, "sub", "b.png"))
        answer = self.api("POST", "/api/clips", {"paths": [folder], "recursive": False})
        self.assertEqual([c["name"] for c in answer["clips"]], ["a.png"])

    def test_thumbnail_and_the_source_with_ranges(self):
        clip_id = self.import_clip()
        status, response, data = self.get(f"/api/clips/{clip_id}/thumb")
        self.assertEqual((status, response.getheader("Content-Type"), data[:3]), (200, "image/jpeg", b"\xff\xd8\xff"))
        path = self.server.session.clip(clip_id).path
        with open(path, "rb") as f:
            whole = f.read()
        status, response, data = self.get(f"/api/clips/{clip_id}/source")
        self.assertEqual((status, data, response.getheader("Accept-Ranges")), (200, whole, "bytes"))
        self.assertEqual(response.getheader("Content-Type"), "video/mp4")
        status, response, data = self.get(
            f"/api/clips/{clip_id}/source",
            headers={"Range": "bytes=10-109", "Cookie": f"anon_session_{self.port}={TOKEN}"},
        )
        self.assertEqual(
            (status, data, response.getheader("Content-Range")), (206, whole[10:110], f"bytes 10-109/{len(whole)}")
        )
        status, response, data = self.get(
            f"/api/clips/{clip_id}/source",
            headers={"Range": "bytes=-50", "Cookie": f"anon_session_{self.port}={TOKEN}"},
        )
        self.assertEqual((status, data), (206, whole[-50:]))
        status, response, data = self.get(
            f"/api/clips/{clip_id}/source",
            headers={"Range": f"bytes={len(whole) - 5}-", "Cookie": f"anon_session_{self.port}={TOKEN}"},
        )
        self.assertEqual((status, data), (206, whole[-5:]))
        status, response, _ = self.get(
            f"/api/clips/{clip_id}/source",
            headers={"Range": f"bytes={len(whole)}-", "Cookie": f"anon_session_{self.port}={TOKEN}"},
        )
        self.assertEqual((status, response.getheader("Content-Range")), (416, f"bytes */{len(whole)}"))

    def test_range_parsing(self):
        parse = server_module.parse_range
        self.assertEqual(parse("bytes=0-0", 10), (0, 0))
        self.assertEqual(parse("bytes=5-", 10), (5, 9))
        self.assertEqual(parse("bytes=5-99", 10), (5, 9))
        self.assertEqual(parse("bytes=-3", 10), (7, 9))
        self.assertEqual(parse("bytes=-30", 10), (0, 9))
        for bad in ("bytes=10-", "bytes=5-2", "bytes=-0", "bytes=-", "items=0-1", "bytes=0-1,3-4"):
            self.assertIsNone(parse(bad, 10), bad)
        self.assertIsNone(parse("bytes=0-1", 0))

    def test_upload_copies_the_bytes_and_refuses_unsupported_files(self):
        with open(self.make_clip(), "rb") as f:
            data = f.read()
        status, _, body = self.request("PUT", "/api/upload?name=dropped.mp4", raw=data)
        answer = json.loads(body)
        self.assertEqual(status, 200)
        [clip] = answer["clips"]
        self.assertEqual((clip["origin"], clip["source_dir"], clip["name"]), ("upload", None, "dropped.mp4"))
        with open(self.server.session.clip(clip["clip_id"]).path, "rb") as f:
            self.assertEqual(f.read(), data)
        status, _, body = self.request("PUT", "/api/upload?name=a.heic", raw=b"x" * 30)
        answer = json.loads(body)
        self.assertEqual((answer["clips"], answer["rejected"][0]["code"]), ([], "error.media.heic"))
        self.assertEqual(os.listdir(os.path.join(self.dir, "work", "uploads")), ["dropped.mp4"])
        self.assertEqual(self.request("PUT", "/api/upload", raw=b"x")[0], 400)

    def test_a_clip_can_be_removed_from_the_session(self):
        clip_id = self.import_clip()
        self.api("DELETE", f"/api/clips/{clip_id}")
        self.assertEqual(self.get(f"/api/clips/{clip_id}/thumb")[0], 404)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "media", "clip.mp4")))


class Analysis(StudioCase):
    def test_the_events_of_an_analysis_follow_the_protocol(self):
        clip_id = self.import_clip()
        reader = self.analyze(clip_id)
        types = reader.types()
        self.assertEqual(types[0], "job.started")
        self.assertEqual(types[-1], "job.finished")
        self.assertEqual(
            [t for t in types if t != "clip.progress"],
            ["job.started", "clip.started", "clip.item", "clip.analyzed", "job.finished"],
        )
        for event in reader.events:
            self.assertEqual(event["v"], 1)
        self.assertEqual(reader.ids, list(range(reader.ids[0], reader.ids[0] + len(reader.ids))))
        started = reader.events[0]
        self.assertEqual(
            (started["kind"], started["clips"][0]["clip_id"], started["clips"][0]["frames"]), ("analyze", clip_id, 60)
        )
        fractions = [e["fraction"] for e in reader.events if e["type"] == "clip.progress"]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)
        progress = next(e for e in reader.events if e["type"] == "clip.progress")
        self.assertEqual(
            {"clip_id", "phase", "fraction", "frame", "frames", "eta_s", "live", "found"}, set(progress) - {"type", "v"}
        )
        item = next(e for e in reader.events if e["type"] == "clip.item")["item"]
        self.assertEqual(
            sorted(item),
            sorted(
                [
                    "item_id",
                    "kind",
                    "judgement",
                    "first_frame",
                    "last_frame",
                    "first_s",
                    "last_s",
                    "frames",
                    "max_score",
                    "blur",
                    "found_by",
                    "second_score",
                    "decided_by",
                    "reviewed",
                    "manual",
                    "thumb_url",
                ]
            ),
        )
        self.assertEqual(
            (item["kind"], item["judgement"]["verdict"], item["blur"], item["reviewed"]), ("plate", "blur", True, False)
        )
        analyzed = next(e for e in reader.events if e["type"] == "clip.analyzed")
        self.assertEqual((analyzed["items"], analyzed["to_check"], analyzed["unchecked_frames"]), (1, 0, 0))
        finished = reader.events[-1]
        self.assertEqual(
            (finished["status"], finished["summary"]["analyzed"], finished["summary"]["failed"]), ("done", 1, 0)
        )
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "analyzed")

    def test_progress_carries_what_is_in_view_and_what_was_found_so_far(self):
        clip_id = self.import_clip()
        reader = self.analyze(clip_id)
        progress = [e for e in reader.events if e["type"] == "clip.progress"]
        live = [e for e in progress if "live" in e]
        self.assertTrue(live)
        self.assertEqual(live[-1]["found"], {"plate": 1})
        box = next(b for e in live for b in e["live"])
        self.assertEqual((box["kind"], len(box["box"]), box["state"] in ("found", "unsure")), ("plate", 4, True))
        self.assertNotIn("live", progress[-1])
        self.assertEqual(progress[-1]["fraction"], 1.0)

    def test_review_decisions_boxes_and_previews(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}"
        analysis = self.api("GET", f"{base}/analysis")
        [item] = analysis["items"]
        rev0 = analysis[
            "rev"
        ]  # the version counter starts at the server's start time (review 2, N7): only the steps count
        self.assertGreater(rev0, 10**12)
        self.assertEqual((analysis["to_check"], analysis["unchecked_frames"]), (0, 0))
        self.assertEqual(self.get(item["thumb_url"])[2][:3], b"\xff\xd8\xff")
        boxes = self.api("GET", f"{base}/boxes?from=28&to=32")
        self.assertEqual(sorted(boxes["frames"]), ["28", "29", "30", "31", "32"])
        [first] = boxes["frames"]["30"]
        self.assertEqual((first["item_id"], first["kind"], first["shape"]), (item["item_id"], "plate", "rect"))
        self.assertEqual(self.get(f"{base}/boxes?from=0&to=700")[0], 400)
        for view in ("original", "anonymized", "boxes", "split"):
            status, response, data = self.get(f"{base}/preview?frame=30&view={view}&w=200&rev=1")
            self.assertEqual((status, response.getheader("Content-Type")), (200, "image/jpeg"), view)
            self.assertEqual(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[1], 200)
        revalidated = self.get(f"{base}/preview?frame=0&view=original&rev=1")[1]
        self.assertEqual(revalidated.getheader("Cache-Control"), "private, no-cache")
        self.assertTrue(revalidated.getheader("ETag"))
        self.assertEqual(self.get(f"{base}/preview?frame=60&view=original")[0], 400)
        self.assertEqual(self.get(f"{base}/preview?view=original")[0], 400)
        self.assertEqual(self.get(f"{base}/preview?frame=0&view=sepia")[0], 400)

        changed = self.api("PATCH", f"{base}/items/{item['item_id']}", {"blur": False})
        self.assertEqual(
            (changed["item"]["blur"], changed["item"]["reviewed"], changed["rev"] - rev0), (False, True, 1)
        )
        self.assertEqual(self.api("GET", f"{base}/boxes?from=30&to=30")["frames"]["30"], [])
        self.assertEqual(self.request("PATCH", f"{base}/items/{item['item_id']}", {"blur": "no"})[0], 400)
        self.assertEqual(self.request("PATCH", f"{base}/items/99", {"blur": True})[0], 404)

        area = self.api(
            "POST", f"{base}/areas", {"frame_from": 10, "frame_to": 20, "box": [5, 5, 45, 25], "follow": "fixed"}, 201
        )
        self.assertEqual(
            (area["item"]["kind"], area["item"]["blur"], area["item"]["manual"], area["rev"] - rev0),
            ("manual", True, True, 2),
        )
        self.assertIsNotNone(area["item"]["thumb_url"])  # a picture of what the area covers (tests/test_ui_review21.py)
        self.assertEqual(sorted(area["boxes"], key=int), [str(t) for t in range(10, 21)])
        self.assertEqual(
            self.request("POST", f"{base}/areas", {"frame_from": 10, "frame_to": 20, "box": [5, 5, 5, 5]})[0], 400
        )
        self.assertEqual(
            self.request("POST", f"{base}/areas", {"frame_from": 10, "frame_to": 99, "box": [5, 5, 45, 25]})[0], 400
        )
        removed = self.api("DELETE", f"{base}/areas/{area['item']['item_id']}")
        self.assertEqual((removed["ok"], removed["rev"] - rev0), (True, 3))
        self.assertEqual(self.request("DELETE", f"{base}/areas/{item['item_id']}")[0], 404)

    def test_the_report_of_the_current_decisions_and_the_analysis_file(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}"
        report = self.api("GET", f"{base}/report?format=json")
        self.assertEqual(
            (report["version"], [(r["kind"], r["blurred"]) for r in report["items"]]), (2, [("plate", True)])
        )
        status, response, data = self.get(f"{base}/report?format=csv")
        self.assertEqual((status, data.decode("utf-8-sig").splitlines()[0].startswith("clip,item_id")), (200, True))
        status, response, data = self.get(f"{base}/analysis.json")
        self.assertEqual((status, json.loads(data)["media"]), (200, "video"))
        self.assertEqual(self.get(f"{base}/report?format=pdf")[0], 400)

    def test_review_routes_need_an_analysis(self):
        clip_id = self.import_clip()
        base = f"/api/clips/{clip_id}"
        for path in (
            f"{base}/analysis",
            f"{base}/boxes?from=0&to=1",
            f"{base}/preview?frame=0&view=boxes",
            f"{base}/report",
        ):
            status, _, data = self.get(path)
            self.assertEqual((status, json.loads(data)["code"]), (409, "error.clip.state"), path)
        self.assertEqual(self.get(f"{base}/preview?frame=0&view=original")[0], 200)
        self.assertEqual(self.request("PATCH", f"{base}/items/0", {"blur": True})[0], 409)

    def test_a_second_job_is_refused_while_one_runs_and_pause_resume_stop_act(self):
        self.stop_server()
        self.modules = [Delayed(plate_module(), 0.08)]
        self.start_server()
        clip_id = self.import_clip()
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, 202)
        status, _, data = self.request("POST", "/api/job/analyze", {"clips": [clip_id]})
        self.assertEqual((status, json.loads(data)["code"]), (409, "error.job.busy"))
        self.assertEqual(self.request("POST", "/api/clips", {"paths": [self.make_clip("x.mp4")]})[0], 409)
        self.assertEqual(self.api("GET", "/api/state")["job"]["kind"], "analyze")
        self.assertEqual(self.api("POST", "/api/job/pause")["state"], "paused")
        self.assertEqual(reader.wait_for("job.state")["state"], "paused")
        frozen = len([e for e in reader.events if e["type"] == "clip.progress"])
        time.sleep(0.4)
        self.assertLessEqual(len([e for e in reader.events if e["type"] == "clip.progress"]) - frozen, 1)
        self.assertEqual(self.api("POST", "/api/job/resume")["state"], "running")
        started = time.monotonic()
        self.assertEqual(self.api("POST", "/api/job/stop")["state"], "stopping")
        finished = reader.wait_for("job.finished")
        self.assertLess(time.monotonic() - started, 30)  # a hang guard; the job must end on its own
        self.assertEqual(finished["status"], "stopped")
        self.wait_idle()
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "imported")
        self.assertNotIn("clip.analyzed", reader.types())
        self.assertEqual(self.request("POST", "/api/job/stop")[0], 409)

    def test_analyzing_again_with_the_same_options_reuses_the_analysis(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        answer = self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, 202)
        self.assertEqual(answer["estimate_s"], 0)
        reader.wait_for("job.finished")
        self.assertNotIn("clip.progress", reader.types())
        self.assertIn("clip.analyzed", reader.types())

    def test_bad_analyze_requests(self):
        for body in ({}, {"clips": []}, {"clips": ["c9"]}, {"clips": "c1"}):
            self.assertIn(self.request("POST", "/api/job/analyze", body)[0], (400, 404), body)
        clip_id = self.import_clip()
        status, _, data = self.request(
            "POST", "/api/job/analyze", {"clips": [clip_id], "options": {"thoroughness": "extreme"}}
        )
        self.assertEqual((status, json.loads(data)["params"]["field"]), (400, "thoroughness"))

    def test_a_file_that_cannot_be_analysed_ends_in_a_job_error_and_the_job_goes_on(self):
        good = self.import_clip("good.mp4")
        bad = self.import_clip("bad.mp4")
        with open(self.server.session.clip(bad).path, "r+b") as f:
            f.seek(0)
            f.write(b"\0" * 4096)
        reader = self.analyze(bad, good)
        errors = [e for e in reader.events if e["type"] == "job.error"]
        self.assertEqual([(e["clip_id"], e["code"]) for e in errors][:1], [(bad, errors[0]["code"])])
        self.assertTrue(errors[0]["code"].startswith("error."))
        self.assertEqual(reader.events[-1]["summary"]["failed"], 1)
        states = {c["clip_id"]: c["status"] for c in self.api("GET", "/api/state")["clips"]}
        self.assertEqual((states[bad], states[good]), ("failed", "analyzed"))

    def test_a_crash_of_the_engine_is_reported_and_the_server_survives(self):
        clip_id = self.import_clip()
        with mock.patch.object(anonymizer.Anonymizer, "analyze", side_effect=RuntimeError("boom")):
            reader = self.analyze(clip_id)
        error = next(e for e in reader.events if e["type"] == "job.error")
        self.assertEqual((error["code"], "boom" in error["detail"]), ("error.engine.crashed", True))
        self.assertEqual(reader.events[-1]["status"], "failed")
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "failed")
        self.assertEqual(self.api("GET", "/api/state")["job"], None)


def worker_env(factory: str) -> dict:
    return {"ANONYMIZER_WORKER_MODULES": f"tests.fakes:{factory}"}


class WorkerCase(StudioCase):
    """The same flows with the engine in a worker process, which can also be killed or die."""

    engine = "process"
    factory = "plate_modules"
    watch = {"silent_after": 600.0, "watch_every": 30.0}

    def setUp(self):
        self.engine_options = {"env": worker_env(self.factory), **self.watch}
        super().setUp()

    def render_and_wait(self, clip_id, options):
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/render", {"clips": [clip_id], "options": options}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        return reader


class InWorkerProcess(WorkerCase):
    def test_analysis_review_render_and_comparison_work_through_the_worker(self):
        clip_id = self.import_clip()
        reader = self.analyze(clip_id)
        analyzed = next(e for e in reader.events if e["type"] == "clip.analyzed")
        self.assertEqual((analyzed["items"], analyzed["unchecked_frames"]), (1, 0))
        base = f"/api/clips/{clip_id}"
        [item] = self.api("GET", f"{base}/analysis")["items"]
        self.assertEqual(self.get(item["thumb_url"])[2][:3], b"\xff\xd8\xff")
        self.assertEqual(self.get(f"{base}/preview?frame=30&view=anonymized&rev=1")[0], 200)
        self.assertEqual(len(self.api("GET", f"{base}/boxes?from=30&to=30")["frames"]["30"]), 1)
        folder = os.path.join(self.dir, "out")
        reader = self.render_and_wait(clip_id, {"destination": folder, "comparison": True})
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        self.assertTrue(os.path.exists(rendered["outputs"][0]))
        self.assertTrue(os.path.exists(rendered["comparison"]))
        self.assertEqual(reader.events[-1]["status"], "done")
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "rendered")


class WorkerThatIsSlow(WorkerCase):
    factory = "slow_plate_modules"

    def test_stop_ends_the_job_through_the_worker(self):
        clip_id = self.import_clip()
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, 202)
        reader.wait_for("clip.progress")
        started = time.monotonic()
        self.api("POST", "/api/job/stop")
        finished = reader.wait_for("job.finished")
        self.assertLess(time.monotonic() - started, 60)  # a hang guard
        self.assertEqual(finished["status"], "stopped")
        self.wait_idle()
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "imported")

    def test_pause_and_resume_pass_through_and_the_job_finishes(self):
        clip_id = self.import_clip()
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, 202)
        reader.wait_for("clip.progress")
        self.assertEqual(self.api("POST", "/api/job/pause")["state"], "paused")
        time.sleep(2.5)
        before = len([e for e in reader.events if e["type"] == "clip.progress"])
        time.sleep(2.0)
        self.assertLessEqual(len([e for e in reader.events if e["type"] == "clip.progress"]) - before, 1)
        self.api("POST", "/api/job/resume")
        self.assertEqual(reader.wait_for("job.finished")["status"], "done")


class WorkerThatCrashes(WorkerCase):
    factory = "crashing_plate_modules"

    def test_a_worker_that_dies_costs_the_job_and_not_the_server(self):
        clip_id = self.import_clip()
        reader = self.analyze(clip_id)
        error = next(e for e in reader.events if e["type"] == "job.error")
        self.assertEqual(
            (error["code"], error["clip_id"], "exit code 3" in error["detail"]), ("error.engine.crashed", clip_id, True)
        )
        self.assertEqual(reader.events[-1]["status"], "failed")
        state = self.api("GET", "/api/state")
        self.assertEqual((state["clips"][0]["status"], state["job"]), ("failed", None))
        self.assertEqual(self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, 202)["clips"], [clip_id])
        self.wait_idle()


class WorkerThatHangs(WorkerCase):
    factory = "hanging_plate_modules"
    watch = {"silent_after": 1.5, "watch_every": 0.3}

    def test_a_silent_worker_is_killed_and_the_page_is_told_why(self):
        clip_id = self.import_clip()
        reader = self.analyze(clip_id)
        error = next(e for e in reader.events if e["type"] == "job.error")
        self.assertEqual((error["code"], error["clip_id"]), ("error.engine.silent", clip_id))
        self.assertEqual(reader.events[-1]["status"], "failed")
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "failed")


class Photos(StudioCase):
    def test_a_photo_is_a_clip_of_one_frame(self):
        noise = (np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)
        photo = os.path.join(self.dir, "media", "f.png")
        Image.fromarray(noise).save(photo)
        self.stop_server()
        self.modules = [
            FixedModule("face", [(140, 60, 200, 120)], 0.85, 0.75, placed=True),
            FixedModule("plate", [(20, 20, 100, 44)]),
        ]
        self.start_server()
        [clip] = self.api("POST", "/api/clips", {"paths": [photo]})["clips"]
        self.assertEqual((clip["media"], clip["frames"], clip["duration_s"], clip["fps"]), ("photo", 1, 0.0, 0.0))
        reader = self.analyze(clip["clip_id"])
        analyzed = next(e for e in reader.events if e["type"] == "clip.analyzed")
        self.assertEqual((analyzed["items"], analyzed["to_check"]), (2, 1))
        base = f"/api/clips/{clip['clip_id']}"
        face = next(i for i in self.api("GET", f"{base}/analysis")["items"] if i["kind"] == "face")
        self.assertEqual(
            (face["judgement"]["verdict"], face["blur"], face["first_frame"], face["last_frame"]),
            ("check", True, 0, 0),  # doubtful and blurred by default (privacy first)
        )
        decided = self.api("PATCH", f"{base}/items/{face['item_id']}", {"blur": True})
        self.assertEqual(decided["to_check"], 0)
        self.assertEqual(self.get(f"{base}/preview?frame=0&view=split")[0], 200)
        self.assertEqual(len(self.api("GET", f"{base}/boxes?from=0&to=0")["frames"]["0"]), 2)


class Rendering(StudioCase):
    def sha(self, path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def test_plan_names_and_estimates_then_render_writes_files_and_reports(self):
        clip_id = self.import_clip()
        source = self.server.session.clip(clip_id).path
        before = self.sha(source)
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        options = {
            "destination": folder,
            "name_pattern": "{name}_anonymized",
            "reports": {"csv": True, "json": True, "pdf": False},
        }
        plan = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": options})
        output = os.path.join(folder, "clip_anonymized.mp4")
        [entry] = plan["clips"]
        low, high = entry.pop("estimate_low_bytes"), entry.pop("estimate_bytes")
        self.assertTrue(0 < low < high, (low, high))  # a range (ui/export_size.py), not the size of the original
        self.assertEqual(entry, {"clip_id": clip_id, "output": output, "exists": False, "blocked": None, "to_check": 0})
        self.assertEqual(plan["total_bytes"], high)
        self.assertEqual(
            (plan["destination"], plan["writable"], plan["total_bytes"] > 0, plan["free_bytes"] > 0),
            (folder, True, True, True),
        )
        self.assertFalse(os.path.exists(folder))

        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        answer = self.api("POST", "/api/render", {"clips": [clip_id], "options": options}, 202)
        self.assertEqual(answer["outputs"], {clip_id: output})
        finished = reader.wait_for("job.finished")
        self.wait_idle()
        self.assertEqual((finished["status"], finished["summary"]["rendered"]), ("done", 1))
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        self.assertEqual(rendered["outputs"], [output])
        self.assertEqual(sorted(rendered["report"]), ["csv", "json"])
        self.assertTrue(all(os.path.exists(p) for p in [output, *rendered["report"].values()]))
        self.assertEqual(self.sha(source), before)
        progress = [e["fraction"] for e in reader.events if e["type"] == "clip.progress" and e["phase"] == "render"]
        self.assertEqual((progress == sorted(progress), progress[-1]), (True, 1.0))
        x0, y0, x1, y1 = (int(v) for v in media.plate_box(30))
        inside = (slice(y0 + 6, y1 - 6), slice(x0 + 6, x1 - 6))
        self.assertLess(
            media.decoded_frame(output, 30)[inside].std(), 0.4 * media.decoded_frame(source, 30)[inside].std()
        )
        with open(rendered["report"]["json"], encoding="utf-8") as f:
            self.assertEqual(json.load(f)["items"][0]["kind"], "plate")
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "rendered")

        again = self.api("POST", "/api/render/plan", {"clips": [clip_id], "options": options})
        self.assertEqual(
            (again["clips"][0]["output"], again["clips"][0]["exists"]),
            (os.path.join(folder, "clip_anonymized-2.mp4"), True),
        )
        self.api("POST", "/api/render", {"clips": [clip_id], "options": options}, 202)
        self.wait_idle()
        self.assertEqual(
            sorted(os.listdir(folder)),
            [
                "clip_anonymized-2.mp4",
                "clip_anonymized-2.report.csv",
                "clip_anonymized-2.report.json",
                "clip_anonymized.mp4",
                "clip_anonymized.report.csv",
                "clip_anonymized.report.json",
            ],
        )

    def test_the_default_destination_is_next_to_the_source_and_names_never_collide_in_one_plan(self):
        one = self.import_clip("one.mp4")
        two = self.import_clip("two.mp4")
        self.analyze(one, two)
        plan = self.api("POST", "/api/render/plan", {"clips": [one, two], "options": {"name_pattern": "same"}})
        outputs = [c["output"] for c in plan["clips"]]
        self.assertEqual(
            outputs,
            [
                os.path.join(self.dir, "media", "anonymized", "same.mp4"),
                os.path.join(self.dir, "media", "anonymized", "same-2.mp4"),
            ],
        )
        self.assertEqual([c["exists"] for c in plan["clips"]], [False, True])

    def test_bad_render_options(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        for options, field in (
            ({"name_pattern": "a/b"}, "name_pattern"),
            ({"name_pattern": "{nope}"}, "name_pattern"),
            ({"strength": 5}, "strength"),
            ({"reports": []}, "reports"),
        ):
            status, _, data = self.request("POST", "/api/render/plan", {"clips": [clip_id], "options": options})
            self.assertEqual((status, json.loads(data)["params"]["field"]), (400, field))

    def test_a_clip_with_frames_nobody_checked_is_blocked_and_never_rendered(self):
        self.stop_server()
        self.modules = [Failing(plate_module())]
        self.start_server()
        clip_id = self.import_clip()
        reader = self.analyze(clip_id)
        analyzed = next(e for e in reader.events if e["type"] == "clip.analyzed")
        self.assertGreater(analyzed["unchecked_frames"], 0)
        self.assertIn("warning.analysis.unchecked", [e.get("code") for e in reader.events])
        plan = self.api(
            "POST", "/api/render/plan", {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out")}}
        )
        blocked = plan["clips"][0]["blocked"]
        self.assertEqual((blocked["code"], blocked["params"]["frames"] > 0), ("error.analysis.incomplete", True))
        status, _, data = self.request(
            "POST", "/api/render", {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out")}}
        )
        self.assertEqual((status, json.loads(data)["code"]), (422, "error.analysis.incomplete"))
        self.assertFalse(os.path.exists(os.path.join(self.dir, "out")))
        self.assertEqual(
            self.api("GET", "/api/state")["clips"][0]["analysis"]["unchecked_frames"], analyzed["unchecked_frames"]
        )

    def test_a_changed_source_blocks_the_render(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        with open(self.server.session.clip(clip_id).path, "ab") as f:
            f.write(b"\0")
        plan = self.api(
            "POST", "/api/render/plan", {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out")}}
        )
        self.assertEqual(plan["clips"][0]["blocked"]["code"], "error.source.changed")

    def test_an_unwritable_destination_is_refused(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        blocker = self.write("file.txt", "x")
        status, _, data = self.request(
            "POST", "/api/render", {"clips": [clip_id], "options": {"destination": os.path.join(blocker, "sub")}}
        )
        self.assertEqual(status, 422)
        self.assertEqual(json.loads(data)["code"], "error.output.not_writable")

    def test_a_render_that_is_stopped_leaves_nothing_under_a_final_name(self):
        self.stop_server()
        self.start_server()
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        real = anonymizer.Anonymizer.render

        def slow(self_, analysis, target, on_event=None, cancel=None):
            def handler(event):
                time.sleep(0.01)
                on_event(event)

            return real(self_, analysis, target, handler, cancel)

        with mock.patch.object(anonymizer.Anonymizer, "render", slow):
            self.api("POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder}}, 202)
            reader.wait_for("clip.started")
            self.api("POST", "/api/job/stop")
            finished = reader.wait_for("job.finished")
            self.wait_idle()
        self.assertEqual(finished["status"], "stopped")
        self.assertEqual([n for n in os.listdir(folder) if not n.startswith(".")], [])
        self.assertEqual(self.api("GET", "/api/state")["clips"][0]["status"], "analyzed")

    def stand_in_browser(self, body: str) -> str:
        """A script that behaves like `chrome --headless --print-to-pdf=FILE URL`; it logs its arguments."""
        path = os.path.join(self.dir, "fake-browser")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\n" + f'echo "$@" >> "{self.dir}/browser-args.txt"\n' + body)
        os.chmod(path, 0o755)
        return path

    PRINT = 'for a in "$@"; do case "$a" in --print-to-pdf=*) out="${a#--print-to-pdf=}";; esac; done\n'

    def render_and_wait(self, clip_id, options):
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/render", {"clips": [clip_id], "options": options}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        return reader

    def test_a_comparison_video_is_written_next_to_the_output(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = self.render_and_wait(clip_id, {"destination": folder, "comparison": True})
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        comparison = os.path.join(folder, "clip_anonymized.compare.mp4")
        self.assertEqual(rendered["comparison"], comparison)
        phases = {e["phase"] for e in reader.events if e["type"] == "clip.progress"}
        self.assertEqual(phases, {"render", "compare"})
        wide = media.probe_size(comparison)
        self.assertEqual(wide, (640, 240))
        self.assertNotIn("job.error", reader.types())
        again = self.render_and_wait(clip_id, {"destination": folder, "comparison": True})
        self.assertTrue(
            next(e for e in again.events if e["type"] == "clip.rendered")["comparison"].endswith("-2.compare.mp4")
        )

    def test_a_photo_gets_no_comparison_and_a_warning_says_so(self):
        photo = os.path.join(self.dir, "media", "p.png")
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(photo)
        [clip] = self.api("POST", "/api/clips", {"paths": [photo]})["clips"]
        self.analyze(clip["clip_id"])
        reader = self.render_and_wait(
            clip["clip_id"], {"destination": os.path.join(self.dir, "out"), "comparison": True}
        )
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        self.assertNotIn("comparison", rendered)
        self.assertIn("comparison.photo_skipped", [e.get("code") for e in reader.events])

    def test_the_pdf_report_is_printed_by_the_browser_on_the_computer(self):
        self.browser = self.stand_in_browser(
            self.PRINT + 'printf "%%PDF-1.4\\n%s\\n" "$(date)" > "$out"; head -c 200 /dev/zero >> "$out"\n'
        )
        self.assertTrue(self.api("GET", "/api/state")["capabilities"]["pdf_report"]["available"])
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = self.render_and_wait(
            clip_id, {"destination": folder, "reports": {"pdf": True, "csv": False, "json": False}}
        )
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        pdf = os.path.join(folder, "clip_anonymized.report.pdf")
        self.assertEqual(rendered["report"], {"pdf": pdf})
        with open(pdf, "rb") as f:
            self.assertEqual(f.read(5), b"%PDF-")
        with open(os.path.join(self.dir, "browser-args.txt"), encoding="utf-8") as f:
            args = f.read().split()
        self.assertIn("--headless=new", args)
        self.assertTrue(any(a.startswith("--print-to-pdf=") for a in args))
        profile = next(a for a in args if a.startswith("--user-data-dir=")).split("=", 1)[1]
        self.assertFalse(os.path.exists(profile))
        self.assertTrue(args[-1].startswith("file://"))
        self.assertEqual([n for n in os.listdir(os.path.join(self.dir, "work")) if n.endswith(".html")], [])
        self.assertNotIn("report.pdf.unavailable", [e.get("code") for e in reader.events])

    def test_without_a_browser_or_with_one_that_fails_the_pdf_is_a_warning_and_the_render_stays(self):
        self.assertEqual(
            self.api("GET", "/api/state")["capabilities"]["pdf_report"],
            {"available": False, "reason": "report.pdf.no_browser", "params": {}},
        )
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        options = {"destination": folder, "reports": {"pdf": True, "csv": True, "json": False}}
        reader = self.render_and_wait(clip_id, options)
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        self.assertEqual(sorted(rendered["report"]), ["csv"])
        self.assertIn("report.pdf.unavailable", [e.get("code") for e in reader.events])
        self.browser = self.stand_in_browser("exit 1\n")
        reader = self.render_and_wait(clip_id, options)
        self.assertIn("report.pdf.unavailable", [e.get("code") for e in reader.events])
        self.assertEqual(reader.events[-1]["status"], "done")
        self.assertFalse([n for n in os.listdir(folder) if n.endswith(".pdf")])


class DemoMode(StudioCase):
    demo = True

    def test_demo_has_limits_no_browsing_and_no_saved_settings(self):
        state = self.api("GET", "/api/state")
        self.assertEqual(
            (state["mode"], state["demo_limits"]["max_seconds"], state["capabilities"]["browse"]), ("demo", 20, False)
        )
        self.assertEqual(self.get("/api/browse")[0], 403)
        self.assertEqual(self.request("POST", "/api/clips", {"paths": [self.make_clip()]})[0], 403)
        self.api("PUT", "/api/settings", {"blur": "solid"})
        self.assertEqual(os.listdir(os.path.join(self.dir, "settings")), [])

    def test_uploads_over_the_limits_are_refused_and_deleted(self):
        with open(self.make_clip(n=60), "rb") as f:
            data = f.read()
        with mock.patch.dict(server_module.DEMO_LIMITS, {"max_seconds": 1}):
            status, _, body = self.request("PUT", "/api/upload?name=long.mp4", raw=data)
        self.assertEqual((status, json.loads(body)["params"]["limit"]), (422, "seconds"))
        with mock.patch.dict(server_module.DEMO_LIMITS, {"max_height": 100}):
            status, _, body = self.request("PUT", "/api/upload?name=tall.mp4", raw=data)
        self.assertEqual((status, json.loads(body)["params"]["limit"]), (422, "height"))
        with mock.patch.dict(server_module.DEMO_LIMITS, {"max_upload_mb": 0}):
            status, _, body = self.request("PUT", "/api/upload?name=big.mp4", raw=data)
        self.assertEqual((status, json.loads(body)["code"]), (413, "error.upload.too_large"))
        self.assertEqual(os.listdir(os.path.join(self.dir, "work", "uploads")), [])
        photo = os.path.join(self.dir, "media", "p.png")
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(photo)
        with open(photo, "rb") as f:
            png = f.read()
        with mock.patch.dict(server_module.DEMO_LIMITS, {"max_photos": 1}):
            self.assertEqual(self.request("PUT", "/api/upload?name=a.png", raw=png)[0], 200)
            self.assertEqual(self.request("PUT", "/api/upload?name=b.png", raw=png)[0], 422)

    def test_results_are_downloads_in_a_temporary_folder(self):
        with open(self.make_clip(), "rb") as f:
            [clip] = json.loads(self.request("PUT", "/api/upload?name=d.mp4", raw=f.read())[2])["clips"]
        reader = self.analyze(clip["clip_id"])
        self.api("POST", "/api/render", {"clips": [clip["clip_id"]]}, 202)
        rendered = reader.wait_for("clip.rendered")
        self.wait_idle()
        [url] = rendered["outputs"]
        self.assertTrue(url.startswith("/api/downloads/"), url)
        status, response, data = self.get(url)
        self.assertEqual(status, 200)
        self.assertIn("attachment", response.getheader("Content-Disposition"))
        self.assertGreater(len(data), 1000)
        out = os.path.join(self.dir, "work", "out")
        self.assertTrue(os.path.isdir(out))
        self.assertEqual(self.get("/api/downloads/f99/x.mp4")[0], 404)

    def test_samples_are_the_public_clips_listed_with_their_licence(self):
        demo = os.path.join(self.dir, "demo")
        os.makedirs(demo)
        clip = os.path.join(demo, "walk.mp4")
        media.moving_plate_clip(clip, n=30, audio=False)
        self.write(
            "demo/SOURCES.csv",
            "file,kind,source_url,author,licence\nwalk.mp4,video,https://example.org/w,Someone,CC BY 4.0\nmissing.mp4,video,x,y,z\n",
        )
        [sample] = self.api("GET", "/api/samples")
        self.assertEqual(
            (sample["sample_id"], sample["licence"], sample["author"], sample["media"]),
            ("walk", "CC BY 4.0", "Someone", "video"),
        )
        answer = self.api("POST", "/api/clips", {"samples": ["walk"]})
        self.assertEqual((answer["clips"][0]["origin"], answer["clips"][0]["name"]), ("upload", "walk.mp4"))
        self.assertEqual(self.request("POST", "/api/clips", {"samples": ["nope"]})[0], 404)

    def test_the_shipped_demo_folder_lists_and_imports(self):
        shipped = os.path.join(os.environ["ANONYMIZER_ROOT"], "demo") if "ANONYMIZER_ROOT" in os.environ else None
        shipped = (
            shipped if shipped and os.path.isdir(shipped) else os.path.join(os.path.dirname(__file__), "..", "demo")
        )
        shutil.copytree(shipped, os.path.join(self.dir, "demo"))
        samples = self.api("GET", "/api/samples")
        self.assertEqual(
            sorted(s["sample_id"] for s in samples),
            ["street-hongkong", "street-madrid", "street-taipei", "street-westminster"],
        )
        for sample in samples:
            self.assertTrue(sample["licence"].startswith("CC"), sample)
            self.assertTrue(
                sample["author"] and sample["source_url"].startswith("https://commons.wikimedia.org/"), sample
            )
        by_id = {s["sample_id"]: s for s in samples}
        self.assertEqual((by_id["street-madrid"]["media"], by_id["street-taipei"]["media"]), ("video", "photo"))
        self.assertTrue(10 <= by_id["street-madrid"]["duration_s"] <= 20)
        answer = self.api("POST", "/api/clips", {"samples": ["street-madrid", "street-taipei"]})
        self.assertEqual([c["name"] for c in answer["clips"]], ["street-madrid.mp4", "street-taipei.jpg"])


class Lifetime(StudioCase):
    def test_quit_stops_the_server(self):
        self.api("POST", "/api/quit")
        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())

    def test_idle_time_counts_open_streams_and_jobs(self):
        self.server.last_request = time.monotonic() - 100
        self.assertGreater(self.server.idle_for(), 99)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        end = time.monotonic() + 5
        while not self.server.streams and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual(self.server.idle_for(), 0.0)


class EventHub(unittest.TestCase):
    def test_events_are_numbered_and_replayed_after_a_given_id(self):
        hub = hub_module.Hub()
        for n in range(5):
            hub.emit("clip.progress", n=n)
        replay = []
        for item in hub.stream(2, ping=0.01):
            if item is None:
                break
            replay.append(item[0])
        self.assertEqual(replay, [3, 4, 5])
        self.assertEqual(hub_module.event_dict(next(iter(hub.stream(4, ping=0.01)))[1])["n"], 4)

    def test_an_id_older_than_the_ring_or_from_a_previous_run_gives_a_resync(self):
        hub = hub_module.Hub(ring=3)
        for n in range(10):
            hub.emit("clip.progress", n=n)
        first = next(iter(hub.stream(2, ping=0.01)))
        self.assertEqual(json.loads(first[1])["type"], "state.resync")
        first = next(iter(hub.stream(500, ping=0.01)))
        self.assertEqual(json.loads(first[1])["type"], "state.resync")
        again = hub.stream(8, ping=0.01)
        self.assertEqual([i[0] for i in (next(again), next(again))], [9, 10])

    def test_a_stream_without_an_id_starts_at_the_present_and_pings(self):
        hub = hub_module.Hub()
        hub.emit("a")
        stream = hub.stream(None, ping=0.01)
        self.assertIsNone(next(stream))
        hub.emit("b")
        self.assertEqual(json.loads(next(stream)[1])["type"], "b")


class ReplayOverHttp(StudioCase):
    def test_last_event_id_replays_what_came_after(self):
        for n in range(4):
            self.server.hub.emit("job.warning", code="x", params={"n": n})
        reader = EventReader(self.port, last_event_id=2)
        self.addCleanup(reader.close)
        reader.wait_for("job.warning")
        end = time.monotonic() + 30
        while len(reader.events) < 2 and time.monotonic() < end:
            time.sleep(0.02)
        self.assertEqual([e["params"]["n"] for e in reader.events], [2, 3])
        stale = EventReader(self.port, last_event_id=999)
        self.addCleanup(stale.close)
        self.assertEqual(stale.wait_for("state.resync")["type"], "state.resync")


if __name__ == "__main__":
    unittest.main()
