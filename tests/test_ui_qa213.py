"""Studio v2.1.3: what the fourth client QA pass (04/10/2026, qa-anon-4) found at the server. One test class per bug; each fails on
v2.1.2 and passes here."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests import test_ui_server as server_tests
    from tests.fakes import FACE, FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class QaCase(server_tests.StudioCase):
    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def photo(self, name="ok.jpg", size=(160, 120), folder="media") -> str:
        os.makedirs(os.path.join(self.dir, folder), exist_ok=True)
        path = os.path.join(self.dir, folder, name)
        Image.fromarray((np.random.RandomState(7).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path


class TwoImportsOfOneFolder(QaCase):
    """Bug 1: two imports of the same folder at the same moment (a double drop, or a second window dropping it) both found its files
    new and each added them: every photo was listed two or three times, and exported as `name_anonymized`, `name_anonymized-2`..."""

    def test_the_same_folder_imported_three_times_at_once_lists_each_file_once(self):
        for n in range(6):
            self.photo(f"p{n}.jpg", folder="media/trip")
        folder = os.path.join(self.dir, "media", "trip")
        answers = []

        def drop():
            status, _, data = self.request("POST", "/api/clips", {"paths": [folder]})
            answers.append((status, json.loads(data)))

        threads = [threading.Thread(target=drop) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual([a[0] for a in answers], [200, 200, 200])
        listed = self.api("GET", "/api/clips")["clips"]
        self.assertEqual(sorted(c["name"] for c in listed), [f"p{n}.jpg" for n in range(6)])
        ids = {a[1]["clips"][i]["clip_id"] for a in answers for i in range(6)}
        self.assertEqual(len(ids), 6)  # each answer names the same six clips


class RequestsThatAreNotAnInternalError(QaCase):
    """Bug 2: a request with a number of thousands of digits in its address (an item number), a frame such as 1e400 and a body of lists
    nested thousands deep each ended in "The local server hit an internal error" (500) instead of saying what was wrong."""

    def setUp(self):
        super().setUp()
        self.clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(self.clip)

    def test_an_absurd_item_number_is_not_found(self):
        big = "9" * 5000
        for method, path in (
            ("GET", f"/api/clips/{self.clip}/items/{big}/thumb"),
            ("PATCH", f"/api/clips/{self.clip}/items/{big}"),
            ("DELETE", f"/api/clips/{self.clip}/areas/{big}"),
        ):
            status, _, data = self.request(method, path, {"blur": True} if method == "PATCH" else None)
            self.assertEqual(status, 404, (method, data[:200]))

    def test_a_frame_that_is_not_a_number_of_a_frame_is_a_400(self):
        base = f"/api/clips/{self.clip}/areas"
        for frame in ("1e400", "-1e400", "NaN", '"x"', "null", "true", "[]", "{}"):
            raw = f'{{"frame_from": {frame}, "frame_to": 0, "box": [0, 0, 10, 10]}}'.encode()
            status, _, data = self.request("POST", base, raw=raw, headers={"Content-Type": "application/json"})
            self.assertEqual(status, 400, (frame, data[:200]))
            self.assertEqual(json.loads(data)["params"], {"field": "frame_from"}, frame)

    def test_a_name_with_a_nul_in_the_static_path_is_not_found(self):
        for path in ("/static/js/main.js%00.png", "/static/%00", "/i18n/en%00.json"):
            status, _, data = self.request("GET", path, cookie=False)
            self.assertEqual(status, 404, (path, data[:200]))

    def test_a_body_nested_thousands_deep_is_a_400(self):
        raw = b'{"a":' * 3000 + b"1" + b"}" * 3000
        status, _, data = self.request("POST", "/api/clips", raw=raw, headers={"Content-Type": "application/json"})
        self.assertEqual((status, json.loads(data)["params"]), (400, {"field": "body"}))
        raw = b'{"paths":' + b"[" * 3000 + b"]" * 3000 + b"}"
        status, _, data = self.request("POST", "/api/clips", raw=raw, headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400, data[:200])


class AreaErrorsNameTheFieldAtFault(QaCase):
    """Bug 3: adding an area with frames out of range, or a `follow` that does not exist, was refused naming the field "box" (the only
    field the message of the check was not about), so a script that read the field blamed the wrong value."""

    def test_each_refusal_names_its_own_field(self):
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip)
        base = f"/api/clips/{clip}/areas"
        good = {"frame_from": 0, "frame_to": 0, "box": [5, 5, 40, 40]}
        cases = [
            ({"frame_from": 3}, "frame_from"),  # a photo has frame 0 only
            ({"frame_from": -1}, "frame_from"),
            ({"frame_to": 4}, "frame_to"),
            ({"follow": "sideways"}, "follow"),
            ({"box": [5, 5, 5, 5]}, "box"),
            ({"box": [1, 2, 3]}, "box"),
        ]
        for change, field in cases:
            body = dict(good, **change)
            status, _, data = self.request("POST", base, body)
            self.assertEqual(status, 400, (change, data[:200]))
            self.assertEqual(json.loads(data)["params"], {"field": field}, change)
        missing = {k: v for k, v in good.items() if k != "frame_to"}
        status, _, data = self.request("POST", base, missing)
        self.assertEqual((status, json.loads(data)["params"]), (400, {"field": "frame_to"}))
        self.api("POST", base, good, expect=201)


class PdfWithALongTempFolder(unittest.TestCase):
    """Bug 4: the Chromium that prints the PDF report makes a socket inside the folder for temporary files and aborted ("Socket path too
    long", with a crash dump left in the browser's data) when that folder's path was long (a sandbox, a deep TMPDIR): no report."""

    def test_the_browser_gets_a_short_folder_for_its_temporary_files(self):
        from unittest import mock

        from anonymizer.ui import pdf

        if os.name != "posix":
            self.skipTest("the limit is that of a Unix socket path")
        long_folder = "/" + "/".join(["deep"] * 20)
        with mock.patch.dict(os.environ, {"TMPDIR": long_folder}):
            self.assertIn(pdf.browser_environment().get("TMPDIR"), ("/tmp", "/var/tmp"))
        with mock.patch.dict(os.environ, {"TMPDIR": "/tmp/short"}):
            self.assertEqual(pdf.browser_environment().get("TMPDIR"), "/tmp/short")

    def test_html_to_pdf_runs_the_browser_in_that_environment(self):
        from unittest import mock

        from anonymizer.ui import pdf

        seen = {}

        def fake_run(command, **kwargs):
            seen.update(kwargs)
            target = next(c for c in command if c.startswith("--print-to-pdf=")).split("=", 1)[1]
            with open(target, "wb") as f:
                f.write(b"%PDF-1.4 " + b"x" * 200)
            return mock.Mock(returncode=0, stderr=b"")

        with tempfile.TemporaryDirectory() as folder, mock.patch("subprocess.run", fake_run):
            page = os.path.join(folder, "r.html")
            open(page, "w").write("<p>x</p>")
            pdf.html_to_pdf("chrome", page, os.path.join(folder, "r.pdf"))
        self.assertIn("env", seen)
        self.assertEqual(seen["env"], pdf.browser_environment())

    def test_a_real_browser_prints_with_a_long_temp_folder(self):
        from unittest import mock

        from anonymizer.ui import pdf

        browser = pdf.find_browser()
        if os.environ.get("ANONYMIZER_CHROME") != "1" or browser is None:
            self.skipTest("set ANONYMIZER_CHROME=1 to print with a real browser")
        with tempfile.TemporaryDirectory() as folder:
            deep = os.path.join(folder, *["a-long-folder-name"] * 6)
            os.makedirs(deep)
            page = os.path.join(folder, "r.html")
            open(page, "w").write("<p>x</p>")
            out = os.path.join(folder, "r.pdf")
            home = os.path.join(folder, "home")
            os.makedirs(home)
            with mock.patch.dict(
                os.environ, {"TMPDIR": deep, "HOME": home, "XDG_CONFIG_HOME": os.path.join(home, "cfg")}
            ):
                pdf.html_to_pdf(browser, page, out)
            self.assertTrue(open(out, "rb").read(5) == b"%PDF-")
            crashes = [r for r, _, files in os.walk(home) for f in files if f.endswith(".dmp")]
            self.assertEqual(crashes, [])


class NamePatternsThatGiveNoName(QaCase):
    """Bug 5: a file name pattern of "." (or "..") was accepted by the server and gave the names "..jpg" and "...jpg" (a hidden file,
    no name); and one with a character Windows cannot save (: " | ? * < >) was accepted and failed later as "not writable"."""

    def setUp(self):
        super().setUp()
        self.clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(self.clip)

    def plan(self, pattern):
        return self.request("POST", "/api/render/plan", {"clips": [self.clip], "options": {"name_pattern": pattern}})

    def test_names_of_dots_and_spaces_and_unsaveable_characters_are_refused(self):
        for pattern in (".", "..", " . ", "{name}: x", 'a"b', "x|y", "q?", "<{name}>", "{name}*"):
            status, _, data = self.plan(pattern)
            self.assertEqual(status, 400, (pattern, data[:200]))
            self.assertEqual(json.loads(data)["params"], {"field": "name_pattern"}, pattern)

    def test_the_patterns_that_name_files_still_work(self):
        for pattern in ("{name}_anonymized", "clip_{index:03d}", "{index}", "x.{name}", "{name} (blurred)"):
            status, _, data = self.plan(pattern)
            self.assertEqual(status, 200, (pattern, data[:200]))


class ClipThumbnailAfterARestartOnTheSamePort(QaCase):
    """Bug 6: the picture of a clip in the list was cached by the browser for an hour under /api/clips/c1/thumb. A Studio started again
    on the same port (a fixed --port, or a bookmark) gives c1 to another file, and the list showed the photos of the last run."""

    def test_the_address_of_the_picture_of_c1_is_not_the_one_of_the_last_run(self):
        def first_clip():
            path = self.photo()
            return self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]

        before = first_clip()
        self.stop_server()
        self.fixed_port = self.port
        self.start_server()
        after = first_clip()
        self.assertEqual((before["clip_id"], after["clip_id"]), ("c1", "c1"))
        self.assertNotEqual(before["thumb_url"], after["thumb_url"])
        status, response, data = self.get(after["thumb_url"])
        self.assertEqual((status, data[:3]), (200, b"\xff\xd8\xff"))


class VersionNumberIsTheReleasesOne(unittest.TestCase):
    """Bug 7: the package, `pip list` and the Studio's `GET /api/state` said "0.1.0.dev0" for a product released as v2.1.x."""

    def test_the_package_the_metadata_and_the_changelog_name_one_version(self):
        import pathlib
        import re

        import anonymizer

        root = pathlib.Path(__file__).resolve().parents[1]
        project = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
        # Keep a Changelog headings: ## [2.2.0] - 2026-10-06
        newest = re.search(r"^## \[(\d+\.\d+\.\d+)\]", (root / "CHANGELOG.md").read_text(), re.M).group(1)
        self.assertEqual((anonymizer.__version__, project), (newest, newest))
        self.assertNotIn("dev", anonymizer.__version__)


class PdfHeaderCellsStayOnOneLine(unittest.TestCase):
    """Bug 11: with a long reason in the table, the narrow columns of the PDF report gave up width and a header such as the Chinese
    "条目" (two characters) was printed on two lines."""

    def rows(self):
        import anonymizer

        return [anonymizer.ReportRow("a.mp4", 0, "face", "blur", True, "face.reason.verified", 1.0, 3.25, 82, 0.93)]

    def texts(self):
        import pathlib

        folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"
        merged = json.loads((folder / "en.json").read_text(encoding="utf-8"))
        merged.update(json.loads((folder / "zh-Hans.json").read_text(encoding="utf-8")))
        merged["face.reason.verified"] = "长" * 400  # a reason that fills the page
        return merged

    def test_the_header_cells_do_not_wrap(self):
        from datetime import date

        from anonymizer.ui import pdf

        page = pdf.report_html("a.mp4", self.rows(), self.texts(), True, date(2026, 10, 5))
        self.assertIn("th{background:#f0f2f5;white-space:nowrap}", page)

    def test_a_real_browser_lays_them_out_on_one_line(self):
        from datetime import date

        from anonymizer.ui import pdf

        if os.environ.get("ANONYMIZER_CHROME") != "1":
            self.skipTest("set ANONYMIZER_CHROME=1 to lay the report out with a real browser")
        from tests.ui.drive import Page

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "r.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(pdf.report_html("a.mp4", self.rows(), self.texts(), True, date(2026, 10, 5)))
            page = Page(
                "file://" + path,
                width=794,
                height=1000,
                port=0 or __import__("tests.ui.browser_case", fromlist=["x"]).free_port(),
            )
            try:
                page.wait_for("document.querySelectorAll('th').length > 0", 30)
                heights = page.eval(
                    "JSON.stringify([...document.querySelectorAll('th')].map((th) => th.getBoundingClientRect().height))"
                )
            finally:
                page.close()
        heights = json.loads(heights)
        self.assertLess(
            max(heights), 40, heights
        )  # 12.5 px text at 1.45 and 6 px of padding above and below: 30 px for one line


class RelativeDestinationIsNotTheServersWorkingFolder(QaCase):
    """Bug 14: a destination typed on the Export screen that did not start at the root ("exports", "Anonymized/photos") was taken from the
    folder the Studio was started in, so the files were written somewhere the person had never seen (and into the checkout, for a
    developer). It is now taken from the person's home folder, and the plan says where."""

    def test_it_is_taken_from_the_home_folder(self):
        from unittest import mock

        home = os.path.join(self.dir, "home")
        os.makedirs(home)
        clip = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip)
        with mock.patch.dict(os.environ, {"HOME": home, "USERPROFILE": home}):
            plan = self.api(
                "POST", "/api/render/plan", {"clips": [clip], "options": {"destination": "Anonymized/photos"}}
            )
            self.assertEqual(plan["destination"], os.path.join(home, "Anonymized", "photos"))
            self.assertEqual(
                plan["clips"][0]["output"], os.path.join(home, "Anonymized", "photos", "ok_anonymized.jpg")
            )
            tilde = self.api("POST", "/api/render/plan", {"clips": [clip], "options": {"destination": "~/x"}})
            self.assertEqual(tilde["destination"], os.path.join(home, "x"))
            here = self.api("POST", "/api/render/plan", {"clips": [clip], "options": {"destination": self.dir}})
            self.assertEqual(here["destination"], self.dir)  # a full path stays as it is


class PreviewOfAChangedOriginalIsNotNotModified(QaCase):
    """Bug 15: the browser asks again for a preview with the tag it has; the tag did not say anything about the original, so after the
    file was changed or deleted the answer was "304 not modified" and Review kept showing the picture of the old file, with no word that
    the original had changed (a browser with nothing cached was told at once)."""

    def test_the_tag_changes_with_the_original(self):
        path = self.photo()
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip)
        url = f"/api/clips/{clip}/preview?frame=0&view=anonymized&w=160"
        status, response, _ = self.get(url)
        self.assertEqual(status, 200)
        tag = response.getheader("ETag")
        status, _, _ = self.request("GET", url, headers={"If-None-Match": tag})
        self.assertEqual(status, 304)  # nothing changed: the browser keeps what it has
        stat = os.stat(path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))  # the file is touched
        status, _, data = self.request("GET", url, headers={"If-None-Match": tag})
        self.assertEqual((status, json.loads(data)["code"]), (422, "error.source.changed"))
        os.remove(path)
        status, _, data = self.request("GET", url, headers={"If-None-Match": tag})
        self.assertEqual((status, json.loads(data)["code"]), (422, "error.source.missing"))


class FrameCountOfAFileWhoseHeaderOverstatesIt(QaCase):
    """Bug 17: for Matroska, WebM and AVI the count of frames read from the header (duration times rate) can be one more than the decoder
    gives (a 2 s clip at 24 fps: 50 against 49). Review then asked for the last frame that is not there: the picture of that frame was a 400
    and the boxes of the last stretch a 500 ("internal error"), so the blur of the last frames was not drawn in the preview."""

    def test_the_clip_takes_the_real_count_when_it_is_analysed(self):
        clip_id = self.import_clip("over.mp4", n=30)
        session = self.server.session
        clip = session.clips[clip_id]
        from dataclasses import replace

        clip.info = replace(clip.info, frames=31)  # what the header said
        reader = self.analyze(clip_id)
        self.assertEqual(reader.wait_for("clip.analyzed")["frames"], 30)
        state = self.api("GET", "/api/state")["clips"][0]
        self.assertEqual(state["frames"], 30)
        base = f"/api/clips/{clip_id}"
        self.assertEqual(self.get(f"{base}/preview?frame=29&view=anonymized&w=160")[0], 200)
        self.assertEqual(self.get(f"{base}/boxes?from=29&to=29")[0], 200)
        status, _, data = self.get(f"{base}/boxes?from=30&to=30")
        self.assertEqual(status, 400, data[:200])  # beyond the last frame: refused, not an internal error
        self.assertEqual(self.get(f"{base}/preview?frame=30&view=anonymized&w=160")[0], 400)


class HostedDemoDoesNotNameItsFolders(QaCase):
    """Bug 20: in the hosted demo the plan and the answer to a render carried the path of the server's session folder
    (`/tmp/anonymizer-studio-xxxx/out/...`): the demo is somebody else's computer."""

    demo = True

    def test_the_answers_hold_file_names_and_no_server_folder(self):
        photo = self.photo()
        with open(photo, "rb") as f:
            data = f.read()
        status, _, answer = self.request("PUT", "/api/upload?name=demo.jpg", raw=data)
        self.assertEqual(status, 200, answer[:200])
        clip = json.loads(answer)["clips"][0]["clip_id"]
        self.analyze(clip)
        folder = self.server.session.work_dir
        plan = self.api("POST", "/api/render/plan", {"clips": [clip], "options": {}})
        self.assertNotIn(folder, json.dumps(plan))
        self.assertEqual(plan["clips"][0]["output"], "demo_anonymized.jpg")
        started = self.api("POST", "/api/render", {"clips": [clip], "options": {}}, expect=202)
        self.assertNotIn(folder, json.dumps(started))
        self.assertEqual(started["outputs"], {clip: "demo_anonymized.jpg"})
        self.wait_idle()
        status, _, body = self.get(f"/api/clips/{clip}/analysis.json")
        self.assertEqual(status, 200)
        self.assertNotIn(folder.encode(), body)
        self.assertEqual(json.loads(body)["source"], "demo.jpg")

    def test_the_settings_do_not_name_the_account_that_runs_the_server(self):
        home = os.path.expanduser("~")
        for path in ("/api/state", "/api/settings"):
            status, _, body = self.get(path)
            self.assertEqual(status, 200)
            self.assertNotIn(home.encode(), body, path)
        status, _, body = self.request("PUT", "/api/settings", {"blur": "pixelate"})
        self.assertNotIn(home.encode(), body)
        self.assertEqual(json.loads(body)["blur"], "pixelate")


class TwoStudiosInOneBrowser(QaCase):
    """Bug 21: the session cookie was called `anon_session` on every Studio, and a browser shares its cookies between all the ports of
    127.0.0.1: opening a second Studio (another folder, the demo, a restart on another port) replaced the cookie of the first, and the
    first window's every request was then refused ("This request is not allowed. If you opened this page from a bookmark...")."""

    def test_the_cookie_of_one_studio_is_not_the_cookie_of_another(self):
        status, response, _ = self.get(f"/?t={server_tests.TOKEN}", cookie=False)
        name = response.getheader("Set-Cookie").split("=", 1)[0]
        self.assertEqual(name, f"anon_session_{self.port}")
        # another Studio sets its own cookie under its own name: this one's token, sent under that other name, opens nothing
        status, _, _ = self.get(
            "/api/state", headers={"Cookie": f"anon_session_{self.port + 1}={server_tests.TOKEN}"}, cookie=False
        )
        self.assertEqual(status, 403)
        # and the other's cookie next to this one's does not take this one's away
        both = f"anon_session_{self.port + 1}=other; anon_session_{self.port}={server_tests.TOKEN}"
        self.assertEqual(self.get("/api/state", headers={"Cookie": both}, cookie=False)[0], 200)
