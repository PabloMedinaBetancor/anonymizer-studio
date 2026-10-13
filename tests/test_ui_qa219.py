"""Studio v2.1.9: what the tenth client QA pass (05/10/2026, qa-anon-10) found at the server. One test class per bug; each fails on
v2.1.8 and passes here."""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

try:
    import numpy as np
    from PIL import Image

    from anonymizer.ui import pdf as pdf_module
    from anonymizer.ui import server as server_module
    from tests import test_ui_server as server_tests
    from tests.fakes import FACE, FixedModule
    from tests.test_ui_server import EventReader
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


def fake_webview(start):
    """A stand-in for pywebview with its one rule that matters here: `start` runs on the main thread or it raises."""
    module = types.ModuleType("webview")
    module.created = []
    module.started = []  # whether each `start` ran on the main thread

    def create_window(title, url, **options):
        module.created.append((title, url))

    def checked_start(*args, **options):
        module.started.append(threading.current_thread() is threading.main_thread())
        if not module.started[-1]:
            raise RuntimeError("pywebview must be run on a main thread.")  # what the real one says
        return start()

    module.create_window = create_window
    module.start = checked_start
    return module


class OptionalWindow(unittest.TestCase):
    """Bug 1: after the README's optional `pip install ".[window]"`, the Studio opened no window and no browser: it called
    pywebview's `start` from a thread (pywebview refuses: "must be run on a main thread"), printed a traceback and went on
    serving to nobody until it stopped itself ten minutes later."""

    def make(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        for folder in ("static", "i18n", "work", "settings"):
            os.makedirs(os.path.join(tmp.name, folder))
        with open(os.path.join(tmp.name, "static", "index.html"), "w") as f:
            f.write("<!doctype html><title>Studio</title>")
        with open(os.path.join(tmp.name, "i18n", "en.json"), "w") as f:
            f.write("{}")
        server = server_module.make_server(
            settings_folder=os.path.join(tmp.name, "settings"),
            static_dir=os.path.join(tmp.name, "static"),
            i18n_dir=os.path.join(tmp.name, "i18n"),
            work_dir=os.path.join(tmp.name, "work"),
            modules=[],
            engine="thread",
        )
        self.addCleanup(server.server_close)
        return server

    def run_main(self, server, window):
        """`main` on the main thread with this server, with a safety stop so that a hang fails the test instead of freezing the suite."""
        watchdog = threading.Timer(20, server.shutdown)
        watchdog.start()
        self.addCleanup(watchdog.cancel)
        with (
            mock.patch.dict(
                os.environ, {"DISPLAY": ":0"}
            ),  # a window needs a display (2.1.10); the test computer may have none
            mock.patch.dict(sys.modules, {"webview": window}),
            mock.patch.object(server_module.webbrowser, "open") as opened,
            mock.patch.object(server_module, "make_server", return_value=server),
            mock.patch.object(server_module, "remove_stale_work_folders"),
            mock.patch.object(server_module, "stop_on_signals"),
            mock.patch.object(server_module, "watch_idle"),
            mock.patch.object(server_module, "user_folder", return_value=None),
            mock.patch("sys.stdout"),
            mock.patch("sys.stderr"),
        ):
            code = server_module.main([])
        self.assertEqual(code, 0)
        return opened

    def test_the_window_runs_on_the_main_thread_and_its_closing_ends_the_studio(self):
        server = self.make()
        window = fake_webview(lambda: None)  # a window that is shown and closed at once
        opened = self.run_main(server, window)
        self.assertEqual(window.created, [("Anonymizer Studio", server.url())])
        self.assertEqual(window.started, [True])
        opened.assert_not_called()  # the window is the page: no second one in the browser

    def test_where_the_window_cannot_open_the_page_opens_in_the_browser(self):
        server = self.make()

        def no_toolkit():
            # what pywebview raises on a computer with neither GTK nor Qt; the Studio is stopped from outside, as a person would
            server.stop_later()
            raise RuntimeError("You must have either QT or GTK with Python extensions installed")

        window = fake_webview(no_toolkit)
        opened = self.run_main(server, window)
        opened.assert_called_once_with(server.url())

    def test_without_pywebview_the_page_opens_in_the_browser(self):
        server = self.make()
        threading.Timer(1, server.shutdown).start()
        opened = self.run_main(server, None)  # sys.modules["webview"] = None: the import fails
        opened.assert_called_once_with(server.url())


class ReportsOfSourcesThatDifferOnlyInTheExtension(server_tests.StudioCase):
    """Bug 3: `x.bmp`, `x.tif` and `x.webp` in one folder were written as `x_anonymized.bmp`, `.tif` and `.webp`, but their
    reports were `x_anonymized.report.csv`, `x_anonymized.report-2.csv` and `-3`: nothing said which report was which picture."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_each_report_keeps_the_extension_of_its_picture_when_the_names_meet(self):
        ids = []
        for name in ("x.bmp", "x.tif"):
            path = os.path.join(self.dir, "media", name)
            Image.fromarray((np.random.RandomState(3).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
            ids.append(self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"])
        self.analyze(*ids)
        folder = os.path.join(self.dir, "out")
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        body = {"destination": folder, "reports": {"pdf": False, "csv": True, "json": True}}
        self.api("POST", "/api/render", {"clips": ids, "options": body}, 202)
        reader.wait_for("job.finished")
        self.assertEqual(
            sorted(os.listdir(folder)),
            [
                "x_anonymized.bmp",
                "x_anonymized.report.csv",
                "x_anonymized.report.json",
                "x_anonymized.tif",
                "x_anonymized.tif.report.csv",
                "x_anonymized.tif.report.json",
            ],
        )
        with open(os.path.join(folder, "x_anonymized.tif.report.csv"), encoding="utf-8-sig") as f:
            self.assertEqual(f.read().splitlines()[1].split(",")[0], "x.tif")  # it is the report of the .tif

    def test_a_single_picture_keeps_the_documented_name(self):
        path = os.path.join(self.dir, "media", "y.png")
        Image.fromarray((np.random.RandomState(3).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api(
            "POST",
            "/api/render",
            {
                "clips": [clip_id],
                "options": {"destination": folder, "reports": {"pdf": False, "csv": True, "json": False}},
            },
            202,
        )
        reader.wait_for("job.finished")
        self.assertEqual(sorted(os.listdir(folder)), ["y_anonymized.png", "y_anonymized.report.csv"])


class PdfCarriesNoBrowserFingerprint(unittest.TestCase):
    """Bug 4: the PDF report's Creator field was the printing Chrome's user agent ("Mozilla/5.0 (X11; Linux x86_64) ...
    HeadlessChrome/153.0.0.0"): the person's operating system and browser version, in a privacy report that is sent on."""

    def test_the_creator_is_the_studio_not_the_browser(self):
        browser = pdf_module.find_browser()
        if browser is None or shutil.which("pdfinfo") is None:
            self.skipTest("needs a Chromium-based browser and pdfinfo")
        with tempfile.TemporaryDirectory() as folder:
            page = os.path.join(folder, "r.html")
            with open(page, "w", encoding="utf-8") as f:
                f.write("<!doctype html><title>Privacy report</title><p>x</p>")
            target = os.path.join(folder, "r.pdf")
            pdf_module.html_to_pdf(browser, page, target)
            info = subprocess.run(["pdfinfo", target], capture_output=True, text=True, check=True).stdout
        creator = re.search(r"^Creator:\s*(.*)$", info, re.M).group(1)
        self.assertEqual(creator, "Anonymizer Studio")
        for given_away in ("Mozilla", "Chrome", "Linux", "Windows", "Mac"):
            self.assertNotIn(given_away, info)


class OpenCvWarningOnTheFirstExample(unittest.TestCase):
    """Bug 5: the README's first example (`anonymize_video`, and every `anonymize_image`) printed "[ WARN:0@0.1] global
    net_impl_backend.cpp:345 setPreferableTarget Targets are not supported by the new graph engine for now" on stderr at each model load:
    a line that reads as a fault, from OpenCV's own log."""

    def test_a_photo_is_anonymized_with_nothing_on_stderr(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models = os.environ.get("ANONYMIZER_MODELS") or os.path.join(root, "models")
        if not os.path.isfile(os.path.join(models, "face_detection_yunet_2023mar.onnx")):
            self.skipTest("needs the face model")
        env = {k: v for k, v in os.environ.items() if k != "OPENCV_LOG_LEVEL"}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [os.path.join(root, "src"), env.get("PYTHONPATH")]))
        with tempfile.TemporaryDirectory() as folder:
            code = (
                "import anonymizer, sys\n"
                "anonymizer.anonymize_image(sys.argv[1], sys.argv[2], anonymizer.Options(detection='fast'))\n"
            )
            done = subprocess.run(
                [
                    sys.executable,
                    *(["-s"] if sys.flags.no_user_site else []),
                    "-c",
                    code,
                    os.path.join(root, "demo", "street-westminster.jpg"),
                    os.path.join(folder, "o.jpg"),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
        if done.returncode != 0 and (
            "error.model.load" in done.stderr or "detection model could not be loaded" in done.stderr
        ):
            self.skipTest(
                "a model of this machine's models folder has no checksum entry"
            )  # a models folder of another install has no RF-DETR line
        self.assertEqual(done.returncode, 0, done.stderr[-500:])
        self.assertEqual(done.stderr.strip(), "")


class VideoContainerIsDocumented(unittest.TestCase):
    """Bug 6: `phone.mov` was exported as `phone_anonymized.mp4` (every video leaves as an H.264 MP4, a rotation flag applied to the
    picture) and no document said so: the Export screen showed the name, the guides were silent."""

    def test_the_faq_says_what_an_output_video_is(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "docs", "faq.md"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("A video always comes out as an MP4 with 8-bit H.264 video", text)
        self.assertIn("A `.mov` leaves as `.mp4`", text)


class InstallLeavesNothingToCommit(unittest.TestCase):
    """Bug 7: `pip install .` (the README's install, from a checkout) leaves `build/` and `UNKNOWN.egg-info/` (a failed run with the old
    pip that Ubuntu ships) or `anonymizer_studio.egg-info/` in the checkout, and `.gitignore` did not list them: `git status` of a
    client who had only installed showed untracked folders."""

    def test_the_folders_pip_makes_are_ignored(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, ".gitignore"), encoding="utf-8") as f:
            lines = [line.strip() for line in f]
        self.assertIn("/build/", lines)
        self.assertIn("*.egg-info/", lines)


class BlockedCookiesAreSaid(unittest.TestCase):
    """Bug 8: a browser that blocks cookies (Firefox's "Block all cookies", a privacy extension) cannot keep the Studio's session: the page
    opened from the printed address said "This page is not connected to the Studio ... If you opened this page from a bookmark, use the
    address the Studio printed", advice that cannot help someone who just did."""

    def test_every_language_names_the_cookies_of_the_address(self):
        folder = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "anonymizer", "ui", "i18n"
        )
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                text = json.load(f)["error.request.forbidden"]
            with self.subTest(language=name):
                self.assertIn("127.0.0.1", text)
                self.assertIn("cookie", text.lower())


class ReadmeTestCount(unittest.TestCase):
    """Bug 2: the README's Development section said "414 test cases" for a suite of about 1,200 (the figure comes from
    `bench/results/engine.json`, measured on 03/10/2026 and never taken again)."""

    def test_the_count_the_readme_quotes_is_near_the_real_one(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "bench", "results", "engine.json"), encoding="utf-8") as f:
            quoted = json.load(f)["repo"]["tests"]["count"]
        real = 0
        for path in glob.glob(os.path.join(root, "tests", "**", "test_*.py"), recursive=True):
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
            real += sum(1 for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test"))
        self.assertLess(
            abs(quoted - real), 0.1 * real, f"the README quotes {quoted} test cases, the tests hold about {real}"
        )


if __name__ == "__main__":
    unittest.main()
