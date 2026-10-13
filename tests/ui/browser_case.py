"""A real Studio server (stand-in detectors) behind a headless Chrome, for tests of what a person sees and does.

    class Example(BrowserCase):
        def test_it(self):
            self.page.eval("document.querySelector('.tabs button').click()")

The page is the real one (src/anonymizer/ui/static); the engine is the stand-in of the other server tests, so nothing needs
the models. The tests are skipped where Chrome or websocket-client is missing. Chrome runs with a temporary profile and a
free debugging port, and is stopped by its own process when the test ends.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import socket
import tempfile
import unittest
from unittest import mock

try:
    from tests import test_ui_server as server_tests
    from tests.ui.drive import Page
except ImportError as e:  # engine libraries or websocket-client missing
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e

UI = pathlib.Path(__file__).resolve().parents[2] / "src" / "anonymizer" / "ui"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class BrowserCase(server_tests.StudioCase):
    """One Chrome for the whole class (starting it is the slow part), one server and one fresh page load per test."""

    width = 1280
    height = 800
    lang = "en"
    page = None
    _pages = {}  # test class -> its Chrome page, started by its first test

    @classmethod
    def tearDownClass(cls):
        page = BrowserCase._pages.pop(cls, None)
        if page is not None:
            profile = page.profile
            page.close()
            shutil.rmtree(profile, ignore_errors=True)
        super().tearDownClass()

    def setUp(self):
        if shutil.which("google-chrome") is None:
            self.skipTest("google-chrome is not installed")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        for folder in ("work", "settings", "media", "home"):
            os.makedirs(os.path.join(self.dir, folder))
        os.symlink(UI / "static", os.path.join(self.dir, "static"))
        os.symlink(UI / "i18n", os.path.join(self.dir, "i18n"))
        self.home = os.path.join(self.dir, "home")
        patch = mock.patch.dict(os.environ, {"HOME": self.home})
        patch.start()
        self.addCleanup(patch.stop)
        self.start_server()
        self.open_page()

    def open_page(self, width=None, height=None):
        """The page of this test's server in the class's Chrome, with the window at the size asked for."""
        url = f"http://127.0.0.1:{self.port}/?t={server_tests.TOKEN}&lang={self.lang}"
        width, height = width or self.width, height or self.height
        page = BrowserCase._pages.get(type(self))
        if page is None:
            page = BrowserCase._pages[type(self)] = Page(url, width=width, height=height, port=free_port())
            page.call(
                "Page.enable"
            )  # without it Chrome ignores the script below: the error list stayed empty whatever the page did
            page.call(
                "Page.addScriptToEvaluateOnNewDocument",
                source="window.__e=[];addEventListener('error',e=>__e.push(e.message));"
                "addEventListener('unhandledrejection',e=>__e.push('rej '+(e.reason&&e.reason.message||e.reason)))",
            )
        else:
            page.call("Page.navigate", url=url)
        self.page = page
        self.resize(width, height)
        page.wait_for("document.querySelectorAll('.tabs button').length === 4", 60)

    def resize(self, width, height):
        self.page.call(
            "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False
        )

    def reload_page(self):
        self.page.eval("location.reload()")
        self.page.sleep(1)
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4", 60)

    def click(self, selector):
        """A real mouse click in the middle of the first element that matches."""
        self.page.click(*self.page.center(selector))

    def js_errors(self):
        return json.loads(self.page.eval("JSON.stringify(window.__e || [])"))

    def import_paths(self, *paths):
        reply = self.api("POST", "/api/clips", {"paths": list(paths)})
        self.reload_page()
        return reply
