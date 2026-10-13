"""Studio v2.1.9: what the tenth client QA pass (05/10/2026, qa-anon-10) found in the page, in a real Chrome. Each test fails on v2.1.8."""

from __future__ import annotations

import unittest

try:
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class NoScriptNoteIsOnTheFirstScreen(BrowserCase):
    """Bug 9: with JavaScript off the note "Anonymizer Studio needs JavaScript" was in the page but 100 % of the window's height below its top
    (the empty `#app` is `height: 100vh`): a laptop's window showed a dark rectangle with a scrollbar and no word."""

    def test_the_note_is_inside_the_window(self):
        from tests.test_ui_server import TOKEN

        self.page.call("Emulation.setDeviceMetricsOverride", width=1280, height=640, deviceScaleFactor=1, mobile=False)
        self.page.call("Emulation.setScriptExecutionDisabled", value=True)
        try:
            self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?t={TOKEN}")
            self.page.sleep(1.5)
            box = self.page.eval(
                "(() => { const r = document.querySelector('.no-script p').getBoundingClientRect();"
                " return [r.top, r.bottom, innerHeight, document.documentElement.scrollHeight]; })()"
            )
        finally:
            self.page.call("Emulation.setScriptExecutionDisabled", value=False)
            self.page.call("Emulation.clearDeviceMetricsOverride")
        top, bottom, height, scroll_height = box
        self.assertLessEqual(bottom, height, f"the note is at {top:.0f}-{bottom:.0f} px in a window {height} px high")
        self.assertLessEqual(scroll_height, height, "the empty page scrolls")


if __name__ == "__main__":
    unittest.main()
