"""UI-only layout checks in a real Chrome on the mock page: no control hidden under another one at 1600 px (after the sealed read).

Needs google-chrome and websocket-client; without them the module is skipped, which is said in the skip reason.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import sys
import threading
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests", "ui"))

if importlib.util.find_spec("websocket") is None:
    raise unittest.SkipTest("websocket-client missing: the layout checks need it")
if not shutil.which("google-chrome"):
    raise unittest.SkipTest("google-chrome not found: the layout checks need a real browser")

from dev_server import serve  # noqa: E402
from drive import Page  # noqa: E402
from screens import OVERFLOW, settle  # noqa: E402

PORT = 8767


def free_port():
    """A debugging port nobody holds: the Chrome of the test before may still be closing the fixed one."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


HIT = """(sel) => {
  const el = document.querySelector(sel);
  const r = el.getBoundingClientRect();
  const top = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
  return !!top && (top === el || el.contains(top));
}"""


class Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threading.Thread(target=serve, args=(PORT,), daemon=True).start()

    def open(self, start):
        page = Page(f"http://127.0.0.1:{PORT}/?mock=fast&lang=en&start={start}", port=free_port())
        self.addCleanup(page.close)
        return page

    def hit(self, page, selector):
        return page.eval(f"({HIT})({selector!r})")

    def test_review_toolbar_controls_are_all_clickable_at_1600_px(self):
        page = self.open("review")
        page.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9")
        count = page.eval("document.querySelectorAll('.review-toolbar button:not([hidden])').length")
        self.assertGreaterEqual(count, 4)
        for index in range(count):
            selector = f".review-toolbar button:not([hidden]):nth-of-type({index + 1})"
            if page.eval(f"!!document.querySelector({selector!r})"):
                self.assertTrue(self.hit(page, selector), f"toolbar control {index + 1} is covered by another element")
        self.assertTrue(self.hit(page, ".review-toolbar .btn-small:last-child"), "Add missed area is covered")

    def test_a_notice_does_not_cover_the_export_button(self):
        page = self.open("export")
        page.wait_for("document.querySelectorAll('.export-card').length === 3")
        page.wait_for(
            "document.querySelector('.page-export .btn-primary') && !document.querySelector('.page-export .btn-primary').disabled"
        )
        page.eval(
            "(() => { let box = document.querySelector('.notices'); if (!box) { box = document.createElement('div'); box.className = 'notices'; document.body.append(box); }"
            " box.insertAdjacentHTML('beforeend', '<div class=\"notice notice-warning\"><span>A long notice, three lines of text, to stand over whatever is below it in a narrow corner of the page.</span><button type=\"button\">x</button></div>'); })()"
        )
        page.sleep(0.5)
        self.assertTrue(self.hit(page, ".page-export .btn-primary"), "a notice covers the Export button")

    def import_the_folder_with_six_refusals(self, page):
        page.wait_for("!!document.querySelector('.side-head button')")
        page.eval("document.querySelector('.side-head button').click()")
        page.wait_for("document.querySelectorAll('.explorer-list li').length > 0")
        page.eval("document.querySelector('dialog .btn-primary').click()")
        page.wait_for("document.querySelectorAll('.notice').length === 6")

    def test_notices_stay_clear_of_every_control_even_six_at_once(self):
        page = self.open("import")
        self.import_the_folder_with_six_refusals(page)
        covered = page.eval(
            "(() => { const box = document.querySelector('.notices').getBoundingClientRect();"
            " return [...document.querySelectorAll('button, input, select, a[href]')]"
            "   .filter((el) => !el.closest('.notices') && el.getClientRects().length > 0)"
            "   .filter((el) => { const r = el.getBoundingClientRect();"
            "     return r.width > 0 && r.height > 0 && r.left < box.right && r.right > box.left && r.top < box.bottom && r.bottom > box.top; })"
            "   .map((el) => el.textContent.trim() || el.getAttribute('aria-label') || el.className); })()"
        )
        self.assertEqual(covered, [], "a notice lies over these controls")

    def test_notices_go_by_themselves_unless_the_pointer_is_on_them(self):
        page = Page(f"http://127.0.0.1:{PORT}/?mock=fast&lang=en&notice_seconds=2", port=free_port())
        self.addCleanup(page.close)
        self.import_the_folder_with_six_refusals(page)
        x, y = page.center(".notice:first-child")
        page.mouse("mouseMoved", x, y)
        page.sleep(4)
        self.assertEqual(
            page.eval("document.querySelectorAll('.notice').length"), 1, "the notice under the pointer went"
        )
        page.mouse("mouseMoved", 5, 5)
        page.wait_for("document.querySelectorAll('.notice').length === 0", 6)

    def open_keep_all_dialog(self, page):
        page.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9")
        page.eval(
            "[...document.querySelectorAll('.review-toolbar button')].find((b) => !b.hidden && b.textContent.includes('visible')).click()"
        )
        page.wait_for("!!document.querySelector('dialog[open]')")

    def test_escape_closes_the_keep_all_doubtful_visible_dialog_and_decides_nothing(self):
        page = self.open("review")
        self.open_keep_all_dialog(page)
        page.key("Escape", code="Escape")
        page.wait_for("!document.querySelector('dialog')", 5)
        still_doubtful = "[...document.querySelectorAll('.review-toolbar button')].some((b) => !b.hidden && b.textContent.includes('visible'))"
        self.assertTrue(page.eval(still_doubtful), "Escape decided the doubtful items")

    def test_leaving_the_review_tab_closes_the_dialog_so_nothing_is_left_blocking(self):
        page = self.open("review")
        self.open_keep_all_dialog(page)
        page.eval("document.querySelectorAll('.tabs button')[3].click()")
        page.wait_for("!document.querySelector('dialog')", 5)
        self.assertTrue(self.hit(page, ".page-export .btn-primary"), "the Export page is still blocked")

    def test_the_review_timeline_has_no_scrollbar_of_its_own_when_its_lanes_fit_at_1280_px(self):
        page = self.open("review")
        page.call("Emulation.setDeviceMetricsOverride", width=1280, height=800, deviceScaleFactor=1, mobile=False)
        page.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9")
        page.sleep(0.5)
        scroll = page.eval(
            "(() => { const e = document.querySelector('.timeline-scroll'); return [e.scrollHeight, e.clientHeight]; })()"
        )
        self.assertLessEqual(
            scroll[0], scroll[1], "the timeline scrolls although its lanes fit (the playhead sticks out of the track)"
        )
        self.assertTrue(
            page.eval("document.documentElement.scrollHeight <= innerHeight"), "the page scrolls vertically"
        )

    def test_no_page_scrolls_sideways_or_sticks_out_at_tablet_and_phone_widths(self):
        """QA report, should-fix 7: the layout had a minimum width of 1100 px, so at 700 or 420 px the page scrolled sideways and
        the right panel was lost. A tablet upright (1200 x 1920), a half-screen window (700) and a phone (420), on the four pages."""
        page = Page(f"http://127.0.0.1:{PORT}/?mock=fast&lang=en", port=free_port())
        self.addCleanup(page.close)
        for width, height in ((1200, 1920), (700, 900), (420, 800)):
            page.call(
                "Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False
            )
            for name, start in (
                ("import", ""),
                ("analyze", "&start=analyze"),
                ("review", "&start=review"),
                ("export", "&start=export"),
            ):
                with self.subTest(width=width, page=name):
                    page.call("Page.navigate", url=f"http://127.0.0.1:{PORT}/?mock=fast&lang=en{start}")
                    page.sleep(1)
                    settle(page, name)
                    self.assertFalse(
                        page.eval("document.documentElement.scrollWidth > innerWidth + 1"), "the page scrolls sideways"
                    )
                    self.assertEqual(page.eval(OVERFLOW), [], "controls stick out of the window or lose their text")
                    main = page.eval(
                        "(() => { const b = [...document.querySelectorAll('.page:not([hidden]) .btn-primary, .page:not([hidden]) .btn')]"
                        ".find((el) => el.getClientRects().length > 0); if (!b) return null; b.scrollIntoView({block: 'center'}); const r = b.getBoundingClientRect(); "
                        "return [r.left >= 0 && r.right <= innerWidth, r.width > 0]; })()"
                    )
                    self.assertEqual(main, [True, True], "the main button of the page cannot be reached")

    def test_a_short_window_with_notices_shrinks_the_review_picture_instead_of_overflowing(self):
        """The Review picture used to keep its size (a 16:9 box 860 px wide) whatever room the page left it: with the notice strip
        above, or a short window, the controls were pushed out and the page grew scrollbars."""
        page = self.open("review")
        page.call("Emulation.setDeviceMetricsOverride", width=1280, height=720, deviceScaleFactor=1, mobile=False)
        page.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9")
        page.eval("document.querySelectorAll('.tabs button')[0].click()")
        self.import_the_folder_with_six_refusals(page)
        page.eval("document.querySelectorAll('.tabs button')[2].click()")
        page.sleep(1)
        facts = page.eval(
            "(() => { const r = (s) => document.querySelector(s).getBoundingClientRect();"
            " const toolbar = r('.review-toolbar'), stage = r('.page-review .stage'), controls = r('.page-review .controls');"
            " return {scrolls: document.documentElement.scrollHeight > innerHeight || document.documentElement.scrollWidth > innerWidth,"
            "  clear_of_toolbar: stage.top >= toolbar.bottom - 1, above_controls: stage.bottom <= controls.top + 1,"
            "  controls_in_window: controls.bottom <= innerHeight, stage_visible: stage.height > 60}; })()"
        )
        self.assertEqual(
            facts,
            {
                "scrolls": False,
                "clear_of_toolbar": True,
                "above_controls": True,
                "controls_in_window": True,
                "stage_visible": True,
            },
        )

    def test_a_touch_button_jumps_to_the_next_item_to_check(self):
        """Tablet report: N and the arrows are keyboard-only, and a touch screen had no way to reach the next doubtful item."""
        page = self.open("review")
        page.wait_for("document.querySelectorAll('.timeline-track .bar-item').length === 9")
        page.wait_for("!!document.querySelector('[data-action=next-check]:not([hidden])')")
        page.eval("document.querySelector('[data-action=next-check]').click()")
        page.wait_for("!!document.querySelector('.item-detail:not([hidden]) h2')")
        self.assertTrue(page.eval("document.querySelector('.item-detail h2').textContent.length > 0"))


if __name__ == "__main__":
    unittest.main()
