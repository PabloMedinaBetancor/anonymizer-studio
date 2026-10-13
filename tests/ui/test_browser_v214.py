"""Studio v2.1.4: what the fifth client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.3 and passes here."""

from __future__ import annotations

import unittest

try:
    from tests.fakes import FACE, Delayed, FixedModule, plate_module
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class PolicyOfThePageIsNotBrokenByIt(BrowserCase):
    """Bug 4: the page now comes with a Content-Security-Policy; the four screens, the video, the thumbnails and the pictures of the review
    must all work under it (a violation is reported to the page as an event, and none may happen)."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_a_walk_through_the_four_screens_breaks_no_rule(self):
        self.page.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source="window.__csp=[];addEventListener('securitypolicyviolation',e=>__csp.push(e.violatedDirective+' '+e.blockedURI))",
        )
        clip_id = self.import_clip("walk.mp4", n=12)
        self.reload_page()
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        self.page.wait_for("document.querySelectorAll('.tabs button')[2].disabled === false", 120)
        self.wait_idle()
        for index in range(4):
            self.page.eval(f"document.querySelectorAll('.tabs button')[{index}].click()")
            self.page.sleep(0.8)
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelector('.stage img, .stage video, .stage canvas') !== null", 30)
        self.page.sleep(1.5)
        self.assertEqual(self.page.eval("typeof window.__csp"), "object")  # the listener was in place
        self.assertEqual(self.page.eval("JSON.stringify(window.__csp || [])"), "[]")
        self.assertEqual(self.js_errors(), [])
        self.assertGreater(self.page.eval("document.querySelectorAll('img').length"), 0)


class StrengthSliderShowsItsValue(BrowserCase):
    """The Strength slider had no value beside it (the margin has): a person could not say what 1.4 looked like or put the same value twice."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]
    SET = (
        "(() => { const s = document.querySelector('#strength'); s.value = %d; s.dispatchEvent(new Event('input', {bubbles: true}));"
        " return document.querySelector('label[for=strength]').textContent; })()"
    )

    def test_the_label_carries_the_value_and_follows_the_slider(self):
        self.page.wait_for("document.querySelector('label[for=strength]') !== null", 30)
        self.assertEqual(self.page.eval("document.querySelector('label[for=strength]').textContent"), "Strength ×1.00")
        self.assertEqual(self.page.eval(self.SET % 140), "Strength ×1.40")
        self.page.sleep(0.5)
        self.assertEqual(self.page.eval("document.querySelector('label[for=strength]').textContent"), "Strength ×1.40")

    def test_the_labels_of_both_sliders_keep_one_line_beside_the_slider(self):
        self.page.wait_for("document.querySelector('label[for=strength]') !== null", 30)
        for width in (1280, 1920):
            self.resize(width, 800)
            self.page.sleep(0.4)
            heights = self.page.eval(
                "['strength', 'margin'].map(id => Math.round(document.querySelector('label[for=' + id + ']').getBoundingClientRect().height))"
            )
            self.assertTrue(
                all(h < 24 for h in heights), (width, heights)
            )  # one line (it was two beside a slider fixed at 170 px)

    def test_it_is_written_the_way_the_language_writes_numbers(self):
        self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?lang=de")
        self.page.wait_for("document.querySelector('label[for=strength]') !== null", 30)
        self.page.sleep(0.5)
        self.assertEqual(self.page.eval(self.SET % 150), "Stärke ×1,50")


class ExportProgressAfterReload(BrowserCase):
    """Polish of 2.1.3: reload in the middle of an export and the Export screen showed neither the progress nor the files already written."""

    modules = [Delayed(plate_module(), 0.02)]

    def test_progress_and_written_files_come_back(self):
        import os

        ids = [self.import_clip(f"x{n}.mp4", n=240) for n in range(2)]
        reader = self.start_analysis(ids)
        dest = os.path.join(self.dir, "out")
        self.api(
            "POST",
            "/api/render",
            {"clips": ids, "options": {"destination": dest, "reports": {"pdf": False, "csv": True, "json": False}}},
            expect=202,
        )
        self.page.wait_for("document.querySelector('.export-card .bar-small') !== null", 60)
        self.reload_page()
        self.page.wait_for("document.querySelectorAll('.export-card').length === 2", 30)
        shown = self.page.eval("[...document.querySelectorAll('.export-card')].map(c => c.innerText).join('|')")
        self.assertRegex(shown, r"Rendering|Queued|Done|Rendered|written", shown)  # a card is not blank
        self.assertEqual(
            self.page.eval(
                "document.querySelectorAll('.export-card .bar-small, .export-card .ok-text, .export-card .dim').length > 0"
            ),
            True,
        )
        reader.wait_for("job.finished")
        self.reload_page()
        self.page.wait_for("document.querySelectorAll('.export-files').length === 2", 30)
        files = self.page.eval("[...document.querySelectorAll('.export-files')].map(u => u.innerText).join('|')")
        self.assertIn("x0_anonymized.mp4", files)
        self.assertIn("x1_anonymized.report.csv", files)

    def start_analysis(self, ids):
        from tests.test_ui_server import EventReader

        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": ids}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.api("POST", "/api/render/plan", {"clips": ids, "options": {}})
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        return EventReader(self.port)


class QuickPlayPauseAndResizeRaiseNoErrors(BrowserCase):
    """Found by random clicking, now that the browser tests really collect the page's errors: two quick presses of Play left an unhandled
    "play() request was interrupted by a call to pause()", and resizing the window in Review reported a ResizeObserver loop."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def open_review(self):
        clip_id = self.import_clip("play.mp4", n=60)
        self.reload_page()
        from tests.test_ui_server import EventReader

        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for(
            "document.querySelector('.btn-play') !== null && document.querySelectorAll('.timeline-track .bar-item').length > 0",
            30,
        )

    def test_play_and_pause_pressed_quickly(self):
        self.open_review()
        for _ in range(6):
            self.page.eval(
                "(() => { const b = document.querySelector('.btn-play'); b.click(); b.click(); b.click(); })()"
            )
            self.page.sleep(0.3)
        self.page.sleep(1)
        self.assertEqual(self.js_errors(), [])

    def test_resizing_the_window_in_review(self):
        self.open_review()
        for width, height in (
            (420, 800),
            (1920, 1080),
            (1280, 800),
            (900, 600),
            (1920, 1080),
            (1200, 1920),
            (1280, 800),
        ):
            self.resize(width, height)
            self.page.sleep(0.4)
        self.assertEqual(self.js_errors(), [])
