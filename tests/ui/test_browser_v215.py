"""Studio v2.1.5: what the sixth client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.4 and passes here."""

from __future__ import annotations

import os
import unittest

try:
    from tests.fakes import FACE, FixedModule
    from tests.test_ui_server import EventReader
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class ExportResultKeepsItsLabelsInOneLine(BrowserCase):
    """Bug 8: the kind of each written file ("PDF report", "Video or photo") was broken in the middle of a word on the Export screen
    once a render was done ("PDF repor" and "t"), at 1280 px and at 420."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_the_pills_beside_the_written_files_do_not_wrap(self):
        clip = self.import_clip("pills.mp4", n=12)
        self.reload_page()
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip]}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        dest = os.path.join(self.dir, "out-with-a-rather-long-name-to-make-the-path-wrap-in-the-narrow-column")
        self.api(
            "POST",
            "/api/render",
            {"clips": [clip], "options": {"destination": dest, "reports": {"pdf": False, "csv": True, "json": True}}},
            202,
        )
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("document.querySelectorAll('.export-files .pill').length >= 3", 30)
        for width in (1280, 420):
            self.resize(width, 800)
            self.page.sleep(0.4)
            heights = self.page.eval(
                "[...document.querySelectorAll('.export-files .pill')].map(p => Math.round(p.getBoundingClientRect().height))"
            )
            self.assertTrue(heights and all(h < 24 for h in heights), (width, heights))  # one line each
        self.resize(1280, 800)
        self.page.sleep(0.4)
        head = self.page.eval(
            "Math.round(document.querySelector('.page-export .head-label').getBoundingClientRect().height)"
        )
        self.assertLess(head, 24, head)  # "Output files" was cut in two ("Output" over "files") beside the long folder


class HeaderFitsAPhoneOf320Pixels(BrowserCase):
    """Bug 9: at 320 px (a small phone, or a window zoomed to 400 %) the header was 7 px wider than the window and the page scrolled sideways."""

    def test_no_screen_scrolls_sideways_at_320(self):
        self.resize(320, 480)
        for lang in ("en", "fr", "pt"):
            self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?lang={lang}")
            self.page.wait_for("document.querySelectorAll('.tabs button').length === 4", 30)
            self.page.sleep(0.4)
            for index in range(4):
                self.page.eval(f"document.querySelectorAll('.tabs button')[{index}].click()")
                self.page.sleep(0.4)
                width = self.page.eval("document.documentElement.scrollWidth")
                self.assertLessEqual(width, 320, (lang, index, width))


class AnalyzeCountsWhatItWouldDo(BrowserCase):
    """Bug 5: "Analyze 2 clips" under two clips that were already analysed with these settings (the server skips them at once)."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    LABEL = "document.querySelector('.page:not([hidden]) .cta .btn-primary').textContent"

    def test_the_button_says_what_it_will_do_and_follows_the_options(self):
        ids = [self.import_clip(f"c{n}.mp4", n=12) for n in range(2)]
        self.reload_page()
        self.page.wait_for(f"{self.LABEL} === 'Analyze 2 clips'", 30)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": ids}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[0].click()")
        self.page.wait_for(f"{self.LABEL} === 'All clips are analyzed'", 30)
        self.assertTrue(self.page.eval("document.querySelector('.page:not([hidden]) .cta .btn-primary').disabled"))
        self.page.eval(
            "[...document.querySelectorAll('.segmented button')].find(b => b.textContent.startsWith('Fast')).click()"
        )
        self.page.wait_for(f"{self.LABEL} === 'Analyze 2 clips'", 30)  # other settings: the analysis is done again
        self.assertEqual(self.js_errors(), [])


class FolderOfRefusedFilesIsOneNotice(BrowserCase):
    """Bug 17: a folder of 300 HEIC photos (what an iPhone gives) was 300 notices, one line each, identical but for the name, that went by
    in a few seconds; now it is the first, in full, and one line for the rest."""

    DROP = """(async () => {
        const transfer = new DataTransfer();
        for (let n = 0; n < 12; n++) transfer.items.add(new File([new Uint8Array(100)], 'IMG_' + String(n).padStart(4, '0') + '.HEIC'));
        const area = document.querySelector('.center-body');
        area.dispatchEvent(new DragEvent('dragover', {dataTransfer: transfer, bubbles: true, cancelable: true}));
        area.dispatchEvent(new DragEvent('drop', {dataTransfer: transfer, bubbles: true, cancelable: true}));
    })()"""

    def test_twelve_refused_files_make_two_notices(self):
        self.page.eval(self.DROP)
        self.page.wait_for("document.querySelectorAll('.notice').length >= 2", 30)
        self.page.sleep(1.0)
        texts = self.page.eval("[...document.querySelectorAll('.notice')].map(n => n.innerText)")
        self.assertEqual(len(texts), 2, texts)
        self.assertIn("IMG_0000.HEIC is a HEIC photo", texts[0])
        self.assertIn("11 more files were not imported, for the same reason", texts[1])
        self.assertIn("IMG_0001.HEIC, IMG_0002.HEIC, IMG_0003.HEIC", texts[1])


if __name__ == "__main__":
    unittest.main()
