"""Studio v2.1.8: what the ninth client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.7 and passes here."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import unittest
from unittest import mock

try:
    import anonymizer
    from tests.fakes import FACE, Delayed, FixedModule
    from tests.test_ui_server import EventReader
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


def open_tab(page, index):
    """Click a tab once it is enabled (on a slow computer the page's state arrives after the tabs are drawn, and a click on a disabled tab does nothing)."""
    page.wait_for(f"document.querySelectorAll('.tabs button')[{index}].disabled === false", 90)
    page.eval(f"document.querySelectorAll('.tabs button')[{index}].click()")


class StoppedComparisonShowsOnTheCard(BrowserCase):
    """Bug 1: a stop during the comparison video left the card "Done" with no comparison and no word about it."""

    def test_the_card_says_the_comparison_video_was_not_made(self):
        path = self.make_clip()
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)

        def stopped(original, anonymized, target, on_event=None, cancel=None):
            cancel.cancel()
            raise anonymizer.Cancelled()

        with mock.patch.object(anonymizer, "compare_video", stopped):
            folder = os.path.join(self.dir, "out")
            self.api(
                "POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder, "comparison": True}}, 202
            )
            reader.wait_for("job.finished")
            self.wait_idle()
        self.reload_page()
        open_tab(self.page, 3)
        self.page.wait_for("document.querySelector('.export-card') !== null", 30)
        card = self.page.eval("document.querySelector('.export-card').innerText")
        self.assertIn("Stopped: the comparison video was not made.", card)
        self.assertNotIn("Comparison\n", card)


class StudioGoneUnderThePage(BrowserCase):
    """Bug 2: with the Studio gone the buttons stayed live and a click said the browser's own English text ("NetworkError when attempting to
    fetch resource"); the page now says in its language that the Studio is not running and how to start it, and offers nothing."""

    lang = "es"

    def test_the_page_says_so_in_its_language_and_offers_nothing(self):
        self.import_paths(self.make_clip())
        self.page.wait_for("document.querySelectorAll('.clip-row, .clip-card').length >= 1", 30)
        self.stop_server()
        self.page.wait_for("!document.querySelector('.lost-bar').hidden", 60)
        bar = self.page.eval("document.querySelector('.lost-bar').innerText")
        self.assertIn("El Studio no est\u00e1 en marcha", bar)
        self.assertIn("python3 -m anonymizer.ui", bar)
        self.assertNotIn("-s -m", bar)
        self.assertEqual(self.page.eval("document.querySelector('.notices').innerText"), "")
        inert = self.page.eval("[...document.querySelectorAll('.topbar, main.page')].every((el) => el.inert)")
        self.assertTrue(inert)
        # a click on a tab does nothing: the screen stays where it was
        before = self.page.eval("document.querySelector('.tabs [aria-current]').innerText")
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")
        self.page.sleep(0.3)
        self.assertEqual(self.page.eval("document.querySelector('.tabs [aria-current]').innerText"), before)
        english = self.page.eval("document.body.innerText")
        self.assertNotIn("NetworkError", english)
        self.assertNotIn("Failed to fetch", english)

    def test_a_studio_that_comes_back_is_usable_again(self):
        self.import_paths(self.make_clip())
        port = self.port
        self.stop_server()
        self.page.wait_for("!document.querySelector('.lost-bar').hidden", 60)
        self.fixed_port = port
        self.start_server()
        self.page.wait_for("document.querySelector('.lost-bar').hidden", 60)
        self.assertFalse(self.page.eval("document.querySelector('main.page').inert"))


class PaperFollowsTheBrowserLanguage(BrowserCase):
    """Bug 4, end to end: the PDF an export writes is Letter for a browser in US English and A4 for any other."""

    browser = shutil.which("google-chrome")
    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def page_size_of_an_export(self, accept_language):
        from tests.ui.test_browser_v217 import save_photo

        if shutil.which("pdfinfo") is None:
            self.skipTest("needs pdfinfo")
        name = f"p-{accept_language}.jpg"
        clip_id = self.api(
            "POST", "/api/clips", {"paths": [save_photo(os.path.join(self.dir, "media", name), 300, 200)]}
        )["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.page.call(
            "Emulation.setUserAgentOverride",
            userAgent="Mozilla/5.0 (X11; Linux x86_64) Chrome/153",
            acceptLanguage=accept_language,
        )
        self.reload_page()
        self.assertEqual(self.page.eval("navigator.language"), accept_language)
        open_tab(self.page, 3)
        self.page.wait_for("!document.querySelector('.page-export .btn-primary').disabled", 60)
        self.page.eval("document.querySelector('.page-export .btn-primary').click()")
        # the click is answered a moment later on a slow computer: wait for the written report, not for "no job" (which is true before the job exists)
        render = None
        end = time.monotonic() + 120
        while time.monotonic() < end:
            clips = self.api("GET", "/api/state")["clips"]
            render = next(c for c in clips if c["name"] == name).get("render")
            if render and render.get("phase") == "done":
                break
            time.sleep(0.3)
        self.assertEqual((render or {}).get("phase"), "done", render)
        info = subprocess.run(["pdfinfo", render["report"]["pdf"]], capture_output=True, text=True, check=True).stdout
        return re.search(r"Page size:.*\((\w+)\)", info).group(1)

    def test_us_english_prints_letter(self):
        self.assertEqual(self.page_size_of_an_export("en-US"), "letter")

    def test_british_english_prints_a4(self):
        self.assertEqual(self.page_size_of_an_export("en-GB"), "A4")


class TheTestsChromeLeavesNothingInTmp(unittest.TestCase):
    """Bug 16 (2.1.7's "Found and not fixed", the part that was left): each Chrome of the tests left a `com.google.Chrome.XXXXXX` folder in the
    temporary folder (180 after one round), and a `Default` folder came back in its profile after the profile had been removed."""

    def test_a_chrome_that_was_opened_and_closed_leaves_nothing(self):
        import shutil
        import tempfile
        import time

        from tests.ui.browser_case import free_port
        from tests.ui.drive import Page

        if shutil.which("google-chrome") is None:
            self.skipTest("google-chrome is not installed")
        with tempfile.TemporaryDirectory(dir="/tmp") as folder:
            with mock.patch.object(tempfile, "tempdir", folder), mock.patch.dict(os.environ, {"TMPDIR": folder}):
                page = Page("about:blank", width=400, height=300, port=free_port())
                page.eval("1 + 1")
                page.close()
            time.sleep(1.0)
            self.assertEqual(os.listdir(folder), [])


class StopOfAnExportSaysHowFar(BrowserCase):
    """Bug 18: Stop in the middle of an export said nothing: the bar went, the clips not reached went back to their plan, and a client did not know
    how many files had been written."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_a_notice_counts_the_clips_written(self):
        paths = [self.make_clip(f"c{i}.mp4", n=60) for i in range(3)]
        ids = [c["clip_id"] for c in self.api("POST", "/api/clips", {"paths": paths})["clips"]]
        self.analyze(*ids)
        self.reload_page()
        open_tab(self.page, 3)
        real = anonymizer.Anonymizer.render

        def slow(self_, analysis, target, on_event=None, cancel=None):
            def handler(event):
                time.sleep(0.2)
                on_event(event)

            return real(self_, analysis, target, handler, cancel)

        with mock.patch.object(anonymizer.Anonymizer, "render", slow):
            self.api(
                "POST", "/api/render", {"clips": ids, "options": {"destination": os.path.join(self.dir, "out")}}, 202
            )
            self.page.wait_for(
                "[...document.querySelectorAll('.export-card')].some((c) => /Done in/.test(c.innerText))", 60
            )
            self.api("POST", "/api/job/stop")
            self.wait_idle()
        self.page.wait_for("/Export stopped/.test(document.querySelector('.notices').innerText)", 30)
        notice = self.page.eval("document.querySelector('.notices').innerText")
        self.assertRegex(notice, r"Clips written: [12] of 3")


class PictureOfAFileThatWentAway(BrowserCase):
    """Bug 14: a file removed from the disk after its analysis left Review with a broken-image icon on black; the only word of why was a notice that
    goes after 20 s."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_the_stage_says_the_picture_cannot_be_shown(self):
        from tests.ui.test_browser_v217 import save_photo

        path = save_photo(os.path.join(self.dir, "media", "gone.jpg"), 600, 400)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.reload_page()
        open_tab(self.page, 2)
        self.page.wait_for("document.querySelector('main.page:not([hidden]) .stage img')?.complete", 20)
        self.page.sleep(0.5)
        shown = "!document.querySelector('main.page:not([hidden]) .stage-missing').hidden"
        self.assertFalse(self.page.eval(shown))  # the file is there: no word
        os.remove(path)
        self.reload_page()
        open_tab(self.page, 2)
        self.page.wait_for(shown, 20)
        self.assertIn(
            "The picture cannot be shown",
            self.page.eval("document.querySelector('main.page:not([hidden]) .stage-missing').innerText"),
        )


class PageWithoutJavaScript(BrowserCase):
    """Bug 13: with JavaScript off the page was blank, with nothing to say why."""

    def test_it_says_what_it_needs(self):
        from tests.test_ui_server import TOKEN

        self.page.call("Emulation.setScriptExecutionDisabled", value=True)
        try:
            self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?t={TOKEN}")
            self.page.sleep(1.5)
            text = self.page.eval("document.body.innerText")
        finally:
            self.page.call("Emulation.setScriptExecutionDisabled", value=False)
        self.assertIn("needs JavaScript", text)
        self.assertIn("necesita JavaScript", text)


class DemoGoneUnderThePage(BrowserCase):
    """Bug 2, in the hosted demo: "run anonymizer-studio in a terminal" means nothing to a visitor; the bar says to reload the page."""

    demo = True

    def test_the_bar_says_to_reload(self):
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4")
        self.stop_server()
        self.page.wait_for("!document.querySelector('.lost-bar').hidden", 60)
        bar = self.page.eval("document.querySelector('.lost-bar').innerText")
        self.assertIn("reload the page", bar)
        self.assertNotIn("anonymizer-studio", bar)


class PageTheStudioRefuses(BrowserCase):
    """Bug 2, the other half: the Studio is there but refuses the page (the address without its session, a bookmark of an earlier start): the
    page said "The Studio is not running" over live buttons, or only a notice that went after 20 s."""

    lang = "de"

    def test_a_bar_says_the_page_is_not_connected_and_the_screens_are_inert(self):
        self.page.call("Network.enable")
        self.page.call("Network.clearBrowserCookies")
        self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?lang=de")
        self.page.wait_for(
            "document.querySelectorAll('.tabs button').length === 4 && !document.querySelector('.lost-bar').hidden", 60
        )
        bar = self.page.eval("document.querySelector('.lost-bar').innerText")
        self.assertIn("Diese Seite ist nicht mit dem Studio verbunden", bar)
        self.assertNotIn("Studio läuft nicht", bar)
        self.assertTrue(self.page.eval("[...document.querySelectorAll('.topbar, main.page')].every((el) => el.inert)"))
        self.assertEqual(self.page.eval("document.querySelectorAll('.notices .notice').length"), 0)


class AnalyzeRowsOfAStoppedRun(BrowserCase):
    """Bug 5: after a stop (or a failure) the Analyze screen's "This clip" and "All clips" rows said "done" while the clips said "Stopped"."""

    modules = [Delayed(FixedModule(FACE, [(10, 10, 60, 60)]), 0.25)]

    def rows(self):
        return self.page.eval(
            "[...document.querySelectorAll('main.page:not([hidden]) .row')].map((r) => r.innerText.replace(/\\s+/g, ' ')).join(' | ')"
        )

    def test_a_stopped_run_does_not_say_done(self):
        ids = [
            c["clip_id"]
            for c in self.api(
                "POST", "/api/clips", {"paths": [self.make_clip("a.mp4"), self.make_clip("b.mp4", n=30)]}
            )["clips"]
        ]
        self.reload_page()
        self.api("POST", "/api/job/analyze", {"clips": ids}, 202)
        self.page.wait_for("document.querySelectorAll('.tabs button')[1].disabled === false", 20)
        open_tab(self.page, 1)
        self.page.sleep(1.0)
        self.api("POST", "/api/job/stop")
        self.wait_idle()
        self.page.wait_for(
            "/Stopped/.test(document.querySelector('main.page:not([hidden])').innerText) && !/analyzing/i.test(document.querySelector('main.page:not([hidden])').innerText.split('PROGRESS')[0])",
            20,
        )
        rows = self.rows()
        self.assertIn("This clip Stopped", rows)
        self.assertIn("All clips 0 of 2 analyzed", rows)
        self.assertNotIn("done", rows)

    def test_a_failed_clip_does_not_say_done(self):
        a, b = self.make_clip("a.mp4"), self.make_clip("b.mp4", n=30)
        ids = [c["clip_id"] for c in self.api("POST", "/api/clips", {"paths": [a, b]})["clips"]]
        self.reload_page()
        self.api("POST", "/api/job/analyze", {"clips": ids}, 202)
        time.sleep(1.0)
        os.remove(b)  # the second clip's file goes while the first is being analyzed
        self.wait_idle()
        self.page.wait_for("document.querySelectorAll('.tabs button')[1].disabled === false", 20)
        open_tab(self.page, 1)
        self.page.wait_for("/Failed/.test(document.querySelector('main.page:not([hidden])').innerText)", 20)
        self.assertIn("All clips 1 of 2 analyzed", self.rows())


class ReviewOfAPhotoWithNothingToCheck(BrowserCase):
    """Bug 7 (2.1.7's "Found and not fixed"): Review offered "Only items to check" on a clip with nothing to check (a click showed an empty
    list), and the selected item of a photo showed "Visible -", "Length Photo" and two disabled "Go to" buttons."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_the_filter_is_not_offered_and_a_photo_item_has_no_time(self):
        from tests.ui.test_browser_v217 import save_photo

        path = save_photo(os.path.join(self.dir, "media", "one.jpg"))
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.reload_page()
        open_tab(self.page, 2)
        self.page.wait_for("document.querySelectorAll('.item-row, .item-list button').length >= 1", 30)
        only = "[...document.querySelectorAll('.review-toolbar button')].find((b) => /Only items to check/.test(b.innerText))"
        self.assertTrue(self.page.eval(f"{only}.disabled"))
        self.page.eval("document.querySelector('.item-row, .item-list button').click()")
        self.page.wait_for("!document.querySelector('.item-detail').hidden", 20)
        detail = self.page.eval(
            "[...document.querySelectorAll('.item-detail > *')].filter((e) => e.offsetParent).map((e) => e.innerText).join(' | ')"
        )
        self.assertNotIn("Visible", detail)
        self.assertNotIn("Length", detail)
        self.assertNotIn("Go to", detail)


class NarrowWindowHeaderAndLegend(BrowserCase):
    """Bug 8 (found by the audit of the four screens at 320 px in the seven languages): "Anonymizer Studio" wrapped over two lines and lay over the
    "Import" tab, and Review's legend ran past the window in German ("Unsicher" cut off)."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60), (100, 10, 150, 60)])]

    def test_nothing_lies_over_the_tabs_and_the_legend_stays_inside(self):
        from tests.test_ui_server import TOKEN
        from tests.ui.test_browser_v217 import save_photo

        clip_id = self.api("POST", "/api/clips", {"paths": [save_photo(os.path.join(self.dir, "media", "short.jpg"))]})[
            "clips"
        ][0]["clip_id"]
        self.analyze(clip_id)
        self.page.call("Emulation.setDeviceMetricsOverride", width=320, height=900, deviceScaleFactor=1, mobile=False)
        self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/?t={TOKEN}&lang=de")
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4", 30)
        brand = (
            "(() => { const b = document.querySelector('.brand strong').getBoundingClientRect(), t = document.querySelector('.tabs button').getBoundingClientRect();"
            " return !(b.width > 2 && b.right > t.left && b.left < t.right && b.bottom > t.top && b.top < t.bottom); })()"
        )
        self.assertTrue(self.page.eval(brand))
        self.assertEqual(
            self.page.eval("document.querySelector('.brand strong').textContent"), "Anonymizer Studio"
        )  # still there for a screen reader
        open_tab(self.page, 2)
        self.page.wait_for("document.querySelector('main.page:not([hidden]) .legend') !== null", 60)
        inside = self.page.eval(
            "[...document.querySelectorAll('main.page:not([hidden]) .legend span')].every((s) => s.getBoundingClientRect().right <= window.innerWidth)"
        )
        self.assertTrue(inside)


class CardNameAfterAReloadInTheMiddleOfAnExport(BrowserCase):
    """Bug 9 (2.1.7's "Found and not fixed"): after a reload in the middle of an export the card of the clip being written was headed with the
    plan's name (`x_anonymized.mp4`) while the render wrote `x_anonymized-2.mp4`, until the clip was done."""

    def test_the_card_names_the_file_being_written(self):
        clip_id = self.api("POST", "/api/clips", {"paths": [self.make_clip("x.mp4", n=90)]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder}}, 202)
        reader.wait_for("job.finished")
        self.wait_idle()
        real = anonymizer.Anonymizer.render

        def slow(self_, analysis, target, on_event=None, cancel=None):
            def handler(event):
                time.sleep(0.25)
                on_event(event)

            return real(self_, analysis, target, handler, cancel)

        with mock.patch.object(anonymizer.Anonymizer, "render", slow):
            self.api("POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder}}, 202)
            time.sleep(0.3)
            self.reload_page()
            open_tab(self.page, 3)
            self.page.wait_for("document.querySelector('.export-card') !== null", 30)
            card = self.page.eval("document.querySelector('.export-card').innerText")
            self.assertRegex(card, "Rendering|Queued")  # the render is still going while the card is read
            self.assertIn("x_anonymized-2.mp4", card.split("\n")[0] + card.split("\n")[1])
            self.wait_idle()


class FolderDialogSaysHomeInTheLanguage(BrowserCase):
    """Bug 10: the folder dialog's first place was "Home" in every language (the server names it)."""

    lang = "es"

    def test_the_place_is_in_the_page_language(self):
        self.page.eval("document.querySelector('.side-head button').click()")
        self.page.wait_for("document.querySelectorAll('dialog .explorer-row').length >= 1", 20)
        rows = self.page.eval(
            "[...document.querySelectorAll('dialog .explorer-row')].map((b) => b.innerText).join('|')"
        )
        self.assertIn("Inicio", rows)
        self.assertNotIn("Home", rows)


class StopSaysItIsStopping(BrowserCase):
    """Bug 11: a stop that takes a while (the first pass of a photo in Careful cannot be broken off: 7.6 s measured on a street photo) left the
    Stop button as it was and the screen as it was; a person clicks again, and wonders."""

    modules = [Delayed(FixedModule(FACE, [(10, 10, 60, 60)]), 3.0)]

    def test_analyze_says_stopping_and_offers_no_second_stop(self):
        from tests.ui.test_browser_v217 import save_photo

        clip_id = self.api("POST", "/api/clips", {"paths": [save_photo(os.path.join(self.dir, "media", "one.jpg"))]})[
            "clips"
        ][0]["clip_id"]
        self.reload_page()
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, 202)
        self.page.wait_for("document.querySelectorAll('.tabs button')[1].disabled === false", 20)
        open_tab(self.page, 1)
        stop = "[...document.querySelectorAll('main.page:not([hidden]) button')].find((b) => b.classList.contains('btn-danger'))"
        self.page.wait_for(f"{stop} && {stop}.innerText === 'Stop'", 20)
        self.click_text_stop = self.page.eval(f"{stop}.click()")
        self.page.wait_for(f"{stop}.innerText === 'Stopping...'", 20)
        self.assertTrue(self.page.eval(f"{stop}.disabled"))
        self.wait_idle()


if __name__ == "__main__":
    unittest.main()
