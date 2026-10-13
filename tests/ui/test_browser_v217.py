"""Studio v2.1.7: what the eighth client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.6 and passes here."""

from __future__ import annotations

import os
import unittest

try:
    from tests.fakes import FACE, Delayed, FixedModule
    from tests.test_ui_server import EventReader
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


def save_photo(path, width=900, height=200):
    import numpy as np
    from PIL import Image

    Image.fromarray((np.random.RandomState(3).rand(height, width, 3) * 255).astype(np.uint8)).save(path)
    return path


class PhotoOnTheAnalyzeScreen(BrowserCase):
    """A photo analysed, and the Analyze screen opened on it (the stand-in finds the faces of the class's `modules`)."""

    def analyzed_photo(self):
        path = save_photo(os.path.join(self.dir, "media", "crowd.jpg"))
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        return clip_id

    def open_analyze(self, items):
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")
        self.page.wait_for(f"document.querySelectorAll('.lane-track .bar-item').length >= {items}", 30)


class ManyMarksStayInTheirLane(PhotoOnTheAnalyzeScreen):
    """Bug 1: a photo's marks on the Analyze screen were 22 px apart with no end: 36 faces in one picture ran over the notes at the right."""

    modules = [FixedModule(FACE, [(10 + 24 * i, 20, 28 + 24 * i, 40) for i in range(36)])]

    def test_the_marks_stay_inside_their_lane_at_every_width(self):
        self.analyzed_photo()
        for width in (1280, 420):
            self.page.call(
                "Emulation.setDeviceMetricsOverride", width=width, height=900, deviceScaleFactor=1, mobile=False
            )
            self.open_analyze(30)
            outside = self.page.eval(
                "(() => { const t = document.querySelector('.lane-track').getBoundingClientRect();"
                " const marks = [...document.querySelectorAll('.lane-track .bar-mark')];"
                " return [marks.length, marks.filter(m => m.getBoundingClientRect().right > t.right + 1 || m.getBoundingClientRect().left < t.left - 1).length]; })()"
            )
            self.assertGreaterEqual(outside[0], 30, (width, outside))
            self.assertEqual(outside[1], 0, (width, outside))  # none sticks out of the track


class AFewMarksKeepTheirEvenSteps(PhotoOnTheAnalyzeScreen):
    """Bug 1 (the guard): while the marks fit, they stay 22 px apart."""

    modules = [FixedModule(FACE, [(10 + 24 * i, 20, 28 + 24 * i, 40) for i in range(3)])]

    def test_marks_are_22_px_apart_while_they_fit(self):
        self.analyzed_photo()
        self.open_analyze(3)
        lefts = self.page.eval(
            "(() => { const t = document.querySelector('.lane-track').getBoundingClientRect();"
            " return [...document.querySelectorAll('.lane-track .bar-mark')].map(m => Math.round(m.getBoundingClientRect().left - t.left)); })()"
        )
        self.assertEqual(lefts, [0, 22, 44])


class AnAddedAreaIsNotAFindOfTheDetectors(PhotoOnTheAnalyzeScreen):
    """Bug 2: an area the person added in Review was drawn on the first row of the Faces lane, over a face."""

    modules = [FixedModule(FACE, [(10 + 24 * i, 20, 28 + 24 * i, 40) for i in range(5)])]

    def test_an_area_added_in_review_is_not_a_find_of_the_detectors(self):
        clip_id = self.analyzed_photo()
        self.api(
            "POST",
            f"/api/clips/{clip_id}/areas",
            {"frame_from": 0, "frame_to": 0, "box": [100, 100, 160, 150], "follow": "fixed"},
            expect=201,
        )
        self.open_analyze(5)
        kinds = self.page.eval("[...document.querySelectorAll('.lane-track .bar-item')].map(b => b.className)")
        self.assertFalse([k for k in kinds if "bar-manual" in k], kinds)
        total = self.page.eval("document.querySelectorAll('.lane-track .bar-item').length")
        self.assertEqual(total, self.page.eval("document.querySelectorAll('.lane-track .bar-face').length"))


class VideoAnalyzeLanesIgnoreAddedAreas(BrowserCase):
    """Bug 2 (video): the added area was a white bar over the first row of the Faces lane."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_no_bar_of_an_added_area_in_the_lanes(self):
        path = self.make_clip("v.mp4", n=12)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.api(
            "POST",
            f"/api/clips/{clip_id}/areas",
            {"frame_from": 1, "frame_to": 8, "box": [20, 20, 60, 50], "follow": "fixed"},
            expect=201,
        )
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")
        self.page.wait_for("document.querySelectorAll('.lane-track .bar-item').length >= 1", 30)
        classes = self.page.eval("[...document.querySelectorAll('.lane-track .bar-item')].map(b => b.className)")
        self.assertFalse([c for c in classes if "bar-manual" in c], classes)


class TwoPhotosOnTheExportScreen(BrowserCase):
    """Two photos analysed and the Export screen open on them, the plan come."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def two_photos_exported(self):
        ids = []
        for name in ("a.jpg", "b.jpg"):
            path = save_photo(os.path.join(self.dir, "media", name), 160, 120)
            ids.append(self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"])
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": ids}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for(
            "[...document.querySelectorAll('.btn-primary')].some(b => /Render 2 clips/.test(b.textContent) && !b.disabled)",
            30,
        )  # the plan has come: the button is enabled
        return ids


class APlanAnsweredDuringTheRender(TwoPhotosOnTheExportScreen):
    """Bug 3: a plan the server answered in the middle of a render ("not possible" for the clips still waiting) stayed on the screen when the
    render ended, over clips that had been written, and the Render button counted only the first."""

    def test_a_plan_answered_during_the_render_does_not_outlive_it(self):
        self.two_photos_exported()
        # The first plan asked from now on is answered as the server answers it during a render: the clip that waits is "blocked". It arrives late.
        self.page.eval(
            """(() => {
              const real = window.fetch; let once = true;
              window.fetch = async (url, init) => {
                if (!String(url).includes('api/render/plan') || !once) return real(url, init);
                once = false;
                const answer = await (await real(url, init)).json();
                answer.clips = answer.clips.map((c, i) => (i === 0 ? c : { ...c, blocked: { code: 'error.clip.state', params: { status: 'queued' } } }));
                await new Promise((r) => setTimeout(r, 1500));
                return new Response(JSON.stringify(answer), { status: 200, headers: { 'Content-Type': 'application/json' } });
              };
            })()"""
        )
        self.page.eval("document.querySelectorAll('.chip')[1].click()")  # a change of options: the page asks for a plan
        self.page.sleep(0.5)
        self.page.eval("[...document.querySelectorAll('.btn-primary')].find(b => /Render/.test(b.textContent)).click()")
        self.page.wait_for("document.querySelectorAll('.export-files').length === 2", 60)
        self.page.sleep(3)  # the late answer has come and gone
        self.assertEqual(self.page.eval("document.querySelectorAll('.export-row .unchecked').length"), 0)
        self.assertIn(
            "Render 2 clips", self.page.eval("document.querySelector('main.page:not([hidden]) .cta').innerText")
        )


class TheCardOfAWrittenFile(TwoPhotosOnTheExportScreen):
    """Bug 4: the card of a written file was headed with the next free name, not the name that was written (after a second render or a reload)."""

    def test_the_card_of_a_written_file_names_what_was_written(self):
        ids = self.two_photos_exported()
        self.api(
            "POST",
            "/api/render",
            {
                "clips": ids,
                "options": {
                    "name_pattern": "{name}_anonymized",
                    "reports": {"pdf": False, "csv": False, "json": False},
                },
            },
            expect=202,
        )
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("document.querySelectorAll('.export-files').length === 2", 30)
        heads = self.page.eval(
            "[...document.querySelectorAll('.export-card')].map(c => c.querySelector('.card-line .mono').textContent)"
        )
        written = self.page.eval(
            "[...document.querySelectorAll('.export-card')].map(c => c.querySelector('.export-files .mono').textContent.split('/').pop())"
        )
        self.assertEqual(heads, written)


class ReviewHintsOfAPhoto(BrowserCase):
    """Bug 7: the Review screen told a photo's reader to "click a bar on the timeline" (a photo has a list, no timeline), and listed Space
    (play) and Left/Right (one frame), which do nothing on a photo."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_a_photo_and_a_video_each_get_their_own_hints(self):
        photo = save_photo(os.path.join(self.dir, "media", "p.jpg"), 160, 120)
        video = self.make_clip("v.mp4", n=12)
        ids = [self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"] for path in (photo, video)]
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": ids}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        page = "main.page:not([hidden])"

        def hints(name):
            self.page.eval(
                f"[...document.querySelectorAll('{page} .clip-card')].find(c => c.innerText.startsWith('{name}')).click()"
            )
            self.page.sleep(0.8)
            return self.page.eval(
                f"[document.querySelector('{page} .side-right .note:not(.shortcuts)').textContent, document.querySelector('{page} .shortcuts').innerText]"
            )

        said, keys = hints("p.jpg")
        self.assertIn("list", said)
        self.assertNotIn("timeline", said)
        self.assertNotIn("play", keys)
        self.assertNotIn("frame", keys)
        self.assertIn("next to check", keys)
        said, keys = hints("v.mp4")
        self.assertIn("timeline", said)
        self.assertIn("play/pause", keys)
        self.assertIn("one frame", keys)


class AFolderDroppedWithAFileThatCannotBeRead(BrowserCase):
    """Bug 8: a file in a dropped folder that the browser cannot read (no permission) made a notice that said "The engine did not answer:
    Failed to fetch"; the server was never asked. A real drop of files from disk, through the browser's own drag events."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_the_file_is_refused_by_name_and_the_others_come_in(self):
        folder = os.path.join(self.dir, "media", "shoot")
        os.makedirs(folder)
        save_photo(os.path.join(folder, "ok.jpg"), 160, 120)
        locked = save_photo(os.path.join(folder, "locked.jpg"), 160, 120)
        os.chmod(locked, 0)
        self.addCleanup(os.chmod, locked, 0o644)
        for kind in ("dragEnter", "dragOver", "drop"):
            self.page.call(
                "Input.dispatchDragEvent",
                type=kind,
                x=600,
                y=400,
                data={"items": [], "files": [folder], "dragOperationsMask": 1},
            )
        self.page.wait_for("document.querySelectorAll('.side-left .clip-card').length >= 1", 30)
        self.page.wait_for("/cannot be opened/.test(document.body.innerText)", 30)
        text = self.page.eval("document.body.innerText")
        self.assertIn("locked.jpg cannot be opened", text)
        self.assertNotIn("did not answer", text)
        names = self.page.eval("[...document.querySelectorAll('.side-left .clip-card')].map(c => c.innerText)")
        self.assertEqual(len(names), 1, names)
        self.assertIn("ok.jpg", names[0])


class AddAreaFormOfAPhoto(BrowserCase):
    """Bug 9: the form of "Add missed area" on a photo offered "Follows the camera / Stays in place" (a photo has one picture and no camera)."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_a_photo_has_no_follow_choice_and_a_video_has(self):
        photo = save_photo(os.path.join(self.dir, "media", "p.jpg"), 160, 120)
        video = self.make_clip("v.mp4", n=12)
        ids = [self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"] for path in (photo, video)]
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": ids}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        page = "main.page:not([hidden])"
        for name, follows in (("p.jpg", False), ("v.mp4", True)):
            self.page.eval(
                f"[...document.querySelectorAll('{page} .clip-card')].find(c => c.innerText.startsWith('{name}')).click()"
            )
            self.page.sleep(0.8)
            self.page.eval(
                f"[...document.querySelectorAll('{page} button')].find(b => b.textContent.trim() === 'Add missed area').click()"
            )
            x, y = self.page.center(f"{page} .stage")
            self.page.drag(x - 40, y - 30, x + 40, y + 30, steps=4, pause=0.03)
            self.page.wait_for(f"!document.querySelector('{page} .area-form').hidden", 10)
            shown = self.page.eval(
                f"[...document.querySelectorAll('{page} .area-form .segmented')].some(s => s.offsetParent !== null)"
            )
            self.assertEqual(shown, follows, name)
            self.page.eval(
                f"[...document.querySelectorAll('{page} .area-form button')].find(b => b.textContent.trim() === 'Cancel').click()"
            )


class AnalysisThatReportsNothingYetShowsItIsWorking(BrowserCase):
    """Bug 16: a photo in Careful reports no progress until its first pass ends (13 s for a street photo): the Analyze screen showed "0 %",
    "estimating" and an empty bar all that time, like a frozen page."""

    modules = [Delayed(FixedModule(FACE, [(10, 10, 60, 60)]), 4.0)]

    def test_the_bars_move_while_nothing_is_reported_and_stop_when_it_is_done(self):
        path = save_photo(os.path.join(self.dir, "media", "slow.jpg"), 160, 120)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        self.reload_page()
        self.page.wait_for("!!document.querySelector('main.page:not([hidden]) .bar.is-waiting .bar-fill')", 20)
        moving = self.page.eval(
            "[...document.querySelectorAll('main.page:not([hidden]) .bar.is-waiting .bar-fill')].map(b => getComputedStyle(b).animationName)"
        )
        self.assertEqual(set(moving), {"bar-waiting"})
        self.wait_idle()
        self.page.wait_for("!document.querySelector('main.page:not([hidden]) .bar.is-waiting')", 20)


class AFastExportSaysHowLongItTook(TwoPhotosOnTheExportScreen):
    """Bug 15 (2.1.6's "Found and not fixed"): a photo's card said "Done in 0 s"."""

    def test_it_is_not_zero_seconds(self):
        ids = self.two_photos_exported()
        self.api(
            "POST",
            "/api/render",
            {"clips": ids, "options": {"reports": {"pdf": False, "csv": False, "json": False}}},
            expect=202,
        )
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("document.querySelectorAll('.export-files').length === 2", 30)
        said = self.page.eval("[...document.querySelectorAll('.ok-text')].map(e => e.textContent)")
        self.assertEqual(len(said), 2)
        for text in said:
            self.assertRegex(text, r"^Done in \d\.\d s$")


class ExportCardsHaveASpaceBetweenThem(TwoPhotosOnTheExportScreen):
    """Bug 10: the cards of the Export list touched each other (the gap was set on a container that is not their parent)."""

    def test_the_cards_do_not_touch(self):
        self.two_photos_exported()
        gaps = self.page.eval(
            "(() => { const c = [...document.querySelectorAll('main.page:not([hidden]) .export-card')].map(e => e.getBoundingClientRect());"
            " return c.slice(1).map((r, i) => Math.round(r.top - c[i].bottom)); })()"
        )
        self.assertEqual(len(gaps), 1)
        self.assertTrue(all(g >= 8 for g in gaps), gaps)


class ALongFileNameStaysInsideTheHeader(BrowserCase):
    """Bug 11: a long file name with no space in it (a camera's or an editor's) pushed the "Source" heading of the Import screen, and the Fit
    button with it, far past the window: the page scrolled sideways. The same for the heading of the Analyze screen."""

    modules = [Delayed(FixedModule(FACE, [(10, 10, 60, 60)]), 3.0)]

    def test_import_and_analyze_headings_are_cut_short(self):
        name = "Very_long_" * 14 + "x.jpg"
        path = save_photo(os.path.join(self.dir, "media", name), 160, 120)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        for width in (1280, 420):
            self.page.call(
                "Emulation.setDeviceMetricsOverride", width=width, height=900, deviceScaleFactor=1, mobile=False
            )
            self.reload_page()
            self.page.wait_for(
                "document.querySelector('main.page:not([hidden]) .source-title').textContent.length > 10", 20
            )
            fits = self.page.eval(
                "[document.documentElement.scrollWidth <= window.innerWidth,"
                " [...document.querySelectorAll('main.page:not([hidden]) .center-head > *')].every(e => e.getBoundingClientRect().right <= window.innerWidth + 1)]"
            )
            self.assertEqual(fits, [True, True], width)
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")
        self.page.wait_for("!!document.querySelector('main.page:not([hidden]) .center-head')", 20)
        self.page.wait_for(
            "/Very_long/.test(document.querySelector('main.page:not([hidden]) .source-title').textContent)", 20
        )
        fits = self.page.eval(
            "[document.documentElement.scrollWidth <= window.innerWidth,"
            " [...document.querySelectorAll('main.page:not([hidden]) .center-head > *')].every(e => e.getBoundingClientRect().right <= window.innerWidth + 1)]"
        )
        self.assertEqual(fits, [True, True])
        self.wait_idle()


class TheTestChromeLeavesNoProfileBehind(unittest.TestCase):
    """Bug 12 (the tests' own): `Page.close()` stopped Chrome and left its temporary profile (about 50 MB) in /tmp, for every caller that did
    not remove it itself (the screenshot, walk-through and demo-video scripts, a second window): hundreds of them were in /tmp on 05/10/2026."""

    def test_the_profile_goes_with_the_page(self):
        import shutil

        if shutil.which("google-chrome") is None:
            self.skipTest("google-chrome is not installed")
        from tests.ui.browser_case import free_port
        from tests.ui.drive import Page

        page = Page("about:blank", width=400, height=300, port=free_port())
        profile = page.profile
        self.assertTrue(os.path.isdir(profile))
        page.close()
        self.assertFalse(os.path.exists(profile))


class ADropBeforeTheStateHasArrived(BrowserCase):
    """Bug 13 (found by the full suite on a slow computer: `test_browser_v216.DroppedFolderSendsOnlyWhatTheServerCanTake` failed there): a folder
    dropped before the page had its state sent every file in it, as 2.1.5 did, because the lists of what the Studio takes come with the state."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_the_folder_is_sorted_even_when_the_state_comes_late(self):
        from tests.ui import test_browser_v216 as v216

        self.page.call(
            "Page.addScriptToEvaluateOnNewDocument",
            source="""(() => { const real = window.fetch;
              window.fetch = (url, init) => String(url).includes('api/state') && !String(url).includes('?')
                ? new Promise((resolve) => setTimeout(() => resolve(real(url, init)), 2500)) : real(url, init); })()""",
        )
        self.page.eval("location.reload()")
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4", 60)
        self.assertIsNone(
            self.page.eval("document.querySelector('.options') ? null : 1")
        )  # the page is up, its state is not
        self.page.eval(v216.DroppedFolderSendsOnlyWhatTheServerCanTake.DROP)
        self.page.wait_for("window.__puts.length >= 2", 30)
        self.page.sleep(1)
        sent = self.page.eval("JSON.stringify(window.__puts)")
        self.assertEqual(sorted(__import__("json").loads(sent)), ["deep.png", "good.png"], sent)


if __name__ == "__main__":
    unittest.main()
