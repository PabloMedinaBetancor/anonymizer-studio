"""Studio v2.1.6: what the seventh client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.5 and passes here."""

from __future__ import annotations

import os
import unittest

try:
    from tests.fakes import FACE, FixedModule
    from tests.test_ui_server import EventReader
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class PageFollowsTheBrowsersLanguage(BrowserCase):
    """Bug 1: a first visit opened in English whatever the browser's language, and a language chosen once could not be told from none."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]
    lang = ""  # the address carries no ?lang=: the page decides

    def open_in(self, accept_language):
        self.page.call(
            "Emulation.setUserAgentOverride",
            userAgent=self.page.eval("navigator.userAgent"),
            acceptLanguage=accept_language,
        )
        self.reload_page()

    def tab_names(self):
        return self.page.eval("[...document.querySelectorAll('.tabs button')].map(b => b.textContent).join('|')")

    def test_a_german_browser_gets_german_and_a_choice_of_english_is_kept(self):
        self.open_in("de-DE,de")
        self.assertEqual(self.page.eval("document.documentElement.lang"), "de")
        self.assertEqual(self.page.eval("document.querySelector('.btn-lang').textContent"), "DE")
        self.open_in("ja,fr-CA")  # the first language the Studio has
        self.assertEqual(self.page.eval("document.documentElement.lang"), "fr")
        self.open_in("ja")  # none it has
        self.assertEqual(self.page.eval("document.documentElement.lang"), "en")
        self.open_in("de")
        self.page.eval("document.querySelector('.btn-lang').click()")
        self.page.wait_for("!!document.querySelector('.language-dialog')", 10)
        self.page.eval(
            "[...document.querySelectorAll('.language-dialog .explorer-row')].find(b => b.textContent === 'English').click()"
        )
        self.page.sleep(1.5)
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4", 30)
        self.assertEqual(self.page.eval("document.documentElement.lang"), "en")
        self.assertEqual(self.api("GET", "/api/settings")["language"], "en")
        self.page.call(
            "Page.navigate", url=f"http://127.0.0.1:{self.port}/"
        )  # the address with no ?lang=, a German browser, English chosen
        self.page.sleep(1.5)
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4", 30)
        self.assertEqual(self.page.eval("document.documentElement.lang"), "en")


class DroppedFolderSendsOnlyWhatTheServerCanTake(BrowserCase):
    """Bug 2: a folder dropped on Import uploaded every file in it, media or not, before the server refused it (a 3 GB archive was copied into the
    work folder and deleted again), and said nothing of the files it dropped."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    DROP = """
    window.__puts = [];
    const realFetch = window.fetch;
    window.fetch = (url, init) => { if (String(url).includes('api/upload')) window.__puts.push(decodeURIComponent(String(url).split('name=')[1])); return realFetch(url, init); };
    const file = (name, bytes) => new File([bytes], name);
    const png = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='), c => c.charCodeAt(0));
    const fileEntry = (name, bytes) => ({ isFile: true, isDirectory: false, name, file: (ok) => ok(file(name, bytes)) });
    const dirEntry = (name, children) => ({ isFile: false, isDirectory: true, name,
      createReader: () => { let given = false; return { readEntries: (ok) => { const batch = given ? [] : children; given = true; ok(batch); } }; } });
    const tree = dirEntry('Shoot', [fileEntry('good.png', png), fileEntry('archive.zip', new Uint8Array(10)), fileEntry('photo.HEIC', new Uint8Array(10)),
      fileEntry('.hidden.png', png), dirEntry('anonymized', [fileEntry('old.png', png)]), dirEntry('Sub', [fileEntry('deep.png', png), fileEntry('notes.txt', new Uint8Array(3))])]);
    const transfer = new DataTransfer();
    transfer.items.add(file('Shoot', new Uint8Array(1)));
    DataTransferItem.prototype.webkitGetAsEntry = () => tree;
    document.querySelector('main.page').dispatchEvent(new DragEvent('drop', { dataTransfer: transfer, bubbles: true, cancelable: true }));
    """

    def test_only_media_goes_up_and_the_rest_is_said(self):
        self.page.eval(self.DROP)
        self.page.wait_for("window.__puts.length >= 2", 30)
        self.page.sleep(1)
        sent = self.page.eval("JSON.stringify(window.__puts)")
        self.assertEqual(sorted(__import__("json").loads(sent)), ["deep.png", "good.png"], sent)
        notices = self.page.eval("[...document.querySelectorAll('.notice')].map(n => n.textContent).join(' | ')")
        self.assertIn("photo.HEIC", notices)
        self.assertRegex(notices, r"2 files were not recognized and were left out \((\.txt, \.zip|\.zip, \.txt)\)")
        self.assertIn("hidden", notices.lower())


class AnalyzeScreenForPhotos(BrowserCase):
    """Bug 3: the Analyze screen drew a photo's items as bars across the whole width (a timeline of one frame), and "Settings used" named the
    audio and the gap filling for a project of photos only."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def photo(self, name):
        import numpy as np
        from PIL import Image

        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(3).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        return path

    def analyzed_screen(self):
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api(
            "POST",
            "/api/job/analyze",
            {"clips": [c["clip_id"] for c in self.api("GET", "/api/clips")["clips"]]},
            expect=202,
        )
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")
        self.page.wait_for("document.querySelectorAll('.lane-track .bar-item').length >= 1", 30)

    def test_a_photo_has_marks_and_no_audio_in_its_settings(self):
        self.import_paths(self.photo("one.jpg"))
        self.analyzed_screen()
        widths = self.page.eval(
            "[...document.querySelectorAll('.lane-track .bar-item')].map(b => Math.round(b.getBoundingClientRect().width))"
        )
        self.assertTrue(widths and max(widths) <= 24, widths)  # marks, not bars across the screen
        self.assertEqual(self.page.eval("document.querySelectorAll('.lane-track .playhead').length"), 0)
        line = self.page.eval("document.querySelector('.settings-used').textContent")
        self.assertNotIn("Audio", line)
        self.assertNotIn("Gaps", line)

    def test_a_project_with_a_video_keeps_bars_and_the_audio(self):
        self.import_paths(self.photo("one.jpg"), self.make_clip("v.mp4", n=12))
        self.analyzed_screen()
        line = self.page.eval("document.querySelector('.settings-used').textContent")
        self.assertIn("Audio", line)
        self.assertIn("Gaps", line)


class SampleDialogAndEmptyProject(BrowserCase):
    """Bug 4: the demo's sample dialog was an empty box while the list loaded (and for good when there was none), and the Import screen of an
    empty project said "Analyze 0 clips" on a disabled button."""

    demo = True

    def test_the_sample_dialog_says_it_is_loading_and_then_that_there_is_none(self):
        self.page.eval(
            "const f = window.fetch; window.fetch = (u, i) => String(u).includes('api/samples') ? new Promise(r => setTimeout(() => r(f(u, i)), 1500)) : f(u, i);"
        )
        self.page.eval("document.querySelector('.side-head .btn').click()")
        self.page.wait_for("!!document.querySelector('.sample-list')", 10)
        self.assertEqual(
            self.page.eval("document.querySelector('.sample-list').textContent.trim()"), "Loading the sample clips..."
        )
        self.page.wait_for("document.querySelector('.sample-list').textContent.includes('No sample clips')", 10)

    def test_an_empty_project_says_analyze_and_not_analyze_zero_clips(self):
        # the button's text is written by the page's first render, after its state has come: not yet when the tabs are drawn (red on a slow machine, now and then)
        self.page.wait_for(
            "[...document.querySelectorAll('button.btn-primary')].some(b => /^Analyze/.test(b.textContent))", 60
        )
        label = self.page.eval(
            "[...document.querySelectorAll('button.btn-primary')].map(b => b.textContent).find(t => /^Analyze/.test(t))"
        )
        self.assertEqual(label, "Analyze")


class ScrollbarsAreThePagesOwn(BrowserCase):
    """Bug 5: Firefox drew the scrollbar of a side panel as a white strip in the dark page (found in Firefox; the page set only color-scheme)."""

    def test_the_scrollbar_has_a_colour_of_the_page(self):
        colour = self.page.eval("getComputedStyle(document.querySelector('.options')).scrollbarColor")
        self.assertNotEqual(colour, "auto")


class AddAreaFormKeepsItsWords(BrowserCase):
    """Bug 7: the form of "Add missed area" is as wide as the picture (648 px at 1280); its buttons and its choice were squeezed until their words broke
    over two lines ("Add" over "area") in English and worse in German, found in Firefox."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]
    lang = "de"

    def test_nothing_in_the_form_breaks_a_word_over_two_lines(self):
        self.assertEqual(self.page.eval("document.documentElement.lang"), "de")  # the address asked for German
        clip = self.import_clip("form.mp4", n=12)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": [clip]}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("!!document.querySelector('.review-toolbar')", 30)
        for width in (1280, 420):
            self.resize(width, 900)
            self.page.sleep(0.5)
            self.page.eval(
                "document.querySelector('.area-form').hidden || document.querySelector('.area-form .btn-small').click()"
            )
            self.page.eval(
                "!document.querySelector('.area-form').hidden || document.querySelector('.review-toolbar .btn-small:last-child').click()"
            )
            r = self.page.eval(
                "(() => { const r = document.querySelector('.page-review .stage').getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; })()"
            )
            self.page.drag(r[0] + r[2] * 0.1, r[1] + r[3] * 0.1, r[0] + r[2] * 0.3, r[1] + r[3] * 0.3)
            self.page.wait_for("!document.querySelector('.area-form').hidden", 10)
            heights = self.page.eval(
                "[...document.querySelectorAll('.area-form button')].map(b => Math.round(b.getBoundingClientRect().height))"
            )
            self.assertTrue(heights and all(h <= 44 for h in heights), (width, heights))
            box = self.page.eval(
                "(() => { const f = document.querySelector('.area-form').getBoundingClientRect(), s = document.querySelector('.page-review .stage').getBoundingClientRect(); return [f.left - s.left, f.right - s.right]; })()"
            )
            self.assertTrue(
                box[0] >= -12 and box[1] <= 12, (width, box)
            )  # the form stays under the picture, not past it
            for button in self.page.eval(
                "[...document.querySelectorAll('.area-form button')].map(b => b.scrollWidth > b.clientWidth + 1 || b.scrollHeight > b.clientHeight + 1)"
            ):
                self.assertFalse(button, width)


class ExportScreenWithTwoSourceFolders(BrowserCase):
    """Bug 8 (the page's part): "Save to" and the heading of the file list gave the first clip's folder as where everything goes."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_it_says_next_to_each_source_and_not_one_folder(self):
        import numpy as np
        from PIL import Image

        ids = []
        for folder in ("one", "two"):
            os.makedirs(os.path.join(self.dir, folder))
            path = os.path.join(self.dir, folder, "p.jpg")
            Image.fromarray((np.random.RandomState(1).rand(60, 80, 3) * 255).astype(np.uint8)).save(path)
            ids.append(self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"])
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api("POST", "/api/job/analyze", {"clips": ids}, expect=202)
        reader.wait_for("job.finished")
        self.wait_idle()
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for(
            "document.querySelector('.export-where') && document.querySelector('.export-where').textContent !== ''", 30
        )
        self.assertEqual(
            self.page.eval("document.querySelector('.export-where').textContent"),
            "A folder named anonymized next to each source",
        )
        self.assertEqual(
            self.page.eval("document.querySelector('#export-destination').placeholder"),
            "A folder named anonymized next to each source",
        )


class ImportScreenOffersNoMaximum(BrowserCase):
    """A release review found that the Import screen had a "Thoroughness" choice, Standard or Maximum; Maximum is not repeatable on video and no better, and is not shown."""

    def test_no_thoroughness_choice(self):
        text = self.page.eval("document.querySelector('.options').innerText")
        self.assertNotIn("Maximum", text)
        self.assertNotIn("Thoroughness", text)
        self.assertIn("Careful", text)


if __name__ == "__main__":
    unittest.main()
