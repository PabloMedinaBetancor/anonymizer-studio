"""Studio v2.1.3: what the fourth client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.2 and passes here."""

from __future__ import annotations

import json
import os
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests.fakes import FACE, Delayed, FixedModule, plate_module
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class ReloadDuringAnalysis(BrowserCase):
    """Bug 8: a page reloaded (or opened late) while a clip was being analysed landed on the Import screen with the clip at 0 % until
    its next progress event, and a clip that had failed said "Failed" with no reason."""

    modules = [Delayed(plate_module(), 0.15)]

    def test_the_page_comes_back_on_the_analysis_with_the_progress_it_has(self):
        clip_id = self.import_clip("long.mp4", n=120)
        self.reload_page()
        self.api("POST", "/api/job/analyze", {"clips": [clip_id]}, expect=202)
        self.page.wait_for("document.querySelector('.tabs button[aria-current=page]')?.textContent === 'Analyze'", 30)
        self.page.wait_for("/[1-9]\\d?%/.test(document.querySelector('.clip-card-progress')?.innerText || '')", 60)
        self.reload_page()
        self.assertEqual(
            self.page.eval("document.querySelector('.tabs button[aria-current=page]').textContent"), "Analyze"
        )
        shown = self.page.eval("document.querySelector('.clip-card-progress').innerText")
        self.assertRegex(shown, r"[1-9]\d?%", shown)  # not the 0 % of a clip the page knows nothing about
        self.page.wait_for("document.querySelectorAll('.tabs button')[2].disabled === false", 120)

    def test_a_clip_that_failed_keeps_its_reason_in_the_state(self):
        clip_id = self.import_clip("f.mp4", n=10)
        session = self.server.session
        clip = session.clips[clip_id]
        clip.status = "failed"
        import anonymizer

        session._error(clip, anonymizer.UnsupportedMedia("error.media.unreadable_video", {"name": "f.mp4"}))
        summary = self.api("GET", "/api/state")["clips"][0]
        self.assertEqual(summary["failure"], {"code": "error.media.unreadable_video", "params": {"name": "f.mp4"}})
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")


class ExportOptionsSurviveAReload(BrowserCase):
    """Bug 9: the destination, the file name, the reports and the comparison video typed on the Export screen, and the clips taken off its
    list, went back to their defaults when the page was reloaded or its language changed (which reloads it); the page also came back on
    Import."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def photo(self, name):
        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(7).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        return path

    def test_what_was_typed_on_export_is_still_there_after_a_reload_and_after_a_change_of_language(self):
        ids = [
            c["clip_id"]
            for c in self.api("POST", "/api/clips", {"paths": [self.photo("a.jpg"), self.photo("b.jpg")]})["clips"]
        ]
        self.analyze(*ids)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[3].disabled", 30)
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("!!document.querySelector('#export-pattern')")
        target = os.path.join(self.dir, "elsewhere")
        self.page.eval(
            "(() => { for (const [id, value] of [['export-destination', %s], ['export-pattern', 'x_{index}']]) {"
            " const input = document.getElementById(id); input.value = value; input.dispatchEvent(new Event('input', {bubbles: true})); } })()"
            % json.dumps(target)
        )
        self.click(".page-export .chip:nth-of-type(2)")  # the CSV report off
        self.click(".page-export .check")  # the first clip off the list
        facts = (
            "JSON.stringify([[...document.querySelectorAll('.page-export .chip')].map((c) => c.getAttribute('aria-pressed')),"
            " [...document.querySelectorAll('.page-export .check')].map((c) => c.getAttribute('aria-checked'))])"
        )
        before = self.page.eval(facts)
        self.assertEqual(json.loads(before)[1], ["false", "true"])
        self.reload_page()
        self.page.wait_for("!!document.getElementById('export-pattern')", 30)
        self.assertEqual(
            self.page.eval("document.querySelector('.tabs button[aria-current=page]').textContent"), "Export"
        )
        self.assertEqual(self.page.eval("document.getElementById('export-destination').value"), target)
        self.assertEqual(self.page.eval("document.getElementById('export-pattern').value"), "x_{index}")
        self.assertEqual(self.page.eval(facts), before)
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")  # Review, on the second clip
        self.page.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card').length === 2", 30)
        self.page.eval("document.querySelectorAll('.page:not([hidden]) .clip-card')[1].click()")
        self.page.wait_for(
            "document.querySelectorAll('.page:not([hidden]) .clip-card')[1].getAttribute('aria-pressed') === 'true'", 30
        )
        self.reload_page()
        self.page.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card').length === 2", 30)
        self.assertEqual(
            self.page.eval(
                "[...document.querySelectorAll('.page:not([hidden]) .clip-card')].map((c) => c.getAttribute('aria-pressed')).join()"
            ),
            "false,true",
        )  # the clip that was open is the one that comes back
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.eval("location.search = '?lang=es'")  # what picking another language does
        self.page.sleep(1)
        self.page.wait_for(
            "document.querySelectorAll('.tabs button').length === 4 && !!document.getElementById('export-pattern')", 60
        )
        self.assertEqual(self.page.eval("document.documentElement.lang"), "es")
        self.assertEqual(self.page.eval("document.getElementById('export-pattern').value"), "x_{index}")
        self.assertEqual(self.page.eval("document.getElementById('export-destination').value"), target)


class LongNamesBreakAtSeparators(BrowserCase):
    """Bug 10: in the 280 px column of the Import screen a long file name wrapped in the middle of a word ("madrid.mp" and "4")."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]
    NAMES = [
        "Living_room_view_from_the_balcony_2024-01-15_final.jpg",
        "street-madrid-\u00f1and\u00fa.jpg",
        "IMG_20240115_123456_HDR_PANO_FULL.jpg",
    ]

    def test_each_line_of_a_name_ends_at_a_separator(self):
        paths = []
        for name in self.NAMES:
            path = os.path.join(self.dir, "media", name)
            Image.fromarray((np.random.RandomState(7).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
            paths.append(path)
        self.import_paths(*paths)
        self.page.wait_for("document.querySelectorAll('.clip-row .clip-name').length === 3", 30)
        lines = json.loads(
            self.page.eval(
                """JSON.stringify([...document.querySelectorAll('.clip-row .clip-name')].map((el) => {
                    const rows = []; const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
                    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
                        for (let i = 0; i < node.length; i++) {
                            const range = document.createRange(); range.setStart(node, i); range.setEnd(node, i + 1);
                            const rect = range.getClientRects()[0]; if (!rect) continue;
                            if (!rows.length || Math.abs(rows[rows.length - 1].top - rect.top) > 4) rows.push({top: rect.top, text: ''});
                            rows[rows.length - 1].text += node.data[i];
                        }
                    }
                    return rows.map((row) => row.text);
                }))"""
            )
        )
        for name, rows in zip(self.NAMES, lines, strict=True):
            self.assertEqual("".join(rows), name)  # the name is whole (a <wbr> adds no text)
            for row in rows[:-1]:
                self.assertIn(row[-1], "_-. ", f"{name}: a line ends in the middle of a word: {rows}")


class PhotoReviewShowsNoClock(BrowserCase):
    """Bug 12: Review showed "00:00:00:00 / 00:00:00:00" under every photo, which has no time."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_a_photo_has_no_timecode_but_a_video_has_one(self):
        photo = os.path.join(self.dir, "media", "p.jpg")
        Image.fromarray((np.random.RandomState(7).rand(120, 160, 3) * 255).astype(np.uint8)).save(photo)
        video = self.make_clip("v.mp4", n=30)
        ids = [c["clip_id"] for c in self.api("POST", "/api/clips", {"paths": [photo, video]})["clips"]]
        self.analyze(*ids)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled", 30)
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for(
            "document.querySelectorAll('.page-review .clip-card, .page:not([hidden]) .clip-card').length >= 2", 30
        )
        visible = "(() => { const e = document.querySelector('.page:not([hidden]) .timecode'); return !!e && e.getClientRects().length > 0; })()"
        names = (
            "[...document.querySelectorAll('.page:not([hidden]) .clip-card')].map((c) => c.innerText.split('\\n')[0])"
        )
        self.assertEqual(self.page.eval(names)[0], "p.jpg")
        self.page.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card').length >= 2", 30)
        self.click(".page:not([hidden]) .clip-card")  # the photo (first in the list)
        self.page.sleep(0.5)
        self.assertFalse(self.page.eval(visible))
        self.page.eval("document.querySelectorAll('.page:not([hidden]) .clip-card')[1].click()")
        self.page.sleep(0.5)
        self.assertTrue(self.page.eval(visible))


class DropAnywhereOnThePage(BrowserCase):
    """Bug 13: only the small box at the bottom of the clip list took a dropped file; dropped anywhere else (the picture, the options, any
    other screen) the browser opened the file and the person left the Studio."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    DROP = """(async (selector) => {
        const canvas = document.createElement('canvas'); canvas.width = 40; canvas.height = 30;
        canvas.getContext('2d').fillRect(0, 0, 20, 20);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
        const transfer = new DataTransfer(); transfer.items.add(new File([blob], 'dropped.jpg', {type: 'image/jpeg'}));
        const over = new DragEvent('dragover', {dataTransfer: transfer, bubbles: true, cancelable: true});
        document.querySelector(selector).dispatchEvent(over);
        const drop = new DragEvent('drop', {dataTransfer: transfer, bubbles: true, cancelable: true});
        document.querySelector(selector).dispatchEvent(drop);
        return [over.defaultPrevented, drop.defaultPrevented, transfer.dropEffect];
    })(%s)"""

    def test_a_file_dropped_on_the_picture_is_imported_and_the_browser_does_not_open_it(self):
        over, dropped, _ = self.page.eval(self.DROP % json.dumps(".center-body"))
        self.assertEqual((over, dropped), (True, True))
        self.page.wait_for("document.querySelectorAll('.clip-row').length === 1", 30)
        self.assertIn("dropped.jpg", self.page.eval("document.querySelector('.clip-row').innerText"))
        self.assertTrue(self.page.eval("location.pathname === '/'"))

    def test_on_another_screen_the_drop_is_refused_but_the_page_stays(self):
        over, dropped, effect = self.page.eval(self.DROP % json.dumps(".tabs"))
        self.assertEqual((over, dropped, effect), (True, True, "none"))
        self.page.sleep(0.5)
        self.assertEqual(self.page.eval("document.querySelectorAll('.clip-row').length"), 0)
        self.assertTrue(self.page.eval("location.pathname === '/'"))


class FocusAfterCancellingAnalyzeAgain(BrowserCase):
    """Bug 16: after "Analyze again and discard your review?" was cancelled with the keyboard, the focus was on the page's body (the
    button had been disabled while the question stood), so the next Tab started again at the top of the page."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)], 0.85, 0.75)]

    def test_cancel_returns_the_focus_to_the_analyze_button(self):
        path = os.path.join(self.dir, "media", "p.jpg")
        Image.fromarray((np.random.RandomState(7).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        clip = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip)
        item = self.api("GET", f"/api/clips/{clip}/analysis")["items"][0]
        self.api("PATCH", f"/api/clips/{clip}/items/{item['item_id']}", {"blur": False})  # a decision of the person
        self.reload_page()
        self.page.eval(
            "document.querySelectorAll('.tabs button')[0].click()"
        )  # the page comes back on the screen it was on
        self.page.sleep(
            1.5
        )  # the stream's first answer reloads the state: a choice made before it would be overwritten
        self.page.eval(
            "[...document.querySelectorAll('.segmented button, [role=radio]')].find((b) => /^Fast/.test(b.textContent.trim())).click()"
        )
        self.page.wait_for(
            "!document.querySelector('.cta .btn-primary').disabled", 30
        )  # (v2.1.5: it is enabled by what is left to do)
        self.page.eval("document.querySelector('.cta .btn-primary').focus()")
        self.page.key(" ", "Space")  # the button is pressed with the keyboard
        self.page.wait_for("!!document.querySelector('dialog[open]')", 15)
        self.page.key("Escape", "Escape")
        self.page.wait_for("!document.querySelector('dialog[open]')", 15)
        self.page.sleep(0.3)
        self.assertTrue(self.page.eval("document.activeElement === document.querySelector('.cta .btn-primary')"))


class HostedDemoTakesFilesByClick(BrowserCase):
    """Bug 19: the hosted demo said "Use Import" and "Choose files or whole folders" on a page whose only button was "Samples": a file could
    only be brought by dragging it, which a tablet or a phone cannot do."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]
    demo = True

    PICK = """(async () => {
        const canvas = document.createElement('canvas'); canvas.width = 40; canvas.height = 30;
        canvas.getContext('2d').fillRect(0, 0, 20, 20);
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
        const transfer = new DataTransfer(); transfer.items.add(new File([blob], 'chosen.jpg', {type: 'image/jpeg'}));
        const input = document.querySelector('input[type=file]');
        input.files = transfer.files;
        input.dispatchEvent(new Event('change', {bubbles: true}));
        return [input.multiple, input.accept];
    })()"""

    def test_the_box_under_the_list_opens_the_chooser_and_what_it_gives_is_imported(self):
        zone = "document.querySelector('.drop-zone')"
        self.page.wait_for(f"{zone}.getAttribute('role') === 'button'", 30)
        self.assertIn("click to choose files", self.page.eval(f"{zone}.textContent"))
        self.assertIn("open the Samples", self.page.eval("document.querySelector('.clip-list .empty').textContent"))
        self.assertEqual(self.page.eval(f"{zone}.tabIndex"), 0)
        multiple, accept = self.page.eval(self.PICK)
        self.assertEqual((multiple, accept), (True, "image/*,video/*"))
        self.page.wait_for("document.querySelectorAll('.clip-row').length === 1", 30)
        self.assertIn("chosen.jpg", self.page.eval("document.querySelector('.clip-row').innerText"))


class LocalStudioKeepsItsDropBoxAsItWas(BrowserCase):
    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def test_it_is_not_a_button_and_says_what_it_said(self):
        self.page.wait_for("document.querySelector('.drop-zone').textContent.length > 0", 30)
        self.assertIsNone(self.page.eval("document.querySelector('.drop-zone').getAttribute('role')"))
        self.assertEqual(
            self.page.eval("document.querySelector('.drop-zone').textContent"),
            "Drop photos, videos or whole folders here",
        )


class DoubleClickIsOneAction(BrowserCase):
    """Bug 23: the second click of a double click fell on what the list had closed up under the pointer: a double click on the cross of a clip took
    two clips off the list (with the analysis and the decisions of the second), and on a folder of the file picker it opened the folder and then
    chose or opened whatever lay under the pointer in it."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def double_click(self, selector):
        x, y = self.page.center(selector)
        for count in (1, 2):
            self.page.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)
            self.page.call(
                "Input.dispatchMouseEvent", type="mousePressed", x=x, y=y, button="left", buttons=1, clickCount=count
            )
            self.page.call(
                "Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y, button="left", buttons=0, clickCount=count
            )
            self.page.sleep(0.08)
        self.page.sleep(0.6)

    def test_a_double_click_on_a_cross_removes_one_clip(self):
        paths = []
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            path = os.path.join(self.dir, "media", name)
            Image.fromarray((np.random.RandomState(7).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
            paths.append(path)
        self.import_paths(*paths)
        self.page.wait_for("document.querySelectorAll('.clip-row').length === 3", 30)
        self.double_click(".clip-row:nth-child(2) .clip-remove")
        names = [c["name"] for c in self.api("GET", "/api/clips")["clips"]]
        self.assertEqual(names, ["a.jpg", "c.jpg"])

    def test_a_double_click_on_a_folder_opens_it_once(self):
        for folder in ("tree/a/b1", "tree/a/b2"):
            os.makedirs(os.path.join(self.home, folder))
        self.page.eval("document.querySelector('.side-head .btn-small').click()")
        self.page.wait_for("!!document.querySelector('dialog[open] .explorer-row')", 30)
        self.page.eval(
            "[...document.querySelectorAll('dialog[open] .explorer-row')].find((b) => b.textContent === 'Home').click()"
        )
        self.page.wait_for(
            "document.querySelector('dialog[open] .explorer-path').textContent === %s" % json.dumps(self.home), 30
        )
        self.double_click("dialog[open] li:nth-child(2) .explorer-row")  # the first row is ".."; this one is `tree`
        self.assertEqual(
            self.page.eval("document.querySelector('dialog[open] .explorer-path').textContent"),
            os.path.join(self.home, "tree"),
        )
