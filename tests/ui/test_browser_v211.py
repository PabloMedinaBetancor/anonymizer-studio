"""Studio v2.1.1, what the second client QA pass (04/10/2026, qa-anon) found in the page: each test drives the real page in a
headless Chrome and fails on v2.1."""

from __future__ import annotations

import os
import time
import unittest

try:
    import numpy as np
    from PIL import Image

    from anonymizer.contract import FACE, PLATE
    from tests import media
    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule, doubtful_plate_module
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries: {e}") from e


class ImportButtonOnTheListOfPlaces(BrowserCase):
    """Bug 1: "Import" with nothing ticked on the list of places sent the empty path, which the server took for its own working
    directory: everything below it was imported."""

    def open_picker(self):
        self.page.eval("document.querySelector('.side-head button').click()")
        self.page.wait_for("document.querySelectorAll('.explorer-list li').length > 0")

    def test_the_button_waits_until_a_folder_is_open_or_a_file_is_ticked(self):
        Image.fromarray((np.random.RandomState(1).rand(60, 80, 3) * 255).astype(np.uint8)).save(
            os.path.join(self.home, "a.jpg")
        )
        self.open_picker()
        disabled = "document.querySelector('dialog.explorer .btn-primary').disabled"
        self.assertTrue(self.page.eval(disabled), "Import is live on the list of places")
        self.page.eval(
            "[...document.querySelectorAll('.explorer-row')].find((b) => b.textContent.trim() === 'Home').click()"
        )
        self.page.wait_for("document.querySelector('.explorer-path').textContent.length > 0")
        self.assertFalse(self.page.eval(disabled), "Import is dead inside a folder")
        self.page.wait_for(
            "[...document.querySelectorAll('.explorer-row')].some((b) => b.textContent.trim() === 'a.jpg')"
        )
        self.page.eval(
            "[...document.querySelectorAll('.explorer-row')].find((b) => b.textContent.trim() === 'a.jpg').click()"
        )
        self.assertFalse(self.page.eval(disabled), "Import is dead with a file ticked")


class KeysStayInsideTheConfirmation(BrowserCase):
    """Bug 6: with "Keep all doubtful visible?" open, B, N and Delete still acted on the Review behind it: B turned the blur of the
    selected item off, Delete removed the selected area."""

    modules = [
        FixedModule(FACE, [(140, 60, 200, 120)], 0.85, 0.75, placed=True),
        FixedModule(PLATE, [(20, 20, 100, 44)]),
    ]  # 2.2: a doubtful face is blurred by default only when placed

    def open_review_of_a_photo(self):
        path = os.path.join(self.dir, "media", "f.png")
        Image.fromarray((np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)).save(path)
        self.clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(self.clip_id)
        self.base = f"/api/clips/{self.clip_id}"
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelectorAll('.item-list li').length >= 2")

    def blur_of(self, kind):
        return {i["kind"]: i["blur"] for i in self.api("GET", f"{self.base}/analysis")["items"]}[kind]

    def open_dialog(self):
        self.page.eval(
            "[...document.querySelectorAll('.review-toolbar button')].find((b) => /doubtful visible/i.test(b.textContent)).click()"
        )
        self.page.wait_for("!!document.querySelector('dialog.confirm-dialog[open]')")

    def test_b_n_and_delete_do_nothing_behind_the_dialog_and_work_again_after_it(self):
        self.open_review_of_a_photo()
        self.page.key("n", "KeyN")  # selects the doubtful face
        self.page.wait_for("!document.querySelector('.item-detail').hidden")
        area = self.api("POST", f"{self.base}/areas", {"frame_from": 0, "frame_to": 0, "box": [5, 100, 60, 150]}, 201)
        area_id = area["item"]["item_id"]
        self.open_dialog()
        self.page.key("b", "KeyB")
        self.page.key("Delete", "Delete")
        self.page.sleep(1.0)
        self.assertTrue(self.blur_of("face"), "B turned the blur off behind the dialog")
        self.assertIn(area_id, [i["item_id"] for i in self.api("GET", f"{self.base}/analysis")["items"]])
        self.page.key("Escape", "Escape")
        self.page.wait_for("!document.querySelector('dialog.confirm-dialog[open]')")
        self.assertTrue(self.blur_of("face"), "Escape decided something")
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelectorAll('.item-list li').length >= 2")
        self.page.key("n", "KeyN")
        self.page.wait_for("!document.querySelector('.item-detail').hidden")
        self.page.key("b", "KeyB")
        self.page.wait_for(
            f"fetch('{self.base}/analysis').then((r) => r.json()).then((a) => a.items.some((i) => i.kind === 'face' && !i.blur))"
        )


class ListsKeepTheNodeUnderThePointer(BrowserCase):
    """Bug 7: the notices and the clip cards of the Analyze screen were built again whole on every progress event (four a second), so
    a click that began on a card or on a notice's close button, and ended a moment later, landed on the list and did nothing."""

    def photo(self, name, seed):
        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(seed).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        return self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]

    def test_a_notice_and_a_finished_clips_card_stay_the_same_nodes_while_another_clip_reports_progress(self):
        done = self.photo("a.png", 1)
        busy = self.photo("b.png", 2)
        self.analyze(done)
        self.reload_page()
        hub = self.server.hub
        hub.emit("job.warning", code="warning.comparison.contains_original", clip_id=done, params={"file": "x.mp4"})
        self.page.wait_for("document.querySelectorAll('.notices .notice').length === 1")
        self.page.eval("document.querySelectorAll('.tabs button')[1].click()")
        self.page.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card').length === 2")
        self.page.eval(
            "window.__keep = [document.querySelector('.notices .notice'), document.querySelector('.page:not([hidden]) .clip-card')]"
        )
        for i in range(1, 6):
            hub.emit("clip.progress", clip_id=busy, phase="analyze", fraction=i / 10, frame=i, frames=100, eta_s=None)
            self.page.sleep(0.25)
        self.page.wait_for("document.querySelectorAll('.page:not([hidden]) .clip-card')[1].textContent.includes('50%')")
        self.assertEqual(self.page.eval("window.__keep.map((node) => node.isConnected)"), [True, True])


class DroppedFolders(BrowserCase):
    """Bug 8: the guide says "drop files or folders on the page", but a dropped folder ended with "The engine did not answer: Failed to
    fetch" (the page tried to upload the folder as if it were a file), and nothing was imported."""

    def drop(self, *paths):
        x, y = self.page.center(".drop-zone")
        data = {"items": [], "files": list(paths), "dragOperationsMask": 1}
        for kind in ("dragEnter", "dragOver", "drop"):
            self.page.call("Input.dispatchDragEvent", type=kind, x=x, y=y, data=data)

    def test_a_dropped_folder_imports_the_photos_in_it_and_below_it(self):
        folder = os.path.join(self.dir, "media", "shoot")
        os.makedirs(os.path.join(folder, "day two"))
        for where, seed in (("a.png", 1), ("day two/b.png", 2), (".hidden.png", 3)):
            Image.fromarray((np.random.RandomState(seed).rand(60, 80, 3) * 255).astype(np.uint8)).save(
                os.path.join(folder, where)
            )
        self.drop(folder)
        self.page.wait_for("document.querySelectorAll('.clip-card').length === 2")
        names = self.page.eval(
            "[...document.querySelectorAll('.clip-card .clip-name')].map((e) => e.textContent).sort()"
        )
        self.assertEqual(names, ["a.png", "b.png"])
        # the hidden file is left out as the server's own folder import leaves it, and since 2.1.6 it is said (the same notice)
        self.assertEqual(
            self.page.eval("document.querySelector('.notices').innerText"),
            "1 hidden entry (a name that starts with a dot) was not imported.",
        )


class ExplorerWhenTheServerIsGone(BrowserCase):
    """Bug 9: with the server stopped, the Import dialog said "The local server hit an internal error", as if it were there and had failed."""

    def test_the_dialog_says_the_server_did_not_answer(self):
        self.page.eval("window.fetch = () => Promise.reject(new TypeError('Failed to fetch'))")
        self.page.eval("document.querySelector('.side-head button').click()")
        self.page.wait_for(
            "!!document.querySelector('dialog.explorer .error') && !document.querySelector('dialog.explorer .error').hidden"
        )
        text = self.page.eval("document.querySelector('dialog.explorer .error').textContent")
        self.assertIn("did not answer", text)
        self.assertNotIn("internal error", text)


class AnalyzeAgainDialog(BrowserCase):
    """Bug 13, in the page: other settings and Analyze on a reviewed clip ask first, and Cancel changes nothing."""

    modules = [doubtful_plate_module()]

    def test_the_dialog_names_the_clips_and_decisions_and_cancel_keeps_them(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        self.api("PATCH", f"/api/clips/{clip_id}/items/0", {"blur": False})
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[0].click()")
        self.page.eval(
            "[...document.querySelectorAll('.segmented button')].find((b) => b.textContent.trim() === 'Fast').click()"
        )
        self.page.sleep(0.5)
        self.click(".page:not([hidden]) .cta .btn-primary")
        self.page.wait_for("!!document.querySelector('dialog.confirm-dialog[open]')")
        text = self.page.eval("document.querySelector('dialog.confirm-dialog').innerText")
        self.assertIn("discard your review", text)
        self.assertIn("1 in all", text)
        self.page.eval(
            "[...document.querySelectorAll('dialog.confirm-dialog button')].find((b) => b.textContent.trim() === 'Cancel').click()"
        )
        self.page.wait_for("!document.querySelector('dialog.confirm-dialog[open]')")
        self.page.sleep(0.5)
        self.assertIsNone(self.api("GET", "/api/state")["job"])
        self.assertFalse(self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"])
        self.click(".page:not([hidden]) .cta .btn-primary")
        self.page.wait_for("!!document.querySelector('dialog.confirm-dialog[open]')")
        reader = server_tests.EventReader(
            self.port
        )  # the end of the new analysis is an event: the state before the job starts also reads "analyzed"
        self.addCleanup(reader.close)
        self.page.eval(
            "[...document.querySelectorAll('dialog.confirm-dialog button')].find((b) => b.textContent.trim() === 'Analyze again').click()"
        )
        reader.wait_for("job.finished")
        self.wait_idle()
        self.assertTrue(self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]["blur"])


class KeyboardReachesAnyItemOfAVideo(BrowserCase):
    """Bug 15: the bars of a video's timeline take only a pointer, and N goes only to items that need a check, so from the keyboard an item
    the engine was sure of could not be selected (to turn off a blur that hides nothing)."""

    def test_comma_and_period_walk_through_every_item_in_time_order(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        items = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        self.assertGreaterEqual(len(items), 1)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelectorAll('.page-review .timeline-track .bar-item').length >= 1")
        self.page.key(".", "Period", ".")
        self.page.wait_for("!document.querySelector('.item-detail').hidden")
        order = sorted(items, key=lambda i: (i["first_frame"], i["item_id"]))
        self.assertIn("#1", self.page.eval("document.querySelector('.item-detail h2').textContent"))
        for number in range(2, len(order) + 1):
            self.page.key(".", "Period", ".")
            self.page.wait_for(f"document.querySelector('.item-detail h2').textContent.includes('#{number}')")
        self.page.key(".", "Period", ".")
        self.page.wait_for("document.querySelector('.item-detail h2').textContent.includes('#1')")
        self.page.key(",", "Comma", ",")
        self.page.wait_for(f"document.querySelector('.item-detail h2').textContent.includes('#{len(order)}')")
        self.assertIn("previous/next item", self.page.eval("document.querySelector('.shortcuts').textContent"))


class LongDestinationOnAPhone(BrowserCase):
    """Bug 16: on a 420 px screen the Export screen's destination path (a long folder) ran past the right edge of the page, cut off, in the
    header of the files list."""

    def test_the_destination_stays_inside_the_window_and_keeps_the_full_path_for_the_pointer(self):
        folder = os.path.join(self.dir, "media", "alongprojectfoldernameXX" * 2, "andasecondonebelowitYYYY" * 2)
        os.makedirs(folder)
        media.moving_plate_clip(os.path.join(folder, "clip.mp4"), audio=False)
        clip_id = self.api("POST", "/api/clips", {"paths": [os.path.join(folder, "clip.mp4")]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.resize(420, 900)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[3].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("document.querySelector('.export-where').textContent.length > 20")
        facts = self.page.eval(
            "(() => { const e = document.querySelector('.export-where'); return {right: e.getBoundingClientRect().right,"
            " width: innerWidth, title: e.title === e.textContent}; })()"
        )
        self.assertLessEqual(facts["right"], facts["width"], facts)
        self.assertTrue(facts["title"])


class PreferencesButton(BrowserCase):
    """Bug 18: the sliders button at the top right (labelled "Preferences" for a screen reader) did nothing at all."""

    def test_it_takes_you_to_the_options(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelectorAll('.tabs button')[2].getAttribute('aria-current') === 'page'")
        self.click(".top-right .btn-icon")
        self.page.wait_for("document.querySelectorAll('.tabs button')[0].getAttribute('aria-current') === 'page'")
        self.assertTrue(self.page.eval("!document.querySelector('.page:not([hidden]) .side-right .options').hidden"))
        self.assertEqual(self.page.eval("document.querySelector('.top-right .btn-icon').title"), "Preferences")


class ReviewOfAChangedOriginal(BrowserCase):
    """Bug 22: when the original file changed after the analysis (edited, synced, replaced) the Review showed a broken picture on black and
    said nothing; the reason came only at Export."""

    def test_the_review_says_the_original_changed(self):
        path = os.path.join(self.dir, "media", "p.png")
        Image.fromarray((np.random.RandomState(4).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        os.utime(path, (time.time() + 100, time.time() + 100))
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelector('.notices').innerText.includes('changed after it was analyzed')")
        self.page.sleep(1.0)
        self.assertEqual(
            self.page.eval("document.querySelectorAll('.notices .notice').length"), 1, "said more than once"
        )


class DoubleClicks(BrowserCase):
    """Bug 23: a double click on Analyze or on Render sent the request twice, and the second was answered with the error "Another job is
    running" (a red notice for something that had worked)."""

    def two_clicks(self, selector):
        self.page.eval(f"(() => {{ const b = document.querySelector({selector!r}); b.click(); b.click(); }})()")

    def test_analyze_and_render_start_once_and_say_nothing(self):
        clip_id = self.import_clip()
        self.reload_page()
        self.two_clicks(".page:not([hidden]) .cta .btn-primary")
        self.page.wait_for(
            "fetch('api/state').then((r) => r.json()).then((s) => s.job === null && s.clips[0].status === 'analyzed')",
            60,
        )
        self.page.sleep(1.0)
        self.assertEqual(self.page.eval("document.querySelector('.notices').innerText"), "")
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("!document.querySelector('.page-export .btn-primary').disabled")
        self.two_clicks(".page-export .btn-primary")
        self.page.wait_for("fetch('api/state').then((r) => r.json()).then((s) => s.job === null)", 60)
        self.page.sleep(1.0)
        self.assertEqual(self.page.eval("document.querySelector('.notices').innerText"), "")
        folder = os.path.join(self.dir, "media", "anonymized")
        outputs = [name for name in os.listdir(folder) if name.endswith(".mp4")]
        self.assertEqual(outputs, ["clip_anonymized.mp4"], clip_id)


class ServerStopsUnderThePage(BrowserCase):
    """Bug 25: the Studio stops by itself (after 10 minutes with no page open, 6 hours with clips: a laptop that slept) and the page said
    nothing; the first thing the person did afterwards ended in "The engine did not answer: Failed to fetch"."""

    def test_the_page_says_the_studio_has_stopped(self):
        self.page.wait_for("document.querySelectorAll('.tabs button').length === 4")
        self.assertEqual(self.page.eval("document.querySelector('.notices').innerText"), "")
        self.stop_server()
        # since 2.1.8 a bar that stays says so (a notice went by itself after 20 s), and the screens are inert
        self.page.wait_for(
            "!document.querySelector('.lost-bar').hidden && document.querySelector('.lost-bar').innerText.includes('The Studio is not running')",
            60,
        )
        self.assertEqual(self.page.eval("document.querySelectorAll('.notices .notice').length"), 0)


class LastDetectorStaysOn(BrowserCase):
    """Bug 26: turning off both faces and plates left the page with nothing to look for, a dead Analyze button and a red "The request was
    not valid (faces)" from the settings it could not save."""

    def toggle(self, label):
        self.page.eval(
            f"[...document.querySelectorAll('.toggle')].find((b) => b.textContent.trim() === {label!r}).click()"
        )
        self.page.sleep(0.6)

    def test_the_second_one_cannot_be_turned_off_and_says_why(self):
        self.page.wait_for("document.querySelectorAll('.toggle').length >= 2")
        self.toggle("Faces")
        self.toggle("License plates")
        self.page.sleep(0.6)
        self.assertEqual(self.page.eval("document.querySelectorAll('.notices .notice').length"), 1)
        text = self.page.eval("document.querySelector('.notices').innerText")
        self.assertIn("Choose at least one", text)
        self.assertNotIn("not valid", text)
        checked = self.page.eval(
            "[...document.querySelectorAll('.toggle')].filter((b) => /Faces|License plates/.test(b.textContent)).map((b) => b.getAttribute('aria-checked'))"
        )
        self.assertEqual(checked, ["false", "true"])
        settings = self.api("GET", "/api/settings")
        self.assertEqual((settings["faces"], settings["plates"]), (False, True))


class SlowImportSaysSo(BrowserCase):
    """Bug 28: importing a folder of large photos takes a quarter of a second each, in one request, and the page showed nothing meanwhile:
    no sign that anything was happening, and nothing to stop a second click."""

    def test_the_import_button_says_importing_until_the_answer_comes(self):
        path = os.path.join(self.dir, "home", "a.png")
        Image.fromarray((np.random.RandomState(8).rand(60, 80, 3) * 255).astype(np.uint8)).save(path)
        self.page.eval(
            "window.__fetch = window.fetch; window.fetch = (url, init) => (init && init.method === 'POST' && String(url).includes('api/clips')"
            " ? new Promise((resolve) => setTimeout(resolve, 1500)).then(() => window.__fetch(url, init)) : window.__fetch(url, init))"
        )
        self.page.eval("document.querySelector('.side-head button').click()")
        self.page.wait_for("document.querySelectorAll('.explorer-list li').length > 0")
        self.page.eval(
            "[...document.querySelectorAll('.explorer-row')].find((b) => b.textContent.trim() === 'Home').click()"
        )
        self.page.wait_for(
            "[...document.querySelectorAll('.explorer-row')].some((b) => b.textContent.trim() === 'a.png')"
        )
        self.page.eval(
            "[...document.querySelectorAll('.explorer-row')].find((b) => b.textContent.trim() === 'a.png').click()"
        )
        self.page.eval("document.querySelector('dialog.explorer .btn-primary').click()")
        self.page.wait_for("document.querySelector('.side-head button').textContent.trim() === 'Importing...'", 5)
        self.assertTrue(self.page.eval("document.querySelector('.side-head button').disabled"))
        self.page.wait_for("document.querySelectorAll('.clip-card').length === 1", 20)
        self.assertEqual(self.page.eval("document.querySelector('.side-head button').textContent.trim()"), "Import")
        self.assertFalse(self.page.eval("document.querySelector('.side-head button').disabled"))


class RemoveAClipFromTheList(BrowserCase):
    """Bug 29: nothing in the page removed a clip from the list (the server could), so an import of the wrong folder had to be analysed
    in full or the Studio restarted."""

    def photo(self, name, seed):
        path = os.path.join(self.dir, "media", name)
        Image.fromarray((np.random.RandomState(seed).rand(60, 80, 3) * 255).astype(np.uint8)).save(path)
        return path

    def test_the_cross_takes_the_clip_off_the_list_and_leaves_the_file(self):
        first, second = self.photo("a.png", 1), self.photo("b.png", 2)
        self.api("POST", "/api/clips", {"paths": [first, second]})
        self.reload_page()
        self.page.wait_for("document.querySelectorAll('.clip-row').length === 2")
        self.assertIn("a.png", self.page.eval("document.querySelector('.clip-remove').title"))
        self.click(".clip-row:first-child .clip-remove")
        self.page.wait_for("document.querySelectorAll('.clip-row').length === 1")
        names = [clip["name"] for clip in self.api("GET", "/api/state")["clips"]]
        self.assertEqual(names, ["b.png"])
        self.assertTrue(os.path.exists(first), "the file itself was touched")
        self.assertEqual(self.page.eval("document.querySelector('.clip-card').getAttribute('aria-pressed')"), "true")


class HiddenMeansHidden(BrowserCase):
    """Bug 30: a rule such as `.field { display: flex }` beat the `hidden` attribute, so "Heavy model only" (meant for Deep) showed under
    Careful, and in Review the buttons for doubtful items showed on a clip that has none."""

    def test_the_studio_offers_fast_and_careful_and_no_heavy_model_option(self):
        self.page.wait_for("document.querySelectorAll('.toggle').length >= 2")
        text = self.page.eval("document.querySelector('.page:not([hidden]) .side-right').innerText")
        self.assertNotIn("Heavy model only", text)
        self.assertNotIn("Deep", text)
        self.assertEqual(
            self.page.eval(
                "[...document.querySelectorAll('.page:not([hidden]) .segmented button')].map((b) => b.textContent.trim())"
            ).count("Careful"),
            1,
        )

    def test_a_clip_with_nothing_doubtful_has_no_bulk_buttons(self):
        path = os.path.join(self.dir, "media", "p.png")
        Image.fromarray((np.random.RandomState(9).rand(60, 80, 3) * 255).astype(np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        items = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"]
        self.assertFalse(
            [i for i in items if i["judgement"]["verdict"] == "check"], "the stand-in found something doubtful"
        )
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("!!document.querySelector('.review-toolbar')")
        self.page.sleep(0.5)
        labels = self.page.eval(
            "[...document.querySelectorAll('.review-toolbar button')].filter((b) => b.getClientRects().length > 0).map((b) => b.textContent.trim())"
        )
        self.assertNotIn("Keep all doubtful visible", labels)
        self.assertNotIn("Blur all doubtful", labels)
        self.assertNotIn("Next item to check", labels)


if __name__ == "__main__":
    unittest.main()
