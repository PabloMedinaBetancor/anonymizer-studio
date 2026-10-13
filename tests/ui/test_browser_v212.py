"""Studio v2.1.2: what the third client QA pass found in the page, driven in a real Chrome. Each fails on v2.1.1 and passes here."""

from __future__ import annotations

import os
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests.fakes import FACE, FixedModule
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class NoticeWaitsForTheButton(BrowserCase):
    """Bug 2: a notice that came while the button was down pushed the page down by its own height, so the button came up on something
    else and the click went nowhere (and a box being drawn slipped under the pointer)."""

    def top_of(self, selector):
        return self.page.eval(f"document.querySelector({selector!r}).getBoundingClientRect().top")

    def test_a_notice_comes_after_the_button_is_released_and_the_page_does_not_move_meanwhile(self):
        selector = ".page:not([hidden]) .btn-small"
        before = self.top_of(selector)
        x, y = self.page.center(selector)
        self.page.mouse("mouseMoved", x, y)
        self.page.mouse("mousePressed", x, y)
        self.server.hub.emit(
            "job.warning", code="warning.comparison.contains_original", clip_id=None, params={"file": "x.mp4"}
        )
        self.page.sleep(0.5)
        self.assertEqual(self.page.eval("document.querySelectorAll('.notices .notice').length"), 0)
        self.assertEqual(self.top_of(selector), before)
        self.page.mouse("mouseReleased", x, y)
        self.page.wait_for("document.querySelectorAll('.notices .notice').length === 1", 30)
        self.assertGreater(self.top_of(selector), before)

    def test_pressing_a_button_while_another_control_has_the_focus_still_holds_the_notice(self):
        self.page.eval(
            "document.querySelector('.page:not([hidden]) .btn-small').focus()"
        )  # a control has the focus: pressing the other one blurs it
        x, y = self.page.center(".page:not([hidden]) .tabs button, .tabs button")
        self.page.mouse("mouseMoved", x, y)
        self.page.mouse("mousePressed", x, y)
        self.server.hub.emit(
            "job.warning", code="warning.comparison.contains_original", clip_id=None, params={"file": "x.mp4"}
        )
        self.page.sleep(0.5)
        self.assertEqual(self.page.eval("document.querySelectorAll('.notices .notice').length"), 0)
        self.page.mouse("mouseReleased", x, y)
        self.page.wait_for("document.querySelectorAll('.notices .notice').length === 1", 30)

    def test_a_notice_with_no_button_held_comes_at_once(self):
        self.server.hub.emit(
            "job.warning", code="warning.comparison.contains_original", clip_id=None, params={"file": "x.mp4"}
        )
        self.page.wait_for("document.querySelectorAll('.notices .notice').length === 1", 30)


class DeleteKeyAsksBeforeRemovingAnArea(BrowserCase):
    """Bug 3: Delete or Backspace removed the selected area at once, with no question and no undo (the key is not even in the guide)."""

    modules = [FixedModule(FACE, [(140, 60, 200, 120)])]

    def test_the_key_asks_and_only_the_confirmation_removes_it(self):
        path = os.path.join(self.dir, "media", "f.png")
        Image.fromarray((np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        base = f"/api/clips/{clip_id}"
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[2].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelectorAll('.item-list li').length >= 1")
        area = self.api("POST", f"{base}/areas", {"frame_from": 0, "frame_to": 0, "box": [5, 100, 60, 150]}, 201)[
            "item"
        ]["item_id"]
        # the area reaches this page by the event of the server (a second window sees what the first draws)
        self.page.wait_for("document.querySelectorAll('.item-list li').length >= 2")
        self.page.eval("[...document.querySelectorAll('.item-list li button, .item-list li')].pop().click()")
        self.page.wait_for("!document.querySelector('.item-detail').hidden")

        def items():
            return [i["item_id"] for i in self.api("GET", f"{base}/analysis")["items"]]

        self.page.key("Delete", "Delete")
        self.page.wait_for("!!document.querySelector('dialog.confirm-dialog[open]')")
        self.assertIn(area, items())
        self.page.key("Escape", "Escape")
        self.page.wait_for("!document.querySelector('dialog.confirm-dialog[open]')")
        self.page.sleep(0.5)
        self.assertIn(area, items(), "the key removed it before asking")
        self.page.key("Backspace", "Backspace")
        self.page.wait_for("!!document.querySelector('dialog.confirm-dialog[open]')")
        self.page.eval("document.querySelector('dialog.confirm-dialog .btn-primary').click()")
        self.page.wait_for(
            f"fetch('{base}/analysis').then((r) => r.json()).then((a) => !a.items.some((i) => i.item_id === {area}))"
        )


class EveryControlCanBeHit(BrowserCase):
    """Bug 4: the toggles were 20 px high, the checkboxes of the lists 20 x 20 and the sliders 16 px: below the 24 px that a pointer needs
    to hit them reliably (and far below a finger)."""

    SMALL = """(() => { const out = [];
      for (const e of document.querySelectorAll('button, a[href], input, select, textarea, [role=switch]')) {
        const r = e.getBoundingClientRect();
        if (r.width === 0 || r.height === 0 || getComputedStyle(e).visibility === 'hidden' || e.closest('[hidden]')) continue;
        if (r.width < 24 || r.height < 24) out.push(e.tagName + '.' + e.className + ' ' + Math.round(r.width) + 'x' + Math.round(r.height));
      } return out; })()"""

    def test_no_control_of_any_screen_is_smaller_than_24_px(self):
        self.sweep(24)

    def test_on_a_touch_screen_nothing_is_smaller_than_36_px(self):
        self.page.call("Emulation.setTouchEmulationEnabled", enabled=True, maxTouchPoints=5)
        self.page.call("Emulation.setEmulatedMedia", features=[{"name": "pointer", "value": "coarse"}])
        self.addCleanup(self.page.call, "Emulation.setTouchEmulationEnabled", enabled=False)
        self.addCleanup(self.page.call, "Emulation.setEmulatedMedia", features=[{"name": "pointer", "value": "fine"}])
        self.assertTrue(self.page.eval("matchMedia('(pointer: coarse)').matches"))
        self.sweep(36)

    def sweep(self, limit):
        path = os.path.join(self.dir, "media", "f.png")
        Image.fromarray((np.random.RandomState(3).rand(160, 240, 3) * 255).astype(np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[3].disabled")
        for index, name in enumerate(["import", "analyze", "review", "export"]):
            self.page.eval(f"document.querySelectorAll('.tabs button')[{index}].click()")
            self.page.sleep(0.4)
            self.assertEqual(self.page.eval(self.SMALL.replace("24", str(limit))), [], f"on {name}")


class KeyboardFocusStaysPut(BrowserCase):
    """Bug 6: the lists of the screens are built again whenever the state changes, and a node that is replaced takes the keyboard focus with it:
    after Space on a clip's checkbox of the Export screen (or on a clip card of Import or Review) the focus was on nothing, the next Tab went back to
    the same checkbox and a second Space acted on nothing; pressing Analyze left the focus on a hidden button."""

    def setUp(self):
        super().setUp()
        for name, seed in (("a.png", 1), ("b.png", 2)):
            path = os.path.join(self.dir, "media", name)
            Image.fromarray((np.random.RandomState(seed).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        ids = [
            self.api("POST", "/api/clips", {"paths": [os.path.join(self.dir, "media", n)]})["clips"][0]["clip_id"]
            for n in ("a.png", "b.png")
        ]
        self.analyze(*ids)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[3].disabled")

    def active(self):
        return self.page.eval(
            "(() => { const a = document.activeElement; return a === document.body ? 'BODY' : a.className.split(' ')[0] + ':' + (a.getAttribute('aria-label') || a.textContent.trim()); })()"
        )

    def space_on(self, tab, selector, index):
        self.page.eval(f"document.querySelectorAll('.tabs button')[{tab}].click()")
        self.page.sleep(0.4)
        self.page.eval(f"document.querySelectorAll({selector!r})[{index}].focus()")
        self.page.key(" ", "Space")
        self.page.sleep(0.5)

    def test_space_on_a_checkbox_or_a_clip_card_leaves_the_focus_on_it(self):
        for tab, selector in (
            (3, ".page:not([hidden]) .check"),
            (0, ".page:not([hidden]) .clip-card"),
            (2, ".page:not([hidden]) .clip-card-review"),
        ):
            self.space_on(tab, selector, 1)
            self.assertNotEqual(self.active(), "BODY", selector)
            self.assertIn(self.active().split(":")[0], ("check", "clip-card"), self.active())
        self.space_on(3, ".page:not([hidden]) .check", 0)
        self.page.key("Tab", "Tab")
        self.assertEqual(self.active(), "check:b.png", "Tab after Space did not go on to the next checkbox")

    def test_pressing_analyze_with_the_keyboard_leaves_the_focus_on_the_new_screen(self):
        self.page.eval("document.querySelectorAll('.tabs button')[0].click()")
        self.page.sleep(
            1.5
        )  # the stream's first answer reloads the state: a choice made before it would be overwritten
        # the clips are analysed with these settings, and Analyze (v2.1.5) does only what is not done: another setting gives it work
        self.page.eval(
            "[...document.querySelectorAll('.segmented button')].find((b) => /^Fast/.test(b.textContent.trim())).click()"
        )
        self.page.wait_for("!document.querySelector('.page:not([hidden]) .cta .btn-primary').disabled", 30)
        self.page.eval("document.querySelector('.page:not([hidden]) .cta .btn-primary').focus()")
        self.page.key(" ", "Space")
        self.page.wait_for("document.querySelectorAll('.tabs button')[1].getAttribute('aria-current') === 'page'", 30)
        self.page.sleep(0.5)
        self.assertNotEqual(self.active(), "BODY")


class ClipRowsFillTheNarrowList(BrowserCase):
    """Bug 7: below 1100 px the list of clips lets its cards wrap, but since the cross to take a clip off the list (2.1.1) the card sits in a row
    and the rule that made the cards share the width was left on the card: on a tablet or a phone each row was as wide as its own text, the rows
    ended at different places, and the cross lay over the end of the name."""

    def test_every_row_takes_the_width_of_the_list_and_the_cross_stays_clear_of_the_name(self):
        for name, seed in (("a-long-name-for-a-photo.png", 1), ("b.png", 2)):
            path = os.path.join(self.dir, "media", name)
            Image.fromarray((np.random.RandomState(seed).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
            self.api("POST", "/api/clips", {"paths": [path]})
        self.open_page(420, 900)
        self.page.wait_for("document.querySelectorAll('.clip-row').length === 2")
        rows = self.page.eval(
            "(() => { const list = document.querySelector('.clip-list').getBoundingClientRect();"
            " return [...document.querySelectorAll('.clip-row')].map((row) => { const r = row.getBoundingClientRect();"
            " const x = row.querySelector('.clip-remove').getBoundingClientRect(); const n = row.querySelector('.clip-name').getBoundingClientRect();"
            " return [Math.round(r.left - list.left), Math.round(list.right - r.right), Math.round(x.left - n.right)]; }); })()"
        )
        for left, right_gap, clearance in rows:
            self.assertEqual((left, right_gap), (0, 0), rows)
            self.assertGreaterEqual(clearance, 0, rows)


class TwoWindowsDoNotUndoEachOther(BrowserCase):
    """Bug 8: every change of an option sent all the options of the window, so a second window that had loaded earlier put back in the server
    what the first had changed since (a blur style, a strength, a margin) while the first window still showed its own choice: the export was
    blurred with a setting nobody had chosen. Now a window sends only what it changed and shows what the others change."""

    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def second_window(self):
        from tests.ui.browser_case import free_port
        from tests.ui.drive import Page

        url = f"http://127.0.0.1:{self.port}/?t={server_tests_token()}&lang=en"
        other = Page(url, width=1280, height=800, port=free_port())
        self.addCleanup(other.close)
        other.wait_for("document.querySelectorAll('.tabs button').length === 4", 60)
        return other

    def choose(self, page, selector, value):
        page.eval(
            f"(() => {{ const e = document.querySelector({selector!r}); e.value = {value!r}; e.dispatchEvent(new Event('input', {{bubbles: true}})); e.dispatchEvent(new Event('change', {{bubbles: true}})); }})()"
        )

    def test_a_change_in_one_window_survives_an_unrelated_change_in_the_other_and_shows_there(self):
        other = self.second_window()
        self.choose(self.page, "#blur-style", "pixelate")
        self.page.sleep(1.0)
        self.assertEqual(self.api("GET", "/api/settings")["blur"], "pixelate")
        other.eval(
            "document.querySelectorAll('.page:not([hidden]) .toggle')[0].click()"
        )  # an unrelated option, in the window that is behind
        other.sleep(1.0)
        settings = self.api("GET", "/api/settings")
        self.assertEqual(
            (settings["blur"], settings["faces"]), ("pixelate", False), "the second window put the old blur style back"
        )
        other.wait_for(
            "document.querySelector('#blur-style').value === 'pixelate'", 5
        )  # and it shows the first window's choice
        self.page.wait_for(
            "document.querySelectorAll('.page:not([hidden]) .toggle')[0].getAttribute('aria-checked') === 'false'", 5
        )

    def test_a_clip_imported_in_one_window_and_the_decisions_on_it_show_in_the_other_without_a_reload(self):
        other = self.second_window()
        path = os.path.join(self.dir, "media", "shared.png")
        Image.fromarray((np.random.RandomState(9).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0][
            "clip_id"
        ]  # what the first window's Import does
        other.wait_for("document.querySelectorAll('.clip-row').length === 1", 30)
        self.analyze(clip_id)
        other.wait_for(
            "!document.querySelectorAll('.tabs button')[2].disabled", 10
        )  # Review opens in the window that did not start the job
        item = self.api("GET", f"/api/clips/{clip_id}/analysis")["items"][0]
        self.api("PATCH", f"/api/clips/{clip_id}/items/{item['item_id']}", {"blur": not item["blur"]})
        other.eval("document.querySelectorAll('.tabs button')[2].click()")
        other.wait_for("document.querySelectorAll('.item-list li, .timeline-track .bar-item').length >= 1", 30)
        self.api("DELETE", f"/api/clips/{clip_id}")
        other.wait_for(
            "document.querySelectorAll('.tabs button')[2].disabled && document.querySelector('.page-review .center').classList.contains('is-idle')",
            10,
        )
        other.eval("document.querySelectorAll('.tabs button')[0].click()")
        other.wait_for("document.querySelectorAll('.clip-row').length === 0", 30)


def server_tests_token():
    from tests import test_ui_server

    return test_ui_server.TOKEN


class IndexExampleFollowsTheClipsThatWillBeWritten(BrowserCase):
    """Bug 16: with a name pattern that uses {index} and the first clip unticked, the example under the file name counted the unticked clip
    ("p_2.png") while the server, and the card of the clip, name the first clip written "p_1.png"."""

    def test_the_example_is_the_name_the_card_gives(self):
        ids = []
        for name, seed in (("a.png", 1), ("b.png", 2)):
            path = os.path.join(self.dir, "media", name)
            Image.fromarray((np.random.RandomState(seed).rand(120, 160, 3) * 255).astype(np.uint8)).save(path)
            ids.append(self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"])
        self.analyze(*ids)
        self.reload_page()
        self.page.wait_for("!document.querySelectorAll('.tabs button')[3].disabled")
        self.page.eval("document.querySelectorAll('.tabs button')[3].click()")
        self.page.wait_for("document.querySelectorAll('.export-row .check').length === 2")
        self.page.eval(
            "(() => { const e = document.querySelector('#export-pattern'); e.value = 'p_{index}'; e.dispatchEvent(new Event('input', {bubbles: true})); })()"
        )
        self.page.eval("document.querySelectorAll('.export-row .check')[0].click()")  # the first clip is not written
        self.page.wait_for("document.querySelector('.export-card:nth-child(2)').textContent.includes('p_1.png')", 30)
        self.assertIn("p_1.png", self.page.eval("document.querySelector('.page-export .hint.mono').textContent"))
