"""Studio v2.1.10: what the eleventh client QA pass (06/10/2026, qa-anon-11) found in the page, in a real Chrome. Each test fails on v2.1.9."""

from __future__ import annotations

import os
import unittest

try:
    from tests.fakes import FACE, PLATE, FixedModule
    from tests.ui.browser_case import BrowserCase
    from tests.ui.test_browser_v217 import save_photo
    from tests.ui.test_browser_v218 import open_tab
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries and websocket-client: {e}") from e


class ReviewPictureInAShortWindow(BrowserCase):
    """Entry 3 (2.1.9's "Found and not fixed"): at 1366x600 (a 1366x768 laptop's browser) the photo of a Review was 254x190 px: the toolbar
    wraps to two rows and the list of items kept its 200 px, so the picture had what was left."""

    boxes = [(10 + 30 * (i % 16), 10 + 40 * (i // 16), 30 + 30 * (i % 16), 40 + 40 * (i // 16)) for i in range(32)]
    modules = [FixedModule(FACE, boxes)]
    width, height = 1366, 600

    def test_the_picture_has_the_room_and_the_list_is_still_there(self):
        path = save_photo(os.path.join(self.dir, "media", "street.jpg"), 640, 480)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.reload_page()
        open_tab(self.page, 2)
        self.page.wait_for("document.querySelectorAll('.item-list .item-row').length >= 8", 60)
        self.page.sleep(1)
        box = self.page.eval(
            "(() => { const s = document.querySelector('.page-review .stage').getBoundingClientRect();"
            " const l = document.querySelector('.item-list'); return [s.width, s.height, l.clientHeight, l.scrollHeight, innerHeight]; })()"
        )
        width, height, list_height, list_scroll, window_height = box
        self.assertEqual(window_height, 600)
        self.assertGreaterEqual(
            height, 280, f"the picture is {width:.0f}x{height:.0f} px in a window {window_height} px high"
        )
        self.assertGreaterEqual(list_height, 58, "the list shows at least one whole row")
        self.assertGreater(list_scroll, list_height, "the rest of the items scroll")


class ItemNumbersInTheList(BrowserCase):
    """Entry 2: in the list of a photo's items the name was cut by an ellipsis before its number ("License plate #...") for every item from 10 on,
    because the certainty beside it ("54% . please check") never wraps: the rows read the same and the number is the only difference."""

    boxes = [(10 + 30 * (i % 16), 10 + 40 * (i // 16), 30 + 30 * (i % 16), 40 + 40 * (i // 16)) for i in range(24)]
    modules = [FixedModule(PLATE, boxes, score=0.55, verify_score=0.5)]
    width, height = 1366, 768

    def test_every_name_is_whole(self):
        path = save_photo(os.path.join(self.dir, "media", "street.jpg"), 640, 480)
        clip_id = self.api("POST", "/api/clips", {"paths": [path]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.reload_page()
        open_tab(self.page, 2)
        self.page.wait_for("document.querySelectorAll('.item-list .item-row').length >= 20", 60)
        self.page.sleep(1)
        cut = self.page.eval(
            "[...document.querySelectorAll('.item-list .item-row-name')].filter((n) => n.scrollWidth > n.clientWidth + 1"
            " || n.scrollHeight > n.clientHeight + 1).map((n) => n.innerText)"
        )
        certainty = self.page.eval("document.querySelector('.item-row-cert').innerText")
        self.assertEqual(cut, [], f"names cut off beside {certainty!r}")


class PageTheStudioRefusesRaisesNothing(BrowserCase):
    """Entry 4 (found in Firefox's own log): the page that the Studio refuses (an address without its session) threw "can't access lexical declaration
    'sentSettings' before initialization" at start: the notice that the Studio tells at once reached the settings code before it was set up."""

    def test_no_error_is_thrown_while_the_page_says_it_is_not_connected(self):
        self.page.call("Network.enable")
        self.page.call("Network.clearBrowserCookies")
        self.page.call("Page.navigate", url=f"http://127.0.0.1:{self.port}/")
        self.page.wait_for(
            "document.querySelectorAll('.tabs button').length === 4 && !document.querySelector('.lost-bar').hidden", 60
        )
        self.page.sleep(1)
        self.assertEqual(self.js_errors(), [])


if __name__ == "__main__":
    unittest.main()
