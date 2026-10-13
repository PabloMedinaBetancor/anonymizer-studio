"""Studio v2.1, what a client meets (QA report of 04/10/2026): each test drives the real page in a headless Chrome."""

from __future__ import annotations

import os
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests import media
    from tests.ui.browser_case import BrowserCase
except ImportError as e:
    raise unittest.SkipTest(f"browser tests need the engine libraries: {e}") from e


def write_photo(path, size=(160, 120), seed=1):
    pixels = (np.random.RandomState(seed).rand(size[1], size[0], 3) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


class ImportPicker(BrowserCase):
    def home_files(self):
        write_photo(os.path.join(self.home, "a.jpg"))
        media.moving_plate_clip(os.path.join(self.home, "b.mp4"), n=10, audio=False)
        Image.new("RGB", (8, 8), "red").save(os.path.join(self.home, "c.gif"))

    def open_picker(self):
        self.page.eval("document.querySelector('.side-head button').click()")
        self.page.wait_for("document.querySelectorAll('.explorer-list li').length > 0")

    def enter_home(self):
        self.page.eval(
            "[...document.querySelectorAll('.explorer-row')].find((b) => b.textContent.trim() === 'Home').click()"
        )
        self.page.wait_for("document.querySelector('.explorer-path').textContent.length > 0")
        self.page.wait_for("document.querySelectorAll('.explorer-list li').length >= 3")

    def test_the_picker_never_shows_the_word_null(self):
        self.home_files()
        self.open_picker()
        self.assertNotIn("null", self.page.eval("document.querySelector('dialog').innerText"))
        self.enter_home()
        self.assertNotIn("null", self.page.eval("document.querySelector('dialog').innerText"))

    def test_each_kind_of_file_has_its_own_icon_and_a_gif_says_why_it_is_greyed(self):
        self.home_files()
        self.open_picker()
        self.enter_home()
        rows = self.page.eval(
            "Object.fromEntries([...document.querySelectorAll('.explorer-row')].map((b) => [b.textContent.trim(),"
            " {icon: b.querySelector('svg').innerHTML, disabled: b.disabled, title: b.title}]))"
        )
        icons = {name: rows[name]["icon"] for name in ("a.jpg", "b.mp4", "c.gif")}
        self.assertEqual(len(set(icons.values())), 3, "photo, video and unsupported file share an icon")
        self.assertTrue(rows["c.gif"]["disabled"])
        self.assertIn("c.gif", rows["c.gif"]["title"])
        self.assertIn("animation", rows["c.gif"]["title"].lower())


class EmptyImport(BrowserCase):
    def visible(self, selector):
        return self.page.eval(
            f"(() => {{ const el = document.querySelector({selector!r}); return !!el && el.getClientRects().length > 0; }})()"
        )

    def test_the_empty_import_screen_says_what_to_do_and_shows_no_stand_in_video(self):
        self.page.wait_for(
            "!!document.querySelector('.import-empty') && document.querySelector('.import-empty').getClientRects().length > 0"
        )
        self.assertTrue(self.visible(".import-empty"))
        self.assertTrue(self.visible(".import-empty .btn-primary"))
        for dead in (".page:not([hidden]) .stage", ".page:not([hidden]) .controls", ".page:not([hidden]) .center-head"):
            self.assertFalse(self.visible(dead), f"{dead} is shown with nothing imported")
        self.assertNotIn("Source", self.page.eval("document.querySelector('.page:not([hidden]) .center').innerText"))

    def test_its_button_opens_the_picker_and_a_clip_brings_the_viewer_back(self):
        write_photo(os.path.join(self.home, "a.jpg"))
        self.click(".import-empty .btn-primary")
        self.page.wait_for("!!document.querySelector('dialog[open]')")
        self.page.eval("document.querySelector('dialog .btn').click()")
        self.import_paths(os.path.join(self.home, "a.jpg"))
        self.page.wait_for("document.querySelectorAll('.clip-card').length === 1")
        self.assertFalse(self.visible(".import-empty"))
        self.assertTrue(self.visible(".page:not([hidden]) .stage"))


class ReviewTimelineEnd(BrowserCase):
    def test_the_last_ruler_label_stays_inside_the_page_so_the_review_has_no_scrollbars(self):
        """QA report, polish: a vertical page scrollbar at 1280 px in Review. The ruler's last label (its tick sits at the very end
        of the track) stuck out of the page by a few pixels: a sideways scrollbar, which took height and made the vertical one."""
        clip = self.make_clip("short.mp4")
        clip_id = self.api("POST", "/api/clips", {"paths": [clip]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        self.resize(1280, 800)
        self.reload_page()
        self.page.eval("document.querySelectorAll('.tabs button')[2].click()")
        self.page.wait_for("document.querySelectorAll('.page-review .timeline-ruler .tick').length >= 2")
        self.page.sleep(0.5)
        facts = self.page.eval(
            "({page_sideways: document.documentElement.scrollWidth > innerWidth,"
            " page_down: document.documentElement.scrollHeight > innerHeight,"
            " timeline_sideways: document.querySelector('.page-review .timeline').scrollWidth >"
            "   document.querySelector('.page-review .timeline').clientWidth})"
        )
        self.assertEqual(facts, {"page_sideways": False, "page_down": False, "timeline_sideways": False})


if __name__ == "__main__":
    unittest.main()
