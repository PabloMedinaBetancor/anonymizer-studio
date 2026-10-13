"""Stop and pause, and what the API guarantees that the old folder job also tested: originals only read, no network, a photo too big for
the free memory fails alone, a leftover hidden partial file is replaced and nothing else is touched."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    from PIL import Image

    import anonymizer
    from anonymizer.engine.events import Cancelled, StopFlag
    from tests import media
    from tests.fakes import plate_module
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class StopAndPause(unittest.TestCase):
    def test_E8_a_pause_after_stop_never_blocks(self):
        flag = StopFlag()
        flag.request()
        flag.pause()
        with self.assertRaises(Cancelled):
            flag.check()

    def test_E8_stop_wakes_a_paused_worker(self):
        flag = StopFlag()
        flag.pause()
        threading.Timer(0.1, flag.request).start()
        start = time.monotonic()
        with self.assertRaises(Cancelled):
            flag.check()
        self.assertLess(time.monotonic() - start, 10.0)  # wakes on Stop; 2 s is for a free machine


class ApiGuarantees(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def test_E1_originals_are_only_read_and_an_existing_output_is_never_replaced(self):
        video, photo = self.path("clip.mp4"), self.path("p.png")
        media.moving_plate_clip(video, n=20, audio=True)
        Image.new("RGB", (64, 48), (120, 130, 140)).save(photo)
        before = {p: sha256(p) for p in (video, photo)}
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            for source in (video, photo):
                analysis = anon.analyze(source)
                out = self.path("out" + os.path.splitext(source)[1])
                anon.render(analysis, out)
                taken = sha256(out)
                with self.assertRaises(anonymizer.OutputExists):
                    anon.render(analysis, out)
                self.assertEqual(sha256(out), taken)
        self.assertEqual({p: sha256(p) for p in (video, photo)}, before)

    def test_E15_the_api_opens_no_network_connection(self):
        def refuse(*args, **kwargs):
            raise AssertionError("network used")

        video, photo = self.path("clip.mp4"), self.path("p.png")
        media.moving_plate_clip(video, n=10)
        Image.new("RGB", (64, 48)).save(photo)
        with mock.patch("socket.socket.connect", refuse), mock.patch("socket.create_connection", refuse):
            with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
                anon.render(anon.analyze(video), self.path("o.mp4"))
                anon.render(anon.analyze(photo), self.path("o.png"))

    def test_E9_a_photo_too_big_for_the_free_memory_fails_alone_with_what_it_needs(self):
        photo = self.path("p.jpg")
        Image.new("RGB", (64, 48)).save(photo, "JPEG")
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            with mock.patch("anonymizer.engine.video.available_memory_mb", lambda: 100):
                with self.assertRaises(anonymizer.AnonymizerError) as caught:
                    anon.analyze(photo)
            self.assertEqual(caught.exception.code, "error.photo.not_enough_memory")
            self.assertEqual(set(caught.exception.params), {"needed_mb", "free_mb", "megapixels"})
            with mock.patch("anonymizer.engine.video.available_memory_mb", lambda: 4000):
                analysis = anon.analyze(photo)
            with mock.patch("anonymizer.engine.video.available_memory_mb", lambda: 100):
                with self.assertRaises(anonymizer.AnonymizerError) as caught:
                    anon.render(analysis, self.path("o.jpg"))
            self.assertEqual(caught.exception.code, "error.photo.not_enough_memory")
        self.assertEqual(sorted(os.listdir(self.dir)), ["p.jpg"])

    def test_E9_a_picture_over_the_pixel_limit_is_unsupported_and_not_a_crash(self):
        photo = self.path("huge.png")
        Image.new("1", (64, 48)).save(photo)
        with mock.patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(anonymizer.UnsupportedMedia):
                anonymizer.media_info(photo)
            with (
                self.assertRaises(anonymizer.AnonymizerError) as caught,
                anonymizer.Anonymizer(modules=[plate_module()]) as anon,
            ):
                anon.analyze(photo)
        self.assertIn(caught.exception.code, ("error.media.unsupported",))

    def test_E2_a_leftover_partial_file_is_replaced_by_its_own_render_and_no_other_hidden_file_is_touched(self):
        video = self.path("clip.mp4")
        media.moving_plate_clip(video, n=20, audio=False)
        mine, other = self.path(".out.mp4.anonymizer-partial"), self.path(".other.mp4.anonymizer-partial")
        for leftover in (mine, other):
            with open(leftover, "wb") as f:
                f.write(b"half a file")
        with anonymizer.Anonymizer(modules=[plate_module()]) as anon:
            anon.render(anon.analyze(video), self.path("out.mp4"))
        self.assertFalse(os.path.exists(mine))
        self.assertEqual(open(other, "rb").read(), b"half a file")
        self.assertTrue(os.path.exists(self.path("out.mp4")))


if __name__ == "__main__":
    unittest.main()
