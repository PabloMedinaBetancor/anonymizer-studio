"""Review 2, N8: a photo fingerprint is a hash of the whole file, and the thumbnails of a saved analysis are replaced as a set."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

try:
    from PIL import Image

    import anonymizer
    from anonymizer.contract import PLATE
    from anonymizer.engine.files import source_fingerprint
    from tests.fakes import FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class Fingerprint(unittest.TestCase):
    def test_N8_a_change_anywhere_in_a_photo_changes_its_fingerprint_even_with_the_same_size_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.bmp")
            Image.fromarray(np.zeros((400, 600, 3), np.uint8)).save(
                path
            )  # uncompressed: far larger than the sampled places
            before = source_fingerprint(path)
            stat = os.stat(path)
            with open(path, "r+b") as f:  # a byte between two of the places a sampled hash would read
                f.seek(stat.st_size // 20)
                f.write(b"\x07")
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(source_fingerprint(path), before)

    def test_N8_a_video_is_still_sampled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.mp4")
            with open(path, "wb") as f:
                f.write(os.urandom(1 << 20))
            stat = os.stat(path)
            before = source_fingerprint(path)
            with open(path, "r+b") as f:
                f.seek(70000)  # between samples: not looked at, by design
                f.write(b"\x07")
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertEqual(source_fingerprint(path), before)


class Thumbnails(unittest.TestCase):
    def analysis(self, tmp: str):
        path = os.path.join(tmp, "p.png")
        Image.fromarray(np.random.default_rng(2).integers(0, 255, (160, 240, 3), dtype=np.uint8)).save(path)
        with anonymizer.Anonymizer(modules=[FixedModule(PLATE, [(20, 20, 90, 60)])]) as anon:
            return anon.analyze(path)

    def test_N8_a_save_cut_after_the_new_set_is_written_leaves_the_earlier_thumbnails_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis = self.analysis(tmp)
            target = os.path.join(tmp, "a.json")
            analysis.save(target)
            folder = os.path.join(tmp, "a.thumbs")
            first = {n: open(os.path.join(folder, n), "rb").read() for n in os.listdir(folder)}
            self.assertTrue(first)
            os.replace(folder, folder + ".old")  # the state between the two renames of a save that was cut
            os.makedirs(folder + ".new")
            loaded = anonymizer.load_analysis(target)
            item = loaded.items[0].item_id
            self.assertEqual(loaded.thumbnail_jpeg(item), first[f"{item}.jpg"])
            loaded.save(target)  # the next save puts everything right again and leaves no side folders
            self.assertEqual(sorted(os.listdir(tmp)), ["a.json", "a.thumbs", "p.png"])
            self.assertEqual({n: open(os.path.join(folder, n), "rb").read() for n in os.listdir(folder)}, first)


if __name__ == "__main__":
    unittest.main()
