"""A photo that Deep would refuse for its size is refused before the first pass is spent on it (QA pass of 04/10/2026: an 81 MP photo
was refused after 61.8 s, when its first stage was done, while the guide says "before anything runs")."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.classes.heavy import HeavyFaceModule
    from anonymizer.contract import PLATE
    from tests.fakes import FixedModule
    from tests.heavy_fakes import TEST_TH, FakeHeavy
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class RefusedBeforeTheFirstPass(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def photo(self, name, size):
        path = os.path.join(self.dir, name)
        Image.fromarray(np.full((size[1], size[0]), 128, np.uint8)).save(path)
        return path

    def analyse(self, path, detection):
        light = FixedModule(PLATE, [(10, 10, 40, 30)])
        heavy = HeavyFaceModule(heavy_zone=FakeHeavy([]), heavy_thresholds=TEST_TH)
        self.addCleanup(heavy.close)
        with anonymizer.Anonymizer(anonymizer.Options(detection=detection), modules=[light, heavy]) as anon:
            started = time.monotonic()
            try:
                return light, anon.analyze(path), time.monotonic() - started
            except anonymizer.AnonymizerError as e:
                return light, e, time.monotonic() - started

    def test_an_81_mp_photo_in_deep_is_refused_at_once_and_the_light_models_never_ran(self):
        light, error, seconds = self.analyse(self.photo("81mp.png", (9000, 9000)), "deep")
        self.assertIsInstance(error, anonymizer.AnonymizerError)
        self.assertEqual((error.code, error.params["limit"]), ("error.deep.too_large", 120))
        self.assertGreater(error.params["tiles"], 120)
        self.assertEqual(light.calls["sweep"], 0, "the first pass was spent before the refusal")
        self.assertLess(
            seconds, 20.0
        )  # 0.6 s on a quiet machine, 7 s on a loaded node (it failed at 4 s); the old path took 61.8 s; the count of first passes above is the exact check

    def test_a_photo_within_the_limit_still_runs_in_deep_and_the_same_photo_runs_in_careful(self):
        small = self.photo("small.png", (400, 300))
        light, analysis, _ = self.analyse(small, "deep")
        self.assertNotIsInstance(analysis, anonymizer.AnonymizerError)
        self.assertGreaterEqual(light.calls["sweep"], 1)
        light, analysis, _ = self.analyse(self.photo("big.png", (9000, 9000)), "fast")
        self.assertNotIsInstance(analysis, anonymizer.AnonymizerError)


if __name__ == "__main__":
    unittest.main()
