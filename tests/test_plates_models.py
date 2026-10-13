"""Heavy plate tests: they load the real models and run sweeps on public images. Off unless ANONYMIZER_HEAVY=1
(and, on a small machine, with a memory cap and nothing else heavy running).

    ANONYMIZER_HEAVY=1 python3 -s -m unittest tests.test_plates_models -v
"""

from __future__ import annotations

import os
import unittest

import cv2
import numpy as np

from anonymizer import contract as c
from anonymizer import geometry
from anonymizer.classes import plates
from bench import plates_truth
from bench.bench_plates import decide_photo

HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
EU = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", ""), "plates", "openalpr-eu")
# Photos with several vehicles, small plates or dark scenes, picked from the EU set
PARITY_IMAGES = ["eu1.jpg", "eu7.jpg", "eu10.jpg", "test_001.jpg", "test_020.jpg", "test_045.jpg", "test_070.jpg"]
PHOTO_PLAN = c.SweepPlan(general_side=1600, tile_side=640, vehicle_tiles=True)
LIGHT_PLAN = c.SweepPlan(general_side=960, tile_side=640, tiles=[0, 2, 5], full_image=False, max_vehicles=3)


def _image(name: str, scale: float = 1.0) -> np.ndarray:
    bgr = cv2.imread(os.path.join(EU, name))
    if bgr is None:
        raise unittest.SkipTest(f"{name} is not in {EU}")
    return bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


@unittest.skipUnless(HEAVY, "set ANONYMIZER_HEAVY=1 to run the tests that load the models")
class WithModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = plates.PlateModule(threads=2)

    @classmethod
    def tearDownClass(cls):
        cls.module.close()

    def test_P1_retry_around_an_empty_place_finds_nothing(self):
        bgr = np.full((600, 800, 3), 128, np.uint8)
        self.assertIsNone(self.module.retry(bgr, (300, 300, 380, 330), c.RetryPurpose.VERIFY))

    def test_P2_a_flat_image_gives_no_zone_at_all(self):
        flat = np.full((720, 1280, 3), 200, np.uint8)
        self.assertEqual(decide_photo(self.module, flat), [])

    def test_P2_pure_noise_gives_nothing_to_blur(self):
        noise = np.random.default_rng(5).integers(0, 255, (640, 640, 3), dtype=np.uint8)
        self.assertEqual([z for z in decide_photo(self.module, noise) if z.verdict is c.Verdict.BLUR], [])

    def test_P6_eu7_blurs_the_large_plate_and_the_22_px_far_plate(self):
        bgr = _image("eu7.jpg")
        blur = [z for z in decide_photo(self.module, bgr) if z.verdict is c.Verdict.BLUR]
        large, far = (848, 954, 1191, 1033), (1706, 467, 1729, 480)  # the far one is not in the upstream annotation
        self.assertTrue(any(plates_truth.covered_share(large, z.blur_box) >= 0.95 for z in blur), "large plate")
        self.assertTrue(any(plates_truth.covered_share(far, z.blur_box) >= 0.95 for z in blur), "far plate")

    def test_P3_no_sweep_plate_off_every_vehicle_ends_blurred_on_the_eu_photos(self):
        for name in PARITY_IMAGES:
            for z in decide_photo(self.module, _image(name)):
                if z.verdict is c.Verdict.BLUR:
                    self.assertTrue(z.on_vehicle, (name, z.box))

    def test_P1_a_module_is_a_class_module_of_the_contract(self):
        self.assertIsInstance(self.module, c.ClassModule)

    def test_P1_sweep_tiles_follow_the_geometry_the_engine_shares(self):
        bgr = _image("eu7.jpg")
        before = self.module.calls.get("plate_tiles", 0)
        self.module.sweep(
            bgr,
            c.SweepPlan(
                general_side=960, tile_side=640, tiles=[0, 1], full_image=False, vehicle_tiles=False, max_vehicles=0
            ),
        )
        self.assertEqual(
            self.module.calls["plate_tiles"] - before,
            len(geometry.sweep_tiles(bgr.shape[1], bgr.shape[0], 640, [0, 1])),
        )


if __name__ == "__main__":
    unittest.main()
