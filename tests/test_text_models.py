"""The real PP-OCRv6 text detector. Off unless ANONYMIZER_HEAVY=1 (the model is not in the repo): it runs the model on a
picture drawn here, checks its boxes, and where RapidOCR is installed (ANONYMIZER_RAPIDOCR=<its lib folder>) compares them with the
ones RapidOCR gives for the same picture with the same settings."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

try:
    import cv2
    import numpy as np

    from anonymizer import contract as c
    from anonymizer.classes import text as text_det
    from anonymizer.classes.plates import TextPlateModule
    from tests.fakes import FixedVehicles
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def drawn() -> np.ndarray:
    img = np.full((480, 640, 3), 90, np.uint8)
    cv2.rectangle(img, (200, 300), (420, 360), (240, 240, 240), -1)
    cv2.putText(img, "AB 1234 CD", (210, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    cv2.putText(img, "CALL 555 0100", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (250, 250, 250), 3)
    return img


def overlap(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@unittest.skipUnless(HEAVY, "set ANONYMIZER_HEAVY=1 to run the tests that load the models")
class RealTextModel(unittest.TestCase):
    def setUp(self):
        source = os.path.join(c.models_dir(), text_det.TEXT_MODEL)
        if not os.path.isfile(source):
            self.skipTest(f"{text_det.TEXT_MODEL} is not in {c.models_dir()}")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        shutil.copy(source, tmp.name)  # the hashes of THIS branch's models/SHA256SUMS decide
        shutil.copy(os.path.join(REPO, "models", "SHA256SUMS"), tmp.name)
        self.folder = tmp.name
        self.detector = text_det.TextDetector(models=self.folder)
        self.addCleanup(self.detector.close)

    def test_the_plate_and_the_other_writing_are_found_as_two_regions(self):
        found = self.detector.detect(drawn(), 0.3)
        boxes = sorted(d.box() for d in found)
        self.assertEqual(len(boxes), 2, boxes)
        self.assertGreater(overlap(boxes[0], (60, 87, 320, 127)), 0.6)
        self.assertGreater(overlap(boxes[1], (205, 305, 450, 352)), 0.6)
        self.assertTrue(all(d.kind == c.TEXT and 0.3 <= d.score <= 1.0 for d in found))
        self.assertEqual(self.detector.run_failures, 0)

    def test_nothing_is_found_on_a_flat_picture(self):
        self.assertEqual(self.detector.detect(np.full((300, 400, 3), 120, np.uint8), 0.3), [])

    def test_the_boxes_agree_with_rapidocrs_for_the_same_settings(self):
        lib = os.environ.get("ANONYMIZER_RAPIDOCR")
        if not lib or not os.path.isdir(lib):
            self.skipTest("set ANONYMIZER_RAPIDOCR=<RapidOCR lib folder> to compare with it")
        sys.path.append(lib)
        import logging

        logging.disable(logging.INFO)
        from rapidocr import RapidOCR

        t = self.detector.thresholds
        ocr = RapidOCR(params={"Det.thresh": t.thresh, "Det.box_thresh": 0.3, "Det.unclip_ratio": t.unclip_ratio})
        result = ocr(drawn(), use_det=True, use_cls=False, use_rec=False)
        reference = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in np.array(result.boxes)]
        mine = [d.box() for d in self.detector.detect(drawn(), 0.3)]
        self.assertEqual(len(mine), len(reference))
        for box in mine:
            self.assertGreater(max(overlap(box, r) for r in reference), 0.9, box)

    def test_the_text_path_on_a_vehicle_crop_with_the_real_model(self):
        bgr = drawn()
        module = TextPlateModule(vehicles=FixedVehicles([(20, 20, 620, 460)]), text=self.detector)
        found = module.sweep(bgr, c.SweepPlan(1600, 640, vehicle_tiles=True))
        self.assertEqual(len(found), 2, [d.box() for d in found])
        self.assertTrue(all(d.on_vehicle for d in found))
        self.assertEqual(module.model_failures, 0)


if __name__ == "__main__":
    unittest.main()
