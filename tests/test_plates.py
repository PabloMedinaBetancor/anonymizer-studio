"""Plate rules and geometry filter: own behaviour. No models
needed: the model output is faked."""

from __future__ import annotations

import itertools
import logging
import types
import unittest
from typing import Optional

import numpy as np

from anonymizer import contract as c
from anonymizer.classes import plates

SCORES = [
    0.0,
    0.05,
    0.10,
    0.14,
    0.15,
    0.24,
    0.25,
    0.29,
    0.30,
    0.31,
    0.49,
    0.50,
    0.51,
    0.59,
    0.60,
    0.61,
    0.79,
    0.80,
    0.81,
    0.84,
    0.85,
    0.86,
    0.99,
    1.0,
]


def _stats(
    n_views: int, n_sweep: int, max_score: float, verify: float, n_verify: int, verify_side: float, vehicle: bool
) -> c.TrackStats:
    return c.TrackStats(c.PLATE, n_views, n_sweep, max_score, verify, n_verify, verify_side, vehicle)


class VideoRules(unittest.TestCase):
    """P3 and the reasons of the video rules."""

    @classmethod
    def setUpClass(cls):
        cls.module = plates.PlateModule.__new__(plates.PlateModule)
        cls.module.thresholds = plates.DEFAULT_THRESHOLDS

    def _grid(self):
        for n_views in (1, 2, 3, 4, 7, 8, 9, 20):
            for n_sweep in sorted({0, 1, 2, 3, n_views} & set(range(n_views + 1))):
                for max_score, verify in itertools.product(SCORES, SCORES):
                    if max_score == 0.0 and n_views:
                        continue
                    for on_vehicle in (False, True):
                        yield _stats(n_views, n_sweep, max_score, verify, 1, 30.0, on_vehicle)

    def test_P3_video_plate_off_every_vehicle_is_never_blurred(self):
        for s in self._grid():
            if not s.on_vehicle:
                self.assertNotEqual(self.module.video_judgement(s).verdict, c.Verdict.BLUR, s)

    def test_P1_every_video_reason_is_an_i18n_key_of_the_plate_namespace(self):
        for s in itertools.islice(self._grid(), 0, None, 37):
            self.assertTrue(self.module.video_judgement(s).reason.startswith("plate.reason."))


class PhotoRules(unittest.TestCase):
    """P3 for photos."""

    @classmethod
    def setUpClass(cls):
        cls.module = plates.PlateModule.__new__(plates.PlateModule)
        cls.module.thresholds = plates.DEFAULT_THRESHOLDS

    def test_P3_photo_plate_off_every_vehicle_is_never_blurred(self):
        for score, verify in itertools.product(SCORES, [None] + SCORES):
            det = c.Detection(10, 10, 60, 25, score, c.PLATE, on_vehicle=False)
            retry = None if verify is None else c.Detection(10, 10, 60, 25, verify, c.PLATE, source=c.RETRY)
            self.assertNotEqual(self.module.photo_judgement(det, retry).verdict, c.Verdict.BLUR)

    def test_P3_a_plate_when_vehicles_were_not_looked_for_is_not_blurred(self):
        det = c.Detection(10, 10, 60, 25, 0.95, c.PLATE, on_vehicle=None)
        retry = c.Detection(10, 10, 60, 25, 0.95, c.PLATE, source=c.RETRY)
        self.assertNotEqual(self.module.photo_judgement(det, retry).verdict, c.Verdict.BLUR)


class FakeSession:
    """An onnxruntime session that returns canned rows: [batch, x0, y0, x1, y1, class, score] in input pixels."""

    def __init__(self, rows=(), size: int = 384, error: Optional[Exception] = None):
        self.rows, self.size, self.error = rows, size, error

    def get_inputs(self):
        return [types.SimpleNamespace(name="images", shape=[1, 3, self.size, self.size])]

    def run(self, _outputs, _feed):
        if self.error:
            raise self.error
        return [np.array(self.rows, np.float32).reshape(-1, 7)]


class GeometryFilter(unittest.TestCase):
    """P2: boxes that cannot be plates are dropped at the model's input scale (384 px here, so no rescaling)."""

    IMAGE = np.zeros((384, 384, 3), np.uint8)

    def _kept(self, rows):
        det = plates.PlateDetector(FakeSession(rows))
        return [d.box() for d in det.detect(self.IMAGE)]

    @staticmethod
    def _row(x0, y0, w, h, score=0.9):
        return [0, x0, y0, x0 + w, y0 + h, 0, score]

    def test_P2_box_taller_than_35_percent_of_the_input_is_dropped(self):
        self.assertEqual(self._kept([self._row(20, 20, 300, 134)]), [(20.0, 20.0, 320.0, 154.0)])
        self.assertEqual(self._kept([self._row(20, 20, 300, 135)]), [])

    def test_P2_box_wider_than_90_percent_of_the_input_is_dropped(self):
        self.assertEqual(len(self._kept([self._row(0, 100, 345, 40)])), 1)
        self.assertEqual(self._kept([self._row(0, 100, 346, 40)]), [])

    def test_P2_box_with_aspect_below_1_1_is_dropped(self):
        self.assertEqual(len(self._kept([self._row(50, 50, 56, 50)])), 1)
        self.assertEqual(self._kept([self._row(50, 50, 54, 50)]), [])
        self.assertEqual(self._kept([self._row(50, 50, 40, 60)]), [])

    def test_P2_score_below_the_minimum_is_dropped_and_the_minimum_itself_is_kept(self):
        self.assertEqual(self._kept([self._row(50, 50, 80, 20, 0.24)]), [])
        self.assertEqual(len(self._kept([self._row(50, 50, 80, 20, 0.25)])), 1)

    def test_P2_degenerate_boxes_are_dropped(self):
        rows = [self._row(50, 50, 0, 20), self._row(50, 50, 80, 0), self._row(50, 50, 80, -5)]
        self.assertEqual(self._kept(rows), [])

    def test_P2_a_model_that_raises_gives_no_plates_but_is_counted_and_logged(self):
        det = plates.PlateDetector(FakeSession(error=RuntimeError("NMS failed")))
        with self.assertLogs(plates.log, logging.WARNING) as logged:
            self.assertEqual(det.detect(self.IMAGE), [])
        self.assertEqual(det.run_failures, 1)
        self.assertIn("NMS failed", logged.output[0])

    def test_regions_smaller_than_8_px_are_not_sent_to_the_model(self):
        det = plates.PlateDetector(FakeSession(error=AssertionError("must not run")))
        self.assertEqual(det.detect(self.IMAGE, (10, 10, 17, 200)), [])

    def test_a_closed_detector_raises_instead_of_reporting_no_plates(self):
        det = plates.PlateDetector(FakeSession())
        det.session = None
        with self.assertRaises(RuntimeError):
            det.detect(self.IMAGE)


class Loading(unittest.TestCase):
    def test_missing_models_raise_model_load_error_naming_the_plate_model(self):
        with self.assertRaises(c.ModelLoadError) as raised:
            plates.PlateModule(models="/nonexistent/models")
        self.assertEqual(raised.exception.kind, c.PLATE)
        self.assertIn(plates.PLATE_MODEL, raised.exception.detail)

    def test_module_describes_itself_as_the_contract_asks(self):
        self.assertEqual(plates.PlateModule.kind, c.PLATE)
        self.assertEqual(plates.PlateModule.blur_shape, "rect")
        self.assertEqual(set(plates.PlateModule.video_margins), {c.BOX_SWEEP, c.BOX_RETRY, c.BOX_GAP, c.BOX_EXTEND})


if __name__ == "__main__":
    unittest.main()
