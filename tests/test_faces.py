"""Face rules: boundaries, reasons and the module contract."""

from __future__ import annotations

import re
import unittest

import numpy as np

from anonymizer import contract as c
from anonymizer.classes.faces import FaceModule, FaceThresholds
from tests.faces_support import make_module


class FaceRuleEdges(unittest.TestCase):
    """The boundaries of the face rules, spelled out."""

    @classmethod
    def setUpClass(cls):
        cls.module = make_module()

    def photo(self, sweep, verify):
        det = c.Detection(0, 0, 40, 40, sweep, c.FACE)
        second = None if verify is None else c.Detection(0, 0, 40, 40, verify, c.FACE, c.RETRY)
        return self.module.photo_judgement(det, second)

    def video(self, views, max_score, verify):
        return self.module.video_judgement(c.TrackStats(c.FACE, views, views, max_score, verify, 1, 40.0, False))

    def test_F2_photo_blurs_from_0_84_of_the_second_look(self):
        self.assertEqual(self.photo(0.5, 0.84).verdict, c.Verdict.BLUR)
        self.assertEqual(self.photo(0.5, 0.839).verdict, c.Verdict.CHECK)

    def test_F2_photo_asks_for_review_from_0_70_or_a_sweep_of_0_80(self):
        self.assertEqual(self.photo(0.5, 0.70).verdict, c.Verdict.CHECK)
        self.assertEqual(self.photo(0.80, None).verdict, c.Verdict.CHECK)
        self.assertEqual(self.photo(0.799, 0.699).verdict, c.Verdict.DROP)

    def test_F2_a_confident_sweep_alone_never_blurs_a_photo(self):
        self.assertEqual(self.photo(1.0, None).verdict, c.Verdict.CHECK)

    def test_F3_video_blurs_a_face_seen_three_times(self):
        self.assertEqual(self.video(3, 0.75, 0.60).verdict, c.Verdict.BLUR)
        self.assertEqual(self.video(3, 0.75, 0.60).reason, "face.reason.seen_often")
        self.assertEqual(self.video(2, 0.75, 0.60).verdict, c.Verdict.DROP)
        self.assertEqual(self.video(3, 0.749, 0.60).verdict, c.Verdict.DROP)
        self.assertEqual(self.video(3, 0.75, 0.599).verdict, c.Verdict.DROP)

    def test_F3_video_keeps_the_photo_rule(self):
        self.assertEqual(self.video(1, 0.5, 0.84).verdict, c.Verdict.BLUR)
        self.assertEqual(self.video(1, 0.5, 0.84).reason, "face.reason.verified")

    def test_F1_wants_verify_uses_the_longest_side(self):
        stats = c.TrackStats(c.FACE, 1, 1, 0.9, 0.5, 1, 40.0, False)
        tall = c.Detection(0, 0, 20, 61, 0.9, c.FACE)
        wide = c.Detection(0, 0, 61, 20, 0.9, c.FACE)
        self.assertTrue(self.module.wants_verify(stats, tall))
        self.assertTrue(self.module.wants_verify(stats, wide))

    def test_reasons_are_i18n_keys_with_their_numbers(self):
        key = re.compile(r"^face\.reason\.[a-z_]+$")
        for j in (
            self.photo(0.5, 0.9),
            self.photo(0.5, 0.75),
            self.photo(0.9, None),
            self.photo(0.1, None),
            self.video(3, 0.8, 0.65),
        ):
            self.assertRegex(j.reason, key)
            self.assertTrue(all(isinstance(v, float) for v in j.params.values()))

    def test_module_follows_the_contract(self):
        self.assertIsInstance(self.module, c.ClassModule)
        self.assertEqual(self.module.kind, c.FACE)
        self.assertEqual(self.module.blur_shape, "ellipse")

    def test_a_missing_model_raises_model_load_error(self):
        with self.assertRaises(c.ModelLoadError) as caught:
            FaceModule(models="/nonexistent/models")
        self.assertEqual(caught.exception.kind, c.FACE)

    def test_model_failures_stay_at_zero_because_no_model_error_is_swallowed(self):
        self.assertEqual(self.module.model_failures, 0)
        self.module.sweep(np.zeros((90, 70, 3), np.uint8), c.SweepPlan(general_side=1600, tile_side=640))
        self.assertEqual(self.module.model_failures, 0)

    def test_thresholds_can_be_replaced_for_experiments(self):
        strict = FaceModule(thresholds=FaceThresholds(photo_verify=0.95))
        det = c.Detection(0, 0, 40, 40, 0.5, c.FACE)
        verify = c.Detection(0, 0, 40, 40, 0.9, c.FACE, c.RETRY)
        self.assertEqual(strict.photo_judgement(det, verify).verdict, c.Verdict.CHECK)
        self.assertEqual(self.module.photo_judgement(det, verify).verdict, c.Verdict.BLUR)


if __name__ == "__main__":
    unittest.main()
