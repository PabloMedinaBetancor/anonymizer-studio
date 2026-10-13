"""The shared contract and geometry: the shape of the contract and the rules of the shared geometry."""

from __future__ import annotations

import unittest

from anonymizer import contract as c
from anonymizer import geometry as g


class Geometry(unittest.TestCase):
    def test_nms_without_containment_merges_by_iou_alone(self):
        big, small = c.Detection(0, 0, 100, 100, 0.9, c.FACE), c.Detection(10, 10, 40, 40, 0.8, c.FACE)
        self.assertEqual(len(g.nms([big, small])), 1)  # the face rule: almost inside is a repeat
        self.assertEqual(len(g.nms([big, small], 0.4, contained=None)), 2)
        twin = c.Detection(1, 1, 101, 101, 0.7, c.FACE)
        self.assertEqual(len(g.nms([big, twin], 0.4, contained=None)), 1)  # IoU still merges the same box


class ContractShape(unittest.TestCase):
    def test_verdicts_and_kinds(self):
        self.assertEqual({v.value for v in c.Verdict}, {"blur", "check", "drop"})
        self.assertEqual({c.FACE, c.PLATE}, {"face", "plate"})
        self.assertEqual({c.BOX_SWEEP, c.BOX_RETRY, c.BOX_GAP, c.BOX_EXTEND}, {"sweep", "retry", "gap", "extend"})

    def test_models_dir_override_and_env(self):
        import os

        self.assertEqual(c.models_dir("/x"), "/x")
        old = os.environ.get("ANONYMIZER_MODELS")
        os.environ["ANONYMIZER_MODELS"] = "/y"
        try:
            self.assertEqual(c.models_dir(), "/y")
        finally:
            if old is None:
                del os.environ["ANONYMIZER_MODELS"]
            else:
                os.environ["ANONYMIZER_MODELS"] = old


if __name__ == "__main__":
    unittest.main()


class ContractV2(unittest.TestCase):
    """The names that v2 adds; frozen at G0; a change after it is a decision, not a side effect."""

    def test_new_origins_and_modes(self):
        self.assertEqual((c.SECOND, c.DEEP, c.BOX_SECOND), ("second", "deep", "second"))
        self.assertEqual(c.FOUND_BY, ("light", "second", "deep", "manual"))
        self.assertEqual(c.DECIDED_BY, ("engine", "person"))
        self.assertEqual(c.DETECTION_MODES, ("fast", "careful", "deep"))
        self.assertEqual(c.UNSURE_MODES, ("placed", "blur", "keep"))
        self.assertEqual(c.FIT_MODES, ("tight", "loose"))
        self.assertEqual(c.RetryPurpose.SECOND_LOOK.value, "second_look")

    def test_zone_priorities_are_ordered_and_named(self):
        order = [c.ZONE_DOUBTFUL, c.ZONE_DROPPED_STRONG, c.ZONE_PERSON_HEAD, c.ZONE_DROPPED_WEAK, c.ZONE_VEHICLE]
        self.assertEqual(order, [1, 2, 3, 4, 5])
        self.assertEqual(set(c.ZONE_WHY), set(order))
        self.assertEqual(c.MAX_VEHICLES, {"fast": 12, "careful": 40, "deep": 40})

    def test_the_vk4_figures_are_the_measured_ones(self):
        self.assertEqual((c.KEY_PERIOD_FRAMES, c.HEAVY_PER_SECOND, c.FACE_BOX_SECOND_MARGIN), (8, 6.0, (0.125, 0.152)))

    def test_defaults_keep_the_v1_callers_working(self):
        stats = c.TrackStats("face", 1, 1, 0.9, 0.0, 0, 0.0, False)
        self.assertEqual((stats.best_second, stats.n_second), (0.0, 0))
        plan = c.SweepPlan(640, 0)
        self.assertEqual((plan.deep, plan.deep_tile), (False, 0))
        zone = c.SuspectZone((0, 0, 10, 10), c.FACE, c.ZONE_DOUBTFUL, 1)
        self.assertIsNone(zone.item)

    def test_a_v1_module_is_still_a_class_module_but_not_a_second_look_module(self):
        class Plain:
            kind = c.FACE
            blur_shape = "ellipse"
            video_margins = {}
            photo_margin = (0.0, 0.0)
            timings = {}
            calls = {}
            model_failures = 0

            def sweep(self, bgr, plan): ...
            def retry(self, bgr, expected, purpose): ...
            def photo_judgement(self, det, verify): ...
            def video_judgement(self, track): ...
            def wants_verify(self, track, det): ...
            def wants_recover(self, track): ...
            def close(self): ...

        self.assertIsInstance(Plain(), c.ClassModule)
        self.assertNotIsInstance(Plain(), c.SecondLookModule)
