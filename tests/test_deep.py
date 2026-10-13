"""Deep mode on photos (criteria DM1-DM4): the heavy model over the whole picture, united with the first pass."""

from __future__ import annotations

import random
import time
import unittest

import numpy as np

from anonymizer.classes import heavy
from anonymizer.classes.heavy import HeavyFaceModule
from anonymizer.contract import DEEP, FACE, PLATE, Detection, Judgement, SweepPlan, Verdict
from anonymizer.engine import deep
from anonymizer.engine.deep import DeepRefused
from anonymizer.engine.zones import VERDICT_ORDER, LookItem
from tests.heavy_fakes import TEST_TH, FakeHeavy, Scripted

BGR = np.zeros((1000, 1500, 3), np.uint8)


def item(box, verdict, score=0.8, kind=FACE) -> LookItem:
    return LookItem(box, kind, score, verdict, Judgement(verdict, f"{kind}.reason.test"), Detection(*box, score, kind))


def module(placed=(), **fake) -> HeavyFaceModule:
    return HeavyFaceModule(heavy_zone=FakeHeavy(placed, **fake), heavy_thresholds=TEST_TH)


A, B, C, D = (100, 100, 140, 140), (300, 100, 340, 140), (500, 100, 540, 140), (1000, 700, 1040, 750)


class Union(unittest.TestCase):
    def run_deep(self, items, placed, **kwargs):
        m = module(placed)
        try:
            return deep.deep_stage(BGR, items, [m], **kwargs), m
        finally:
            m.close()

    def test_dm1_the_sweep_is_united_with_the_first_pass(self):
        items = [item(A, Verdict.BLUR), item(B, Verdict.CHECK), item(C, Verdict.DROP, 0.7)]
        placed = [(A, 0.99), (B, 0.95), (C, 0.93), (D, 0.92)]
        result, _ = self.run_deep(items, placed)
        by_place = {i.box: i for i in result.items}
        self.assertEqual(len(result.items), 4)
        self.assertEqual((by_place[A].verdict, by_place[A].found_by), (Verdict.BLUR, "light"))
        self.assertGreater(by_place[A].params[DEEP], 0.9)  # seen by both: the first label kept, the second in params
        self.assertEqual((by_place[B].verdict, by_place[B].found_by), (Verdict.BLUR, "light"))
        self.assertEqual(by_place[B].judgement.reason, "face.reason.second_confirmed")
        self.assertEqual((by_place[C].verdict, by_place[C].found_by), (Verdict.BLUR, DEEP))
        self.assertEqual((by_place[D].verdict, by_place[D].found_by), (Verdict.BLUR, DEEP))
        self.assertEqual((result.added, result.confirmed), (2, 1))

    def test_dm1_a_face_seen_in_several_tiles_is_one_item(self):
        result, m = self.run_deep([], [((700, 400, 740, 450), 0.95)])
        self.assertGreater(len(m.heavy_zone.calls), 3)
        self.assertEqual(len(result.items), 1)

    def test_dm1_heavy_only_leaves_out_the_light_result_of_the_classes_with_a_heavy_model(self):
        items = [item(A, Verdict.BLUR), item(B, Verdict.CHECK), item((600, 600, 700, 640), Verdict.BLUR, kind=PLATE)]
        result, _ = self.run_deep(items, [(D, 0.95)], heavy_only=True)
        self.assertEqual([(i.kind, i.found_by) for i in result.items], [(PLATE, "light"), (FACE, DEEP)])

    def test_dm1_the_sweep_sources_are_deep_and_the_caller_list_is_not_changed(self):
        m = module([(D, 0.95)])
        found = m.deep_sweep(BGR, SweepPlan(general_side=1024, tile_side=640, tiles=[], deep=True))
        self.assertEqual([d.source for d in found], [DEEP])
        items = [item(A, Verdict.CHECK)]
        self.run_deep(items, [(A, 0.99)])
        self.assertEqual(items[0].verdict, Verdict.CHECK)

    def test_dm1_deep_never_lowers_over_random_pictures(self):
        rng = random.Random(9)
        verdicts = (Verdict.BLUR, Verdict.CHECK, Verdict.DROP)

        def box():
            x, y, s = rng.randint(0, 1300), rng.randint(0, 800), rng.randint(15, 90)
            return (x, y, x + s, y + s)

        for trial in range(120):
            items = [item(box(), rng.choice(verdicts), rng.uniform(0.5, 0.95)) for _ in range(rng.randint(0, 20))]
            placed = [(box(), rng.uniform(0.3, 1.0)) for _ in range(rng.randint(0, 25))]
            placed += [(i.box, rng.uniform(0.3, 1.0)) for i in items if rng.random() < 0.5]
            result, _ = self.run_deep(items, placed, heavy_only=False)
            for before, after in zip(items, result.items, strict=False):
                self.assertGreaterEqual(VERDICT_ORDER[after.verdict], VERDICT_ORDER[before.verdict], trial)
                self.assertTrue(
                    after.box[0] <= before.box[0]
                    and after.box[1] <= before.box[1]
                    and after.box[2] >= before.box[2]
                    and after.box[3] >= before.box[3],
                    trial,
                )

    def test_dm1_a_failed_call_is_counted_and_the_other_calls_still_run(self):
        m = module([(D, 0.95)], fail_on={1})
        result = deep.deep_stage(BGR, [], [m])
        self.assertEqual((m.model_failures, result.failures), (1, 1))
        self.assertGreater(len(m.heavy_zone.calls), 3)


class Tiles(unittest.TestCase):
    def test_dm1b_a_deep_tile_is_never_bigger_than_640(self):
        m = module()
        picture = np.zeros((2000, 3000, 3), np.uint8)
        m.deep_sweep(picture, SweepPlan(general_side=0, tile_side=2000, tiles=None, deep=True, deep_tile=2000))
        sides = [max(r[2] - r[0], r[3] - r[1]) for r, _ in m.heavy_zone.calls]
        self.assertTrue(sides and max(sides) <= 640)

    def test_dm1b_the_limit_of_tiles_follows_the_tile_size(self):
        self.assertEqual(deep.max_tiles_for(640), 120)
        self.assertEqual(deep.max_tiles_for(320), 480)
        self.assertEqual((deep.tile_side(0), deep.tile_side(1000), deep.tile_side(400)), (640, 640, 400))

    def test_dm1b_a_tile_is_never_smaller_than_320_and_a_negative_side_is_an_error(self):
        self.assertEqual((deep.tile_side(1), deep.tile_side(100), deep.tile_side(319), deep.tile_side(320)), (320,) * 4)
        with self.assertRaises(ValueError):
            deep.tile_side(-5)
        small = deep.call_count(4000, 3000, deep.tile_side(8))
        self.assertEqual(small, deep.call_count(4000, 3000, 320))
        self.assertGreater(small["tiles"], 100)  # a tiny tile cannot turn a big picture into "a few calls"

    def test_dm1b_the_module_sweeps_in_tiles_no_smaller_than_the_floor_whatever_the_plan_says(self):
        m = module()
        m.deep_sweep(
            np.zeros((2000, 3000, 3), np.uint8),
            SweepPlan(general_side=0, tile_side=50, tiles=None, deep=True, deep_tile=50),
        )
        sides = [max(r[2] - r[0], r[3] - r[1]) for r, _ in m.heavy_zone.calls]
        self.assertTrue(sides and min(sides) >= 320 - 1)

    def test_dm1b_every_piece_of_every_call_is_at_most_the_model_side(self):
        d = Scripted([])
        m = HeavyFaceModule(heavy_zone=FakeHeavy(), heavy_deep=d)
        picture = np.zeros((3000, 4000, 3), np.uint8)
        m.deep_sweep(picture, SweepPlan(general_side=1024, tile_side=640, tiles=[0, 5], deep=True))
        self.assertTrue(d.pieces and max(max(p) for p in d.pieces) <= 640)


class Estimate(unittest.TestCase):
    def test_dm2_the_estimate_is_calls_times_the_calibration(self):
        e = deep.estimate(1500, 1000, 400.0)
        counts = deep.call_count(1500, 1000)
        self.assertEqual((e.calls, e.tiles), (counts["calls"], counts["tiles"]))
        self.assertAlmostEqual(e.seconds, counts["calls"] * 0.4, places=2)
        self.assertGreater(e.tiles, 4)

    def test_dm2_a_panorama_is_refused_with_the_number_of_tiles(self):
        with self.assertRaises(DeepRefused) as raised:
            deep.estimate(16000, 4000, 400.0)
        self.assertEqual(raised.exception.key, "error.deep.too_large")
        self.assertGreater(raised.exception.params["tiles"], 120)
        self.assertEqual(raised.exception.params["limit"], 120)

    def test_dm2_the_limit_is_inclusive_and_can_be_set(self):
        tiles = deep.call_count(1500, 1000)["tiles"]
        deep.estimate(1500, 1000, 1.0, max_tiles=tiles)
        with self.assertRaises(DeepRefused):
            deep.estimate(1500, 1000, 1.0, max_tiles=tiles - 1)

    def test_dm2_the_stage_refuses_at_the_exact_limit_of_tiles_and_not_before(self):
        tiles = deep.call_count(1500, 1000)["tiles"]
        m = module()
        deep.deep_stage(BGR, [], [m], max_tiles=tiles)  # at the limit: runs
        self.assertGreater(len(m.heavy_zone.calls), tiles)
        m = module()
        with self.assertRaises(DeepRefused) as raised:
            deep.deep_stage(BGR, [], [m], max_tiles=tiles - 1)
        self.assertEqual(raised.exception.params, {"tiles": tiles, "limit": tiles - 1})
        self.assertEqual(m.heavy_zone.calls, [])

    def test_dm2_a_calibration_during_which_the_model_failed_is_an_error_not_a_time(self):
        m = module(fail_on={0, 1, 2, 3})
        with self.assertRaises(heavy.HeavyRunError):
            deep.calibrate(m)

    def test_dm2_nothing_runs_when_the_picture_is_refused(self):
        m = module()
        with self.assertRaises(DeepRefused):
            deep.deep_stage(BGR, [], [m], max_tiles=2)
        self.assertEqual(m.heavy_zone.calls, [])

    def test_dm2_the_calibration_is_a_median_of_timed_calls_after_the_one_that_opens_the_session(self):
        m = module(delay=0.02)
        ms = deep.calibrate(m)
        self.assertGreater(ms, 15)
        self.assertLess(ms, 400)
        self.assertEqual(len(m.heavy_zone.calls), 1 + deep.CALIBRATION_CALLS)


class Stop(unittest.TestCase):
    def test_dm3_the_check_is_looked_at_between_calls_and_can_stop_the_sweep(self):
        class Halt(Exception):
            pass

        seen = []

        def check():
            seen.append(1)
            if len(seen) >= 3:
                raise Halt

        m = module(delay=0.05)
        started = time.perf_counter()
        with self.assertRaises(Halt):
            deep.deep_stage(BGR, [], [m], check=check)
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertLessEqual(len(m.heavy_zone.calls), 3)
        self.assertIsNone(m.heavy_zone.check)  # the hook is taken off again

    def test_dm3_the_detectors_look_at_the_check_between_the_pieces_of_one_call(self):
        d = Scripted([])
        m = HeavyFaceModule(heavy_zone=FakeHeavy(), heavy_deep=d)
        calls = []
        deep.deep_stage(np.zeros((1000, 1500, 3), np.uint8), [], [m], check=lambda: calls.append(1))
        self.assertGreaterEqual(len(calls), len(d.pieces))

    def test_dm3_a_call_never_takes_more_than_one_piece_of_the_model_side(self):
        d = Scripted([])
        m = HeavyFaceModule(heavy_zone=FakeHeavy(), heavy_deep=d)
        deep.deep_stage(np.zeros((2000, 3000, 3), np.uint8), [], [m])
        self.assertLessEqual(max(max(p) for p in d.pieces), heavy.MAX_INPUT_SIDE)


class Demo(unittest.TestCase):
    def test_dm4_the_hosted_demo_refuses_deep_and_still_allows_careful_and_fast(self):
        with self.assertRaises(DeepRefused) as raised:
            deep.allow_detection("deep", demo=True)
        self.assertEqual(raised.exception.key, "error.demo.deep")
        for detection in ("careful", "fast"):
            deep.allow_detection(detection, demo=True)
        deep.allow_detection("deep", demo=False)


if __name__ == "__main__":
    unittest.main()
