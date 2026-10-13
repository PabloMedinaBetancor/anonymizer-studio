"""The second look on photos: the second-look table, the budget, failures, the enlargement and "never lowers" (SL3-SL7, PF8)."""

from __future__ import annotations

import inspect
import os
import random
import shutil
import tempfile
import unittest

import numpy as np

from anonymizer import contract
from anonymizer.classes import heavy
from anonymizer.classes.faces import FaceModule
from anonymizer.classes.heavy import HeavyFaceModule
from anonymizer.contract import (
    FACE,
    SECOND,
    ZONE_DOUBTFUL,
    ZONE_DROPPED_STRONG,
    ZONE_DROPPED_WEAK,
    Detection,
    Judgement,
    SecondLookModule,
    SuspectZone,
    TrackStats,
    Verdict,
)
from anonymizer.engine import second_look
from anonymizer.engine.zones import VERDICT_ORDER, LookItem
from bench.bench_cells import first_pass
from tests.heavy_fakes import TEST_TH, FakeHeavy, Scripted

BGR = np.zeros((1000, 1500, 3), np.uint8)
TH = TEST_TH


def item(box, verdict, score=0.8, kind=FACE) -> LookItem:
    det = Detection(*box, score, kind)
    return LookItem(box, kind, score, verdict, Judgement(verdict, "face.reason.test"), det)


def module(placed=(), **fake) -> HeavyFaceModule:
    return HeavyFaceModule(heavy_zone=FakeHeavy(placed, **fake), heavy_thresholds=TEST_TH)


def run(items, placed=(), **kwargs):
    m = module(placed)
    try:
        return second_look.second_stage(BGR, items, [m], **kwargs), m
    finally:
        m.close()


class Table(unittest.TestCase):
    """Each row of 4.3 that the second look decides."""

    def test_a_doubtful_item_the_heavy_model_is_sure_of_becomes_blur_with_the_union_box(self):
        (result, _) = run([item((100, 100, 140, 140), Verdict.CHECK)], [((96, 98, 146, 144), 0.95)])
        (it,) = result.items
        self.assertEqual(it.verdict, Verdict.BLUR)
        self.assertEqual(it.judgement.reason, "face.reason.second_confirmed")
        self.assertEqual(it.box, (96, 98, 146, 144))
        self.assertEqual((it.found_by, it.second_score), ("light", 0.95))
        self.assertEqual((result.confirmed, result.added), (1, 0))

    def test_a_doubtful_item_the_heavy_model_half_believes_stays_doubtful_with_its_score(self):
        (result, _) = run([item((100, 100, 140, 140), Verdict.CHECK)], [((100, 100, 140, 140), 0.8)])
        (it,) = result.items
        self.assertEqual((it.verdict, it.judgement.reason, it.second_score), (Verdict.CHECK, "face.reason.test", 0.8))

    def test_a_doubtful_item_the_heavy_model_does_not_find_stays_doubtful(self):
        (result, m) = run([item((100, 100, 140, 140), Verdict.CHECK)])
        self.assertEqual([i.verdict for i in result.items], [Verdict.CHECK])
        self.assertEqual(len(m.heavy_zone.calls), 1)

    def test_a_dropped_candidate_the_heavy_model_finds_comes_back(self):
        for score, verdict in ((0.95, Verdict.BLUR), (0.8, Verdict.CHECK)):
            (result, _) = run([item((100, 100, 140, 140), Verdict.DROP, 0.7)], [((98, 99, 142, 143), score)])
            (it,) = result.items
            self.assertEqual((it.verdict, it.found_by), (verdict, SECOND), score)
            self.assertEqual(it.judgement.reason, "face.reason.second_found")
            self.assertEqual(result.added, 1)

    def test_a_dropped_candidate_the_heavy_model_cannot_confirm_stays_dropped(self):
        (result, _) = run([item((100, 100, 140, 140), Verdict.DROP, 0.7)], [((100, 100, 140, 140), 0.4)])
        self.assertEqual([i.verdict for i in result.items], [Verdict.DROP])

    def test_the_head_of_a_person_with_no_candidate_gives_a_new_item(self):
        person = Detection(300, 300, 400, 700, 0.9, "person")
        (result, _) = run([], [((330, 310, 370, 350), 0.93)], persons=[person])
        (it,) = result.items
        self.assertEqual((it.verdict, it.found_by, it.kind), (Verdict.BLUR, SECOND, FACE))
        self.assertEqual(result.added, 1)

    def test_a_sure_item_is_never_visited_and_never_doubled(self):
        (result, m) = run([item((100, 100, 140, 140), Verdict.BLUR)], [((100, 100, 140, 140), 0.99)])
        self.assertEqual(m.heavy_zone.calls, [])
        self.assertEqual(len(result.items), 1)

    def test_one_face_found_twice_makes_one_item(self):
        person = Detection(100, 80, 200, 600, 0.9, "person")
        items = [item((110, 90, 150, 130), Verdict.DROP, 0.55)]
        (result, _) = run(items, [((112, 92, 150, 131), 0.95)], persons=[person])
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].verdict, Verdict.BLUR)

    def test_the_caller_list_is_not_changed(self):
        items = [item((100, 100, 140, 140), Verdict.CHECK)]
        run(items, [((100, 100, 140, 140), 0.95)])
        self.assertEqual(items[0].verdict, Verdict.CHECK)


class Budget(unittest.TestCase):
    def crowd(self, n):
        return [item((10 + 30 * i, 100, 30 + 30 * i, 120), Verdict.CHECK) for i in range(n)]

    def test_sl5_a_crowd_above_the_cap_is_visited_up_to_the_cap_and_the_rest_is_counted(self):
        cap = second_look.DEFAULT_MAX_ZONES
        (result, m) = run(self.crowd(cap + 20))
        self.assertEqual((result.zones_total, result.zones_visited, result.zones_skipped), (cap + 20, cap, 20))
        self.assertEqual(len(m.heavy_zone.calls), cap)
        self.assertEqual(result.zones_visited + result.zones_skipped, result.zones_total)

    def test_sl5_the_cap_can_be_set_and_zero_visits_nothing(self):
        for cap, visited in ((5, 5), (0, 0), (500, 60)):
            (result, _) = run(self.crowd(60), max_zones=cap)
            self.assertEqual((result.zones_visited, result.zones_skipped), (visited, 60 - visited), cap)

    def test_sl5_what_the_cap_leaves_out_is_the_lowest_priority(self):
        items = self.crowd(10) + [
            item((900, 500, 930, 530), Verdict.DROP, 0.7),
            item((900, 600, 930, 630), Verdict.DROP, 0.55),
        ]
        (result, _) = run(items, max_zones=11)
        why = [v.zone.why for v in result.visits]
        self.assertEqual(why, [ZONE_DOUBTFUL] * 10 + [ZONE_DROPPED_STRONG])
        self.assertEqual(result.zones_skipped, 1)

    def test_sl5_the_progress_callback_counts_the_visits(self):
        seen = []
        run(self.crowd(3), progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(seen, [(1, 3), (2, 3), (3, 3)])

    def test_the_check_hook_can_stop_the_run(self):
        class Stop(Exception):
            pass

        def check():
            raise Stop

        with self.assertRaises(Stop):
            run(self.crowd(3), check=check)


class Failures(unittest.TestCase):
    def test_sl6_a_failed_look_is_counted_and_the_other_zones_are_still_visited(self):
        m = module(fail_on={1})
        items = [item((100 + 100 * i, 100, 140 + 100 * i, 140), Verdict.CHECK) for i in range(3)]
        result = second_look.second_stage(BGR, items, [m])
        self.assertEqual(m.model_failures, 1)
        self.assertEqual((result.failures, result.zones_visited), (1, 3))
        self.assertEqual([v.failed for v in result.visits], [False, True, False])
        self.assertEqual([i.verdict for i in result.items], [Verdict.CHECK] * 3)

    def test_sl6_a_module_that_lets_the_error_through_is_counted_by_the_engine(self):
        m = module()
        m.second_look = lambda bgr, zone: (_ for _ in ()).throw(heavy.HeavyRunError("boom"))
        result = second_look.second_stage(BGR, [item((100, 100, 140, 140), Verdict.CHECK)], [m])
        self.assertEqual((m.model_failures, result.failures), (1, 1))

    def test_sl6_a_call_over_the_limit_is_counted(self):
        original = second_look.CALL_LIMIT_S
        second_look.CALL_LIMIT_S = 0.005
        try:
            m = module(delay=0.02)
            result = second_look.second_stage(BGR, [item((100, 100, 140, 140), Verdict.CHECK)], [m])
        finally:
            second_look.CALL_LIMIT_S = original
        self.assertEqual(result.slow_calls, 1)


def track(**kw) -> TrackStats:
    base = dict(
        kind=FACE,
        n_views=1,
        n_sweep_views=1,
        max_score=0.5,
        best_verify=0.0,
        n_verify=0,
        verify_side=0.0,
        on_vehicle=False,
    )
    return TrackStats(**{**base, **kw})


class VideoJudgement(unittest.TestCase):
    """VK1: the class rule of a video track reads what the heavy model saw at the key frames, on the real module with the shipped thresholds."""

    def setUp(self):
        self.module = HeavyFaceModule(heavy_zone=FakeHeavy())  # the real class and its real thresholds
        self.addCleanup(self.module.close)
        self.th = self.module.hth

    def test_vk1_a_track_the_light_rule_drops_is_blurred_when_the_heavy_model_is_sure(self):
        j = self.module.video_judgement(track(best_second=self.th.h_hi, n_second=2))
        self.assertEqual((j.verdict, j.reason), (Verdict.BLUR, "face.reason.second_found"))
        self.assertEqual((j.params["best_second"], j.params["n_second"]), (self.th.h_hi, 2.0))

    def test_vk1_between_the_cuts_it_is_doubtful_and_below_the_low_cut_it_is_dropped_as_before(self):
        self.assertEqual(
            self.module.video_judgement(track(best_second=self.th.h_lo, n_second=1)).verdict, Verdict.CHECK
        )
        self.assertEqual(
            self.module.video_judgement(track(best_second=self.th.h_hi - 0.001, n_second=1)).verdict, Verdict.CHECK
        )
        low = self.module.video_judgement(track(best_second=self.th.h_lo - 0.001, n_second=1))
        self.assertEqual((low.verdict, low.reason), (Verdict.DROP, "face.reason.weak"))

    def test_vk1_a_doubtful_track_the_heavy_model_confirms_is_blurred_with_the_confirmed_reason(self):
        doubtful = dict(max_score=0.85, n_sweep_views=1)  # the light rule: CHECK (a strong first score only)
        self.assertEqual(self.module.video_judgement(track(**doubtful)).verdict, Verdict.CHECK)
        j = self.module.video_judgement(track(best_second=0.99, n_second=1, **doubtful))
        self.assertEqual((j.verdict, j.reason), (Verdict.BLUR, "face.reason.second_confirmed"))

    def test_vk1_a_heavy_look_that_found_nothing_changes_nothing(self):
        for stats in (track(max_score=0.85), track(best_verify=0.9, max_score=0.9), track()):
            looked = self.module.video_judgement(track(**{**stats.__dict__, "n_second": 3}))
            light = FaceModule.video_judgement(self.module, stats)
            self.assertEqual(self.module.video_judgement(stats), light)
            # The heavy look adds what it saw to the params (the report shows it) but changes neither verdict nor reason.
            self.assertEqual((looked.verdict, looked.reason), (light.verdict, light.reason))

    def test_vk1_never_lowers_whatever_the_heavy_score_is(self):
        rng = random.Random(11)
        rank = {Verdict.DROP: 0, Verdict.CHECK: 1, Verdict.BLUR: 2}
        for _ in range(400):
            stats = track(
                n_sweep_views=rng.randint(0, 6),
                max_score=rng.uniform(0.3, 1.0),
                best_verify=rng.choice([0.0, rng.uniform(0.0, 1.0)]),
                best_second=rng.choice([0.0, rng.uniform(0.0, 1.0)]),
                n_second=rng.randint(0, 3),
            )
            light = FaceModule.video_judgement(self.module, stats)
            both = self.module.video_judgement(stats)
            self.assertGreaterEqual(rank[both.verdict], rank[light.verdict], stats)
            if both.verdict is light.verdict:
                self.assertEqual(both.reason, light.reason)

    def test_vk1_a_track_the_light_rule_blurs_stays_blurred_with_its_own_reason(self):
        sure = track(best_verify=0.9, max_score=0.9)
        light = FaceModule.video_judgement(self.module, sure)
        self.assertEqual(light.verdict, Verdict.BLUR)
        both = self.module.video_judgement(track(best_verify=0.9, max_score=0.9, best_second=0.2, n_second=1))
        self.assertEqual((both.verdict, both.reason), (light.verdict, light.reason))


class PerPiece(unittest.TestCase):
    """SL6 and E8 are about one model run, not about a whole zone that needed several pieces."""

    def slow_zone_module(self, seconds_per_piece, max_side=64):
        import time

        def rows(_, img):
            time.sleep(seconds_per_piece)
            return []

        return HeavyFaceModule(heavy_zone=Scripted(rows, max_side=max_side), heavy_thresholds=TEST_TH)

    def test_sl6_four_pieces_that_are_each_fast_are_not_one_slow_call(self):
        original = second_look.CALL_LIMIT_S
        second_look.CALL_LIMIT_S = 0.05
        try:
            m = self.slow_zone_module(0.03)
            result = second_look.second_stage(BGR, [item((100, 100, 300, 300), Verdict.CHECK)], [m])
        finally:
            second_look.CALL_LIMIT_S = original
        self.assertGreater(len(m.heavy_zone.pieces), 2)
        self.assertGreater(result.visits[0].seconds, 0.05)  # the zone as a whole took longer than the limit
        self.assertEqual(result.slow_calls, 0)  # but no model run did

    def test_sl6_one_slow_piece_is_one_slow_call(self):
        original = second_look.CALL_LIMIT_S
        second_look.CALL_LIMIT_S = 0.05
        try:
            m = self.slow_zone_module(0.08, max_side=10000)
            result = second_look.second_stage(BGR, [item((100, 100, 140, 140), Verdict.CHECK)], [m])
        finally:
            second_look.CALL_LIMIT_S = original
        self.assertEqual(result.slow_calls, 1)

    def test_e8_the_stop_check_is_looked_at_between_the_pieces_of_a_zone_and_taken_off_afterwards(self):
        m = self.slow_zone_module(0.0)
        calls = []
        second_look.second_stage(BGR, [item((100, 100, 300, 300), Verdict.CHECK)], [m], check=lambda: calls.append(1))
        self.assertGreaterEqual(len(calls), len(m.heavy_zone.pieces) + 1)
        self.assertIsNone(m.heavy_zone.check)

    def test_e8_a_stop_raised_inside_a_zone_ends_the_run_and_the_hook_is_still_taken_off(self):
        class Halt(Exception):
            pass

        m = self.slow_zone_module(0.0)
        count = []

        def check():
            count.append(1)
            if len(count) == 3:
                raise Halt

        with self.assertRaises(Halt):
            second_look.second_stage(BGR, [item((100, 100, 300, 300), Verdict.CHECK)], [m], check=check)
        self.assertIsNone(m.heavy_zone.check)


class Boundaries(unittest.TestCase):
    def test_a_dropped_candidate_at_exactly_065_is_strong_and_just_under_it_is_weak(self):
        from anonymizer.engine.zones import build_zones

        strong = build_zones([item((10, 10, 50, 50), Verdict.DROP, 0.65)])
        weak = build_zones([item((10, 10, 50, 50), Verdict.DROP, 0.6499)])
        self.assertEqual((strong[0].why, weak[0].why), (ZONE_DROPPED_STRONG, ZONE_DROPPED_WEAK))

    def test_a_heavy_score_exactly_at_the_cuts_takes_the_higher_verdict(self):
        m = module()
        for score, verdict in (
            (TH.h_hi, Verdict.BLUR),
            (TH.h_hi - 0.001, Verdict.CHECK),
            (TH.h_lo, Verdict.CHECK),
            (TH.h_lo - 0.001, Verdict.DROP),
        ):
            found = Detection(0, 0, 10, 10, score, FACE, SECOND)
            self.assertEqual(m.second_judgement(None, found).verdict, verdict, score)

    def test_a_heavy_box_over_a_sure_item_and_a_doubtful_one_is_taken_as_the_sure_one(self):
        sure = item((100, 100, 140, 140), Verdict.BLUR)
        doubtful = item((110, 105, 150, 145), Verdict.CHECK)
        (result, _) = run([sure, doubtful], [((105, 102, 145, 142), 0.99)])
        self.assertEqual([i.verdict for i in result.items], [Verdict.BLUR, Verdict.CHECK])
        self.assertEqual(result.items[1].box, doubtful.box)  # nothing was folded into the doubtful one
        self.assertEqual(len(result.items), 2)


class TinyZones(unittest.TestCase):
    def test_a_zone_of_a_few_pixels_is_cut_wider_so_that_it_is_really_looked_at(self):
        region, scale = heavy.enlargement((100, 100, 104, 104), 1500, 1000, 192)
        self.assertGreaterEqual(region[2] - region[0], 16)
        self.assertGreaterEqual(region[3] - region[1], 16)
        self.assertLessEqual(scale, 6.0)
        corner, _ = heavy.enlargement((0, 0, 3, 3), 1500, 1000, 192)
        self.assertEqual(corner[:2], (0.0, 0.0))
        self.assertGreaterEqual(corner[2] - corner[0], 16)
        far, _ = heavy.enlargement((1498, 998, 1500, 1000), 1500, 1000, 192)
        self.assertEqual(far[2:], (1500.0, 1000.0))
        self.assertGreaterEqual(far[2] - far[0], 16)

    def test_a_tiny_dropped_candidate_reaches_the_model(self):
        (result, m) = run([item((100, 100, 104, 104), Verdict.DROP, 0.7)])
        self.assertEqual(len(m.heavy_zone.calls), 1)
        region = m.heavy_zone.calls[0][0]
        self.assertGreaterEqual(region[2] - region[0], 16)
        self.assertEqual(result.zones_visited, 1)


class NeverLowers(unittest.TestCase):
    def test_pf8_nothing_found_or_a_weak_score_never_lowers_a_doubtful_item(self):
        for placed in ((), [((100, 100, 140, 140), 0.55)], [((100, 100, 140, 140), 0.3)]):
            (result, _) = run([item((100, 100, 140, 140), Verdict.CHECK)], placed)
            self.assertEqual(result.items[0].verdict, Verdict.CHECK, placed)

    def test_pf8_the_second_stage_has_no_way_to_turn_a_veto_on(self):
        self.assertNotIn("veto", inspect.signature(second_look.second_stage).parameters)
        self.assertNotIn("veto", inspect.signature(second_look.merge).parameters)

    def test_sl4_monotone_over_random_pictures_and_random_heavy_detectors(self):
        rng = random.Random(4)
        verdicts = (Verdict.BLUR, Verdict.CHECK, Verdict.DROP)

        def box():
            x, y, s = rng.randint(0, 1300), rng.randint(0, 800), rng.randint(15, 90)
            return (x, y, x + s, y + int(s * rng.uniform(0.8, 1.4)))

        for trial in range(250):
            items = [item(box(), rng.choice(verdicts), rng.uniform(0.5, 0.95)) for _ in range(rng.randint(0, 25))]
            placed = [(box(), rng.uniform(0.3, 1.0)) for _ in range(rng.randint(0, 30))]
            placed += [
                (i.box, rng.uniform(0.3, 1.0)) for i in items if rng.random() < 0.5
            ]  # heavy boxes on the same places
            persons = [Detection(*box(), 0.9, "person") for _ in range(rng.randint(0, 4))]
            result, _ = run(items, placed, persons=persons, max_zones=rng.choice([None, 3, 8]))
            self.assertGreaterEqual(len(result.items), len(items), trial)
            for before, after in zip(items, result.items, strict=False):
                self.assertGreaterEqual(VERDICT_ORDER[after.verdict], VERDICT_ORDER[before.verdict], (trial, before))
                self.assertTrue(
                    after.box[0] <= before.box[0]
                    and after.box[1] <= before.box[1]
                    and after.box[2] >= before.box[2]
                    and after.box[3] >= before.box[3],
                    (trial, before.box, after.box),
                )
            for new in result.items[len(items) :]:
                self.assertEqual(new.found_by, SECOND, trial)
                self.assertIn(new.verdict, (Verdict.BLUR, Verdict.CHECK), trial)


class Enlargement(unittest.TestCase):
    def test_sl3_the_zone_is_cut_with_a_quarter_of_context_and_enlarged_to_the_target_side(self):
        region, scale = heavy.enlargement((200, 300, 240, 340), 1500, 1000, 256)
        self.assertEqual(region, (190.0, 290.0, 250.0, 350.0))
        self.assertAlmostEqual(scale, 256 / 60)

    def test_sl3_the_context_never_goes_beyond_the_picture_and_the_scale_stays_in_range(self):
        region, _ = heavy.enlargement((0, 0, 40, 40), 1500, 1000, 256)
        self.assertEqual(region[:2], (0.0, 0.0))
        _, small = heavy.enlargement((10, 10, 12, 12), 1500, 1000, 256)
        _, big = heavy.enlargement((0, 0, 1400, 900), 1500, 1000, 256)
        self.assertEqual((small, big), (6.0, 0.5))

    def test_sl3_a_box_found_in_the_enlarged_crop_comes_back_within_half_a_pixel(self):
        truth = (213.4, 310.2, 233.9, 331.6)  # a face inside a zone of its own
        zone = SuspectZone((200, 300, 240, 340), FACE, ZONE_DOUBTFUL, 0, None)

        def rows(_, img):
            # the crop is the zone with 25 % context enlarged by `scale`: find the face where the image says it was drawn
            ys, xs = np.where(img[:, :, 0] > 200)
            return [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, 0.95)]

        picture = np.zeros((1000, 1500, 3), np.uint8)
        picture[round(truth[1]) : round(truth[3]), round(truth[0]) : round(truth[2])] = 255
        m = HeavyFaceModule(heavy_zone=Scripted(rows), heavy_thresholds=TEST_TH)
        (found,) = m.second_look(picture, zone)
        for got, want in zip(
            found.box(), (round(truth[0]), round(truth[1]), round(truth[2]), round(truth[3])), strict=True
        ):
            self.assertAlmostEqual(got, want, delta=0.5)
        self.assertEqual(found.source, SECOND)


class Loading(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        source = contract.models_dir()
        shutil.copy(os.path.join(source, "SHA256SUMS"), self.folder)
        shutil.copy(os.path.join(source, heavy.YUNET_FILE), self.folder)
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_sl7_careful_without_the_heavy_model_file_is_a_model_load_error_that_says_where_to_look(self):
        with self.assertRaises(contract.ModelLoadError) as raised:
            HeavyFaceModule(models=self.folder)
        message = str(raised.exception)
        self.assertIn(heavy.RETINAFACE_FILE, message)
        for way in ("ANONYMIZER_MODELS", "--models", "models_dir="):
            self.assertIn(way, message)

    def test_sl7_fast_never_loads_a_heavy_model(self):
        m = FaceModule(models=self.folder)  # the same folder, without the heavy file: the light module is fine
        self.assertNotIsInstance(m, SecondLookModule)
        self.assertFalse(hasattr(m, "heavy_zone"))
        m.close()

    def test_a_module_with_a_heavy_model_is_a_second_look_module_and_closes_what_it_opened(self):
        fake = FakeHeavy()
        m = HeavyFaceModule(heavy_zone=fake)
        self.assertIsInstance(m, SecondLookModule)
        m.close()
        self.assertTrue(fake.closed)


HEAVY = os.environ.get("ANONYMIZER_HEAVY") == "1"
SET_A = os.path.join(os.environ.get("ANONYMIZER_BENCH_DATA", ""), "faces", "set-a")


@unittest.skipUnless(HEAVY, "set ANONYMIZER_HEAVY=1 to run the tests that load the models")
class WithRealModels(unittest.TestCase):
    def test_the_real_heavy_model_on_the_zones_of_a_crowd_photo_only_adds(self):
        import glob

        import cv2

        from anonymizer.classes import context

        path = sorted(glob.glob(os.path.join(SET_A, "*.jpg")))[0]
        bgr = cv2.imread(path)
        m = HeavyFaceModule()
        persons = context.load_person_detector()
        try:
            items = first_pass(m, bgr)
            people = context.find_persons(persons, bgr)
            result = second_look.second_stage(bgr, items, [m], persons=people)
        finally:
            m.close()
        self.assertGreater(len(items), 10)
        self.assertGreater(result.zones_visited, 5)
        self.assertEqual((result.failures, m.model_failures), (0, 0))
        self.assertEqual(result.slow_calls, 0)
        for before, after in zip(items, result.items, strict=False):
            self.assertGreaterEqual(VERDICT_ORDER[after.verdict], VERDICT_ORDER[before.verdict])
        self.assertTrue(all(v.seconds < second_look.CALL_LIMIT_S for v in result.visits))


if __name__ == "__main__":
    unittest.main()
