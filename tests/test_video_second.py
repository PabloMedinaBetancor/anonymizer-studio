"""K1-K5, VK1-VK3, VK5, VK6, DM5 and PT7: the heavy model in video at key frames, the tracker that takes its
views, the budget, the box refinement and the text path on tracked vehicles. Every heavy detector here is a stand-in that finds blobs of
colour (tests.fakes), so the clips are small synthetic ones with the colour drawn where a face or a plate is."""

from __future__ import annotations

import fractions
import os
import tempfile
import unittest

try:
    import av
    import numpy as np

    from anonymizer.contract import (
        DEEP,
        FACE,
        PLATE,
        SECOND,
        SWEEP,
        VEHICLE,
        ZONE_DOUBTFUL,
        ZONE_PERSON_HEAD,
        Detection,
        TrackStats,
    )
    from anonymizer.engine import keyframes as kfm
    from anonymizer.engine import tracks as tk
    from anonymizer.engine import video as vid
    from anonymizer.engine.events import Cancelled, ClipError, StopFlag
    from tests import media
    from tests.fakes import ColourTextDetector, SecondColourModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

RED, GREEN, DARK_RED = (0, 0, 255), (0, 255, 0), (0, 0, 200)
W, H, FPS = 320, 240, 25
CALLS_PER_FRAME = 1  # a deep sweep of a 320 x 240 frame: only the general pass (the frame is smaller than a tile)


def background(t: int, cut_at=None) -> np.ndarray:
    """Stripes give the optical flow something to hold on to; after `cut_at` the shot is another (a change of shot)."""
    if cut_at is not None and t >= cut_at:
        img = np.full((H, W, 3), (220, 220, 220), np.uint8)
        img[::24, :] = 30
        return img
    img = np.full((H, W, 3), (60, 90, 120), np.uint8)
    img[::16, :] = 200
    img[:, ::20] = 30
    return img


def write_clip(path: str, n: int, draw, fps: int = FPS, cut_at=None) -> None:
    """`draw(img, t)` paints the frame t over the background; the clip is H.264 so the colours come back a little changed."""
    with av.open(path, "w") as out:
        vs = out.add_stream("libx264", rate=fps)
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        vs.options = {"crf": "14", "preset": "ultrafast"}
        for t in range(n):
            img = background(t, cut_at)
            draw(img, t)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts, frame.time_base = t, fractions.Fraction(1, fps)
            for p in vs.encode(frame):
                out.mux(p)
        for p in vs.encode(None):
            out.mux(p)


def rect(img, box, colour) -> None:
    x0, y0, x1, y1 = (int(v) for v in box)
    img[y0:y1, x0:x1] = colour


def coverage(truth, boxes) -> float:
    """The share of the truth box that the blur boxes cover."""
    tx0, ty0, tx1, ty1 = truth
    mask = np.zeros((int(ty1 - ty0), int(tx1 - tx0)), bool)
    for b, _, _ in boxes:
        x0, y0 = int(max(tx0, b[0]) - tx0), int(max(ty0, b[1]) - ty0)
        x1, y1 = int(min(tx1, b[2]) - tx0), int(min(ty1, b[3]) - ty0)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return float(mask.mean())


class Clips(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def analyse(self, draw, n=60, modules=None, cut_at=None, **kw):
        path = self.path("clip.mp4")
        write_clip(path, n, draw, cut_at=cut_at)
        return vid.analyze_video(path, modules, "standard", "motion", **kw)

    def second(self, light=RED, heavy=(RED,), verify=0.5, **kw):
        return SecondColourModule(FACE, light, list(heavy), verify_score=verify, **kw)


class VK1TrackerAcceptsSecondViews(unittest.TestCase):
    def setUp(self):
        self.tracker = tk.Tracker(25.0)
        for t in range(3):
            self.tracker.step(
                t, tk.IDENTITY.copy(), False, [Detection(100, 100, 160, 160, 0.8, FACE)] if t == 0 else [], True
            )
        self.track = self.tracker.tracks[0]

    def test_vk1_a_second_view_over_a_light_one_is_the_union_of_the_two_boxes(self):
        d = Detection(95, 90, 150, 170, 0.95, FACE, SECOND)
        view = self.tracker.add_second(self.track, 0, d)
        self.assertEqual(view.box(), (95, 90, 160, 170))
        self.assertEqual((view.source, view.score), (SECOND, 0.95))
        self.assertEqual((self.track.n_second, self.track.best_second), (1, 0.95))
        self.assertIs(self.track.views[0], view)

    def test_vk1_the_union_is_never_smaller_than_the_light_box_and_the_score_never_lower(self):
        inside = Detection(110, 110, 150, 150, 0.4, FACE, SECOND)
        view = self.tracker.add_second(self.track, 0, inside)
        self.assertEqual(view.box(), (100, 100, 160, 160))
        self.assertEqual(view.score, 0.8)
        self.assertEqual(self.track.best_second, 0.4)

    def test_vk1_the_light_sweep_view_still_counts_as_a_sweep_view_for_the_class_rule(self):
        before = self.track.stats()
        self.tracker.add_second(self.track, 0, Detection(100, 100, 160, 160, 0.95, FACE, SECOND))
        after = self.track.stats()
        self.assertEqual((before.n_sweep_views, before.n_second, before.best_second), (1, 0, 0.0))
        self.assertEqual((after.n_sweep_views, after.n_second, after.best_second), (1, 1, 0.95))

    def test_vk1_a_second_view_where_the_track_has_none_continues_the_track_like_a_recovery(self):
        self.tracker.add_second(self.track, 2, Detection(102, 100, 162, 160, 0.9, FACE, SECOND))
        self.assertEqual(sorted(self.track.views), [0, 2])
        self.assertEqual((self.track.views[2].source, self.track.misses), (SECOND, 0))

    def test_vk1_a_heavy_detection_no_track_asked_for_joins_the_one_that_expects_it_or_starts_one(self):
        joined, new = self.tracker.offer(2, Detection(104, 100, 164, 160, 0.9, FACE, SECOND))
        self.assertEqual((joined.id, new), (self.track.id, False))
        started, new = self.tracker.offer(2, Detection(250, 30, 300, 80, 0.9, FACE, SECOND))
        self.assertEqual((new, started.n_second, len(self.tracker.tracks)), (True, 1, 2))

    def test_vk1_the_class_rule_reads_best_second_as_it_reads_best_verify(self):
        module = SecondColourModule(FACE, RED, [RED])
        self.assertEqual(module.video_judgement(self.track.stats()).verdict.value, "check")
        self.tracker.add_second(self.track, 0, Detection(100, 100, 160, 160, 0.95, FACE, SECOND))
        self.assertEqual(module.video_judgement(self.track.stats()).verdict.value, "blur")

    def test_vk1_merged_tracks_keep_what_the_second_look_saw(self):
        a, b = tk.Track(0, FACE), tk.Track(1, FACE)
        for track, frames in ((a, (0, 1, 2)), (b, (1, 2, 3))):
            for t in frames:
                track.views[t] = Detection(100, 100, 160, 160, 0.8, FACE, SWEEP)
        b.best_second, b.n_second, b.second_side, b.second_over_sweep = 0.9, 2, 70.0, 1
        (kept,) = tk.merge_duplicates([a, b])
        self.assertEqual((kept.best_second, kept.n_second, kept.second_side, kept.second_over_sweep), (0.9, 2, 70.0, 1))

    def test_vk1_a_heavy_view_has_the_margin_of_its_own_origin(self):
        margins = tk.with_second_margin({"sweep": (0.12, 0.18), "retry": (0.15, 0.20)})
        self.assertAlmostEqual(margins["second"][0], 0.135)
        self.assertAlmostEqual(margins["second"][1], 0.19)
        own = {"sweep": (0.1, 0.1), "retry": (0.2, 0.2), "second": (0.05, 0.05)}
        self.assertEqual(tk.with_second_margin(own)["second"], (0.05, 0.05))
        self.assertEqual(tk.margin_origin(DEEP), "second")
        self.assertEqual(tk.margin_origin(SWEEP), "sweep")


class VK1TheRealFaceRule(unittest.TestCase):
    """HeavyFaceModule.video_judgement reads `best_second` as it reads `best_verify` (RV-VK1 W1), with the product's own rule and thresholds
    (here the test ones: h_hi 0.9, h_lo 0.7)."""

    @classmethod
    def setUpClass(cls):
        from anonymizer.classes.heavy import HeavyFaceModule
        from tests.heavy_fakes import TEST_TH, FakeHeavy

        cls.module = HeavyFaceModule(heavy_zone=FakeHeavy(), heavy_thresholds=TEST_TH)

    @classmethod
    def tearDownClass(cls):
        cls.module.close()

    @staticmethod
    def stats(sweep_views=3, max_score=0.6, verify=0.0, second=0.0, n_second=1):
        return TrackStats(FACE, sweep_views + n_second, sweep_views, max_score, verify, 0, 0.0, False, second, n_second)

    def test_vk1_a_track_the_heavy_model_is_sure_of_is_blur_and_says_who_found_it(self):
        found = self.module.video_judgement(self.stats(sweep_views=0, second=0.95))
        self.assertEqual((found.verdict.value, found.reason), ("blur", "face.reason.second_found"))
        confirmed = self.module.video_judgement(
            self.stats(sweep_views=3, max_score=0.85, second=0.95)
        )  # light: doubtful
        self.assertEqual((confirmed.verdict.value, confirmed.reason), ("blur", "face.reason.second_confirmed"))
        self.assertEqual(confirmed.params["second"], 0.95)

    def test_vk1_a_track_the_heavy_model_half_believes_is_doubtful_and_one_it_doubts_follows_the_light_rule(self):
        self.assertEqual(self.module.video_judgement(self.stats(sweep_views=0, second=0.8)).verdict.value, "check")
        weak = self.module.video_judgement(self.stats(sweep_views=0, second=0.6, max_score=0.1))
        self.assertEqual(weak.verdict.value, "drop")

    def test_vk1_the_two_cut_values_are_inclusive(self):
        hi, lo = self.module.hth.h_hi, self.module.hth.h_lo
        for score, expected in ((hi, "blur"), (hi - 0.001, "check"), (lo, "check"), (lo - 0.001, "drop")):
            got = self.module.video_judgement(self.stats(sweep_views=0, second=score, max_score=0.1))
            self.assertEqual(got.verdict.value, expected, score)

    def test_vk1_the_second_look_never_lowers_what_the_light_rule_decided(self):
        sure = self.stats(sweep_views=3, verify=0.95, second=0.8)  # light: BLUR; heavy: only doubtful
        judgement = self.module.video_judgement(sure)
        self.assertEqual((judgement.verdict.value, judgement.reason), ("blur", "face.reason.verified"))
        self.assertEqual(judgement.params["second"], 0.8)
        light = self.stats(sweep_views=3, max_score=0.9, n_second=0)
        self.assertNotIn(
            "second", self.module.video_judgement(light).params
        )  # a track the heavy model never saw is read as in v1


class VK1JoinAndPersonReach(unittest.TestCase):
    """The two small rules of the key frames that RV-VK1 found unpinned (W3, W4)."""

    def setUp(self):
        self.tracker = tk.Tracker(25.0)
        self.tracker.step(0, tk.IDENTITY.copy(), False, [Detection(100, 100, 160, 160, 0.8, FACE)], True)
        self.track = self.tracker.tracks[0]
        self.kf = kfm.KeyFrames([], 25.0, "careful")

    def test_vk1_a_heavy_detection_far_from_the_track_does_not_enlarge_it_and_starts_its_own(self):
        far = Detection(250, 30, 300, 80, 0.95, FACE, SECOND)
        self.kf._join(self.tracker, self.track, 0, (100, 100, 160, 160), [far])
        self.assertEqual(self.track.views[0].box(), (100, 100, 160, 160))
        self.assertEqual(self.track.n_second, 0)
        self.assertEqual(len(self.tracker.tracks), 2)
        self.assertTrue(
            self.kf.state[1]["looked"]
        )  # the new track was born from a heavy look: K1 does not look at it again
        near = Detection(104, 102, 166, 164, 0.95, FACE, SECOND)
        self.kf._join(self.tracker, self.track, 0, (100, 100, 160, 160), [near])
        self.assertEqual(self.track.views[0].box(), (100, 100, 166, 164))
        self.assertEqual(self.track.n_second, 1)

    def test_vk1_a_person_has_a_face_track_when_its_centre_lies_in_the_upper_half_of_the_person_box(self):
        person = Detection(100, 40, 180, 240, 0.9, "person")  # upper half: y 40-140
        self.tracker.step(1, tk.IDENTITY.copy(), False, [], False)
        for centre, expected in (((140, 100), True), ((140, 150), False), ((190, 100), False), ((90, 100), False)):
            self.track.state = np.array([*centre, 40.0, 50.0])
            self.assertIs(self.kf._has_face_track(self.tracker, person), expected, centre)
        self.track.state = np.array([140.0, 100.0, 40.0, 50.0])
        self.track.kind = PLATE  # a plate under the person's head is not a face
        self.assertFalse(self.kf._has_face_track(self.tracker, person))


class K1Birth(Clips):
    FACE_BOX = (100, 60, 160, 130)

    def draw(self, img, t):
        rect(img, self.FACE_BOX, RED)

    def test_k1_a_track_its_light_verify_did_not_make_sure_of_gets_one_heavy_look_at_its_birth(self):
        module = self.second(verify=0.5)
        analysis = self.analyse(self.draw, modules=[module], detection="careful")
        self.assertEqual(len(module.zones), 1)
        zone = module.zones[0]
        self.assertEqual((zone.kind, zone.why), (FACE, ZONE_DOUBTFUL))
        self.assertTrue(
            zone.box[0] <= 105 and zone.box[1] <= 65 and zone.box[2] >= 155 and zone.box[3] >= 125, zone.box
        )
        (track,) = analysis.tracks
        self.assertEqual((track.n_second, analysis.verdict(track.id).value), (1, "blur"))
        self.assertEqual(analysis.second["k1"], 1)
        margins = analysis.specs[FACE].margins  # a heavy view has its own margin, between the sweep's and the retry's
        self.assertIn("second", margins)

    def test_k1_the_summary_says_how_many_of_the_looks_found_something(self):
        found = self.analyse(self.draw, modules=[self.second(verify=0.5)], detection="careful")
        self.assertEqual((found.second["k1"], found.second["found_k1"]), (1, 1))
        empty = self.analyse(self.draw, modules=[self.second(verify=0.5, heavy=(DARK_RED,))], detection="careful")
        self.assertEqual((empty.second["k1"], empty.second["found_k1"]), (1, 0))  # the heavy model saw nothing there

    def test_k1_a_track_the_light_verify_made_sure_of_is_not_looked_at(self):
        module = self.second(verify=0.9)
        analysis = self.analyse(self.draw, modules=[module], detection="careful")
        self.assertEqual(module.zones, [])
        self.assertEqual((analysis.second["calls"], analysis.second["skipped"]), (0, 0))

    def test_k1_fast_never_calls_the_heavy_model(self):
        module = self.second(verify=0.5)
        analysis = self.analyse(self.draw, modules=[module], detection="fast")
        self.assertEqual((module.zones, module.heavy_zone.calls), ([], 0))
        self.assertEqual(analysis.tracks[0].n_second, 0)

    def test_k1_a_view_seen_one_and_a_half_times_larger_is_looked_at_again(self):
        def growing(img, t):
            side = 30 + t
            rect(img, (100, 60, 100 + side, 60 + side), RED)

        module = self.second(verify=0.5)
        self.analyse(growing, n=70, modules=[module], detection="careful")
        sides = [zone.box[2] - zone.box[0] for zone in module.zones]
        self.assertGreaterEqual(len(sides), 2)
        for before, after in zip(sides, sides[1:], strict=False):
            self.assertGreaterEqual(after, 1.5 * before - 8)  # the zone box has the light box's pad


class K2Loss(Clips):
    FACE_BOX = (100, 60, 160, 130)

    def seen_then(self, afterwards):
        def draw(img, t):
            rect(img, self.FACE_BOX, RED if t < 30 else afterwards)

        return draw

    def test_k2_a_track_that_the_light_models_lose_is_looked_for_by_the_heavy_one_and_continues_where_it_finds_it(self):
        module = self.second(verify=0.9, heavy=(RED, DARK_RED))
        analysis = self.analyse(self.seen_then(DARK_RED), modules=[module], detection="careful")
        (track,) = analysis.tracks
        self.assertGreaterEqual(analysis.second["k2"], 1)
        self.assertEqual(module.zones[0].why, ZONE_DOUBTFUL)
        heavy_frames = [f for f, d in track.views.items() if d.source == SECOND]
        self.assertTrue(heavy_frames and min(heavy_frames) >= 30)
        self.assertGreater(track.last, 50)  # light lost it at 30; the heavy views carried it on
        # not one look per sweep: the next one is a period (KEY_PERIOD_FRAMES = 8 frames) later
        self.assertLessEqual(analysis.second["k2"], 5)

    def test_k2_a_track_the_heavy_model_does_not_find_either_is_looked_for_once_and_ends_with_the_extension_margins(
        self,
    ):
        module = self.second(verify=0.9)
        analysis = self.analyse(
            self.seen_then((60, 90, 120)),
            modules=[module],
            detection="careful",
            key_params=kfm.KeyFrameParams(period=4),
        )
        (track,) = analysis.tracks
        self.assertEqual(len(module.zones), 1)  # even with a short period: what was not found is not looked for again
        x0, y0, x1, y1 = module.zones[0].box
        self.assertTrue(x0 <= 105 and y0 <= 65 and x1 >= 155 and y1 >= 125, module.zones[0].box)  # the predicted box
        self.assertEqual(track.n_second, 0)
        self.assertLess(track.last, 30)
        boxes = vid.boxes_to_blur(analysis, {track.id})
        self.assertIn(track.last + 1, boxes)  # extended
        self.assertNotIn(analysis.frames - 1, boxes)

    def test_k2_a_track_the_class_rule_drops_is_not_looked_for(self):
        module = self.second(
            verify=0.1, score=0.4, heavy=(DARK_RED,)
        )  # weak and never confirmed: DROP; the heavy model sees no red
        analysis = self.analyse(self.seen_then((60, 90, 120)), modules=[module], detection="careful")
        self.assertEqual(analysis.listed(), [])
        self.assertEqual(
            (len(module.zones), analysis.second["k1"], analysis.second["k2"]), (1, 1, 0)
        )  # its birth, not its loss


def person_at(box):
    return lambda bgr: [Detection(*box, 0.9, "person")]


class K3K4Persons(Clips):
    PERSON = (100, 40, 180, 200)  # head zone: (92, 40, 188, 96)
    HEAD_FACE = (120, 50, 160, 90)

    def heads(self, module):
        return [z for z in module.zones if z.why == ZONE_PERSON_HEAD]

    def hidden_face(self, img, t):
        rect(img, self.HEAD_FACE, RED)

    def test_k3_at_a_change_of_shot_the_heads_of_persons_with_no_face_track_are_visited(self):
        module = self.second(light=GREEN, heavy=(RED,))  # the light models never see this face: only the heavy one does
        params = kfm.KeyFrameParams(period=1000)
        analysis = self.analyse(
            self.hidden_face,
            n=60,
            modules=[module],
            cut_at=30,
            detection="careful",
            key_params=params,
            persons=person_at(self.PERSON),
        )
        self.assertTrue(analysis.cuts, "the clip must have a change of shot")
        zones = self.heads(module)
        self.assertEqual(len(zones), 2)  # the first frame and the cut
        self.assertEqual(zones[0].box, tuple(kfm.head_zone(self.PERSON)))
        self.assertEqual((analysis.second["k3"], analysis.second["k4"]), (2, 0))
        found = [t for t in analysis.tracks if t.n_second]
        self.assertTrue(found)
        self.assertTrue(all(d.source == SECOND for t in found for d in list(t.views.values())[:1]))

    def test_k3_a_person_whose_head_already_has_a_face_track_is_not_visited(self):
        module = self.second(verify=0.9)  # the light model sees the face and the track is sure
        self.analyse(
            self.hidden_face,
            n=40,
            modules=[module],
            detection="careful",
            persons=person_at(self.PERSON),
            key_params=kfm.KeyFrameParams(period=1000),
        )
        self.assertEqual(self.heads(module), [])

    def test_k4_every_n_frames_the_persons_with_no_face_track_get_their_head_visited(self):
        module = self.second(
            light=GREEN, heavy=(DARK_RED,)
        )  # a person with no face to find: nothing ever starts a track

        def nobody(img, t):
            pass

        params = kfm.KeyFrameParams(period=10)
        analysis = self.analyse(
            nobody, n=60, modules=[module], detection="careful", key_params=params, persons=person_at(self.PERSON)
        )
        self.assertEqual((analysis.second["k3"], analysis.second["k4"]), (1, 5))  # frame 0, then 10, 20, 30, 40, 50
        self.assertEqual(len(self.heads(module)), 6)

    def test_k3_k4_need_a_person_finder_and_the_careful_mode(self):
        module = self.second(light=GREEN, heavy=(RED,))
        self.analyse(self.hidden_face, n=30, modules=[module], detection="careful")
        self.analyse(self.hidden_face, n=30, modules=[module], detection="fast", persons=person_at(self.PERSON))
        self.assertEqual(module.zones, [])


class K5DeepAndDM5(Clips):
    def draw(self, img, t):
        rect(img, (100, 60, 160, 130), RED)

    def test_k5_deep_sweeps_the_whole_frame_at_the_first_frame_and_every_n_deep_frames(self):
        module = self.second(light=GREEN, heavy=(RED,))
        params = kfm.KeyFrameParams(deep_period=20)
        analysis = self.analyse(self.draw, n=60, modules=[module], detection="deep", key_params=params)
        self.assertEqual(module.deep_calls, 3 * CALLS_PER_FRAME)  # frames 0, 20, 40
        self.assertEqual(analysis.second["k5"], 3)
        (track,) = [t for t in analysis.tracks if t.kind == FACE]
        self.assertTrue(all(track.views[f].source == DEEP for f in (0, 20, 40)))
        self.assertTrue(
            {d.source for d in track.views.values()} <= {DEEP, SECOND}
        )  # between sweeps: the loss looks (K2) carry it
        self.assertGreater(track.n_second, 0)

    def test_k5_a_change_of_shot_is_a_deep_key_frame_too(self):
        module = self.second(light=GREEN, heavy=(RED,))
        params = kfm.KeyFrameParams(deep_period=1000)
        analysis = self.analyse(self.draw, n=60, modules=[module], cut_at=30, detection="deep", key_params=params)
        self.assertTrue(analysis.cuts)
        self.assertEqual(module.deep_calls, 2 * CALLS_PER_FRAME)

    def test_dm5_deep_in_video_is_not_every_frame_and_careful_never_sweeps_the_whole_frame(self):
        module = self.second(light=GREEN, heavy=(RED,))
        analysis = self.analyse(self.draw, n=60, modules=[module], detection="deep")
        self.assertLess(module.deep_calls, analysis.frames / 10 * CALLS_PER_FRAME)
        careful = self.second(light=GREEN, heavy=(RED,))
        self.analyse(self.draw, n=60, modules=[careful], detection="careful")
        self.assertEqual(careful.deep_calls, 0)

    def test_dm5_a_big_frame_is_swept_as_the_deep_stage_of_photos_does_one_call_for_the_general_pass_and_one_per_tile(
        self,
    ):
        from anonymizer.engine import deep

        module = self.second(light=GREEN, heavy=(RED,))
        frame = np.zeros((1000, 1500, 3), np.uint8)
        kf = kfm.KeyFrames([module], 25.0, "deep")
        kf._whole_frame(module, frame)
        self.assertEqual(
            module.deep_calls, 1 + deep.call_count(1500, 1000)["tiles"]
        )  # the general pass, then each tile
        self.assertGreater(module.deep_calls, 2)

    def test_k5_the_deep_key_frame_runs_the_whole_frame_sweep_of_every_heavy_module(self):
        from anonymizer.engine import deep

        module = self.second(light=GREEN, heavy=(RED,))
        kf = kfm.KeyFrames([module], 25.0, "deep", kfm.KeyFrameParams(deep_ms_per_frame=1.0))
        kf.deep_sweep(tk.Tracker(25.0), 0, np.zeros((1000, 1500, 3), np.uint8), False, 100)
        self.assertEqual(module.deep_calls, 1 + deep.call_count(1500, 1000)["tiles"])
        self.assertEqual(kf.summary()["k5"], 1)  # one deep key frame, whatever its number of calls

    def test_dm5_stop_is_felt_between_the_calls_of_one_whole_frame_sweep(self):
        module = self.second(light=GREEN, heavy=(RED,))
        stop = StopFlag()
        inner = module.deep_sweep

        def sweep_then_stop(bgr, plan):
            found = inner(bgr, plan)
            stop.request()
            return found

        module.deep_sweep = sweep_then_stop
        kf = kfm.KeyFrames([module], 25.0, "deep", stop=stop)
        with self.assertRaises(Cancelled):
            kf._whole_frame(module, np.zeros((1000, 1500, 3), np.uint8))
        self.assertEqual(module.deep_calls, 1)

    def test_dm5_the_estimate_is_the_sweeps_times_the_cost_of_one(self):
        params = kfm.KeyFrameParams(deep_period=50, deep_ms_per_frame=2000.0)
        self.assertEqual(kfm.estimate_deep_work(500, 25.0, params), 11 * 2.0)
        self.assertIsNone(kfm.estimate_deep_work(500, 25.0, kfm.KeyFrameParams()))


class VK2Budget(Clips):
    def test_vk2_a_second_never_gets_more_calls_than_the_budget_and_the_lowest_priority_goes_first(self):
        budget = kfm.Budget(4, 25.0)  # frames 0-24 are the first second, 25-49 the second one
        self.assertEqual([budget.take(0, kfm.K4) for _ in range(3)], [True, True, False])  # K4 may use half
        self.assertEqual([budget.take(5, kfm.K3) for _ in range(2)], [True, False])  # K3 three quarters: one more
        self.assertEqual(
            [budget.take(9, kfm.K1) for _ in range(2)], [True, False]
        )  # a birth takes the last of the four
        self.assertTrue(budget.take(25, kfm.K4))  # the next second starts again
        self.assertEqual((budget.calls, budget.skipped), (5, 3))

    def test_vk2_a_look_refused_and_asked_for_again_counts_once_and_not_at_all_when_it_is_served(self):
        budget = kfm.Budget(2, 25.0)
        self.assertTrue(budget.take(0, kfm.K1, ("k1", 1)) and budget.take(1, kfm.K1, ("k1", 2)))
        for sweep in range(
            2, 12, 2
        ):  # the third birth is refused at every sweep of the second: one look skipped, not five
            self.assertFalse(budget.take(sweep, kfm.K1, ("k1", 3)))
        self.assertEqual((budget.calls, budget.skipped), (2, 1))
        self.assertTrue(budget.take(25, kfm.K1, ("k1", 3)))  # served in the next second
        self.assertEqual((budget.calls, budget.skipped), (3, 0))
        self.assertFalse(budget.take(26, kfm.K4))
        self.assertFalse(budget.take(27, kfm.K4))
        self.assertEqual(budget.skipped, 2)  # looks nobody names (periodic ones) count each refusal

    def test_vk2_births_beyond_the_budget_are_skipped_counted_and_kept_in_the_analysis(self):
        boxes = [(20 + 55 * i, 60, 60 + 55 * i, 100) for i in range(5)]

        def draw(img, t):
            for b in boxes:
                rect(img, b, RED)

        module = self.second(verify=0.5)
        analysis = self.analyse(
            draw, n=30, modules=[module], detection="careful", key_params=kfm.KeyFrameParams(heavy_per_second=2)
        )
        self.assertLessEqual(len(module.zones), 4)  # 30 frames at 25 fps touch two seconds: 2 + 2
        self.assertGreater(analysis.second["skipped"], 0)
        self.assertLessEqual(
            analysis.second["skipped"], len(boxes)
        )  # one skipped look per face, not one per sweep it was refused at
        self.assertEqual(analysis.second["calls"], len(module.zones))
        reloaded = vid.VideoAnalysis.from_json(analysis.to_json())
        self.assertEqual(reloaded.second, analysis.second)

    def test_vk2_a_budget_of_zero_makes_no_heavy_call_and_says_so(self):
        module = self.second(verify=0.5)
        analysis = self.analyse(
            lambda img, t: rect(img, (100, 60, 160, 130), RED),
            n=30,
            modules=[module],
            detection="careful",
            key_params=kfm.KeyFrameParams(heavy_per_second=0),
        )
        self.assertEqual((module.heavy_zone.calls, analysis.second["calls"]), (0, 0))
        self.assertGreater(analysis.second["skipped"], 0)


class VK2TheApiSaysWhatTheBudgetRefused(Clips):
    BOXES = [(10 + 38 * i, 60, 40 + 38 * i, 100) for i in range(8)]

    def test_vk2_a_warning_with_the_count_and_the_counts_in_the_summary(self):
        import anonymizer

        def draw(img, t):
            for b in self.BOXES:
                rect(img, b, RED)

        path = self.path("clip.mp4")
        write_clip(path, 20, draw)
        module = self.second(verify=0.5)
        events = []
        with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), modules=[module]) as anon:
            analysis = anon.analyze(path, events.append)
        warnings = [e for e in events if isinstance(e, anonymizer.JobWarning)]
        (warning,) = [w for w in warnings if w.code == "warning.second_look.zones_skipped"]
        summary = analysis.summary()
        self.assertEqual(warning.params["count"], summary["zones_skipped"])
        self.assertGreater(summary["zones_skipped"], 0)
        self.assertEqual(summary["zones_visited"], len(module.zones))
        item = next(i for i in analysis.items if i.second_score is not None)
        self.assertEqual(
            (item.found_by, item.second_score), ("light", 0.95)
        )  # the light sweep found it; the heavy model confirmed

    def test_vk2_a_track_only_the_heavy_model_found_says_so_in_the_report(self):
        import anonymizer

        path = self.path("clip.mp4")
        write_clip(path, 20, lambda img, t: rect(img, (100, 60, 160, 130), RED))
        module = self.second(light=GREEN, heavy=(RED,))
        with anonymizer.Anonymizer(anonymizer.Options(detection="deep"), modules=[module]) as anon:
            analysis = anon.analyze(path)
        (item,) = analysis.items
        self.assertEqual((item.found_by, item.kind), ("deep", FACE))


class VK3TheBoxIsRefined(Clips):
    TRUTH = (100, 60, 200, 160)  # the face; the light model sees only the inner 60 %

    def draw(self, img, t):
        rect(img, self.TRUTH, RED)
        rect(img, (120, 80, 180, 140), GREEN)

    def test_vk3_the_final_blur_covers_the_truth_box_at_the_key_frame(self):
        module = self.second(light=GREEN, heavy=(GREEN, RED), verify=0.5)
        analysis = self.analyse(self.draw, n=30, modules=[module], detection="careful")
        (track,) = analysis.tracks
        boxes = vid.boxes_to_blur(analysis, {track.id})
        self.assertGreaterEqual(coverage(self.TRUTH, boxes[0]), 0.95)

    def test_vk3_the_control_the_light_model_alone_leaves_the_face_part_uncovered(self):
        module = self.second(light=GREEN, heavy=(GREEN, RED), verify=0.5)
        analysis = self.analyse(self.draw, n=30, modules=[module], detection="fast")
        (track,) = analysis.tracks
        boxes = vid.boxes_to_blur(analysis, {track.id})
        self.assertLess(coverage(self.TRUTH, boxes[0]), 0.95)


class VK5RenderUnchanged(Clips):
    def test_vk5_a_render_from_the_saved_analysis_equals_the_fresh_one_and_runs_no_detector(self):
        module = self.second(light=GREEN, heavy=(GREEN, RED), verify=0.5)
        source = self.path("clip.mp4")

        def draw(img, t):
            rect(img, (100, 60, 200, 160), RED)
            rect(img, (120, 80, 180, 140), GREEN)

        write_clip(source, 30, draw)
        analysis = vid.analyze_video(source, [module], "standard", "motion", detection="careful")
        calls = (module.calls["sweep"], module.calls.get("second", 0), module.heavy_zone.calls)
        ids = {t.id for t in analysis.tracks}
        vid.render_video(source, self.path("fresh.mp4"), analysis, ids)
        saved = vid.VideoAnalysis.from_json(analysis.to_json())
        vid.render_video(source, self.path("saved.mp4"), saved, ids)
        self.assertEqual(calls, (module.calls["sweep"], module.calls.get("second", 0), module.heavy_zone.calls))
        for t in (0, 10, 29):
            self.assertTrue(
                np.array_equal(
                    media.decoded_frame(self.path("fresh.mp4"), t), media.decoded_frame(self.path("saved.mp4"), t)
                )
            )
        self.assertTrue(any(d["source"] == "second" for d in self.views(analysis)), "the heavy views are in the file")

    @staticmethod
    def views(analysis):
        out = []
        for tj in analysis.to_json()["tracks"]:
            out += [{"source": v[5]} for v in tj["views"].values()]
        return out


class VK6DeepEstimateAndStop(Clips):
    def draw(self, img, t):
        rect(img, (100, 60, 160, 130), RED)

    def test_vk6_a_clip_whose_deep_estimate_is_over_the_limit_is_refused_with_the_number_before_any_heavy_call(self):
        module = self.second(light=GREEN, heavy=(RED,))
        params = kfm.KeyFrameParams(deep_period=1, deep_ms_per_frame=1000.0, deep_max_seconds_of_work=30.0)
        path = self.path("clip.mp4")
        write_clip(path, 60, self.draw)
        with self.assertRaises(ClipError) as raised:
            vid.analyze_video(path, [module], "standard", "motion", detection="deep", key_params=params)
        self.assertEqual(raised.exception.code, "error.deep.too_much_work")
        self.assertEqual(raised.exception.params, {"minutes": 2, "limit": 0})  # 61 sweeps of 1 s = 61 s
        self.assertEqual((module.deep_calls, module.heavy_zone.calls), (0, 0))

    def test_vk6_a_clip_under_the_limit_runs(self):
        module = self.second(light=GREEN, heavy=(RED,))
        params = kfm.KeyFrameParams(deep_period=20, deep_ms_per_frame=10.0, deep_max_seconds_of_work=30.0)
        self.analyse(self.draw, n=60, modules=[module], detection="deep", key_params=params)
        self.assertEqual(module.deep_calls, 3 * CALLS_PER_FRAME)

    def test_vk6_without_a_calibration_the_first_sweep_is_measured_and_the_clip_refused_if_it_is_too_much(self):
        module = self.second(light=GREEN, heavy=(RED,))
        original = module.heavy_zone.detect

        def slow(*a, **kw):
            import time

            time.sleep(0.05)
            return original(*a, **kw)

        module.heavy_zone.detect = slow
        params = kfm.KeyFrameParams(deep_period=1, deep_max_seconds_of_work=1.0)  # 61 sweeps of 50 ms are 3 s
        with self.assertRaises(ClipError) as raised:
            self.analyse(self.draw, n=60, modules=[module], detection="deep", key_params=params)
        self.assertEqual(raised.exception.code, "error.deep.too_much_work")
        self.assertEqual(module.deep_calls, CALLS_PER_FRAME)

    def test_vk6_stop_is_felt_before_the_next_heavy_call(self):
        module = self.second(verify=0.5)
        stop = StopFlag()
        module.heavy_zone.on_call = lambda n: stop.request() if n == 1 else None
        boxes = [(20 + 55 * i, 60, 60 + 55 * i, 100) for i in range(5)]

        def draw(img, t):
            for b in boxes:
                rect(img, b, RED)

        with self.assertRaises(Cancelled):
            self.analyse(draw, n=30, modules=[module], detection="careful", stop=stop)
        self.assertEqual(module.heavy_zone.calls, 1)


class GrowingVehicles:
    """A vehicle detector whose answer depends on the sweep: `boxes[k]` is the list of vehicles at the k-th whole-picture call."""

    session = None

    def __init__(self, boxes):
        self.boxes = boxes
        self.sweeps = 0

    def detect(self, bgr, region=None, min_score=None):
        k = self.sweeps
        self.sweeps += 1
        boxes = self.boxes[min(k, len(self.boxes) - 1)]
        return [Detection(*b, 0.9, VEHICLE) for b in boxes]


class PT7TextPathOnTrackedVehicles(Clips):
    """The white plate of tests.media.moving_plate_clip moves 2 px per frame over a vehicle that fills the middle of the picture."""

    VEHICLE = (10, 70, 300, 170)

    def module(self, vehicles):
        from anonymizer.classes.plates import TextPlateModule

        text = ColourTextDetector([((250, 250, 250), 0.9)])
        return TextPlateModule(vehicles=vehicles, text=text), text

    def clip(self, n=100):
        """A white rectangle of the plate's size moving 2 px per frame (no letters: the colour stand-in would find their holes)."""
        path = self.path("plate.mp4")
        write_clip(path, n, lambda img, t: rect(img, (20 + 2 * t, 100, 120 + 2 * t, 130), (250, 250, 250)), fps=30)
        return path

    def go(self, vehicles, n=100, **kw):
        module, text = self.module(vehicles)
        analysis = vid.analyze_video(self.clip(n), [module], "standard", "motion", **kw)
        return module, text, analysis

    def test_pt7_the_text_detector_runs_at_the_vehicles_key_frames_and_not_at_every_sweep(self):
        vehicles = GrowingVehicles([[self.VEHICLE]])
        module, text, analysis = self.go(vehicles, key_params=kfm.KeyFrameParams(period=20))
        self.assertEqual(len(text.crops), 5)  # frames 0, 20, 40, 60, 80
        self.assertGreater(vehicles.sweeps, 40)  # the vehicles are found at every sweep
        self.assertEqual(analysis.second["k1"], 1)
        self.assertEqual(analysis.second["k4"], 4)

    def test_pt7_the_regions_are_tracked_as_plate_items_and_the_blur_follows_the_plate_between_looks(self):
        module, text, analysis = self.go(GrowingVehicles([[self.VEHICLE]]), key_params=kfm.KeyFrameParams(period=20))
        (track,) = [t for t in analysis.tracks if t.kind == PLATE]
        self.assertEqual(analysis.verdict(track.id).value, "blur")
        self.assertEqual(analysis.judgements[track.id].reason, "plate.reason.vehicle_text")
        self.assertTrue(track.on_vehicle)
        self.assertEqual(sorted(track.views), [0, 20, 40, 60, 80])  # the looks; the blur between them is carried
        boxes = vid.boxes_to_blur(analysis, {track.id})
        for t in range(0, 80):
            self.assertGreaterEqual(coverage(media.plate_box(t), boxes[t]), 0.9, t)

    def test_pt7_a_vehicle_that_grows_one_and_a_half_times_is_looked_at_again(self):
        sweeps = [[(10, 70, min(110 + 8 * k, 210), 170)] for k in range(60)]  # 100 px wide, then up to 200
        module, text, analysis = self.go(GrowingVehicles(sweeps), n=60, key_params=kfm.KeyFrameParams(period=1000))
        sides = [max(shape[:2]) for shape in text.crops]
        self.assertEqual(len(text.crops), 2)  # the birth and the look at 1.5 times the size
        self.assertEqual((analysis.second["k1"], analysis.second["k4"]), (2, 0))
        self.assertTrue(all(side == 960 for side in sides))  # every crop is enlarged to T_V

    def test_pt7_a_vehicle_the_detector_stops_finding_is_looked_for_once_at_its_predicted_box(self):
        sweeps = [[self.VEHICLE]] * 5 + [[]]
        module, text, analysis = self.go(GrowingVehicles(sweeps), n=60, key_params=kfm.KeyFrameParams(period=1000))
        self.assertEqual(len(text.crops), 2)  # the birth and one look for the loss
        self.assertEqual((analysis.second["k1"], analysis.second["k2"]), (1, 1))

    def test_pt7_vehicles_too_small_to_read_are_counted_in_the_summary_of_a_video(self):
        import anonymizer

        module, _ = self.module(GrowingVehicles([[(10, 70, 60, 100)]]))  # 50 x 30
        with anonymizer.Anonymizer(modules=[module]) as anon:
            analysis = anon.analyze(self.clip(30))
        self.assertGreaterEqual(analysis.summary()["vehicles_too_small"], 1)
        self.assertEqual(module.vehicles_visited, 0)

    def test_pt7_the_looks_obey_the_budget_and_what_it_refuses_is_counted(self):
        module, text, analysis = self.go(
            GrowingVehicles([[self.VEHICLE]]), n=60, key_params=kfm.KeyFrameParams(heavy_per_second=0)
        )
        self.assertEqual(len(text.crops), 0)
        self.assertGreater(analysis.second["skipped"], 0)
        self.assertEqual([t for t in analysis.tracks if t.kind == PLATE], [])

    def test_pt7_the_text_path_works_in_every_detection_mode_and_a_text_failure_makes_the_frame_not_checked(self):
        module, text, analysis = self.go(GrowingVehicles([[self.VEHICLE]]), n=40, detection="fast")
        self.assertTrue([t for t in analysis.tracks if t.kind == PLATE])
        module, text = self.module(GrowingVehicles([[self.VEHICLE]]))
        text.fail_on = {1}
        analysis = vid.analyze_video(self.clip(40), [module], "standard", "motion")
        self.assertEqual(analysis.unchecked, [0])

    def test_pt7_a_region_seen_in_three_views_is_blurred_whatever_its_score_and_in_two_it_is_doubtful(self):
        module, _ = self.module(GrowingVehicles([[self.VEHICLE]]))
        rule = module.text_path
        weak = (rule.tau_lo + rule.tau_hi) / 2

        def stats(views, score):
            return TrackStats(PLATE, views, views, score, 0.0, 0, 0.0, True)

        self.assertEqual(module.video_judgement(stats(3, weak)).verdict.value, "blur")
        self.assertEqual(module.video_judgement(stats(2, weak)).verdict.value, "check")
        self.assertEqual(module.video_judgement(stats(9, rule.tau_lo - 0.01)).verdict.value, "drop")
        self.assertEqual(module.video_judgement(stats(1, rule.tau_hi)).verdict.value, "blur")

    def test_pt7_a_photo_still_gets_its_text_on_every_vehicle_in_one_sweep(self):
        from anonymizer.contract import SweepPlan

        module, text = self.module(GrowingVehicles([[self.VEHICLE]]))
        img = background(0)
        rect(img, (50, 100, 150, 130), (250, 250, 250))
        found = module.sweep(img, SweepPlan(960, 640, max_vehicles=6))
        self.assertEqual(len(found), 1)
        self.assertEqual(len(text.crops), 1)
        self.assertEqual(module.sweep(img, SweepPlan(960, 640, max_vehicles=0)), [])
        self.assertEqual(len(text.crops), 1)  # the video census finds vehicles and visits none...
        self.assertEqual(
            (module.vehicles_visited, module.vehicles_not_visited), (1, 0)
        )  # ...and does not count them as skipped


if __name__ == "__main__":
    unittest.main()
