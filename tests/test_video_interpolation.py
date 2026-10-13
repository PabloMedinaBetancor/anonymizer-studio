"""VK4: the tool that measures how far the interpolated blur stays on the face between two heavy key frames. The planning, the
camera-motion record, the offline replay, the margin and the speed proposal are tested here with synthetic windows; the real figure is
`bench/results/video-interpolation-*.json`, run with the real heavy model on the public clips."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np

    from anonymizer.contract import Detection
    from bench import bench_video_interpolation as vi
    from tests.test_video_second import rect, write_clip
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

W, H = 1920, 1080
RAW_BASE = {
    "thresholds": {"h_hi": 0.98, "h_lo": 0.52, "look_min": 0.5},
    "machine": {"host": "test"},
    "heavy_model": {"name": "fake", "sha256": "0" * 64},
}


def box_at(x, y, side=60.0):
    return [x, y, x + side, y + side * 1.25, 0.95]


def window(path, motions=None, offsets=vi.GRID, faces=1):
    """A window of WINDOW frames: `path(face, t)` is the box of a face at offset t; `motions` the camera motion (default: still camera)."""
    motions = motions or [[1, 0, 0, 0, 1, 0]] * vi.WINDOW
    return {
        "motions": motions,
        "cut_at": None,
        "detections": {str(t): [path(f, t) for f in range(faces)] for t in offsets},
        "sweep_ms": [],
        "zone_ms": [],
    }


def raw_of(windows, name="clip"):
    return {
        **RAW_BASE,
        "clips": {
            name: {
                "width": W,
                "height": H,
                "total_frames": 1000,
                "windows": {str(i * 100): w for i, w in enumerate(windows)},
            }
        },
    }


class Planning(unittest.TestCase):
    CLIPS = {
        "a": {"total_frames": 2000},
        "b": {"total_frames": 500},
        "c": {"total_frames": 40},
    }  # c is shorter than a window

    def test_vk4_the_windows_are_spread_in_proportion_to_the_clips_and_fit_inside_them(self):
        windows = vi.plan_windows(self.CLIPS, 10, {})
        self.assertEqual({n for n, _ in windows}, {"a", "b"})
        self.assertGreaterEqual(sum(1 for n, _ in windows if n == "a"), 6)
        for name, start in windows:
            self.assertTrue(0 <= start and start + vi.WINDOW <= self.CLIPS[name]["total_frames"], (name, start))

    def test_vk4_windows_of_one_clip_do_not_overlap_and_the_plan_is_the_same_every_time(self):
        windows = vi.plan_windows(self.CLIPS, 12, {})
        for name in ("a", "b"):
            starts = sorted(s for n, s in windows if n == name)
            self.assertTrue(all(b - a >= vi.WINDOW for a, b in zip(starts, starts[1:], strict=False)), starts)
        self.assertEqual(windows, vi.plan_windows(self.CLIPS, 12, {}))
        self.assertNotEqual(windows, vi.plan_windows(self.CLIPS, 12, {}, seed=1))

    def test_vk4_no_window_comes_near_a_truth_frame(self):
        truth = {"a": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800], "b": [100, 300]}
        windows = vi.plan_windows(self.CLIPS, 12, truth)
        self.assertTrue(windows)
        for name, start in windows:
            for t in truth[name]:
                self.assertFalse(start - 15 <= t <= start + vi.WINDOW + 15, (name, start, t))


class Replay(unittest.TestCase):
    def evaluate(self, windows):
        return vi.evaluate(raw_of(windows))

    def test_vk4_a_face_that_moves_in_a_straight_line_is_covered_at_every_spacing_and_the_claim_is_made(self):
        windows = [
            window(lambda f, t, k=k: box_at(200 + 300 * f + k * 30 + t, 300 + t // 2, 80), faces=2) for k in range(16)
        ]
        result = self.evaluate(windows)
        for n in vi.SPACINGS:
            cell = result["spacings"][str(n)]
            self.assertGreaterEqual(cell["n"], vi.MIN_CASES, n)
            self.assertEqual((cell["p95"], cell["pair_rate"]), (0.0, 1.0), n)
            self.assertTrue(cell["holds"], n)
        self.assertEqual((result["period_proposal"], result["claim_made"]), (48, True))

    def test_vk4_a_face_that_accelerates_is_lost_to_the_join_at_the_long_spacings_and_the_claim_stops_there(self):
        windows = [window(lambda f, t, k=k: box_at(200 + k * 30 + t + 0.1 * t * t, 300, 80)) for k in range(40)]
        result = self.evaluate(windows)
        self.assertEqual(
            result["spacings"]["48"]["pair_rate"], 0.0
        )  # moved 280 px in 48 frames: no longer the same face for the join
        self.assertFalse(result["spacings"]["48"]["holds"])
        chosen = result["period_proposal"]
        self.assertTrue(chosen is not None and chosen < 48 and result["spacings"][str(chosen)]["holds"])
        self.assertTrue(all(not result["spacings"][str(n)]["holds"] for n in vi.SPACINGS if n > chosen))

    def test_vk4_an_error_larger_than_the_margins_is_measured_not_hidden(self):
        # the face walks to the right and stops at the second key frame: the straight line between the key frames is wrong in between
        def path(f, t, k):
            return box_at(
                300 + 40 * k + (60 if t >= 12 else 0) * (1 if t < 16 else 0) + (60 if t >= 16 else 0), 300, 80
            )

        windows = [window(lambda f, t, k=k: path(f, t, k), faces=2) for k in range(16)]
        cell = self.evaluate(windows)["spacings"]["16"]
        self.assertGreater(cell["p95"], vi.P95_LIMIT)
        self.assertFalse(cell["holds"])

    def test_vk4_the_camera_motion_of_the_window_carries_a_face_the_camera_pans_past(self):
        # a still face seen by a camera whose pan speeds up: 0.2 px more a frame every frame, so the face's path in the picture is a parabola
        def shift(t):
            return -0.2 * t if t else 0.0

        pan = [[1, 0, shift(t), 0, 1, 0] for t in range(vi.WINDOW)]
        x = lambda t: 1000 + sum(shift(u) for u in range(1, t + 1))  # noqa: E731
        windows = [window(lambda f, t, k=k: box_at(x(t) + 10 * k, 300, 80), motions=pan, faces=1) for k in range(40)]
        result = self.evaluate(windows)
        self.assertEqual((result["spacings"]["48"]["p95"], result["spacings"]["48"]["pair_rate"]), (0.0, 1.0))
        self.assertEqual(result["period_proposal"], 48)
        no_motion = [window(lambda f, t, k=k: box_at(x(t) + 10 * k, 300, 80), faces=1) for k in range(40)]
        left_out = self.evaluate(
            no_motion
        )  # the same boxes with the camera motion left out: the parabola is not carried
        self.assertFalse(left_out["spacings"]["48"]["holds"])
        self.assertLess(left_out["period_proposal"] or 0, 48)

    def test_vk4_a_face_the_heavy_model_sees_only_between_the_key_frames_is_counted_apart_never_dropped(self):
        def path(f, t):
            return box_at(300, 300, 80) if f == 0 else box_at(1500, 700, 80)

        windows = [window(path, faces=2) for _ in range(3)]
        for w in windows:  # the second face exists only at offset 12, between the key frames
            for t in list(w["detections"]):
                if t != "12":
                    w["detections"][t] = w["detections"][t][:1]
        cell = self.evaluate(windows)["spacings"]["16"]
        self.assertEqual(cell["unmatched_heavy_boxes"], 3)
        self.assertEqual(cell["tracked_pairs"], 3)
        self.assertEqual(cell["p95"], 0.0)

    def test_vk4_a_spacing_at_which_most_faces_cannot_be_followed_to_the_second_key_frame_is_not_claimed(self):
        def path(f, t):
            return box_at(100 + 250 * f, 300, 80)

        windows = []
        for _ in range(6):
            w = window(path, faces=6)
            for t in list(w["detections"]):
                if int(t) >= 24:  # four of the six faces are not seen by the heavy model from offset 24 on
                    w["detections"][t] = w["detections"][t][:2]
            windows.append(w)
        result = self.evaluate(windows)
        short, long = result["spacings"]["16"], result["spacings"]["24"]
        self.assertEqual((short["pair_rate"], short["p95"], short["holds"]), (1.0, 0.0, True))
        self.assertEqual(
            (long["pair_rate"], long["p95"], long["n"] >= vi.MIN_CASES, long["holds"]), (0.333, 0.0, True, False)
        )
        self.assertEqual(result["period_proposal"], 16)

    def test_vk4_too_few_cases_never_make_the_claim(self):
        result = self.evaluate([window(lambda f, t: box_at(300 + t, 300, 80))])
        self.assertLess(result["spacings"]["48"]["n"], vi.MIN_CASES)
        self.assertEqual((result["period_proposal"], result["claim_made"]), (None, False))

    def test_vk4_detections_under_the_score_are_not_a_face(self):
        weak = [window(lambda f, t: [300 + t, 300, 380, 400, 0.3]) for _ in range(4)]
        result = self.evaluate(weak)
        self.assertEqual(result["spacings"]["48"]["n"], 0)

    def test_vk4_the_cases_are_reported_by_distance_to_the_nearest_key_frame_and_by_clip(self):
        windows = [window(lambda f, t: box_at(300 + t, 300, 80), faces=2) for _ in range(4)]
        cell = self.evaluate(windows)["spacings"]["16"]
        self.assertEqual(
            sorted(int(d) for d in cell["by_distance"]), [2, 4, 6, 8]
        )  # offsets 2..12 at spacing 16: min(j, 16 - j)
        self.assertEqual(list(cell["by_clip"]), ["clip"])
        self.assertEqual(cell["by_clip"]["clip"]["n"], cell["n"])

    def test_vk4_what_the_result_says_about_itself(self):
        result = self.evaluate([window(lambda f, t: box_at(300 + t, 300, 80))])
        for key in (
            "machine",
            "heavy_model",
            "caveat",
            "p95_limit",
            "min_cases",
            "min_pair_rate",
            "windows",
            "min_score",
        ):
            self.assertIn(key, result)
        self.assertEqual((result["p95_limit"], result["windows"]), (0.01, 1))


class DevOnly(unittest.TestCase):
    TRUTH = {"clips": {"a": {"frames": {"10": {}, "20": {}}}, "b": {"frames": {"5": {}}}}}

    def test_vk4_only_the_frames_the_split_marks_dev_are_read(self):
        table = {"video/a/10": "dev", "video/a/20": "sealed", "video/b/5": "dev"}
        self.assertEqual(vi.dev_truth_frames(self.TRUTH, table), {"a": [10], "b": [5]})

    def test_vk4_a_frame_the_split_does_not_name_stops_the_run(self):
        with self.assertRaises(SystemExit) as stopped:
            vi.dev_truth_frames(self.TRUTH, {"video/a/10": "dev", "video/a/20": "dev"})
        self.assertIn("video/b/5", str(stopped.exception))

    def test_vk4_a_part_that_is_neither_dev_nor_sealed_is_an_error(self):
        with self.assertRaises(SystemExit):
            vi.dev_truth_frames(self.TRUTH, {"video/a/10": "dev", "video/a/20": "test", "video/b/5": "dev"})

    def test_vk4_only_the_dev_frames_are_decoded_and_swept(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_clip(
                os.path.join(tmp, "a.mp4"), 40, lambda img, t: rect(img, (2 * t, 100, 2 * t + 20, 140), (0, 0, 255))
            )
            truth = {"clips": {"a": {"frames": {"10": {"boxes": [{"box": [20, 100, 40, 140]}]}, "20": {"boxes": []}}}}}
            clips = {"a": {"file": "a.mp4", "width": 320, "height": 240}}
            swept = []

            def sweep(bgr):
                swept.append(1)
                return [Detection(18, 98, 42, 142, 0.9, "face")]

            got = vi.collect_truth_frames(tmp, truth, clips, sweep, {"a": [10]})
        self.assertEqual(list(got["a"]), ["10"])
        self.assertEqual(len(swept), 1)  # the sealed frame 20 was never swept
        self.assertEqual(got["a"]["10"]["truth"], [[20, 100, 40, 140]])


class SealedDiscipline(unittest.TestCase):
    """RV-VK4 W1, W5: a sealed clip is never selected, by `collect` or by the careful run; the clip names come from the split."""

    TRUTH = {
        "clips": {
            "a": {"file": "a.webm", "frames": {"10": {"boxes": [{"box": [0, 0, 5, 5]}]}}},
            "b": {"file": "b.webm", "frames": {"5": {"boxes": [{"box": [0, 0, 5, 5]}]}}},
            "s": {"file": "s.webm", "frames": {"7": {"boxes": [{"box": [0, 0, 5, 5]}]}}},
            "n": {"file": "n.webm", "frames": {"3": {"boxes": [{"box": [0, 0, 5, 5], "ignore": True}]}}},
        }
    }
    TABLE = {"video/a/10": "dev", "video/b/5": "dev", "video/s/7": "sealed", "video/n/3": "dev", "face/x/1": "sealed"}

    def dev(self):
        return vi.dev_truth_frames(self.TRUTH, self.TABLE)

    def test_vk4_the_dev_clip_names_come_from_the_split_and_a_clip_is_one_unit(self):
        self.assertEqual(vi.dev_clip_names(self.TABLE), ["a", "b", "n"])
        with self.assertRaises(SystemExit):
            vi.dev_clip_names({"video/a/1": "dev", "video/a/2": "sealed"})

    def test_vk4_a_sealed_clip_is_never_selected_with_or_without_names(self):
        self.assertEqual(
            sorted(vi.select_clips(self.TRUTH, self.dev(), None)), ["a", "b"]
        )  # s sealed, n has no labelled face
        self.assertEqual(sorted(vi.select_clips(self.TRUTH, self.dev(), ["b"])), ["b"])
        for name in ("s", "nowhere"):
            with self.assertRaises(SystemExit) as stopped:
                vi.select_clips(self.TRUTH, self.dev(), ["a", name])
            self.assertIn(name, str(stopped.exception))

    def test_vk4_the_careful_run_stops_on_a_sealed_name_before_it_opens_any_clip(self):
        from bench import bench_video_careful as careful

        with tempfile.TemporaryDirectory() as tmp:
            truth, split = os.path.join(tmp, "truth.json"), os.path.join(tmp, "split.json")
            with open(truth, "w", encoding="utf-8") as f:
                json.dump(self.TRUTH, f)
            with open(split, "w", encoding="utf-8") as f:
                json.dump(self.TABLE, f)
            argv = ["--clips", tmp, "--truth-video", truth, "--split-file", split, "--clip-names", "s"]
            with mock.patch.object(careful, "trim") as trim, self.assertRaises(SystemExit):
                careful.main(argv)
            trim.assert_not_called()
            with (
                mock.patch.object(careful, "trim", return_value=0) as trim,
                mock.patch.object(careful, "run_clip", return_value={"seconds": 0}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                careful.main(argv[:-2])  # no names: the dev clips with a labelled face, from the split
            self.assertEqual([os.path.basename(c.args[0]) for c in trim.call_args_list], ["a.webm", "b.webm"])


class Selection(unittest.TestCase):
    """RV-VK4 W3, W4, W2: the claim rule, the frame rate, the tail."""

    def test_vk4_the_chosen_spacing_has_every_smaller_one_holding(self):
        self.assertEqual(vi.largest_holding_prefix({4: True, 6: True, 8: False, 12: True, 16: True}), 6)
        self.assertIsNone(vi.largest_holding_prefix({4: False, 6: True, 8: True}))
        self.assertEqual(vi.largest_holding_prefix({n: True for n in vi.SPACINGS}), vi.SPACINGS[-1])
        self.assertIsNone(vi.largest_holding_prefix({}))

    def test_vk4_each_clip_is_replayed_with_its_own_frame_rate_and_the_claim_is_in_seconds_too(self):
        raw = raw_of([window(lambda f, t: box_at(300 + t, 300, 80), faces=2) for _ in range(16)], name="slow")
        raw["clips"]["slow"]["fps"] = 25.0
        raw["clips"]["fast"] = {**raw["clips"]["slow"], "fps": 60.0}
        raw["clips"]["old"] = {k: v for k, v in raw["clips"]["slow"].items() if k != "fps"}
        result = vi.evaluate(raw, given_fps={"old": 30.0})
        self.assertEqual(result["period_proposal"], 48)
        self.assertEqual(result["period_proposal_seconds"], {"slow": 1.92, "fast": 0.8, "old": 1.6})
        self.assertEqual(
            {k: (v["fps"], v["source"]) for k, v in result["clip_rates"].items()},
            {"slow": (25.0, "recorded"), "fast": (60.0, "recorded"), "old": (30.0, "clip file")},
        )
        self.assertEqual(result["spacings"]["48"]["seconds"], {"min": 0.8, "max": 1.92})
        assumed = vi.evaluate({**raw, "clips": {"x": raw["clips"]["old"]}}, fps=24.0)
        self.assertEqual(assumed["clip_rates"]["x"], {"fps": 24.0, "source": "assumed"})

    def test_vk4_the_frame_rate_reaches_the_gap_filling_of_the_clip(self):
        seen = []
        real = vi.interpolate_pair

        def spy(a, b, spacing, motions, width, height, fps, margins):
            seen.append(fps)
            return real(a, b, spacing, motions, width, height, fps, margins)

        raw = raw_of([window(lambda f, t: box_at(300 + t, 300, 80)) for _ in range(2)])
        raw["clips"]["clip"]["fps"] = 50.0
        with mock.patch.object(vi, "interpolate_pair", spy):
            vi.evaluate(raw)
        self.assertTrue(seen and set(seen) == {50.0})

    def test_vk4_the_table_has_the_tail_and_the_claim_names_the_clips_over_the_limit(self):
        shares = [0.0] * 90 + [0.05] * 6 + [0.2] * 3 + [0.7]
        tail = vi.tail_of(shares)
        self.assertEqual((tail["max"], tail["over_10pct"], tail["over_50pct"]), (0.7, 0.04, 0.01))
        self.assertEqual(tail["p99"], round(float(np.percentile(shares, 99)), 4))
        self.assertEqual(vi.tail_of([])["max"], None)
        windows = [window(lambda f, t: box_at(300 + t, 300, 80), faces=2) for _ in range(16)]
        result = vi.evaluate(raw_of(windows))
        cell = result["spacings"]["48"]
        for key in ("p99", "max", "over_10pct", "over_50pct"):
            self.assertEqual(cell[key], 0.0)
            self.assertIn(key, cell["by_clip"]["clip"])
        self.assertIn("48 frames", result["claim"])
        self.assertIn("pooled over 1 clips", result["claim"])
        table = {"8": {**cell, "by_clip": {"hk": {"p95": 0.019}, "ok": {"p95": 0.0}}}}
        rates = {"hk": {"fps": 25.0}, "ok": {"fps": 25.0}}
        text = vi.claim_text(table, 8, rates)
        self.assertIn("hk 1.9 %", text)
        self.assertNotIn("ok 0.0", text)
        self.assertIn("not made", vi.claim_text(table, None, rates))

    def test_vk4_the_measured_margin_can_replace_the_placeholder_in_the_replay(self):
        windows = [window(lambda f, t: box_at(300 + t, 300, 80), faces=2) for _ in range(4)]
        default = vi.evaluate(raw_of(windows))
        measured = vi.evaluate(raw_of(windows), second_margin=(0.125, 0.152))
        self.assertEqual(measured["margins_used"]["second"], [0.125, 0.152])
        self.assertIsInstance(default["margins_used"]["second"], (str, list))


class JoinAndCases(unittest.TestCase):
    """RV-VK4 W6: one-line cases of the join rule, the chain gap, the denominator, the margin and the cut of a window."""

    MOTIONS = [np.array([[1, 0, 0], [0, 1, 0]], float)] * 60
    PARAMS = vi.tk.TrackerParams()

    def same(self, a, b):
        return vi.same_face(a, b, self.MOTIONS, 0, 4, self.PARAMS)

    def test_vk4_a_box_that_overlaps_is_the_same_face_however_far_its_centre_is_and_a_close_one_is_too_with_little_overlap(
        self,
    ):
        a = (0, 0, 100, 100)
        self.assertTrue(self.same(a, (-70, 36, 170, 276)))  # IoU 0.105 and a centre 1.06 sides away
        self.assertTrue(
            self.same(a, (85, 0, 185, 100))
        )  # IoU 0.08 (under the minimum) but the centre is 0.85 sides away
        self.assertFalse(self.same(a, (200, 0, 300, 100)))  # neither
        self.assertFalse(self.same(a, (0, 300, 100, 400)))

    def test_vk4_a_face_of_a_size_that_cannot_be_the_same_is_not_joined_even_when_the_boxes_overlap(self):
        self.assertFalse(self.same((0, 0, 100, 100), (-100, -100, 200, 200)))  # three times the size, IoU 0.11
        self.assertFalse(self.same((0, 0, 100, 100), (40, 40, 60, 60)))  # a fifth of the size

    def box_chain_window(self, offsets):
        return window(lambda f, t: box_at(300, 300, 80), offsets=offsets)

    def evaluate_window(self, window, spacing):
        from anonymizer.classes.faces import FaceModule

        return vi.evaluate_window(window, spacing, W, H, 25.0, dict(FaceModule.video_margins), 0.52, self.PARAMS)

    def test_vk4_a_face_missed_for_more_than_two_steps_is_not_followed_across_the_gap(self):
        far = self.evaluate_window(self.box_chain_window((0, 48)), 48)  # a gap of 48 frames: more than 2 x 16
        self.assertEqual((far["tracked"], far["key_faces"]), (0, 1))
        edge = self.evaluate_window(self.box_chain_window((0, 32)), 32)  # exactly 32: still followed
        self.assertEqual((edge["tracked"], edge["key_faces"]), (1, 1))

    def test_vk4_a_weak_box_between_the_key_frames_is_not_counted_as_a_face_nobody_follows(self):
        w = self.box_chain_window(vi.GRID)
        w["detections"]["12"].append([900, 500, 960, 575, 0.3])  # under the score
        got = self.evaluate_window(w, 16)
        self.assertEqual(got["unmatched"], 0)
        w["detections"]["12"][-1][4] = 0.9  # the same box as a real detection that no face carries
        self.assertEqual(self.evaluate_window(w, 16)["unmatched"], 1)

    def test_vk4_a_heavy_box_that_already_holds_the_label_needs_no_margin_and_never_a_negative_one(self):
        got = vi.box_second_margin(
            {"c": {"0": {"truth": [[110, 110, 190, 190]], "heavy": [[100, 100, 200, 200, 0.9]]}}}
        )
        self.assertEqual((got["x"], got["y"]), (0.0, 0.0))

    def test_vk4_a_window_ends_at_the_change_of_shot_and_the_frame_that_changes_is_not_swept(self):
        shot_a = np.full((240, 320, 3), 60, np.uint8)
        shot_a[::16] = 200
        shot_b = np.full((240, 320, 3), 220, np.uint8)
        frames = {t: (shot_a if t < 24 else shot_b).copy() for t in range(vi.WINDOW)}  # the cut falls on a grid offset
        record = vi.window_record(frames, 1.0, lambda bgr: [])
        self.assertEqual(record["cut_at"], 24)
        self.assertEqual(sorted(int(k) for k in record["detections"]), [t for t in vi.GRID if t < 24])


class Proposals(unittest.TestCase):
    def test_vk4_the_margin_is_the_90th_percentile_of_what_the_heavy_box_needs_to_contain_the_labelled_one(self):
        frames = {"c": {}}
        for i in range(10):  # the labelled box sticks out by i / 10 of the heavy width on the right and 0.05 above
            frames["c"][str(i)] = {"truth": [[100, 100, 200 + 10 * i, 200]], "heavy": [[100, 105, 200, 200, 0.9]]}
        got = vi.box_second_margin(frames)
        self.assertEqual((got["n"], got["percentile"]), (10, 90))
        self.assertAlmostEqual(got["x"], 0.81, places=2)
        self.assertAlmostEqual(got["y"], 0.053, places=2)
        self.assertIsNone(vi.box_second_margin({}))
        self.assertIsNone(
            vi.box_second_margin({"c": {"0": {"truth": [[0, 0, 10, 10]], "heavy": [[500, 500, 600, 600, 0.9]]}}})
        )

    def test_vk4_heavy_calls_per_second_follow_the_speed_of_a_zone_look(self):
        self.assertEqual(vi.propose_heavy_per_second(100.0)["value"], 5)
        self.assertEqual(vi.propose_heavy_per_second(10.0)["value"], 12)  # capped
        self.assertEqual(vi.propose_heavy_per_second(5000.0)["value"], 1)  # never zero
        self.assertIsNone(vi.propose_heavy_per_second(None))

    def test_vk4_uncovered_is_the_share_of_the_heavy_box_outside_the_blur(self):
        self.assertEqual(vi.uncovered([0, 0, 10, 10], [0, 0, 10, 10]), 0.0)
        self.assertEqual(vi.uncovered([0, 0, 10, 10], [20, 20, 30, 30]), 1.0)
        self.assertAlmostEqual(vi.uncovered([0, 0, 10, 10], [5, 0, 20, 10]), 0.5)
        self.assertEqual(vi.uncovered([2, 2, 8, 8], [0, 0, 10, 10]), 0.0)


class Recording(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def test_vk4_the_record_has_the_camera_motion_of_every_frame_and_the_heavy_boxes_at_the_grid_only(self):
        rng = np.random.default_rng(3)
        scene = rng.integers(30, 220, (270, 800), dtype=np.uint8)
        scene = np.repeat(np.repeat(scene[::6, ::6], 6, axis=0), 6, axis=1)  # blocky texture the optical flow can hold
        scene = np.repeat(scene[:, :, None], 3, axis=2)
        frames = {
            t: np.ascontiguousarray(scene[:, 2 * t : 2 * t + 640]) for t in range(vi.WINDOW)
        }  # a camera panning 2 px a frame
        calls = []

        def sweep(bgr):
            calls.append(1)
            return [Detection(10, 20, 70, 95, 0.9, "face")]

        record = vi.window_record(frames, 0.5, sweep)
        self.assertEqual(len(record["motions"]), vi.WINDOW)
        self.assertEqual(sorted(int(k) for k in record["detections"]), list(vi.GRID))
        self.assertEqual(len(calls), len(vi.GRID))
        self.assertIsNone(record["cut_at"])
        shifts = [m[0][2] for m in record["motions"][1:]]
        self.assertTrue(
            all(-3.0 < s < -1.0 for s in shifts), shifts[:5]
        )  # a pan right is a shift of the content to the left
        json.dumps(record)  # plain data

    def test_vk4_a_change_of_shot_ends_the_window_for_the_heavy_model(self):
        shot_a = np.full((240, 320, 3), 60, np.uint8)
        shot_a[::16] = 200
        shot_b = np.full((240, 320, 3), 220, np.uint8)
        frames = {t: (shot_a if t < 20 else shot_b).copy() for t in range(vi.WINDOW)}
        record = vi.window_record(frames, 1.0, lambda bgr: [])
        self.assertEqual(record["cut_at"], 20)
        self.assertEqual(sorted(int(k) for k in record["detections"]), [t for t in vi.GRID if t < 20])

    def test_vk4_the_windows_are_read_from_a_clip_in_order_with_the_frames_they_name(self):
        path = os.path.join(self.dir, "clip.mp4")
        write_clip(path, 130, lambda img, t: rect(img, (2 * t, 100, 2 * t + 20, 140), (0, 0, 255)))
        got = {start: frames for start, frames in vi.read_clip_windows(path, [60, 5])}
        self.assertEqual(sorted(got), [5, 60])
        for start, frames in got.items():
            self.assertEqual(sorted(frames), list(range(vi.WINDOW)))
            for offset in (0, 20, 48):
                red = np.where((frames[offset][:, :, 2] > 200) & (frames[offset][:, :, 1] < 60))[1]
                self.assertAlmostEqual(float(red.min()), 2 * (start + offset), delta=3)

    def test_vk4_the_command_line_replays_a_raw_file_into_a_results_file(self):
        raw = raw_of([window(lambda f, t: box_at(300 + t, 300, 80), faces=2) for _ in range(16)])
        raw_path, out_path = os.path.join(self.dir, "raw.json"), os.path.join(self.dir, "out.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vi.main(["evaluate", "--raw", raw_path, "--out", out_path]), 0)
        with open(out_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["period_proposal"], 48)


if __name__ == "__main__":
    unittest.main()
