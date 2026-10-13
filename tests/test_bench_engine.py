"""bench/bench_engine.py: the figures it derives are consistent with the data they come from (no detector runs here)."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

try:
    from bench import bench_engine
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class GapRound(unittest.TestCase):
    def test_motion_is_at_least_as_good_as_linear_and_hold_on_every_camera(self):
        r = bench_engine.gap_round()
        self.assertGreaterEqual(r["motion_pct"], r["linear_pct"])
        self.assertGreaterEqual(r["linear_pct"], r["hold_pct"])
        for camera, rates in r["by_camera"].items():
            self.assertGreaterEqual(rates["motion_pct"], rates["hold_pct"], camera)
        self.assertGreater(r["object_frames"], 5000)

    def test_it_is_deterministic(self):
        self.assertEqual(bench_engine.gap_round(), bench_engine.gap_round())


class StandardVersusMaximum(unittest.TestCase):
    def test_figures_follow_from_the_committed_results(self):
        r = bench_engine.standard_vs_maximum()
        with open(os.path.join(bench_engine.HERE, "results", "faces-video.json"), encoding="utf-8") as f:
            raw = json.load(f)["faces_video"]
        self.assertNotIn("reason", r)
        hit = {q: sum(raw[q][c]["faces_hit_blur"] for c in raw[q]) for q in ("standard", "maximum")}
        self.assertEqual(
            (r["faces_blurred_by_standard"], r["faces_blurred_by_maximum"]), (hit["standard"], hit["maximum"])
        )
        self.assertLessEqual(r["faces_blurred_by_both"], min(hit.values()))
        self.assertEqual(r["coverage_pct"], round(100 * r["faces_blurred_by_both"] / hit["maximum"], 2))
        slow = sum(raw["maximum"][c]["analysis_seconds"] for c in raw["maximum"])
        fast = sum(raw["standard"][c]["analysis_seconds"] for c in raw["standard"])
        self.assertEqual(r["speedup"], round(slow / fast, 2))

    def test_without_the_results_it_says_why_instead_of_a_number(self):
        with mock.patch("builtins.open", side_effect=FileNotFoundError(2, "no", "faces-video.json")):
            r = bench_engine.standard_vs_maximum()
        self.assertEqual(list(r), ["reason"])


class Counting(unittest.TestCase):
    def test_pct(self):
        self.assertEqual(bench_engine.pct(1, 3), 33.33)
        self.assertIsNone(bench_engine.pct(0, 0))


if __name__ == "__main__":
    unittest.main()
