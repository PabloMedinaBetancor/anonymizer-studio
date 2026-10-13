"""The video face truth (bench/truth/faces/video.json) and the coverage measure of bench/bench_faces_video.py."""

from __future__ import annotations

import os
import unittest

from bench import bench_faces_video as bfv


class Truth(unittest.TestCase):
    def setUp(self):
        self.truth = bfv.load_truth()

    def test_F4_both_clips_have_their_labelled_frames_and_well_formed_boxes(self):
        self.assertEqual(
            {c: len(v["frames"]) for c, v in self.truth["clips"].items()}, {"preciados": 28, "hongkong": 24}
        )
        for clip, spec in self.truth["clips"].items():
            for frame, entry in spec["frames"].items():
                self.assertTrue(frame.isdigit())
                for b in entry["boxes"]:
                    x0, y0, x1, y1 = b["box"]
                    self.assertTrue(0 <= x0 < x1 <= spec["width"] and 0 <= y0 < y1 <= spec["height"], (clip, frame, b))
                    self.assertIsInstance(b["ignore"], bool)

    def test_F4_the_hong_kong_corridor_has_no_face_and_preciados_has_some(self):
        counted = {
            c: sum(1 for e in v["frames"].values() for b in e["boxes"] if not b["ignore"])
            for c, v in self.truth["clips"].items()
        }
        self.assertEqual(counted["hongkong"], 0)
        self.assertGreater(counted["preciados"], 15)

    def test_F4_the_clips_are_listed_with_their_licence(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "bench", "SOURCES.csv"), encoding="utf-8") as f:
            text = f.read()
        for clip in self.truth["clips"].values():
            self.assertIn("faces/video/" + clip["file"], text)
        self.assertIn("CC-BY-SA", text.split("faces/video/preciados.ogv")[1].split("\n")[0])


class Coverage(unittest.TestCase):
    def test_a_zone_that_contains_the_box_covers_all_of_it(self):
        self.assertEqual(bfv.covered_share([10, 10, 30, 40], [([0, 0, 50, 60], "rect")]), 1.0)

    def test_nothing_covers_nothing_and_a_far_zone_does_not_count(self):
        self.assertEqual(bfv.covered_share([10, 10, 30, 40], []), 0.0)
        self.assertEqual(bfv.covered_share([10, 10, 30, 40], [([100, 100, 150, 150], "ellipse")]), 0.0)

    def test_half_of_the_box_covered_is_half(self):
        self.assertAlmostEqual(bfv.covered_share([0, 0, 20, 20], [([0, 0, 10, 20], "rect")]), 0.5, places=2)

    def test_an_ellipse_covers_less_than_its_rectangle(self):
        share = bfv.covered_share([0, 0, 40, 40], [([0, 0, 40, 40], "ellipse")])
        self.assertGreater(share, 0.7)
        self.assertLess(share, 0.85)

    def test_the_wilson_interval_of_9_out_of_22_is_wide(self):
        low, high = bfv.wilson(9, 22)
        self.assertAlmostEqual(low, 23.3, delta=0.5)
        self.assertAlmostEqual(high, 61.3, delta=0.5)
        self.assertEqual(bfv.wilson(0, 0), [None, None])

    def test_size_classes(self):
        self.assertEqual([bfv.size_class([0, 0, w, 10]) for w in (10, 30, 60)], ["<24", "24-48", ">48"])


class Summary(unittest.TestCase):
    def test_the_quoted_figure_is_the_hits_over_the_labelled_faces_of_all_clips(self):
        runs = {
            "standard": {
                "a": {"faces": 20, "faces_hit_blur": 5, "recall_blur_or_check_pct": 30.0},
                "b": {"faces": 0, "faces_hit_blur": 0, "recall_blur_or_check_pct": None},
            }
        }
        video = bfv.summary(runs)["faces"]["video"]
        self.assertEqual(video["frame_recall_standard_pct"], 25.0)
        self.assertEqual(video["frame_recall_standard_blur_or_check_pct"], 30.0)
        self.assertEqual(video["labelled_faces_standard"], 20)


if __name__ == "__main__":
    unittest.main()
