"""Every results file says what it was measured against and how far to trust it (review 1, M1 to M5): the truth it was measured with,
the date and the machine, how many cases stand behind a percentage, and which figures are not to be quoted."""

from __future__ import annotations

import json
import os
import unittest

try:
    from bench import results_meta
except ImportError as e:
    raise unittest.SkipTest(f"bench libraries missing: {e}") from e

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "results")


def load(name: str) -> dict:
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


class Provenance(unittest.TestCase):
    def test_M5_each_result_carries_the_hash_of_the_truth_it_was_measured_with_and_it_still_matches(self):
        for name in ("plates.json", "faces.json", "faces-video.json", "engine.json"):
            data = load(name)
            hashes = data.get("truth_sha256") or data["engine"]["standard_vs_maximum"].get("truth_sha256")
            self.assertTrue(hashes, name)
            for relative, digest in hashes.items():
                self.assertEqual(
                    results_meta.sha256_file(os.path.join(ROOT, relative)),
                    digest,
                    f"{name}: {relative} changed since the figure was measured: run the benchmark again",
                )

    def test_M5_each_result_says_when_and_on_what_machine(self):
        for name in ("plates.json", "faces.json", "faces-video.json", "engine.json"):
            data = load(name)
            self.assertRegex(data["run"]["date"], r"^\d{4}-\d{2}-\d{2}$", name)
            self.assertTrue(data["hardware"]["cpu"], name)


class Trust(unittest.TestCase):
    def test_M3_the_percentages_of_recall_carry_n_and_an_interval(self):
        plates, faces = load("plates.json"), load("faces.json")
        for scale in ("full", "half", "30"):
            base = f"plates.photo.eu.recall_{scale}"
            self.assertEqual(plates[f"{base}_n"], 108)
            low, high = plates[f"{base}_ci95_pct"]
            self.assertLessEqual(low, plates[f"{base}_pct"])
            self.assertGreaterEqual(high, plates[f"{base}_pct"])
        self.assertEqual(plates["plates.photo.streets.recall_blur_n"], plates["plates.photo.streets.plates"])
        self.assertGreater(
            plates["plates.photo.streets.recall_blur_ci95_pct"][1]
            - plates["plates.photo.streets.recall_blur_ci95_pct"][0],
            30,
        )
        photo = faces["faces"]["photo"]
        for key in ("recall_blur", "recall_blur_or_check", "recall_blur_ellipse"):
            self.assertEqual(photo[key]["n"], photo["faces_total"])
            low, high = photo[key]["ci95_pct"]
            self.assertLessEqual(low, photo[f"{key}_pct"])
            self.assertGreaterEqual(high, photo[f"{key}_pct"])
        self.assertIn("optimistic", faces["caveat"])

    def test_M1_what_is_not_to_be_quoted_says_so_and_why(self):
        video, engine = load("faces-video.json"), load("engine.json")
        for prefix in ("faces.video", "faces_video"):
            self.assertIs(video["_publication"][prefix]["publishable"], False)
            self.assertIn("22 faces", video["_publication"][prefix]["why"])
        coverage = engine["_publication"]["engine.standard_vs_maximum.coverage_pct"]
        self.assertIs(coverage["publishable"], False)
        self.assertIn("noise", coverage["why"])
        self.assertEqual(video["faces"]["video"]["frame_recall_standard"]["n"], 22)

    def test_M2_the_synthetic_gap_figures_say_they_are_with_perfect_recovery(self):
        gaps = load("engine.json")["engine"]["gaps"]["synthetic"]
        self.assertIn("perfect recovery", gaps["oracle"])
        self.assertGreater(gaps["motion_pct"], gaps["hold_pct"])

    def test_intervals_are_wilson_s(self):
        self.assertEqual(results_meta.wilson(9, 22), [23.3, 61.3])
        self.assertEqual(results_meta.wilson(0, 0), [None, None])
        self.assertEqual(results_meta.interval(7, 8)["ci95_pct"], results_meta.wilson(7, 8))


class Review2(unittest.TestCase):
    def test_N5_a_rate_per_100_photos_carries_its_events_its_n_and_an_interval_and_the_rule_of_three_when_there_were_none(
        self,
    ):
        zero = results_meta.rate_per_100(0, 0, 71)
        self.assertEqual((zero["per_100"], zero["n"], zero["rule_of_three_upper_pct"]), (0.0, 71, 4.2))
        self.assertEqual(zero["ci95_photos_pct"], [0.0, 5.1])
        some = results_meta.rate_per_100(3, 3, 60)
        self.assertEqual((some["per_100"], some["events"], some["photos_affected"]), (5.0, 3, 3))
        self.assertNotIn("rule_of_three_upper_pct", some)

    def test_N5_the_false_rates_of_every_results_file_say_how_many_events_on_how_many_photos(self):
        plates, faces = load("plates.json"), load("faces.json")
        for key in ("plates.photo.false_blur", "plates.photo.false_check"):
            rate = plates[key]
            self.assertEqual(rate["n"], plates["plates.photo.streets.photos"])
            self.assertEqual(round(rate["per_100"], 2), plates[key + "_per_100"])
        self.assertEqual(plates["plates.photo.false_blur"]["events"], 3)
        rate = faces["faces"]["photo"]["false_blur"]
        self.assertEqual(rate["n"], faces["faces"]["photo"]["false_blur_photos_without_people"])
        self.assertEqual(rate["per_100"], faces["faces"]["photo"]["false_blur_per_100"])
        if rate["photos_affected"] == 0:
            self.assertIn("rule_of_three_upper_pct", rate)

    def test_N5_the_street_plate_recall_is_not_publishable(self):
        marks = load("plates.json")["_publication"]
        for key in (
            "plates.photo.streets.recall_blur_pct",
            "plates.photo.streets.recall_blur_n",
            "plates.photo.streets.recall_blur_ci95_pct",
        ):
            self.assertIs(marks[key]["publishable"], False)
            self.assertIn("minority", marks[key]["why"])

    def test_N5_the_speed_is_also_measured_on_a_clip_with_something_to_blur_and_names_its_resolution(self):
        engine = load("engine.json")["engine"]
        self.assertEqual(engine["speed"]["resolution"], "1920x1080")
        busy = engine["speed_busy"]
        self.assertEqual(busy["resolution"], "1280x720")
        self.assertIn("s_per_video_s_720p", busy)
        self.assertGreater(busy["blurred_frames_pct"], 50.0)
        self.assertIn("blurred_frames_pct", engine["speed"])  # the 1080p clip says how little it has to blur
        self.assertGreater(engine["rerender_busy"]["blurred_frames_pct"], 50.0)

    def test_N5_the_photo_time_is_one_figure_for_faces_and_plates_together_on_one_set(self):
        photo = load("engine.json")["engine"]["photo_speed"]
        self.assertEqual(photo["photos"], load("plates.json")["plates.photo.streets.photos"])
        self.assertIn("median", photo["resolution_px"])
        self.assertGreater(photo["analyze_and_write_ms_per_photo"], photo["analyze_ms_per_photo"])

    def test_R3_5_the_rerender_of_the_clip_with_nothing_to_blur_is_not_publishable_and_the_busy_one_is(self):
        data = load("engine.json")
        marks = data["_publication"]
        self.assertIs(marks["engine.rerender"]["publishable"], False)
        self.assertIn("rerender_busy", marks["engine.rerender"]["why"])
        self.assertFalse([k for k in marks if k.startswith("engine.rerender_busy")])
        self.assertEqual(data["engine"]["speed"]["items"], 0)  # the reason the first is refused

    def test_the_stop_latency_is_measured_and_says_whether_the_two_seconds_were_met(self):
        stop = load("engine.json")["engine"]["stop"]
        self.assertEqual(stop["stops"], 20)
        self.assertEqual(stop["requirement_s"], 2.0)
        self.assertEqual(stop["met"], stop["latency_max_s"] <= 2.0)
        self.assertEqual(stop["latency_s"], {"median": stop["latency_median_s"], "max": stop["latency_max_s"]})


class PrivacyFirstFigures(unittest.TestCase):
    """PF6: the gain is never printed without its cost, and the v1 names are still there (M8)."""

    def test_pf6_the_faces_result_prints_what_is_hidden_by_default_next_to_what_is_falsely_hidden(self):
        photo = load("faces.json")["faces"]["photo"]
        hidden, ellipse = photo["hidden_default"], photo["hidden_default_ellipse"]
        self.assertEqual(hidden["n"], photo["faces_total"])
        self.assertEqual(round(100.0 * hidden["hits"] / hidden["n"], 2), photo["hidden_default_pct"])
        self.assertLessEqual(hidden["ci95_pct"][0], photo["hidden_default_pct"])
        self.assertGreaterEqual(hidden["ci95_pct"][1], photo["hidden_default_pct"])
        self.assertEqual(ellipse["n"], hidden["n"])
        cost = photo["false_hidden_default"]
        self.assertEqual(cost["n"], photo["false_blur_photos_without_people"])
        for key in ("per_100", "events", "photos_affected", "ci95_photos_pct"):
            self.assertIn(key, cost)
        self.assertIn("false_hidden_default_in_set_a", photo)

    def test_pf6_the_ellipse_figure_is_never_above_the_rectangle_figure_and_both_are_printed(self):
        photo = load("faces.json")["faces"]["photo"]
        self.assertLessEqual(photo["hidden_default_ellipse_pct"], photo["hidden_default_pct"])
        self.assertLessEqual(photo["recall_blur_ellipse_pct"], photo["hidden_default_ellipse_pct"])

    def test_pf6_hidden_by_default_is_at_least_what_is_blurred_and_a_doubtful_zone_never_cancels_a_blur(self):
        photo = load("faces.json")["faces"]["photo"]
        self.assertGreaterEqual(photo["hidden_default_pct"], photo["recall_blur_pct"])
        self.assertEqual(
            photo["hidden_default_pct"], photo["recall_blur_or_check_pct"]
        )  # the same measurement under the new default
        self.assertGreaterEqual(photo["false_hidden_default"]["events"], photo["false_blur"]["events"])
        for size in photo["by_size"].values():
            self.assertGreaterEqual(size["recall_blur_or_check_pct"], size["recall_blur_pct"])

    def test_pf6_the_plates_result_prints_the_default_at_three_scales_and_the_false_hidden_cost(self):
        plates = load("plates.json")
        for scale in ("full", "half", "30"):
            base = f"plates.photo.hidden_default_{scale}"
            self.assertEqual(plates[f"{base}_n"], plates[f"plates.photo.eu.recall_{scale}_n"])
            self.assertGreaterEqual(plates[f"{base}_pct"], plates[f"plates.photo.eu.recall_{scale}_pct"])
            low, high = plates[f"{base}_ci95_pct"]
            self.assertLessEqual(low, plates[f"{base}_pct"])
            self.assertGreaterEqual(high, plates[f"{base}_pct"])
        cost = plates["plates.photo.false_hidden_default"]
        self.assertEqual(cost["n"], plates["plates.photo.streets.photos"])
        self.assertEqual(
            cost["events"], plates["plates.photo.false_blur"]["events"] + plates["plates.photo.false_check"]["events"]
        )
        self.assertGreaterEqual(
            cost["photos_affected"],
            max(
                plates["plates.photo.false_blur"]["photos_affected"],
                plates["plates.photo.false_check"]["photos_affected"],
            ),
        )

    def test_pf6_the_v1_names_stay_so_that_unsure_keep_remains_reproducible(self):
        plates, photo = load("plates.json"), load("faces.json")["faces"]["photo"]
        for key in ("recall_blur_pct", "recall_blur_or_check_pct", "recall_blur_ellipse_pct", "false_blur"):
            self.assertIn(key, photo)
        for key in ("plates.photo.eu.recall_full_pct", "plates.photo.false_blur", "plates.photo.false_check"):
            self.assertIn(key, plates)


if __name__ == "__main__":
    unittest.main()
