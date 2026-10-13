"""M3-M5: the registered measurements of the sealed read. The statistics helpers and the plate cell are tested on small made-up inputs; the plate
measurement itself refuses a sealed key unless the read hands it over (`allow_sealed`), and is registered under its claim's name."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

try:
    from bench import read_sealed as rs
    from bench import sealed_measure as sm
except ImportError as e:
    raise unittest.SkipTest(f"bench libraries missing: {e}") from e


class Cover(unittest.TestCase):
    def test_a_plate_is_covered_by_the_union_of_the_boxes(self):
        plate = (10.0, 10.0, 30.0, 20.0)
        self.assertAlmostEqual(sm.covered_share(plate, [(10, 10, 20, 20)]), 0.5, delta=0.06)
        self.assertGreaterEqual(sm.covered_share(plate, [(10, 10, 20, 20), (19, 10, 31, 20)]), 0.95)
        self.assertEqual(sm.covered_share(plate, [(40, 40, 50, 50)]), 0.0)

    def test_hidden_needs_95_percent_and_the_rule_is_not_moved(self):
        self.assertEqual(sm.COVERED, 0.95)  # M9
        plate = (0.0, 0.0, 100.0, 10.0)
        self.assertTrue(sm.plate_hidden(plate, [(0, 0, 100, 10)]))
        self.assertFalse(sm.plate_hidden(plate, [(0, 0, 90, 10)]))

    def test_the_margin_grows_the_box_and_is_clipped_to_the_picture(self):
        self.assertEqual(sm.with_margin((10, 10, 20, 20), (0.1, 0.2), 100, 100), (9.0, 8.0, 21.0, 22.0))
        self.assertEqual(sm.with_margin((0, 0, 20, 20), (0.5, 0.5), 25, 25), (0.0, 0.0, 25.0, 25.0))

    def test_the_union_share_counts_overlaps_once(self):
        both = sm.union_share([(0, 0, 40, 40), (20, 20, 60, 60)], 100, 100)
        one = sm.union_share([(0, 0, 40, 40)], 100, 100)
        self.assertGreater(both, one)
        self.assertLess(both, 2 * one)


class Intervals(unittest.TestCase):
    def test_the_bootstrap_resamples_photos_not_plates(self):
        per_photo = {"a": [4, 4], "b": [0, 4], "c": [4, 4], "d": [0, 4]}
        low, high = sm.bootstrap_by_photo(per_photo)
        self.assertLess(low, 50.0)
        self.assertGreater(high, 50.0)
        self.assertEqual(sm.bootstrap_by_photo({}), [None, None])

    def test_the_pairing_is_the_exact_mcnemar_of_the_discordant_plates(self):
        shipped = {f"p#{i}": [i < 9, i < 9] for i in range(10)}
        p0 = {f"p#{i}": [i < 4, i < 4] for i in range(10)}
        got = sm.paired(shipped, p0, 1)
        self.assertEqual((got["gained"], got["lost"]), (5, 0))
        self.assertLess(got["mcnemar_p"], 0.1)


class Cell(unittest.TestCase):
    def rows(self):
        def plate(shipped, text, p0, readable=True):
            return {"box": [0, 0, 10, 5], "readable": readable, "shipped": shipped, "text_alone": text, "p0": p0}

        return [
            {
                "photo": "a.jpg",
                "plates": [plate(True, False, False), plate(True, True, True)],
                "bands": 1,
                "band_share": 0.04,
                "text_share": 0.02,
                "both_share": 0.06,
                "vehicles_searched": 1,
            },
            {
                "photo": "b.jpg",
                "plates": [plate(False, False, True)],
                "bands": 0,
                "band_share": 0.0,
                "text_share": 0.0,
                "both_share": 0.0,
                "vehicles_searched": 0,
            },
        ]

    def test_the_cell_counts_the_cells_and_prints_the_cost_beside_them(self):
        cell = sm._plates_scale(self.rows())
        self.assertEqual(cell["plates"], 3)
        self.assertEqual({k: v["k"] for k, v in cell["hidden"].items()}, {"shipped": 2, "text_alone": 1, "p0": 2})
        self.assertEqual(cell["shipped_vs_text_alone"]["gained"], 1)
        self.assertEqual((cell["shipped_vs_p0"]["gained"], cell["shipped_vs_p0"]["lost"]), (1, 1))
        self.assertEqual(cell["cost"]["photos_with_a_band"], 1)
        self.assertEqual(cell["cost"]["photos_without_a_searchable_vehicle"], 1)
        self.assertEqual(
            cell["cost"]["bands_on_photos_without_a_searchable_vehicle"], 0
        )  # the claim "adds nothing where there is no vehicle"
        self.assertAlmostEqual(cell["cost"]["band_share_of_picture_mean_banded_photos"], 0.04)
        self.assertEqual(json.loads(json.dumps(cell))["plates"], 3)  # JSON-able


class FacesAndReference(unittest.TestCase):
    def test_the_reference_blurs_each_detection_scaled_by_deface_s_mask_scale(self):
        zones = sm.reference_zones([[100, 100, 200, 200, 0.9], [10, 10, 20, 20, 0.1]], 0.5, 1000, 1000)
        self.assertEqual(len(zones), 1)  # the weak detection is below the cut
        self.assertEqual(zones[0].margin_box, (85.0, 85.0, 215.0, 215.0))  # 1.3 times, about the centre
        edge = sm.reference_zones([[0, 0, 100, 100, 0.9]], 0.5, 120, 120)[0].margin_box
        self.assertEqual((edge[0], edge[1], edge[2], edge[3]), (0.0, 0.0, 115.0, 115.0))  # clipped to the picture

    def test_the_cut_is_the_lowest_of_the_grid_within_the_false_hidden_budget_and_chosen_on_dev_photos(self):
        truth = {"boxes": [{"box": [100, 100, 200, 200]}]}
        scan = {
            "set-a/with.jpg": {"width": 400, "height": 400, "detections": [[100, 100, 200, 200, 0.95]], "truth": truth},
            "set-b/empty.jpg": {
                "width": 400,
                "height": 400,
                "detections": [[300, 300, 360, 360, 0.30]],
                "truth": {"boxes": []},
            },
        }
        chosen = sm.choose_reference_threshold(scan, ["set-b/empty.jpg"], budget=3.0)
        self.assertGreater(
            chosen["threshold"], 0.30
        )  # the 0.30 detection on a photo without people is a false hidden: the cut moves above it
        self.assertEqual(chosen["grid"][0]["threshold"], 0.2)
        self.assertEqual(chosen["grid"][0]["false_per_100"], 100.0)
        self.assertEqual(chosen["mask_scale"], 1.3)

    def test_the_frozen_cut_is_in_the_truth_folder_and_comes_from_the_grid(self):
        data = json.load(open(sm.REFERENCE_THRESHOLD_FILE, encoding="utf-8"))
        self.assertIn(data["threshold"], sm.REFERENCE_GRID)
        self.assertEqual(data["rule"][:21], "the lowest score cut ")

    def test_the_face_cell_prints_its_cost_and_the_false_hidden_beside_the_recall(self):
        faces = {"a.jpg#0": [True, True, True], "a.jpg#1": [False, False, True], "b.jpg#0": [True, False, True]}
        cell = sm._face_cell_summary(faces, {"a.jpg": 0, "e.jpg": 2}, ["e.jpg", "f.jpg"], [1.0, 3.0])
        self.assertEqual((cell["faces"], cell["hidden_by_default"]), (3, 2))
        self.assertEqual(cell["false_hidden_photos_without_people"]["photos_affected"], 1)
        self.assertEqual(cell["false_hidden_photos_without_people"]["per_100_photos"], 50.0)
        self.assertEqual(cell["seconds_per_photo_median"], 2.0)

    def test_the_face_and_video_claims_are_registered_and_a_dry_run_refuses_a_sealed_key(self):
        self.assertTrue({"faces_v2", "video_v2", "plates_v2"} <= set(rs.MEASUREMENTS))
        with tempfile.TemporaryDirectory() as tmp:
            split = os.path.join(tmp, "split.json")
            with open(split, "w", encoding="utf-8") as f:
                json.dump({"items": {"blind/x.jpg": {"part": "sealed"}, "video/c/1": {"part": "sealed"}}}, f)
            with self.assertRaises(rs.SealedError):
                sm.faces_v2(["blind/x.jpg"], split_path=split)
            with self.assertRaises(rs.SealedError):
                sm.video_v2(["video/c/1"], split_path=split)

    def test_the_speed_claim_needs_the_median_ratio_and_its_upper_bound(self):
        cells = {
            "fast": {"seconds_by_photo": [1.0, 1.0, 1.0, 1.0]},
            "careful": {"seconds_by_photo": [2.0, 2.5, 3.0, 2.8]},
        }
        got = sm._speed_ratio(cells)
        self.assertTrue(got["claim_3x_holds"])
        slow = {"fast": {"seconds_by_photo": [1.0] * 4}, "careful": {"seconds_by_photo": [5.0] * 4}}
        self.assertFalse(sm._speed_ratio(slow)["claim_3x_holds"])
        self.assertEqual(sm._speed_ratio({"fast": {"seconds_by_photo": [1.0]}}), {})

    def test_a_blind_key_reads_its_truth_from_the_blind_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blind.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"images": {"blind-001.jpg": {"boxes": [{"box": [1, 2, 3, 4], "ignore": False}]}}}, f)
            old, sm.BLIND_FACES_TRUTH = sm.BLIND_FACES_TRUTH, path
            try:
                self.assertEqual(sm._face_truth("blind/blind-001.jpg", "")["boxes"][0]["box"], [1, 2, 3, 4])
            finally:
                sm.BLIND_FACES_TRUTH = old

    def test_a_failing_cell_is_recorded_and_does_not_take_the_others_with_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = os.path.join(tmp, "split.json")
            with open(split, "w", encoding="utf-8") as f:
                json.dump({"items": {"set-a/x.jpg": {"part": "dev"}}}, f)
            result = sm.faces_v2([], modes=("not-a-cell", "also-not-a-cell"), split_path=split)
        self.assertEqual(sorted(result["errors"]), ["also-not-a-cell", "not-a-cell"])
        self.assertIn("error", result["cells"]["not-a-cell"])
        self.assertEqual(result["speed"], {})  # no timings, no ratio, and no crash

    def test_a_failing_clip_is_recorded_and_the_video_claim_is_made_on_the_rest(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            split = os.path.join(tmp, "split.json")
            with open(split, "w", encoding="utf-8") as f:
                json.dump({"items": {"video/c/1": {"part": "dev", "sha256": "x"}}}, f)
            truth = os.path.join(tmp, "truth.json")
            with open(truth, "w", encoding="utf-8") as f:
                json.dump({"clips": {"c": {"file": "c.webm", "frames": {"1": {"boxes": [{"box": [0, 0, 5, 5]}]}}}}}, f)
            with (
                mock.patch.object(sm, "check_clip_hash"),
                mock.patch("anonymizer.Anonymizer", side_effect=RuntimeError("boom")),
            ):
                result = sm.video_v2(["video/c/1"], split_path=split, truth_path=truth, clips_dir=tmp)
        self.assertEqual(sorted(result["errors"]), ["c/careful", "c/fast"])
        self.assertIn("error", result["per_clip"]["c"]["modes"]["fast"])
        self.assertEqual(result["cells"]["fast"]["faces"], 0)  # nothing scored, nothing invented


class Registration(unittest.TestCase):
    def test_the_plate_claim_is_registered_under_its_name(self):
        self.assertIn("plates_v2", rs.MEASUREMENTS)

    def test_a_dry_run_refuses_a_sealed_key_and_an_unknown_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = os.path.join(tmp, "split.json")
            with open(split, "w", encoding="utf-8") as f:
                json.dump({"items": {"plates/dev.jpg": {"part": "dev"}, "plates/sealed.jpg": {"part": "sealed"}}}, f)
            with self.assertRaises(rs.SealedError):
                sm.plates_v2(["plates/sealed.jpg"], split_path=split)
            with self.assertRaises(rs.SealedError):
                sm.plates_v2(["plates/unknown.jpg"], split_path=split)


class Hashes(unittest.TestCase):
    def test_a_clip_that_moved_or_has_no_recorded_hash_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.mp4")
            with open(path, "wb") as f:
                f.write(b"clip")
            good = rs.sha256_file(path)
            sm.check_clip_hash(path, {good})
            sm.check_clip_hash(path, {good, None})
            for bad in ({"0" * 64}, {None}, {good, "0" * 64}):
                with self.assertRaises(SystemExit):
                    sm.check_clip_hash(path, set(bad))

    def test_the_reference_model_is_checked_against_the_sources_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "centerface.onnx")
            with open(path, "wb") as f:
                f.write(b"not the model")
            with self.assertRaises(SystemExit):
                sm.CenterFace(path)


if __name__ == "__main__":
    unittest.main()
