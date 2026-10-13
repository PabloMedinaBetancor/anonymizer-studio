"""The cells of the pilot: the same zones for every detector (SL10), the replay of a recorded run, the paired tests and the
rules that choose the thresholds on dev."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from anonymizer.classes.heavy import HeavyFaceModule, HeavyThresholds
from anonymizer.contract import FACE, Detection, Judgement, Verdict
from anonymizer.engine import deep, second_look
from anonymizer.engine.zones import LookItem
from bench import bench_cells, cells_eval
from tests.heavy_fakes import TEST_TH, FakeHeavy

BGR = np.zeros((1000, 1500, 3), np.uint8)


def item(box, verdict, score=0.8) -> LookItem:
    return LookItem(box, FACE, score, verdict, Judgement(verdict, "face.reason.test"), Detection(*box, score, FACE))


def photo_items():
    return [
        item((100, 100, 140, 140), Verdict.CHECK),
        item((300, 100, 340, 140), Verdict.CHECK),
        item((500, 100, 540, 140), Verdict.DROP, 0.7),
        item((700, 100, 740, 140), Verdict.DROP, 0.55),
        item((900, 100, 940, 140), Verdict.BLUR),
    ]


PLACED_A = [
    ((98, 99, 143, 142), 0.95),
    ((298, 98, 342, 142), 0.8),
    ((502, 99, 541, 141), 0.92),
    ((1000, 400, 1040, 450), 0.9),
]
PLACED_B = [((99, 100, 141, 141), 0.6), ((500, 100, 540, 140), 0.99)]
PERSONS = [Detection(950, 300, 1100, 800, 0.9, "person")]


def recorders():
    return {
        name: HeavyFaceModule(heavy_zone=bench_cells.Recorder(FakeHeavy(placed)), heavy_thresholds=TEST_TH)
        for name, placed in (("a", PLACED_A), ("b", PLACED_B))
    }


class Record(unittest.TestCase):
    def test_sl10_every_detector_is_recorded_on_the_same_zones_of_the_same_photo(self):
        record = bench_cells.record_photo(BGR, photo_items(), PERSONS, recorders())
        self.assertEqual(len(record["zones"]), 5)  # two doubtful, two dropped, one head
        for model in ("a", "b"):
            self.assertEqual(len(record["cells"][model]["dets"]), len(record["zones"]), model)
            self.assertEqual(len(record["cells"][model]["seconds"]), len(record["zones"]), model)
        for model in ("a", "b"):
            self.assertEqual(len(record["cells"][model]["piece_s"]), len(record["zones"]), model)
        self.assertEqual(json.loads(json.dumps(record)), record)  # it is plain data

    def test_sl10_a_replay_gives_the_items_of_the_real_run_at_any_cap(self):
        modules = recorders()
        record = bench_cells.record_photo(BGR, photo_items(), PERSONS, modules)
        replayer = cells_eval.Replayer()
        th = TEST_TH
        try:
            for cap in (cells_eval.NOTHING, 3, 1):
                real_module = HeavyFaceModule(heavy_zone=FakeHeavy(PLACED_A), heavy_thresholds=TEST_TH)
                real = second_look.second_stage(BGR, photo_items(), [real_module], PERSONS, max_zones=cap)
                replay = replayer.run(record, record["cells"]["a"]["dets"], th, cap)
                key = lambda r: [(i.box, i.verdict, i.found_by, i.second_score) for i in r.items]  # noqa: E731
                self.assertEqual(key(replay), key(real), cap)
                self.assertEqual((replay.zones_visited, replay.zones_skipped), (real.zones_visited, real.zones_skipped))
        finally:
            replayer.close()

    def test_sl10_the_light_models_alone_are_a_replay_with_no_visit(self):
        record = bench_cells.record_photo(BGR, photo_items(), PERSONS, recorders())
        replayer = cells_eval.Replayer()
        try:
            base = replayer.run(record, None, TEST_TH, 0)
        finally:
            replayer.close()
        self.assertEqual(base.zones_visited, 0)
        self.assertEqual([i.verdict for i in base.items], [i.verdict for i in photo_items()])

    def split_file(self, table):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(table, f)
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_f6_the_split_file_gives_the_dev_part_and_nothing_else(self):
        path = self.split_file({"set-a/x.jpg": "dev", "set-a/y.jpg": "sealed", "set-b/z.jpg": "dev"})
        self.assertEqual(bench_cells.split_names(path, "dev"), {"set-a/x.jpg", "set-b/z.jpg"})

    def test_f6_the_sealed_part_is_refused_everywhere_with_a_pointer_to_the_one_read(self):
        path = self.split_file({"set-a/x.jpg": "dev", "set-a/y.jpg": "sealed"})
        for call in (
            lambda: bench_cells.split_names(path, "sealed"),
            lambda: bench_cells.dev_part(path, "sealed"),
            lambda: bench_cells.photo_keys({"set-a/x.jpg": "dev"}, 0, ("a",), part="sealed"),
        ):
            with self.assertRaises(SystemExit) as raised:
                call()
            self.assertIn("read_sealed", str(raised.exception))

    def test_f6_the_split_v2_file_of_the_truth_is_read_with_its_image_hashes(self):
        path = self.split_file(
            {
                "rule": {"a": "x"},
                "items": {
                    "set-a/x.jpg": {"origin": "a", "part": "dev", "sha256": "ab" * 32},
                    "set-a/y.jpg": {"origin": "a", "part": "sealed", "sha256": "cd" * 32},
                },
            }
        )
        table, hashes = bench_cells.read_split(path)
        self.assertEqual(table, {"set-a/x.jpg": "dev", "set-a/y.jpg": "sealed"})
        self.assertEqual(hashes["set-a/x.jpg"], "ab" * 32)
        self.assertEqual(bench_cells.split_names(path, "dev"), {"set-a/x.jpg"})

    def test_f6_an_image_that_is_not_the_one_the_split_was_fixed_on_is_refused(self):
        import cv2
        import numpy as np

        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder, True)
        os.makedirs(os.path.join(folder, "val"))
        cv2.imwrite(os.path.join(folder, "val", "w.jpg"), np.full((40, 40, 3), 128, np.uint8))
        with open(os.path.join(folder, "val", "w.jpg"), "rb") as f:
            good = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(bench_cells.read_image("wider/w.jpg", {"wider/w.jpg": good}, folder).shape[:2], (40, 40))
        with self.assertRaises(SystemExit) as raised:
            bench_cells.read_image("wider/w.jpg", {"wider/w.jpg": "0" * 64}, folder)
        self.assertIn("SHA-256 differs", str(raised.exception))
        bench_cells.read_image("wider/w.jpg", {}, folder)  # a flat split has no hashes: nothing to check

    def test_f6_speed_and_memory_use_only_the_dev_photos_and_need_the_split(self):
        path = self.split_file(
            {"set-a/b.jpg": "sealed", "set-a/a.jpg": "dev", "set-b/c.jpg": "dev", "set-a/d.jpg": "dev"}
        )
        names = [os.path.basename(f) for f in bench_cells.dev_set_a_files(path)]
        self.assertEqual(names, ["a.jpg", "d.jpg"])
        for command in ("speed", "memory"):
            extra = ["--model", "retinaface-r34"] if command == "speed" else []
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                bench_cells.main([command, *extra])
            self.assertEqual(raised.exception.code, 2)

    def test_f6_a_part_that_is_neither_dev_nor_sealed_is_an_error(self):
        with self.assertRaises(SystemExit) as raised:
            bench_cells.load_split(self.split_file({"set-a/x.jpg": "dev", "set-a/y.jpg": "test"}))
        self.assertIn("unknown part", str(raised.exception))
        with self.assertRaises(SystemExit):
            bench_cells.load_split(self.split_file({}))

    def test_f6_a_photo_the_split_does_not_know_is_an_error_not_a_dev_photo(self):
        names = sorted(bench_cells.load_truth("b")["images"])
        table = {f"set-b/{n}": "dev" for n in names[:-1]}  # the last photo is missing from the split
        with self.assertRaises(SystemExit) as raised:
            bench_cells.photo_keys(table, 0, ("b",))
        self.assertIn("not in the split file", str(raised.exception))
        table[f"set-b/{names[-1]}"] = "sealed"
        keys = bench_cells.photo_keys(table, 0, ("b",))
        self.assertEqual(keys, [f"set-b/{n}" for n in names[:-1]])  # the sealed one is left out, never run

    def test_f6_evaluate_refuses_a_record_that_holds_a_photo_outside_the_dev_part(self):
        record = {
            "models": {"m": {}},
            "target_side_px": 192,
            "look_min": 0.5,
            "machine": {},
            "photos": {"set-a/x.jpg": {}, "set-a/y.jpg": {}},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(record, f)
        self.addCleanup(os.unlink, f.name)
        with self.assertRaises(SystemExit) as raised:
            bench_cells.evaluate([f.name], {"set-a/x.jpg"}, None, None)
        self.assertIn("outside the dev part", str(raised.exception))

    def test_f6_the_command_line_needs_a_split_file(self):
        for command in ("collect", "evaluate"):
            extra = ["--raw", "x.json"] if command == "evaluate" else []
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stderr(io.StringIO()):
                bench_cells.main([command, *extra, "--out", "x.json"])
            self.assertEqual(raised.exception.code, 2)


class Statistics(unittest.TestCase):
    def test_mcnemar_exact_known_values(self):
        self.assertEqual(cells_eval.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(cells_eval.mcnemar_exact(5, 0), 2 / 32)
        self.assertAlmostEqual(cells_eval.mcnemar_exact(10, 2), 2 * 79 / 4096)
        self.assertEqual(cells_eval.mcnemar_exact(3, 3), 1.0)

    def test_paired_counts_what_one_cell_hides_that_the_other_does_not(self):
        base = {"p#0": [False, False, 20], "p#1": [True, True, 30], "q#0": [False, True, 25], "q#1": [False, False, 25]}
        better = {"p#0": [False, True, 20], "p#1": [True, True, 30], "q#0": [False, True, 25], "q#1": [False, True, 25]}
        got = cells_eval.paired(better, base, 1)
        self.assertEqual((got["n"], got["gained"], got["lost"]), (4, 2, 0))
        self.assertEqual(got["gain_per_1000"], 500.0)
        self.assertAlmostEqual(got["mcnemar_p"], 0.5)
        self.assertEqual(cells_eval.paired(base, better, 1)["gain_per_1000"], -500.0)


class DeepCells(unittest.TestCase):
    def test_dm6_the_replay_of_a_deep_sweep_gives_the_items_of_the_real_one_in_both_modes(self):
        for heavy_only in (False, True):
            recorder = bench_cells.Recorder(FakeHeavy(PLACED_A + [((1100, 700, 1140, 760), 0.95)]))
            m = HeavyFaceModule(heavy_zone=recorder, heavy_deep=recorder, heavy_thresholds=TEST_TH)
            real = deep.deep_stage(BGR, photo_items(), [m], heavy_only=heavy_only)
            record = bench_cells.record_photo(BGR, photo_items(), PERSONS, {})
            replayer = cells_eval.Replayer()
            try:
                replay = cells_eval.run_deep(replayer, record, recorder.log, TEST_TH, heavy_only)
            finally:
                replayer.close()
            key = lambda r: sorted((i.box, i.verdict.value, i.found_by) for i in r.items)  # noqa: E731
            self.assertEqual(key(replay), key(real), heavy_only)
            self.assertEqual((replay.calls, replay.added, replay.confirmed), (real.calls, real.added, real.confirmed))

    def test_dm6_deep_must_hide_a_point_more_than_careful_with_p_under_005_or_it_is_the_audit_mode(self):
        self.assertTrue(cells_eval.deep_beats_careful(1.0, 0.049))
        self.assertFalse(cells_eval.deep_beats_careful(0.99, 0.001))
        self.assertFalse(cells_eval.deep_beats_careful(3.0, 0.05))
        self.assertFalse(cells_eval.deep_beats_careful(-1.0, 0.001))


class Markers(unittest.TestCase):
    def report(self):
        def cell(pct, false_per_100, ms=None):
            cell = {
                "faces": 400,
                "blur_pct": pct - 5,
                "hidden_default_pct": pct,
                "hidden_default_ci95_pct": [pct - 2, pct + 2],
                "false_hidden_without_people": {
                    "photos": 50,
                    "events": 1,
                    "photos_affected": 1,
                    "photos_affected_per_100": false_per_100,
                },
                "by_source": {"set-a": {"faces": 400}},
            }
            if ms is not None:
                cell["second_look_ms_per_photo"] = ms
            return cell

        heavy_cell = {
            **cell(90.0, 2.0, 500.0),
            "paired_over_C0": {"blur_or_check": {"gain_per_1000": 25.0}},
            "paired_over_C1": {"blur_or_check": {"gain_per_1000": 20.0}},
            "thresholds": {"h_hi": 0.9, "h_lo": 0.6},
            "second_look": {"zones_skipped": 3},
        }
        return {
            "cells": {"C0": cell(85.0, 0.0), "retinaface-r34": heavy_cell, "yunet-zoomed": cell(86.0, 2.0, 40.0)},
            "recorded_on": {"cpu": "x"},
            "truth_sha256": {"a": "b"},
        }

    def test_bn1_every_recall_figure_has_its_false_hidden_figure_and_n_under_the_same_mode(self):
        markers = cells_eval.v2_markers(self.report())
        for mode in ("fast", "careful", "control"):
            base = f"faces.photo.v2.{mode}"
            for key in ("hidden_default_pct", "hidden_default_n", "hidden_default_ci95_pct", "false_hidden_default"):
                self.assertIn(f"{base}.{key}", markers, f"{base}.{key}")
        self.assertEqual(markers["faces.photo.v2.careful.over_fast"]["gain_per_1000"], 25.0)
        self.assertEqual(markers["faces.photo.v2.careful.over_control"]["gain_per_1000"], 20.0)
        self.assertEqual(markers["faces.photo.v2.careful.zones_skipped"], 3)

    def test_bn1_the_names_are_v2_names_and_the_machine_and_the_truth_travel_with_the_figures(self):
        markers = cells_eval.v2_markers(self.report())
        self.assertTrue(all(k.startswith("faces.photo.v2.") for k in markers))
        self.assertEqual(markers["faces.photo.v2.machine"], {"cpu": "x"})
        self.assertIn("faces.photo.v2.truth_sha256", markers)
        self.assertNotIn("faces.photo.hidden_default_pct", markers)  # v1's name is not reused (M8)

    def test_bn1_deep_is_named_with_its_verdict_when_it_was_scored(self):
        report = self.report()
        report["deep"] = {
            "retinaface-r34": {
                "C4_union": {
                    **self.report()["cells"]["C0"],
                    "ms_per_photo": 6000.0,
                    "paired_over_C2": {"gain_per_1000": 6.1},
                },
                "dm6": {"deep_beats_careful": False},
            }
        }
        markers = cells_eval.v2_markers(report)
        self.assertIs(markers["faces.photo.v2.deep.verdict"]["deep_beats_careful"], False)
        self.assertEqual(markers["faces.photo.v2.deep.second_look_ms_per_photo"], 6000.0)
        self.assertIn("faces.photo.v2.deep.false_hidden_default", markers)


class TargetSide(unittest.TestCase):
    def report(self, gain, extra, ms=100.0):
        base = {"false_hidden_without_people": {"photos_affected_per_100": 0.0}}
        cell = {
            "paired_over_C0": {"blur_or_check": {"gain_per_1000": gain, "mcnemar_p": 0.01}},
            "false_hidden_without_people": {"photos_affected_per_100": extra},
            "second_look_ms_per_photo": ms,
        }
        return {"cells": {"C0": base, "m": cell}}

    def test_the_enlargement_is_the_one_with_the_largest_gain_inside_the_false_hidden_budget(self):
        reports = {192: self.report(10.0, 1.0), 256: self.report(18.0, 2.0), 320: self.report(25.0, 5.0)}
        got = cells_eval.choose_target_side(reports, "m")
        self.assertEqual(got["target_side"], 256)  # 320 gains more but adds 5 false hidden photos per 100
        self.assertEqual([r["allowed"] for r in got["rows"]], [True, True, False])

    def test_a_tie_goes_to_the_smaller_and_faster_side_and_no_allowed_side_gives_none(self):
        self.assertEqual(
            cells_eval.choose_target_side({192: self.report(12.0, 0.0), 256: self.report(12.0, 0.0)}, "m")[
                "target_side"
            ],
            192,
        )
        self.assertIsNone(cells_eval.choose_target_side({256: self.report(30.0, 9.0)}, "m")["target_side"])


class WiderTruth(unittest.TestCase):
    def test_a_wider_face_that_is_invalid_or_under_10_px_high_is_ignored_not_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "subset.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "0--Parade/a.jpg": [
                            {"box": [10, 10, 60, 70], "invalid": False},
                            {"box": [100, 100, 130, 105], "invalid": False},
                            {"box": [200, 200, 260, 280], "invalid": True},
                        ]
                    },
                    f,
                )
            truth = bench_cells.wider_truth(folder)
        self.assertEqual([b["ignore"] for b in truth["0--Parade__a.jpg"]["boxes"]], [False, True, True])
        self.assertEqual(bench_cells.image_path("wider/0--Parade__a.jpg", "/x"), "/x/val/0--Parade__a.jpg")


class Thresholds(unittest.TestCase):
    def truths(self, names):
        return {
            n: {"boxes": [{"box": [0, 0, 100, 100], "ignore": False}, {"box": [500, 0, 600, 100], "ignore": True}]}
            for n in names
        }

    def test_h_hi_is_the_lowest_score_with_precision_98_percent_on_dev(self):
        names = [f"set-a/{i}.jpg" for i in range(40)]
        cell = {}
        for i, n in enumerate(names):
            hit = [5, 5, 95, 95, 0.9 + (i % 5) / 100]  # real, high scores
            junk = [200, 200, 260, 260, 0.6 + (i % 10) / 100]  # matches nothing, scores 0.60-0.69
            maybe = [10, 10, 90, 90, 0.7 + (i % 10) / 100]  # real, scores 0.70-0.79
            cell[n] = [[hit, junk, maybe]]
        got = cells_eval.choose_h_hi(cell, self.truths(names))
        self.assertAlmostEqual(got["h_hi"], 0.7)

    def test_a_detection_on_an_ignore_box_is_neither_hit_nor_false(self):
        names = [f"set-a/{i}.jpg" for i in range(30)]
        cell = {n: [[[505, 5, 595, 95, 0.95], [5, 5, 95, 95, 0.95]]] for n in names}
        got = cells_eval.choose_h_hi(cell, self.truths(names))
        self.assertEqual(got["h_hi"], 0.5)
        self.assertEqual(got["table"][0]["n"], 30)

    def test_the_cap_lets_95_percent_of_the_photos_be_visited_whole(self):
        photos = {f"p{i}": {"zones": [0] * n} for i, n in enumerate([1] * 60 + [12] * 35 + [47, 80, 91, 100, 103])}
        got = cells_eval.choose_max_zones(photos)
        self.assertEqual(got["max_zones"], 15)
        self.assertEqual(got["zones_per_photo"]["max"], 103)

    def _veto_data(self, n, real):
        photos, cell, truths = {}, {}, {}
        for i in range(n):
            name = f"set-a/{i}.jpg"
            photos[name] = {
                "items": [
                    {"box": [10, 10, 50, 50], "score": 0.9, "verdict": "check", "reason": "r", "det": [10, 10, 50, 50]}
                ],
                "zones": [{"box": [10, 10, 50, 50], "why": 1, "item": 0}],
            }
            cell[name] = [[]]  # the heavy model finds nothing in any zone
            truths[name] = {"boxes": [{"box": [10, 10, 50, 50], "ignore": False}] if i < real else []}
        return cell, photos, truths

    def test_the_veto_stays_off_when_three_of_120_doubtful_items_were_real_faces(self):
        got = cells_eval.veto_study(*self._veto_data(120, 3), HeavyThresholds())
        self.assertEqual((got["doubtful_zones_with_nothing_found"], got["real_faces"]), (120, 3))
        self.assertFalse(got["veto_allowed"])

    def test_the_veto_stays_off_with_too_few_zones_even_when_none_was_real(self):
        got = cells_eval.veto_study(*self._veto_data(90, 0), HeavyThresholds())
        self.assertFalse(got["veto_allowed"])  # n < 100
        self.assertFalse(
            cells_eval.veto_study(*self._veto_data(120, 0), HeavyThresholds())["veto_allowed"]
        )  # upper bound 3.1 %

    def test_the_veto_is_allowed_only_with_enough_zones_and_almost_no_real_face(self):
        got = cells_eval.veto_study(*self._veto_data(130, 0), HeavyThresholds())
        self.assertTrue(got["veto_allowed"])


if __name__ == "__main__":
    unittest.main()
