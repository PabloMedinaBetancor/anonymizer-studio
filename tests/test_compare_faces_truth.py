"""bench/compare_faces_truth.py and bench/pick_faces_verify_sample.py on toy data whose answer is known by hand."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from bench import compare_faces_truth as cft
from bench import pick_faces_verify_sample as pick


def box(x0, y0, x1, y1, ignore=False):
    return {"box": [x0, y0, x1, y1], "kind": "face", "ignore": ignore, "note": ""}


def doc(**images):
    return {"images": {k: {"width": 500, "height": 400, "boxes": v} for k, v in images.items()}}


class Compare(unittest.TestCase):
    def test_identical_truths_agree_completely(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22), box(100, 100, 160, 170)], "b.jpg": []})
        r = cft.compare(a, a)
        self.assertEqual((r["agree"], r["only_a"], r["only_v"]), (2, 0, 0))
        self.assertEqual((r["author_precision"], r["author_coverage"]), (1.0, 1.0))
        self.assertEqual(r["discrepancies"], [])

    def test_a_face_only_the_verifier_sees_lowers_coverage_not_precision(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22)]})
        v = doc(**{"a.jpg": [box(10, 10, 20, 22), box(300, 300, 360, 370)]})
        r = cft.compare(a, v)
        self.assertEqual((r["agree"], r["only_a"], r["only_v"]), (1, 0, 1))
        self.assertEqual((r["author_precision"], r["author_coverage"]), (1.0, 0.5))
        self.assertEqual(r["discrepancies"][0]["kind"], "only_in_verifier")
        self.assertEqual(r["by_size"][">48"]["only_v"], 1)

    def test_a_face_only_the_author_labels_lowers_precision(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22), box(300, 300, 330, 340)]})
        v = doc(**{"a.jpg": [box(10, 10, 20, 22)]})
        r = cft.compare(a, v)
        self.assertEqual((r["author_precision"], r["author_coverage"]), (0.5, 1.0))
        self.assertEqual(r["by_size"]["24-48"]["only_a"], 1)

    def test_a_small_shift_pairs_at_the_default_iou_but_not_at_a_strict_one(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22)]})
        v = doc(**{"a.jpg": [box(12, 10, 22, 22)]})  # IoU = 80 / 160 = 0.5 less a hair: 8*12 / (120 + 120 - 96) = 0.667
        self.assertEqual(cft.compare(a, v, 0.3)["agree"], 1)
        self.assertEqual(cft.compare(a, v, 0.7)["agree"], 0)

    def test_ignore_is_uncertain_not_a_miss(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22, ignore=True), box(200, 200, 215, 215, ignore=True)]})
        v = doc(**{"a.jpg": [box(10, 10, 20, 22)]})
        r = cft.compare(a, v)
        self.assertEqual((r["agree"], r["uncertain"], r["only_a"], r["only_v"], r["ignored_a"]), (0, 1, 0, 0, 1))
        self.assertEqual([d["kind"] for d in r["discrepancies"]], ["ignore_differs"])
        self.assertIsNone(r["author_precision"])

    def test_only_the_images_in_the_list_and_in_both_files_count(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22)], "b.jpg": [box(10, 10, 20, 22)], "c.jpg": []})
        v = doc(**{"a.jpg": [box(10, 10, 20, 22)], "b.jpg": []})
        self.assertEqual(cft.compare(a, v)["images"], 2)
        self.assertEqual(cft.compare(a, v, only={"a.jpg"})["images"], 1)

    def test_misses_the_verifier_confirms_are_counted(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22), box(100, 100, 110, 112)]})
        v = doc(**{"a.jpg": [box(10, 10, 20, 22)]})
        misses = [
            {"image": "set-a/a.jpg", "box": [10, 10, 20, 22]},
            {"image": "set-a/a.jpg", "box": [100, 100, 110, 112]},
        ]
        r = cft.compare(a, v, misses=misses)
        self.assertEqual((r["misses_in_sample"], r["misses_confirmed_by_verifier"]), (2, 1))

    def test_a_malformed_truth_is_refused(self):
        bad = doc(**{"a.jpg": [{"box": [5, 5, 3, 9], "kind": "face", "ignore": False}]})
        self.assertTrue(cft.problems(bad))
        self.assertEqual(cft.problems(doc(**{"a.jpg": [box(1, 1, 5, 5)]})), [])

    def test_render_mentions_the_sizes_and_the_discrepancies(self):
        a = doc(**{"a.jpg": [box(10, 10, 20, 22)]})
        text = cft.render(cft.compare(a, doc(**{"a.jpg": []})))
        self.assertIn("only_in_author", text)
        self.assertIn("<24 px", text)


class Sample(unittest.TestCase):
    def test_the_sample_holds_every_flagged_photo_and_is_repeatable(self):
        lines = pick.pick(60, 8, 12, 20261003)
        with open(pick.DETAILS, encoding="utf-8") as f:
            details = json.load(f)
        flagged = {d["image"] for d in details["misses"] + details["false_blurs"]}
        self.assertTrue(flagged <= set(lines))
        self.assertEqual(lines, sorted(lines))
        self.assertEqual(lines, pick.pick(60, 8, 12, 20261003))
        self.assertGreaterEqual(len([x for x in lines if x.startswith("set-b/")]), 12)
        self.assertNotEqual(lines, pick.pick(60, 8, 12, 1))

    def test_the_cli_writes_the_list(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "list.txt")
            pick.main(["--out", out])
            with open(out, encoding="utf-8") as f:
                self.assertTrue(f.read().startswith("set-"))


if __name__ == "__main__":
    unittest.main()
