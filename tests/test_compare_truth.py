"""bench/compare_truth.py on toy truths whose answer is known by hand."""

from __future__ import annotations

import unittest

from bench import compare_truth as ct


def box(x0, y0, x1, y1, readable=True, ignore=False):
    return {"box": [x0, y0, x1, y1], "kind": "plate", "readable": readable, "ignore": ignore}


def doc(**images):
    return {
        "set": "t",
        "licence": "CC0-1.0",
        "images": {k: {"width": 500, "height": 400, "boxes": v} for k, v in images.items()},
    }


class Compare(unittest.TestCase):
    def test_identical_truths_agree_completely(self):
        a = doc(**{"a.jpg": [box(10, 10, 60, 30), box(100, 100, 160, 130)], "b.jpg": []})
        report = ct.compare(a, a)
        self.assertEqual((report["agree"], report["only_a"], report["only_b"]), (2, 0, 0))
        self.assertEqual(report["concordance"], 1.0)
        self.assertEqual(report["discrepancies"], [])

    def test_a_box_on_one_side_only_is_a_discrepancy_of_that_side(self):
        a = doc(**{"a.jpg": [box(10, 10, 60, 30), box(300, 300, 360, 330)]})
        b = doc(**{"a.jpg": [box(11, 10, 61, 30)]})
        report = ct.compare(a, b)
        self.assertEqual((report["agree"], report["only_a"], report["only_b"]), (1, 1, 0))
        self.assertAlmostEqual(report["concordance"], 0.5)
        self.assertEqual([r["kind"] for r in report["discrepancies"]], ["only_in_a"])
        self.assertEqual(report["discrepancies"][0]["box_a"], [300, 300, 360, 330])

    def test_boxes_that_barely_overlap_are_not_paired(self):
        a = doc(**{"a.jpg": [box(0, 0, 100, 40)]})
        b = doc(**{"a.jpg": [box(80, 0, 180, 40)]})  # IoU = 20*40 / (2*4000 - 800) = 0.111
        report = ct.compare(a, b)
        self.assertEqual((report["agree"], report["only_a"], report["only_b"]), (0, 1, 1))
        self.assertEqual(report["concordance"], 0.0)

    def test_pairing_is_one_to_one_and_prefers_the_best_overlap(self):
        a = doc(**{"a.jpg": [box(0, 0, 100, 40)]})
        b = doc(**{"a.jpg": [box(40, 0, 140, 40), box(2, 0, 102, 40)]})
        report = ct.compare(a, b)
        self.assertEqual((report["agree"], report["only_b"]), (1, 1))
        self.assertEqual(report["discrepancies"][0]["box_b"], [40, 0, 140, 40])

    def test_ignore_is_not_a_miss_but_a_flag_difference_is_listed(self):
        a = doc(**{"a.jpg": [box(10, 10, 60, 30, ignore=True), box(200, 200, 250, 220, ignore=True)]})
        b = doc(**{"a.jpg": [box(10, 10, 60, 30)]})
        report = ct.compare(a, b)
        self.assertEqual((report["agree"], report["uncertain"], report["only_a"], report["ignored_a"]), (0, 1, 0, 1))
        self.assertEqual([r["kind"] for r in report["discrepancies"]], ["ignore_differs"])
        self.assertIsNone(report["concordance"])

    def test_readable_difference_and_poor_overlap_come_after_the_missing_boxes(self):
        a = doc(**{"a.jpg": [box(10, 10, 60, 30, readable=True), box(100, 100, 200, 140), box(300, 300, 340, 320)]})
        b = doc(**{"a.jpg": [box(10, 10, 60, 30, readable=False), box(130, 100, 230, 140)]})
        report = ct.compare(a, b)
        self.assertEqual(
            [r["kind"] for r in report["discrepancies"]], ["only_in_a", "readable_differs", "poor_overlap"]
        )

    def test_only_the_images_both_name_and_the_listed_ones_are_compared(self):
        a = doc(**{"a.jpg": [box(10, 10, 60, 30)], "b.jpg": [box(10, 10, 60, 30)], "c.jpg": [box(10, 10, 60, 30)]})
        b = doc(**{"a.jpg": [], "b.jpg": [box(10, 10, 60, 30)]})
        self.assertEqual(ct.compare(a, b)["images"], 2)
        report = ct.compare(a, b, only=ct.read_image_list_text(["streets/b.jpg", "# note", ""]))
        self.assertEqual((report["images"], report["agree"], report["only_a"]), (1, 1, 0))

    def test_larger_missing_boxes_come_first(self):
        a = doc(**{"a.jpg": [box(0, 0, 20, 10), box(100, 100, 300, 160)]})
        report = ct.compare(a, doc(**{"a.jpg": []}))
        self.assertEqual([r["area"] for r in report["discrepancies"]], [12000.0, 200.0])

    def test_the_report_names_the_concordance(self):
        text = ct.render(ct.compare(doc(**{"a.jpg": [box(1, 1, 50, 20)]}), doc(**{"a.jpg": []})))
        self.assertIn("concordance: 0.0 %", text)
        self.assertIn("only_in_a", text)


if __name__ == "__main__":
    unittest.main()
