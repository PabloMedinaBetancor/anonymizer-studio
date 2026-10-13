"""P10: the plate ground truth follows its schema, and every image it names is in the sources list."""

from __future__ import annotations

import csv
import glob
import os
import unittest

from bench import plates_truth

SOURCES = os.path.join(os.path.dirname(plates_truth.HERE), "bench", "SOURCES.csv")


def _good() -> dict:
    return {
        "set": "t",
        "licence": "CC0-1.0",
        "images": {
            "a.jpg": {
                "width": 100,
                "height": 50,
                "boxes": [{"box": [10, 10, 40, 20], "kind": "plate", "readable": True, "ignore": False}],
            }
        },
    }


class Schema(unittest.TestCase):
    def test_P10_every_truth_file_follows_the_schema(self):
        files = glob.glob(os.path.join(plates_truth.TRUTH_DIR, "*.json"))
        self.assertTrue(files)
        for path in files:
            name = os.path.splitext(os.path.basename(path))[0]
            self.assertEqual(plates_truth.problems(plates_truth.load(name)), [], name)

    def test_P10_a_valid_document_has_no_problems(self):
        self.assertEqual(plates_truth.problems(_good()), [])

    def test_P10_box_outside_the_image_is_reported(self):
        doc = _good()
        doc["images"]["a.jpg"]["boxes"][0]["box"] = [10, 10, 101, 20]
        self.assertEqual(len(plates_truth.problems(doc)), 1)

    def test_P10_empty_or_inverted_box_is_reported(self):
        for box in ([10, 10, 10, 20], [40, 10, 10, 20], [10, 30, 40, 20]):
            doc = _good()
            doc["images"]["a.jpg"]["boxes"][0]["box"] = box
            self.assertEqual(len(plates_truth.problems(doc)), 1, box)

    def test_P10_missing_or_extra_keys_are_reported(self):
        doc = _good()
        del doc["images"]["a.jpg"]["boxes"][0]["ignore"]
        self.assertTrue(plates_truth.problems(doc))
        doc = _good()
        doc["images"]["a.jpg"]["boxes"][0]["text"] = "ABC123"
        self.assertTrue(plates_truth.problems(doc))
        self.assertTrue(plates_truth.problems({"set": "t"}))

    def test_P10_readable_must_be_a_boolean_not_a_number(self):
        doc = _good()
        doc["images"]["a.jpg"]["boxes"][0]["readable"] = 1
        self.assertTrue(plates_truth.problems(doc))

    def test_P10_unknown_licence_is_reported(self):
        doc = _good()
        doc["licence"] = "all rights reserved"
        self.assertTrue(plates_truth.problems(doc))


class Sources(unittest.TestCase):
    def test_P10_every_image_of_every_truth_file_has_a_sources_row_with_url_author_and_licence(self):
        with open(SOURCES, encoding="utf-8", newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
        for path in glob.glob(os.path.join(plates_truth.TRUTH_DIR, "*.json")):
            doc = plates_truth.load(os.path.splitext(os.path.basename(path))[0])
            for image in doc["images"]:
                row = rows.get(f"plates/{doc['set']}/{image}")
                self.assertIsNotNone(row, f"{doc['set']}/{image} is not in SOURCES.csv")
                for column in ("source_url", "author", "licence", "licence_url", "sha256"):
                    self.assertTrue(row[column], f"{image}: empty {column}")
                self.assertIn(row["licence"], plates_truth.LICENCES)
                if doc["licence"] != "mixed":
                    self.assertEqual(row["licence"], doc["licence"])

    def test_P10_truth_sizes_match_the_images_when_the_bench_data_is_there(self):
        import cv2

        data = os.environ.get("ANONYMIZER_BENCH_DATA", "")
        for path in glob.glob(os.path.join(plates_truth.TRUTH_DIR, "*.json")):
            doc = plates_truth.load(os.path.splitext(os.path.basename(path))[0])
            folder = os.path.join(data, "plates", doc["set"])
            if not os.path.isdir(folder):
                self.skipTest(f"no bench images in {folder}")
            for image, entry in doc["images"].items():
                shape = cv2.imread(os.path.join(folder, image)).shape
                self.assertEqual((entry["width"], entry["height"]), (shape[1], shape[0]), image)


if __name__ == "__main__":
    unittest.main()
