"""The face ground truth and its sources (F9): structure, licences, no private material."""

from __future__ import annotations

import csv
import json
import os
import re
import unittest

from tests import docs_support
from tests.faces_support import missing

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRUTH = os.path.join(ROOT, "bench", "truth", "faces")
SOURCES = os.path.join(ROOT, "bench", "SOURCES.csv")
LICENCE = re.compile(r"^(CC0|CC BY(-SA)? \d\.\d)")
PRIVATE = (
    docs_support.load_internal_names()
)  # the private list is named by ANON_DOCS_PRIVATE_NAMES; without it this scan is skipped


def load(tag: str) -> dict:
    path = os.path.join(TRUTH, f"set-{tag}.json")
    if not os.path.isfile(path):
        missing(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class FaceTruth(unittest.TestCase):
    def test_F9_both_sets_follow_the_documented_schema(self):
        for tag in ("a", "b"):
            data = load(tag)
            self.assertEqual(data["set"], tag)
            self.assertTrue(data["licence"])
            self.assertTrue(data["images"], tag)
            for name, entry in data["images"].items():
                width, height = entry["width"], entry["height"]
                self.assertTrue(name.endswith(".jpg") and "/" not in name, name)
                self.assertGreater(width, 0)
                self.assertGreater(height, 0)
                for item in entry["boxes"]:
                    self.assertEqual(set(item), {"box", "kind", "ignore", "note"}, (name, item))
                    x0, y0, x1, y1 = item["box"]
                    self.assertTrue(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, (name, item["box"]))
                    self.assertEqual(item["kind"], "face")
                    self.assertIsInstance(item["ignore"], bool)
                    if item["ignore"]:
                        self.assertTrue(item["note"], f"{name}: an ignored box needs a note")

    def test_F9_set_a_has_faces_and_set_b_has_none_that_count(self):
        counted_a = sum(not b["ignore"] for e in load("a")["images"].values() for b in e["boxes"])
        counted_b = sum(not b["ignore"] for e in load("b")["images"].values() for b in e["boxes"])
        self.assertGreater(counted_a, 100)
        self.assertEqual(counted_b, 0)
        self.assertTrue(all(any(not b["ignore"] for b in e["boxes"]) for e in load("a")["images"].values()))

    def test_F9_every_labelled_file_has_a_source_row_with_a_free_licence(self):
        with open(SOURCES, encoding="utf-8", newline="") as f:
            rows = {r["file"]: r for r in csv.DictReader(f)}
        for tag in ("a", "b"):
            for name in load(tag)["images"]:
                row = rows.get(f"faces/set-{tag}/{name}")
                self.assertIsNotNone(row, f"no SOURCES.csv row for {name}")
                self.assertRegex(row["licence"], LICENCE, name)
                self.assertTrue(row["source_url"].startswith("https://commons.wikimedia.org/wiki/File"), name)
                self.assertTrue(row["author"], name)
                self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")

    def test_F9_no_private_material_in_the_sources(self):
        with open(SOURCES, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["file"].startswith("faces/"):
                    self.assertIsNone(
                        PRIVATE.search(row["file"] + " " + row["source_url"]) if PRIVATE else None, row["file"]
                    )


if __name__ == "__main__":
    unittest.main()
