"""D1: the truth-label package is current, its hashes are right and every labelled image says where it came from."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(REPO, "docs", "truth-dataset")


def script():
    spec = importlib.util.spec_from_file_location(
        "make_truth_dataset", os.path.join(REPO, "scripts", "make_truth_dataset.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(name):
    with open(os.path.join(PACKAGE, name), encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


class TruthDataset(unittest.TestCase):
    def test_the_package_is_what_the_script_builds_from_bench_truth(self):
        self.assertEqual(script().main(["--check"]), 0, "run python3 scripts/make_truth_dataset.py")

    def test_the_manifest_hashes_are_those_of_the_files(self):
        listed = rows("MANIFEST.csv")
        self.assertGreaterEqual(len(listed), 10)
        for row in listed:
            with open(os.path.join(PACKAGE, row["file"]), "rb") as f:
                data = f.read()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"], row["file"])
            self.assertEqual(len(data), int(row["bytes"]))

    def test_every_labelled_file_but_the_blind_video_frames_has_its_page_licence_and_hash(self):
        for row in rows("IMAGES.csv"):
            if row["set"] == "blind-video-frames":
                continue
            for column in ("source_url", "licence", "source_sha256"):
                self.assertTrue(row[column], f"{row['set']}/{row['name']} has no {column}")
            self.assertEqual(len(row["source_sha256"]), 64)

    def test_the_openalpr_boxes_are_apart_and_under_their_own_licence(self):
        for row in rows("MANIFEST.csv"):
            self.assertEqual(
                row["file"].startswith("labels-agpl/"), row["licence_of_labels"] == "AGPL-3.0", row["file"]
            )

    def test_no_image_or_video_is_in_the_package(self):
        for _, _, names in os.walk(PACKAGE):
            for name in names:
                if name == "LICENSE":  # the AGPL-3.0 text, beside the AGPL labels
                    continue
                self.assertIn(os.path.splitext(name)[1], {".json", ".csv", ".md"}, name)

    def test_the_agpl_labels_carry_the_licence_text(self):
        # the AGPL asks for a copy of the licence to travel with the labels, not only a link
        with open(os.path.join(PACKAGE, "labels-agpl", "LICENSE"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", text)
        self.assertIn("Version 3, 19 November 2007", text)
        self.assertIn("END OF TERMS AND CONDITIONS", text)


if __name__ == "__main__":
    unittest.main()
