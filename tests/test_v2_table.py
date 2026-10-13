"""D1: the v2 results in the documents come from the sealed read and keep their conditions (M6, RM4, C4, VK8)."""

from __future__ import annotations

import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = (os.path.join("docs", "MEASURED.template.md"),)
RECALL = re.compile(
    r"\{results\.faces_v2\.(?:independent_parts|parts\.\w+|cells)\.\w+\.(?:pct|hidden_by_default_pct)\}"
)
FALSE_HIDDEN = re.compile(r"\{results\.faces_v2\.cells\.\w+\.false_hidden_photos_without_people\.per_100_photos\}")
SECONDS = re.compile(r"\{results\.faces_v2\.cells\.\w+\.seconds_per_photo_median\}|not timed")


def read(doc):
    with open(os.path.join(REPO, doc), encoding="utf-8") as f:
        return f.read()


class V2Table(unittest.TestCase):
    def test_m6_every_level_row_has_its_false_hidden_rate_and_its_time(self):
        for doc in DOCS:
            rows = [r for r in read(doc).splitlines() if r.startswith("|") and "{results.faces_v2.cells." in r]
            # the full measurements keep the reference rows; the README names no other product (a release review, 05/10/2026)
            self.assertGreaterEqual(
                len(rows), 5 if "MEASURED" in doc else 3, doc
            )  # the two Deep rows left the table in 2.1.2 (the Studio does not offer Deep) and are one note
            for row in rows:
                if RECALL.search(row):
                    self.assertRegex(row, FALSE_HIDDEN, f"{doc}: recall without false hidden: {row[:60]}")
                    self.assertRegex(row, SECONDS, f"{doc}: recall without time: {row[:60]}")
            # Deep is out of the headline table, but what is said of it keeps its conditions: its false hidden rate and its time are in the note
            note = next(
                line for line in read(doc).splitlines() if line.startswith("**Experimental, not offered: Deep.**")
            )
            self.assertRegex(note, FALSE_HIDDEN, doc)
            self.assertRegex(note, SECONDS, doc)
            self.assertIn("not part of the headline", note, doc)

    def test_rm4_the_reference_that_does_worse_or_better_stays_in_the_table(self):
        # In the full measurements only: the README names no other product (a release review, 05/10/2026)
        for doc in [d for d in DOCS if d.endswith("MEASURED.template.md")]:
            text = read(doc)
            self.assertIn("Reference `deface` (CenterFace) at the cut chosen on dev", text, doc)
            self.assertIn("Reference `deface` at its own default cut", text, doc)

    def test_the_claims_that_failed_are_said_plainly(self):
        for doc in DOCS:
            text = read(doc)
            self.assertIn("The false hidden rate is not claimed to be low", text, doc)
            # the full measurements say it in plain words; the README keeps the old sentence
            second_look = (
                "did not measurably add to Fast"
                if doc.endswith("MEASURED.template.md")
                else "There is no video headline for the second look"
            )
            self.assertIn(second_look, text, doc)
            if doc.endswith("MEASURED.template.md"):
                self.assertIn("EgoBlur, which was not run", text, doc)

    def test_no_document_keeps_a_placeholder_the_numbers_gate_cannot_fill(self):
        # The gate only sees names that fit the marker pattern: a hyphen in a name hides it and the raw text would be published.
        marker = re.compile(r"\{[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,}\}")
        shaped = re.compile(r"\{[a-z][\w-]*(?:\.[\w-]+)+\}")
        for doc in DOCS:
            for found in shaped.findall(read(doc)):
                self.assertRegex(found, marker, f"{doc}: {found} is not a marker the numbers gate can fill")


if __name__ == "__main__":
    unittest.main()
