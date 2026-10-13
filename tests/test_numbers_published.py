"""The documents a client opens carry figures, never raw {metric} markers (QA report of 04/10/2026, should-fix 6: the README showed
"{faces.photo.by_size.lt24.recall_blur_pct}%"). README.md, docs/accuracy.md and docs/MEASURED.md are the filled copies of their templates,
written by scripts/fill_numbers.py --publish from bench/results/; the numbers gate reads the templates."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = ("README.md", os.path.join("docs", "accuracy.md"), os.path.join("docs", "MEASURED.md"))
MARKER = re.compile(r"\{[a-z][a-z0-9_]*(?:\.[a-z0-9_]+){2,}\}")
FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)


def read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def script():
    spec = importlib.util.spec_from_file_location("fill_numbers", os.path.join(REPO, "scripts", "fill_numbers.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublishedDocuments(unittest.TestCase):
    def test_no_published_document_carries_a_raw_marker(self):
        for doc in PUBLISHED:
            self.assertEqual(MARKER.findall(read(doc)), [], doc)

    def test_the_marker_pattern_sees_the_one_the_tester_found(self):
        self.assertTrue(MARKER.search("only {faces.photo.by_size.lt24.recall_blur_pct}% of the time"))
        self.assertFalse(MARKER.search("the pattern {name} of the export"))

    def test_every_published_document_is_what_its_template_fills_to(self):
        done = subprocess.run(
            [
                sys.executable,
                *(["-s"] if sys.flags.no_user_site else []),
                os.path.join(REPO, "scripts", "fill_numbers.py"),
                "--verify",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_each_published_document_has_its_template(self):
        for doc in PUBLISHED:
            template = doc.replace(".md", ".template.md")
            self.assertTrue(os.path.isfile(os.path.join(REPO, template)), template)


if __name__ == "__main__":
    unittest.main()
