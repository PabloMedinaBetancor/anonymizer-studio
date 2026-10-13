"""The PDF report: the page it prints, finding the browser, and printing (with a stand-in browser, and with a real one
when ANONYMIZER_CHROME=1)."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from datetime import date
from unittest import mock

try:
    import anonymizer
    from anonymizer.ui import pdf
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

HERE = os.path.dirname(os.path.abspath(__file__))
EN = os.path.join(os.path.dirname(HERE), "src", "anonymizer", "ui", "i18n", "en.json")
ROWS = [
    anonymizer.ReportRow("a.mp4", 0, "face", "blur", True, "face.reason.verified", 1.0, 3.25, 82, 0.93),
    anonymizer.ReportRow("a.mp4", 1, "plate", "check", False, "plate.reason.unconfirmed", 62.0, 64.5, 40, 0.41),
    anonymizer.ReportRow("a.mp4", 2, "manual", "blur", True, "manual.reason.added", 5.0, 6.0, 25, 1.0),
]


def english() -> dict:
    with open(EN, encoding="utf-8") as f:
        return json.load(f)


class Page(unittest.TestCase):
    def test_every_item_is_a_row_with_its_decision_and_reason(self):
        page = pdf.report_html("a.mp4", ROWS, english(), True, date(2026, 10, 3))
        self.assertIn("<title>Privacy report</title>", page)
        self.assertIn("Created 2026-10-03", page)
        self.assertIn("3 items: 2 blurred, 1 left visible.", page)
        self.assertEqual(page.count("<tr"), 4)  # the header and the three items
        self.assertIn("License plate", page)
        self.assertIn("On a vehicle but not confirmed. Please check.", page)
        self.assertIn("Left visible", page)
        self.assertIn("1:02.0", page)
        self.assertIn("41%", page)
        self.assertIn("tr class=visible", page)

    def test_photos_have_no_time_columns(self):
        page = pdf.report_html("p.jpg", ROWS[:1], english(), False, date(2026, 10, 3))
        self.assertNotIn("<th class=n>From", page)
        self.assertNotIn("0:01.0", page)

    def test_names_and_texts_are_escaped(self):
        rows = [anonymizer.ReportRow("x", 0, "face", "blur", True, "face.reason.verified", 0, 1, 1, 1.0)]
        page = pdf.report_html(
            '<script>alert("x")</script>.mp4', rows, {"report.title": "A & B <i>"}, True, date(2026, 1, 1)
        )
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("A &amp; B &lt;i&gt;", page)

    def test_an_empty_report_says_so_and_a_missing_text_shows_its_key(self):
        page = pdf.report_html("a.mp4", [], {}, True, date(2026, 10, 3))
        self.assertIn("report.none", page)
        self.assertIn("report.title", page)


class Browser(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def script(self, name: str, body: str) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\n" + body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def test_the_first_browser_on_the_path_is_found(self):
        found = {"chromium": "/usr/bin/chromium"}
        with mock.patch.object(pdf.shutil, "which", lambda name: found.get(name)):
            self.assertEqual(pdf.find_browser(), "/usr/bin/chromium")
        with mock.patch.object(pdf.shutil, "which", lambda name: None), mock.patch.object(pdf, "MAC_BROWSERS", ()):
            self.assertIsNone(pdf.find_browser())

    def test_printing_checks_what_came_out(self):
        page = os.path.join(self.dir, "r.html")
        with open(page, "w", encoding="utf-8") as f:
            f.write("<p>x</p>")
        printing = 'for a in "$@"; do case "$a" in --print-to-pdf=*) out="${a#--print-to-pdf=}";; esac; done\n'
        good = self.script("good", printing + 'printf "%%PDF-1.4\\n" > "$out"; head -c 300 /dev/zero >> "$out"\n')
        target = os.path.join(self.dir, "good.pdf")
        pdf.html_to_pdf(good, page, target)
        self.assertTrue(os.path.exists(target))
        for name, body in (
            ("none", "exit 0\n"),
            ("fails", "exit 3\n"),
            (
                "text",
                printing
                + 'echo not a pdf and long enough to pass the size check, which is a hundred bytes or more, yes > "$out"\n',
            ),
        ):
            with self.assertRaises(RuntimeError, msg=name):
                pdf.html_to_pdf(self.script(name, body), page, os.path.join(self.dir, name + ".pdf"))

    def test_a_browser_that_hangs_is_given_up_on(self):
        page = os.path.join(self.dir, "r.html")
        open(page, "w").close()
        hang = self.script("hang", "sleep 30\n")
        with mock.patch.object(pdf, "TIMEOUT_SECONDS", 0.5), self.assertRaises(RuntimeError):
            pdf.html_to_pdf(hang, page, os.path.join(self.dir, "h.pdf"))

    @unittest.skipUnless(
        os.environ.get("ANONYMIZER_CHROME") == "1", "set ANONYMIZER_CHROME=1 to print with a real browser"
    )
    def test_a_real_browser_prints_the_report(self):
        browser = pdf.find_browser()
        if browser is None:
            self.skipTest("no Chromium-based browser here")
        page = os.path.join(self.dir, "report.html")
        with open(page, "w", encoding="utf-8") as f:
            f.write(pdf.report_html("a.mp4", ROWS, english(), True, date(2026, 10, 3)))
        target = os.path.join(self.dir, "report.pdf")
        pdf.html_to_pdf(browser, page, target)
        with open(target, "rb") as f:
            data = f.read()
        self.assertTrue(data.startswith(b"%PDF-"))
        self.assertGreater(len(data), 2000)


if __name__ == "__main__":
    unittest.main()
