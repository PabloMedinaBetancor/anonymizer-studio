"""Studio v2.1.8: what the ninth client QA pass (05/10/2026, qa-anon-9) found at the server. One test class per bug; each fails on
v2.1.7 and passes here."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unittest
from unittest import mock

try:
    import anonymizer
    from tests import test_ui_server as server_tests
    from tests.test_ui_server import EventReader
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class StopDuringTheComparisonVideo(server_tests.StudioCase):
    """Bug 1: a stop while the comparison video was being made left the clip "Done" with no comparison and no word about it."""

    def test_the_clip_says_the_comparison_was_stopped(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        folder = os.path.join(self.dir, "out")
        reader = EventReader(self.port)
        self.addCleanup(reader.close)

        def stopped_while_comparing(original, anonymized, target, on_event=None, cancel=None):
            cancel.cancel()  # what a click on Stop does to the running job
            raise anonymizer.Cancelled()

        with mock.patch.object(anonymizer, "compare_video", stopped_while_comparing):
            self.api(
                "POST", "/api/render", {"clips": [clip_id], "options": {"destination": folder, "comparison": True}}, 202
            )
            finished = reader.wait_for("job.finished")
            self.wait_idle()
        self.assertEqual(finished["status"], "stopped")
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        self.assertTrue(rendered["comparison_stopped"])
        self.assertNotIn("comparison", rendered)
        warning = next(
            e for e in reader.events if e["type"] == "job.warning" and e["code"] == "warning.comparison.stopped"
        )
        self.assertEqual(warning["params"], {"name": "clip.mp4"})
        # what a window opened afterwards is shown: the card keeps the word, the folder holds no half-made video
        state = self.api("GET", "/api/state")
        self.assertTrue(state["clips"][0]["render"]["comparison_stopped"])
        self.assertEqual(sorted(n for n in os.listdir(folder) if "compare" in n), [])

    def test_a_finished_comparison_is_not_marked_stopped(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        self.api(
            "POST",
            "/api/render",
            {"clips": [clip_id], "options": {"destination": os.path.join(self.dir, "out"), "comparison": True}},
            202,
        )
        reader.wait_for("job.finished")
        self.wait_idle()
        rendered = next(e for e in reader.events if e["type"] == "clip.rendered")
        self.assertNotIn("comparison_stopped", rendered)
        self.assertTrue(rendered["comparison"].endswith(".compare.mp4"))


class DemoStateShowsNoServerPathWhileRendering(server_tests.StudioCase):
    """Bug 9, in the hosted demo: the name of the file being written is in the clip's state; it is a name, never a path of the server."""

    demo = True

    def test_the_output_is_a_name(self):
        import json
        import time

        path = self.make_clip()
        with open(path, "rb") as f:
            status, _, body = self.request("PUT", "/api/upload?name=clip.mp4", raw=f.read())
        self.assertEqual(status, 200, body[:200])
        clip_id = json.loads(body)["clips"][0]["clip_id"]
        self.analyze(clip_id)
        real = anonymizer.Anonymizer.render

        def slow(self_, analysis, target, on_event=None, cancel=None):
            def handler(event):
                time.sleep(0.05)
                on_event(event)

            return real(self_, analysis, target, handler, cancel)

        with mock.patch.object(anonymizer.Anonymizer, "render", slow):
            self.api("POST", "/api/render", {"clips": [clip_id], "options": {}}, 202)
            time.sleep(0.3)
            state = self.api("GET", "/api/state")
            self.wait_idle()
        render = state["clips"][0]["render"]
        self.assertEqual(render["output"], "clip_anonymized.mp4")
        self.assertNotIn(self.dir, json.dumps(state))


class PdfPaperSize(server_tests.StudioCase):
    """Bug 4: the PDF was Letter-sized (612 x 792 pt) in every country, for a European printer to scale; it is A4, and Letter for a browser in US English."""

    def page_size(self, **options):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        reader = EventReader(self.port)
        self.addCleanup(reader.close)
        folder = os.path.join(self.dir, "out")
        body = {"destination": folder, "reports": {"pdf": True, "csv": False, "json": False}, **options}
        self.api("POST", "/api/render", {"clips": [clip_id], "options": body}, 202)
        rendered = reader.wait_for("clip.rendered")
        reader.wait_for("job.finished")
        info = subprocess.run(["pdfinfo", rendered["report"]["pdf"]], capture_output=True, text=True, check=True).stdout
        return tuple(round(float(v)) for v in re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info).groups())

    def setUp(self):
        if shutil.which("google-chrome") is None or shutil.which("pdfinfo") is None:
            self.skipTest("needs google-chrome and pdfinfo")
        self.browser = shutil.which("google-chrome")
        super().setUp()

    def test_a4_by_default_and_letter_when_the_page_asks_for_it(self):
        self.assertEqual(self.page_size(), (595, 842))
        self.assertEqual(self.page_size(paper="letter"), (612, 792))

    def test_an_unknown_paper_is_refused(self):
        clip_id = self.import_clip()
        self.analyze(clip_id)
        status, _, _ = self.request("POST", "/api/render", {"clips": [clip_id], "options": {"paper": "a0"}})
        self.assertEqual(status, 400)


class PdfTimesInThePagesLanguage(unittest.TestCase):
    """Bug 15: the From and To of the PDF's rows were written with a decimal point (0:01.3) in every language; the page writes "3,5 s" in German."""

    def page(self, language):
        import json
        import pathlib
        from datetime import date

        from anonymizer.ui import pdf

        folder = pathlib.Path(__file__).resolve().parents[1] / "src" / "anonymizer" / "ui" / "i18n"
        texts = json.loads((folder / f"{language}.json").read_text(encoding="utf-8"))
        rows = [anonymizer.ReportRow("a.mp4", 0, "face", "blur", True, "face.reason.verified", 1.3, 3.3, 82, 0.93)]
        return pdf.report_html("a.mp4", rows, texts, True, date(2026, 10, 5), decimal=pdf.decimal_for(language))

    def test_the_comma_in_the_languages_that_write_one(self):
        for language in ("es", "de", "fr", "it", "pt"):
            page = self.page(language)
            self.assertIn("0:01,3", page, language)
            self.assertIn("0:03,3", page, language)
            self.assertNotIn("0:01.3", page, language)

    def test_the_point_in_english_and_chinese(self):
        for language in ("en", "zh-Hans"):
            self.assertIn("0:01.3", self.page(language), language)


if __name__ == "__main__":
    unittest.main()
