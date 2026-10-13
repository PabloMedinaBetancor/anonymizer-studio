"""Studio v2.1.7: what the eighth client QA pass (05/10/2026, qa-anon-8) found at the server. One test class per bug; each fails on
v2.1.6 and passes here."""

from __future__ import annotations

import os
import re
import unittest

try:
    import numpy as np
    from PIL import Image

    from tests import test_ui_server as server_tests
    from tests.fakes import FACE, FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e


class QaCase(server_tests.StudioCase):
    modules = [FixedModule(FACE, [(10, 10, 60, 60)])]

    def photo(self, name="ok.jpg", size=(160, 120), folder="media", seed=7) -> str:
        os.makedirs(os.path.join(self.dir, folder), exist_ok=True)
        path = os.path.join(self.dir, folder, name)
        Image.fromarray((np.random.RandomState(seed).rand(size[1], size[0], 3) * 255).astype(np.uint8)).save(path)
        return path


class FolderImportIsListedAsAFileManagerDoes(QaCase):
    """Bug 5: a folder import put the capitals first (`Taipei`, `Westminster`, then `madrid`), as the folder dialog does not."""

    def test_names_are_in_the_order_the_dialog_lists_them(self):
        for name in ("Westminster.jpg", "madrid.jpg", "Taipei.jpg", "alpha.jpg", "Zulu.jpg"):
            self.photo(name, folder="shoot")
        self.photo("beta.jpg", folder="shoot/Sub")
        self.photo("Aaa.jpg", folder="shoot/sub2")
        answer = self.api("POST", "/api/clips", {"paths": [os.path.join(self.dir, "shoot")]})
        self.assertEqual(
            [c["name"] for c in answer["clips"]],
            ["alpha.jpg", "madrid.jpg", "Taipei.jpg", "Westminster.jpg", "Zulu.jpg", "beta.jpg", "Aaa.jpg"],
        )
        browsed = self.api("GET", "/api/browse?path=" + os.path.join(self.dir, "shoot"))
        self.assertEqual(
            [e["name"] for e in browsed["entries"] if e["kind"] != "dir"],
            ["alpha.jpg", "madrid.jpg", "Taipei.jpg", "Westminster.jpg", "Zulu.jpg"],
        )


class CsvFieldsWithASemicolonOrATabAreQuoted(QaCase):
    """Bug 6: the Text Import of LibreOffice splits at a semicolon and a tab as well as at the comma by default, so a file named
    `beach;house.jpg` moved the rest of its row one column to the right in the CSV report."""

    def test_the_names_that_a_spreadsheet_would_split_are_quoted_and_the_rest_is_as_before(self):
        import csv
        import io

        import anonymizer

        names = ["beach;house 1.jpg", "tab\there.jpg", 'say "hi", ok.jpg', "plain \u00d1u.jpg", "-dash.jpg"]
        rows = [
            anonymizer.ReportRow(n, i, "face", "blur", True, "face.reason.verified", 0, 0.5, 1, 0.9)
            for i, n in enumerate(names)
        ]
        path = os.path.join(self.dir, "work", "r.csv")
        anonymizer.Report(rows).to_csv(path)
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8-sig")
        self.assertIn('\r\n"beach;house 1.jpg",0,face,', raw)
        self.assertIn('\r\n"tab\there.jpg",1,face,', raw)
        self.assertIn('\r\n"say ""hi"", ok.jpg",2,face,', raw)
        self.assertIn("\r\nplain \u00d1u.jpg,3,face,", raw)  # not quoted when nothing needs it
        read = list(csv.reader(io.StringIO(raw, newline="")))
        self.assertEqual([r[0] for r in read[1:]], names[:4] + ["'-dash.jpg"])
        self.assertTrue(all(len(r) == len(read[0]) for r in read))
        # for every other row it is what the csv module writes
        plain = io.StringIO(newline="")
        writer = csv.writer(plain)
        writer.writerow([f for f in anonymizer.ReportRow.__dataclass_fields__])
        self.assertTrue(raw.startswith(plain.getvalue()))


class APlateBandHasNoCertainty(QaCase):
    """Bug 14: the report and the Review screen said "Certainty 100%" of a plate band, a region where no text was found
    that is blurred because a plate usually sits there; the 1.0 is a constant the engine gives such an item, not a confidence."""

    def rows(self):
        import anonymizer

        return [
            anonymizer.ReportRow("a.jpg", 0, "plate", "blur", True, "plate.reason.vehicle_text", 0, 0, 1, 0.8),
            anonymizer.ReportRow("a.jpg", 1, "plate", "blur", True, "plate.reason.position_band", 0, 0, 1, 1.0),
        ]

    def test_the_pdf_shows_a_dash_and_says_what_it_is(self):
        import json
        from datetime import date

        from anonymizer.ui import pdf

        for code in ("en", "es", "fr", "de", "pt", "it", "zh-Hans"):
            with open(os.path.join(os.path.dirname(pdf.__file__), "i18n", f"{code}.json"), encoding="utf-8") as f:
                texts = json.load(f)
            page = pdf.report_html("a.jpg", self.rows(), texts, False, date(2026, 10, 5))
            body = [r for r in re.findall(r"<tr[^>]*>(.*?)</tr>", page) if "<td" in r]
            last = [re.findall(r"<td[^>]*>(.*?)</td>", r)[-1] for r in body]
            self.assertEqual(len(last), 2, code)
            self.assertIn("80", last[0], code)  # the text on a vehicle keeps its certainty
            self.assertEqual(last[1], "\u2013", code)  # the band's certainty is a dash
            self.assertIn("<p class=small>", page, code)  # and the footnote says why
            self.assertIn("1.0", page, code)
        only_text = pdf.report_html("a.jpg", self.rows()[:1], texts, False, date(2026, 10, 5))
        self.assertNotIn("<p class=small>", only_text)  # no band, no footnote
        self.assertNotIn("\u2013</td>", only_text)

    def test_the_csv_and_json_keep_the_constant_and_the_docs_say_so(self):
        import anonymizer

        path = os.path.join(self.dir, "work", "band.csv")
        anonymizer.Report(self.rows()).to_csv(path)
        with open(path, encoding="utf-8-sig") as f:
            self.assertIn("plate.reason.position_band,0,0,1,1.0,", f.read().replace("\r\n", "\n").split("\n")[2] + ",")
        docs = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        with open(os.path.join(docs, "reference", "errors.md"), encoding="utf-8") as f:
            self.assertIn("plate.reason.position_band", f.read())
        with open(os.path.join(docs, "concepts", "how-it-works.md"), encoding="utf-8") as f:
            self.assertIn("constant", f.read())


if __name__ == "__main__":
    unittest.main()
