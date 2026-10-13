"""Review and report details a client sees (QA report of 04/10/2026, polish): items are numbered 1 to n in the report as in the
page (the engine's ids skip the candidates it dropped, so the PDF listed 1, 2, 4, 5, 8 and an area as "#22"), and an area the person
added has a picture of what it covers, like the detected items."""

from __future__ import annotations

import io
import os
import re
import unittest
from datetime import date

try:
    import numpy as np
    from PIL import Image

    import anonymizer
    from anonymizer.ui import pdf
    from tests import test_ui_server as server_tests
    from tests.fakes import FixedModule
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

TEXTS = {
    "report.col.item": "Item",
    "report.col.kind": "Kind",
    "report.kind.face": "Face",
    "report.summary": "{items} items",
}


class ReportNumbers(unittest.TestCase):
    def test_the_pdf_counts_its_rows_from_one_whatever_the_engine_ids_are(self):
        rows = [
            anonymizer.ReportRow("a.jpg", engine_id, "face", "blur", True, "r", 0.0, 0.0, 1, 0.9)
            for engine_id in (3, 7, 22)
        ]
        page = pdf.report_html("a.jpg", rows, TEXTS, False, date(2026, 10, 4))
        first_cells = re.findall(r"<tr[^>]*><td[^>]*>(\d+)</td>", page)
        self.assertEqual(first_cells, ["1", "2", "3"])


class AddedAreaPicture(server_tests.StudioCase):
    modules = [FixedModule("face", [(20, 20, 60, 60)], 0.95, 0.9)]

    def photo(self) -> str:
        path = os.path.join(self.dir, "media", "p.png")
        pixels = (np.random.RandomState(5).rand(200, 300, 3) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(path)
        return path

    def test_an_added_area_has_a_thumbnail_of_its_own_box(self):
        clip_id = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        added = self.api(
            "POST",
            f"/api/clips/{clip_id}/areas",
            {"frame_from": 0, "frame_to": 0, "box": [100, 80, 180, 140], "follow": "fixed"},
            expect=201,
        )
        item = added["item"]
        self.assertTrue(item["manual"])
        self.assertIsNotNone(item["thumb_url"], "an added area has no picture")
        status, response, data = self.get(item["thumb_url"])
        self.assertEqual((status, response.getheader("Content-Type")), (200, "image/jpeg"))
        with Image.open(io.BytesIO(data)) as crop:
            self.assertLessEqual(max(crop.size), 240)
            self.assertGreater(crop.width, 80)  # the box and a margin around it, not a point
            self.assertLess(crop.width, 130)

    def test_removing_the_area_removes_its_picture(self):
        clip_id = self.api("POST", "/api/clips", {"paths": [self.photo()]})["clips"][0]["clip_id"]
        self.analyze(clip_id)
        item = self.api(
            "POST",
            f"/api/clips/{clip_id}/areas",
            {"frame_from": 0, "frame_to": 0, "box": [10, 10, 60, 60], "follow": "fixed"},
            expect=201,
        )["item"]
        self.api("DELETE", f"/api/clips/{clip_id}/areas/{item['item_id']}")
        self.assertEqual(self.get(f"/api/clips/{clip_id}/items/{item['item_id']}/thumb")[0], 404)


if __name__ == "__main__":
    unittest.main()
