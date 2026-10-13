"""The words the Studio shows: no engine jargon in a message a client reads (QA report of 04/10/2026, polish)."""

from __future__ import annotations

import json
import os
import unittest

UI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "anonymizer", "ui", "i18n")


class Wording(unittest.TestCase):
    def texts(self, language):
        with open(os.path.join(UI, f"{language}.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_the_skipped_zones_warning_has_no_engine_jargon_and_keeps_its_count(self):
        for language in ("en", "es", "fr", "de", "pt", "it", "zh-Hans"):
            text = self.texts(language)["warning.second_look.zones_skipped"]
            self.assertIn("{count}", text, language)
        english = self.texts("en")["warning.second_look.zones_skipped"].lower()
        self.assertNotIn("budget", english)
        self.assertNotIn("per second", english)


if __name__ == "__main__":
    unittest.main()
