"""S12: every new key is in the seven languages with the same placeholders, and the removed ones are in none."""

from __future__ import annotations

import json
import pathlib
import re
import unittest

I18N = pathlib.Path(__file__).resolve().parents[2] / "src" / "anonymizer" / "ui" / "i18n"
LANGUAGES = ("en", "es", "fr", "de", "pt", "it", "zh-Hans")
NEW = (
    "options.detection",
    "options.detection.fast",
    "options.detection.careful",
    "options.detection.careful.slower",
    "analyze.second_look",
    "review.badge.second",
    "review.second_score",
    "review.only_second",
    "review.doubtful.keep_visible",
    "review.doubtful.blur_all",
    "error.demo.deep",
)
REMOVED = (
    "options.detection.deep",
    "options.detection.deep.slower",
    "options.detection.deep.hint",
    "options.heavy_only",
    "options.heavy_only.hint",
    "analyze.deep_look",
    "error.review.pending",
    "export.confirm_visible",
    "export.confirm_visible.one",
    "export.visible_title",
    "export.render_anyway",
    "export.review_first",
)


def load(lang: str) -> dict:
    return json.loads((I18N / f"{lang}.json").read_text(encoding="utf-8"))


def placeholders(text: str) -> set:
    return set(re.findall(r"\{(\w+)\}", text))


class S12NewKeys(unittest.TestCase):
    def test_s12_every_new_key_is_in_the_seven_languages_with_the_placeholders_of_english(self):
        english = load("en")
        for lang in LANGUAGES:
            keys = load(lang)
            for key in NEW:
                self.assertIn(key, keys, f"{lang} lacks {key}")
        for lang in LANGUAGES[1:]:
            keys = load(lang)
            for key, text in english.items():
                self.assertEqual(placeholders(keys[key]), placeholders(text), f"{lang} {key}")

    def test_s12_the_seven_languages_have_the_same_keys_and_none_of_the_removed_ones(self):
        english = set(load("en"))
        for lang in LANGUAGES:
            keys = set(load(lang))
            self.assertEqual(keys, english, lang)
            self.assertFalse(keys.intersection(REMOVED), lang)

    def test_s12_no_new_text_is_left_in_english_inside_another_language(self):
        english = load("en")
        for lang in LANGUAGES[1:]:
            keys = load(lang)
            for key in NEW:
                self.assertNotEqual(keys[key], english[key], f"{lang} {key} is still the English text")


if __name__ == "__main__":
    unittest.main()
