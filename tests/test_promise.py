"""PF5, the `promise` check. It is a unit test: the promise is said in words everywhere as "blurred unless you turn it off". The words of v1 (the Not
sure items stay visible unless someone turns them on) may only appear where `unsure="keep"` is explained."""

from __future__ import annotations

import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md",
    "docs/PRIVACY-PATHS.md",
    "docs/UI-PROTOCOL.md",
)
STALE = re.compile(r"stay visible|stays visible|left visible unless|exported visible", re.IGNORECASE)
# A paragraph that explains the option that keeps them visible, or the warning only that option raises, may use the old words.
EXPLAINS_KEEP = re.compile(r'unsure="keep"|unsure=\\"keep\\"|`"keep"`|"keep" is v1|check_left_visible|visible_check')
LANGUAGES = ("en", "es", "fr", "de", "pt", "it", "zh-Hans")
# The v1 promise in each language, as the six keys removed by PF4 (and `export.to_check`) said it, plus paraphrases in English.
LEGACY = {
    "en": r"remain visible|remains visible|left unblurred|unblurred by default|exported without being blurred",
    "es": r"quedan visibles|queda visible|se exportar\u00e1n visibles",
    "fr": r"restent visibles|reste visible|export\u00e9s visibles|export\u00e9 visible",
    "de": r"bleiben im Export sichtbar|bleibt im Export sichtbar|sichtbar exportiert",
    "pt": r"continuam vis\u00edveis|continua vis\u00edvel|exportados vis\u00edveis|exportado vis\u00edvel",
    "it": r"restano visibili|resta visibile|esportati visibili|esportato visibile",
    "zh-Hans": r"仍然可见|将以可见",
}
# Words that say "blur" in each language: the keys that state the promise must carry one.
BLUR_WORD = {
    "en": "blur",
    "es": "difumin",
    "fr": "flout",
    "de": "weichgezeichnet",
    "pt": "desfoc",
    "it": "sfoc",
    "zh-Hans": "模糊",
}
PROMISE_KEYS = ("export.to_check", "export.to_check.one", "review.doubtful.blur_body", "review.doubtful.blur_body.one")


def stale_paragraphs(text: str):
    for paragraph in re.split(r"\n\s*\n", text):
        if STALE.search(paragraph) and not EXPLAINS_KEEP.search(paragraph):
            yield paragraph.strip()[:120]


class PF5ThePromiseInWords(unittest.TestCase):
    def test_pf5_the_documents_do_not_say_that_not_sure_items_are_exported_visible(self):
        for name in DOCUMENTS:
            path = ROOT / name
            if not path.exists():
                continue
            self.assertEqual(list(stale_paragraphs(path.read_text(encoding="utf-8"))), [], name)

    def test_pf5_the_i18n_files_do_not_say_it_either(self):
        for lang in LANGUAGES:
            keys = json.loads((ROOT / "src/anonymizer/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for key, text in keys.items():
                if key.startswith("warning.review.check_left_visible"):
                    continue  # raised only with unsure="keep"
                self.assertIsNone(STALE.search(text), f"{lang} {key}")

    def test_pf5_no_language_says_the_v1_promise_in_its_own_words(self):
        for lang in LANGUAGES:
            legacy = re.compile(LEGACY[lang], re.IGNORECASE)
            keys = json.loads((ROOT / "src/anonymizer/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for key, text in keys.items():
                if key.startswith("warning.review.check_left_visible"):
                    continue  # raised only with unsure="keep"
                self.assertIsNone(legacy.search(text), f"{lang} {key}: {text}")

    def test_pf5_the_keys_that_state_the_promise_say_blur_in_every_language(self):
        for lang in LANGUAGES:
            keys = json.loads((ROOT / "src/anonymizer/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for key in PROMISE_KEYS:
                self.assertIn(BLUR_WORD[lang], keys[key].lower(), f"{lang} {key}")

    def test_pf5_the_check_catches_a_stale_paragraph_and_lets_the_keep_explanation_through(self):
        self.assertEqual(
            len(list(stale_paragraphs("Not sure items are exported visible unless someone turns them on."))), 1
        )
        self.assertEqual(list(stale_paragraphs('With `unsure="keep"` the Not sure items stay visible.')), [])

    def test_pf5_the_readme_and_the_guide_say_what_the_default_rule_is_and_where_the_old_one_went(self):
        # 2.2 reversed the rule by decision: Not sure items are blurred only when placed
        for name in ("README.md", "docs/concepts/how-it-works.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertRegex(text, r'unsure="placed"|"placed"', name)
            self.assertIn('unsure="blur"', text, name)


if __name__ == "__main__":
    unittest.main()
