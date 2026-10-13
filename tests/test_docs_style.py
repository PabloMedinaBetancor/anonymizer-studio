"""The documentation reads as written by a person: no banned word or form (the lists are in `tests/docs_support.py`: stock marketing words, filler openings, empty headings and recap endings), and nothing personal.

Scans every page, `README.md`, `AGENTS.md`, `llms.txt`, the changelog's source and the docstrings of the public API.
Code blocks and inline code are left out of the style scan; the privacy scan reads them too. A failure prints
`file:line: what was found`. Files that are still the 2.2.0 text (`LEGACY` in tests/docs_support.py) and the docstrings
are scanned too, and what is found there is a warning until the final check (`scripts/docs.py check --final`).
The private list of internal names is named by `ANON_DOCS_PRIVATE_NAMES`; no such name is spelled in this file.
"""

from __future__ import annotations

import inspect
import re
import unittest

import anonymizer
from tests import docs_support as ds


def public_docstrings():
    """(dotted name, docstring) for every public name of `anonymizer.__all__` and its public methods."""
    for name in anonymizer.__all__:
        obj = getattr(anonymizer, name, None)
        if obj is None:
            continue
        if obj.__doc__ and not isinstance(obj, (str, int, float, tuple, dict, list, set, frozenset)):
            yield name, inspect.getdoc(obj) or ""
        if inspect.isclass(obj):
            for member, value in vars(obj).items():
                if member.startswith("_"):
                    continue
                target = value.fget if isinstance(value, property) else value
                if callable(target) and getattr(target, "__doc__", None):
                    yield f"{name}.{member}", inspect.getdoc(target) or ""


SYNTHETIC_INTERNAL = re.compile(r"team-lead-zeta|ModelOmega")


class StyleScanTest(unittest.TestCase):
    def finish(self, found: list[str], what: str):
        current, legacy = ds.split_legacy(found)
        ds.report_legacy(legacy)
        ds.expect_none(self, current + (legacy if ds.final() else []), what)

    def test_no_page_uses_a_banned_word_or_form(self):
        found = []
        for path in ds.prose_files():
            if path.suffix not in {".md", ".txt"}:
                continue
            faq = path.name.lower() == "faq.md"
            found += [
                f"{ds.rel(path)}:{n}: {what}"
                for n, what in ds.style_findings(path.read_text(encoding="utf-8"), faq=faq)
            ]
        self.finish(found, "banned phrases in pages")

    def test_no_docstring_of_the_public_api_uses_a_banned_word_or_form(self):
        found = []
        for name, doc in public_docstrings():
            found += [f"docstring {name}:{n}: {what}" for n, what in ds.docstring_style_findings(doc)]
        self.finish(found, "banned phrases in docstrings")

    def test_nothing_personal_is_in_a_public_page(self):
        found = []
        for path in ds.prose_files():
            found += [
                f"{ds.rel(path)}:{n}: {what}" for n, what in ds.privacy_findings(path.read_text(encoding="utf-8"))
            ]
        self.finish(found, "personal details in public pages")


class ScannerCatchesSeededFaultsTest(unittest.TestCase):
    """Each rule fires on an invented fault and stays quiet on the allowed form (the scan catches that and nothing else)."""

    def kinds(self, text, **kw):
        return [what for _, what in ds.style_findings(text, **kw)]

    def test_banned_words_and_forms_are_found(self):
        for text, expect in [
            ("The Studio is a robust blurrer.", "banned word: robust"),
            ("You can simply run it.", "banned word: simply"),
            ("Then just run the script.", "banned word: just run"),
            ("This is easy to install.", "banned word: easy"),
            ("Moreover, it blurs plates.", "banned opener: Moreover"),
            ("It blurs. Note that it is slow.", "banned opener: Note that"),
            ("It works \u2014 mostly.", "em dash"),
            ("It works!", "exclamation mark"),
            ("It works \u2713", "emoji or decorative symbol: '\u2713'"),
            ("It does not only blur but also lists.", "banned form: not only ... but also"),
            ("# Overview\n", "empty heading: Overview"),
            ("## Why use it?\n", "question as a heading"),
            ("We ensure the result.", "banned word: ensure"),
        ]:
            with self.subTest(text=text):
                self.assertIn(expect, self.kinds(text))

    def test_allowed_forms_are_not_flagged(self):
        quiet = [
            "```python\nprint('simply robust!')  # \u2014\n```\n",
            "Use `just run` or `robust` in code.\n",
            "<!-- simply an easy comment -->\n",
            "It is not equal (`a != b`) and `![x](y)` is an image.\n",
            "He was just there.\n",  # "just" is banned only before a verb of instruction
            "Range 49.7\u201386.5 uses an en dash.\n",
        ]
        for text in quiet:
            with self.subTest(text=text):
                self.assertEqual(self.kinds(text), [])
        self.assertEqual(self.kinds("## Why Anonymizer Studio?\n", faq=True), [])

    def test_personal_details_are_found(self):
        for text, expect in [
            ("Open /ho" + "me/someone/work.", "home path"),
            ("Write to someone@" + "gmail.com.", "e-mail address"),
            ("Call +34 600" + " 123 456.", "phone number"),
            ("Ask the team-lead-zeta.", "internal name"),
            ("Made with ModelOmega.", "internal name"),
        ]:
            with self.subTest(text=text):
                found = ds.privacy_findings(text, internal=SYNTHETIC_INTERNAL)
                self.assertTrue(any(expect in what for _, what in found), text)
        self.assertEqual(ds.privacy_findings("Mail a@example.com about 1,330 MB in 553 pages, 3.616 s, 2026-0042."), [])

    def test_the_private_name_lists_of_several_files_are_joined(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            one, two = Path(folder) / "one.txt", Path(folder) / "two.txt"
            one.write_text("# a comment\nfirst-secret\n", encoding="utf-8")
            two.write_text("\nsecond-secret\n", encoding="utf-8")
            pattern = ds.load_internal_names(f"{one}{ds.os.pathsep}{two}")
        self.assertTrue(pattern.search("the second-secret name"))
        self.assertTrue(pattern.search("the first-secret name"))
        self.assertIsNone(pattern.search("a comment"))
        self.assertIsNone(ds.load_internal_names(""))


class LegacySplitTest(unittest.TestCase):
    def test_findings_in_files_still_in_the_old_text_are_split_off_and_the_rest_stay(self):
        current, legacy = ds.split_legacy(
            [
                "docs/how-to/batch-photos.md:3: banned word: easy",
                "docs/MISSES.md:9: em dash",
                "docstring Options:2: em dash",
            ]
        )
        self.assertEqual(current, ["docs/how-to/batch-photos.md:3: banned word: easy"])
        self.assertEqual(legacy, ["docs/MISSES.md:9: em dash", "docstring Options:2: em dash"])


if __name__ == "__main__":
    unittest.main()
