"""Every relative link and anchor in the documentation resolves, every image exists, and no page links outside `docs/`
(the site could not follow it). External links are listed, never fetched.

Anchors follow MkDocs' heading slugs. A failure prints `file:line: target: problem`. Files that are still the 2.2.0 text
(`LEGACY` in tests/docs_support.py) are checked too; what is wrong in them is a warning until the final check.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from tests import docs_support as ds


class LinksTest(unittest.TestCase):
    def test_every_internal_link_and_anchor_resolves(self):
        problems, _external = ds.check_links(ds.doc_pages())
        current, legacy = ds.split_legacy([str(p) for p in problems])
        ds.report_legacy(legacy)
        ds.expect_none(self, current + (legacy if ds.final() else []), "broken links")

    def test_every_page_of_the_navigation_exists(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("docs_script", ds.ROOT / "scripts" / "docs.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [p for p in module.nav_pages() if not (ds.DOCS / p).is_file()]
        ds.expect_none(self, missing, "pages named in mkdocs.yml that do not exist")


class LinkCheckerCatchesSeededFaultsTest(unittest.TestCase):
    def check(self, files: dict[str, str], page: str):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        for name, text in files.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        problems, external = ds.check_links([folder / page])
        return [(p.target, p.problem) for p in problems], external

    def test_a_dead_link_a_dead_anchor_and_a_missing_image_are_found(self):
        files = {
            "a.md": "# A\n\n[gone](missing.md) [bad](b.md#nope) [ok](b.md#second-part) ![x](nope.png)\n[self](#a) [badself](#zzz)\n",
            "b.md": "# B\n\n## Second part\n\n## `Anonymizer.analyze`\n",
        }
        problems, _ = self.check(files, "a.md")
        self.assertEqual(
            problems,
            [
                ("missing.md", "no such file"),
                ("b.md#nope", "no such anchor"),
                ("nope.png", "no such file"),
                ("#zzz", "no such anchor"),
            ],
        )

    def test_slugs_follow_mkdocs_including_code_and_duplicates(self):
        files = {
            "a.md": "[1](b.md#anonymizeranalyze) [2](b.md#same) [3](b.md#same_1) [4](b.md#custom)\n",
            "b.md": "# B\n\n## `Anonymizer.analyze`\n\n## Same\n\n## Same\n\n## Other {#custom}\n",
        }
        self.assertEqual(self.check(files, "a.md")[0], [])

    def test_links_in_code_are_not_links_and_external_ones_are_listed_not_fetched(self):
        files = {"a.md": "`[x](missing.md)`\n\n```text\n[y](missing.md)\n```\n\n[web](https://example.org/x)\n"}
        problems, external = self.check(files, "a.md")
        self.assertEqual(problems, [])
        self.assertEqual([e[2] for e in external], ["https://example.org/x"])

    def test_a_mail_link_is_a_problem_in_a_public_page(self):
        problems, _ = self.check({"a.md": "[write](mailto:someone@example.com)\n"}, "a.md")
        self.assertEqual(problems, [("mailto:someone@example.com", "a mail or phone link in a public page")])


class SiteScanTest(unittest.TestCase):
    def scan(self, files: dict[str, str]):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        for name, text in files.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text(text, encoding="utf-8")
        return [p for p in ds.outside_requests(folder) if not p.startswith("mkdocs.yml")]

    def test_a_script_style_font_and_css_import_from_another_host_are_found(self):
        html = (
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css">'
            '<script src="https://cdn.example/x.js"></script><img src="//host/p.png">'
        )
        found = self.scan({"index.html": html, "a.css": "@font-face{src:url(https://f.example/x.woff2)}"})
        self.assertEqual(len(found), 4, found)

    def test_local_files_text_links_and_the_svg_namespace_are_fine(self):
        html = '<link rel="stylesheet" href="assets/a.css"><a href="https://example.org/">text</a><link rel="canonical" href="https://x.example/">'
        css = ".a{background:url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'/>\")}"
        self.assertEqual(self.scan({"index.html": html, "a.css": css}), [])


if __name__ == "__main__":
    unittest.main()
