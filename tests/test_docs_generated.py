"""Every generated documentation file equals what its script writes now, so the docs cannot drift from the code.

`llms-full.txt` comes from `scripts/docs.py`; the API reference from `scripts/api_docs.py`; the
published numbers from `scripts/fill_numbers.py` (checked with `--verify`). Fix a failure
by running `python scripts/docs.py build`, then read the diff.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import docs_support as ds


def load_docs_script():
    spec = importlib.util.spec_from_file_location("docs_script", ds.ROOT / "scripts" / "docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratedFilesTest(unittest.TestCase):
    def test_the_generated_files_are_the_committed_ones(self):
        stale = load_docs_script().stale_files()
        ds.expect_none(self, stale, "stale generated files (run `python scripts/docs.py build`)")

    def test_llms_full_is_every_site_page_in_navigation_order_and_regenerates_identically(self):
        script = load_docs_script()
        first = script.render_llms_full()
        self.assertEqual(first, script.render_llms_full())
        headers = [line[2:] for line in first.splitlines() if line.startswith("# docs/") and line.endswith(".md")]
        pages = {f"docs/{p.relative_to(ds.DOCS).as_posix()}" for p in ds.DOCS.rglob("*.md") if not ds.is_excluded(p)}
        self.assertEqual(set(headers), pages)
        for dev in ("docs/UI-PROTOCOL.md", "docs/PRIVACY-PATHS.md", "docs/MEASURED.md", "docs/MISSES.md"):
            self.assertNotIn(dev, headers)
        listed = [f"docs/{p}" for p in script.nav_pages() if (ds.DOCS / p).is_file()]
        self.assertEqual([h for h in headers if h in listed], listed)

    def test_the_help_of_the_docs_script_changes_nothing(self):
        before = {p: p.stat().st_mtime_ns for p in [ds.ROOT / "llms-full.txt"] if p.exists()}
        done = subprocess.run(
            [sys.executable, str(ds.ROOT / "scripts" / "docs.py"), "--help"], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(done.returncode, 0, done.stderr[-300:])
        self.assertIn("python scripts/docs.py", done.stdout)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})


class StaleFileIsCaughtTest(unittest.TestCase):
    def test_a_missing_or_edited_llms_full_is_reported_stale(self):
        script = load_docs_script()
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        original = script.LLMS_FULL
        script.LLMS_FULL = folder / "llms-full.txt"
        self.addCleanup(lambda: setattr(script, "LLMS_FULL", original))
        self.assertIn("llms-full.txt", script.stale_files())
        script.LLMS_FULL.write_text(script.render_llms_full(), encoding="utf-8")
        self.assertNotIn("llms-full.txt", script.stale_files())
        script.LLMS_FULL.write_text(script.LLMS_FULL.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
        self.assertIn("llms-full.txt", script.stale_files())


if __name__ == "__main__":
    unittest.main()
