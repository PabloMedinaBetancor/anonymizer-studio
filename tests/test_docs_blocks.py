"""Every ```python block of every page, of `README.md` and `AGENTS.md`, and every `examples/*.py`, runs on the fakes.

This is the fast half of the example gate (the default suite): the blocks run in a fresh sandbox folder that holds
`demo/` with synthetic stand-ins of the five sample files, on the detectors of `tests/fakes.py`, so a block that does not
run, that names a file that is not there or that calls the library wrongly fails here in seconds. What the blocks print
is not compared here (the fakes find other things than the real models); `scripts/docs.py check` runs the same blocks
again on the real models and compares each output with the `# output` block the page shows (tests/test_docs_examples.py).

A failure names the block as `docs/how-to/x.md block 3`. The block conventions are in `tests/docs_runner.py`'s docstring.
Files that are still the 2.2.0 text are left to tests/test_docs.py until they move (`LEGACY` in tests/docs_support.py).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    for _library in ("cv2", "anonymizer.classes"):  # the fakes and the stand-ins need them
        importlib.import_module(_library)
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from tests import docs_runner
from tests import docs_support as ds

SCRIPTS = sorted((ds.ROOT / "examples").glob("*.py")) if (ds.ROOT / "examples").is_dir() else []


class PagesRunOnTheFakes(unittest.TestCase):
    """One test per page that has python blocks (added below), so a failure names the page."""


def _page_test(path: Path):
    def test(self):
        name = ds.rel(path)
        results = ds.run_blocks(name, ds.python_blocks(path), "fakes")
        failures = ds.failure_lines(name, results)
        self.assertFalse(failures, "\n".join(failures))

    return test


for _path in ds.python_pages():
    setattr(
        PagesRunOnTheFakes,
        "test_blocks_of_" + ds.rel(_path).replace("/", "_").replace(".", "_").replace("-", "_"),
        _page_test(_path),
    )


class ExampleScriptsRunOnTheFakes(unittest.TestCase):
    """Every `examples/*.py` runs to the end as a program in a sandbox (a script that says `# docs-test: real-only - why` on its first line is left to the real-model gate)."""


def _script_test(script: Path):
    def test(self):
        first = script.read_text(encoding="utf-8").splitlines()[:1]
        if first and ds.re.match(r"#\s*docs-test:\s*(?:skip|real-only)\s*-\s*", first[0]):
            self.skipTest(first[0])
        done = subprocess.run(
            [sys.executable, str(ds.RUNNER), "script", str(script), "fakes"],
            cwd=ds.ROOT,
            env=ds.child_env(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(done.returncode, 0, f"examples/{script.name}: exit {done.returncode}\n{done.stderr[-1500:]}")

    return test


for _script in SCRIPTS:
    setattr(ExampleScriptsRunOnTheFakes, "test_script_" + _script.stem, _script_test(_script))


class HarnessCatchesSeededFaultsTest(unittest.TestCase):
    """The harness on invented pages: it finds a broken block and a stale output, and nothing else."""

    def blocks(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "page.md"
            path.write_text(text, encoding="utf-8")
            return ds.python_blocks(path)

    def spec(self, blocks, page="docs/x.md"):
        return ds.spec_of(page, blocks)

    def test_blocks_are_numbered_by_page_and_text_blocks_are_ignored(self):
        text = "# T\n\n```python\nprint(1)\n```\n\n```text\nnot code\n```\n\n```python\nprint(2)\n```\n"
        self.assertEqual([(b.number, b.line) for b in self.blocks(text)], [(1, 3), (2, 11)])

    def test_a_text_block_that_starts_with_output_is_the_expected_output(self):
        text = "```python\nprint(1)\n```\n\n```text\n# output\n1\n```\n\n```python\nprint(2)\n```\n"
        self.assertEqual([b.expected_output for b in self.blocks(text)], ["1", None])

    def test_the_skip_and_real_only_comments_need_their_reason_and_are_carried(self):
        text = (
            "<!-- docs-test: skip - needs a Windows machine -->\n```python\nimport msvcrt\n```\n"
            "<!-- docs-test: real-only - the fakes find no faces -->\n```python\nx = 1\n```\n```python\ny = 1\n```\n"
        )
        blocks = self.blocks(text)
        self.assertEqual([b.skip_reason for b in blocks], ["needs a Windows machine", None, None])
        self.assertEqual([b.real_only for b in blocks], [None, "the fakes find no faces", None])

    def test_a_fence_inside_a_longer_fence_is_not_a_block(self):
        self.assertEqual(self.blocks("````markdown\n```python\nprint(1)\n```\n````\n"), [])

    def test_a_broken_block_is_named_by_page_and_number_and_the_rest_are_not_run(self):
        blocks = [
            ds.Block("docs/x.md", 1, 3, "value = 2\n", None, None),
            ds.Block("docs/x.md", 2, 9, "print(value + undefined_name)\n", None, None),
            ds.Block("docs/x.md", 3, 15, "print('after')\n", None, None),
        ]
        results = docs_runner.run_blocks_here(self.spec(blocks), "fakes")
        self.assertEqual([r["status"] for r in results], ["ok", "fail", "not-run"])
        self.assertTrue(
            ds.failure_lines("docs/x.md", results)[0].startswith("docs/x.md block 2 (line 9): NameError: na")
        )

    def test_a_real_only_block_is_skipped_on_the_fakes_and_run_on_the_real_models(self):
        blocks = [ds.Block("docs/x.md", 1, 3, "raise SystemExit(3)\n", None, None, "the fakes find no faces")]
        self.assertEqual(docs_runner.run_blocks_here(self.spec(blocks), "fakes")[0]["status"], "skipped")
        self.assertEqual(docs_runner.run_blocks_here(self.spec(blocks), "real")[0]["status"], "fail")

    def test_a_stale_output_fails_on_the_real_models_and_is_not_compared_on_the_fakes(self):
        blocks = [ds.Block("docs/x.md", 1, 3, "print(41 + 1)\n", "43", None)]
        self.assertEqual(docs_runner.run_blocks_here(self.spec(blocks), "fakes")[0]["status"], "ok")
        stale = docs_runner.run_blocks_here(self.spec(blocks), "real")[0]
        self.assertEqual(stale["status"], "fail")
        self.assertEqual(stale["output_mismatch"], {"expected": "43", "actual": "42"})
        current = [ds.Block("docs/x.md", 1, 3, "print(41 + 1)\n", "42", None)]
        self.assertEqual(docs_runner.run_blocks_here(self.spec(current), "real")[0]["status"], "ok")

    def test_three_dots_in_an_expected_line_stand_for_any_text_on_that_line(self):
        self.assertTrue(docs_runner.output_matches("took ... s\nfaces: 3", "took 4.2 s\nfaces: 3\n"))
        self.assertFalse(docs_runner.output_matches("took ... s\nfaces: 3", "took 4.2 s\nfaces: 4"))
        self.assertFalse(docs_runner.output_matches("faces: 3", "faces: 3\nplates: 1"))

    def test_the_sandbox_holds_the_four_demo_files_and_nothing_a_page_before_wrote(self):
        blocks = [
            ds.Block(
                "docs/a.md",
                1,
                1,
                "import os\nprint(sorted(os.listdir('.')), sorted(os.listdir('demo')), os.listdir('examples/data'))\nopen('left-behind.txt', 'w').close()\n",
                None,
                None,
            )
        ]
        first = ds.run_blocks("docs/a.md", blocks, "fakes")[0]
        self.assertEqual(first["status"], "ok", first)
        self.assertEqual(
            first["actual_output"].strip(),
            f"['demo', 'examples'] {sorted(docs_runner.DEMO_FILES)} ['street-madrid-5s.mp4']",
        )
        second = ds.run_blocks("docs/b.md", blocks, "fakes")[0]
        self.assertEqual(
            second["actual_output"].strip(),
            f"['demo', 'examples'] {sorted(docs_runner.DEMO_FILES)} ['street-madrid-5s.mp4']",
        )

    def test_the_stand_ins_have_the_names_of_the_repositorys_demo_files_and_the_fakes_find_something_in_them(self):
        for name in docs_runner.DEMO_FILES:
            self.assertTrue((ds.ROOT / "demo" / name).is_file(), name)
        for name in docs_runner.EXAMPLE_FILES:
            self.assertTrue((ds.ROOT / name).is_file(), name)
        blocks = [
            ds.Block(
                "docs/a.md",
                1,
                1,
                "import anonymizer\nwith anonymizer.Anonymizer() as anon:\n"
                "    photo = anon.analyze('demo/street-westminster.jpg')\n    clip = anon.analyze('demo/street-madrid.mp4')\n"
                "assert photo.items and clip.items, (len(photo.items), len(clip.items))\n",
                None,
                None,
            )
        ]
        result = ds.run_blocks("docs/a.md", blocks, "fakes")[0]
        self.assertEqual(result["status"], "ok", result)

    def test_a_script_exits_with_its_own_code_in_the_sandbox(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "ok.py"
            script.write_text(
                "import os, sys\nsys.exit(0 if os.path.isfile('demo/street-taipei.jpg') else 5)\n", encoding="utf-8"
            )
            done = subprocess.run(
                [sys.executable, str(ds.RUNNER), "script", str(script), "fakes"], cwd=ds.ROOT, env=ds.child_env()
            )
            self.assertEqual(done.returncode, 0)
            script.write_text("raise SystemExit(7)\n", encoding="utf-8")
            done = subprocess.run(
                [sys.executable, str(ds.RUNNER), "script", str(script), "fakes"], cwd=ds.ROOT, env=ds.child_env()
            )
            self.assertEqual(done.returncode, 7)


if __name__ == "__main__":
    unittest.main()
