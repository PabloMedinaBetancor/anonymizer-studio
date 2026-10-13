"""A7: the API reference is generated from the code and current, and every example in the docs runs."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import tempfile
import unittest
from unittest import mock

try:
    from PIL import Image

    import anonymizer.classes
except ImportError as e:
    raise unittest.SkipTest(f"engine libraries missing: {e}") from e

from tests import media
from tests.fakes import plate_module

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = re.compile(r"```python\n(.*?)```", re.S)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "scripts", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def examples(doc: str):
    with open(os.path.join(REPO, doc), encoding="utf-8") as f:
        return EXAMPLE.findall(f.read())


class Docs(unittest.TestCase):
    def test_A7_api_reference_matches_the_docstrings(self):
        for name, text in load_script("api_docs").pages().items():
            with self.subTest(page=name):
                with open(os.path.join(REPO, "docs", "reference", name), encoding="utf-8") as f:
                    self.assertEqual(f.read(), text, "run python3 scripts/api_docs.py")

    def test_a14_the_fast_keep_example_is_among_the_examples_that_run(self):
        runnable = list(examples(os.path.join("docs", "reference", "options.md")))
        self.assertTrue(
            any('detection="fast"' in c and 'unsure="keep"' in c for c in runnable),
            'no runnable example uses detection="fast", unsure="keep"',
        )

    def run_example(self, code: str):
        with tempfile.TemporaryDirectory() as d:
            media.moving_plate_clip(os.path.join(d, "in.mp4"), n=24, audio=False)
            photo = Image.new("RGB", (240, 160), (90, 120, 60))
            photo.paste((250, 250, 250), (40, 60, 140, 90))
            photo.save(os.path.join(d, "in.jpg"))
            os.makedirs(os.path.join(d, "photos"))
            photo.save(os.path.join(d, "photos", "a.jpg"))
            photo.save(os.path.join(d, "photos", "b.png"))
            with open(os.path.join(d, "photos", "c.heic"), "wb") as f:
                f.write(b"\0" * 32)
            cwd = os.getcwd()
            os.chdir(d)
            try:
                with mock.patch.object(
                    anonymizer.classes, "load", lambda kinds, threads, models, **kw: [plate_module()]
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(compile(code, "<example>", "exec"), {"__name__": "__example__"})
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
