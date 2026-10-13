"""scripts/fetch_reference_plate_model.py reads its arguments: `--help` prints the usage and downloads nothing, an unknown
argument is refused before any download, and a file already there that matches is left alone."""

from __future__ import annotations

import contextlib
import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_reference_plate_model as fetch  # noqa: E402


class Arguments(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(fetch.urllib.request, "urlopen", side_effect=AssertionError("the network was used")):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    code = fetch.main(argv)
                except SystemExit as stop:
                    code = stop.code
        return code, out.getvalue(), err.getvalue()

    def test_help_prints_the_usage_and_exits_zero_without_the_network(self):
        code, out, _ = self.run_main(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("usage:", out)
        self.assertIn("--models", out)

    def test_an_unknown_argument_is_refused_before_any_download(self):
        code, _, err = self.run_main(["--nonsense"])
        self.assertEqual(code, 2)
        self.assertIn("unrecognized arguments", err)

    def test_a_download_that_cannot_start_exits_one_and_keeps_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.object(fetch.urllib.request, "urlopen", side_effect=OSError("no route")):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = fetch.main(["--models", folder])
            self.assertEqual(code, 1)
            self.assertIn("download failed", err.getvalue())
            self.assertEqual(list(pathlib.Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
